from enum import Enum

from . import events
from .packet import (
    QuicErrorCode,
    QuicResetStreamAtFrame,
    QuicResetStreamFrame,
    QuicStopSendingFrame,
    QuicStreamFrame,
)
from .packet_builder import QuicDeliveryState
from .rangeset import RangeSet


class FinalSizeError(Exception):
    pass


class StreamFinishedError(Exception):
    pass


class StreamStateError(Exception):
    pass


class ResetStreamAtState(Enum):
    NONE = 0
    PENDING = 1
    WAITING_ACK = 2
    ACKED = 3


class QuicStreamReceiver:
    """
    The receive part of a QUIC stream.

    It finishes:
    - immediately for a send-only stream
    - upon reception of a STREAM_RESET frame
    - upon reception of a data frame with the FIN bit set
    """

    def __init__(self, stream_id: int | None, readable: bool) -> None:
        self.highest_offset = 0  # the highest offset ever seen
        self.is_finished = not readable
        self.stop_pending = False

        self._buffer = bytearray()
        self._buffer_start = 0  # the offset for the start of the buffer
        self._final_size: int | None = None
        self._ranges = RangeSet()
        self._stream_id = stream_id
        self._stop_error_code: int | None = None
        # set when a RESET_STREAM_AT is received and the stream is not finished
        self._pending_reset_event: events.StreamReset | None = None
        self._reset_error_code: int | None = None
        self._reliable_size: int | None = None

    def get_stop_frame(self) -> QuicStopSendingFrame:
        self.stop_pending = False
        return QuicStopSendingFrame(
            error_code=self._stop_error_code,
            stream_id=self._stream_id,
        )

    def get_pending_reset_event(self) -> events.StreamReset | None:
        if (
            self._reliable_size is not None
            and not self.is_finished
            and self._buffer_start >= self._reliable_size
        ):
            # we are done receiving
            self.is_finished = True
            event = self._pending_reset_event
            self._pending_reset_event = None
            return event
        return None

    def starting_offset(self) -> int:
        return self._buffer_start

    def handle_frame(self, frame: QuicStreamFrame) -> events.StreamDataReceived | None:
        """
        Handle a frame of received data.
        """
        pos = frame.offset - self._buffer_start
        count = len(frame.data)
        frame_end = frame.offset + count

        # we should receive no more data beyond FIN!
        if self._final_size is not None:
            if frame_end > self._final_size:
                raise FinalSizeError("Data received beyond final size")
            elif frame.fin and frame_end != self._final_size:
                raise FinalSizeError("Cannot change final size")
        if frame.fin:
            # Data at or beyond this final size already arrived (RFC 9000 section 4.5).
            if frame_end < self.highest_offset:
                raise FinalSizeError("Data received beyond final size")
            self._final_size = frame_end
        if frame_end > self.highest_offset:
            self.highest_offset = frame_end

        # if we were already done receiving, data is either a duplicate or arrived
        # after a reset, so there is nothing more to deliver
        if self.is_finished:
            return None

        # fast path: new in-order chunk
        if pos == 0 and count and not self._buffer:
            self._buffer_start += count
            if frame.fin and self._reliable_size is None:
                # all data up to the FIN has been received, we're done receiving
                self.is_finished = True
            return events.StreamDataReceived(
                data=frame.data,
                end_stream=frame.fin and self._reliable_size is None,
                stream_id=self._stream_id,
            )

        # discard duplicate data
        if pos < 0:
            frame.data = frame.data[-pos:]
            frame.offset -= pos
            pos = 0
            count = len(frame.data)

        # marked received range
        if frame_end > frame.offset:
            self._ranges.add(frame.offset, frame_end)

        # add new data
        gap = pos - len(self._buffer)
        if gap > 0:
            self._buffer += bytearray(gap)
        self._buffer[pos : pos + count] = frame.data

        # return data from the front of the buffer
        data = self._pull_data()
        end_stream = (
            self._buffer_start == self._final_size and self._reliable_size is None
        )
        if end_stream:
            # all data up to the FIN has been received, we're done receiving
            self.is_finished = True
        if data or end_stream:
            return events.StreamDataReceived(
                data=data, end_stream=end_stream, stream_id=self._stream_id
            )
        else:
            return None

    def handle_reset(
        self, *, final_size: int, error_code: int = QuicErrorCode.NO_ERROR
    ) -> events.StreamReset | None:
        """
        Handle an abrupt termination of the receiving part of the QUIC stream.
        """
        event = self.handle_reset_at(
            final_size=final_size, error_code=error_code, reliable_size=0
        )
        return event

    def handle_reset_at(
        self,
        final_size: int,
        error_code: int = QuicErrorCode.NO_ERROR,
        reliable_size: int = 0,
    ) -> events.StreamReset | None:
        """Handle a reset, retaining delivery up to the smallest Reliable Size."""
        if self._final_size is not None and final_size != self._final_size:
            raise FinalSizeError("Cannot change final size")
        # Data at or beyond this final size already arrived (RFC 9000 section 4.5).
        if final_size < self.highest_offset:
            raise FinalSizeError("Data received beyond final size")
        if self._reset_error_code is not None and self._reset_error_code != error_code:
            raise StreamStateError("error_code must not change")

        self._reset_error_code = error_code
        self._final_size = final_size
        if final_size > self.highest_offset:
            self.highest_offset = final_size

        # if we were already done receiving, there is nothing to report
        if self.is_finished:
            return None
        if self._reliable_size is not None:
            reliable_size = min(reliable_size, self._reliable_size)
        self._reliable_size = reliable_size
        self._pending_reset_event = events.StreamReset(
            error_code=error_code, stream_id=self._stream_id
        )
        return self.get_pending_reset_event()

    def on_stop_sending_delivery(self, delivery: QuicDeliveryState) -> None:
        """
        Callback when a STOP_SENDING is ACK'd.
        """
        if delivery != QuicDeliveryState.ACKED:
            self.stop_pending = True

    def stop(self, error_code: int = QuicErrorCode.NO_ERROR) -> None:
        """
        Request the peer stop sending data on the QUIC stream.
        """
        self._stop_error_code = error_code
        self.stop_pending = True

    def _pull_data(self) -> bytes:
        """
        Remove data from the front of the buffer.
        """
        try:
            has_data_to_read = self._ranges[0].start == self._buffer_start
        except IndexError:
            has_data_to_read = False
        if not has_data_to_read:
            return b""

        r = self._ranges.shift()
        pos = r.stop - r.start
        data = bytes(self._buffer[:pos])
        del self._buffer[:pos]
        self._buffer_start = r.stop
        return data


class QuicStreamSender:
    """
    The send part of a QUIC stream.

    It finishes:
    - immediately for a receive-only stream
    - upon acknowledgement of a STREAM_RESET frame
    - upon acknowledgement of a data frame with the FIN bit set
    """

    def __init__(self, stream_id: int | None, writable: bool) -> None:
        self.buffer_is_empty = True
        self.highest_offset = 0
        self.is_finished = not writable
        self.reset_pending = False

        self._acked = RangeSet()
        self._acked_fin = False
        self._buffer = bytearray()
        self._buffer_fin: int | None = None
        self._buffer_start = 0  # the offset for the start of the buffer
        self._buffer_stop = 0  # the offset for the stop of the buffer
        self._pending = RangeSet()
        self._pending_eof = False
        self._reset_error_code: int | None = None
        self._stream_id = stream_id

        self._reset_at_state: ResetStreamAtState = ResetStreamAtState.NONE
        self._reset_at_error_code: int | None = None
        self._buffer_reliable_size: int | None = None

    @property
    def next_offset(self) -> int:
        """
        The offset for the next frame to send.

        This is used to determine the space needed for the frame's `offset` field.
        """
        try:
            return self._pending[0].start
        except IndexError:
            return self._buffer_stop

    @property
    def reset_at_pending(self) -> bool:
        return self._reset_at_state == ResetStreamAtState.PENDING

    @property
    def reset_final_size(self) -> int:
        """
        The Final Size for the next RESET_STREAM or RESET_STREAM_AT frame.

        The highest offset covers all data sent so far, including a FIN, and
        the Final Size of any reset frame already sent, so those stay committed.
        Until a RESET_STREAM_AT is sent, its Final Size only needs to cover the
        current Reliable Size, which may still shrink. A RESET_STREAM needs
        nothing beyond what was sent. Committing more than that would consume
        flow control credit the peer might never grant.
        """
        if self._reset_error_code is None and self._buffer_reliable_size is not None:
            return max(self.highest_offset, self._buffer_reliable_size)
        return self.highest_offset

    def get_frame(
        self, max_size: int, max_offset: int | None = None
    ) -> QuicStreamFrame | None:
        """
        Get a frame of data to send.

        `max_size` is the number of data bytes which fit in the packet once the
        frame header is written, and is negative if the header does not fit.
        `max_offset` is the highest offset flow control lets the frame reach.
        """
        assert self._reset_error_code is None, "cannot call get_frame() after reset()"

        # Without room for the header, not even a FIN without data can be sent.
        if max_size < 0:
            return None

        # Get the first pending data range.
        try:
            r = self._pending[0]
        except IndexError:
            if self._pending_eof:
                # FIN only
                self._pending_eof = False
                return QuicStreamFrame(fin=True, offset=self._buffer_fin)

            self.buffer_is_empty = True
            return None

        # apply flow control
        start = r.start
        stop = min(r.stop, start + max_size)
        if max_offset is not None and stop > max_offset:
            stop = max_offset
        if stop <= start:
            return None

        # create frame
        frame = QuicStreamFrame(
            data=bytes(
                self._buffer[start - self._buffer_start : stop - self._buffer_start]
            ),
            offset=start,
        )
        self._pending.subtract(start, stop)

        # track the highest offset ever sent
        if stop > self.highest_offset:
            self.highest_offset = stop

        # if the buffer is empty and EOF was written, set the FIN bit
        if self._buffer_fin == stop:
            frame.fin = True
            self._pending_eof = False

        return frame

    def get_reset_frame(self) -> QuicResetStreamFrame:
        self.reset_pending = False
        return QuicResetStreamFrame(
            error_code=self._reset_error_code,
            final_size=self.reset_final_size,
            stream_id=self._stream_id,
        )

    def get_reset_at_frame(self) -> QuicResetStreamAtFrame:
        self._reset_at_state = ResetStreamAtState.WAITING_ACK
        # Once sent, the Final Size cannot change.
        self.highest_offset = self.reset_final_size
        return QuicResetStreamAtFrame(
            error_code=self._reset_at_error_code,
            final_size=self.highest_offset,
            stream_id=self._stream_id,
            reliable_size=self._buffer_reliable_size,
        )

    def on_data_delivery(
        self, delivery: QuicDeliveryState, start: int, stop: int, fin: bool
    ) -> None:
        """
        Callback when sent data is ACK'd.
        """
        # If the frame had the FIN bit set, its end MUST match otherwise
        # we have a programming error.
        assert not fin or stop == self._buffer_fin, (
            "on_data_delivered() was called with inconsistent fin / stop"
        )

        # If a reset has been requested, stop processing data delivery.
        # The transition to the finished state only depends on the reset
        # being acknowledged.
        if self._reset_error_code is not None:
            return

        if delivery == QuicDeliveryState.ACKED:
            if stop > start:
                # Some data has been ACK'd, discard it.
                self._acked.add(start, stop)
                first_range = self._acked[0]
                if first_range.start == self._buffer_start:
                    size = first_range.stop - first_range.start
                    self._acked.shift()
                    self._buffer_start += size
                    del self._buffer[:size]

            if fin:
                # The FIN has been ACK'd.
                self._acked_fin = True

            # TODO: Refactoring needed.
            if self._buffer_start == self._buffer_fin and self._acked_fin:
                if self._reset_at_state == ResetStreamAtState.NONE:
                    # All data and the FIN have been ACK'd, we're done sending.
                    self.is_finished = True

            if (
                self._reset_at_state == ResetStreamAtState.ACKED
                and self._buffer_start >= self._buffer_reliable_size
            ):
                self.buffer_is_empty = True
                self.is_finished = True
        else:
            if self._buffer_reliable_size is not None:
                stop = min(stop, self._buffer_reliable_size)
                fin = False
            if stop > start:
                # Some data has been lost, reschedule it.
                self.buffer_is_empty = False
                self._pending.add(start, stop)

            if fin:
                # The FIN has been lost, reschedule it.
                self.buffer_is_empty = False
                self._pending_eof = True

    def on_reset_delivery(self, delivery: QuicDeliveryState) -> None:
        """
        Callback when a reset is ACK'd.
        """
        if delivery == QuicDeliveryState.ACKED:
            # The reset has been ACK'd, we're done sending.
            self.is_finished = True
        else:
            # The reset has been lost, reschedule it.
            self.reset_pending = True

    def on_reset_at_delivery(
        self, delivery: QuicDeliveryState, reliable_size: int | None = None
    ) -> None:
        """
        Callback when a RESET_STREAM_AT is ACK'd.
        """
        if (
            self._reset_error_code is not None
            or self.is_finished
            or (
                reliable_size is not None
                and reliable_size != self._buffer_reliable_size
            )
            or self._reset_at_state == ResetStreamAtState.ACKED
        ):
            return
        if delivery == QuicDeliveryState.ACKED:
            self._reset_at_state = ResetStreamAtState.ACKED
            if self._buffer_start >= self._buffer_reliable_size:
                self.is_finished = True
        else:
            # The RESET_STREAM_AT has been lost, reschedule it.
            self._reset_at_state = ResetStreamAtState.PENDING

    def reset(self, error_code: int) -> None:
        """
        Abruptly terminate the sending part of the QUIC stream.

        Once this method has been called, or once the stream has finished
        sending, any further calls to it will have no effect.
        """
        if self._reset_error_code is None and not self.is_finished:
            if (
                self._reset_at_error_code is not None
                and self._reset_at_error_code != error_code
            ):
                raise ValueError("Cannot change the error code of a reset")
            self._reset_error_code = error_code
            self.reset_pending = True
            self._reset_at_state = ResetStreamAtState.NONE

            # Prevent any more data from being sent or re-sent.
            self.buffer_is_empty = True

    def reset_at(self, error_code: int, reliable_size: int) -> None:
        """
        Abruptly terminate the sending part of the QUIC stream at a specific offset.
        """
        if self._reset_error_code is not None:
            raise ValueError("Cannot call reset_at() after reset()")
        if (
            self._reset_at_error_code is not None
            and self._reset_at_error_code != error_code
        ):
            raise ValueError("Cannot change the error code of a reset")
        if reliable_size < 0 or reliable_size > self._buffer_stop:
            raise ValueError("reliable_size must be within written data")
        if (
            self._buffer_reliable_size is not None
            and reliable_size > self._buffer_reliable_size
        ):
            raise ValueError("reliable_size must not increase")
        if self.is_finished:
            return

        self._reset_at_error_code = error_code
        self._buffer_reliable_size = reliable_size
        self._reset_at_state = ResetStreamAtState.PENDING
        self._pending_eof = False
        if reliable_size < self._buffer_stop:
            self._pending.subtract(reliable_size, self._buffer_stop)
        self.buffer_is_empty = not len(self._pending)

    def write(self, data: bytes, end_stream: bool = False) -> None:
        """
        Write some data bytes to the QUIC stream.
        """
        if self._buffer_fin is not None:
            raise RuntimeError("Cannot send data after FIN")
        if self._reset_error_code is not None:
            raise RuntimeError("Cannot send data after the stream was reset")
        if self._reset_at_error_code is not None:
            raise RuntimeError("Cannot send data after the stream was reset")
        size = len(data)

        if size:
            self.buffer_is_empty = False
            self._pending.add(self._buffer_stop, self._buffer_stop + size)
            self._buffer += data
            self._buffer_stop += size
        if end_stream:
            self.buffer_is_empty = False
            self._buffer_fin = self._buffer_stop
            self._pending_eof = True


class QuicStream:
    def __init__(
        self,
        stream_id: int | None = None,
        max_stream_data_local: int = 0,
        max_stream_data_remote: int = 0,
        readable: bool = True,
        writable: bool = True,
    ) -> None:
        self.is_blocked = False
        self.max_stream_data_local = max_stream_data_local
        self.max_stream_data_local_sent = max_stream_data_local
        self.max_stream_data_remote = max_stream_data_remote
        self.receiver = QuicStreamReceiver(stream_id=stream_id, readable=readable)
        self.sender = QuicStreamSender(stream_id=stream_id, writable=writable)
        self.stream_id = stream_id

    @property
    def is_finished(self) -> bool:
        return self.receiver.is_finished and self.sender.is_finished

"""Regressions for draft-ietf-quic-reliable-stream-reset-11."""

from unittest import TestCase

from aioquic import tls
from aioquic.buffer import Buffer, encode_uint_var
from aioquic.quic import events
from aioquic.quic.connection import QuicConnectionError
from aioquic.quic.packet import (
    QuicErrorCode,
    QuicFrameType,
    QuicStreamFrame,
    QuicTransportParameters,
    pull_quic_transport_parameters,
    push_quic_transport_parameters,
)
from aioquic.quic.packet_builder import QuicDeliveryState
from aioquic.quic.stream import (
    QuicStream,
    StreamStateError,
)

from .test_connection import (
    SessionTicketStore,
    client_and_server,
    client_receive_context,
    consume_events,
    encode_transport_parameters,
    roundtrip,
)


class TransportParameterTest(TestCase):
    def test_empty_value(self):
        params = pull_quic_transport_parameters(Buffer(data=bytes.fromhex("1d00")))
        self.assertTrue(params.reset_stream_at)
        buf = Buffer(capacity=2)
        push_quic_transport_parameters(buf, params)
        self.assertEqual(buf.data, bytes.fromhex("1d00"))

    def test_absent_and_disabled(self):
        for flag in (None, False):
            buf = Buffer(capacity=20)
            push_quic_transport_parameters(
                buf, QuicTransportParameters(reset_stream_at=flag)
            )
            self.assertEqual(buf.data, b"")
        self.assertIsNone(
            pull_quic_transport_parameters(Buffer(data=b"")).reset_stream_at
        )

    def test_nonempty_rejected(self):
        for data in ("1d0100", "1d020000", "1d01"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                pull_quic_transport_parameters(Buffer(data=bytes.fromhex(data)))

    def test_old_identifier_is_not_support(self):
        params = pull_quic_transport_parameters(
            Buffer(data=bytes.fromhex("c017f7586d2cb57100"))
        )
        self.assertIsNone(params.reset_stream_at)


class StreamResetTest(TestCase):
    def sender(self):
        sender = QuicStream(stream_id=0).sender
        sender.write(b"abcdefgh", end_stream=True)
        sender.get_frame(8)
        return sender

    def test_reliable_size_decreases_and_final_size_is_stable(self):
        sender = self.sender()
        sender.reset_at(7, 6)
        first = sender.get_reset_at_frame()
        sender.reset_at(7, 3)
        second = sender.get_reset_at_frame()
        self.assertEqual((first.final_size, second.final_size), (8, 8))
        self.assertEqual(second.reliable_size, 3)
        with self.assertRaises(ValueError) as cm:
            sender.reset_at(7, 4)
        self.assertEqual(str(cm.exception), "reliable_size must not increase")
        with self.assertRaises(ValueError) as cm:
            sender.reset_at(8, 3)
        self.assertEqual(str(cm.exception), "Cannot change the error code of a reset")
        self.assertEqual(sender.get_reset_at_frame().reliable_size, 3)

    def test_invalid_size_does_not_mutate_sender(self):
        sender = self.sender()
        for size in (-1, 9):
            with self.assertRaises(ValueError):
                sender.reset_at(7, size)
        sender.reset_at(8, 4)
        self.assertEqual(sender.get_reset_at_frame().error_code, 8)

    def test_unsent_reliable_prefix(self):
        sender = QuicStream().sender
        sender.write(b"abcdefgh")
        sender.reset_at(7, 4)
        self.assertEqual(sender.get_reset_at_frame().final_size, 4)
        self.assertEqual(sender.get_frame(20).data, b"abcd")
        self.assertIsNone(sender.get_frame(20))
        sender.on_reset_at_delivery(QuicDeliveryState.ACKED, 4)
        self.assertFalse(sender.is_finished)
        sender.on_data_delivery(QuicDeliveryState.ACKED, 0, 4, False)
        self.assertTrue(sender.is_finished)

    def test_unsent_fin_does_not_commit_final_size(self):
        sender = QuicStream().sender
        sender.write(b"abcdefgh", end_stream=True)
        sender.reset_at(7, 0)
        self.assertEqual(sender.get_reset_at_frame().final_size, 0)
        self.assertIsNone(sender.get_frame(20))

        sender = QuicStream().sender
        sender.write(b"abcdefgh", end_stream=True)
        self.assertEqual(sender.get_frame(4).data, b"abcd")
        sender.reset_at(7, 2)
        self.assertEqual(sender.get_reset_at_frame().final_size, 4)

    def test_unsent_final_size_follows_reliable_size(self):
        sender = QuicStream().sender
        sender.write(b"abcdefgh")
        sender.reset_at(7, 8)
        sender.reset_at(7, 2)
        self.assertEqual(sender.get_reset_at_frame().final_size, 2)

        # an ordinary reset only covers what was sent
        sender = QuicStream().sender
        sender.write(b"abcdefgh")
        self.assertEqual(sender.get_frame(1).data, b"a")
        sender.reset_at(7, 8)
        sender.reset(7)
        self.assertEqual(sender.get_reset_frame().final_size, 1)

        # a sent RESET_STREAM_AT commits its final size
        sender = QuicStream().sender
        sender.write(b"abcdefgh")
        sender.reset_at(7, 8)
        sender.get_reset_at_frame()
        sender.reset_at(7, 2)
        self.assertEqual(sender.get_reset_at_frame().final_size, 8)
        sender.reset(7)
        self.assertEqual(sender.get_reset_frame().final_size, 8)

    def test_lost_data_is_clipped_and_fin_not_retransmitted(self):
        sender = self.sender()
        sender.reset_at(7, 3)
        sender.get_reset_at_frame()
        sender.on_data_delivery(QuicDeliveryState.LOST, 0, 8, True)
        frame = sender.get_frame(20)
        self.assertEqual((frame.offset, frame.data, frame.fin), (0, b"abc", False))
        self.assertIsNone(sender.get_frame(20))

    def test_stale_ack_and_loss_do_not_complete_new_reset(self):
        sender = self.sender()
        sender.reset_at(7, 6)
        sender.get_reset_at_frame()
        sender.reset_at(7, 3)
        sender.get_reset_at_frame()
        sender.on_data_delivery(QuicDeliveryState.ACKED, 0, 8, True)
        sender.on_reset_at_delivery(QuicDeliveryState.ACKED, 6)
        self.assertFalse(sender.is_finished)
        sender.on_reset_at_delivery(QuicDeliveryState.LOST, 6)
        self.assertFalse(sender.reset_at_pending)
        sender.on_reset_at_delivery(QuicDeliveryState.LOST, 3)
        self.assertTrue(sender.reset_at_pending)
        sender.get_reset_at_frame()
        sender.on_reset_at_delivery(QuicDeliveryState.ACKED, 3)
        self.assertTrue(sender.is_finished)
        sender.on_reset_at_delivery(QuicDeliveryState.LOST, 3)
        self.assertFalse(sender.reset_at_pending)

    def test_ordinary_reset_cancels_reliable_reset(self):
        sender = self.sender()
        sender.reset_at(7, 4)
        sender.get_reset_at_frame()
        sender.reset(7)
        self.assertFalse(sender.reset_at_pending)
        sender.on_reset_at_delivery(QuicDeliveryState.LOST, 4)
        self.assertFalse(sender.reset_at_pending)
        self.assertEqual(sender.get_reset_frame().final_size, 8)
        with self.assertRaises(ValueError) as cm:
            sender.reset_at(7, 0)
        self.assertEqual(str(cm.exception), "Cannot call reset_at() after reset()")

    def test_reset_must_keep_reliable_reset_error_code(self):
        sender = self.sender()
        sender.reset_at(7, 4)
        with self.assertRaises(ValueError) as cm:
            sender.reset(8)
        self.assertEqual(str(cm.exception), "Cannot change the error code of a reset")
        self.assertTrue(sender.reset_at_pending)
        self.assertFalse(sender.reset_pending)

        # once reset, further calls have no effect, whatever the error code
        sender.reset(7)
        self.assertTrue(sender.reset_pending)
        sender.reset(8)
        self.assertEqual(sender.get_reset_frame().error_code, 7)

    def test_receive_prefix_then_reset_without_clean_eof(self):
        receiver = QuicStream(stream_id=0).receiver
        self.assertIsNone(receiver.handle_reset_at(8, 7, 8))
        data = receiver.handle_frame(
            QuicStreamFrame(offset=0, data=b"abcdefgh", fin=True)
        )
        self.assertEqual(
            data,
            events.StreamDataReceived(stream_id=0, data=b"abcdefgh", end_stream=False),
        )
        self.assertEqual(
            receiver.get_pending_reset_event(),
            events.StreamReset(stream_id=0, error_code=7),
        )
        self.assertTrue(receiver.is_finished)
        self.assertIsNone(receiver.get_pending_reset_event())
        self.assertIsNone(receiver.handle_reset_at(8, 7, 8))
        with self.assertRaises(StreamStateError):
            receiver.handle_reset_at(8, 8, 8)

    def test_receive_reordered_sizes_and_ordinary_reset(self):
        receiver = QuicStream().receiver
        self.assertIsNone(receiver.handle_reset_at(8, 7, 6))
        self.assertIsNone(receiver.handle_reset_at(8, 7, 3))
        self.assertIsNone(receiver.handle_reset_at(8, 7, 6))
        receiver.handle_frame(QuicStreamFrame(offset=0, data=b"abc"))
        self.assertIsNotNone(receiver.get_pending_reset_event())
        self.assertEqual(receiver.highest_offset, 8)
        with self.assertRaises(StreamStateError):
            receiver.handle_reset(final_size=8, error_code=8)
        other = QuicStream().receiver
        other.handle_reset(final_size=8, error_code=7)
        self.assertIsNone(other.handle_reset_at(8, 7, 6))

    def test_receive_smaller_size_completes_immediately(self):
        receiver = QuicStream().receiver
        receiver.handle_frame(QuicStreamFrame(offset=0, data=b"abc"))
        receiver.handle_reset_at(8, 7, 6)
        self.assertIsNotNone(receiver.handle_reset_at(8, 7, 3))
        self.assertTrue(receiver.is_finished)


class ConnectionResetTest(TestCase):
    def test_negotiation_is_directional(self):
        with client_and_server(server_options={"reliable_stream_reset": True}) as (
            client,
            server,
        ):
            self.assertTrue(client._remote_reset_stream_at)
            self.assertFalse(server._remote_reset_stream_at)
            client.send_stream_data(0, b"abcdefgh")
            client.reset_stream_at(0, 7, 3)
            with self.assertRaises(ValueError):
                server.reset_stream_at(0, 7, 3)
            consume_events(server)
            roundtrip(client, server)
            self.assertEqual(
                server.next_event(),
                events.StreamDataReceived(stream_id=0, data=b"abc", end_stream=False),
            )
            self.assertEqual(
                server.next_event(), events.StreamReset(stream_id=0, error_code=7)
            )

    def test_api_rejects_without_support_or_handshake(self):
        with client_and_server() as (client, server):
            with self.assertRaises(ValueError):
                client.reset_stream_at(0, 7, 0)
            self.assertNotIn(0, client._streams)
        with client_and_server(handshake=False) as (client, server):
            with self.assertRaises(ValueError):
                client.reset_stream_at(0, 7, 0)

    def test_api_rejects_misuse_with_value_error(self):
        with client_and_server(server_options={"reliable_stream_reset": True}) as (
            client,
            server,
        ):
            client.send_stream_data(0, b"abcdefgh")
            client.reset_stream_at(0, 7, 4)
            for call in (
                lambda: client.reset_stream_at(0, 7, 5),
                lambda: client.reset_stream_at(0, 8, 4),
                lambda: client.reset_stream(0, 8),
            ):
                with self.assertRaises(ValueError):
                    call()
            client.reset_stream(0, 7)
            with self.assertRaises(ValueError):
                client.reset_stream_at(0, 7, 0)

    def test_receive_validation(self):
        with client_and_server(client_options={"reliable_stream_reset": True}) as (
            client,
            server,
        ):
            for final, reliable, code in (
                (3, 4, QuicErrorCode.FRAME_ENCODING_ERROR),
                (
                    client._local_max_stream_data_bidi_remote + 1,
                    0,
                    QuicErrorCode.FLOW_CONTROL_ERROR,
                ),
                (client._local_max_data.value + 1, 0, QuicErrorCode.FLOW_CONTROL_ERROR),
            ):
                data = b"".join(encode_uint_var(x) for x in (1, 7, final, reliable))
                with (
                    self.subTest(final=final),
                    self.assertRaises(QuicConnectionError) as cm,
                ):
                    client._handle_reset_stream_frame(
                        client_receive_context(client),
                        QuicFrameType.RESET_STREAM_AT,
                        Buffer(data=data),
                    )
                self.assertEqual(cm.exception.error_code, code)

    def test_receive_frame_encoding_error_regardless_of_stream_state(self):
        with client_and_server(client_options={"reliable_stream_reset": True}) as (
            client,
            server,
        ):
            client._handle_reset_stream_frame(
                client_receive_context(client),
                QuicFrameType.RESET_STREAM,
                Buffer(data=b"\x01\x07\x03"),
            )
            self.assertTrue(client._streams[1].receiver.is_finished)
            # the same frame is rejected on a stream which was already reset, and
            # on a stream the client cannot receive on
            for stream_id in (1, 2):
                with (
                    self.subTest(stream_id=stream_id),
                    self.assertRaises(QuicConnectionError) as cm,
                ):
                    client._handle_reset_stream_frame(
                        client_receive_context(client),
                        QuicFrameType.RESET_STREAM_AT,
                        Buffer(data=bytes([stream_id, 7, 3, 4])),
                    )
                self.assertEqual(
                    cm.exception.error_code, QuicErrorCode.FRAME_ENCODING_ERROR
                )

    def test_unadvertised_receive_is_rejected(self):
        with client_and_server() as (client, server):
            with self.assertRaises(QuicConnectionError) as cm:
                client._handle_reset_stream_frame(
                    client_receive_context(client),
                    QuicFrameType.RESET_STREAM_AT,
                    Buffer(data=b"\x01\x07\x00\x00"),
                )
            self.assertEqual(cm.exception.error_code, QuicErrorCode.PROTOCOL_VIOLATION)

    def test_flow_control_defers_reset(self):
        with client_and_server(server_options={"reliable_stream_reset": True}) as (
            client,
            server,
        ):
            client.send_stream_data(0, b"abcdefgh")
            stream = client._streams[0]
            client.reset_stream_at(0, 7, 4)
            stream.max_stream_data_remote = 2
            roundtrip(client, server)
            self.assertTrue(stream.sender.reset_at_pending)
            stream.max_stream_data_remote = 100
            client._remote_max_data = 2
            roundtrip(client, server)
            self.assertTrue(stream.sender.reset_at_pending)
            client._remote_max_data = 100
            roundtrip(client, server)
            self.assertFalse(stream.sender.reset_at_pending)
            self.assertEqual(client._remote_max_data_used, 4)

    def test_unsent_fin_does_not_block_reset(self):
        for reliable_size, expected_data in ((0, b""), (1, b"a")):
            with (
                self.subTest(reliable_size=reliable_size),
                client_and_server(
                    server_options={"max_stream_data": 4, "reliable_stream_reset": True}
                ) as (client, server),
            ):
                client.send_stream_data(0, b"abcdefgh", end_stream=True)
                client.reset_stream_at(0, 7, reliable_size)
                roundtrip(client, server)
                roundtrip(server, client)

                data = b""
                resets = []
                while (event := server.next_event()) is not None:
                    if isinstance(event, events.StreamDataReceived):
                        data += event.data
                        self.assertFalse(event.end_stream)
                    elif isinstance(event, events.StreamReset):
                        resets.append(event)
                self.assertEqual(data, expected_data)
                self.assertEqual(
                    resets, [events.StreamReset(stream_id=0, error_code=7)]
                )
                self.assertTrue(client._streams[0].sender.is_finished)
                self.assertEqual(client._remote_max_data_used, reliable_size)

    def test_unsent_fin_does_not_block_stop_sending_reset(self):
        with client_and_server(
            server_options={"max_stream_data": 4, "reliable_stream_reset": True}
        ) as (client, server):
            client.send_stream_data(0, b"abcdefgh", end_stream=True)
            client.reset_stream_at(0, 7, 1)
            client._handle_stop_sending_frame(
                client_receive_context(client),
                QuicFrameType.STOP_SENDING,
                Buffer(data=b"\x00\x09"),
            )
            roundtrip(client, server)
            roundtrip(server, client)

            resets = []
            while (event := server.next_event()) is not None:
                if isinstance(event, events.StreamReset):
                    resets.append(event)
            self.assertEqual(resets, [events.StreamReset(stream_id=0, error_code=7)])
            self.assertTrue(client._streams[0].sender.is_finished)
            # the RESET_STREAM_AT was never sent, so nothing beyond the sent data
            # is committed
            self.assertEqual(client._remote_max_data_used, 0)

    def test_unsent_reliable_size_does_not_block_reset(self):
        # The peer has received nothing, so it never raises its limit of 4 bytes.
        # The reset must not keep needing credit for a Reliable Size of 8 which
        # was never sent.
        for stop_sending in (False, True):
            with (
                self.subTest(stop_sending=stop_sending),
                client_and_server(
                    server_options={"max_stream_data": 4, "reliable_stream_reset": True}
                ) as (client, server),
            ):
                client.send_stream_data(0, b"abcdefgh")
                client.reset_stream_at(0, 7, 8)
                if stop_sending:
                    client._handle_stop_sending_frame(
                        client_receive_context(client),
                        QuicFrameType.STOP_SENDING,
                        Buffer(data=b"\x00\x09"),
                    )
                else:
                    client.reset_stream_at(0, 7, 0)
                roundtrip(client, server)
                roundtrip(server, client)

                resets = []
                while (event := server.next_event()) is not None:
                    self.assertNotIsInstance(event, events.StreamDataReceived)
                    if isinstance(event, events.StreamReset):
                        resets.append(event)
                self.assertEqual(
                    resets, [events.StreamReset(stream_id=0, error_code=7)]
                )
                self.assertTrue(client._streams[0].sender.is_finished)
                self.assertEqual(client._remote_max_data_used, 0)

    def test_reset_flow_control_counted_once(self):
        with client_and_server(client_options={"reliable_stream_reset": True}) as (
            client,
            server,
        ):
            data = b"\x01\x07\x08\x04"
            for _ in range(2):
                client._handle_reset_stream_frame(
                    client_receive_context(client),
                    QuicFrameType.RESET_STREAM_AT,
                    Buffer(data=data),
                )
            self.assertEqual(client._local_max_data.used, 8)

    def test_nonempty_parameter_is_transport_parameter_error(self):
        with client_and_server() as (client, server):
            with self.assertRaises(QuicConnectionError) as cm:
                client._parse_transport_parameters(bytes.fromhex("1d0100"))
            self.assertEqual(
                cm.exception.error_code, QuicErrorCode.TRANSPORT_PARAMETER_ERROR
            )

    def test_ticket_remembers_capability(self):
        with client_and_server() as (client, server):
            client._parse_transport_parameters(
                encode_transport_parameters(
                    QuicTransportParameters(reset_stream_at=True)
                ),
                from_session_ticket=True,
            )
            self.assertTrue(client._remote_reset_stream_at)

    def test_server_declines_resumption_when_capability_disabled(self):
        tickets = []
        store = SessionTicketStore()
        with client_and_server(
            server_options={"reliable_stream_reset": True},
            client_kwargs={"session_ticket_handler": tickets.append},
            server_kwargs={"session_ticket_handler": store.add},
        ):
            pass
        self.assertTrue(tickets)
        with client_and_server(
            client_options={"session_ticket": tickets[0]},
            server_kwargs={"session_ticket_fetcher": store.pop},
        ) as (client, server):
            self.assertFalse(server.tls.early_data_accepted)
            self.assertFalse(client.tls.session_resumed)
            self.assertFalse(client._remote_reset_stream_at)

    def test_reset_to_ordinary_reset_keeps_flow_control(self):
        # an unsent RESET_STREAM_AT commits nothing
        with client_and_server(server_options={"reliable_stream_reset": True}) as (
            client,
            server,
        ):
            client.send_stream_data(0, b"abcdefgh")
            stream = client._streams[0]
            client.reset_stream_at(0, 7, 4)
            client.reset_stream(0, 7)
            stream.max_stream_data_remote = 2
            roundtrip(client, server)
            self.assertFalse(stream.sender.reset_pending)
            self.assertEqual(server._streams[0].receiver._final_size, 0)
            self.assertEqual(client._remote_max_data_used, 0)

        # a sent RESET_STREAM_AT commits its final size, counted once
        with client_and_server(server_options={"reliable_stream_reset": True}) as (
            client,
            server,
        ):
            client.send_stream_data(0, b"abcdefgh")
            stream = client._streams[0]
            client.reset_stream_at(0, 7, 4)
            stream.max_stream_data_remote = 4
            roundtrip(client, server)
            self.assertFalse(stream.sender.reset_at_pending)
            client.reset_stream(0, 7)
            roundtrip(client, server)
            self.assertFalse(stream.sender.reset_pending)
            self.assertEqual(server._streams[0].receiver._final_size, 4)
            self.assertEqual(client._remote_max_data_used, 4)

    def test_packet_loss_retransmits_prefix_and_reset(self):
        import time

        with client_and_server(server_options={"reliable_stream_reset": True}) as (
            client,
            server,
        ):
            consume_events(server)
            client.send_stream_data(0, b"abcdefgh")
            stream = client._streams[0]
            client.reset_stream_at(0, 7, 3)
            datagrams = client.datagrams_to_send(now=time.time())
            self.assertTrue(datagrams)
            space = client._spaces[tls.Epoch.ONE_RTT]
            packets = list(space.sent_packets.values())
            self.assertTrue(
                any(
                    handler == stream.sender.on_reset_at_delivery and args == (3,)
                    for packet in packets
                    for handler, args in packet.delivery_handlers
                )
            )
            # Drop the datagrams and invoke the loss recovery callbacks.
            for packet in packets:
                for handler, args in packet.delivery_handlers:
                    handler(QuicDeliveryState.LOST, *args)
            self.assertTrue(stream.sender.reset_at_pending)
            self.assertEqual(list(stream.sender._pending), [range(0, 3)])
            roundtrip(client, server)
            self.assertEqual(
                server.next_event(),
                events.StreamDataReceived(stream_id=0, data=b"abc", end_stream=False),
            )
            self.assertEqual(
                server.next_event(), events.StreamReset(stream_id=0, error_code=7)
            )
            self.assertTrue(stream.sender.is_finished)

    def test_reordered_data_and_error_mapping(self):
        with client_and_server(client_options={"reliable_stream_reset": True}) as (
            client,
            server,
        ):
            client._handle_reset_stream_frame(
                client_receive_context(client),
                QuicFrameType.RESET_STREAM_AT,
                Buffer(data=b"\x01\x07\x08\x03"),
            )
            with self.assertRaises(QuicConnectionError) as cm:
                client._handle_reset_stream_frame(
                    client_receive_context(client),
                    QuicFrameType.RESET_STREAM,
                    Buffer(data=b"\x01\x08\x08"),
                )
            self.assertEqual(cm.exception.error_code, QuicErrorCode.STREAM_STATE_ERROR)

    def test_stop_sending_preserves_reset_error_code(self):
        with client_and_server(server_options={"reliable_stream_reset": True}) as (
            client,
            server,
        ):
            client.send_stream_data(0, b"abcdefgh")
            client.reset_stream_at(0, 7, 3)
            stream = client._streams[0]
            for _ in range(2):
                client._handle_stop_sending_frame(
                    client_receive_context(client),
                    QuicFrameType.STOP_SENDING,
                    Buffer(data=b"\x00\x09"),
                )
            self.assertFalse(stream.sender.reset_at_pending)
            self.assertEqual(stream.sender.get_reset_frame().error_code, 7)

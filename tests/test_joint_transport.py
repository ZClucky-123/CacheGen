import socket
import threading
import time
import unittest
from mini_repro.transport import DynamicTransport
from experiments.joint_wire import frame, receive_wire


class TransportTests(unittest.TestCase):
    def test_dynamic_lengths_and_one_connection(self):
        with DynamicTransport(io_idle_timeout_s=.5) as transport:
            for i, body in enumerate((b'a', b'b' * 10000)):
                wire = frame(body, {'token_start': i, 'token_end': i + 1})
                future = transport.send(wire, identity={'block_index': i})
                received, row = future.result(timeout=2)
                self.assertEqual(received, wire)
                self.assertGreater(row['data_available_ns'], row['net_start_ns'])
                self.assertEqual(row['wire_bytes'], len(wire))

    def test_truncation_and_idle_timeout(self):
        a, b = socket.socketpair()
        try:
            a.sendall(b'CG')
            a.close()
            with self.assertRaisesRegex(RuntimeError, 'transport_disconnected'):
                receive_wire(b, io_idle_timeout_s=.02)
        finally:
            b.close()
        a, b = socket.socketpair()
        try:
            with self.assertRaisesRegex(TimeoutError, 'transport_timeout'):
                receive_wire(b, io_idle_timeout_s=.02)
        finally:
            a.close()
            b.close()

    def test_bad_magic_aborts_sender_without_idle_timeout_wait(self):
        started=time.perf_counter()
        with DynamicTransport(io_idle_timeout_s=5) as transport:
            with self.assertRaisesRegex(ValueError,'magic'):
                transport.send(b'BAD!' + bytes(262144)).result(timeout=1)
        self.assertLess(time.perf_counter()-started,1)

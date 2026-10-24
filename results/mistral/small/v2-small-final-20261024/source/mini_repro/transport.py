"""Measured, paced local socket transport with bounded copies and no NIC changes."""
import math
import socket
import threading
import time


def rate_limited_transfer(payload, bandwidth_gbps, chunk_bytes=None, *, validate=False):
    if bandwidth_gbps is not None and (not math.isfinite(bandwidth_gbps) or bandwidth_gbps <= 0):
        raise ValueError('bandwidth must be positive or None for unlimited')
    if chunk_bytes is not None and chunk_bytes <= 0:
        raise ValueError('chunk size must be positive')
    # About 1 ms of target traffic per send, bounded at 64 KiB..1 MiB.
    quantum = chunk_bytes or min(1 << 20, max(65536, int((bandwidth_gbps or 10) * 125000)))
    start = time.perf_counter()
    sender, receiver = socket.socketpair()
    sender.settimeout(60)
    receiver.settimeout(60)
    errors = []
    view = memoryview(payload)

    def send():
        try:
            with sender:
                for offset in range(0, len(view), quantum):
                    end = min(offset + quantum, len(view))
                    sender.sendall(view[offset:end])
                    if bandwidth_gbps is not None:
                        remaining = start + end * 8 / (bandwidth_gbps * 1e9) - time.perf_counter()
                        if remaining > 0:
                            time.sleep(remaining)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=send, daemon=True)
    worker.start()
    buffer = bytearray(len(payload))
    target = memoryview(buffer)
    offset = 0
    try:
        with receiver:
            while offset < len(buffer):
                count = receiver.recv_into(target[offset:], min(len(buffer) - offset, 1 << 20))
                if not count:
                    raise RuntimeError('Truncated local transport')
                offset += count
    finally:
        worker.join(timeout=65)
    if worker.is_alive():
        raise RuntimeError('Transport worker did not terminate')
    if errors:
        raise errors[0]
    received = bytes(buffer)
    elapsed = time.perf_counter() - start
    # Debug equality costs are deliberately outside the reported transfer interval.
    if validate and received != payload:
        raise RuntimeError('Local transport corrupted payload')
    return received, elapsed

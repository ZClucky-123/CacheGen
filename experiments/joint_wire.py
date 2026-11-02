"""Self-describing dynamic frames; no receiver access to sender payload lengths."""
import json
import socket
import struct
import time

HEADER = struct.Struct('!4sIQ')
M45_HEADER = struct.Struct('<4sHHIHHHHIIIIIQQQ')
MAX_FRAME_BYTES = 67108864


def frame(payload, metadata):
    meta = json.dumps(dict(metadata, payload_bytes=len(payload)), sort_keys=True,
                      separators=(',', ':')).encode()
    return HEADER.pack(b'CGV3', len(meta), len(payload)) + meta + payload


def unframe(wire):
    if len(wire) < HEADER.size:
        raise ValueError('Truncated frame')
    magic, n, size = HEADER.unpack_from(wire)
    if magic != b'CGV3' or len(wire) != HEADER.size + n + size:
        raise ValueError('Invalid frame size/magic')
    metadata = json.loads(wire[HEADER.size:HEADER.size + n])
    if metadata.get('payload_bytes') != size:
        raise ValueError('Invalid payload length')
    return metadata, wire[HEADER.size + n:]


def receive_wire(endpoint, request=None, *, io_idle_timeout_s=30., max_frame_bytes=MAX_FRAME_BYTES,
                 cancelled=None, on_progress=None):
    """Idle time starts after NET_START; byte progress alone resets the deadline."""
    if request is not None:
        io_idle_timeout_s = getattr(request, 'io_idle_timeout_s', io_idle_timeout_s)
    last = time.perf_counter_ns()
    endpoint.settimeout(min(.1, io_idle_timeout_s))

    def exact(count):
        nonlocal last
        result = bytearray()
        while len(result) < count:
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError('transport_cancelled')
            if time.perf_counter_ns() - last >= io_idle_timeout_s * 1e9:
                raise TimeoutError('transport_timeout')
            try:
                data = endpoint.recv(min(count - len(result), 1 << 20))
            except socket.timeout:
                continue
            if not data:
                raise RuntimeError('transport_disconnected')
            result.extend(data)
            last = time.perf_counter_ns()
            if on_progress:
                on_progress(len(data), last)
        return bytes(result)

    first = exact(4)
    if first == b'CGV3':
        header = first + exact(HEADER.size - 4)
        _, meta, body = HEADER.unpack(header)
        size = HEADER.size + meta + body
    elif first == b'M45F':
        header = first + exact(60)
        fields = M45_HEADER.unpack(header)
        size = 64 + fields[-3] + fields[-2]
    else:
        raise ValueError('Unknown dynamic frame magic')
    if size > max_frame_bytes or size < len(header):
        raise ValueError('Frame exceeds configured limit')
    return header + exact(size - len(header))

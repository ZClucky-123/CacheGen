"""CPU socket threads with pre-send pacing and bounded receiver credits.

GPU work belongs to the caller's restore callback on the main thread. Serial
credits return after restoration; pipeline credits return when a message is
taken. Socket buffering therefore cannot bypass the application lookahead cap.
"""
import math
import queue
import socket
import threading
import time


def _bandwidth(value):
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise ValueError('bandwidth must be positive or None for unlimited')


def _acquire(semaphore, cancelled):
    while not cancelled.is_set():
        if semaphore.acquire(timeout=.02):
            return
    raise RuntimeError('Transport cancelled')


def _get(items, cancelled):
    while not cancelled.is_set():
        try:
            return items.get(timeout=.02)
        except queue.Empty:
            pass
    raise RuntimeError('Transport cancelled')


def transfer_chunks(messages, bandwidth_gbps, restore, *, mode='chunk_serial',
                    queue_capacity=1, chunk_bytes=None, origin=None, analyze_timeline=True):
    """Transfer prepared complete messages over ONE connection and restore them.

    All times are relative to origin, taken by TTFT callers before socket setup.
    Effective network intervals start after receive capacity/buffer readiness;
    they include pre-send pacing and socket work, excluding capacity/GPU waits.
    Equality/debug checks should run after the first-token measurement.
    """
    _bandwidth(bandwidth_gbps)
    if mode not in ('whole', 'chunk_serial', 'chunk_pipeline'):
        raise ValueError('Unknown transfer mode')
    if type(queue_capacity) is not int or queue_capacity < 1:
        raise ValueError('queue capacity must be positive')
    if chunk_bytes is not None and (type(chunk_bytes) is not int or chunk_bytes <= 0):
        raise ValueError('chunk size must be positive')
    if not messages or any(not isinstance(m, bytes) for m in messages):
        raise ValueError('Expected nonempty sequence of immutable byte messages')
    if mode == 'whole' and len(messages) != 1:
        raise ValueError('whole mode requires one message')
    origin = time.perf_counter() if origin is None else origin
    clock = lambda: time.perf_counter() - origin
    quantum = chunk_bytes or min(1 << 20, max(65536, int((bandwidth_gbps or 10) * 125000)))
    sender, receiver = socket.socketpair()
    sender.settimeout(.2)
    receiver.settimeout(.2)
    cancelled = threading.Event()
    slots = threading.BoundedSemaphore(queue_capacity)
    serial_gate = threading.Semaphore(1)
    ready = queue.Queue(maxsize=1)
    completed = queue.Queue(maxsize=queue_capacity)
    errors = []
    rows = [{} for _ in messages]
    received_messages = []

    def sending():
        try:
            for index, payload in enumerate(messages):
                wait_start = clock()
                permitted = _get(ready, cancelled)
                if permitted != index:
                    raise RuntimeError('Send permission order differs')
                send_start = clock()
                row = rows[index]
                row.update(send_permission_wait_start=wait_start, send_start=send_start,
                           send_permission_wait_seconds=send_start - wait_start)
                pacing, blocking = 0., 0.
                view = memoryview(payload)
                for offset in range(0, len(view), quantum):
                    end = min(offset + quantum, len(view))
                    # The tail earns its actual bytes BEFORE send, too.
                    if bandwidth_gbps is not None:
                        deadline = origin + send_start + end * 8 / (bandwidth_gbps * 1e9)
                        before = time.perf_counter()
                        delay = deadline - before
                        if delay > 0 and cancelled.wait(delay):
                            raise RuntimeError('Transport cancelled during pacing')
                        pacing += time.perf_counter() - before
                    offset_sent = offset
                    while offset_sent < end:
                        if cancelled.is_set():
                            raise RuntimeError('Transport cancelled during send')
                        before = time.perf_counter()
                        try:
                            count = sender.send(view[offset_sent:end])
                        except socket.timeout:
                            count = 0
                        blocking += time.perf_counter() - before
                        offset_sent += count
                row.update(send_end=clock(), pacing_wait_seconds=pacing,
                           send_blocking_seconds=blocking)
        except BaseException as error:
            errors.append(error)
            cancelled.set()

    def receiving():
        try:
            for index, payload in enumerate(messages):
                row = rows[index]
                permission_start = clock()
                if mode != 'chunk_pipeline':
                    _acquire(serial_gate, cancelled)
                capacity_start = clock()
                _acquire(slots, cancelled)
                capacity_end = clock()
                buffer = bytearray(len(payload))
                target = memoryview(buffer)
                row.update(chunk_index=index, message_bytes=len(payload),
                    receiver_permission_wait_seconds=capacity_start - permission_start,
                    capacity_wait_start=capacity_start, capacity_wait_end=capacity_end,
                    capacity_wait_seconds=capacity_end - capacity_start,
                    receive_start=clock())
                ready.put(index)
                offset = 0
                while offset < len(buffer):
                    if cancelled.is_set():
                        raise RuntimeError('Transport cancelled during receive')
                    try:
                        count = receiver.recv_into(target[offset:], min(len(buffer) - offset, 1 << 20))
                    except socket.timeout:
                        continue
                    if not count:
                        raise RuntimeError('Truncated local transport')
                    offset += count
                row['receive_end'] = clock()
                received = bytes(buffer)
                row['data_available'] = clock()
                row['effective_transfer_seconds'] = row['data_available'] - row['receive_start']
                row['actual_bandwidth_gbps'] = len(payload) * 8 / max(row['effective_transfer_seconds'], 1e-12) / 1e9
                completed.put((index, received))
        except BaseException as error:
            errors.append(error)
            cancelled.set()

    workers = [threading.Thread(target=fn, name='cachegen-' + name, daemon=True)
               for fn, name in ((sending, 'sender'), (receiving, 'receiver'))]
    value = None
    try:
        for worker in workers:
            worker.start()
        for expected in range(len(messages)):
            wait_start = clock()
            index, received = _get(completed, cancelled)
            if index != expected:
                raise RuntimeError('Receive order differs')
            row = rows[index]
            row.update(dequeue_time=clock(), main_queue_wait_start=wait_start)
            row['main_queue_wait_seconds'] = row['dequeue_time'] - wait_start
            row['queue_residence_seconds'] = row['dequeue_time'] - row['data_available']
            if mode == 'chunk_pipeline':
                slots.release()
            row['restore_start'] = clock()
            value = restore(index, received)  # callback synchronizes GPU completion
            row['restore_end'] = clock()
            row['restore_seconds'] = row['restore_end'] - row['restore_start']
            received_messages.append(received)
            if mode != 'chunk_pipeline':
                slots.release()
                serial_gate.release()
    except BaseException:
        if errors:
            raise errors[0]
        raise
    finally:
        cancelled.set()
        for endpoint in (sender, receiver):
            try:
                endpoint.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            endpoint.close()
        for worker in workers:
            if worker.ident is not None:
                worker.join(timeout=2)
        if any(w.is_alive() for w in workers):
            raise RuntimeError('Transport worker did not terminate')
    if errors:
        raise errors[0]
    result = {'value': value, 'received': received_messages, 'chunks': rows,
              'transport_wall_seconds': clock(), 'queue_capacity': queue_capacity,
              'pacing_quantum_bytes': quantum}
    if analyze_timeline:
        result.update(overlap_timeline(rows))
    return result


def overlap_timeline(rows):
    """Postprocessing; TTFT callers run this only AFTER the CPU token is ready."""
    intervals = []
    for recv_row in rows:
        for restore_row in rows:
            lo = max(recv_row['receive_start'], restore_row['restore_start'])
            hi = min(recv_row['data_available'], restore_row['restore_end'])
            if hi > lo:
                intervals.append({'receiving_chunk': recv_row['chunk_index'],
                                  'restoring_chunk': restore_row['chunk_index'], 'start': lo, 'end': hi})
    return {'overlap_intervals': intervals,
            'overlap_seconds': sum(r['end'] - r['start'] for r in intervals)}


def rate_limited_transfer(payload, bandwidth_gbps, chunk_bytes=None, *, validate=False):
    start = time.perf_counter()
    result = transfer_chunks([payload], bandwidth_gbps, lambda _, data: data,
                             mode='whole', chunk_bytes=chunk_bytes, origin=start, analyze_timeline=False)
    received, elapsed = result['value'], time.perf_counter() - start
    if validate and received != payload:
        raise RuntimeError('Local transport corrupted payload')
    return received, elapsed


class DynamicTransport:
    """One serial link for joint packets materialized after request start.

    The receiver reads each frame's header. Controller metadata never supplies
    its length. The runtime owns the capacity-one receive credit and freezes
    identity in on_start immediately before network service begins.
    """
    def __init__(self, *, bandwidth_bps=None, io_idle_timeout_s=30.,
                 max_frame_bytes=67108864, cancelled=None, origin_ns=None):
        from concurrent.futures import ThreadPoolExecutor
        self.sender, self.receiver = socket.socketpair()
        self.sender.settimeout(.1)
        self.cancelled = cancelled or threading.Event()
        self.bandwidth_bps = bandwidth_bps
        self.io_idle_timeout_s = io_idle_timeout_s
        self.max_frame_bytes = max_frame_bytes
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='joint-link')
        self.origin_ns = time.perf_counter_ns() if origin_ns is None else origin_ns

    def send(self, wire, *, identity=None, on_start=None):
        if not isinstance(wire, bytes) or not wire or len(wire) > self.max_frame_bytes:
            raise ValueError('Expected bounded immutable frame')
        return self.pool.submit(self._transfer, wire, identity, on_start)

    def _transfer(self, wire, identity, on_start):
        from experiments.joint_wire import receive_wire
        if self.cancelled.is_set():
            raise RuntimeError('transport_cancelled')
        start = time.perf_counter_ns()
        if on_start:
            frozen_start = on_start(start)
            if frozen_start is not None:
                start = frozen_start
        row = {'identity': identity, 'net_start_ns': start, 'wire_bytes': len(wire)}
        errors = []

        def send_body():
            try:
                offset = 0
                last = time.perf_counter_ns()
                while offset < len(wire):
                    if self.cancelled.is_set():
                        raise RuntimeError('transport_cancelled')
                    # Limiter reads its independent wall-clock trace; it exposes
                    # only completed byte/time observations to the controller.
                    rate = self.bandwidth_bps
                    if callable(rate):
                        rate = rate((time.perf_counter_ns() - self.origin_ns) / 1e9)
                    if rate is not None and (not math.isfinite(rate) or rate <= 0):
                        raise ValueError('Invalid link bit/s')
                    end = min(offset + 16384, len(wire))
                    if rate is not None and self.cancelled.wait((end - offset) * 8 / rate):
                        raise RuntimeError('transport_cancelled')
                    while offset < end:
                        try:
                            sent = self.sender.send(memoryview(wire)[offset:end])
                        except socket.timeout:
                            sent = 0
                        if sent:
                            offset += sent
                            last = time.perf_counter_ns()
                        elif time.perf_counter_ns() - last >= self.io_idle_timeout_s * 1e9:
                            raise TimeoutError('transport_timeout')
                        if self.cancelled.is_set():
                            raise RuntimeError('transport_cancelled')
            except BaseException as exc:
                errors.append(exc)
                self.cancelled.set()
                try:
                    self.sender.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        sending = threading.Thread(target=send_body, name='joint-send', daemon=True)
        sending.start()
        try:
            result = receive_wire(self.receiver, io_idle_timeout_s=self.io_idle_timeout_s,
                                  max_frame_bytes=self.max_frame_bytes, cancelled=self.cancelled)
            row['data_available_ns'] = time.perf_counter_ns()
        except BaseException:
            self.cancelled.set()
            for endpoint in (self.sender,self.receiver):
                try:
                    endpoint.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            raise
        finally:
            sending.join(timeout=self.io_idle_timeout_s + .2)
        if errors:
            raise errors[0]
        row['bandwidth_bps'] = 8 * len(wire) * 1e9 / max(1, row['data_available_ns'] - start)
        return result, row

    def close(self):
        self.cancelled.set()
        for endpoint in (self.sender, self.receiver):
            try:
                endpoint.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            endpoint.close()
        self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

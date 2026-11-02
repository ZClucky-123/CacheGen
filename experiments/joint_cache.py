"""Per-context CPU LRU. Pins never block GPU completion or force eviction."""
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time


@dataclass
class Entry:
    wire: bytes
    pins: int = 0
    created_ns: int = 0


class PacketCache:
    def __init__(self, capacity=268435456, on_change=lambda: None):
        if capacity < 0:
            raise ValueError('Negative cache capacity')
        self.capacity = capacity
        self.entries = OrderedDict()
        self.pending = {}
        self.lock = threading.RLock()
        self.on_change = on_change
        self.hits = self.evictions = 0
        self.resident_bytes = 0

    def peek(self, key):
        with self.lock:
            entry = self.entries.get(key)
            return None if entry is None else entry.wire

    def put(self, key, wire):
        if not isinstance(wire, bytes):
            raise TypeError('Only immutable CPU wire may enter packet cache')
        with self.lock:
            if key in self.entries:
                if self.entries[key].wire != wire:
                    raise ValueError('Immutable cache key changed wire')
                return True
            while self.resident_bytes + len(wire) > self.capacity:
                victim = next((k for k, e in self.entries.items() if not e.pins), None)
                if victim is None:
                    return False
                self.resident_bytes -= len(self.entries.pop(victim).wire)
                self.evictions += 1
            self.entries[key] = Entry(wire, created_ns=time.perf_counter_ns())
            self.resident_bytes += len(wire)
        self.on_change()
        return True

    @contextmanager
    def acquire(self, key):
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                raise KeyError(key)
            entry.pins += 1
            self.entries.move_to_end(key)
            self.hits += 1
        self.on_change()
        try:
            yield entry.wire
        finally:
            with self.lock:
                entry.pins -= 1
            self.on_change()

    def reserve(self, key):
        with self.lock:
            if key in self.pending:
                return self.pending[key], False
            future = Future()
            self.pending[key] = future
            return future, True

    def complete(self, key, wire=None, error=None):
        if error is None:
            self.put(key, wire)
        with self.lock:
            future = self.pending.pop(key)
        if error is None:
            future.set_result(wire)
        else:
            future.set_exception(error)
        self.on_change()

    def snapshot(self):
        with self.lock:
            return tuple((k, len(e.wire), e.pins) for k, e in self.entries.items())

    def clear(self):
        with self.lock:
            if self.pending or any(e.pins for e in self.entries.values()):
                raise RuntimeError('Cache still leased')
            self.entries.clear()
            self.resident_bytes = 0
        self.on_change()

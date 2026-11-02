"""Read-only master references, retained until context close or explicit release."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import uuid


@dataclass
class SourceRef:
    context_id: str
    context_ids: tuple
    raw_kv: object
    source_revision: str
    readers: int = 0
    releasing: bool = False
    metadata: dict = field(default_factory=dict)


class SourceStore:
    def __init__(self, on_change=lambda: None):
        self._items = {}
        self._condition = threading.Condition()
        self.on_change = on_change

    def register_context(self, context_ids, raw_kv, source_revision, *, context_id=None):
        with self._condition:
            key = context_id or uuid.uuid4().hex
            if key in self._items:
                old = self._items[key]
                if old.readers or old.source_revision == str(source_revision):
                    raise ValueError('Replacing a source requires a new revision and no readers')
            ref = SourceRef(key, tuple(context_ids), raw_kv, str(source_revision))
            self._items[key] = ref
        self.on_change()
        return ref

    def get(self, context_id):
        with self._condition:
            return self._items[context_id]

    def available(self, context_id):
        with self._condition:
            item = self._items[context_id]
            return item.raw_kv is not None and not item.releasing

    @contextmanager
    def lease(self, context_id, revision):
        with self._condition:
            item = self._items[context_id]
            if item.source_revision != revision or item.releasing or item.raw_kv is None:
                raise RuntimeError('source_unavailable')
            item.readers += 1
            raw = item.raw_kv
        try:
            yield raw
        finally:
            with self._condition:
                item.readers -= 1
                if item.releasing and item.readers == 0:
                    item.raw_kv = None
                self._condition.notify_all()

    def release_master(self, context_id, *, wait=True):
        with self._condition:
            item = self._items[context_id]
            item.releasing = True
            if wait:
                self._condition.wait_for(lambda: item.readers == 0)
            if item.readers == 0:
                item.raw_kv = None
        self.on_change()

    def close(self, context_id):
        self.release_master(context_id)
        with self._condition:
            del self._items[context_id]
        self.on_change()

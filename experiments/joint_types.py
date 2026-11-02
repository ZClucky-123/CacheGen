"""Joint v2 control identities; all absolute times share perf_counter_ns."""
from dataclasses import dataclass, field
import math
import time
import uuid


@dataclass(frozen=True)
class RequestClock:
    request_id: str
    t0_ns: int
    deadline_ns: int
    now: object = field(default=time.perf_counter_ns, repr=False, compare=False)

    def remaining_s(self):
        return (self.deadline_ns - self.now()) / 1e9


def begin_request(slo_s, *, now=time.perf_counter_ns):
    if not math.isfinite(slo_s) or slo_s < 0:
        raise ValueError('slo_s must be finite and nonnegative')
    start = now()
    return RequestClock(uuid.uuid4().hex, start, start + math.ceil(slo_s * 1e9), now)


@dataclass(frozen=True)
class DecisionIdentity:
    request_id: str
    block_index: int
    source_revision: str
    decision_id: int
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class Recipe:
    context_id: str
    source_revision: str
    block_index: int
    start: int
    end: int
    quality: int
    encoding: str
    model_version: str = 'mistral-fastchat8'
    quantization_version: str = 'original-v1'
    protocol: str = 'cachegen-joint-v2'

    @property
    def cache_key(self):
        return (self.context_id, self.source_revision, self.model_version,
                self.quantization_version, self.start, self.end, self.quality,
                self.encoding, self.protocol)


@dataclass(frozen=True)
class PreparedPacket:
    recipe: Recipe
    wire: bytes
    identity: DecisionIdentity
    tag: int | None = None
    generation: tuple = ()

    @property
    def size(self):
        return len(self.wire)


@dataclass(frozen=True)
class FrozenSend:
    identity: DecisionIdentity
    cache_key: tuple
    quality: int
    encoding: str
    start: int
    end: int
    net_start_ns: int


@dataclass(frozen=True)
class JobEvent:
    identity: DecisionIdentity
    event: str
    time_ns: int
    detail: dict = field(default_factory=dict)

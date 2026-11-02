"""Metadata-only nanosecond simulation of one GPU, CPU worker and link.

Generation and restore share the GPU. A completed encode retains the source
CPU slot until sealing, and a received frame retains the sole receive slot
until restore starts (pipeline) or ends (serial).
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import math

from experiments.joint_profiles import seconds_ns, NS
from experiments.joint_codec import workspace_state_from_snapshot


@dataclass(frozen=True)
class CandidateEstimate:
    block_index: int
    start: int
    end: int
    q: int
    encoding: str
    cache_key: tuple
    size50: int
    size90: int
    g50_ns: int
    g90_ns: int
    c50_ns: int
    c90_ns: int
    r50_ns: int
    r90_ns: int
    packet_state: str = 'missing'
    source_available: bool = True
    access50_ns: int = 0
    access90_ns: int = 0
    tag: int | None = None
    size_kind: str = 'estimated'
    eligible: bool = True
    encode_requirements: tuple = ()
    restore_requirements: tuple = ()
    encode_buffer_state: str | None = None
    restore_buffer_state: str | None = None
    generation_cost_available: bool = True

    @property
    def kind(self):
        return 'text' if self.encoding in ('text', 'whole_text') else 'kv'

    @property
    def quality(self):
        return self.q

    def value(self, name, percentile):
        return getattr(self, name + ('50' if percentile == .5 else '90') + '_ns')


@dataclass(frozen=True)
class Plan:
    quality: int
    candidates: tuple
    route: str = 'chunks'

    @property
    def signature(self):
        return self.quality, self.route, tuple(c.cache_key for c in self.candidates)


@dataclass(frozen=True)
class RunningTask:
    resource: str
    kind: str
    block_index: int
    remaining50_ns: int
    remaining90_ns: int
    candidate: CandidateEstimate | None = None


@dataclass(frozen=True)
class CacheEntry:
    cache_key: tuple
    size: int
    leases: int = 0


@dataclass(frozen=True)
class Snapshot:
    now_ns: int
    state_version: int = 0
    mode: str = 'pipeline'
    beta50_bps: float = 1e9
    beta90_bps: float = 1e9
    history_valid: bool = False
    committed_blocks: int = 0
    inflight: tuple = ()
    cache_entries: tuple = ()
    cache_capacity: int = 268435456
    stages: tuple = ()
    source_slot_busy: bool = False
    receive_slot_block: int | None = None
    previous_gpu: str = 'none'
    encoder_buffer: str = 'first'
    restore_buffer: str = 'first'
    encoder_capacity: int = 0
    restore_capacity: int = 0
    next_send_block: int | None = None
    generation_lookahead: int = 1
    current_encoding: str | None = None
    suffix_tokens: int = 0
    total_tokens: int = 0
    workspace: dict = field(default_factory=lambda: {'encode': {}, 'restore': {}})


@dataclass(frozen=True)
class Simulation:
    end_ns: int
    snapshot_time_ns: int
    state_version: int
    events: tuple
    wait_ns: dict
    cache_class: str
    generation_count: int
    ready_count: int
    cache_peak_bytes: int
    temporary_peak_bytes: int
    evictions: tuple
    gpu_encode_ns: int = 0
    gpu_restore_ns: int = 0
    link_idle_ns: int = 0
    workspace: dict = field(default_factory=dict)


def _mode(value):
    if value in ('pipeline', 'chunk_pipeline'):
        return 'pipeline'
    if value in ('serial', 'chunk_serial', 'whole'):
        return 'serial'
    raise ValueError('invalid execution mode')


def _cost(candidate, stage, percentile, profile, previous, buffer_state, mode, prefix):
    """Transition/resource keys can override candidate's already-converted costs."""
    default = candidate.value({'encode': 'g', 'seal': 'c', 'restore': 'r'}[stage], percentile)
    if profile is None or not getattr(profile, 'costs', None):
        return default
    matches = []
    for key, values in profile.transition_rows(candidate):
        required = {'q': 3 if candidate.kind == 'text' else candidate.q, 'encoding': candidate.encoding,
                    'N': candidate.end - candidate.start, 'prefix': prefix,
                    'previous_gpu': previous, 'buffer_state': buffer_state,
                    'mode': mode, 'tag': candidate.tag}
        good = True
        score = 0
        for name, value in required.items():
            if name not in key or key[name] is None:
                continue
            actual = key[name]
            if name == 'mode':
                actual = _mode(actual)
            if name == 'previous_gpu' and actual == 'initialize':
                actual = 'init'
            if actual != value:
                good = False
                break
            score += 1
        if good:
            name = {'encode': 'g', 'seal': 'c', 'restore': 'r'}[stage] + ('50_ns' if percentile == .5 else '90_ns')
            if name in values:
                matches.append((score, values[name]))
    if not matches:
        return default
    return max(matches, key=lambda x: x[0])[1]


def simulate_plan(plan, snapshot, profile=None, percentile=.50):
    if percentile not in (.50, .90):
        raise ValueError('percentile must be .50 or .90')
    mode = _mode(snapshot.mode)
    beta = snapshot.beta50_bps if percentile == .5 else snapshot.beta90_bps
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('bandwidth must be finite positive bit/s')
    now = snapshot.now_ns
    start_time = now
    workspace = {direction: dict(snapshot.workspace.get(direction, {}))
                 for direction in ('encode', 'restore')}
    encode_context = {}

    def workspace_state(candidate, direction, default):
        requirements = dict(getattr(candidate, direction + '_requirements'))
        if requirements:
            return workspace_state_from_snapshot(workspace, requirements, direction)
        if candidate.kind == 'text':
            return 'allocates_each_restore'
        return default

    def reserve_workspace(candidate, direction):
        for slot, capacity in getattr(candidate, direction + '_requirements'):
            workspace[direction][slot] = max(workspace[direction].get(slot, 0), capacity)
    events = []
    states = {}
    candidates = {}
    selected_keys = {c.cache_key for c in plan.candidates}
    for candidate in plan.candidates:
        if candidate.block_index < snapshot.committed_blocks:
            continue
        states[candidate.block_index] = 'pending'
        candidates[candidate.block_index] = candidate
    for candidate, stage in snapshot.stages:
        if candidate.block_index >= snapshot.committed_blocks:
            candidates.setdefault(candidate.block_index, candidate)
            if candidates[candidate.block_index].cache_key == candidate.cache_key:
                states[candidate.block_index] = stage
    if not candidates:
        h = (getattr(profile, 'h50_ns', 0) if percentile == .5 else getattr(profile, 'h90_ns', 0))
        return Simulation(now + h, now, snapshot.state_version, (), {}, 'warm', 0, 0, 0, 0, (), workspace=workspace)
    indices = sorted(candidates)
    committed = snapshot.committed_blocks
    next_send = snapshot.next_send_block
    if next_send is None:
        next_send = indices[0]
    # Resource -> (finish, kind, block, candidate, original_start).
    running = {}
    for task in snapshot.inflight:
        duration = task.remaining50_ns if percentile == .5 else task.remaining90_ns
        if duration < 0:
            raise ValueError('negative remaining task time')
        if task.resource in running:
            raise ValueError('two running jobs on one resource')
        running[task.resource] = (now + duration, task.kind, task.block_index, task.candidate, now)
        if task.candidate is not None and task.block_index >= committed:
            candidates.setdefault(task.block_index, task.candidate)
            if candidates[task.block_index].cache_key == task.candidate.cache_key:
                states[task.block_index] = {'encode': 'encoding', 'seal': 'sealing',
                                            'network': 'network', 'restore': 'restoring',
                                            'select': 'selecting'}.get(task.kind, 'pending')
        if task.kind == 'network':
            next_send = max(next_send, task.block_index + 1)
        if task.candidate is not None and task.kind in ('encode', 'restore'):
            if task.kind == 'encode':
                state = task.candidate.encode_buffer_state or workspace_state(task.candidate, 'encode', snapshot.encoder_buffer)
                encode_context[task.candidate.cache_key] = (state, snapshot.previous_gpu)
            # Started nonpreemptible tasks finish their real allocation before
            # a later same-direction task can acquire the GPU/workspace.
            reserve_workspace(task.candidate, task.kind)
    indices = sorted(candidates)
    first_plan = min((c.block_index for c in plan.candidates), default=indices[0])
    receive_slot = snapshot.receive_slot_block
    for task in snapshot.inflight:
        if task.kind in ('network', 'restore') and receive_slot is None:
            if task.kind == 'network' or mode == 'serial':
                receive_slot = task.block_index
    for i, stage in states.items():
        if stage == 'received' and receive_slot is None:
            receive_slot = i
        if stage in ('network', 'received', 'restoring', 'done'):
            next_send = max(next_send, i + 1)
    source_slot = snapshot.source_slot_busy or any(t.kind in ('encode', 'seal') for t in snapshot.inflight)
    previous_gpu = snapshot.previous_gpu
    encoder_buffer = snapshot.encoder_buffer
    restore_buffer = snapshot.restore_buffer
    encoder_capacity = snapshot.encoder_capacity
    restore_capacity = snapshot.restore_capacity
    history_valid = snapshot.history_valid
    cache = OrderedDict()
    leases = {}
    for entry in snapshot.cache_entries:
        if isinstance(entry, CacheEntry):
            key, size, leased = entry.cache_key, entry.size, entry.leases
        elif isinstance(entry, dict):
            key, size, leased = tuple(entry['cache_key']), entry['size'], entry.get('leases', 0)
        else:
            key, size, *rest = entry
            leased = rest[0] if rest else 0
        cache[key] = size
        leases[key] = leased
    for candidate in candidates.values():
        if candidate.packet_state == 'ready' and candidate.cache_key not in cache:
            # A snapshot without a full LRU may still expose actual ready metadata.
            cache[candidate.cache_key] = candidate.size50
            leases[candidate.cache_key] = 0
    ready_before = {c.cache_key for c in candidates.values()
                    if states[c.block_index] in ('ready', 'network', 'received', 'restoring')}
    held = set(ready_before)
    generated = {c.cache_key for i, c in candidates.items()
                 if states.get(i) in ('encode_queued', 'encoding', 'seal_queued', 'sealing')}
    temporary = {}
    cache_peak = sum(cache.values())
    temporary_peak = 0
    evictions = []
    wait_ns = {'gpu_encode': 0, 'gpu_restore': 0, 'cpu_seal': 0, 'cpu_select': 0,
               'network': 0}
    queued_at = {(i, stage): now for i, stage in states.items()}
    orphan_seals = []
    gpu_encode_ns = gpu_restore_ns = 0
    net_busy_ns = 0
    net_first = None
    net_last = None

    def emit(kind, i):
        events.append((now, kind, i))

    def change(i, stage):
        states[i] = stage
        queued_at[i, stage] = now

    def is_current(c):
        return c is not None and c.block_index in candidates and candidates[c.block_index].cache_key == c.cache_key

    def insert(candidate):
        nonlocal cache_peak, temporary_peak
        key = candidate.cache_key
        size = candidate.size50 if percentile == .5 else candidate.size90
        held.add(key)
        if key in cache:
            cache.move_to_end(key)
            return
        while sum(cache.values()) + size > snapshot.cache_capacity:
            victim = next((k for k in cache if k not in held and not leases.get(k, 0)), None)
            if victim is None:
                temporary[key] = size
                temporary_peak = max(temporary_peak, sum(temporary.values()))
                return
            evictions.append(victim)
            del cache[victim]
        cache[key] = size
        cache_peak = max(cache_peak, sum(cache.values()))

    def complete(resource, task):
        nonlocal source_slot, receive_slot, previous_gpu, committed, history_valid
        _, kind, i, candidate, _ = task
        current = is_current(candidate)
        if kind == 'encode':
            previous_gpu = 'encode'
            emit('ENCODE_GPU_DONE', i)
            if current:
                change(i, 'seal_queued')
            else:
                # A nonpreemptible old decision still owns its fields/pinned
                # slot. Its seal precedes new selection work and may cache a
                # valid old wire, but never makes that wire the chosen route.
                orphan_seals.append((candidate, now))
        elif kind == 'seal':
            source_slot = False
            emit('PACKET_READY', i)
            if candidate is not None:
                insert(candidate)
            if current:
                change(i, 'ready')
        elif kind == 'select':
            emit('SELECT_DONE', i)
            if current:
                key = candidate.cache_key
                if key in cache or key in temporary:
                    if key in cache:
                        cache.move_to_end(key)
                    held.add(key)
                    ready_before.add(key)
                    change(i, 'ready')
                elif candidate.kind == 'text':
                    if not candidate.generation_cost_available:
                        raise ValueError('cost_model_missing')
                    generated.add(key)
                    change(i, 'seal_queued')
                else:
                    if not candidate.source_available:
                        raise ValueError('source_unavailable')
                    if not candidate.generation_cost_available:
                        raise ValueError('cost_model_missing')
                    generated.add(key)
                    change(i, 'encode_queued')
        elif kind == 'network':
            emit('DATA_AVAILABLE', i)
            if current:
                change(i, 'received')
            history_valid = history_valid or (candidate is not None and candidate.size50 >= 262144)
        elif kind == 'restore':
            previous_gpu = 'restore'
            emit('RESTORE_DONE', i)
            if current:
                change(i, 'done')
                committed = i + 1
                held.discard(candidate.cache_key)
                temporary.pop(candidate.cache_key, None)
            if mode == 'serial':
                receive_slot = None
        elif kind == 'init':
            previous_gpu = 'init'

    def start(resource, kind, i, candidate, duration):
        if duration < 0:
            raise ValueError('negative service time')
        running[resource] = (now + duration, kind, i, candidate, now)

    def can_select(i):
        if mode == 'serial':
            return i == committed and all(stage == 'done' for j, stage in states.items() if j < i)
        if i == first_plan and not any(j < i for j in states):
            return True
        if not snapshot.generation_lookahead:
            return i == committed
        if i > next_send:
            return False
        previous = max((j for j in states if j < i), default=None)
        if previous is None:
            return True
        if history_valid:
            return states[previous] in ('network', 'received', 'restoring', 'done')
        return states[previous] in ('received', 'restoring', 'done')

    while True:
        # Process all simultaneous completions before dispatching priority queues.
        finished = [(resource, task) for resource, task in running.items() if task[0] <= now]
        for resource, task in finished:
            del running[resource]
        for resource, task in finished:
            complete(resource, task)
        if all(states.get(c.block_index) == 'done' for c in plan.candidates):
            break
        if 'cpu' not in running:
            seals = [i for i in indices if states.get(i) == 'seal_queued']
            selections = [i for i in indices if states.get(i) == 'pending' and can_select(i)]
            if orphan_seals:
                c, queued = orphan_seals.pop(0)
                wait_ns['cpu_seal'] += now - queued
                buffer_state, predecessor = encode_context.get(c.cache_key, (c.encode_buffer_state or encoder_buffer, previous_gpu))
                duration = _cost(c, 'seal', percentile, profile, predecessor, buffer_state, mode, c.start)
                start('cpu', 'seal', c.block_index, c, duration)
            elif seals:
                i = seals[0]
                c = candidates[i]
                wait_ns['cpu_seal'] += now - queued_at.get((i, 'seal_queued'), now)
                buffer_state, predecessor = encode_context.get(c.cache_key,
                    (c.encode_buffer_state or workspace_state(c, 'encode', encoder_buffer), previous_gpu))
                duration = _cost(c, 'seal', percentile, profile, predecessor, buffer_state, mode, c.start)
                change(i, 'sealing')
                start('cpu', 'seal', i, c, duration)
            elif selections:
                i = selections[0]
                c = candidates[i]
                # Current SELECT has already run; only its commit/access remains.
                name = ('commit' if i == first_plan else 'select') + ('50_ns' if percentile == .5 else '90_ns')
                duration = getattr(profile, name, 0) + c.value('access', percentile)
                wait_ns['cpu_select'] += now - queued_at.get((i, 'pending'), now)
                change(i, 'selecting')
                start('cpu', 'select', i, c, duration)
        if 'gpu' not in running:
            restores = [i for i in indices if states.get(i) == 'received' and i == committed]
            encodes = [i for i in indices if states.get(i) == 'encode_queued']
            if restores:
                i = restores[0]
                c = candidates[i]
                wait_ns['gpu_restore'] += now - queued_at.get((i, 'received'), now)
                if mode == 'pipeline':
                    receive_slot = None
                capacity = c.end - c.start
                default_state = ('expand' if restore_capacity and capacity > restore_capacity else restore_buffer)
                buffer_state = workspace_state(c, 'restore', default_state)
                duration = _cost(c, 'restore', percentile, profile, previous_gpu, buffer_state, mode, c.start)
                reserve_workspace(c, 'restore')
                gpu_restore_ns += duration
                restore_buffer = 'reuse'
                restore_capacity = max(restore_capacity, capacity)
                change(i, 'restoring')
                emit('RESTORE_START', i)
                start('gpu', 'restore', i, c, duration)
            elif encodes and not source_slot:
                i = encodes[0]
                c = candidates[i]
                wait_ns['gpu_encode'] += now - queued_at.get((i, 'encode_queued'), now)
                capacity = c.end - c.start
                default_state = ('expand' if encoder_capacity and capacity > encoder_capacity else encoder_buffer)
                buffer_state = workspace_state(c, 'encode', default_state)
                encode_context[c.cache_key] = (buffer_state, previous_gpu)
                duration = _cost(c, 'encode', percentile, profile, previous_gpu, buffer_state, mode, c.start)
                reserve_workspace(c, 'encode')
                gpu_encode_ns += duration
                encoder_buffer = 'reuse'
                encoder_capacity = max(encoder_capacity, capacity)
                source_slot = True
                change(i, 'encoding')
                emit('ENCODE_START', i)
                start('gpu', 'encode', i, c, duration)
        if 'network' not in running and receive_slot is None and states.get(next_send) == 'ready':
            i = next_send
            c = candidates[i]
            size = c.size50 if percentile == .5 else c.size90
            duration = math.ceil(8 * size * NS / beta)
            wait_ns['network'] += now - queued_at.get((i, 'ready'), now)
            receive_slot = i
            next_send += 1
            change(i, 'network')
            emit('NET_START', i)
            net_busy_ns += duration
            net_first = now if net_first is None else net_first
            net_last = now + duration
            start('network', 'network', i, c, duration)
        if not running:
            raise ValueError('simulation_deadlock: inconsistent snapshot, prefix or unavailable resource')
        # NET_START can open the generation window at this same instant.
        # Dispatch that zero-delay CPU work before advancing to a network end.
        if 'cpu' not in running and any(states.get(i) == 'pending' and can_select(i) for i in indices):
            continue
        now = min(task[0] for task in running.values())
    total = snapshot.total_tokens or max(c.end for c in plan.candidates)
    h = profile.h_ns(total, snapshot.suffix_tokens, percentile) if profile is not None else 0
    gen_count = len(generated & selected_keys)
    ready_count = sum(c.cache_key not in generated for c in plan.candidates)
    cache_class = 'cold' if ready_count == 0 else ('warm' if gen_count == 0 else 'mixed')
    return Simulation(now + h, start_time, snapshot.state_version, tuple(events), wait_ns,
                      cache_class, gen_count, ready_count, cache_peak, temporary_peak,
                      tuple(evictions), gpu_encode_ns, gpu_restore_ns,
                      max(0, (net_last or start_time) - (net_first or start_time) - net_busy_ns), workspace)

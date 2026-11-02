"""Joint request execution on one GPU executor, one CPU controller and one link."""
from concurrent.futures import CancelledError, Future
from contextlib import ExitStack, nullcontext, contextmanager
from dataclasses import asdict
import math
import queue
import threading
import time
from types import SimpleNamespace

from experiments.joint_types import (begin_request, DecisionIdentity, Recipe,
                                     PreparedPacket, FrozenSend)
from experiments.joint_source import SourceStore
from experiments.joint_cache import PacketCache
from experiments.joint_wire import frame, unframe
from mini_repro.transport import DynamicTransport


class GPUExecutor:
    """Nonpreemptive priority queue. Every CUDA operation uses the default stream."""
    def __init__(self, device=None, on_change=lambda: None, on_failure=lambda kind,error:None):
        self.device = device
        self.tasks = queue.PriorityQueue()
        self.serial = 0
        self.lock = threading.RLock()
        self.running = None
        self.previous = 'none'
        self.on_change = on_change
        self.on_failure = on_failure
        self.worker = threading.Thread(target=self._work, name='joint-gpu', daemon=True)
        self.worker.start()

    def submit(self, kind, fn, *, valid=lambda: True, on_start=None, metadata=None):
        future = Future()
        with self.lock:
            self.serial += 1
            self.tasks.put((0 if kind in ('restore', 'init', 'first') else 1,
                            self.serial, kind, fn, future, valid, on_start, metadata))
        self.on_change()
        return future

    def _work(self):
        while True:
            item = self.tasks.get()
            _, _, kind, fn, future, valid, on_start, metadata = item
            if kind == 'stop':
                return
            if not valid():
                future.cancel()
                self.on_change()
                continue
            if not future.set_running_or_notify_cancel():
                continue
            with self.lock:
                self.running = (kind, time.perf_counter_ns(), metadata)
            self.on_change()
            try:
                if self.device is not None:
                    import torch
                    scope = torch.cuda.device(self.device)
                else:
                    scope = nullcontext()
                with scope:
                    if self.device is not None:
                        import torch
                        stream = torch.cuda.stream(torch.cuda.default_stream(self.device))
                    else:
                        stream = nullcontext()
                    with stream:
                        if on_start:
                            on_start()
                        result = fn()
                        if self.device is not None:
                            torch.cuda.default_stream(self.device).synchronize()
                future.set_result(result)
            except BaseException as exc:
                if self.device is not None:
                    # A recoverable Python/OOM failure may follow asynchronous
                    # launches. Complete trusted device work before the Future
                    # can release any task-owned lease or buffer.
                    try:
                        import torch
                        torch.cuda.default_stream(self.device).synchronize()
                    except BaseException as device_exc:
                        exc=RuntimeError(f'device_error: {device_exc}')
                self.on_failure(kind,exc)
                future.set_exception(exc)
            finally:
                with self.lock:
                    self.running = None
                    self.previous = kind
                self.on_change()

    def close(self):
        # Running CUDA is allowed to finish; no lease is released before join.
        with self.lock:
            while True:
                try:
                    item = self.tasks.get_nowait()
                    item[4].cancel()
                except queue.Empty:
                    break
            self.serial += 1
            self.tasks.put((2, self.serial, 'stop', None, None, None, None, None))
        self.worker.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class JointState:
    def __init__(self, clock):
        self.clock = clock
        self.lock = threading.RLock()
        self.active = True
        self.error = None
        self.state_version = 0
        self.choices = {}
        self.choice_values = {}
        self.frozen = {}
        self.committed_tokens = 0
        self.attempted = set()
        self.failed = set()
        self.wake = threading.Event()

    def changed(self):
        with self.lock:
            self.state_version += 1
        self.wake.set()

    def select(self, index, revision, value, version):
        with self.lock:
            if not self.active or index in self.frozen or version != self.state_version:
                raise RuntimeError('stale_snapshot')
            previous = self.choices.get(index)
            if previous and self.choice_values[index] == value:
                return previous
            identity = DecisionIdentity(self.clock.request_id, index, revision,
                                        1 if previous is None else previous.decision_id + 1)
            self.choices[index] = identity
            self.choice_values[index] = value
            self.state_version += 1
        return identity

    def valid(self, identity):
        with self.lock:
            return self.active and self.choices.get(identity.block_index) == identity

    def freeze(self, packet, now):
        with self.lock:
            i, r = packet.identity.block_index, packet.recipe
            if not self.valid(packet.identity) or i in self.frozen or i != len(self.frozen):
                raise RuntimeError('stale_send_identity')
            if self.choice_values[i] != (r.quality,r.encoding) or packet.identity.source_revision != r.source_revision:
                raise RuntimeError('stale_wire_identity')
            record = FrozenSend(packet.identity, r.cache_key, r.quality, r.encoding,
                                r.start, r.end, now)
            self.frozen[i] = record
            self.state_version += 1
            return record

    def commit(self, record):
        with self.lock:
            if not self.active or self.frozen.get(record.identity.block_index) != record:
                raise RuntimeError('stale_restore_identity')
            if record.start != self.committed_tokens:
                raise RuntimeError('invalid_committed_prefix')
            self.committed_tokens = record.end
            self.state_version += 1
        self.wake.set()

    def cancel(self):
        with self.lock:
            self.active = False
            self.state_version += 1
        self.wake.set()


def _ns(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid measured service cost')
    return math.ceil(seconds * 1e9)


@contextmanager
def _request_guard(state):
    try:
        yield
    except BaseException:
        state.cancel()
        raise


class Runtime:
    def __init__(self, engine, codec, profile, config):
        self.engine, self.codec, self.profile = engine, codec, profile
        self.config = config.get('joint', config)
        self.sources = SourceStore()
        self.caches = {}
        self.current_state = None
        self.current_transport = None
        self.cost_index = {}
        costs = getattr(profile,'costs',self.config.get('costs',[]))
        if isinstance(costs,dict):
            costs=list(costs.values())
        for row in costs:
            self.cost_index.setdefault((row.get('q'),row.get('encoding'),row.get('N'),
                                       row.get('prefix',0),row.get('mode','pipeline')),[]).append(row)

    def register_context(self, ids, raw, revision):
        start = time.perf_counter_ns()
        ref = self.sources.register_context(ids, raw, revision)
        ref.metadata.update(registration_start_ns=start, registration_end_ns=time.perf_counter_ns(), used=False)
        ref.metadata['source_bytes'] = raw.numel()*raw.element_size() if hasattr(raw,'numel') else None
        self.caches[ref.context_id] = PacketCache(self.config.get('packet_cache_bytes_per_context', 268435456))
        return ref

    def close_context(self, context):
        self.sources.close(context.context_id)
        self.caches.pop(context.context_id).clear()

    def _recipe(self,context,index,start,end,q,encoding):
        spec=self.engine.spec
        model='@'.join(str(spec.get(k,'cpu-reference')) for k in ('id','revision','weight_loading'))
        return Recipe(context.context_id,context.source_revision,index,start,end,q,encoding,model_version=model)

    def cancel(self):
        if self.current_state:
            self.current_state.cancel()
        if self.current_transport:
            self.current_transport.close()

    def _allocate(self, context):
        if hasattr(self.engine, 'allocate_final'):
            return self.engine.allocate_final(len(context.context_ids))
        import torch
        spec = self.engine.spec
        return torch.empty((spec['layers'], 2, spec['kv_heads'], len(context.context_ids), spec['head_dim']),
                           dtype=torch.float16, device='cuda')

    def _workspace_snapshot(self):
        snapshot = (self.codec.workspace_snapshot() if hasattr(self.codec, 'workspace_snapshot')
                    else {'encode': {}, 'restore': {}})
        return {direction: dict(snapshot.get(direction, {})) for direction in ('encode', 'restore')}

    def _candidates(self, context, first, ready, state, encodings, route=None, fixed=False, workspace=None):
        from experiments.joint_scheduler import CandidateEstimate
        from experiments.joint_codec import workspace_requirements, workspace_state_from_snapshot
        cache = self.caches[context.context_id]
        workspace = self._workspace_snapshot() if workspace is None else workspace
        spec = self.engine.spec
        T, chunk = len(context.context_ids), self.config.get('chunk_tokens', 256)
        candidates = []
        blocks=[(i,i*chunk,min(T,(i+1)*chunk),False) for i in range(first,(T+chunk-1)//chunk)]
        if first==0 and self.profile.text_admitted and route!='chunks':
            blocks.append((0,0,T,True))
        for i,start,end,whole in blocks:
            if route=='whole_text' and not whole:
                continue
            for q in self.config.get('qualities', [1, 2, 3]):
                if whole and q!=3:
                    continue
                if T == 4096 and q == 1 and not fixed:
                    continue
                methods = ('whole_text',) if whole else tuple(encodings) + (('text',) if self.profile.text_admitted else ())
                for m in methods:
                    recipe = self._recipe(context,i,start,end,q,m)
                    if recipe.cache_key in state.failed:
                        continue
                    wire = ready.get(recipe.cache_key)
                    if wire is None:
                        wire = cache.peek(recipe.cache_key)
                    hit = wire is not None
                    available = m in ('text','whole_text') or self.sources.available(context.context_id)
                    if not hit and (not available or recipe.cache_key in state.attempted):
                        continue
                    matches = self.cost_index.get((3 if m in ('text','whole_text') else q,m,end-start,start,self._mode),[])
                    if not matches:
                        continue
                    tag=None
                    if m in ('ac_count','ac_cdf'):
                        if hit:
                            from experiments.joint_codec import inspect_wire
                            tag=inspect_wire(wire).get('cdf_tag')
                        else:
                            tag=0x10 if m=='ac_cdf' else 0x12
                    requirements = {direction: (workspace_requirements(q,m,end-start,
                        spec['layers'],spec['kv_heads'],spec['head_dim'],direction,tag)
                        if m not in ('text','whole_text') else {}) for direction in ('encode','restore')}
                    buffer_states = {direction: (workspace_state_from_snapshot(workspace, required, direction)
                        if required else 'allocates_each_restore') for direction, required in requirements.items()}
                    # Concrete byte capacities determine first/reuse/expand for
                    # this snapshot. Unknown future predecessors use a maximum
                    # over applicable transitions; the simulator reprices later
                    # task states as virtual allocations change capacities.
                    def val(name, default=None):
                        direction = 'encode' if name.startswith(('g50_','g90_','c50_','c90_')) else (
                                    'restore' if name.startswith(('r50_','r90_')) else None)
                        rows = [r for r in matches if (tag is None or r.get('tag') in (None,tag)) and
                                (direction is None or r.get('buffer_state') in (None,buffer_states[direction]))]
                        values = [r[name] for r in rows if name in r]
                        return max(values) if values else default
                    generation_cost_available = all(val(k) is not None for k in ('g50_s','g90_s','c50_s','c90_s'))
                    if not hit and not generation_cost_available:
                        continue
                    if any(val(k) is None for k in ('r50_s','r90_s')):
                        continue
                    if hit:
                        s50 = s90 = len(wire)
                    elif m == 'fixed_bits':
                        from experiments.joint_codec import fixed_wire_size
                        s50 = s90 = fixed_wire_size(spec['layers'], spec['kv_heads'], spec['head_dim'], end-start, q)
                    else:
                        s50, s90 = val('size50'), val('size90')
                        if s50 is None or s90 is None:
                            continue
                    candidates.append(CandidateEstimate(i,start,end,q,m,recipe.cache_key,
                        int(s50),int(s90),_ns(val('g50_s',0)),
                        _ns(val('g90_s',0)),_ns(val('c50_s',0)),
                        _ns(val('c90_s',0)),_ns(val('r50_s')),_ns(val('r90_s')),
                        packet_state='ready' if hit else 'missing',source_available=available,
                        access50_ns=_ns(val('access50_s',0)),access90_ns=_ns(val('access90_s',0)),
                        tag=tag,size_kind='exact' if hit or m=='fixed_bits' else 'estimated',
                        encode_requirements=tuple(requirements['encode'].items()),
                        restore_requirements=tuple(requirements['restore'].items()),
                        encode_buffer_state=buffer_states['encode'],restore_buffer_state=buffer_states['restore'],
                        generation_cost_available=generation_cost_available))
        return tuple(candidates)

    def run(self, context, suffix, slo_s, *, bandwidth_bps=None, quality_mode=None,
            fixed_quality=None, encodings=None, validation_only=False, residual_mode='normal', clock=None,
            route=None, diagnostic_only=False, link_id='local', initial_history=None):
        """Execute only committed selections; no unselected candidate reads KV."""
        from experiments.joint_policy import choose_joint, refresh_decision, BandwidthHistory
        from experiments.joint_scheduler import Snapshot, RunningTask
        from experiments.joint_types import RequestClock
        if clock is None:
            clock = begin_request(slo_s)
            if not context.metadata['used']:
                t0 = context.metadata['registration_start_ns']
                clock = RequestClock(clock.request_id,t0,t0+math.ceil(slo_s*1e9))
        context.metadata['used'] = True
        state = JointState(clock)
        self.current_state = state
        cache = self.caches[context.context_id]
        self._mode = self.config.get('execution_mode', 'pipeline')
        encodings = encodings or self.config.get('encodings', ('ac_count','ac_cdf','fixed_bits'))
        feedback = BandwidthHistory(link_id=link_id)
        if initial_history is not None:
            feedback.import_completed(initial_history,clock.t0_ns)
        chunk, T = self.config.get('chunk_tokens',256), len(context.context_ids)
        count = (T+chunk-1)//chunk
        ready, jobs, restore_jobs, records, network_rows, generations, events = {}, {}, {}, [], [], [], []
        sent_candidates={}
        pruned_blocks=set()
        actual_route='chunks'
        encode_metrics={}
        cache_peak_bytes=cache.resident_bytes
        temporary_peak_bytes=0
        calls_before=dict(getattr(self.codec,'calls',{}))
        next_send = completed = 0
        net = None
        net_packet = None
        receive_credit = threading.Event()
        receive_credit.set()
        held = {}
        cache.on_change = state.changed
        self.sources.on_change = state.changed
        cuda_device = 'cuda' if str(getattr(self.codec,'device','cpu')).startswith('cuda') else None

        def event(name, identity=None, **detail):
            events.append(dict(event=name,time_ns=time.perf_counter_ns(),
                               identity=asdict(identity) if identity else None,**detail))
            state.wake.set()

        def restore(packet, wire, record):
            if not state.active or state.frozen.get(record.identity.block_index) != record:
                raise RuntimeError('stale_restore_identity')
            if record.start != state.committed_tokens:
                raise RuntimeError('invalid_committed_prefix')
            event('RESTORE_START',record.identity)
            if self._mode == 'pipeline':
                receive_credit.set()
            if packet.recipe.encoding in ('text','whole_text'):
                meta, body = unframe(wire)
                ids = self.engine.restore_text(body)
                if ids != list(context.context_ids[record.start:record.end]):
                    raise ValueError('Text token identity differs')
                prefix = None if record.start == 0 else final[:, :, :, :record.start, :]
                computed = self.engine.prefill(ids, prefix)
                final[:, :, :, record.start:record.end, :] = computed[:, :, :, record.start:record.end, :]
            else:
                self.codec.restore(wire, final, record.start)
            if cuda_device:
                import torch
                torch.cuda.default_stream().synchronize()
            state.commit(record)
            event('RESTORE_DONE',record.identity,**getattr(self.codec,'last_restore_metrics',{}))
            if self._mode != 'pipeline':
                receive_credit.set()

        def snapshot_metadata():
            now = time.perf_counter_ns()
            running = []
            with gpu.lock:
                if gpu.running:
                    kind, began, candidate = gpu.running
                    if candidate is not None:
                        a,b = ('g50_ns','g90_ns') if kind=='encode' else ('r50_ns','r90_ns')
                        running.append(RunningTask('gpu',kind,candidate.block_index,
                            max(0,getattr(candidate,a)-(now-began)),max(0,getattr(candidate,b)-(now-began)),candidate))
            if net is not None and not net.done():
                cand = net_packet[1]
                elapsed = now - state.frozen[cand.block_index].net_start_ns if cand.block_index in state.frozen else 0
                running.append(RunningTask('network','network',cand.block_index,
                    max(0,math.ceil(8*len(net_packet[0].wire)*1e9/feedback.beta50_bps)-elapsed),
                    max(0,math.ceil(8*len(net_packet[0].wire)*1e9/feedback.beta90_bps)-elapsed),cand))
            stages=[]
            for i,(_,candidate,_,_,_) in jobs.items():
                if not any(t.block_index==i and t.kind=='encode' for t in running):
                    stages.append((candidate,'encode_queued'))
            for i,candidate in sent_candidates.items():
                if candidate.end<=state.committed_tokens:
                    continue
                if not any(t.block_index==i for t in running):
                    stages.append((candidate,'received'))
            if pending_packet is not None:
                stages.append((pending_packet[1],'ready'))
            return Snapshot(now,state.state_version,mode=self._mode,beta50_bps=feedback.beta50_bps,
                            beta90_bps=feedback.beta90_bps,history_valid=feedback.valid,
                            committed_blocks=completed,inflight=tuple(running),cache_entries=cache.snapshot(),
                            cache_capacity=cache.capacity,stages=tuple(stages),
                            source_slot_busy=bool(jobs),receive_slot_block=None if receive_credit.is_set() else next_send-1,
                            previous_gpu=gpu.previous,next_send_block=next_send,workspace=self._workspace_snapshot(),
                            generation_lookahead=self.config.get('generation_lookahead',1),
                            current_encoding=state.choice_values.get(next_send,(None,None))[1],
                            total_tokens=T,suffix_tokens=len(suffix))

        def snapshot():
            # Copy the version and queue/resource metadata in one short control
            # critical section. Forecasting, CUDA and socket work occur outside.
            with state.lock:
                return snapshot_metadata()

        def decide(index):
            began = time.perf_counter_ns()
            while state.active:
                snap = snapshot()
                fixed=(quality_mode or self.config.get('quality_mode','adaptive'))=='fixed'
                candidates = self._candidates(context,index,ready,state,encodings,route,fixed,workspace=snap.workspace)
                if not candidates and not self.sources.available(context.context_id) and not self.profile.text_admitted:
                    raise RuntimeError('source_unavailable')
                whole=next((c for c in candidates if c.encoding=='whole_text'),None)
                request = SimpleNamespace(clock=clock,min_quality=self.config.get('min_quality',1),
                    quality_mode='adaptive' if route=='whole_text' else quality_mode or self.config.get('quality_mode','adaptive'),
                    fixed_quality=fixed_quality or self.config.get('fixed_quality'),total_tokens=T,
                    whole_text=whole,validation_only=validation_only,residual_mode=residual_mode,
                    diagnostic_only=diagnostic_only)
                decision = choose_joint(candidates,snap,self.profile,request)
                fresh=snapshot()
                if fresh.state_version!=snap.state_version:
                    continue
                refreshed=refresh_decision(decision,fresh,self.profile,request)
                if not decision.predicted_slo_miss and refreshed.predicted_slo_miss:
                    # Selection CPU work consumed the previous plan's slack.
                    # Reevaluate quality/routes at the new time; eventually the
                    # fastest allowed miss remains valid if the budget expires.
                    continue
                decision=refreshed
                candidate = decision.candidate
                try:
                    identity = state.select(index,context.source_revision,(candidate.q,candidate.encoding),fresh.state_version)
                except RuntimeError as exc:
                    if str(exc)=='stale_snapshot':
                        continue
                    raise
                row = asdict(decision.forecast)
                row.update(request_id=clock.request_id,block_index=index,quality=candidate.q,route=decision.route,
                           encoding=candidate.encoding,decision_id=identity.decision_id,
                           decision_ns=time.perf_counter_ns(),select_ns=time.perf_counter_ns()-began)
                records.append(row)
                event('SELECT_DONE',identity,encoding=candidate.encoding,q=candidate.q)
                return candidate, identity
            raise RuntimeError('request_cancelled')

        def materialize(candidate, identity):
            recipe = self._recipe(context,candidate.block_index,candidate.start,candidate.end,
                                  candidate.q,candidate.encoding)
            key = recipe.cache_key
            if key in ready:
                return PreparedPacket(recipe,ready[key],identity)
            wire = cache.peek(key)
            if wire is not None:
                wire = held.setdefault(candidate.block_index,ExitStack()).enter_context(cache.acquire(key))
                ready[key] = wire
                return PreparedPacket(recipe,wire,identity)
            state.attempted.add(key)
            shared, owner = cache.reserve(key)
            if not owner:
                jobs[candidate.block_index]=(shared,candidate,identity,recipe,False)
                return None

            def generate():
                event('ENCODE_START',identity)
                start = time.perf_counter_ns()
                if recipe.encoding in ('text','whole_text'):
                    result = ('text',list(context.context_ids[recipe.start:recipe.end]))
                else:
                    with self.sources.lease(context.context_id,context.source_revision) as raw:
                        piece = raw[:, :, :, recipe.start:recipe.end, :]
                        try:
                            result = self.codec.encode_gpu(piece,recipe.quality,recipe.encoding,
                                                           recipe.start,T)
                        finally:
                            if cuda_device:
                                import torch
                                torch.cuda.default_stream().synchronize()
                elapsed = time.perf_counter_ns()-start
                encode_metrics[identity.job_id]=dict(getattr(self.codec,'last_metrics',{}))
                event('ENCODE_GPU_DONE',identity,g_encode_ns=elapsed)
                return result, elapsed
            future = gpu.submit('encode',generate,valid=lambda: state.valid(identity), metadata=candidate)
            jobs[candidate.block_index]=(future,candidate,identity,recipe,True)
            return None

        try:
            def failed_task(kind,error):
                if kind!='encode' or 'device_error' in str(error).lower():
                    state.error=error
                    state.cancel()
            with GPUExecutor(cuda_device,state.changed,failed_task) as gpu, DynamicTransport(
                bandwidth_bps=bandwidth_bps,io_idle_timeout_s=self.config.get('io_idle_timeout_s',30),
                max_frame_bytes=self.config.get('max_frame_bytes',67108864),
                cancelled=threading.Event(),origin_ns=clock.t0_ns) as link, _request_guard(state):
                self.current_transport = link
                def allocate_final():
                    if cuda_device:
                        import torch
                        torch.cuda.reset_peak_memory_stats()
                    event('FINAL_ALLOCATE_START')
                    return self._allocate(context)
                final = gpu.submit('init',allocate_final).result()
                event('FINAL_ALLOCATED')
                pending_packet = None
                while completed < count:
                    if not state.active:
                        raise state.error or RuntimeError('request_cancelled')
                    # CPU seal has priority over the next selector invocation.
                    for i,(future,cand,identity,recipe,owner) in list(jobs.items()):
                        if not future.done():
                            continue
                        del jobs[i]
                        try:
                            if owner:
                                staged,g_ns = future.result()
                                seal_start = time.perf_counter_ns()
                                if recipe.encoding in ('text','whole_text'):
                                    wire = frame(self.engine.text_payload(staged[1]),
                                        {'token_start':recipe.start,'token_end':recipe.end,'method':'text'})
                                else:
                                    wire = self.codec.seal(staged)
                                cache.complete(recipe.cache_key,wire)
                                c_ns = time.perf_counter_ns()-seal_start
                                generations.append(dict(block_index=i,q=recipe.quality,encoding=recipe.encoding,
                                    g_encode_ns=g_ns,c_seal_ns=c_ns,wire_bytes=len(wire),sent=False,
                                    **encode_metrics.get(identity.job_id,{})))
                            else:
                                wire = future.result()
                            ready[recipe.cache_key] = wire
                            cache_peak_bytes=max(cache_peak_bytes,cache.resident_bytes)
                            temporary_peak_bytes=max(temporary_peak_bytes,
                                sum(len(w) for k,w in ready.items() if cache.peek(k) is None))
                            if cache.peek(recipe.cache_key) is not None:
                                held.setdefault(i,ExitStack()).enter_context(cache.acquire(recipe.cache_key))
                            event('PACKET_READY',identity,wire_bytes=len(wire))
                            if state.valid(identity):
                                pending_packet = (PreparedPacket(recipe,wire,identity),cand)
                        except BaseException as exc:
                            if owner and recipe.cache_key in cache.pending:
                                cache.complete(recipe.cache_key,error=exc)
                            state.failed.add(recipe.cache_key)
                            generations.append(dict(block_index=i,q=recipe.quality,encoding=recipe.encoding,
                                failed=True,error=str(exc),failed_at_ns=time.perf_counter_ns(),
                                **getattr(self.codec,'last_metrics',{})))
                            event('GENERATION_FAILED',identity,error=str(exc))
                            if any(word in str(exc).lower() for word in ('device_error','device-side','illegal memory','device assert')):
                                raise RuntimeError('device_error') from exc
                        state.changed()
                    if net is not None and net.done():
                        wire,row = net.result()
                        packet,cand = net_packet
                        network_rows.append(row)
                        feedback.observe(len(wire),row['net_start_ns'],row['data_available_ns'])
                        state.changed()
                        event('DATA_AVAILABLE',packet.identity)
                        record = state.frozen[packet.identity.block_index]
                        restore_jobs[packet.identity.block_index] = gpu.submit('restore',
                            lambda p=packet,w=wire,r=record: restore(p,w,r),valid=lambda:state.active,metadata=cand)
                        net = net_packet = None
                    for i,future in list(restore_jobs.items()):
                        if future.done():
                            future.result()
                            del restore_jobs[i]
                            if i in held:
                                held.pop(i).close()
                            for key in list(ready):
                                if key[4] == i * chunk:
                                    del ready[key]
                            completed += 1
                    for i,record in tuple(state.frozen.items()):
                        if i in pruned_blocks:
                            continue
                        # Freeze ends replacement eligibility. Drop unselected
                        # temporaries and retain only the selected wire lease.
                        if i in held:
                            held.pop(i).close()
                            if cache.peek(record.cache_key) is not None:
                                held[i]=ExitStack()
                                held[i].enter_context(cache.acquire(record.cache_key))
                        for key in list(ready):
                            if key[4]==record.start and key!=record.cache_key:
                                del ready[key]
                        pruned_blocks.add(i)
                    if completed >= count:
                        break
                    # Window one: next block may materialize only after current
                    # NET_START and only with valid completed history.
                    can_select = (next_send < count and next_send not in jobs
                        and pending_packet is None
                        and (next_send == completed or
                             (self._mode=='pipeline' and feedback.valid
                              and self.config.get('generation_lookahead',1)>0
                              and next_send <= completed+1 and next_send <= len(state.frozen))))
                    if can_select:
                        cand,identity = decide(next_send)
                        packet = materialize(cand,identity)
                        if packet:
                            pending_packet=(packet,cand)
                    if pending_packet and net is None and receive_credit.is_set():
                        packet,cand = pending_packet
                        # Fresh exact-length comparison; previous alternatives
                        # remain pinned until this block has frozen its send.
                        fresh,identity = decide(next_send)
                        if fresh.cache_key != cand.cache_key:
                            pending_packet=None
                            other=materialize(fresh,identity)
                            if other:
                                pending_packet=(other,fresh)
                            continue
                        packet=PreparedPacket(packet.recipe,packet.wire,identity)
                        receive_credit.clear()
                        def freeze(now, p=packet):
                            record = state.freeze(p,now)
                            event('NET_START',p.identity)
                            return record.net_start_ns
                        net_packet=(packet,fresh)
                        net=link.send(packet.wire,identity=asdict(identity),on_start=freeze)
                        sent_candidates[packet.identity.block_index]=fresh
                        if packet.recipe.encoding=='whole_text':
                            count=1
                            actual_route='whole_text'
                        pending_packet=None
                        next_send += 1
                        for item in generations:
                            if item['block_index']==packet.identity.block_index and item['q']==packet.recipe.quality and item['encoding']==packet.recipe.encoding:
                                item['sent']=True
                    state.wake.wait(.001)
                    state.wake.clear()
                def first_token():
                    if not state.active:
                        raise RuntimeError('request_cancelled')
                    return self.engine.first(list(context.context_ids),suffix,final)
                try:
                    token=gpu.submit('first',first_token,valid=lambda:state.active).result()
                except CancelledError as error:
                    raise RuntimeError('request_cancelled') from error
                with state.lock:
                    if not state.active:
                        raise state.error or RuntimeError('request_cancelled')
                    first_ns=time.perf_counter_ns()
                    state.active=False
                if cuda_device:
                    import torch
                    cuda_peak_bytes=torch.cuda.max_memory_allocated()
                else:
                    cuda_peak_bytes=None
        except BaseException as error:
            state.cancel()
            self.last_failure=dict(request_id=clock.request_id,t0_ns=clock.t0_ns,
                failed_at_ns=time.perf_counter_ns(),decisions=records,events=events,
                generation=generations,network=network_rows,error=str(error))
            error.joint_failure=self.last_failure
            raise
        finally:
            for stack in held.values():
                stack.close()
            for future,cand,identity,recipe,owner in jobs.values():
                if future.done() and not future.cancelled() and future.exception() is None and owner:
                    fields,_=future.result()
                    if getattr(fields,'owner',None) is not None:
                        fields.owner.release_fields()
                if owner and recipe.cache_key in cache.pending:
                    cache.complete(recipe.cache_key,error=RuntimeError('request_cancelled'))
            cache.on_change=lambda:None
            self.sources.on_change=lambda:None
            self.current_transport=None
            self.current_state=None
        return dict(final_kv=final,token=token,first_token_ns=first_ns,t0_ns=clock.t0_ns,
            deadline_ns=clock.deadline_ns,request_id=clock.request_id,ttft_s=(first_ns-clock.t0_ns)/1e9,
            slo_met=first_ns<=clock.deadline_ns,decisions=records,events=events,network=network_rows,
            generation=generations,cache=cache.snapshot(),route=actual_route,
            completed_history=feedback.samples,link_id=link_id,
            codec_calls={k:v-calls_before.get(k,0) for k,v in getattr(self.codec,'calls',{}).items()},
            source_bytes=context.metadata.get('source_bytes'),cache_peak_bytes=cache_peak_bytes,
            temporary_peak_bytes=temporary_peak_bytes,cuda_peak_bytes=cuda_peak_bytes)

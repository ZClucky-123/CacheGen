"""Joint quality/representation selection using complete resource forecasts."""
from dataclasses import dataclass, field, replace
from itertools import product

from experiments.joint_profiles import RuntimeProfile, seconds_ns, BandwidthHistory
from experiments.joint_scheduler import CandidateEstimate, Plan, Snapshot, simulate_plan


ENCODINGS = ('ac_count', 'ac_cdf', 'fixed_bits', 'text', 'whole_text')


@dataclass(frozen=True)
class Forecast:
    snapshot_time_ns: int
    state_version: int
    F50_ns: int
    F_base_ns: int
    residual_group: tuple
    delta_s: float
    F_safe_ns: int
    cache_class: str
    simulation50: object = None
    simulation_base: object = None


@dataclass(frozen=True)
class Decision:
    quality: int
    encoding: str
    kind: str
    route: str
    candidate: CandidateEstimate
    plan: Plan
    forecast: Forecast
    predicted_slo_miss: bool
    based_on_state_version: int
    reason: str
    stats: dict = field(default_factory=dict)

    @property
    def q(self):
        return self.quality

    @property
    def m(self):
        return None if self.kind == 'text' else self.encoding


def _attr(request, name, default=None):
    return request.get(name, default) if isinstance(request, dict) else getattr(request, name, default)


def _deadline(request):
    clock = _attr(request, 'clock', request)
    return clock['deadline_ns'] if isinstance(clock, dict) else clock.deadline_ns


def _collect(request):
    return _attr(request, 'validation_only', False) and _attr(request, 'residual_mode', '') == 'collect'


def enumerate_plans(candidates, quality, *, whole_text=None, text_admitted=True, deduplicate=True):
    """Current route a + one uniform suffix route b, with no materialization."""
    candidates = tuple(candidates)
    by_block = {}
    for candidate in candidates:
        if candidate.encoding == 'whole_text':
            continue
        if candidate.q != quality and candidate.kind != 'text':
            continue
        if candidate.kind == 'text' and not text_admitted:
            continue
        by_block.setdefault(candidate.block_index, {})[candidate.encoding] = candidate
    plans = []
    if by_block:
        indices = sorted(by_block)
        head = indices[0]
        suffix_routes = set.intersection(*(set(by_block[i]) for i in indices[1:])) if len(indices) > 1 else set(by_block[head])
        for head_route, suffix_route in product(ENCODINGS, ENCODINGS):
            if head_route not in by_block[head] or suffix_route not in suffix_routes:
                continue
            blocks = (by_block[head][head_route],) + tuple(by_block[i][suffix_route] for i in indices[1:])
            plans.append(Plan(quality, blocks))
    if quality == 3 and whole_text is not None and text_admitted:
        plans.append(Plan(3, (whole_text,), 'whole_text'))
    if not deduplicate:
        return tuple(plans)
    return tuple({plan.signature: plan for plan in plans}.values())


def _forecast(plan, snapshot, profile, request):
    f50 = simulate_plan(plan, snapshot, profile, .50)
    base = simulate_plan(plan, snapshot, profile, .90)
    total = _attr(request, 'total_tokens', snapshot.total_tokens or max(c.end for c in plan.candidates))
    group = (('serial' if snapshot.mode in ('serial', 'chunk_serial', 'whole') else 'pipeline'),
             plan.route, total, base.cache_class)
    if _collect(request) or _attr(request, 'diagnostic_only', False):
        resolved, delta = group, 0.
    else:
        resolved, delta = profile.residual_table.lookup(group)
    return Forecast(snapshot.now_ns, snapshot.state_version, f50.end_ns, base.end_ns,
                    resolved, delta, base.end_ns + seconds_ns(delta), base.cache_class, f50, base)


def _tie(plan, forecast, snapshot):
    first = plan.candidates[0]
    ready = first.packet_state == 'ready'
    keep = first.encoding == snapshot.current_encoding
    return (forecast.F50_ns, not ready, not keep,
            tuple(ENCODINGS.index(c.encoding) for c in plan.candidates), -plan.quality)


def _select(candidates, snapshot, profile, request, optimized):
    candidates = tuple(candidates)
    mode = _attr(request, 'quality_mode', 'adaptive')
    if mode not in ('adaptive', 'fixed'):
        raise ValueError('quality_mode must be adaptive or fixed')
    validation = _attr(request, 'validation_only', False)
    diagnostic = _attr(request, 'diagnostic_only', False)
    if not validation and not diagnostic and mode == 'adaptive':
        profile.require_formal()
    fixed = _attr(request, 'fixed_quality')
    if mode == 'fixed' and fixed not in (1, 2, 3):
        raise ValueError('fixed_quality must be 1, 2 or 3')
    minimum = _attr(request, 'min_quality', 1)
    total = _attr(request, 'total_tokens', snapshot.total_tokens)
    if mode == 'adaptive' and not validation and not diagnostic and str(total) not in profile.quality_by_T:
        raise ValueError('quality_admission_missing')
    admitted = tuple(profile.quality_by_T.get(str(total), profile.admitted_qualities))
    qualities = [fixed] if mode == 'fixed' else [q for q in (3, 2, 1) if q >= minimum and q in admitted and q in profile.admitted_qualities]
    usable = tuple(c for c in candidates if c.eligible and c.packet_state != 'failed' and
                   (c.source_available or c.packet_state in ('ready', 'generating') or c.kind == 'text'))
    if mode == 'fixed':
        usable = tuple(c for c in usable if c.kind == 'kv')
    whole = _attr(request, 'whole_text')
    if mode == 'fixed' or snapshot.committed_blocks or snapshot.next_send_block not in (None, 0):
        whole = None
    if whole is not None and not whole.eligible:
        whole = None
    stats = {'qualities_evaluated': [], 'plans_evaluated': 0, 'simulation_calls': 0,
             'prediction_cache_hits': 0, 'metadata_candidates': len(usable)}
    evaluated = []
    memo = {}
    deadline = _deadline(request)
    feasible_quality = None
    residual_missing = False
    source_missing = False
    for q in qualities:
        plans = enumerate_plans(usable, q, whole_text=whole,
                                text_admitted=profile.text_admitted,
                                deduplicate=optimized)
        if not plans:
            continue
        stats['qualities_evaluated'].append(q)
        quality_results = []
        for plan in plans:
            # Cache lifetime is this call only: snapshot time/version/bandwidth
            # and the already-loaded profile are invariant within the call.
            key = plan.signature
            try:
                if optimized and key in memo:
                    forecast = memo[key]
                    stats['prediction_cache_hits'] += 1
                else:
                    forecast = _forecast(plan, snapshot, profile, request)
                    stats['simulation_calls'] += 2
                    memo[key] = forecast
            except ValueError as exc:
                error = str(exc)
                if error not in ('residual_profile_missing', 'cost_model_missing', 'source_unavailable'):
                    raise
                residual_missing |= error == 'residual_profile_missing'
                source_missing |= error == 'source_unavailable'
                continue
            stats['plans_evaluated'] += 1
            quality_results.append((plan, forecast))
        evaluated.extend(quality_results)
        feasible = [(p, f) for p, f in quality_results if f.F_safe_ns <= deadline]
        if feasible:
            feasible_quality = q if feasible_quality is None else max(feasible_quality, q)
            if optimized:
                break
    if not evaluated:
        raise ValueError('residual_profile_missing' if residual_missing else (
                         'source_unavailable' if source_missing else 'cost_model_missing'))
    if feasible_quality is not None:
        allowed = [(p, f) for p, f in evaluated if p.quality == feasible_quality and f.F_safe_ns <= deadline]
    else:
        allowed = evaluated
    plan, forecast = min(allowed, key=lambda x: _tie(x[0], x[1], snapshot))
    candidate = plan.candidates[0]
    return Decision(plan.quality, candidate.encoding, candidate.kind, plan.route, candidate,
                    plan, forecast, forecast.F_safe_ns > deadline, snapshot.state_version,
                    'predicted_slo_miss' if feasible_quality is None else 'highest_feasible_quality_fastest_plan', stats)


def choose_joint(estimates, state, profile, request):
    return _select(estimates, state, profile, request, True)


def choose_joint_exhaustive(estimates, state, profile, request):
    """Unoptimized reference: duplicate plans, all qualities and no memoization."""
    return _select(estimates, state, profile, request, False)


def refresh_decision(decision, snapshot, profile, request):
    """Reprice only work after the latest commit-time snapshot.

    The runtime must re-run selection if state_version changed, or if repricing
    changes feasibility. No virtual current SELECT is added by the simulator.
    """
    if snapshot.state_version != decision.based_on_state_version:
        raise ValueError('stale_state_version')
    forecast = _forecast(decision.plan, snapshot, profile, request)
    return replace(decision, forecast=forecast,
                   predicted_slo_miss=forecast.F_safe_ns > _deadline(request))

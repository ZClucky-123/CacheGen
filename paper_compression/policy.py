"""Causal policy: explicit measured probe, completed history, and time margin.

No current/future trace argument is accepted here. This is a disclosed conservative
implementation of the paper's recompute-first idea, not identical pseudocode.
"""
import math


def fixed_strategy_method(strategy, selected, highest):
    if strategy == 'history':
        return None
    if strategy == 'text':
        return 'text'
    if strategy == 'fixed':
        return selected
    if strategy == 'fixed_highest_quality':
        return highest
    raise ValueError('Unknown adaptive strategy')


def bandwidth(probe, completed):
    source = completed[-1] if completed else probe
    if source['bytes'] <= 0 or not math.isfinite(source['seconds']) or source['seconds'] <= 0:
        raise ValueError('A positive, completed bandwidth measurement is required')
    return {'estimate_gbps': source['bytes'] * 8 / source['seconds'] / 1e9,
            'estimate_source': 'previous_completed_chunk' if completed else 'measured_probe',
            'source_chunk_index': len(completed) - 1 if completed else None}


def choose(chunks, index, probe, completed, slo, elapsed, first_seconds, safety_fraction,
           min_reserve, quality_order):
    if len(completed) != index or not 0 <= index < len(chunks):
        raise ValueError('Require exactly the completed prefix')
    if not (slo > 0 and elapsed >= 0 and first_seconds >= 0 and 0 <= safety_fraction < 1 and min_reserve >= 0):
        raise ValueError('Invalid SLO or reserve')
    estimate = bandwidth(probe, completed)
    gbps = estimate['estimate_gbps']
    reserve = first_seconds + max(min_reserve, (slo - elapsed) * safety_fraction)
    budget = slo - elapsed - reserve
    def cost(chunk, method):
        row = chunk[method]
        return row['bytes'] * 8 / (gbps * 1e9) + row['restore_seconds']
    methods = ['text'] + list(quality_order)
    forecasts = {m: sum(cost(c, m) for c in chunks[index:]) for m in methods}
    if forecasts['text'] <= budget:
        method = 'text'
    else:
        method = next((m for m in quality_order if forecasts[m] <= budget), None)
        if method is None:
            method = min(methods, key=lambda m: forecasts[m])
    return {**estimate, 'method': method, 'chunk_index': index, 'elapsed_seconds': elapsed,
            'remaining_budget_seconds': budget, 'reserved_seconds': reserve,
            'candidate_forecasts_seconds': forecasts, 'forecast_feasible': forecasts[method] <= budget}

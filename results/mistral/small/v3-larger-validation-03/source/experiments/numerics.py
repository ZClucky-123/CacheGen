"""Audit FP16 greedy sensitivity without changing model outputs or QA scores.

Different query shapes can perturb FP16 logits by one representable step. A
different greedy token is accepted ONLY as an explicitly recorded numerical tie
when both choices lie within two FP16 ULPs, and the complete teacher-forced
logit vectors satisfy the existing numerical check. Same-payload transport modes
still require exact tensors and generated IDs; this helper does not replace it.
"""
import math
import torch
from mini_repro.run import legacy_to_blob

ATOL = .1
RTOL = .02
MAX_TIE_ULPS = 2


def compare_kv(reference, candidate):
    """Keep the strict pointwise check and add per-head RMS diagnostics.

    KV vectors contain components near zero, where pointwise relative error
    loses meaning. RMS keeps absolute tolerance in per-component units across
    different head dimensions while bounding each complete attention-head vector,
    not hiding bad positions behind a global average. No equality is claimed for FP16
    full-vs-chunk prefill; raw reassembly/mode identity remain exact checks.
    """
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise RuntimeError('KV geometry or dtype differs')
    if not bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all()):
        raise RuntimeError('KV contains nonfinite values')
    difference = candidate.float() - reference.float()
    normalizer = math.sqrt(reference.shape[-1])
    error_norm = torch.linalg.vector_norm(difference, dim=-1) / normalizer
    reference_norm = torch.linalg.vector_norm(reference.float(), dim=-1) / normalizer
    bound = ATOL + RTOL * reference_norm
    pointwise_bad = difference.abs() > ATOL + RTOL * reference.float().abs()
    bad_vectors = error_norm > bound
    elementwise_ok = not bool(pointwise_bad.any())
    return {'policy': 'pointwise_with_head_rms_diagnostics_v1', 'atol': ATOL, 'rtol': RTOL,
            'absolute_tolerance_units': 'per_component_rms',
            'max_abs_error': float(difference.abs().max()), 'mean_abs_error': float(difference.abs().mean()),
            'elementwise_outlier_count': int(pointwise_bad.sum()),
            'elementwise_allclose': elementwise_ok,
            'head_vector_count': error_norm.numel(), 'failed_head_vectors': int(bad_vectors.sum()),
            'max_head_vector_error_over_bound': float((error_norm / bound).max()),
            'rms_within_bound': not bool(bad_vectors.any()), 'passed': elementwise_ok}


def compare_logits(reference, candidate):
    reference, candidate = reference.float(), candidate.float()
    a, b = int(reference.argmax()), int(candidate.argmax())
    gap_a = float(reference[a] - reference[b])
    gap_b = float(candidate[b] - candidate[a])
    # Measure representable FP16 spacing at both competing logits, rather than
    # inventing a fixed score margin or altering the greedy decoding rule.
    values = torch.stack((reference[a], reference[b], candidate[a], candidate[b])).half().tolist()
    # PyTorch 2.3 has no CUDA nextafter(Half). IEEE binary16 has ten fraction
    # bits; at powers of two use the larger adjacent spacing. Subnormal floor
    # is 2^-24. This is representational spacing, not a tuned score tolerance.
    ulp = max(2. ** -24 if abs(v) < 2. ** -14 else 2. ** (math.frexp(abs(v))[1] - 11) for v in values)
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    elementwise = finite and torch.allclose(reference, candidate, atol=ATOL, rtol=RTOL)
    rms_error = float(torch.linalg.vector_norm(reference - candidate) / math.sqrt(reference.numel()))
    reference_rms = float(torch.linalg.vector_norm(reference) / math.sqrt(reference.numel()))
    competitive = finite and torch.allclose(reference[[a, b]], candidate[[a, b]], atol=ATOL, rtol=RTOL)
    close = elementwise
    same = a == b
    near_tie = not same and gap_a <= MAX_TIE_ULPS * ulp and gap_b <= MAX_TIE_ULPS * ulp
    return {'reference_token_id': a, 'candidate_token_id': b, 'argmax_identical': same,
            'numerically_close': bool(close), 'fp16_near_tie': near_tie,
            'logit_policy': 'pointwise_v1',
            'elementwise_allclose': bool(elementwise), 'competitive_logits_close': bool(competitive),
            'rms_logit_error': rms_error, 'reference_rms_logit': reference_rms,
            'reference_choice_gap': gap_a, 'candidate_choice_gap': gap_b,
            'fp16_ulp': ulp, 'max_tie_ulps': MAX_TIE_ULPS,
            'max_abs_logit_error': float((reference - candidate).abs().max()),
            'passed': bool(close and (same or near_tie))}


def cached_reference(engine, suffix, blob, limit):
    """Actual optimized greedy scores with fresh cache ownership each step."""
    current, ids, generated, scores = blob, suffix, [], []
    for _ in range(limit):
        logits, cache = engine.forward(ids, current)
        score = logits[0, -1]
        scores.append(score)
        token = int(score.argmax())
        if token == engine.tokenizer.eos_token_id:
            break
        generated.append(token)
        current, ids = legacy_to_blob(cache), [token]
    return generated, scores


def audit_generation(engine, suffix, blob, limit, reference_ids, reference_scores, actual_ids=None):
    """Compare on the reference prefix, so divergence cannot hide later errors."""
    if actual_ids is None:
        actual_ids = engine.generate(suffix, blob, limit)['generated_ids']
    current, ids, steps = blob, suffix, []
    for index, reference in enumerate(reference_scores):
        logits, cache = engine.forward(ids, current)
        row = {'step': index, **compare_logits(reference, logits[0, -1])}
        steps.append(row)
        if not row['passed']:
            raise RuntimeError(f'Cache generation exceeds numerical bounds: {row}')
        token = row['reference_token_id']
        if token == engine.tokenizer.eos_token_id:
            break
        current, ids = legacy_to_blob(cache), [token]
    identical = reference_ids == actual_ids
    divergence = None
    if not identical:
        def token_at(values, index):
            if index < len(values):
                return values[index]
            return engine.tokenizer.eos_token_id if len(values) < limit else None
        divergence = next(i for i in range(limit) if token_at(reference_ids, i) != token_at(actual_ids, i))
        first = steps[divergence]
        if not first['fp16_near_tie'] or first['candidate_token_id'] != token_at(actual_ids, divergence):
            raise RuntimeError('First generation divergence is not an audited FP16 near-tie')
    return {'generated_ids_identical': identical, 'reference_generated_ids': reference_ids,
            'candidate_generated_ids': actual_ids, 'teacher_forced_logit_steps': steps,
            'first_divergence_step': divergence,
            'classification': 'identical' if identical else 'fp16_near_tie_divergence',
            'status': 'numerically_valid', 'actual_outputs_unmodified': True}

"""Bandwidth estimators receive completed history, never a current/future trace."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CompletedTransfer:
    chunk_index: int
    message_bytes: int
    effective_transfer_seconds: float


def history_bandwidth(completed, initial_gbps):
    if not math.isfinite(initial_gbps) or initial_gbps <= 0:
        raise ValueError('Fixed initial bandwidth must be positive')
    if not completed:
        return {'estimate_gbps': initial_gbps, 'estimate_source': 'fixed_initial', 'source_chunk_index': None}
    previous = completed[-1]
    if not isinstance(previous, CompletedTransfer) or previous.message_bytes <= 0 or previous.effective_transfer_seconds <= 0:
        raise ValueError('Expected completed effective network measurements')
    return {'estimate_gbps': previous.message_bytes * 8 / previous.effective_transfer_seconds / 1e9,
            'estimate_source': 'previous_completed_message', 'source_chunk_index': previous.chunk_index}


def history_decision(chunks, index, completed, initial_gbps, slo, elapsed, reserve):
    """No argument can supply the current/future trace to this policy."""
    from mini_repro.adaptive import next_decision
    if len(completed) != index or any(c.chunk_index != i for i, c in enumerate(completed)):
        raise ValueError('Policy requires exactly the previously completed chunks')
    estimate = history_bandwidth(completed, initial_gbps)
    return {**next_decision(chunks, index, estimate['estimate_gbps'], slo, elapsed, 'adaptive', reserve), **estimate}

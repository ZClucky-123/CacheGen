"""GPU cross-entropy size estimates; never run an arithmetic encoder here."""
import math
import torch
from .entropy import SAFE_TOKENS
from .quantization import delta_symbols


def symbol_bits(symbols, cdf):
    """Use the actual smoothed, 16bit integer CDF, not fitted test entropy."""
    cdf = cdf.to(symbols.device).to(torch.int32).bitwise_and(65535)
    cdf[..., -1] = 65536
    frequencies = cdf[..., 1:] - cdf[..., :-1]
    costs = 16. - torch.log2(frequencies.float())
    total = torch.zeros((), dtype=torch.float64, device=symbols.device)
    # Bound gather intermediates for 4K Mistral caches.
    for start in range(0, symbols.shape[2], 32):
        block = symbols[:, :, start:start + 32].transpose(1, 2).long()
        total += costs[:, start:start + 32].gather(-1, block).double().sum()
    return total


def estimate(q, profile, metadata_bytes):
    """Declared wire model, including segment lengths and termination allowance.

    metadata_bytes is a fixed pre-registered budget, not fitted to test payloads.
    Actual torch.save framing is measured separately during final verification.
    """
    delta, escaped = delta_symbols(q.delta, q.config)
    bits = symbol_bits(q.anchor, profile.anchor_cdf) + symbol_bits(delta, profile.delta_cdf)
    channels = q.anchor.shape[2]
    streams = sum(math.ceil(t.shape[1] / SAFE_TOKENS) * t.shape[0] * channels
                  for t in (q.anchor, delta))
    scale_bytes = q.anchor_scales.numel() * q.anchor_scales.element_size()
    if q.delta_scales is not None:
        scale_bytes += q.delta_scales.numel() * q.delta_scales.element_size()
    escape_bytes = escaped.numel() * 4
    # Each channel stream has int32 length plus a 2-bit termination and <=7 pad bits.
    # Use a conservative 2-byte allowance; this is an estimate, not an exact bound.
    lengths_bytes, termination_bytes = streams * 4, streams * 2
    entropy_bytes = float(bits.item()) / 8.
    return {'estimated_bytes': entropy_bytes + scale_bytes + escape_bytes + metadata_bytes
            + lengths_bytes + termination_bytes,
            'cross_entropy_bytes': entropy_bytes, 'scale_bytes': scale_bytes,
            'escape_bytes': escape_bytes, 'escape_count': escaped.numel(),
            'segment_lengths_bytes': lengths_bytes, 'termination_allowance_bytes': termination_bytes,
            'metadata_budget_bytes': metadata_bytes, 'size_kind': 'theoretical_estimate',
            'profile_bytes_in_packet': 0}

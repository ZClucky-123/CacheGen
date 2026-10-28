"""§5.2 anchor grouping and disclosed vectorwise/absolute delta quantization."""
from dataclasses import dataclass
import torch
from .config import CodecConfig


@dataclass
class QuantizedKV:
    anchor: torch.Tensor       # [K layers then V layers, anchors, channels], uint8
    delta: torch.Tensor        # same row/channel layout; non-anchor tokens, int32
    anchor_scales: torch.Tensor
    delta_scales: object
    shape: tuple
    config: CodecConfig


def positions(tokens, group_size, device):
    if type(tokens) is not int or tokens < 1:
        raise ValueError('KV must contain at least one token')
    all_positions = torch.arange(tokens, device=device)
    anchors = all_positions[all_positions.remainder(group_size) == 0]
    others = all_positions[all_positions.remainder(group_size) != 0]
    return anchors, others, torch.div(others, group_size, rounding_mode='floor')


def split(blob, config):
    if blob.ndim != 5 or blob.shape[0] != config.layers or blob.shape[1] != 2:
        raise ValueError('Expected [layers,2,heads,tokens,head_dim]')
    if blob.dtype not in (torch.float16, torch.float32) or not bool(torch.isfinite(blob).all()):
        raise ValueError('Expected finite FP16/FP32 KV')
    layers, _, heads, tokens, dim = blob.shape
    if heads < 1 or dim < 1:
        raise ValueError('KV head geometry must be positive')
    groups = blob.permute(1, 0, 3, 2, 4).reshape(2 * layers, tokens, heads * dim)
    anchor_pos, delta_pos, parents = positions(tokens, config.group_size, blob.device)
    anchor = groups.index_select(1, anchor_pos).float()
    delta = groups.index_select(1, delta_pos).float() - anchor.index_select(1, parents)
    return anchor, delta


def quantize(blob, config):
    anchor, delta = split(blob, config)
    maxima = anchor.abs().amax(-1, keepdim=True)
    anchor_scales = torch.where(maxima > 0, maxima / 127., torch.ones_like(maxima))
    anchor_symbols = (torch.round(anchor / anchor_scales).clamp(-127, 127) + 127).to(torch.uint8)
    if not config.use_delta or config.anchor_reference == 'reconstructed':
        _, dpos, parents = positions(blob.shape[3], config.group_size, blob.device)
        groups = blob.permute(1, 0, 3, 2, 4).reshape(2 * config.layers, blob.shape[3], -1)
        values = groups.index_select(1, dpos).float()
        reference = (anchor_symbols.float() - 127.) * anchor_scales
        delta = values if not config.use_delta else values - reference.index_select(1, parents)
    widths = torch.tensor(config.widths() * 2, device=blob.device, dtype=torch.float32)[:, None, None]
    delta_scales = None
    if config.delta_scale_mode == 'vectorwise':
        maxima = delta.abs().amax(-1, keepdim=True)
        delta_scales = torch.where(maxima > 0, maxima / config.vector_range, torch.ones_like(maxima))
        integers = torch.round(delta / delta_scales / widths)
    else:
        integers = torch.round(delta / widths)
    if not bool(torch.isfinite(integers).all()) or (integers.numel() and float(integers.abs().max()) >= 2147483648):
        raise ValueError('Delta symbols exceed the exact int32 side channel')
    return QuantizedKV(anchor_symbols, integers.to(torch.int32), anchor_scales,
                       delta_scales, tuple(blob.shape), config)


def delta_symbols(delta, config):
    radius = config.delta_radius
    overflow = (delta < -radius) | (delta > radius)
    escape = config.delta_alphabet - 1
    symbols = torch.where(overflow, torch.full_like(delta, escape), delta + radius).to(torch.uint8)
    return symbols, delta[overflow].to(torch.int32)


def restore_delta(symbols, escaped, config):
    if symbols.dtype != torch.uint8 or escaped.dtype != torch.int32 or escaped.ndim != 1:
        raise ValueError('Invalid delta symbol or escape dtype/shape')
    escape = config.delta_alphabet - 1
    if symbols.numel() and int(symbols.max()) > escape:
        raise ValueError('Invalid delta symbol')
    mask = symbols == escape
    if int(mask.sum()) != escaped.numel():
        raise ValueError('Escape side channel does not match symbol positions')
    delta = symbols.to(torch.int32) - config.delta_radius
    delta[mask] = escaped.to(device=delta.device, dtype=torch.int32)
    return delta


def reconstruct(q):
    config = q.config
    if len(q.shape) != 5 or any(type(n) is not int or n < 1 for n in q.shape):
        raise ValueError('Invalid KV shape')
    layers, two, heads, tokens, dim = q.shape
    if layers != config.layers or two != 2:
        raise ValueError('KV geometry and quantizer differ')
    apos, dpos, parents = positions(tokens, config.group_size, q.anchor.device)
    shape_a = (2 * layers, len(apos), heads * dim)
    shape_d = (2 * layers, len(dpos), heads * dim)
    if tuple(q.anchor.shape) != shape_a or tuple(q.delta.shape) != shape_d:
        raise ValueError('Anchor/delta token counts differ from declared geometry')
    if q.anchor.dtype != torch.uint8 or q.delta.dtype != torch.int32 or bool((q.anchor == 255).any()):
        raise ValueError('Invalid anchor/delta symbol representation')
    if q.anchor_scales.dtype != torch.float32:
        raise ValueError('Anchor scales must be FP32')
    if tuple(q.anchor_scales.shape) != (2 * layers, len(apos), 1):
        raise ValueError('Invalid anchor scale geometry')
    if not bool(torch.isfinite(q.anchor_scales).all() and (q.anchor_scales > 0).all()):
        raise ValueError('Invalid anchor scales')
    anchor = (q.anchor.float() - 127.) * q.anchor_scales
    widths = torch.tensor(config.widths() * 2, device=anchor.device, dtype=torch.float32)[:, None, None]
    delta = q.delta.float() * widths
    if config.delta_scale_mode == 'vectorwise':
        if q.delta_scales is None or tuple(q.delta_scales.shape) != (2 * layers, len(dpos), 1):
            raise ValueError('Invalid delta scale geometry')
        if q.delta_scales.dtype != torch.float32:
            raise ValueError('Delta scales must be FP32')
        if not bool(torch.isfinite(q.delta_scales).all() and (q.delta_scales > 0).all()):
            raise ValueError('Invalid delta scales')
        delta = delta * q.delta_scales
    elif q.delta_scales is not None:
        raise ValueError('Absolute-bin packets must not carry normalized scales')
    groups = torch.empty((2 * layers, tokens, heads * dim), device=anchor.device, dtype=torch.float32)
    groups[:, apos, :] = anchor
    groups[:, dpos, :] = anchor.index_select(1, parents) + delta if config.use_delta else delta
    result = groups.reshape(2, layers, tokens, heads, dim).permute(1, 0, 3, 2, 4).half()
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Reconstructed KV exceeds finite FP16 range')
    return result

"""Shape-consistent FP16 transformer projections and explicit causal masks.

Weights/activations remain FP16 (including original FastChat 8-bit weights).
The last-position LM head remains untouched. Fixed-row projection tiles avoid
different GEMM reductions for whole versus chunked prefill. No CUDA kernels or
quantization schedules are changed.
"""
import torch
import torch.nn.functional as F


def configure_execution(execution):
    backend = execution['sdpa_backend']
    torch.backends.cuda.enable_flash_sdp(backend in ('auto', 'flash'))
    torch.backends.cuda.enable_mem_efficient_sdp(backend in ('auto', 'efficient'))
    torch.backends.cuda.enable_math_sdp(backend in ('auto', 'math'))
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = execution['fp16_reduced_precision_reduction']


class TokenTiledLinear(torch.nn.Module):
    def __init__(self, base, tile_tokens):
        super().__init__()
        self.base, self.tile_tokens = base, tile_tokens

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, inputs):
        from fastchat.model.compression import CLinear, decompress, default_compression_config
        weight = decompress(self.base.weight, default_compression_config) if isinstance(self.base, CLinear) else self.base.weight
        bias = None if self.base.bias is None else self.base.bias.to(weight.dtype)
        shape = inputs.shape
        flat = inputs.reshape(-1, shape[-1]).to(weight.dtype)
        pieces = []
        for start in range(0, flat.shape[0], self.tile_tokens):
            block = flat[start:start + self.tile_tokens]
            count = block.shape[0]
            if count < self.tile_tokens:
                block = F.pad(block, (0, 0, 0, self.tile_tokens - count))
            pieces.append(F.linear(block, weight, bias)[:count])
        return torch.cat(pieces, dim=0).reshape(*shape[:-1], weight.shape[0])


def explicit_causal_mask(module, args, kwargs):
    mask = kwargs.get('attention_mask')
    if mask is not None and mask.ndim == 4:
        return args, kwargs
    ids = kwargs.get('input_ids', args[0] if args else None)
    embeds = kwargs.get('inputs_embeds')
    if ids is None and embeds is None:
        return args, kwargs
    if ids is not None:
        batch, length = ids.shape
        device = ids.device
    else:
        batch, length = embeds.shape[:2]
        device = embeds.device
    cache = kwargs.get('past_key_values')
    previous = 0 if cache is None else cache.get_seq_length() if hasattr(cache, 'get_seq_length') else cache[0][0].shape[-2]
    total = previous + length
    positions = kwargs.get('cache_position')
    if positions is None:
        positions = torch.arange(previous, total, device=device)
    future = torch.arange(total, device=device)[None, :] > positions[:, None]
    dtype = module.embed_tokens.weight.dtype
    causal = torch.zeros((batch, 1, length, total), dtype=dtype, device=device)
    causal.masked_fill_(future[None, None], torch.finfo(dtype).min)
    if mask is not None:
        causal.masked_fill_((mask[:, :total] == 0)[:, None, None, :], torch.finfo(dtype).min)
    return args, {**kwargs, 'attention_mask': causal}


def install_stable_transformer(model, tile_tokens=256):
    from fastchat.model.compression import CLinear
    if model.config._attn_implementation != 'sdpa':
        raise ValueError('Stable transformer execution requires the SDPA adapter')
    if getattr(model, '_cachegen_stable_tile_tokens', None) is not None:
        if model._cachegen_stable_tile_tokens != tile_tokens:
            raise ValueError('Cannot change the stable tile of an installed model')
        return 0
    count = 0
    for parent in list(model.model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, (torch.nn.Linear, CLinear)):
                setattr(parent, name, TokenTiledLinear(child, tile_tokens))
                count += 1
    model.model.register_forward_pre_hook(explicit_causal_mask, with_kwargs=True)
    model._cachegen_stable_tile_tokens = tile_tokens
    model.eval()
    return count

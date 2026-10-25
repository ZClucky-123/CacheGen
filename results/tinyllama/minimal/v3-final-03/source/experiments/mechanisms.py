"""Controlled entropy/precision ablations using the existing CUDA symbol range.

Each precision setting quantizes ONCE. Packed and arithmetic paths consume the
same symbols/scales, share the container and dequantizer, and include all metadata.
No changes to CUDA kernels or to the original uniform4/uniform8 baselines.
"""
import io
import math
from dataclasses import dataclass
import numpy as np
import torch

PRECISIONS = ('uniform16', 'uniform32', 'layered1', 'layered2')
ENTROPY = ('packed', 'arithmetic')


def pack_bits(values, bits):
    if bits not in (4, 5):
        raise ValueError('Only existing 4/5-bit CUDA symbol ranges are supported')
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if values.size and int(values.max()) >= 1 << bits:
        raise ValueError('Symbol exceeds configured bit width')
    bit_values = ((values[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(-1)
    return np.packbits(bit_values, bitorder='little').tobytes()


def unpack_bits(payload, bits, count):
    if bits not in (4, 5) or count < 0 or len(payload) != (count * bits + 7) // 8:
        raise ValueError('Invalid packed symbol dimensions')
    bits_array = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder='little')[:count * bits]
    return (bits_array.reshape(count, bits).astype(np.uint16) * (1 << np.arange(bits))).sum(1).astype(np.uint8)


def precision_bins(spec, precision):
    if precision not in PRECISIONS:
        raise ValueError('Unknown precision allocation')
    layers = spec['layers']
    if precision.startswith('uniform'):
        return [int(precision[7:])] * (2 * layers)
    level = int(precision[-1])
    keys = {1: (16, 16, 12), 2: (32, 16, 16)}[level]
    values = {1: (16, 12), 2: (32, 16)}[level]
    b1, b2, _ = spec['key_boundaries']
    return [keys[0] if i < b1 else keys[1] if i < b2 else keys[2] for i in range(layers)] + [
        values[0] if i < spec['value_boundary'] else values[1] for i in range(layers)]


@dataclass
class Quantized:
    symbols: torch.Tensor  # [K layers then V layers, tokens, flattened channels]
    scales: torch.Tensor
    bins: list
    shape: tuple
    precision: str


def quantize(blob, spec, precision):
    from lmcache.storage_backend.serde.cachegen_encoder import torch_quant_vectorized
    layers, _, heads, tokens, dim = blob.shape
    bins = precision_bins(spec, precision)
    groups = blob.permute(1, 0, 3, 2, 4).reshape(2 * layers, tokens, heads * dim)
    # The original serializer constructs FLOAT32 bins. Integer bins would
    # promote division with FP16 maxima to FP16, overflowing small/zero groups.
    symbols, scales = torch_quant_vectorized(torch.tensor(bins, dtype=torch.float32, device=blob.device), groups)
    if not bool(torch.isfinite(scales).all()) or any(int(symbols[i].min()) < 0 or int(symbols[i].max()) >= b for i, b in enumerate(bins)):
        raise ValueError('Quantized symbols exceed supported range')
    return Quantized(symbols.contiguous(), scales, bins, tuple(blob.shape), precision)


def dequantize(symbols, scales, bins, shape):
    # Same reconstruction for BOTH entropy paths; float32 math then FP16 KV.
    from lmcache.storage_backend.serde.cachegen_decoder import do_dequantize
    bins_tensor = torch.tensor(bins, device=symbols.device)
    groups = do_dequantize(symbols.float(), bins_tensor, scales)
    layers, _, heads, tokens, dim = shape
    return groups.reshape(2, layers, tokens, heads, dim).permute(1, 0, 3, 2, 4).half()


def serialize(quantized, entropy):
    if entropy not in ENTROPY:
        raise ValueError('Unknown entropy representation')
    q = quantized
    obj = {'format': 'cachegen-mechanism-v3', 'entropy': entropy, 'precision': q.precision,
           'shape': list(q.shape), 'bins': q.bins, 'scales': q.scales.cpu(), 'dtype': 'float16',
           'internal_segment_max_tokens': 256}
    if entropy == 'packed':
        symbols_cpu = q.symbols.cpu().numpy()
        obj['bits'] = [math.ceil(math.log2(b)) for b in q.bins]
        obj['streams'] = [torch.from_numpy(np.frombuffer(pack_bits(row, width), dtype=np.uint8).copy())
                          for row, width in zip(symbols_cpu, obj['bits'])]
        symbol_bytes = sum(p.numel() for p in obj['streams'])
        cdf_bytes = 0
    else:
        import torchac_cuda
        from lmcache.storage_backend.serde.cachegen_encoder import encode_ntokens
        # Existing per-layer/channel CDF grouping and existing <=256-token kernel.
        cdf = torchac_cuda.calculate_cdf(q.symbols, max(q.bins))
        layers2, tokens, channels = q.symbols.shape
        output = torch.zeros((layers2, channels, 256), device=q.symbols.device, dtype=torch.uint8)
        lengths = torch.zeros((layers2, channels), device=q.symbols.device, dtype=torch.int32)
        obj['cdf'] = cdf.cpu()
        obj['segments'] = []
        for start in range(0, tokens, 256):
            end = min(start + 256, tokens)
            bytestream = encode_ntokens(cdf, q.symbols[:, start:end, :], output, lengths)
            obj['segments'].append({'data': bytestream.cpu(), 'lengths': lengths.cpu().clone(), 'tokens': end - start})
        symbol_bytes = sum(s['data'].numel() for s in obj['segments'])
        cdf_bytes = obj['cdf'].numel() * obj['cdf'].element_size()
    stream = io.BytesIO()
    torch.save(obj, stream)
    payload = stream.getvalue()
    return payload, {'symbol_data_bytes': symbol_bytes, 'cdf_tensor_bytes': cdf_bytes,
                     'serialized_bytes': len(payload), 'metadata_and_container_bytes': len(payload) - symbol_bytes,
                     'precision': q.precision, 'entropy': entropy, 'bins': q.bins,
                     'mean_symbol_bits': sum(math.ceil(math.log2(b)) for b in q.bins) / len(q.bins)}


def deserialize(payload, device='cuda'):
    from lmcache.storage_backend.serde.cachegen_decoder import decode_chunk
    from lmcache.storage_backend.serde.cachegen_basics import CacheGenGPUBytestream
    obj = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if obj['format'] != 'cachegen-mechanism-v3':
        raise ValueError('Unknown mechanism container')
    layers, _, heads, tokens, dim = obj['shape']
    shape = (2 * layers, tokens, heads * dim)
    if obj['entropy'] == 'packed':
        rows = [torch.from_numpy(unpack_bits(p.numpy().tobytes(), b, tokens * heads * dim).copy())
                for p, b in zip(obj['streams'], obj['bits'])]
        symbols = torch.stack(rows).reshape(shape).to(device)
    elif obj['entropy'] == 'arithmetic':
        symbols = torch.zeros(shape, dtype=torch.uint8, device=device)
        cdf = obj['cdf'].to(device)
        start = 0
        for part in obj['segments']:
            end = start + part['tokens']
            segment = CacheGenGPUBytestream(part['data'].to(device), part['lengths'].to(device), part['tokens'])
            decode_chunk(cdf, segment, symbols[:, start:end, :])
            start = end
        if start != tokens:
            raise ValueError('Arithmetic segments lost tail')
    else:
        raise ValueError('Unknown entropy mode')
    blob = dequantize(symbols, obj['scales'].to(device), obj['bins'], obj['shape'])
    return symbols, blob

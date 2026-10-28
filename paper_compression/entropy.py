"""Real arithmetic streams using pre-shared CDFs; no per-packet CDF fitting.

Integer CDF representation follows https://github.com/fab-jul/torchac .
CPU is a reference implementation, never a substitute in CUDA timing reports.
"""
import torch

SAFE_TOKENS = 96
OUTPUT_BYTES = 256
CPU_CDF_BUDGET = 16 << 20


def _cpu_tiles(shape, alphabet):
    layers, tokens, channels = shape
    for layer in range(layers):
        for channel in range(0, channels, 32):
            width = min(32, channels - channel)
            count = max(1, CPU_CDF_BUDGET // (2 * width * (alphabet + 1)))
            for start in range(0, tokens, count):
                yield {'layer': layer, 'channel_start': channel, 'channels': width,
                       'token_start': start, 'tokens': min(count, tokens - start)}


def _expanded_cpu_cdf(cdf, tile):
    layer, channel, width = tile['layer'], tile['channel_start'], tile['channels']
    return cdf[layer, channel:channel + width].unsqueeze(1).expand(-1, tile['tokens'], -1).contiguous()


def encode(symbols, cdf, backend):
    if symbols.dtype != torch.uint8 or symbols.ndim != 3:
        raise ValueError('Arithmetic symbols must be [layers,tokens,channels] uint8')
    rows, tokens, channels = symbols.shape
    if cdf.dtype != torch.int16 or cdf.ndim != 3 or tuple(cdf.shape[:2]) != (rows, channels) or not 2 <= cdf.shape[-1] <= 257:
        raise ValueError('Symbol/CDF geometry mismatch')
    if symbols.numel() and int(symbols.max()) >= cdf.shape[-1] - 1:
        raise ValueError('Arithmetic symbol outside model alphabet')
    if backend == 'cpu':
        import torchac
        source, cdf = symbols.cpu(), cdf.cpu()
        segments = []
        for tile in _cpu_tiles(tuple(symbols.shape), cdf.shape[-1] - 1):
            layer, start, count = tile['layer'], tile['token_start'], tile['tokens']
            channel, width = tile['channel_start'], tile['channels']
            block = source[layer, start:start + count, channel:channel + width].T.contiguous().to(torch.int16)
            data = torchac.encode_int16_normalized_cdf(_expanded_cpu_cdf(cdf, tile), block)
            segments.append({**tile, 'data': torch.tensor(list(data), dtype=torch.uint8)})
        return {'backend': 'torchac-cpu-v1', 'shape': list(symbols.shape), 'segments': segments}
    if backend != 'cuda':
        raise ValueError('Choose an explicit cpu or cuda arithmetic backend')
    from .cuda_backend import extension
    native = extension()
    source, cdf = symbols.to('cuda'), cdf.to('cuda').contiguous()
    buffer = torch.empty((rows, channels, OUTPUT_BYTES), device='cuda', dtype=torch.uint8)
    lengths = torch.empty((rows, channels), device='cuda', dtype=torch.int32)
    segments = []
    for start in range(0, tokens, SAFE_TOKENS):
        end = min(start + SAFE_TOKENS, tokens)
        block = source[:, start:end, :].contiguous().view(torch.int8)
        native.encode_fast_new(cdf, block, buffer, lengths)
        if not bool(((lengths > 0) & (lengths <= OUTPUT_BYTES)).all()):
            raise ValueError('CUDA arithmetic stream exceeded its output buffer')
        mask = torch.arange(OUTPUT_BYTES, device='cuda')[None, None, :] < lengths[..., None]
        segments.append({'token_start': start, 'tokens': end - start,
                         'data': buffer[mask].cpu(), 'lengths': lengths.cpu().clone()})
    return {'backend': 'cuda-ac256-v1', 'shape': list(symbols.shape), 'segments': segments}


def decode(obj, cdf, expected_shape, device):
    if tuple(obj['shape']) != tuple(expected_shape):
        raise ValueError('Declared arithmetic shape differs from KV geometry')
    rows, tokens, channels = expected_shape
    if tuple(cdf.shape[:2]) != (rows, channels):
        raise ValueError('CDF shape differs from arithmetic shape')
    if obj['backend'] == 'torchac-cpu-v1':
        import torchac
        cdf = cdf.cpu()
        result = torch.empty(expected_shape, dtype=torch.uint8)
        expected = list(_cpu_tiles(expected_shape, cdf.shape[-1] - 1))
        if len(expected) != len(obj['segments']):
            raise ValueError('Missing or extra CPU arithmetic tiles')
        for tile, segment in zip(expected, obj['segments']):
            if any(segment[k] != v for k, v in tile.items()):
                raise ValueError('CPU arithmetic tile geometry changed')
            data = segment['data']
            if data.dtype != torch.uint8 or data.ndim != 1 or not data.numel():
                raise ValueError('Invalid CPU arithmetic stream')
            block = torchac.decode_int16_normalized_cdf(_expanded_cpu_cdf(cdf, tile),
                                                       data.cpu().numpy().tobytes())
            if block.numel() and (int(block.min()) < 0 or int(block.max()) >= cdf.shape[-1] - 1):
                raise ValueError('Decoded symbol outside alphabet')
            layer, start, count = tile['layer'], tile['token_start'], tile['tokens']
            channel, width = tile['channel_start'], tile['channels']
            result[layer, start:start + count, channel:channel + width] = block.T.to(torch.uint8)
        return result.to(device)
    if obj['backend'] != 'cuda-ac256-v1':
        raise ValueError('Unknown arithmetic backend')
    from .cuda_backend import extension
    native = extension()
    cdf = cdf.to('cuda').contiguous()
    result = torch.empty(expected_shape, device='cuda', dtype=torch.uint8)
    if len(obj['segments']) != (tokens + SAFE_TOKENS - 1) // SAFE_TOKENS:
        raise ValueError('Missing or extra CUDA segments')
    for start, segment in zip(range(0, tokens, SAFE_TOKENS), obj['segments']):
        count = min(SAFE_TOKENS, tokens - start)
        if segment['token_start'] != start or segment['tokens'] != count:
            raise ValueError('CUDA segment geometry changed')
        lengths, data = segment['lengths'], segment['data']
        if lengths.dtype != torch.int32 or tuple(lengths.shape) != (rows, channels):
            raise ValueError('CUDA stream length geometry invalid')
        if not bool(((lengths > 0) & (lengths <= OUTPUT_BYTES)).all()):
            raise ValueError('Invalid CUDA bytestream lengths')
        if data.dtype != torch.uint8 or data.ndim != 1 or int(lengths.sum()) != data.numel():
            raise ValueError('CUDA stream length sum differs from data')
        prefix = lengths.to(device='cuda', dtype=torch.int64).reshape(-1).cumsum(0).reshape(rows, channels)
        native.decode_fast_prefsum(cdf, data.to('cuda'), prefix, result[:, start:start + count, :])
    if result.numel() and int(result.max()) >= cdf.shape[-1] - 1:
        raise ValueError('Decoded CUDA symbol outside alphabet')
    return result.to(device)

"""Anchor/delta wire container referencing a shared, fingerprinted probability model."""
import io
import torch
from .config import CodecConfig
from .quantization import QuantizedKV, quantize, reconstruct, delta_symbols, restore_delta
from . import entropy

FORMAT = 'cachegen-paper-anchor-delta-v1'


class PaperCodec:
    def __init__(self, profile, model_identity, backend='cuda'):
        profile.validate()
        profile.require_model(model_identity)
        if backend not in ('cpu', 'cuda'):
            raise ValueError('Explicit arithmetic backend required')
        self.profile = profile
        self.config = profile.config
        self.identity = profile.identity
        self.backend = backend
        self.cdfs = {'anchor': profile.anchor_cdf, 'delta': profile.delta_cdf}
        if backend == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested; no silent CPU fallback')
            self.cdfs = {k: v.to('cuda') for k, v in self.cdfs.items()}

    def encode_quantized(self, q):
        if q.config != self.config:
            raise ValueError('Packet quantizer differs from offline probability model')
        layers, _, heads, tokens, dim = q.shape
        geometry = self.profile.metadata['geometry']
        if (layers, heads, dim) != (geometry['layers'], geometry['heads'], geometry['head_dim']):
            raise ValueError('KV geometry differs from model profile')
        if not 1 <= tokens <= 32768:
            raise ValueError('This packet format supports 1..32768 tokens')
        delta, escaped = delta_symbols(q.delta, self.config)
        obj = {'format': FORMAT, 'profile_identity': self.identity,
               'config_identity': self.config.identity, 'shape': list(q.shape),
               'anchor_scales': q.anchor_scales.cpu(),
               'delta_scales': None if q.delta_scales is None else q.delta_scales.cpu(),
               'escape_delta_values': escaped.cpu(),
               'anchor': entropy.encode(q.anchor, self.cdfs['anchor'], self.backend),
               'delta': entropy.encode(delta, self.cdfs['delta'], self.backend)}
        stream = io.BytesIO()
        torch.save(obj, stream)
        data = stream.getvalue()
        symbol_bytes = sum(s['data'].numel() for key in ('anchor', 'delta') for s in obj[key]['segments'])
        stats = {'complete_bytes': len(data), 'arithmetic_stream_bytes': symbol_bytes,
                 'escape_count': escaped.numel(), 'escape_bytes': escaped.numel() * 4,
                 'profile_identity': self.identity, 'profile_bytes_in_packet': 0,
                 'backend': obj['anchor']['backend'], 'parameter_status': self.config.parameter_status}
        return data, stats

    def encode(self, blob):
        return self.encode_quantized(quantize(blob, self.config))

    def decode_quantized(self, payload, device='cuda'):
        obj = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        if obj.get('format') != FORMAT:
            raise ValueError('Unknown paper packet format')
        if obj.get('profile_identity') != self.identity or obj.get('config_identity') != self.config.identity:
            raise ValueError('Packet probability profile or quantizer fingerprint mismatch')
        shape = tuple(obj['shape'])
        if len(shape) != 5 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError('Invalid packet KV shape')
        layers, two, heads, tokens, dim = shape
        geo = self.profile.metadata['geometry']
        if (layers, two, heads, dim) != (geo['layers'], 2, geo['heads'], geo['head_dim']):
            raise ValueError('Packet KV differs from model geometry')
        if type(tokens) is not int or not 1 <= tokens <= 32768:
            raise ValueError('Invalid packet token count')
        if obj['anchor']['backend'] != obj['delta']['backend']:
            raise ValueError('Mixed entropy backends in a packet')
        anchors = (tokens + self.config.group_size - 1) // self.config.group_size
        channels = heads * dim
        a = entropy.decode(obj['anchor'], self.cdfs['anchor'], (2 * layers, anchors, channels), device)
        d = entropy.decode(obj['delta'], self.cdfs['delta'], (2 * layers, tokens - anchors, channels), device)
        escapes = obj['escape_delta_values']
        if escapes.dtype != torch.int32 or escapes.ndim != 1:
            raise ValueError('Invalid escape side channel')
        restored = restore_delta(d, escapes, self.config)
        return QuantizedKV(a, restored, obj['anchor_scales'].to(device),
                           None if obj['delta_scales'] is None else obj['delta_scales'].to(device),
                           shape, self.config)

    def decode(self, payload, device='cuda'):
        return reconstruct(self.decode_quantized(payload, device))

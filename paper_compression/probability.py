"""Separate offline channel/layer probability models for anchors and deltas."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import torch
from .config import CodecConfig
from .quantization import delta_symbols


def make_cdf(counts):
    """Integer arithmetic CDF: strictly positive frequencies, endpoint 2**16."""
    alphabet = counts.shape[-1]
    if counts.dtype != torch.int64 or bool((counts < 0).any()) or not 1 <= alphabet <= 256:
        raise ValueError('Invalid histogram or byte alphabet')
    smoothed = counts + 1
    totals = smoothed.sum(-1, keepdim=True)
    cumulative = torch.cat((torch.zeros_like(totals), smoothed.cumsum(-1)), dim=-1)
    scaled = torch.div(cumulative * (65536 - alphabet), totals, rounding_mode='floor')
    result = scaled + torch.arange(alphabet + 1, dtype=torch.int64)
    return result.to(torch.int16).contiguous()


def validate_cdf(cdf, rows, channels, alphabet):
    if cdf.dtype != torch.int16 or tuple(cdf.shape) != (rows, channels, alphabet + 1):
        raise ValueError('CDF geometry/dtype mismatch')
    if not bool((cdf[..., 0] == 0).all() and (cdf[..., -1] == 0).all()):
        raise ValueError('CDF must start at zero and wrap 65536 to int16 zero')
    expanded = cdf.to(torch.int32).bitwise_and(65535)
    expanded[..., -1] = 65536
    if not bool((expanded[..., 1:] > expanded[..., :-1]).all()):
        raise ValueError('CDF must give every symbol positive mass')


@dataclass
class ProbabilityProfile:
    metadata: dict
    anchor_cdf: torch.Tensor
    delta_cdf: torch.Tensor

    @property
    def config(self):
        return CodecConfig.from_dict(self.metadata['codec'])

    @property
    def identity(self):
        digest = hashlib.sha256(json.dumps(self.metadata, sort_keys=True, separators=(',', ':'),
                                          allow_nan=False).encode())
        for tensor in (self.anchor_cdf, self.delta_cdf):
            digest.update(tensor.cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def validate(self):
        cfg = self.config
        geometry = self.metadata['geometry']
        if geometry['layers'] != cfg.layers or geometry['heads'] < 1 or geometry['head_dim'] < 1:
            raise ValueError('Profile geometry differs from codec')
        channels = geometry['heads'] * geometry['head_dim']
        validate_cdf(self.anchor_cdf, 2 * cfg.layers, channels, 256)
        validate_cdf(self.delta_cdf, 2 * cfg.layers, channels, cfg.delta_alphabet)
        sources = self.metadata['calibration_sources']
        if not sources or len(sources) != len(set(sources)):
            raise ValueError('Declare distinct calibration sources')

    def require_model(self, model_identity):
        if self.metadata['model_identity'] != model_identity:
            raise ValueError('Profile belongs to another model/runtime/weight configuration')

    def require_held_out(self, source_id, dataset_sha256):
        if dataset_sha256 != self.metadata['dataset_sha256']:
            raise ValueError('Evaluation and profile use different dataset namespaces')
        if str(source_id) in self.metadata['calibration_sources']:
            raise ValueError('Calibration/evaluation source leakage')

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create protects an already published shared model.
        with path.open('xb') as stream:
            torch.save({'format': 'cachegen-paper-cdf-v1', 'metadata': self.metadata,
                        'anchor_cdf': self.anchor_cdf.cpu(), 'delta_cdf': self.delta_cdf.cpu(),
                        'identity': self.identity}, stream)

    @classmethod
    def load(cls, path):
        obj = torch.load(path, map_location='cpu', weights_only=True)
        if obj.get('format') != 'cachegen-paper-cdf-v1':
            raise ValueError('Unknown probability profile format')
        profile = cls(obj['metadata'], obj['anchor_cdf'], obj['delta_cdf'])
        profile.validate()
        if profile.identity != obj['identity']:
            raise ValueError('Probability model content fingerprint mismatch')
        return profile


class ProfileBuilder:
    def __init__(self, config, model_identity, dataset_sha256):
        self.config = config
        self.model_identity = model_identity
        self.dataset_sha256 = dataset_sha256
        self.geometry = None
        self.counts = None
        self.sources = {}

    def add(self, quantized, source_id, input_sha256):
        source_id = str(source_id)
        if source_id in self.sources:
            raise ValueError('Calibration source supplied twice')
        if quantized.config != self.config:
            raise ValueError('Calibration quantizer changed')
        layers, _, heads, _, dim = quantized.shape
        geometry = {'layers': layers, 'heads': heads, 'head_dim': dim}
        if self.geometry is not None and self.geometry != geometry:
            raise ValueError('Calibration KV geometry changed')
        if self.counts is None:
            self.geometry = geometry
            self.counts = {
                'anchor': torch.zeros((2 * layers, heads * dim, 256), dtype=torch.int64),
                'delta': torch.zeros((2 * layers, heads * dim, self.config.delta_alphabet), dtype=torch.int64)}
        delta, _ = delta_symbols(quantized.delta, self.config)
        for kind, symbols in (('anchor', quantized.anchor), ('delta', delta)):
            counts = self.counts[kind]
            alphabet = counts.shape[-1]
            # Bounded host histogram work; never materialize all token/channel IDs as int64.
            for layer in range(symbols.shape[0]):
                for start in range(0, symbols.shape[2], 64):
                    block = symbols[layer, :, start:start + 64].transpose(0, 1).cpu().to(torch.int64)
                    group_ids = torch.arange(block.shape[0])[:, None] * alphabet
                    hist = torch.bincount((block + group_ids).reshape(-1),
                                          minlength=block.shape[0] * alphabet)
                    counts[layer, start:start + block.shape[0]] += hist.reshape(-1, alphabet)
        self.sources[source_id] = input_sha256

    def finish(self):
        if self.counts is None:
            raise ValueError('No calibration samples supplied')
        metadata = {'codec': self.config.to_dict(), 'geometry': self.geometry,
                    'model_identity': self.model_identity, 'dataset_sha256': self.dataset_sha256,
                    'calibration_sources': sorted(self.sources), 'calibration_inputs': self.sources,
                    'cdf_precision': 16, 'smoothing_count': 1,
                    'parameter_status': self.config.parameter_status}
        result = ProbabilityProfile(metadata, make_cdf(self.counts['anchor']), make_cdf(self.counts['delta']))
        result.validate()
        return result


class GPUProfileBuilder(ProfileBuilder):
    """One candidate at a time; bounded GPU histograms and CPU persisted CDFs."""
    def add(self, quantized, source_id, input_sha256):
        source_id = str(source_id)
        if source_id in self.sources or quantized.config != self.config:
            raise ValueError('Duplicate calibration source or changed quantizer')
        layers, _, heads, _, dim = quantized.shape
        geometry = {'layers': layers, 'heads': heads, 'head_dim': dim}
        if self.geometry is not None and self.geometry != geometry:
            raise ValueError('Calibration geometry changed')
        if quantized.anchor.device.type != 'cuda':
            raise ValueError('GPU calibration requires CUDA symbols')
        if self.counts is None:
            self.geometry = geometry
            self.counts = {kind: torch.zeros((2 * layers, heads * dim, alphabet), dtype=torch.int64,
                                             device=quantized.anchor.device)
                           for kind, alphabet in (('anchor', 256), ('delta', self.config.delta_alphabet))}
        delta, _ = delta_symbols(quantized.delta, self.config)
        for kind, symbols in (('anchor', quantized.anchor), ('delta', delta)):
            self.counts[kind] += grouped_histogram(symbols, 'layer_channel', self.counts[kind].shape[-1])
        self.sources[source_id] = input_sha256

    def finish(self):
        if self.counts is not None:
            self.counts = {k: v.cpu() for k, v in self.counts.items()}
        return super().finish()


def grouped_histogram(symbols, grouping, alphabet=256):
    """Same symbols, four groupings; never allocate all indices as int64 at once."""
    rows, tokens, channels = symbols.shape
    groups = {'global': 1, 'layer': rows, 'layer_channel': rows * channels,
              'token': tokens}[grouping]
    output = torch.zeros(groups * alphabet, dtype=torch.int64, device=symbols.device)
    r = torch.arange(rows, device=symbols.device)[:, None, None]
    t = torch.arange(tokens, device=symbols.device)[None, :, None]
    for start in range(0, channels, 32):
        block = symbols[:, :, start:start + 32].long()
        c = torch.arange(start, start + block.shape[2], device=symbols.device)[None, None, :]
        ids = {'global': 0, 'layer': r, 'layer_channel': r * channels + c, 'token': t}[grouping]
        output += torch.bincount((block + ids * alphabet).reshape(-1), minlength=groups * alphabet)
    result = output.reshape(groups, alphabet)
    return result.reshape(rows, channels, alphabet) if grouping == 'layer_channel' else result

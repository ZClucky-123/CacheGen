"""Explicit paper constants and disclosed quantizer interpretations. No ML imports."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CodecConfig:
    layers: int
    group_size: int = 10
    anchor_bits: int = 8
    delta_bin_widths: tuple = (.5, 1., 1.5)
    delta_scale_mode: str = 'vectorwise'
    vector_range: float = 127.
    delta_radius: int = 127
    anchor_reference: str = 'original'
    use_delta: bool = True
    parameter_status: str = 'interpretation_pending_verification'

    def __post_init__(self):
        if type(self.layers) is not int or self.layers < 3:
            raise ValueError('Three layer groups require at least three layers')
        if self.group_size != 10 or self.anchor_bits != 8:
            raise ValueError('Paper path requires groups of 10 and 8bit anchors')
        if len(self.delta_bin_widths) != 3 or any(not math.isfinite(v) or v <= 0 for v in self.delta_bin_widths):
            raise ValueError('Provide three positive bin widths')
        if list(self.delta_bin_widths) != sorted(self.delta_bin_widths):
            raise ValueError('Earlier layers must not have coarser bins')
        if self.delta_scale_mode not in ('vectorwise', 'absolute'):
            raise ValueError('Declare vectorwise or absolute bin units')
        if not math.isfinite(self.vector_range) or self.vector_range <= 0:
            raise ValueError('Declare a positive normalized vector range')
        if type(self.delta_radius) is not int or not 1 <= self.delta_radius <= 127:
            raise ValueError('Delta alphabet including escape must fit uint8')
        if self.anchor_reference not in ('original', 'reconstructed'):
            raise ValueError('Declare original or reconstructed anchor reference')
        if type(self.use_delta) is not bool:
            raise ValueError('use_delta must be boolean')
        if self.parameter_status != 'interpretation_pending_verification':
            raise ValueError('Do not relabel unresolved quantization parameters as verified')

    @property
    def boundaries(self):
        return (self.layers // 3, 2 * self.layers // 3, self.layers)

    @property
    def delta_alphabet(self):
        return 2 * self.delta_radius + 2

    def widths(self):
        b1, b2, _ = self.boundaries
        return [self.delta_bin_widths[0] if i < b1 else self.delta_bin_widths[1]
                if i < b2 else self.delta_bin_widths[2] for i in range(self.layers)]

    def to_dict(self):
        value = asdict(self)
        value['delta_bin_widths'] = list(self.delta_bin_widths)
        return value

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value['delta_bin_widths'] = tuple(value.get('delta_bin_widths', (.5, 1., 1.5)))
        return cls(**value)

    @property
    def identity(self):
        return fingerprint(self.to_dict())

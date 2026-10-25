"""Shared application chunks and self-contained, byte-counted wire envelopes."""
import hashlib
import json
import struct
from dataclasses import dataclass
from functools import cached_property
from dataclasses import asdict
import torch
from experiments.engine import timed

HEADER = struct.Struct('!4sIQ')
MAGIC = b'CGV3'
INTERNAL_SEGMENT_TOKENS = 256


def frame(payload, metadata):
    meta = {**metadata, 'payload_bytes': len(payload)}
    encoded = json.dumps(meta, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return HEADER.pack(MAGIC, len(encoded), len(payload)) + encoded + payload


def unframe(wire):
    if len(wire) < HEADER.size:
        raise ValueError('Truncated frame header')
    magic, metadata_bytes, payload_bytes = HEADER.unpack_from(wire)
    if magic != MAGIC or len(wire) != HEADER.size + metadata_bytes + payload_bytes:
        raise ValueError('Invalid complete message size/magic')
    metadata = json.loads(wire[HEADER.size:HEADER.size + metadata_bytes])
    if metadata['payload_bytes'] != payload_bytes:
        raise ValueError('Payload size differs from metadata')
    return metadata, wire[HEADER.size + metadata_bytes:]


@dataclass(frozen=True)
class PreparedChunk:
    index: int
    start: int
    end: int
    method: str
    wire: bytes
    restore: object
    encode_seconds: float
    metadata: dict

    @cached_property
    def sha256(self):
        return hashlib.sha256(self.wire).hexdigest()


def prepare_chunk(engine, ids, blob, method, index, start, end):
    if method == 'text':
        payload, seconds = timed(lambda: engine.text_payload(ids[start:end]))
        restore = engine.restore_text
    else:
        payload, seconds, restore = engine.encode(blob[:, :, :, start:end, :], method)
    meta = {'chunk_index': index, 'token_start': start, 'token_end': end,
            'method': method, 'shape': list(blob[:, :, :, start:end, :].shape),
            'dtype': str(blob.dtype),
            'quantization_schedule': asdict(engine.codec_schedule(int(method[-1]))) if method.startswith('cachegen') else None,
            'internal_segment_max_tokens': INTERNAL_SEGMENT_TOKENS if method.startswith('cachegen') else None,
            'internal_segment_tokens': [min(INTERNAL_SEGMENT_TOKENS, end - start - i)
                for i in range(0, end - start, INTERNAL_SEGMENT_TOKENS)] if method.startswith('cachegen') else []}
    wire, frame_seconds = timed(lambda: frame(payload, meta))
    meta['payload_bytes'] = len(payload)
    return PreparedChunk(index, start, end, method, wire, restore, seconds + frame_seconds, meta)


def prepare(engine, ids, blob, method, chunk_size):
    return tuple(prepare_chunk(engine, ids, blob, method, index, start, min(start + chunk_size, len(ids)))
                 for index, start in enumerate(range(0, len(ids), chunk_size)))


def restore_chunk(engine, prepared, wire, merged):
    meta, payload = unframe(wire)
    if meta != prepared.metadata:
        raise ValueError('Chunk order, token range, method or shape differs')
    previous = 0 if merged is None else merged.shape[3]
    if previous != prepared.start:
        raise ValueError('Restoration has a missing or duplicated prefix')
    if prepared.method == 'text':
        ids = prepared.restore(payload)  # Actual receiver-side tokenization/repair.
        if len(ids) != prepared.end - prepared.start:
            raise ValueError('Text chunk token count differs')
        result = engine.prefill(ids, merged)
    else:
        decoded = prepared.restore(payload)
        if list(decoded.shape) != meta['shape']:
            raise ValueError('Decoded chunk geometry differs')
        result = decoded if merged is None else torch.cat((merged, decoded), dim=3)
    if result.shape[3] != prepared.end:
        raise ValueError('Chunk restoration lost the tail')
    return result


def preparation_metadata(chunks):
    return {'prepared_payload_sha256': [c.sha256 for c in chunks],
            'bytes': sum(len(c.wire) for c in chunks),
            'offline_encode_seconds': sum(c.encode_seconds for c in chunks),
            'application_chunk_count': len(chunks),
            'application_chunks': [{**c.metadata, 'message_bytes': len(c.wire),
                                     'envelope_bytes': len(c.wire) - c.metadata['payload_bytes']} for c in chunks]}

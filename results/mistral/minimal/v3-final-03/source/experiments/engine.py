"""Reusable model/cache/codec execution, independent of the experiment matrix."""
import io
import json
import difflib
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from mini_repro.run import blob_to_legacy, legacy_to_blob
from mini_repro.experiment_common import pack_uniform, unpack_uniform


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def fresh_cache(blob):
    return None if blob is None else DynamicCache.from_legacy_cache(blob_to_legacy(blob))


def validate_blob(blob, expected):
    if blob.shape != expected.shape or blob.dtype != expected.dtype or not bool(torch.isfinite(blob).all()):
        raise RuntimeError('Restored KV shape, dtype, or finite-value check failed')


def snapshot_path(spec):
    import os
    root = Path(os.environ.get('HF_HUB_CACHE', str(Path(os.environ['HF_HOME']) / 'hub')))
    return root / ('models--' + spec['id'].replace('/', '--')) / 'snapshots' / spec['revision']


class Engine:
    def __init__(self, spec, *, load=True):
        self.spec = spec
        self.snapshot = snapshot_path(spec)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), local_files_only=True)
        if load:
            self.load()

    def load(self):
        if self.spec['weight_loading'] == 'fastchat8':
            # Exact same FastChat groupwise weight path as the historical Mistral run.
            from fastchat.model import load_model
            self.model, _ = load_model(str(self.snapshot), device='cuda', num_gpus=1,
                                      max_gpu_memory='20GiB', load_8bit=True, cpu_offloading=False, debug=False)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(str(self.snapshot), local_files_only=True,
                torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda')
        self.model.eval()
        c = self.model.config
        actual = (c.num_hidden_layers, c.num_key_value_heads, c.hidden_size // c.num_attention_heads)
        expected = tuple(self.spec[k] for k in ('layers', 'kv_heads', 'head_dim'))
        if actual != expected or self.spec['context_limit'] > c.max_position_embeddings:
            raise ValueError(f'Adapter/model geometry mismatch: {actual} vs {expected}')
        if getattr(c, 'pretraining_tp', 1) != 1 or getattr(c, 'sliding_window', None):
            raise ValueError('This adapter requires pretraining_tp=1 and full attention')

    @torch.inference_mode()
    def forward(self, ids, blob=None, *, project=True, last_only=True):
        if not ids:
            raise ValueError('Cannot forward empty input')
        previous = 0 if blob is None else blob.shape[3]
        total = previous + len(ids)
        if total > self.spec['context_limit']:
            raise ValueError('Input/cache exceeds configured context window')
        inputs = torch.tensor([ids], dtype=torch.long, device='cuda')
        positions = torch.arange(previous, total, device='cuda')
        out = self.model.model(input_ids=inputs, past_key_values=fresh_cache(blob),
            attention_mask=torch.ones((1, total), dtype=torch.long, device='cuda'),
            position_ids=positions.unsqueeze(0), cache_position=positions, use_cache=True, return_dict=True)
        logits = None
        if project:
            hidden = out.last_hidden_state[:, -1:] if last_only else out.last_hidden_state
            logits = self.model.lm_head(hidden).float()
        return logits, out.past_key_values

    def prefill(self, ids, blob=None):
        _, cache = self.forward(ids, blob, project=False)
        result = legacy_to_blob(cache)
        expected = (self.spec['layers'], 2, self.spec['kv_heads'],
                    len(ids) + (0 if blob is None else blob.shape[3]), self.spec['head_dim'])
        if tuple(result.shape) != expected or result.dtype != torch.float16:
            raise RuntimeError(f'Unexpected KV: {result.shape}/{result.dtype}; expected {expected}/FP16')
        return result

    def first(self, context, suffix, blob=None):
        return self.first_state(context, suffix, blob)[0]

    def first_state(self, context, suffix, blob=None):
        logits, cache = self.forward(suffix if blob is not None else context + suffix, blob)
        return int(logits[0, -1].argmax()), cache

    def generate(self, suffix, blob, limit, initial=None):
        ids = suffix
        generated = []
        current = blob
        reason = 'token_limit'
        for step in range(limit):
            if step == 0 and initial is not None:
                token, cache = initial
            else:
                logits, cache = self.forward(ids, current)
                token = int(logits[0, -1].argmax())
            if token == self.tokenizer.eos_token_id:
                reason = 'eos'
                break
            generated.append(token)
            if step + 1 < limit:
                current = legacy_to_blob(cache)
                ids = [token]
        return {'prediction': self.tokenizer.decode(generated, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False), 'generated_ids': generated, 'stop_reason': reason}

    @torch.inference_mode()
    def nll(self, blob, context, continuation):
        # Replay the last context token against restored KV[0:L-1]. Its logits
        # score continuation[0], then teacher forcing scores every remaining token.
        prefix = blob[:, :, :, :-1, :] if blob.shape[3] > 1 else None
        logits, _ = self.forward(context[-1:] + continuation[:-1], prefix, last_only=False)
        targets = torch.tensor(continuation, device='cuda')
        losses = F.cross_entropy(logits[0].float(), targets, reduction='none')
        if not bool(torch.isfinite(losses).all()):
            raise RuntimeError('Nonfinite NLL')
        return {'nll': float(losses.mean()), 'token_nll': losses.tolist(),
                'scored_ids': continuation, 'tokens_scored': len(continuation)}

    def codec_schedule(self, level):
        from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig
        keys = {1: (16, 16, 12), 2: (32, 16, 16), 3: (32, 32, 32)}[level]
        values = {1: (16, 12), 2: (32, 16), 3: (32, 32)}[level]
        spec = self.spec
        return CacheGenConfig(*spec['key_boundaries'], *keys, spec['value_boundary'], *values)

    def codec(self, tokens, level):
        from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
        from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
        from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
        schedule = self.codec_schedule(level)
        config = LMCacheEngineConfig.from_defaults(chunk_size=tokens)
        metadata = LMCacheEngineMetadata(self.spec['id'], 1, 0, 'huggingface')
        return (CacheGenSerializer(config, metadata, cachegen_config=schedule),
                CacheGenDeserializer(config, metadata, cachegen_config=schedule))

    def encode(self, blob, method):
        if method == 'raw':
            def serialize():
                stream = io.BytesIO()
                torch.save(blob.cpu(), stream)
                return stream.getvalue()
            payload, seconds = timed(serialize)
            restore = lambda p: torch.load(io.BytesIO(p), map_location='cuda', weights_only=True)
        elif method.startswith('uniform'):
            payload, seconds = timed(lambda: pack_uniform(blob, int(method[7:])))
            restore = unpack_uniform
        else:
            encoder, decoder = self.codec(blob.shape[3], int(method[-1]))
            payload, seconds = timed(lambda: encoder.to_bytes(blob))
            restore = decoder.from_bytes
        return payload, seconds, restore

    def restore_function(self, method, tokens):
        """Reopen a persisted prepared payload without encoding it a second time."""
        if method == 'text':
            return self.restore_text
        if method == 'raw':
            return lambda p: torch.load(io.BytesIO(p), map_location='cuda', weights_only=True)
        if method in ('uniform4', 'uniform8'):
            return unpack_uniform
        if method in ('cachegen1', 'cachegen2', 'cachegen3'):
            return self.codec(tokens, int(method[-1]))[1].from_bytes
        raise ValueError(f'Unknown representation: {method}')

    def text_payload(self, ids):
        text = self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        patches = []
        if encoded != ids:
            for tag, a, b, c, d in difflib.SequenceMatcher(None, encoded, ids, autojunk=False).get_opcodes():
                if tag != 'equal':
                    patches.append([a, b - a, ids[c:d]])
        # Standalone subword chunks need explicit token-boundary repair. These
        # bytes are transmitted and counted, rather than replaying hidden IDs.
        payload = json.dumps({'format': 'utf8_token_patch_v1', 'text': text, 'patches': patches},
                             ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        if self.restore_text(payload) != ids:
            raise ValueError('Text wire failed exact token identity')
        return payload

    def restore_text(self, payload):
        saved = json.loads(payload.decode('utf-8'))
        if saved['format'] != 'utf8_token_patch_v1':
            raise ValueError('Unknown text wire format')
        ids = self.tokenizer.encode(saved['text'], add_special_tokens=False)
        for start, count, insert in reversed(saved['patches']):
            if not 0 <= start <= start + count <= len(ids):
                raise ValueError('Invalid text token patch')
            ids[start:start + count] = insert
        return ids

    def text_metadata(self, payload):
        saved = json.loads(payload.decode('utf-8'))
        return {'text_wire': saved['format'], 'text_utf8_bytes': len(saved['text'].encode('utf-8')),
                'token_patch_count': len(saved['patches']),
                'patch_inserted_tokens': sum(len(p[2]) for p in saved['patches'])}

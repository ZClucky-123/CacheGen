import json
import os
import time
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['QUANT_LEVEL'] = '2'
import torch
from src.utils import define_model_and_tokenizer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer

MODEL = 'mistral-community/Mistral-7B-v0.2'
REVISION = '80cf49b18de6354fd8a6d1e81c177dd830e05eea'
started = time.monotonic()
model, tokenizer = define_model_and_tokenizer(MODEL, num_gpus=1, max_gpu_memory=14)
model.eval()
inputs = tokenizer('Explain in one short sentence why caching helps repeated computation.', return_tensors='pt').to('cuda')
with torch.inference_mode():
    output = model(**inputs, use_cache=True)
    assert torch.isfinite(output.logits).all(), 'Nonfinite model logits'
    cache = output.past_key_values
    if hasattr(cache, 'to_legacy_cache'):
        cache = cache.to_legacy_cache()
    blob = torch.stack([torch.stack((k[0], v[0]), dim=0) for k, v in cache])
    config = LMCacheEngineConfig.from_defaults(chunk_size=blob.shape[-2])
    metadata = LMCacheEngineMetadata(MODEL, 1, 0, 'huggingface')
    encoder = CacheGenSerializer(config, metadata)
    decoder = CacheGenDeserializer(config, metadata)
    payload = encoder.to_bytes(blob)
    restored = decoder.from_bytes(payload)
    assert restored.shape == blob.shape, 'KV shape mismatch'
    assert torch.isfinite(restored).all(), 'Nonfinite decoded KV'
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    restored_cache = tuple((restored[i, 0].unsqueeze(0), restored[i, 1].unsqueeze(0)) for i in range(restored.shape[0]))
    continued = model(input_ids=next_token, past_key_values=restored_cache, use_cache=True)
    assert torch.isfinite(continued.logits).all(), 'Nonfinite logits after decoded KV reuse'
    generated = model.generate(**inputs, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    torch.cuda.synchronize()
    result = {
        'status': 'passed', 'model': MODEL, 'revision': REVISION,
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'compiled_architectures': ['sm_75', 'sm_89'],
        'model_loading': 'original src.utils.define_model_and_tokenizer (FastChat load_8bit=True)',
        'input_tokens': inputs.input_ids.shape[-1], 'kv_shape': list(blob.shape),
        'raw_kv_bytes': blob.numel() * blob.element_size(), 'encoded_bytes': len(payload),
        'kv_mse': float((restored.float() - blob.float()).square().mean()),
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'generated_text': text, 'elapsed_seconds': round(time.monotonic() - started, 2),
    }
Path('/home/aaa/CacheGen/.venv/mistral-smoke.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
print('MISTRAL_GPU_SMOKE_PASSED', flush=True)

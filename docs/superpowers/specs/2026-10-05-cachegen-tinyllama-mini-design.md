# TinyLlama CacheGen Mini Reproduction Design

Approved in chat on 2026-10-05.

Goal: On the local RTX 4060 8 GB, exercise the author's CacheGen CUDA codec on real TinyLlama KV caches and compare size and answer preservation with uncompressed and uniform 8-bit caches.

Scope: Use TinyLlama/TinyLlama-1.1B-Chat-v1.0 in FP16. Use six 512/1024-token prefixes from bundled LongChat for compression measurements and five local fact-retrieval prompts for generation comparison. Record byte sizes, encode/decode times, KV error and answers. No training, paid API or remote benchmarking.

Constraints: Store environment, caches, model and outputs on D:. Prefer download mirrors and verify model weight hash. Keep author codec intact, add model-specific configuration and a separate mini runner. Do not claim the paper's 7B/70B speedups or long-context quality are reproduced.
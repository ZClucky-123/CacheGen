# TinyLlama CacheGen Three Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Provide runnable offline scripts for quality-size, TTFT, and adaptive streaming checks on the local TinyLlama setup.

**Architecture:** Extend the existing mini runner's model/KV helpers. Add focused shared experiment primitives and three command-line modules, each writing separate JSON. Keep the author's scripts untouched.

**Tech Stack:** Python 3.10, PyTorch 2.3.1 CUDA 12.1, Transformers 4.42.3, author's CacheGen GPU codec, unittest, Python sockets.

## Global Constraints

- All data, model weights, code and output stay on D:.
- No new download, package installation, paid API, or Linux GPU driver.
- Maximum context plus continuation or query is 2048 TinyLlama tokens.
- Existing `results/tinyllama/metrics.json` remains intact.
- Report measured local time separately from estimated trace transfer time.

---

### Task 1: Codec levels and reusable baselines

**Files:** `LMCache/lmcache/storage_backend/serde/cachegen_basics.py`, `mini_repro/run.py`, `mini_repro/experiment_common.py`, `tests/test_experiment_common.py`, `tests/test_tinyllama_config.py`.

- [x] Write tests for three 22-layer codec levels, packed 4-bit roundtrip, deterministic positioned facts, and deadline selection.
- [x] Run tests and confirm failures from missing behavior.
- [x] Implement levels and shared helpers; retain level 2 as the old default.
- [x] Run focused tests and a real 16-token CUDA codec roundtrip for levels 1/2/3.

### Task 2: Quality versus size

**Files:** `mini_repro/quality_size.py`, `tests/test_quality_size.py`.

- [x] Test held-out next-token NLL alignment and size accounting on small tensors.
- [x] Implement offline CLI for bundled LongChat continuation NLL and position-varied factual QA at configurable sizes/counts.
- [x] Run one natural prefix and one QA item per position with all baselines; inspect JSON.

### Task 3: Local TTFT emulation

**Files:** `mini_repro/ttft.py`, `tests/test_ttft.py`.

- [x] Test throttled socket transfer sends and receives exact bytes and records elapsed time.
- [x] Implement text, uniform-8, and CacheGen paths; time transfer, decode and first token separately.
- [x] Run one context at one bandwidth and verify identical context tokens and valid timing fields.

### Task 4: Adaptive trace experiment and documentation

**Files:** `mini_repro/adaptive.py`, `tests/test_adaptive.py`, `mini_repro/README.md`, `mini_repro/run_wsl.sh`.

- [x] Test selection decisions for fast/slow bandwidth and fixed strategy summaries.
- [x] Implement offline trace CLI with three levels and text fallback, per-case results and SLO summaries.
- [x] Run a one-case, one-trace smoke check; then run full tests and Git diff checks.
- [x] Document commands, output definitions, limitations and download-free operation; commit and push verified changes.

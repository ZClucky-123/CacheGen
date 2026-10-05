# TinyLlama CacheGen Mini Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Run a small local CacheGen method check with TinyLlama and report measured compression and answer preservation.

**Architecture:** Keep the original author scripts intact. Add TinyLlama codec configuration and a standalone mini runner that loads the model directly with Transformers, reuses the author's codec and uniform quantization, and writes JSON results.

**Tech Stack:** Ubuntu WSL2 on D:, Python 3.10, PyTorch 2.3.1 CUDA 12.1, Transformers 4.42.3, TinyLlama 1.1B, author's torchac_cuda.

**Spec:** docs/superpowers/specs/2026-10-05-cachegen-tinyllama-mini-design.md

## Global Constraints

- All large installations, caches, model weights and results stay on D:.
- Use no paid API or Linux GPU driver.
- Report actual measured results, including failures, without claiming full-paper reproduction.

## Review Focus

- TinyLlama's 22-layer grouped-query KV layout must survive codec roundtrip.
- Prefix token counts must fit TinyLlama's 2048-token context.
- Raw, uniform-quantized and CacheGen byte counts must include metadata.
- Generation must reuse only the context KV while passing the same question tokens.
- CUDA compiler must match the selected PyTorch CUDA build.

---

### Task 1: Prepare environment and data selection

- [x] Confirm Python venv, CUDA compiler and mirror connectivity in WSL.
- [x] Install minimal pinned dependencies and build torchac_cuda after download permission.
- [x] Create deterministic selection of six bundled LongChat prefixes and five local QA prompts.
- [x] Verify selected token lengths and save a manifest in the results JSON.

### Task 2: Add TinyLlama codec support

- [x] Write a failing test for TinyLlama's 22-layer configuration.
- [x] Add a TinyLlama branch to CacheGenConfig without changing existing model configs.
- [x] Run the focused test and Python syntax checks.

### Task 3: Build the mini runner

- [x] Write focused tests for data selection, byte accounting and local scoring.
- [x] Implement standalone FP16 model loading, KV extraction, author-codec roundtrip, uniform 8-bit baseline and deterministic generation.
- [x] Run focused tests and CUDA codec smoke checks.

### Task 4: Execute and report

- [x] Run six compression cases and five QA cases within the 2048-token limit.
- [x] Save raw JSON results and a short README with exact versions, disk use and limitations.
- [x] Verify Git diff, result files and D: WSL model/cache locations.

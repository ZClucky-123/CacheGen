# Shared v3 Stable Release Implementation Plan

> **For agentic workers:** Execute inline using systematic debugging, test-driven development and verification-before-completion. This is the continuation of the user's authorized repair, not a new feature proposal.

**Goal:** Deliver an executable, fixed source version with real large-context validation and honest paper boundaries.

**Architecture:** Keep the current shared Engine, TTFT, history policy and ablations. Finish the immutable readiness batch before changing fingerprinted sources. Fix only demonstrated remaining framework defects, then freeze and verify the delivered version.

**Tech Stack:** Python 3.10, PyTorch 2.3.1+cu121, Transformers 4.42.3, original CacheGen CUDA kernels, tmux, SQLite.

**Spec:** User's v3 experiment requirements, local CacheGen SIGCOMM PDF §5–7/Appendix C, and explicit latest scope: no anchor/delta or paper scheduler additions.

## Global Constraints

- No new anchor/delta codec or paper scheduler; do not claim complete paper reproduction.
- Keep quantization, answers, QA/NLL thresholds and same-payload exact equality checks.
- No source edits during an immutable measurement batch; no old/new observation splicing.
- Preserve prior runs and archives. Subsequently authorized commit identity: 张驰 <241880426@smail.nju.edu.cn>; inspect push access without exposing credentials.

## Review Focus

- Actual shipped registry must discover larger runs for both models.
- Hard process environment loss must preserve committed records and support fresh tmux resume.
- Frozen source and execution profile must be identifiable against measured evidence.
- Released commands must use the real model/environment and common GPU lock.
- Paper claims must distinguish public-code subset, user's controlled additions and unimplemented paper components.

### Task 1: Finish existing readiness evidence

- [x] Complete 469 observations without code changes.
- [x] Check all 15 text KV comparisons, six NLL checks, 105 transport pairs, 4K cross-attempt equality and preserved 315/441-row hashes.
- [x] Verify artifact hashes (190 files) and record environment-restart interruption separately from intentional TERM.

### Task 2: Fix shipped default status scope

Files: `experiments/registry.json`, `experiments/workspace.py`, `tests/test_workspace.py`.

- [x] Run real-registry test and observe small selected instead of active larger for both models.
- [x] Correct registered status root to model root. Existing launch commands still choose their own small default config and output scale; no new registry setting required.
- [x] Run that regression and all 112 existing tests, plus 3 old Mistral compatibility tests.

### Task 3: Freeze and deliver executable source

Files: `docs/reproduction/LARGER_READINESS_20261025.md`, paper audit, fixed release directory and manifest.

- [x] Document PDF checksum/page references and the explicitly excluded anchor/delta and paper scheduler.
- [x] Freeze sources, data and environment references; verify SHA256 list and launch commands.
- [x] Run GPU preflight and correctness against frozen source; preserve evidence and give exact monitoring/resume commands.

## Execution ledger

- Paper audit: PDF is 19 pages, SIGCOMM version; §5.2 anchor/delta and offline model CDF are absent from current public-code path. User explicitly declined adding them and the exact paper scheduler.
- Readiness attempt 1: intentional TERM at 315 observations/4K; 14 prepared representations verified. Attempt 2: environment processes disappeared at 441 observations (container init changed); SQLite readable, no cgroup OOM event. Attempt 3 resumes same identity.
- No new feature implementation or worktree reset is needed; user requested repairs in this already modified workspace. Continue inline without approval pauses.
- Final verification: 112 tests and 3 compatibility tests pass. Frozen-source rerun initially found a missing compatibility symlink; restored the same relative layout and reran all 112 successfully. TinyLlama frozen-source 274-row full acceptance and 15 large-context/six NLL checks pass. Mistral 469-row archive has 190 verified files. Measurement code unchanged after readiness; final differences are registry scope and its test.

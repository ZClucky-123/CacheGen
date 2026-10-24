# Experiment workspace implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. User authorized implementation and GitHub upload in the current directory.

**Goal:** Consolidate files under CacheGen and make future experiments relocatable and extensible.
**Architecture:** Registry-based standard-library CLI, shared path discovery and shell environment; family code separated from immutable runs, reports and local dependencies. Preserve old imports and relative compatibility links.
**Tech Stack:** Python 3.10+, Bash, existing PyTorch/CUDA environment.
**Spec:** `docs/superpowers/specs/2026-10-24-experiment-workspace-design.md`.

## Global constraints

- Work in `/home/aaa/CacheGen` on the existing feature branch; no extra checkout because the task reorganizes these exact local files.
- Preserve historical experiment data and report bundle bytes; no model measurement rerun.
- Upload this repository only; exclude environments, model weights and download/build dependencies.

## Review focus

- Checkout moved to a directory containing spaces; direct legacy scripts must still resolve imports.
- Fresh clone has no latest result; status must explain this without a traceback.
- Result/cache overrides must be honored consistently by CLI and shell entrypoints.
- Interruptions and overlapping GPU runs must retain results and reject concurrent measurement.
- A new registered experiment must run without editing dispatcher code.

### Task 1: Move data and add extensible dispatch

Files: `experiments/{workspace.py,__main__.py,registry.json}`, `tests/test_workspace.py`, directory layout and compatibility links.

- [ ] Snapshot originals, move result/report/cache directories with checksum verification, move Mistral source and remaining home files.
- [ ] Write and run failing tests for relocation, registry execution, no-result status, overrides and dry-run.
- [ ] Implement standard-library path, dispatch and status interfaces, then pass the tests.

### Task 2: Adapt existing entrypoints

Files: `scripts/experiment-env.sh`, Mistral entrypoints, `mini_repro/run_local.sh`, setup scripts, compatibility tests.

- [ ] Add failing tests for config forwarding, legacy entrypoints from a relocated tree and GPU lock exclusion.
- [ ] Remove active absolute paths, share environment discovery, snapshot configs and preserve existing measurement algorithms.
- [ ] Run original and new tests and verify historical artifacts.

### Task 3: Document, review and upload

Files: repository README, experiment/result/report indexes, Git ignore rules and audit records.

- [ ] Replace obsolete navigation, document extension/configuration/compatibility contracts and local dependencies.
- [ ] Back up index, exclude staged CUDA and include verified success results and reports.
- [ ] Run full suite, integrity and staged content checks; get independent code review.
- [ ] Commit and upload current feature branch, verifying the remote commit and final home directory.

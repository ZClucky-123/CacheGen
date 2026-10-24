# tinyllama / minimal / cachegen-shared-v2

Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 @ fe8a4ea1ffedaf415f4da2f062534de366a451e6; weights: fp16.

All 6 core tasks and 134 expected observations verified. TriviaQA is supplementary inside the quality task.

## Protocol and limits

NLL scores every held-out token 1..N. The final context token is replayed against restored KV[0:L-1]; no historical v1 NLL is pooled. Negative ΔNLL alone does not show improved model quality.

QA is deterministic greedy decoding until EOS or the configured token limit. Raw generation and token IDs are retained. Leading LF characters are stripped, then only the first LF line is scored with standard normalized EM/F1 and predeclared dataset aliases. Controlled QA and TriviaQA are separate. Historical phrase matching is diagnostic only.

The per-length descriptive gate remains ΔNLL ≤0.02 and EM retention ≥90% of raw-correct controlled questions at that same length. With zero raw-correct questions the result is unassessable. No thresholds or aliases were tuned on results.

TTFT starts before socket setup and ends after the first generated token ID is available on CPU, including transfer, restore/tokenization, and remaining-input inference with CUDA synchronization. It excludes offline prefill/encoding, validation, and subsequent generation. Only the final required hidden state is projected to vocabulary for both text and cache methods. No network interface shaping is used.

Offline prefill and representation encoding are measured separately. Their durations are once-per-context annotations repeated on rows for traceability, not costs incurred on each timed transfer. Adaptive calibration additionally records all-level encoding and preparation wall time, including raw QA/warmups/validation; it is not a minimal production cache-build estimate.

Text wire format utf8_token_patch_v1 sends UTF-8 and explicit token-repair metadata for standalone subword boundaries or concatenated token segments. The receiver actually retokenizes and applies only the transmitted edits; all framing and token patches count in bytes. Patch counts are recorded. This is not an unmodified plain-text-only wire benchmark.

Adaptive mode: socket_serial. socket_serial means actual paced local socket bytes and sequential restoration/recomputation. trace_estimate means estimated bytes/bandwidth network time plus executed local work, never a measured end-to-end network result. SLO applies to TTFT; full answer latency is separate.

Each adaptive cached chunk comes from a full original-context prefill, so later-layer KV retains original-prefix dependence. Text fallback actually recomputes on the assembled prior KV with explicit positions. This is the repository heuristic, not the paper scheduler. Transfer/decode overlap and anchor/delta coding are absent.

Byte ratios divide sums of actual serialized FP16 containers by sums of transmitted containers, including scales, CDFs, shapes and other metadata. Tensor-only ratios are separately named. Local socket target bandwidth is not measured throughput; see calibration.json and ttft.csv.

## Paired quality by length

| context_tokens | method | raw_serialized_over_method | paired_nll_delta | raw_em_correct | em_retention | acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | raw | 1 | 0 | 1 | 1 | pass |
| 128 | uniform4 | 3.932 | -0.030269 | 1 | 1 | pass |
| 128 | uniform8 | 1.9833 | 0.00070243 | 1 | 1 | pass |
| 128 | cachegen1 | 3.7063 | 0.13354 | 1 | 1 | fail |
| 128 | cachegen2 | 2.3607 | -0.027343 | 1 | 1 | pass |
| 128 | cachegen3 | 2.1167 | 0.0091892 | 1 | 1 | pass |
| 257 | raw | 1 | 0 | 1 | 1 | pass |
| 257 | uniform4 | 3.9353 | -0.062844 | 1 | 1 | pass |
| 257 | uniform8 | 1.9839 | 0.004735 | 1 | 1 | pass |
| 257 | cachegen1 | 4.8281 | -0.023568 | 1 | 1 | pass |
| 257 | cachegen2 | 3.346 | -0.061345 | 1 | 1 | pass |
| 257 | cachegen3 | 2.8649 | 0.026119 | 1 | 1 | fail |

## Separate QA datasets

| dataset | method | independent_questions_or_facts | conditions | mean_em | mean_f1 |
| --- | --- | --- | --- | --- | --- |
| controlled | raw | 1 | 6 | 0.33333 | 0.46667 |
| controlled | uniform4 | 1 | 6 | 0.33333 | 0.4 |
| controlled | uniform8 | 1 | 6 | 0.33333 | 0.46667 |
| controlled | cachegen1 | 1 | 6 | 0.33333 | 0.4 |
| controlled | cachegen2 | 1 | 6 | 0.33333 | 0.4 |
| controlled | cachegen3 | 1 | 6 | 0.5 | 0.56667 |
| triviaqa | raw | 1 | 1 | 0 | 0.4 |
| triviaqa | uniform4 | 1 | 1 | 0 | 0.4 |
| triviaqa | uniform8 | 1 | 1 | 0 | 0.4 |
| triviaqa | cachegen1 | 1 | 1 | 0 | 0.4 |
| triviaqa | cachegen2 | 1 | 1 | 0 | 0.4 |
| triviaqa | cachegen3 | 1 | 1 | 0 | 0.66667 |

## TTFT

| context_tokens | target_bandwidth_gbps | method | timing_observations | median_seconds | mean_actual_bandwidth_gbps | speedup_vs_text |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 0.1 | cachegen1 | 2 | 0.093212 | 0.099476 | 0.21024 |
| 128 | 0.1 | cachegen2 | 2 | 0.13386 | 0.09953 | 0.1464 |
| 128 | 0.1 | cachegen3 | 2 | 0.14557 | 0.099579 | 0.13462 |
| 128 | 0.1 | raw | 2 | 0.26628 | 0.099635 | 0.073593 |
| 128 | 0.1 | text | 2 | 0.019597 | 0.0099584 | 1 |
| 128 | 0.1 | uniform4 | 2 | 0.090092 | 0.099561 | 0.21752 |
| 128 | 0.1 | uniform8 | 2 | 0.14852 | 0.099598 | 0.13195 |
| 128 | 10 | cachegen1 | 2 | 0.022046 | 6.9326 | 0.97839 |
| 128 | 10 | cachegen2 | 2 | 0.022603 | 7.7938 | 0.95428 |
| 128 | 10 | cachegen3 | 2 | 0.02272 | 8.041 | 0.94936 |
| 128 | 10 | raw | 2 | 0.024433 | 8.6905 | 0.88281 |
| 128 | 10 | text | 2 | 0.021569 | 0.013972 | 1 |
| 128 | 10 | uniform4 | 2 | 0.021741 | 6.7239 | 0.9921 |
| 128 | 10 | uniform8 | 2 | 0.022708 | 7.4959 | 0.94988 |
| 257 | 0.1 | cachegen1 | 2 | 0.13149 | 0.099536 | 0.15124 |
| 257 | 0.1 | cachegen2 | 2 | 0.17478 | 0.099675 | 0.11377 |
| 257 | 0.1 | cachegen3 | 2 | 0.20002 | 0.099668 | 0.099421 |
| 257 | 0.1 | raw | 2 | 0.49898 | 0.09958 | 0.039853 |
| 257 | 0.1 | text | 2 | 0.019886 | 0.019608 | 1 |
| 257 | 0.1 | uniform4 | 2 | 0.14829 | 0.099562 | 0.1341 |
| 257 | 0.1 | uniform8 | 2 | 0.26927 | 0.099739 | 0.07385 |
| 257 | 10 | cachegen1 | 2 | 0.022908 | 8.0917 | 0.80201 |
| 257 | 10 | cachegen2 | 2 | 0.024653 | 7.7136 | 0.74523 |
| 257 | 10 | cachegen3 | 2 | 0.024396 | 8.5058 | 0.75308 |
| 257 | 10 | raw | 2 | 0.033479 | 8.4221 | 0.54878 |
| 257 | 10 | text | 2 | 0.018372 | 0.02985 | 1 |
| 257 | 10 | uniform4 | 2 | 0.022528 | 7.3703 | 0.81554 |
| 257 | 10 | uniform8 | 2 | 0.024606 | 8.72 | 0.74668 |

## Four adaptive configurations

| task | strategy | mode | fact_trace_conditions | timing_observations | median_ttft_seconds | slo_violation_rate | condition_mean_em | text_fallback_chunks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive-scale-0.01-slo-0.2 | adaptive | socket_serial | 1 | 1 | 0.088229 | 0 | 0 | 3 |
| adaptive-scale-0.01-slo-0.2 | adaptive_static | socket_serial | 1 | 1 | 0.090374 | 0 | 0 | 3 |
| adaptive-scale-0.01-slo-0.2 | fixed1 | socket_serial | 1 | 1 | 1.3958 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.2 | fixed2 | socket_serial | 1 | 1 | 2.2197 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.2 | fixed3 | socket_serial | 1 | 1 | 2.4256 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.2 | text | socket_serial | 1 | 1 | 0.090892 | 0 | 0 | 3 |
| adaptive-scale-0.01-slo-0.5 | adaptive | socket_serial | 1 | 1 | 0.43487 | 0 | 0 | 1 |
| adaptive-scale-0.01-slo-0.5 | adaptive_static | socket_serial | 1 | 1 | 0.34369 | 0 | 0 | 1 |
| adaptive-scale-0.01-slo-0.5 | fixed1 | socket_serial | 1 | 1 | 1.3934 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.5 | fixed2 | socket_serial | 1 | 1 | 2.2215 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.5 | fixed3 | socket_serial | 1 | 1 | 2.436 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.5 | text | socket_serial | 1 | 1 | 0.096411 | 0 | 0 | 3 |
| adaptive-scale-1-slo-0.2 | adaptive | socket_serial | 1 | 1 | 0.073755 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | adaptive_static | socket_serial | 1 | 1 | 0.069443 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | fixed1 | socket_serial | 1 | 1 | 0.052395 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | fixed2 | socket_serial | 1 | 1 | 0.054635 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | fixed3 | socket_serial | 1 | 1 | 0.055855 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | text | socket_serial | 1 | 1 | 0.095085 | 0 | 0 | 3 |
| adaptive-scale-1-slo-0.5 | adaptive | socket_serial | 1 | 1 | 0.064023 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | adaptive_static | socket_serial | 1 | 1 | 0.055734 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | fixed1 | socket_serial | 1 | 1 | 0.042328 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | fixed2 | socket_serial | 1 | 1 | 0.05516 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | fixed3 | socket_serial | 1 | 1 | 0.057036 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | text | socket_serial | 1 | 1 | 0.097181 | 0 | 0 | 3 |

## Statistical interpretation

2000 seeded source-cluster bootstrap draws; with few sources these are descriptive, not equivalence evidence

Independent documents/questions/facts, derived length/position/trace conditions and timing repeats are separately counted. Cluster intervals keep all conditions and repeats of a source together. Small-sample P95 is descriptive. A completed matrix establishes execution coverage, not paper-scale reproduction or statistical equivalence.

## Evidence

config.json, selection.json, metadata.json, preflight.json, cache_checks.json, calibration.json, records.jsonl, events.jsonl, records.sqlite3, manifest.json, failures.jsonl (if any), dependencies.txt, hardware.txt, source/, artifacts.json, and run.log preserve the protocol, data, code, environment, order, outputs and checkpoints.

Resume: `bash experiments/run.sh --resume /home/aaa/CacheGen/results/tinyllama/minimal/v2-final-check`. Integrity: `python3 -m experiments.suite --verify /home/aaa/CacheGen/results/tinyllama/minimal/v2-final-check`.

## Figures

![Size and NLL](figures/quality.png)

![TTFT](figures/ttft.png)

![Adaptive](figures/adaptive.png)

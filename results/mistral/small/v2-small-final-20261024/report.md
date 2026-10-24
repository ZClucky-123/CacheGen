# mistral / small / cachegen-shared-v2

Model: mistral-community/Mistral-7B-v0.2 @ 80cf49b18de6354fd8a6d1e81c177dd830e05eea; weights: fastchat8.

All 6 core tasks and 552 expected observations verified. TriviaQA is supplementary inside the quality task.

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
| 256 | raw | 1 | 0 | 6 | 1 | pass |
| 256 | uniform4 | 3.9839 | 0.028702 | 6 | 1 | fail |
| 256 | uniform8 | 1.996 | -0.0017103 | 6 | 1 | pass |
| 256 | cachegen1 | 5.5314 | 0.018124 | 6 | 1 | pass |
| 256 | cachegen2 | 3.6834 | 0.023021 | 6 | 1 | fail |
| 256 | cachegen3 | 3.1013 | 0.013114 | 6 | 1 | pass |
| 512 | raw | 1 | 0 | 6 | 1 | pass |
| 512 | uniform4 | 3.9842 | 0.00084239 | 6 | 1 | pass |
| 512 | uniform8 | 1.996 | 0.00031152 | 6 | 1 | pass |
| 512 | cachegen1 | 6.804 | 0.059052 | 6 | 1 | fail |
| 512 | cachegen2 | 4.8394 | 0.0093304 | 6 | 1 | pass |
| 512 | cachegen3 | 3.8741 | 0.021964 | 6 | 1 | fail |

## Separate QA datasets

| dataset | method | independent_questions_or_facts | conditions | mean_em | mean_f1 |
| --- | --- | --- | --- | --- | --- |
| controlled | raw | 2 | 12 | 1 | 1 |
| controlled | uniform4 | 2 | 12 | 1 | 1 |
| controlled | uniform8 | 2 | 12 | 1 | 1 |
| controlled | cachegen1 | 2 | 12 | 1 | 1 |
| controlled | cachegen2 | 2 | 12 | 1 | 1 |
| controlled | cachegen3 | 2 | 12 | 1 | 1 |
| triviaqa | raw | 4 | 4 | 0.75 | 0.91667 |
| triviaqa | uniform4 | 4 | 4 | 0.75 | 0.91667 |
| triviaqa | uniform8 | 4 | 4 | 0.75 | 0.91667 |
| triviaqa | cachegen1 | 4 | 4 | 0.75 | 0.91667 |
| triviaqa | cachegen2 | 4 | 4 | 0.5 | 0.66667 |
| triviaqa | cachegen3 | 4 | 4 | 0.5 | 0.66667 |

## TTFT

| context_tokens | target_bandwidth_gbps | method | timing_observations | median_seconds | mean_actual_bandwidth_gbps | speedup_vs_text |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 0.1 | cachegen2 | 6 | 0.79377 | 0.099767 | 0.10677 |
| 256 | 0.1 | cachegen3 | 6 | 0.93727 | 0.099746 | 0.090424 |
| 256 | 0.1 | text | 6 | 0.084752 | 0.023792 | 1 |
| 256 | 0.1 | uniform8 | 6 | 1.4189 | 0.099591 | 0.059731 |
| 256 | 1 | cachegen2 | 6 | 0.14313 | 0.96734 | 0.55335 |
| 256 | 1 | cachegen3 | 6 | 0.1562 | 0.9691 | 0.50704 |
| 256 | 1 | text | 6 | 0.079199 | 0.023474 | 1 |
| 256 | 1 | uniform8 | 6 | 0.20404 | 0.95296 | 0.38815 |
| 256 | 10 | cachegen2 | 6 | 0.069455 | 8.1232 | 1.1413 |
| 256 | 10 | cachegen3 | 6 | 0.071817 | 7.5984 | 1.1038 |
| 256 | 10 | text | 6 | 0.07927 | 0.024531 | 1 |
| 256 | 10 | uniform8 | 6 | 0.08012 | 7.5405 | 0.98938 |
| 512 | 0.1 | cachegen2 | 6 | 1.187 | 0.099534 | 0.10413 |
| 512 | 0.1 | cachegen3 | 6 | 1.4676 | 0.099445 | 0.084223 |
| 512 | 0.1 | text | 6 | 0.1236 | 0.049698 | 1 |
| 512 | 0.1 | uniform8 | 6 | 2.8188 | 0.098743 | 0.04385 |
| 512 | 1 | cachegen2 | 6 | 0.18446 | 0.96873 | 0.62789 |
| 512 | 1 | cachegen3 | 6 | 0.21651 | 0.96157 | 0.53494 |
| 512 | 1 | text | 6 | 0.11582 | 0.044365 | 1 |
| 512 | 1 | uniform8 | 6 | 0.39267 | 0.89618 | 0.29496 |
| 512 | 10 | cachegen2 | 6 | 0.082081 | 7.6095 | 1.4087 |
| 512 | 10 | cachegen3 | 6 | 0.092409 | 6.8015 | 1.2512 |
| 512 | 10 | text | 6 | 0.11563 | 0.050102 | 1 |
| 512 | 10 | uniform8 | 6 | 0.16589 | 3.9973 | 0.697 |

## Four adaptive configurations

| task | strategy | mode | fact_trace_conditions | timing_observations | median_ttft_seconds | slo_violation_rate | condition_mean_em | text_fallback_chunks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive-scale-0.01-slo-0.2 | adaptive | socket_serial | 6 | 12 | 0.24786 | 1 | 1 | 36 |
| adaptive-scale-0.01-slo-0.2 | adaptive_static | socket_serial | 6 | 12 | 0.24799 | 1 | 1 | 36 |
| adaptive-scale-0.01-slo-0.2 | fixed1 | socket_serial | 6 | 12 | 6.5374 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.2 | fixed2 | socket_serial | 6 | 12 | 10.134 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.2 | fixed3 | socket_serial | 6 | 12 | 11.883 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.2 | text | socket_serial | 6 | 12 | 0.24927 | 1 | 1 | 36 |
| adaptive-scale-0.01-slo-0.5 | adaptive | socket_serial | 6 | 12 | 0.36728 | 0.083333 | 1 | 30 |
| adaptive-scale-0.01-slo-0.5 | adaptive_static | socket_serial | 6 | 12 | 0.24815 | 0 | 1 | 36 |
| adaptive-scale-0.01-slo-0.5 | fixed1 | socket_serial | 6 | 12 | 6.5366 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.5 | fixed2 | socket_serial | 6 | 12 | 10.137 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.5 | fixed3 | socket_serial | 6 | 12 | 11.878 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.5 | text | socket_serial | 6 | 12 | 0.24827 | 0 | 1 | 36 |
| adaptive-scale-1-slo-0.2 | adaptive | socket_serial | 6 | 12 | 0.17258 | 0 | 1 | 6 |
| adaptive-scale-1-slo-0.2 | adaptive_static | socket_serial | 6 | 12 | 0.16822 | 0.083333 | 1 | 6 |
| adaptive-scale-1-slo-0.2 | fixed1 | socket_serial | 6 | 12 | 0.14592 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.2 | fixed2 | socket_serial | 6 | 12 | 0.1937 | 0.5 | 1 | 0 |
| adaptive-scale-1-slo-0.2 | fixed3 | socket_serial | 6 | 12 | 0.2176 | 0.5 | 1 | 0 |
| adaptive-scale-1-slo-0.2 | text | socket_serial | 6 | 12 | 0.24822 | 1 | 1 | 36 |
| adaptive-scale-1-slo-0.5 | adaptive | socket_serial | 6 | 12 | 0.21748 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | adaptive_static | socket_serial | 6 | 12 | 0.21702 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | fixed1 | socket_serial | 6 | 12 | 0.14711 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | fixed2 | socket_serial | 6 | 12 | 0.20162 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | fixed3 | socket_serial | 6 | 12 | 0.21218 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | text | socket_serial | 6 | 12 | 0.2474 | 0 | 1 | 36 |

## Statistical interpretation

2000 seeded source-cluster bootstrap draws; with few sources these are descriptive, not equivalence evidence

Independent documents/questions/facts, derived length/position/trace conditions and timing repeats are separately counted. Cluster intervals keep all conditions and repeats of a source together. Small-sample P95 is descriptive. A completed matrix establishes execution coverage, not paper-scale reproduction or statistical equivalence.

## Evidence

config.json, selection.json, metadata.json, preflight.json, cache_checks.json, calibration.json, records.jsonl, events.jsonl, records.sqlite3, manifest.json, failures.jsonl (if any), dependencies.txt, hardware.txt, source/, artifacts.json, and run.log preserve the protocol, data, code, environment, order, outputs and checkpoints.

Resume: `bash experiments/run.sh --resume /home/aaa/CacheGen/results/mistral/small/v2-small-final-20261024`. Integrity: `python3 -m experiments.suite --verify /home/aaa/CacheGen/results/mistral/small/v2-small-final-20261024`.

## Figures

![Size and NLL](figures/quality.png)

![TTFT](figures/ttft.png)

![Adaptive](figures/adaptive.png)

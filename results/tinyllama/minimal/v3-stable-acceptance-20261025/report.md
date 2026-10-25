# tinyllama / minimal / cachegen-shared-v3

Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 @ fe8a4ea1ffedaf415f4da2f062534de366a451e6; weights: fp16.

All 7 core tasks and 274 expected observations verified. TriviaQA is supplementary inside the quality task.

## Protocol and limits

NLL scores every held-out token 1..N. The final context token is replayed against restored KV[0:L-1]; no historical v1 NLL is pooled. Negative ΔNLL alone does not show improved model quality.

QA is deterministic greedy decoding until EOS or the configured token limit. Raw generation and token IDs are retained. Leading LF characters are stripped, then only the first LF line is scored with standard normalized EM/F1 and predeclared dataset aliases. Controlled QA and TriviaQA are separate. Historical phrase matching is diagnostic only.

The per-length descriptive gate remains ΔNLL ≤0.02 and EM retention ≥90% of raw-correct controlled questions at that same length. With zero raw-correct questions the result is unassessable. No thresholds or aliases were tuned on results.

TTFT starts before socket setup and ends after the first generated token ID is available on CPU, including transfer, restore/tokenization, and remaining-input inference with CUDA synchronization. It excludes offline prefill/encoding, validation, and subsequent generation. Only the final required hidden state is projected to vocabulary for both text and cache methods. No network interface shaping is used.

Offline prefill and representation encoding are measured separately. Their durations are once-per-context annotations repeated on rows for traceability, not costs incurred on each timed transfer. Adaptive calibration additionally records all-level encoding and preparation wall time, including raw QA/warmups/validation; it is not a minimal production cache-build estimate.

Text wire format utf8_token_patch_v1 sends UTF-8 and explicit token-repair metadata for standalone subword boundaries or concatenated token segments. The receiver actually retokenizes and applies only the transmitted edits; all framing and token patches count in bytes. Patch counts are recorded. This is not an unmodified plain-text-only wire benchmark.

The independent whole-text reference retains its original optimized single context+suffix forward. Both chunk controls instead execute the same dependent context prefill per chunk, then suffix-only inference. Pipeline benefit is primarily evaluated against chunk serial with identical messages, not attributed from a whole/chunk comparison.

FP16 full-prefill and cache-reuse shapes can produce different greedy tokens at near-tied logits. cache_checks.json records actual IDs, first-divergence position and all reference-prefix logit comparisons. A nonidentical sequence is classified as numerical sensitivity only when its first divergent choices are within two FP16 representable steps and all vectors satisfy the existing atol=0.1/rtol=0.02 check; otherwise execution fails. Actual generations and QA/NLL scores are never repaired. Same-payload serial/pipeline tensors and generated IDs still require exact equality.

Text full-vs-chunk KV retains the strict elementwise atol=0.1/rtol=0.02 check. Per-head RMS and all elementwise outliers are additional diagnostics, never a substitute for the strict check. Stable execution uses fixed FP16 projection row tiles and explicit causal masks with the original efficient SDPA; all methods use the same execution settings and the final-position LM head remains untouched. Arithmetic calls use safe 96-token internal segments inside the existing <=256-token CUDA kernel, preventing rare 16-bit-CDF symbols from overflowing its 256-byte shared buffer. Actual segment sizes and their complete stream metadata count in bytes.

Execution settings (also checkpointed and fingerprinted): {"explicit_causal_mask": false, "fp16_reduced_precision_reduction": true, "linear_tile_tokens": 0, "sdpa_backend": "auto"}. Results with automatic kernels and stable kernels are separate batches; execution scheduling can affect absolute timing.

Adaptive mode: socket_serial. Protocol v3 executes measured, serial paced socket transfers; it rejects trace_estimate for history feedback. SLO applies to TTFT; full answer latency is separate.

History throughput variant adaptive_history uses a fixed first-block estimate of 1.0 Gbps, then the previous completed full-message bytes * 8 / effective receive interval. The interval starts after receiver capacity/buffer readiness, includes pre-send pacing and socket overhead, and excludes queue-capacity and previous GPU waits. The history policy cannot access current or future trace values; those are executor inputs and posthoc diagnostics. Both adaptive variants share candidate costs, remaining budget, first-token reserve and fallback rules. This is a block-level trace simulation, not the complete paper scheduler.

Each adaptive cached chunk comes from a full original-context prefill, so later-layer KV retains original-prefix dependence. Text fallback actually recomputes on the assembled prior KV with explicit positions. This is the repository heuristic, not the paper scheduler. Adaptive execution is serial. TTFT separately compares whole, chunk serial and bounded chunk pipeline; anchor/delta coding is absent.

Byte ratios divide sums of actual serialized FP16 containers by sums of transmitted containers, including scales, CDFs, shapes and other metadata. Tensor-only ratios are separately named. Local socket target bandwidth is not measured throughput; see calibration.json and ttft.csv.

## Paired quality by length

| context_tokens | method | raw_serialized_over_method | paired_nll_delta | raw_em_correct | em_retention | acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | raw | 1 | 0 | 1 | 1 | pass |
| 128 | uniform4 | 3.932 | -0.030269 | 1 | 1 | pass |
| 128 | uniform8 | 1.9833 | 0.00070243 | 1 | 1 | pass |
| 128 | cachegen1 | 3.4734 | 0.13354 | 1 | 1 | fail |
| 128 | cachegen2 | 2.2641 | -0.027343 | 1 | 1 | pass |
| 128 | cachegen3 | 2.0388 | 0.0091892 | 1 | 1 | pass |
| 257 | raw | 1 | 0 | 1 | 1 | pass |
| 257 | uniform4 | 3.9353 | -0.062844 | 1 | 1 | pass |
| 257 | uniform8 | 1.9839 | 0.004735 | 1 | 1 | pass |
| 257 | cachegen1 | 4.6352 | -0.023568 | 1 | 1 | pass |
| 257 | cachegen2 | 3.2513 | -0.061345 | 1 | 1 | pass |
| 257 | cachegen3 | 2.7935 | 0.026119 | 1 | 1 | fail |

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

| context_tokens | target_bandwidth_gbps | method | mode | chunk_size | timing_observations | median_seconds | mean_actual_bandwidth_gbps | speedup_vs_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 0.1 | cachegen1 | chunk_pipeline | 128 | 2 | 0.099976 | 0.099303 | 0.45715 |
| 128 | 0.1 | cachegen1 | chunk_serial | 128 | 2 | 0.085726 | 0.099413 | 0.43992 |
| 128 | 0.1 | cachegen1 | whole | 128 | 2 | 0.086078 | 0.099101 | 0.18103 |
| 128 | 0.1 | cachegen2 | chunk_pipeline | 128 | 2 | 0.12366 | 0.099441 | 0.36959 |
| 128 | 0.1 | cachegen2 | chunk_serial | 128 | 2 | 0.1265 | 0.09951 | 0.29812 |
| 128 | 0.1 | cachegen2 | whole | 128 | 2 | 0.12461 | 0.099512 | 0.12505 |
| 128 | 0.1 | cachegen3 | chunk_pipeline | 128 | 2 | 0.1452 | 0.099502 | 0.31476 |
| 128 | 0.1 | cachegen3 | chunk_serial | 128 | 2 | 0.14096 | 0.099552 | 0.26754 |
| 128 | 0.1 | cachegen3 | whole | 128 | 2 | 0.14077 | 0.099524 | 0.11069 |
| 128 | 0.1 | raw | chunk_pipeline | 128 | 2 | 0.25505 | 0.099616 | 0.17919 |
| 128 | 0.1 | raw | chunk_serial | 128 | 2 | 0.25206 | 0.099666 | 0.14962 |
| 128 | 0.1 | raw | whole | 128 | 2 | 0.25355 | 0.099636 | 0.061456 |
| 128 | 0.1 | text | chunk_pipeline | 128 | 2 | 0.045704 | 0.017607 | 1 |
| 128 | 0.1 | text | chunk_serial | 128 | 2 | 0.037712 | 0.027455 | 1 |
| 128 | 0.1 | text | whole | 128 | 2 | 0.015582 | 0.02881 | 1 |
| 128 | 0.1 | uniform4 | chunk_pipeline | 128 | 2 | 0.089831 | 0.099242 | 0.50877 |
| 128 | 0.1 | uniform4 | chunk_serial | 128 | 2 | 0.089875 | 0.099147 | 0.41961 |
| 128 | 0.1 | uniform4 | whole | 128 | 2 | 0.086556 | 0.099325 | 0.18003 |
| 128 | 0.1 | uniform8 | chunk_pipeline | 128 | 2 | 0.13798 | 0.099559 | 0.33124 |
| 128 | 0.1 | uniform8 | chunk_serial | 128 | 2 | 0.14131 | 0.099623 | 0.26687 |
| 128 | 0.1 | uniform8 | whole | 128 | 2 | 0.13828 | 0.099378 | 0.11269 |
| 128 | 10 | cachegen1 | chunk_pipeline | 128 | 2 | 0.026022 | 3.4553 | 1.4724 |
| 128 | 10 | cachegen1 | chunk_serial | 128 | 2 | 0.024497 | 4.015 | 1.3662 |
| 128 | 10 | cachegen1 | whole | 128 | 2 | 0.022396 | 5.3365 | 0.8976 |
| 128 | 10 | cachegen2 | chunk_pipeline | 128 | 2 | 0.028723 | 4.9803 | 1.3339 |
| 128 | 10 | cachegen2 | chunk_serial | 128 | 2 | 0.02543 | 5.6776 | 1.3161 |
| 128 | 10 | cachegen2 | whole | 128 | 2 | 0.029747 | 5.7922 | 0.67578 |
| 128 | 10 | cachegen3 | chunk_pipeline | 128 | 2 | 0.023093 | 6.4491 | 1.6592 |
| 128 | 10 | cachegen3 | chunk_serial | 128 | 2 | 0.028768 | 6.3476 | 1.1633 |
| 128 | 10 | cachegen3 | whole | 128 | 2 | 0.020198 | 5.466 | 0.99527 |
| 128 | 10 | raw | chunk_pipeline | 128 | 2 | 0.025013 | 6.8474 | 1.5318 |
| 128 | 10 | raw | chunk_serial | 128 | 2 | 0.032229 | 6.1979 | 1.0384 |
| 128 | 10 | raw | whole | 128 | 2 | 0.026125 | 7.6626 | 0.76948 |
| 128 | 10 | text | chunk_pipeline | 128 | 2 | 0.038315 | 0.045063 | 1 |
| 128 | 10 | text | chunk_serial | 128 | 2 | 0.033467 | 0.041641 | 1 |
| 128 | 10 | text | whole | 128 | 2 | 0.020103 | 0.044056 | 1 |
| 128 | 10 | uniform4 | chunk_pipeline | 128 | 2 | 0.026714 | 5.1581 | 1.4343 |
| 128 | 10 | uniform4 | chunk_serial | 128 | 2 | 0.026269 | 4.8322 | 1.274 |
| 128 | 10 | uniform4 | whole | 128 | 2 | 0.022457 | 4.2971 | 0.89515 |
| 128 | 10 | uniform8 | chunk_pipeline | 128 | 2 | 0.030814 | 5.3962 | 1.2434 |
| 128 | 10 | uniform8 | chunk_serial | 128 | 2 | 0.018025 | 7.1798 | 1.8567 |
| 128 | 10 | uniform8 | whole | 128 | 2 | 0.021938 | 6.0059 | 0.91636 |
| 257 | 0.1 | cachegen1 | chunk_pipeline | 128 | 2 | 0.2001 | 0.098955 | 0.44457 |
| 257 | 0.1 | cachegen1 | chunk_serial | 128 | 2 | 0.20662 | 0.099126 | 0.36945 |
| 257 | 0.1 | cachegen1 | whole | 257 | 2 | 0.12458 | 0.099483 | 0.17648 |
| 257 | 0.1 | cachegen2 | chunk_pipeline | 128 | 2 | 0.29877 | 0.099453 | 0.29775 |
| 257 | 0.1 | cachegen2 | chunk_serial | 128 | 2 | 0.30079 | 0.099521 | 0.25378 |
| 257 | 0.1 | cachegen2 | whole | 257 | 2 | 0.17409 | 0.099601 | 0.12629 |
| 257 | 0.1 | cachegen3 | chunk_pipeline | 128 | 2 | 0.32072 | 0.09949 | 0.27737 |
| 257 | 0.1 | cachegen3 | chunk_serial | 128 | 2 | 0.32529 | 0.099511 | 0.23466 |
| 257 | 0.1 | cachegen3 | whole | 257 | 2 | 0.19123 | 0.099671 | 0.11498 |
| 257 | 0.1 | raw | chunk_pipeline | 128 | 2 | 0.4916 | 0.099425 | 0.18096 |
| 257 | 0.1 | raw | chunk_serial | 128 | 2 | 0.50549 | 0.099528 | 0.15101 |
| 257 | 0.1 | raw | whole | 257 | 2 | 0.50532 | 0.099102 | 0.04351 |
| 257 | 0.1 | text | chunk_pipeline | 128 | 2 | 0.088958 | 0.012867 | 1 |
| 257 | 0.1 | text | chunk_serial | 128 | 2 | 0.076335 | 0.021069 | 1 |
| 257 | 0.1 | text | whole | 257 | 2 | 0.021986 | 0.032353 | 1 |
| 257 | 0.1 | uniform4 | chunk_pipeline | 128 | 2 | 0.15206 | 0.098322 | 0.58502 |
| 257 | 0.1 | uniform4 | chunk_serial | 128 | 2 | 0.15401 | 0.098875 | 0.49564 |
| 257 | 0.1 | uniform4 | whole | 257 | 2 | 0.14156 | 0.09931 | 0.15532 |
| 257 | 0.1 | uniform8 | chunk_pipeline | 128 | 2 | 0.28041 | 0.099065 | 0.31724 |
| 257 | 0.1 | uniform8 | chunk_serial | 128 | 2 | 0.27084 | 0.099427 | 0.28184 |
| 257 | 0.1 | uniform8 | whole | 257 | 2 | 0.26858 | 0.09963 | 0.081861 |
| 257 | 10 | cachegen1 | chunk_pipeline | 128 | 2 | 0.028799 | 3.4626 | 3.0641 |
| 257 | 10 | cachegen1 | chunk_serial | 128 | 2 | 0.032765 | 4.5852 | 2.6366 |
| 257 | 10 | cachegen1 | whole | 257 | 2 | 0.025442 | 4.192 | 0.64786 |
| 257 | 10 | cachegen2 | chunk_pipeline | 128 | 2 | 0.035599 | 3.4798 | 2.4788 |
| 257 | 10 | cachegen2 | chunk_serial | 128 | 2 | 0.026641 | 6.1879 | 3.2427 |
| 257 | 10 | cachegen2 | whole | 257 | 2 | 0.038334 | 4.7322 | 0.42997 |
| 257 | 10 | cachegen3 | chunk_pipeline | 128 | 2 | 0.034278 | 3.6829 | 2.5744 |
| 257 | 10 | cachegen3 | chunk_serial | 128 | 2 | 0.043874 | 4.5754 | 1.969 |
| 257 | 10 | cachegen3 | whole | 257 | 2 | 0.028139 | 5.5134 | 0.58575 |
| 257 | 10 | raw | chunk_pipeline | 128 | 2 | 0.033802 | 7.0357 | 2.6106 |
| 257 | 10 | raw | chunk_serial | 128 | 2 | 0.041441 | 5.9143 | 2.0846 |
| 257 | 10 | raw | whole | 257 | 2 | 0.036444 | 6.7964 | 0.45227 |
| 257 | 10 | text | chunk_pipeline | 128 | 2 | 0.088245 | 0.045311 | 1 |
| 257 | 10 | text | chunk_serial | 128 | 2 | 0.086389 | 0.034042 | 1 |
| 257 | 10 | text | whole | 257 | 2 | 0.016483 | 0.068931 | 1 |
| 257 | 10 | uniform4 | chunk_pipeline | 128 | 2 | 0.03102 | 3.7665 | 2.8448 |
| 257 | 10 | uniform4 | chunk_serial | 128 | 2 | 0.028941 | 3.889 | 2.985 |
| 257 | 10 | uniform4 | whole | 257 | 2 | 0.02923 | 4.8718 | 0.56389 |
| 257 | 10 | uniform8 | chunk_pipeline | 128 | 2 | 0.032663 | 4.7045 | 2.7017 |
| 257 | 10 | uniform8 | chunk_serial | 128 | 2 | 0.030723 | 6.8762 | 2.8119 |
| 257 | 10 | uniform8 | whole | 257 | 2 | 0.031408 | 6.1775 | 0.52478 |

## Paired pipeline comparison (same bytes)

| context_tokens | target_bandwidth_gbps | method | chunk_size | paired_observations | median_paired_speedup | mean_pipeline_overlap_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 0.1 | cachegen1 | 128 | 2 | 0.85749 | 0 |
| 128 | 0.1 | cachegen2 | 128 | 2 | 1.0234 | 0 |
| 128 | 0.1 | cachegen3 | 128 | 2 | 0.97068 | 0 |
| 128 | 0.1 | raw | 128 | 2 | 0.98866 | 0 |
| 128 | 0.1 | text | 128 | 2 | 0.83182 | 0 |
| 128 | 0.1 | uniform4 | 128 | 2 | 1.0002 | 0 |
| 128 | 0.1 | uniform8 | 128 | 2 | 1.0257 | 0 |
| 128 | 10 | cachegen1 | 128 | 2 | 0.99944 | 0 |
| 128 | 10 | cachegen2 | 128 | 2 | 0.88878 | 0 |
| 128 | 10 | cachegen3 | 128 | 2 | 1.2643 | 0 |
| 128 | 10 | raw | 128 | 2 | 1.3141 | 0 |
| 128 | 10 | text | 128 | 2 | 0.93706 | 0 |
| 128 | 10 | uniform4 | 128 | 2 | 0.95124 | 0 |
| 128 | 10 | uniform8 | 128 | 2 | 0.61196 | 0 |
| 257 | 0.1 | cachegen1 | 128 | 2 | 1.0327 | 0.006864 |
| 257 | 0.1 | cachegen2 | 128 | 2 | 1.0067 | 0.0044264 |
| 257 | 0.1 | cachegen3 | 128 | 2 | 1.0149 | 0.0056533 |
| 257 | 0.1 | raw | 128 | 2 | 1.0286 | 0.00067925 |
| 257 | 0.1 | text | 128 | 2 | 0.85863 | 0.00091748 |
| 257 | 0.1 | uniform4 | 128 | 2 | 1.0212 | 0.0022491 |
| 257 | 0.1 | uniform8 | 128 | 2 | 0.96596 | 0.0022618 |
| 257 | 10 | cachegen1 | 128 | 2 | 1.199 | 0.0036858 |
| 257 | 10 | cachegen2 | 128 | 2 | 0.77489 | 0.0061922 |
| 257 | 10 | cachegen3 | 128 | 2 | 1.3691 | 0.0055428 |
| 257 | 10 | raw | 128 | 2 | 1.2396 | 0.00042851 |
| 257 | 10 | text | 128 | 2 | 0.97846 | 0.00023157 |
| 257 | 10 | uniform4 | 128 | 2 | 1.018 | 0.0014058 |
| 257 | 10 | uniform8 | 128 | 2 | 0.96572 | 0.00074392 |

Pipeline speedup is serial/pipeline for each matched source, length, bandwidth, method, chunk size and repeat. Values below 1 mean slower pipeline. Chunk payload hashes and first tokens must match. Receiver capacity and send-permission waits, pacing, socket blocking, restoration, and overlap intervals are exported in ttft_timeline.csv / ttft_overlap.csv. All GPU restoration timestamps wait for CUDA completion.

## Adaptive configurations

| task | strategy | throughput_estimator | mode | fact_trace_conditions | timing_observations | median_ttft_seconds | slo_violation_rate | condition_mean_em | text_fallback_chunks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive-scale-0.01-slo-0.2 | adaptive | current_trace | socket_serial | 1 | 1 | 0.099111 | 0 | 0 | 3 |
| adaptive-scale-0.01-slo-0.2 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 2.0254 | 1 | 0 | 2 |
| adaptive-scale-0.01-slo-0.2 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.11199 | 0 | 0 | 3 |
| adaptive-scale-0.01-slo-0.2 | fixed1 | current_trace | socket_serial | 1 | 1 | 1.4681 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.2 | fixed2 | current_trace | socket_serial | 1 | 1 | 2.2886 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.2 | fixed3 | current_trace | socket_serial | 1 | 1 | 2.5144 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.2 | text | current_trace | socket_serial | 1 | 1 | 0.094681 | 0 | 0 | 3 |
| adaptive-scale-0.01-slo-0.5 | adaptive | current_trace | socket_serial | 1 | 1 | 0.42667 | 0 | 0 | 1 |
| adaptive-scale-0.01-slo-0.5 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 2.0636 | 1 | 0 | 2 |
| adaptive-scale-0.01-slo-0.5 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.33716 | 0 | 0 | 1 |
| adaptive-scale-0.01-slo-0.5 | fixed1 | current_trace | socket_serial | 1 | 1 | 1.4669 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.5 | fixed2 | current_trace | socket_serial | 1 | 1 | 2.293 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.5 | fixed3 | current_trace | socket_serial | 1 | 1 | 2.5101 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.5 | text | current_trace | socket_serial | 1 | 1 | 0.079083 | 0 | 0 | 3 |
| adaptive-scale-1-slo-0.2 | adaptive | current_trace | socket_serial | 1 | 1 | 0.069458 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 0.07088 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.053213 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | fixed1 | current_trace | socket_serial | 1 | 1 | 0.046914 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | fixed2 | current_trace | socket_serial | 1 | 1 | 0.061732 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | fixed3 | current_trace | socket_serial | 1 | 1 | 0.063743 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | text | current_trace | socket_serial | 1 | 1 | 0.098209 | 0 | 0 | 3 |
| adaptive-scale-1-slo-0.5 | adaptive | current_trace | socket_serial | 1 | 1 | 0.055426 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 0.062428 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.05643 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | fixed1 | current_trace | socket_serial | 1 | 1 | 0.047663 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | fixed2 | current_trace | socket_serial | 1 | 1 | 0.048978 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | fixed3 | current_trace | socket_serial | 1 | 1 | 0.059054 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | text | current_trace | socket_serial | 1 | 1 | 0.084345 | 0 | 0 | 3 |

## Controlled mechanism ablations

| context_tokens | precision | entropy | mean_symbol_bits | mean_symbol_data_bytes | mean_complete_bytes | paired_nll_delta | mean_em | mean_encode_excluding_quantize_seconds | mean_decode_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 257 | layered2 | arithmetic | 4.2045 | 876792 | 1781056 | -0.061345 | 0.33333 | 0.0041851 | 0.003166 |
| 257 | layered2 | packed | 4.2045 | 1521440 | 1555919 | -0.061345 | 0.33333 | 0.079414 | 0.15625 |
| 257 | uniform16 | arithmetic | 4 | 806055 | 1349888 | -0.061364 | 0.33333 | 0.0038196 | 0.0031172 |
| 257 | uniform16 | packed | 4 | 1447424 | 1481999 | -0.061364 | 0.33333 | 0.099865 | 0.18419 |
| 257 | uniform32 | arithmetic | 5 | 1168643 | 2072896 | 0.026119 | 0.66667 | 0.0039105 | 0.0031848 |
| 257 | uniform32 | packed | 5 | 1809280 | 1842959 | 0.026119 | 0.66667 | 0.075029 | 0.14588 |

Each precision setting quantizes once; both entropy paths consume identical symbols/scales and share dequantization. Encoding timings exclude the separately recorded common quantization cost; complete sizes include CDF, stream lengths, scales, shapes and container overhead. uniform16 and uniform32 bracket layered2; differences in precision budget confound quality attribution. This small sweep shows size/quality tradeoffs, not isolated proof of layer allocation benefit.

## Statistical interpretation

2000 seeded source-cluster bootstrap draws; with few sources these are descriptive, not equivalence evidence

Independent documents/questions/facts, derived length/position/trace conditions and timing repeats are separately counted. Cluster intervals keep all conditions and repeats of a source together. Small-sample P95 is descriptive. A completed matrix establishes execution coverage, not paper-scale reproduction or statistical equivalence.

## Evidence

config.json, selection.json, metadata.json, preflight.json, cache_checks.json, ttft_checks.json, calibration.json, records.jsonl, events.jsonl, records.sqlite3, manifest.json, failures.jsonl (if any), dependencies.txt, hardware.txt, source/, artifacts.json, and run.log preserve the protocol, data, code, environment, order, outputs and checkpoints.

Resume: `bash experiments/run.sh --resume /home/aaa/CacheGen/results/tinyllama/minimal/v3-stable-acceptance-20261025`. Integrity: `python3 -m experiments.suite --verify /home/aaa/CacheGen/results/tinyllama/minimal/v3-stable-acceptance-20261025`.

## Figures

![Size and NLL](figures/quality.png)

![TTFT](figures/ttft.png)

![Adaptive](figures/adaptive.png)

![Paired pipeline speedup](figures/pipeline_speedup.png)

![Receive and restore timeline](figures/receive_restore_timeline.png)

![Mechanism size-quality sweep](figures/ablation.png)

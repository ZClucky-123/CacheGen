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

| context_tokens | target_bandwidth_gbps | method | mode | chunk_size | timing_observations | median_seconds | mean_actual_bandwidth_gbps | speedup_vs_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 0.1 | cachegen1 | chunk_pipeline | 128 | 2 | 0.095696 | 0.099321 | 0.46313 |
| 128 | 0.1 | cachegen1 | chunk_serial | 128 | 2 | 0.090769 | 0.09918 | 0.44647 |
| 128 | 0.1 | cachegen1 | whole | 128 | 2 | 0.094886 | 0.099006 | 0.20643 |
| 128 | 0.1 | cachegen2 | chunk_pipeline | 128 | 2 | 0.12016 | 0.099469 | 0.36884 |
| 128 | 0.1 | cachegen2 | chunk_serial | 128 | 2 | 0.12732 | 0.099245 | 0.31831 |
| 128 | 0.1 | cachegen2 | whole | 128 | 2 | 0.12846 | 0.099473 | 0.15247 |
| 128 | 0.1 | cachegen3 | chunk_pipeline | 128 | 2 | 0.13551 | 0.099438 | 0.32705 |
| 128 | 0.1 | cachegen3 | chunk_serial | 128 | 2 | 0.14379 | 0.099449 | 0.28184 |
| 128 | 0.1 | cachegen3 | whole | 128 | 2 | 0.14145 | 0.099179 | 0.13848 |
| 128 | 0.1 | raw | chunk_pipeline | 128 | 2 | 0.25106 | 0.09959 | 0.17653 |
| 128 | 0.1 | raw | chunk_serial | 128 | 2 | 0.2613 | 0.099389 | 0.15509 |
| 128 | 0.1 | raw | whole | 128 | 2 | 0.25707 | 0.099768 | 0.076192 |
| 128 | 0.1 | text | chunk_pipeline | 128 | 2 | 0.044319 | 0.020828 | 1 |
| 128 | 0.1 | text | chunk_serial | 128 | 2 | 0.040526 | 0.021361 | 1 |
| 128 | 0.1 | text | whole | 128 | 2 | 0.019587 | 0.024102 | 1 |
| 128 | 0.1 | uniform4 | chunk_pipeline | 128 | 2 | 0.090037 | 0.099057 | 0.49224 |
| 128 | 0.1 | uniform4 | chunk_serial | 128 | 2 | 0.088879 | 0.099135 | 0.45597 |
| 128 | 0.1 | uniform4 | whole | 128 | 2 | 0.089338 | 0.099451 | 0.21925 |
| 128 | 0.1 | uniform8 | chunk_pipeline | 128 | 2 | 0.14688 | 0.099439 | 0.30174 |
| 128 | 0.1 | uniform8 | chunk_serial | 128 | 2 | 0.14957 | 0.099573 | 0.27095 |
| 128 | 0.1 | uniform8 | whole | 128 | 2 | 0.14406 | 0.09948 | 0.13597 |
| 128 | 10 | cachegen1 | chunk_pipeline | 128 | 2 | 0.023779 | 4.5761 | 1.9423 |
| 128 | 10 | cachegen1 | chunk_serial | 128 | 2 | 0.026855 | 4.4267 | 1.1923 |
| 128 | 10 | cachegen1 | whole | 128 | 2 | 0.020951 | 4.9498 | 1.1813 |
| 128 | 10 | cachegen2 | chunk_pipeline | 128 | 2 | 0.020669 | 5.6348 | 2.2345 |
| 128 | 10 | cachegen2 | chunk_serial | 128 | 2 | 0.020513 | 5.635 | 1.561 |
| 128 | 10 | cachegen2 | whole | 128 | 2 | 0.035894 | 5.4594 | 0.68952 |
| 128 | 10 | cachegen3 | chunk_pipeline | 128 | 2 | 0.023996 | 6.4489 | 1.9246 |
| 128 | 10 | cachegen3 | chunk_serial | 128 | 2 | 0.025051 | 5.8376 | 1.2782 |
| 128 | 10 | cachegen3 | whole | 128 | 2 | 0.025752 | 3.7596 | 0.96106 |
| 128 | 10 | raw | chunk_pipeline | 128 | 2 | 0.031605 | 7.4068 | 1.4613 |
| 128 | 10 | raw | chunk_serial | 128 | 2 | 0.035935 | 5.2857 | 0.89105 |
| 128 | 10 | raw | whole | 128 | 2 | 0.042494 | 6.9728 | 0.58242 |
| 128 | 10 | text | chunk_pipeline | 128 | 2 | 0.046184 | 0.02811 | 1 |
| 128 | 10 | text | chunk_serial | 128 | 2 | 0.03202 | 0.042524 | 1 |
| 128 | 10 | text | whole | 128 | 2 | 0.02475 | 0.041672 | 1 |
| 128 | 10 | uniform4 | chunk_pipeline | 128 | 2 | 0.024712 | 4.2989 | 1.8689 |
| 128 | 10 | uniform4 | chunk_serial | 128 | 2 | 0.028202 | 4.7241 | 1.1354 |
| 128 | 10 | uniform4 | whole | 128 | 2 | 0.018873 | 4.8285 | 1.3114 |
| 128 | 10 | uniform8 | chunk_pipeline | 128 | 2 | 0.034779 | 5.76 | 1.3279 |
| 128 | 10 | uniform8 | chunk_serial | 128 | 2 | 0.023529 | 6.8976 | 1.3609 |
| 128 | 10 | uniform8 | whole | 128 | 2 | 0.025524 | 5.6791 | 0.96964 |
| 257 | 0.1 | cachegen1 | chunk_pipeline | 128 | 2 | 0.19125 | 0.09924 | 0.51365 |
| 257 | 0.1 | cachegen1 | chunk_serial | 128 | 2 | 0.19608 | 0.099229 | 0.41333 |
| 257 | 0.1 | cachegen1 | whole | 257 | 2 | 0.12497 | 0.09949 | 0.18444 |
| 257 | 0.1 | cachegen2 | chunk_pipeline | 128 | 2 | 0.29561 | 0.099472 | 0.33232 |
| 257 | 0.1 | cachegen2 | chunk_serial | 128 | 2 | 0.29478 | 0.099226 | 0.27494 |
| 257 | 0.1 | cachegen2 | whole | 257 | 2 | 0.17547 | 0.099651 | 0.13136 |
| 257 | 0.1 | cachegen3 | chunk_pipeline | 128 | 2 | 0.31858 | 0.099518 | 0.30836 |
| 257 | 0.1 | cachegen3 | chunk_serial | 128 | 2 | 0.31562 | 0.099426 | 0.25678 |
| 257 | 0.1 | cachegen3 | whole | 257 | 2 | 0.18678 | 0.099702 | 0.1234 |
| 257 | 0.1 | raw | chunk_pipeline | 128 | 2 | 0.49635 | 0.099386 | 0.19792 |
| 257 | 0.1 | raw | chunk_serial | 128 | 2 | 0.49489 | 0.099709 | 0.16376 |
| 257 | 0.1 | raw | whole | 257 | 2 | 0.50282 | 0.099572 | 0.045841 |
| 257 | 0.1 | text | chunk_pipeline | 128 | 2 | 0.098237 | 0.018651 | 1 |
| 257 | 0.1 | text | chunk_serial | 128 | 2 | 0.081045 | 0.017543 | 1 |
| 257 | 0.1 | text | whole | 257 | 2 | 0.02305 | 0.027857 | 1 |
| 257 | 0.1 | uniform4 | chunk_pipeline | 128 | 2 | 0.14931 | 0.098777 | 0.65792 |
| 257 | 0.1 | uniform4 | chunk_serial | 128 | 2 | 0.14868 | 0.099264 | 0.54509 |
| 257 | 0.1 | uniform4 | whole | 257 | 2 | 0.14369 | 0.09954 | 0.16041 |
| 257 | 0.1 | uniform8 | chunk_pipeline | 128 | 2 | 0.27152 | 0.099199 | 0.3618 |
| 257 | 0.1 | uniform8 | chunk_serial | 128 | 2 | 0.26253 | 0.099513 | 0.3087 |
| 257 | 0.1 | uniform8 | whole | 257 | 2 | 0.26427 | 0.09969 | 0.087219 |
| 257 | 10 | cachegen1 | chunk_pipeline | 128 | 2 | 0.026383 | 3.9309 | 2.7881 |
| 257 | 10 | cachegen1 | chunk_serial | 128 | 2 | 0.028205 | 5.0279 | 2.8783 |
| 257 | 10 | cachegen1 | whole | 257 | 2 | 0.031061 | 4.1793 | 0.87679 |
| 257 | 10 | cachegen2 | chunk_pipeline | 128 | 2 | 0.034558 | 4.1923 | 2.1286 |
| 257 | 10 | cachegen2 | chunk_serial | 128 | 2 | 0.03853 | 4.4027 | 2.107 |
| 257 | 10 | cachegen2 | whole | 257 | 2 | 0.035395 | 4.2788 | 0.76943 |
| 257 | 10 | cachegen3 | chunk_pipeline | 128 | 2 | 0.042915 | 4.2714 | 1.7141 |
| 257 | 10 | cachegen3 | chunk_serial | 128 | 2 | 0.030435 | 4.7225 | 2.6674 |
| 257 | 10 | cachegen3 | whole | 257 | 2 | 0.031273 | 4.4359 | 0.87084 |
| 257 | 10 | raw | chunk_pipeline | 128 | 2 | 0.029334 | 7.4767 | 2.5076 |
| 257 | 10 | raw | chunk_serial | 128 | 2 | 0.043368 | 4.7276 | 1.8719 |
| 257 | 10 | raw | whole | 257 | 2 | 0.04427 | 3.7998 | 0.61518 |
| 257 | 10 | text | chunk_pipeline | 128 | 2 | 0.073559 | 0.034049 | 1 |
| 257 | 10 | text | chunk_serial | 128 | 2 | 0.081183 | 0.029663 | 1 |
| 257 | 10 | text | whole | 257 | 2 | 0.027234 | 0.047334 | 1 |
| 257 | 10 | uniform4 | chunk_pipeline | 128 | 2 | 0.024245 | 3.914 | 3.034 |
| 257 | 10 | uniform4 | chunk_serial | 128 | 2 | 0.035765 | 4.8688 | 2.2699 |
| 257 | 10 | uniform4 | whole | 257 | 2 | 0.024089 | 5.0735 | 1.1306 |
| 257 | 10 | uniform8 | chunk_pipeline | 128 | 2 | 0.033741 | 3.7463 | 2.1801 |
| 257 | 10 | uniform8 | chunk_serial | 128 | 2 | 0.028442 | 6.3166 | 2.8543 |
| 257 | 10 | uniform8 | whole | 257 | 2 | 0.024794 | 6.6234 | 1.0984 |

## Paired pipeline comparison (same bytes)

| context_tokens | target_bandwidth_gbps | method | chunk_size | paired_observations | median_paired_speedup | mean_pipeline_overlap_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 0.1 | cachegen1 | 128 | 2 | 0.95108 | 0 |
| 128 | 0.1 | cachegen2 | 128 | 2 | 1.0599 | 0 |
| 128 | 0.1 | cachegen3 | 128 | 2 | 1.0623 | 0 |
| 128 | 0.1 | raw | 128 | 2 | 1.041 | 0 |
| 128 | 0.1 | text | 128 | 2 | 0.92097 | 0 |
| 128 | 0.1 | uniform4 | 128 | 2 | 0.99804 | 0 |
| 128 | 0.1 | uniform8 | 128 | 2 | 1.0184 | 0 |
| 128 | 10 | cachegen1 | 128 | 2 | 1.1712 | 0 |
| 128 | 10 | cachegen2 | 128 | 2 | 0.99137 | 0 |
| 128 | 10 | cachegen3 | 128 | 2 | 1.0666 | 0 |
| 128 | 10 | raw | 128 | 2 | 1.136 | 0 |
| 128 | 10 | text | 128 | 2 | 0.70839 | 0 |
| 128 | 10 | uniform4 | 128 | 2 | 1.1556 | 0 |
| 128 | 10 | uniform8 | 128 | 2 | 0.75539 | 0 |
| 257 | 0.1 | cachegen1 | 128 | 2 | 1.0254 | 0.0039709 |
| 257 | 0.1 | cachegen2 | 128 | 2 | 0.99715 | 0.0037153 |
| 257 | 0.1 | cachegen3 | 128 | 2 | 0.99211 | 0.003993 |
| 257 | 0.1 | raw | 128 | 2 | 0.99718 | 0.0012162 |
| 257 | 0.1 | text | 128 | 2 | 0.82398 | 0.00053387 |
| 257 | 0.1 | uniform4 | 128 | 2 | 0.99602 | 0.0017942 |
| 257 | 0.1 | uniform8 | 128 | 2 | 0.97026 | 0.0016321 |
| 257 | 10 | cachegen1 | 128 | 2 | 1.0731 | 0.002773 |
| 257 | 10 | cachegen2 | 128 | 2 | 1.1197 | 0.004903 |
| 257 | 10 | cachegen3 | 128 | 2 | 0.71759 | 0.0055657 |
| 257 | 10 | raw | 128 | 2 | 1.4784 | 0.00021946 |
| 257 | 10 | text | 128 | 2 | 1.1014 | 0.00028983 |
| 257 | 10 | uniform4 | 128 | 2 | 1.4693 | 0.00083668 |
| 257 | 10 | uniform8 | 128 | 2 | 0.87296 | 0.0012657 |

Pipeline speedup is serial/pipeline for each matched source, length, bandwidth, method, chunk size and repeat. Values below 1 mean slower pipeline. Chunk payload hashes and first tokens must match. Receiver capacity and send-permission waits, pacing, socket blocking, restoration, and overlap intervals are exported in ttft_timeline.csv / ttft_overlap.csv. All GPU restoration timestamps wait for CUDA completion.

## Adaptive configurations

| task | strategy | throughput_estimator | mode | fact_trace_conditions | timing_observations | median_ttft_seconds | slo_violation_rate | condition_mean_em | text_fallback_chunks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive-scale-0.01-slo-0.2 | adaptive | current_trace | socket_serial | 1 | 1 | 0.080091 | 0 | 0 | 3 |
| adaptive-scale-0.01-slo-0.2 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 1.9647 | 1 | 0 | 2 |
| adaptive-scale-0.01-slo-0.2 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.075778 | 0 | 0 | 3 |
| adaptive-scale-0.01-slo-0.2 | fixed1 | current_trace | socket_serial | 1 | 1 | 1.3908 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.2 | fixed2 | current_trace | socket_serial | 1 | 1 | 2.214 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.2 | fixed3 | current_trace | socket_serial | 1 | 1 | 2.4173 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.2 | text | current_trace | socket_serial | 1 | 1 | 0.083976 | 0 | 0 | 3 |
| adaptive-scale-0.01-slo-0.5 | adaptive | current_trace | socket_serial | 1 | 1 | 0.44674 | 0 | 0 | 1 |
| adaptive-scale-0.01-slo-0.5 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 1.9519 | 1 | 0 | 2 |
| adaptive-scale-0.01-slo-0.5 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.33485 | 0 | 0 | 1 |
| adaptive-scale-0.01-slo-0.5 | fixed1 | current_trace | socket_serial | 1 | 1 | 1.3877 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.5 | fixed2 | current_trace | socket_serial | 1 | 1 | 2.2145 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.5 | fixed3 | current_trace | socket_serial | 1 | 1 | 2.4294 | 1 | 0 | 0 |
| adaptive-scale-0.01-slo-0.5 | text | current_trace | socket_serial | 1 | 1 | 0.074944 | 0 | 0 | 3 |
| adaptive-scale-1-slo-0.2 | adaptive | current_trace | socket_serial | 1 | 1 | 0.055815 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 0.052359 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.054255 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | fixed1 | current_trace | socket_serial | 1 | 1 | 0.056075 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | fixed2 | current_trace | socket_serial | 1 | 1 | 0.058292 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | fixed3 | current_trace | socket_serial | 1 | 1 | 0.073337 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.2 | text | current_trace | socket_serial | 1 | 1 | 0.10943 | 0 | 0 | 3 |
| adaptive-scale-1-slo-0.5 | adaptive | current_trace | socket_serial | 1 | 1 | 0.057785 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 0.074473 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.070814 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | fixed1 | current_trace | socket_serial | 1 | 1 | 0.042773 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | fixed2 | current_trace | socket_serial | 1 | 1 | 0.072633 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | fixed3 | current_trace | socket_serial | 1 | 1 | 0.067831 | 0 | 0 | 0 |
| adaptive-scale-1-slo-0.5 | text | current_trace | socket_serial | 1 | 1 | 0.096231 | 0 | 0 | 3 |

## Controlled mechanism ablations

| context_tokens | precision | entropy | mean_symbol_bits | mean_symbol_data_bytes | mean_complete_bytes | paired_nll_delta | mean_em | mean_encode_excluding_quantize_seconds | mean_decode_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 257 | layered2 | arithmetic | 4.2045 | 872028 | 1730888 | -0.061345 | 0.33333 | 0.0034063 | 0.0028107 |
| 257 | layered2 | packed | 4.2045 | 1521440 | 1555919 | -0.061345 | 0.33333 | 0.098431 | 0.18951 |
| 257 | uniform16 | arithmetic | 4 | 801515 | 1299848 | -0.061364 | 0.33333 | 0.0028102 | 0.0025851 |
| 257 | uniform16 | packed | 4 | 1447424 | 1481999 | -0.061364 | 0.33333 | 0.097748 | 0.18326 |
| 257 | uniform32 | arithmetic | 5 | 1162696 | 2021512 | 0.026119 | 0.66667 | 0.0042532 | 0.0031641 |
| 257 | uniform32 | packed | 5 | 1809280 | 1842959 | 0.026119 | 0.66667 | 0.10036 | 0.20506 |

Each precision setting quantizes once; both entropy paths consume identical symbols/scales and share dequantization. Encoding timings exclude the separately recorded common quantization cost; complete sizes include CDF, stream lengths, scales, shapes and container overhead. uniform16 and uniform32 bracket layered2; differences in precision budget confound quality attribution. This small sweep shows size/quality tradeoffs, not isolated proof of layer allocation benefit.

## Statistical interpretation

2000 seeded source-cluster bootstrap draws; with few sources these are descriptive, not equivalence evidence

Independent documents/questions/facts, derived length/position/trace conditions and timing repeats are separately counted. Cluster intervals keep all conditions and repeats of a source together. Small-sample P95 is descriptive. A completed matrix establishes execution coverage, not paper-scale reproduction or statistical equivalence.

## Evidence

config.json, selection.json, metadata.json, preflight.json, cache_checks.json, ttft_checks.json, calibration.json, records.jsonl, events.jsonl, records.sqlite3, manifest.json, failures.jsonl (if any), dependencies.txt, hardware.txt, source/, artifacts.json, and run.log preserve the protocol, data, code, environment, order, outputs and checkpoints.

Resume: `bash experiments/run.sh --resume /home/aaa/CacheGen/results/tinyllama/minimal/v3-final-03`. Integrity: `python3 -m experiments.suite --verify /home/aaa/CacheGen/results/tinyllama/minimal/v3-final-03`.

## Figures

![Size and NLL](figures/quality.png)

![TTFT](figures/ttft.png)

![Adaptive](figures/adaptive.png)

![Paired pipeline speedup](figures/pipeline_speedup.png)

![Receive and restore timeline](figures/receive_restore_timeline.png)

![Mechanism size-quality sweep](figures/ablation.png)

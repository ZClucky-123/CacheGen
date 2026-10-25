# mistral / small / cachegen-shared-v3

Model: mistral-community/Mistral-7B-v0.2 @ 80cf49b18de6354fd8a6d1e81c177dd830e05eea; weights: fastchat8.

All 7 core tasks and 469 expected observations verified. TriviaQA is supplementary inside the quality task.

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

Execution settings (also checkpointed and fingerprinted): {"explicit_causal_mask": true, "fp16_reduced_precision_reduction": false, "linear_tile_tokens": 256, "sdpa_backend": "efficient"}. Results with automatic kernels and stable kernels are separate batches; execution scheduling can affect absolute timing.

Adaptive mode: socket_serial. Protocol v3 executes measured, serial paced socket transfers; it rejects trace_estimate for history feedback. SLO applies to TTFT; full answer latency is separate.

History throughput variant adaptive_history uses a fixed first-block estimate of 1.0 Gbps, then the previous completed full-message bytes * 8 / effective receive interval. The interval starts after receiver capacity/buffer readiness, includes pre-send pacing and socket overhead, and excludes queue-capacity and previous GPU waits. The history policy cannot access current or future trace values; those are executor inputs and posthoc diagnostics. Both adaptive variants share candidate costs, remaining budget, first-token reserve and fallback rules. This is a block-level trace simulation, not the complete paper scheduler.

Each adaptive cached chunk comes from a full original-context prefill, so later-layer KV retains original-prefix dependence. Text fallback actually recomputes on the assembled prior KV with explicit positions. This is the repository heuristic, not the paper scheduler. Adaptive execution is serial. TTFT separately compares whole, chunk serial and bounded chunk pipeline; anchor/delta coding is absent.

Byte ratios divide sums of actual serialized FP16 containers by sums of transmitted containers, including scales, CDFs, shapes and other metadata. Tensor-only ratios are separately named. Local socket target bandwidth is not measured throughput; see calibration.json and ttft.csv.

## Paired quality by length

| context_tokens | method | raw_serialized_over_method | paired_nll_delta | raw_em_correct | em_retention | acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | raw | 1 | 0 | 3 | 1 | pass |
| 1024 | uniform4 | 3.9843 | 0.0047287 | 3 | 1 | pass |
| 1024 | uniform8 | 1.9961 | 7.4744e-05 | 3 | 1 | pass |
| 1024 | cachegen1 | 6.8362 | 0.042479 | 3 | 1 | fail |
| 1024 | cachegen2 | 5.2494 | 0.0039594 | 3 | 1 | pass |
| 1024 | cachegen3 | 4.1263 | -0.0048168 | 3 | 1 | pass |
| 2048 | raw | 1 | 0 | 3 | 1 | pass |
| 2048 | uniform4 | 3.9844 | -0.0074892 | 3 | 1 | pass |
| 2048 | uniform8 | 1.9961 | 0.0013958 | 3 | 1 | pass |
| 2048 | cachegen1 | 7.2552 | -0.0034454 | 3 | 1 | pass |
| 2048 | cachegen2 | 5.7356 | -0.0028452 | 3 | 1 | pass |
| 2048 | cachegen3 | 4.4178 | -0.0070562 | 3 | 1 | pass |
| 4096 | raw | 1 | 0 | 3 | 1 | pass |
| 4096 | uniform4 | 3.9844 | -0.01235 | 3 | 1 | pass |
| 4096 | uniform8 | 1.9961 | -0.0019653 | 3 | 1 | pass |
| 4096 | cachegen1 | 7.4809 | -0.00088759 | 3 | 1 | pass |
| 4096 | cachegen2 | 6.0094 | -0.012674 | 3 | 1 | pass |
| 4096 | cachegen3 | 4.5754 | -0.0075193 | 3 | 1 | pass |

## Separate QA datasets

| dataset | method | independent_questions_or_facts | conditions | mean_em | mean_f1 |
| --- | --- | --- | --- | --- | --- |
| controlled | raw | 1 | 9 | 1 | 1 |
| controlled | uniform4 | 1 | 9 | 1 | 1 |
| controlled | uniform8 | 1 | 9 | 1 | 1 |
| controlled | cachegen1 | 1 | 9 | 1 | 1 |
| controlled | cachegen2 | 1 | 9 | 1 | 1 |
| controlled | cachegen3 | 1 | 9 | 1 | 1 |
| triviaqa | raw | 2 | 2 | 1 | 1 |
| triviaqa | uniform4 | 2 | 2 | 1 | 1 |
| triviaqa | uniform8 | 2 | 2 | 1 | 1 |
| triviaqa | cachegen1 | 2 | 2 | 1 | 1 |
| triviaqa | cachegen2 | 2 | 2 | 1 | 1 |
| triviaqa | cachegen3 | 2 | 2 | 1 | 1 |

## TTFT

| context_tokens | target_bandwidth_gbps | method | mode | chunk_size | timing_observations | median_seconds | mean_actual_bandwidth_gbps | speedup_vs_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 10 | cachegen1 | chunk_pipeline | 256 | 5 | 0.13557 | 5.8395 | 2.7884 |
| 1024 | 10 | cachegen1 | chunk_serial | 256 | 5 | 0.14855 | 6.7753 | 2.5354 |
| 1024 | 10 | cachegen1 | whole | 1024 | 5 | 0.1406 | 6.5261 | 1.5999 |
| 1024 | 10 | cachegen2 | chunk_pipeline | 256 | 5 | 0.16915 | 5.398 | 2.2349 |
| 1024 | 10 | cachegen2 | chunk_serial | 256 | 5 | 0.17334 | 7.1712 | 2.1727 |
| 1024 | 10 | cachegen2 | whole | 1024 | 5 | 0.1535 | 5.9288 | 1.4654 |
| 1024 | 10 | cachegen3 | chunk_pipeline | 256 | 5 | 0.17262 | 6.2061 | 2.1899 |
| 1024 | 10 | cachegen3 | chunk_serial | 256 | 5 | 0.19447 | 7.4811 | 1.9367 |
| 1024 | 10 | cachegen3 | whole | 1024 | 5 | 0.17361 | 6.2912 | 1.2957 |
| 1024 | 10 | raw | chunk_pipeline | 256 | 5 | 0.52125 | 5.0399 | 0.72523 |
| 1024 | 10 | raw | chunk_serial | 256 | 5 | 0.56406 | 5.2513 | 0.66771 |
| 1024 | 10 | raw | whole | 1024 | 5 | 0.58541 | 5.3896 | 0.38425 |
| 1024 | 10 | text | chunk_pipeline | 256 | 5 | 0.37802 | 0.072989 | 1 |
| 1024 | 10 | text | chunk_serial | 256 | 5 | 0.37663 | 0.075067 | 1 |
| 1024 | 10 | text | whole | 1024 | 5 | 0.22495 | 0.22861 | 1 |
| 1024 | 10 | uniform4 | chunk_pipeline | 256 | 5 | 0.15536 | 6.2125 | 2.4332 |
| 1024 | 10 | uniform4 | chunk_serial | 256 | 5 | 0.14699 | 6.9605 | 2.5622 |
| 1024 | 10 | uniform4 | whole | 1024 | 5 | 0.19361 | 4.5984 | 1.1618 |
| 1024 | 10 | uniform8 | chunk_pipeline | 256 | 5 | 0.21664 | 7.1146 | 1.7449 |
| 1024 | 10 | uniform8 | chunk_serial | 256 | 5 | 0.21557 | 7.4212 | 1.7471 |
| 1024 | 10 | uniform8 | whole | 1024 | 5 | 0.34429 | 5.0972 | 0.65336 |
| 2048 | 10 | cachegen1 | chunk_pipeline | 256 | 5 | 0.19793 | 5.6241 | 3.5231 |
| 2048 | 10 | cachegen1 | chunk_serial | 256 | 5 | 0.21954 | 7.3813 | 3.214 |
| 2048 | 10 | cachegen1 | whole | 2048 | 5 | 0.22371 | 5.1185 | 1.788 |
| 2048 | 10 | cachegen2 | chunk_pipeline | 256 | 5 | 0.24785 | 5.6462 | 2.8134 |
| 2048 | 10 | cachegen2 | chunk_serial | 256 | 5 | 0.26525 | 7.3786 | 2.6601 |
| 2048 | 10 | cachegen2 | whole | 2048 | 5 | 0.26153 | 4.9136 | 1.5294 |
| 2048 | 10 | cachegen3 | chunk_pipeline | 256 | 5 | 0.23961 | 6.6671 | 2.9102 |
| 2048 | 10 | cachegen3 | chunk_serial | 256 | 5 | 0.29613 | 7.3443 | 2.3827 |
| 2048 | 10 | cachegen3 | whole | 2048 | 5 | 0.32841 | 5.1297 | 1.218 |
| 2048 | 10 | raw | chunk_pipeline | 256 | 5 | 0.9883 | 4.8656 | 0.70557 |
| 2048 | 10 | raw | chunk_serial | 256 | 5 | 0.98262 | 5.364 | 0.71807 |
| 2048 | 10 | raw | whole | 2048 | 5 | 1.1534 | 5.4248 | 0.34679 |
| 2048 | 10 | text | chunk_pipeline | 256 | 5 | 0.69732 | 0.079248 | 1 |
| 2048 | 10 | text | chunk_serial | 256 | 5 | 0.70559 | 0.074241 | 1 |
| 2048 | 10 | text | whole | 2048 | 5 | 0.39999 | 0.47537 | 1 |
| 2048 | 10 | uniform4 | chunk_pipeline | 256 | 5 | 0.22476 | 6.5645 | 3.1026 |
| 2048 | 10 | uniform4 | chunk_serial | 256 | 5 | 0.23622 | 7.3245 | 2.9871 |
| 2048 | 10 | uniform4 | whole | 2048 | 5 | 0.34953 | 5.2855 | 1.1444 |
| 2048 | 10 | uniform8 | chunk_pipeline | 256 | 5 | 0.38942 | 7.2755 | 1.7906 |
| 2048 | 10 | uniform8 | chunk_serial | 256 | 5 | 0.36709 | 7.738 | 1.9221 |
| 2048 | 10 | uniform8 | whole | 2048 | 5 | 0.60351 | 5.3428 | 0.66277 |
| 4096 | 10 | cachegen1 | chunk_pipeline | 256 | 5 | 0.33609 | 5.3972 | 4.1561 |
| 4096 | 10 | cachegen1 | chunk_serial | 256 | 5 | 0.38661 | 6.4263 | 3.6327 |
| 4096 | 10 | cachegen1 | whole | 4096 | 5 | 0.36949 | 5.3586 | 2.2104 |
| 4096 | 10 | cachegen2 | chunk_pipeline | 256 | 5 | 0.44091 | 5.1116 | 3.168 |
| 4096 | 10 | cachegen2 | chunk_serial | 256 | 5 | 0.50814 | 6.508 | 2.7639 |
| 4096 | 10 | cachegen2 | whole | 4096 | 5 | 0.44928 | 5.3349 | 1.8179 |
| 4096 | 10 | cachegen3 | chunk_pipeline | 256 | 5 | 0.4941 | 5.3749 | 2.827 |
| 4096 | 10 | cachegen3 | chunk_serial | 256 | 5 | 0.57608 | 6.59 | 2.4379 |
| 4096 | 10 | cachegen3 | whole | 4096 | 5 | 0.54103 | 5.3694 | 1.5096 |
| 4096 | 10 | raw | chunk_pipeline | 256 | 5 | 2.112 | 4.6832 | 0.66138 |
| 4096 | 10 | raw | chunk_serial | 256 | 5 | 2.1736 | 5.2505 | 0.64614 |
| 4096 | 10 | raw | whole | 4096 | 5 | 2.0179 | 5.5068 | 0.40474 |
| 4096 | 10 | text | chunk_pipeline | 256 | 5 | 1.3968 | 0.078904 | 1 |
| 4096 | 10 | text | chunk_serial | 256 | 5 | 1.4044 | 0.074455 | 1 |
| 4096 | 10 | text | whole | 4096 | 5 | 0.81674 | 0.60518 | 1 |
| 4096 | 10 | uniform4 | chunk_pipeline | 256 | 5 | 0.39856 | 5.1788 | 3.5046 |
| 4096 | 10 | uniform4 | chunk_serial | 256 | 5 | 0.42299 | 4.9066 | 3.3203 |
| 4096 | 10 | uniform4 | whole | 4096 | 5 | 0.593 | 5.0641 | 1.3773 |
| 4096 | 10 | uniform8 | chunk_pipeline | 256 | 5 | 0.80522 | 4.2844 | 1.7347 |
| 4096 | 10 | uniform8 | chunk_serial | 256 | 5 | 0.73919 | 5.2948 | 1.9 |
| 4096 | 10 | uniform8 | whole | 4096 | 5 | 1.1208 | 5.4424 | 0.72872 |

## Paired pipeline comparison (same bytes)

| context_tokens | target_bandwidth_gbps | method | chunk_size | paired_observations | median_paired_speedup | mean_pipeline_overlap_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 10 | cachegen1 | 256 | 5 | 1.1296 | 0.019536 |
| 1024 | 10 | cachegen2 | 256 | 5 | 1.0806 | 0.027667 |
| 1024 | 10 | cachegen3 | 256 | 5 | 1.1359 | 0.024603 |
| 1024 | 10 | raw | 256 | 5 | 1.0726 | 0.024294 |
| 1024 | 10 | text | 256 | 5 | 0.99684 | 0.00051973 |
| 1024 | 10 | uniform4 | 256 | 5 | 0.92929 | 0.0054761 |
| 1024 | 10 | uniform8 | 256 | 5 | 1.0334 | 0.0084162 |
| 2048 | 10 | cachegen1 | 256 | 5 | 1.1173 | 0.04775 |
| 2048 | 10 | cachegen2 | 256 | 5 | 1.0702 | 0.056803 |
| 2048 | 10 | cachegen3 | 256 | 5 | 1.2315 | 0.06056 |
| 2048 | 10 | raw | 256 | 5 | 0.98538 | 0.0538 |
| 2048 | 10 | text | 256 | 5 | 1.0119 | 0.0010867 |
| 2048 | 10 | uniform4 | 256 | 5 | 1.0435 | 0.012841 |
| 2048 | 10 | uniform8 | 256 | 5 | 0.97423 | 0.011533 |
| 4096 | 10 | cachegen1 | 256 | 5 | 1.1306 | 0.097421 |
| 4096 | 10 | cachegen2 | 256 | 5 | 1.1511 | 0.11617 |
| 4096 | 10 | cachegen3 | 256 | 5 | 1.1462 | 0.12572 |
| 4096 | 10 | raw | 256 | 5 | 1.0357 | 0.16047 |
| 4096 | 10 | text | 256 | 5 | 1.005 | 0.0023875 |
| 4096 | 10 | uniform4 | 256 | 5 | 1.0613 | 0.027238 |
| 4096 | 10 | uniform8 | 256 | 5 | 1.0205 | 0.11944 |

Pipeline speedup is serial/pipeline for each matched source, length, bandwidth, method, chunk size and repeat. Values below 1 mean slower pipeline. Chunk payload hashes and first tokens must match. Receiver capacity and send-permission waits, pacing, socket blocking, restoration, and overlap intervals are exported in ttft_timeline.csv / ttft_overlap.csv. All GPU restoration timestamps wait for CUDA completion.

## Adaptive configurations

| task | strategy | throughput_estimator | mode | fact_trace_conditions | timing_observations | median_ttft_seconds | slo_violation_rate | condition_mean_em | text_fallback_chunks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive-scale-0.01-slo-0.2 | adaptive | current_trace | socket_serial | 1 | 1 | 0.3853 | 1 | 1 | 4 |
| adaptive-scale-0.01-slo-0.2 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 9.1722 | 1 | 1 | 3 |
| adaptive-scale-0.01-slo-0.2 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.38207 | 1 | 1 | 4 |
| adaptive-scale-0.01-slo-0.2 | fixed1 | current_trace | socket_serial | 1 | 1 | 19.143 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.2 | fixed2 | current_trace | socket_serial | 1 | 1 | 28.521 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.2 | fixed3 | current_trace | socket_serial | 1 | 1 | 33.42 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.2 | text | current_trace | socket_serial | 1 | 1 | 0.38482 | 1 | 1 | 4 |
| adaptive-scale-0.01-slo-0.5 | adaptive | current_trace | socket_serial | 1 | 1 | 0.387 | 0 | 1 | 4 |
| adaptive-scale-0.01-slo-0.5 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 15.716 | 1 | 1 | 3 |
| adaptive-scale-0.01-slo-0.5 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.3803 | 0 | 1 | 4 |
| adaptive-scale-0.01-slo-0.5 | fixed1 | current_trace | socket_serial | 1 | 1 | 19.146 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.5 | fixed2 | current_trace | socket_serial | 1 | 1 | 28.522 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.5 | fixed3 | current_trace | socket_serial | 1 | 1 | 33.423 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.5 | text | current_trace | socket_serial | 1 | 1 | 0.38413 | 0 | 1 | 4 |
| adaptive-scale-1-slo-0.2 | adaptive | current_trace | socket_serial | 1 | 1 | 0.28836 | 1 | 1 | 2 |
| adaptive-scale-1-slo-0.2 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 0.4165 | 1 | 1 | 3 |
| adaptive-scale-1-slo-0.2 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.33766 | 1 | 1 | 2 |
| adaptive-scale-1-slo-0.2 | fixed1 | current_trace | socket_serial | 1 | 1 | 0.31881 | 1 | 1 | 0 |
| adaptive-scale-1-slo-0.2 | fixed2 | current_trace | socket_serial | 1 | 1 | 0.43167 | 1 | 1 | 0 |
| adaptive-scale-1-slo-0.2 | fixed3 | current_trace | socket_serial | 1 | 1 | 0.49143 | 1 | 1 | 0 |
| adaptive-scale-1-slo-0.2 | text | current_trace | socket_serial | 1 | 1 | 0.38461 | 1 | 1 | 4 |
| adaptive-scale-1-slo-0.5 | adaptive | current_trace | socket_serial | 1 | 1 | 0.4195 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 0.50056 | 1 | 1 | 3 |
| adaptive-scale-1-slo-0.5 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.39954 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | fixed1 | current_trace | socket_serial | 1 | 1 | 0.31524 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | fixed2 | current_trace | socket_serial | 1 | 1 | 0.44163 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | fixed3 | current_trace | socket_serial | 1 | 1 | 0.48093 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | text | current_trace | socket_serial | 1 | 1 | 0.37835 | 0 | 1 | 4 |

## Controlled mechanism ablations

| context_tokens | precision | entropy | mean_symbol_bits | mean_symbol_data_bytes | mean_complete_bytes | paired_nll_delta | mean_em | mean_encode_excluding_quantize_seconds | mean_decode_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | layered2 | arithmetic | 4.1875 | 18212735 | 25559566 | 0.02447 | 1 | 0.029427 | 0.017096 |
| 1024 | layered2 | packed | 4.1875 | 35127296 | 35275859 | 0.02447 | 1 | 1.6603 | 3.8274 |
| 1024 | uniform16 | arithmetic | 4 | 16747593 | 21997198 | 0.034237 | 1 | 0.037679 | 0.016128 |
| 1024 | uniform16 | packed | 4 | 33554432 | 33702995 | 0.034237 | 1 | 1.6619 | 3.7763 |
| 1024 | uniform32 | arithmetic | 5 | 25171279 | 32518094 | -0.005704 | 1 | 0.056581 | 0.019074 |
| 1024 | uniform32 | packed | 5 | 41943040 | 42091603 | -0.005704 | 1 | 1.6885 | 4.3032 |

Each precision setting quantizes once; both entropy paths consume identical symbols/scales and share dequantization. Encoding timings exclude the separately recorded common quantization cost; complete sizes include CDF, stream lengths, scales, shapes and container overhead. uniform16 and uniform32 bracket layered2; differences in precision budget confound quality attribution. This small sweep shows size/quality tradeoffs, not isolated proof of layer allocation benefit.

## Statistical interpretation

2000 seeded source-cluster bootstrap draws; with few sources these are descriptive, not equivalence evidence

Independent documents/questions/facts, derived length/position/trace conditions and timing repeats are separately counted. Cluster intervals keep all conditions and repeats of a source together. Small-sample P95 is descriptive. A completed matrix establishes execution coverage, not paper-scale reproduction or statistical equivalence.

## Evidence

config.json, selection.json, metadata.json, preflight.json, cache_checks.json, ttft_checks.json, calibration.json, records.jsonl, events.jsonl, records.sqlite3, manifest.json, failures.jsonl (if any), dependencies.txt, hardware.txt, source/, artifacts.json, and run.log preserve the protocol, data, code, environment, order, outputs and checkpoints.

Resume: `bash experiments/run.sh --resume /home/aaa/CacheGen/results/mistral/small/v3-larger-validation-03`. Integrity: `python3 -m experiments.suite --verify /home/aaa/CacheGen/results/mistral/small/v3-larger-validation-03`.

## Figures

![Size and NLL](figures/quality.png)

![TTFT](figures/ttft.png)

![Adaptive](figures/adaptive.png)

![Paired pipeline speedup](figures/pipeline_speedup.png)

![Receive and restore timeline](figures/receive_restore_timeline.png)

![Mechanism size-quality sweep](figures/ablation.png)

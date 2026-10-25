# mistral / minimal / cachegen-shared-v3

Model: mistral-community/Mistral-7B-v0.2 @ 80cf49b18de6354fd8a6d1e81c177dd830e05eea; weights: fastchat8.

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
| 128 | raw | 1 | 0 | 3 | 1 | pass |
| 128 | uniform4 | 3.9833 | 0.049066 | 3 | 1 | fail |
| 128 | uniform8 | 1.9959 | -0.0015075 | 3 | 1 | pass |
| 128 | cachegen1 | 3.9229 | 0.022863 | 3 | 1 | fail |
| 128 | cachegen2 | 2.4536 | 0.070045 | 3 | 1 | fail |
| 128 | cachegen3 | 2.1868 | -0.00090361 | 3 | 1 | pass |
| 257 | raw | 1 | 0 | 3 | 1 | pass |
| 257 | uniform4 | 3.9839 | -0.013791 | 3 | 1 | pass |
| 257 | uniform8 | 1.996 | 0.0025992 | 3 | 1 | pass |
| 257 | cachegen1 | 5.2719 | -0.049421 | 3 | 1 | pass |
| 257 | cachegen2 | 3.5711 | -0.037849 | 3 | 1 | pass |
| 257 | cachegen3 | 3.0232 | 0.046722 | 3 | 1 | fail |

## Separate QA datasets

| dataset | method | independent_questions_or_facts | conditions | mean_em | mean_f1 |
| --- | --- | --- | --- | --- | --- |
| controlled | raw | 1 | 6 | 1 | 1 |
| controlled | uniform4 | 1 | 6 | 1 | 1 |
| controlled | uniform8 | 1 | 6 | 1 | 1 |
| controlled | cachegen1 | 1 | 6 | 1 | 1 |
| controlled | cachegen2 | 1 | 6 | 1 | 1 |
| controlled | cachegen3 | 1 | 6 | 1 | 1 |
| triviaqa | raw | 1 | 1 | 0 | 0.66667 |
| triviaqa | uniform4 | 1 | 1 | 0 | 0.66667 |
| triviaqa | uniform8 | 1 | 1 | 0 | 0.66667 |
| triviaqa | cachegen1 | 1 | 1 | 0 | 0.66667 |
| triviaqa | cachegen2 | 1 | 1 | 0 | 0.66667 |
| triviaqa | cachegen3 | 1 | 1 | 0 | 0.66667 |

## TTFT

| context_tokens | target_bandwidth_gbps | method | mode | chunk_size | timing_observations | median_seconds | mean_actual_bandwidth_gbps | speedup_vs_text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 0.1 | cachegen1 | chunk_pipeline | 128 | 2 | 0.41023 | 0.099733 | 0.28417 |
| 128 | 0.1 | cachegen1 | chunk_serial | 128 | 2 | 0.40755 | 0.099568 | 0.2748 |
| 128 | 0.1 | cachegen1 | whole | 128 | 2 | 0.40272 | 0.099736 | 0.1661 |
| 128 | 0.1 | cachegen2 | chunk_pipeline | 128 | 2 | 0.62212 | 0.09977 | 0.18739 |
| 128 | 0.1 | cachegen2 | chunk_serial | 128 | 2 | 0.61362 | 0.099723 | 0.18252 |
| 128 | 0.1 | cachegen2 | whole | 128 | 2 | 0.61294 | 0.099783 | 0.10913 |
| 128 | 0.1 | cachegen3 | chunk_pipeline | 128 | 2 | 0.68527 | 0.099427 | 0.17012 |
| 128 | 0.1 | cachegen3 | chunk_serial | 128 | 2 | 0.6799 | 0.09976 | 0.16473 |
| 128 | 0.1 | cachegen3 | whole | 128 | 2 | 0.6882 | 0.099621 | 0.097196 |
| 128 | 0.1 | raw | chunk_pipeline | 128 | 2 | 1.4263 | 0.09969 | 0.081732 |
| 128 | 0.1 | raw | chunk_serial | 128 | 2 | 1.4451 | 0.099115 | 0.077501 |
| 128 | 0.1 | raw | whole | 128 | 2 | 1.424 | 0.099701 | 0.046974 |
| 128 | 0.1 | text | chunk_pipeline | 128 | 2 | 0.11658 | 0.017823 | 1 |
| 128 | 0.1 | text | chunk_serial | 128 | 2 | 0.112 | 0.022277 | 1 |
| 128 | 0.1 | text | whole | 128 | 2 | 0.06689 | 0.020842 | 1 |
| 128 | 0.1 | uniform4 | chunk_pipeline | 128 | 2 | 0.39897 | 0.09971 | 0.29219 |
| 128 | 0.1 | uniform4 | chunk_serial | 128 | 2 | 0.40566 | 0.099705 | 0.27608 |
| 128 | 0.1 | uniform4 | whole | 128 | 2 | 0.40349 | 0.099382 | 0.16578 |
| 128 | 0.1 | uniform8 | chunk_pipeline | 128 | 2 | 0.73745 | 0.099758 | 0.15808 |
| 128 | 0.1 | uniform8 | chunk_serial | 128 | 2 | 0.73745 | 0.099646 | 0.15187 |
| 128 | 0.1 | uniform8 | whole | 128 | 2 | 0.74159 | 0.099812 | 0.090198 |
| 128 | 10 | cachegen1 | chunk_pipeline | 128 | 2 | 0.071451 | 5.5161 | 1.5602 |
| 128 | 10 | cachegen1 | chunk_serial | 128 | 2 | 0.065162 | 7.73 | 1.7225 |
| 128 | 10 | cachegen1 | whole | 128 | 2 | 0.064068 | 7.9721 | 1.0144 |
| 128 | 10 | cachegen2 | chunk_pipeline | 128 | 2 | 0.07176 | 8.2061 | 1.5535 |
| 128 | 10 | cachegen2 | chunk_serial | 128 | 2 | 0.069783 | 7.1137 | 1.6084 |
| 128 | 10 | cachegen2 | whole | 128 | 2 | 0.072996 | 6.9002 | 0.89036 |
| 128 | 10 | cachegen3 | chunk_pipeline | 128 | 2 | 0.070647 | 7.7393 | 1.5779 |
| 128 | 10 | cachegen3 | chunk_serial | 128 | 2 | 0.077661 | 6.1769 | 1.4453 |
| 128 | 10 | cachegen3 | whole | 128 | 2 | 0.070781 | 7.5588 | 0.91822 |
| 128 | 10 | raw | chunk_pipeline | 128 | 2 | 0.1087 | 6.7327 | 1.0255 |
| 128 | 10 | raw | chunk_serial | 128 | 2 | 0.10065 | 7.4987 | 1.1151 |
| 128 | 10 | raw | whole | 128 | 2 | 0.10362 | 7.5816 | 0.62724 |
| 128 | 10 | text | chunk_pipeline | 128 | 2 | 0.11148 | 0.043307 | 1 |
| 128 | 10 | text | chunk_serial | 128 | 2 | 0.11224 | 0.030723 | 1 |
| 128 | 10 | text | whole | 128 | 2 | 0.064993 | 0.03147 | 1 |
| 128 | 10 | uniform4 | chunk_pipeline | 128 | 2 | 0.06599 | 6.6862 | 1.6893 |
| 128 | 10 | uniform4 | chunk_serial | 128 | 2 | 0.066752 | 7.4969 | 1.6815 |
| 128 | 10 | uniform4 | whole | 128 | 2 | 0.066516 | 6.8233 | 0.97709 |
| 128 | 10 | uniform8 | chunk_pipeline | 128 | 2 | 0.076882 | 6.675 | 1.45 |
| 128 | 10 | uniform8 | chunk_serial | 128 | 2 | 0.071951 | 7.8452 | 1.56 |
| 128 | 10 | uniform8 | whole | 128 | 2 | 0.0733 | 6.8046 | 0.88667 |
| 257 | 0.1 | cachegen1 | chunk_pipeline | 128 | 2 | 0.95506 | 0.099706 | 0.2388 |
| 257 | 0.1 | cachegen1 | chunk_serial | 128 | 2 | 0.9637 | 0.099656 | 0.23457 |
| 257 | 0.1 | cachegen1 | whole | 257 | 2 | 0.58723 | 0.099544 | 0.1469 |
| 257 | 0.1 | cachegen2 | chunk_pipeline | 128 | 2 | 1.5346 | 0.099765 | 0.14861 |
| 257 | 0.1 | cachegen2 | chunk_serial | 128 | 2 | 1.543 | 0.099782 | 0.1465 |
| 257 | 0.1 | cachegen2 | whole | 257 | 2 | 0.82817 | 0.099701 | 0.10416 |
| 257 | 0.1 | cachegen3 | chunk_pipeline | 128 | 2 | 1.6794 | 0.099569 | 0.1358 |
| 257 | 0.1 | cachegen3 | chunk_serial | 128 | 2 | 1.6929 | 0.099669 | 0.13353 |
| 257 | 0.1 | cachegen3 | whole | 257 | 2 | 0.96406 | 0.099697 | 0.089479 |
| 257 | 0.1 | raw | chunk_pipeline | 128 | 2 | 2.8313 | 0.099087 | 0.080552 |
| 257 | 0.1 | raw | chunk_serial | 128 | 2 | 2.8349 | 0.099055 | 0.079738 |
| 257 | 0.1 | raw | whole | 257 | 2 | 2.8521 | 0.099049 | 0.030246 |
| 257 | 0.1 | text | chunk_pipeline | 128 | 2 | 0.22806 | 0.013793 | 1 |
| 257 | 0.1 | text | chunk_serial | 128 | 2 | 0.22605 | 0.020756 | 1 |
| 257 | 0.1 | text | whole | 257 | 2 | 0.086263 | 0.02334 | 1 |
| 257 | 0.1 | uniform4 | chunk_pipeline | 128 | 2 | 0.74544 | 0.099487 | 0.30595 |
| 257 | 0.1 | uniform4 | chunk_serial | 128 | 2 | 0.749 | 0.099572 | 0.3018 |
| 257 | 0.1 | uniform4 | whole | 257 | 2 | 0.75243 | 0.099654 | 0.11465 |
| 257 | 0.1 | uniform8 | chunk_pipeline | 128 | 2 | 1.4349 | 0.09945 | 0.15895 |
| 257 | 0.1 | uniform8 | chunk_serial | 128 | 2 | 1.4325 | 0.09973 | 0.1578 |
| 257 | 0.1 | uniform8 | whole | 257 | 2 | 1.4427 | 0.099116 | 0.059794 |
| 257 | 10 | cachegen1 | chunk_pipeline | 128 | 2 | 0.085149 | 5.0186 | 2.5347 |
| 257 | 10 | cachegen1 | chunk_serial | 128 | 2 | 0.090676 | 6.2585 | 2.4176 |
| 257 | 10 | cachegen1 | whole | 257 | 2 | 0.071814 | 7.2507 | 1.1305 |
| 257 | 10 | cachegen2 | chunk_pipeline | 128 | 2 | 0.096324 | 5.6162 | 2.2407 |
| 257 | 10 | cachegen2 | chunk_serial | 128 | 2 | 0.10203 | 6.9291 | 2.1486 |
| 257 | 10 | cachegen2 | whole | 257 | 2 | 0.087441 | 6.4294 | 0.92843 |
| 257 | 10 | cachegen3 | chunk_pipeline | 128 | 2 | 0.105 | 5.8317 | 2.0555 |
| 257 | 10 | cachegen3 | chunk_serial | 128 | 2 | 0.10648 | 6.7768 | 2.0587 |
| 257 | 10 | cachegen3 | whole | 257 | 2 | 0.10384 | 6.0528 | 0.78182 |
| 257 | 10 | raw | chunk_pipeline | 128 | 2 | 0.15645 | 5.6955 | 1.3796 |
| 257 | 10 | raw | chunk_serial | 128 | 2 | 0.14894 | 5.423 | 1.4719 |
| 257 | 10 | raw | whole | 257 | 2 | 0.20454 | 5.0049 | 0.39691 |
| 257 | 10 | text | chunk_pipeline | 128 | 2 | 0.21583 | 0.034453 | 1 |
| 257 | 10 | text | chunk_serial | 128 | 2 | 0.21922 | 0.029678 | 1 |
| 257 | 10 | text | whole | 257 | 2 | 0.081183 | 0.038937 | 1 |
| 257 | 10 | uniform4 | chunk_pipeline | 128 | 2 | 0.086741 | 4.4954 | 2.4882 |
| 257 | 10 | uniform4 | chunk_serial | 128 | 2 | 0.08142 | 5.9948 | 2.6924 |
| 257 | 10 | uniform4 | whole | 257 | 2 | 0.086993 | 4.9148 | 0.93321 |
| 257 | 10 | uniform8 | chunk_pipeline | 128 | 2 | 0.086485 | 7.375 | 2.4956 |
| 257 | 10 | uniform8 | chunk_serial | 128 | 2 | 0.099167 | 7.5842 | 2.2106 |
| 257 | 10 | uniform8 | whole | 257 | 2 | 0.10533 | 5.4772 | 0.77076 |

## Paired pipeline comparison (same bytes)

| context_tokens | target_bandwidth_gbps | method | chunk_size | paired_observations | median_paired_speedup | mean_pipeline_overlap_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 0.1 | cachegen1 | 128 | 2 | 0.99359 | 0 |
| 128 | 0.1 | cachegen2 | 128 | 2 | 0.98634 | 0 |
| 128 | 0.1 | cachegen3 | 128 | 2 | 0.99222 | 0 |
| 128 | 0.1 | raw | 128 | 2 | 1.0132 | 0 |
| 128 | 0.1 | text | 128 | 2 | 0.96075 | 0 |
| 128 | 0.1 | uniform4 | 128 | 2 | 1.0168 | 0 |
| 128 | 0.1 | uniform8 | 128 | 2 | 1 | 0 |
| 128 | 10 | cachegen1 | 128 | 2 | 0.92329 | 0 |
| 128 | 10 | cachegen2 | 128 | 2 | 0.9783 | 0 |
| 128 | 10 | cachegen3 | 128 | 2 | 1.0989 | 0 |
| 128 | 10 | raw | 128 | 2 | 0.92613 | 0 |
| 128 | 10 | text | 128 | 2 | 1.0069 | 0 |
| 128 | 10 | uniform4 | 128 | 2 | 1.012 | 0 |
| 128 | 10 | uniform8 | 128 | 2 | 0.94125 | 0 |
| 257 | 0.1 | cachegen1 | 128 | 2 | 1.0091 | 0.0080194 |
| 257 | 0.1 | cachegen2 | 128 | 2 | 1.0055 | 0.0085832 |
| 257 | 0.1 | cachegen3 | 128 | 2 | 1.008 | 0.008704 |
| 257 | 0.1 | raw | 128 | 2 | 1.0013 | 0.0059572 |
| 257 | 0.1 | text | 128 | 2 | 0.99139 | 0.00085652 |
| 257 | 0.1 | uniform4 | 128 | 2 | 1.0048 | 0.0020321 |
| 257 | 0.1 | uniform8 | 128 | 2 | 0.99839 | 0.010075 |
| 257 | 10 | cachegen1 | 128 | 2 | 1.0722 | 0.0075383 |
| 257 | 10 | cachegen2 | 128 | 2 | 1.0599 | 0.0090331 |
| 257 | 10 | cachegen3 | 128 | 2 | 1.0122 | 0.011681 |
| 257 | 10 | raw | 128 | 2 | 0.95446 | 0.0058001 |
| 257 | 10 | text | 128 | 2 | 1.0157 | 0.0003682 |
| 257 | 10 | uniform4 | 128 | 2 | 0.94138 | 0.0059418 |
| 257 | 10 | uniform8 | 128 | 2 | 1.1476 | 0.0014668 |

Pipeline speedup is serial/pipeline for each matched source, length, bandwidth, method, chunk size and repeat. Values below 1 mean slower pipeline. Chunk payload hashes and first tokens must match. Receiver capacity and send-permission waits, pacing, socket blocking, restoration, and overlap intervals are exported in ttft_timeline.csv / ttft_overlap.csv. All GPU restoration timestamps wait for CUDA completion.

## Adaptive configurations

| task | strategy | throughput_estimator | mode | fact_trace_conditions | timing_observations | median_ttft_seconds | slo_violation_rate | condition_mean_em | text_fallback_chunks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive-scale-0.01-slo-0.2 | adaptive | current_trace | socket_serial | 1 | 1 | 0.22115 | 1 | 1 | 3 |
| adaptive-scale-0.01-slo-0.2 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 6.0319 | 1 | 1 | 2 |
| adaptive-scale-0.01-slo-0.2 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.22316 | 1 | 1 | 3 |
| adaptive-scale-0.01-slo-0.2 | fixed1 | current_trace | socket_serial | 1 | 1 | 7.4959 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.2 | fixed2 | current_trace | socket_serial | 1 | 1 | 12.264 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.2 | fixed3 | current_trace | socket_serial | 1 | 1 | 13.548 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.2 | text | current_trace | socket_serial | 1 | 1 | 0.22833 | 1 | 1 | 3 |
| adaptive-scale-0.01-slo-0.5 | adaptive | current_trace | socket_serial | 1 | 1 | 0.2332 | 0 | 1 | 3 |
| adaptive-scale-0.01-slo-0.5 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 10.804 | 1 | 1 | 2 |
| adaptive-scale-0.01-slo-0.5 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.23005 | 0 | 1 | 3 |
| adaptive-scale-0.01-slo-0.5 | fixed1 | current_trace | socket_serial | 1 | 1 | 7.4918 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.5 | fixed2 | current_trace | socket_serial | 1 | 1 | 12.257 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.5 | fixed3 | current_trace | socket_serial | 1 | 1 | 13.539 | 1 | 1 | 0 |
| adaptive-scale-0.01-slo-0.5 | text | current_trace | socket_serial | 1 | 1 | 0.22304 | 0 | 1 | 3 |
| adaptive-scale-1-slo-0.2 | adaptive | current_trace | socket_serial | 1 | 1 | 0.17298 | 0 | 1 | 1 |
| adaptive-scale-1-slo-0.2 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 0.23692 | 1 | 1 | 2 |
| adaptive-scale-1-slo-0.2 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.19281 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.2 | fixed1 | current_trace | socket_serial | 1 | 1 | 0.152 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.2 | fixed2 | current_trace | socket_serial | 1 | 1 | 0.22718 | 1 | 1 | 0 |
| adaptive-scale-1-slo-0.2 | fixed3 | current_trace | socket_serial | 1 | 1 | 0.23422 | 1 | 1 | 0 |
| adaptive-scale-1-slo-0.2 | text | current_trace | socket_serial | 1 | 1 | 0.21836 | 1 | 1 | 3 |
| adaptive-scale-1-slo-0.5 | adaptive | current_trace | socket_serial | 1 | 1 | 0.22311 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | adaptive_history | previous_completed_message | socket_serial | 1 | 1 | 0.23432 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | adaptive_static | current_trace | socket_serial | 1 | 1 | 0.24188 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | fixed1 | current_trace | socket_serial | 1 | 1 | 0.1534 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | fixed2 | current_trace | socket_serial | 1 | 1 | 0.21739 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | fixed3 | current_trace | socket_serial | 1 | 1 | 0.22682 | 0 | 1 | 0 |
| adaptive-scale-1-slo-0.5 | text | current_trace | socket_serial | 1 | 1 | 0.22438 | 0 | 1 | 3 |

## Controlled mechanism ablations

| context_tokens | precision | entropy | mean_symbol_bits | mean_symbol_data_bytes | mean_complete_bytes | paired_nll_delta | mean_em | mean_encode_excluding_quantize_seconds | mean_decode_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 257 | layered2 | arithmetic | 4.1875 | 4548152 | 9433288 | -0.037849 | 1 | 0.010563 | 0.0070383 |
| 257 | layered2 | packed | 4.1875 | 8816128 | 8866515 | -0.037849 | 1 | 0.40412 | 0.77873 |
| 257 | uniform16 | arithmetic | 4 | 4189014 | 6977032 | -0.017038 | 1 | 0.0098589 | 0.0055632 |
| 257 | uniform16 | packed | 4 | 8421376 | 8471763 | -0.017038 | 1 | 0.55582 | 1.0493 |
| 257 | uniform32 | arithmetic | 5 | 6257516 | 11142664 | 0.046722 | 1 | 0.020293 | 0.0083544 |
| 257 | uniform32 | packed | 5 | 10526720 | 10577107 | 0.046722 | 1 | 0.55797 | 1.1967 |

Each precision setting quantizes once; both entropy paths consume identical symbols/scales and share dequantization. Encoding timings exclude the separately recorded common quantization cost; complete sizes include CDF, stream lengths, scales, shapes and container overhead. uniform16 and uniform32 bracket layered2; differences in precision budget confound quality attribution. This small sweep shows size/quality tradeoffs, not isolated proof of layer allocation benefit.

## Statistical interpretation

2000 seeded source-cluster bootstrap draws; with few sources these are descriptive, not equivalence evidence

Independent documents/questions/facts, derived length/position/trace conditions and timing repeats are separately counted. Cluster intervals keep all conditions and repeats of a source together. Small-sample P95 is descriptive. A completed matrix establishes execution coverage, not paper-scale reproduction or statistical equivalence.

## Evidence

config.json, selection.json, metadata.json, preflight.json, cache_checks.json, ttft_checks.json, calibration.json, records.jsonl, events.jsonl, records.sqlite3, manifest.json, failures.jsonl (if any), dependencies.txt, hardware.txt, source/, artifacts.json, and run.log preserve the protocol, data, code, environment, order, outputs and checkpoints.

Resume: `bash experiments/run.sh --resume /home/aaa/CacheGen/results/mistral/minimal/v3-final-03`. Integrity: `python3 -m experiments.suite --verify /home/aaa/CacheGen/results/mistral/minimal/v3-final-03`.

## Figures

![Size and NLL](figures/quality.png)

![TTFT](figures/ttft.png)

![Adaptive](figures/adaptive.png)

![Paired pipeline speedup](figures/pipeline_speedup.png)

![Receive and restore timeline](figures/receive_restore_timeline.png)

![Mechanism size-quality sweep](figures/ablation.png)

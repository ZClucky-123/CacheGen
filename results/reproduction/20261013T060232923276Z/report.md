# Small-scale CacheGen experiment report

**Run purpose:** `formal-small-scale`.

Completed jobs: 6/6. Source: [manifest.json](manifest.json).

This evaluates the repository codec on TinyLlama, using its exploratory quantization schedules. The codec lacks the paper's anchor/delta stage; single-host transfer and offline adaptation do not reproduce the paper's distributed streaming system.

## Paired size and quality

Descriptive tolerances at each context length: mean paired NLL increase ≤ 0.02; QA retention on questions answered correctly by raw KV ≥ 90%. A pass requires both natural continuation contexts and at least one raw-correct question. This small sample does not establish statistical noninferiority. Missing method answers count as failures.

NLL scores teacher-forced continuation tokens 2..N, omitting the first held-out token; QA uses synthetic inserted facts. Compression ratio is paired raw byte sum / method byte sum. These are quality proxies, not the paper's benchmark scores. Details: [quality.csv](quality.csv).

| Job | Tokens | Method | Contexts | Size ratio | ΔNLL | Raw-correct QA | QA retention | Descriptive pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quality | 512 | cachegen1 | 20 | 5.934 | 0.06533 | 30 | 1 | no |
| quality | 512 | cachegen2 | 20 | 4.35 | 0.03032 | 30 | 1 | no |
| quality | 512 | cachegen3 | 20 | 3.557 | -0.005347 | 30 | 1 | yes |
| quality | 512 | raw | 20 | 1 | 0 | 30 | 1 | yes |
| quality | 512 | uniform4 | 20 | 3.937 | 0.02833 | 30 | 1 | no |
| quality | 512 | uniform8 | 20 | 1.984 | 0.0003859 | 30 | 1 | yes |
| quality | 1024 | cachegen1 | 20 | 6.393 | 0.02959 | 30 | 0.9667 | no |
| quality | 1024 | cachegen2 | 20 | 4.937 | 0.01954 | 30 | 1 | yes |
| quality | 1024 | cachegen3 | 20 | 3.93 | 0.002848 | 30 | 1 | yes |
| quality | 1024 | raw | 20 | 1 | 0 | 30 | 1 | yes |
| quality | 1024 | uniform4 | 20 | 3.938 | 0.01893 | 30 | 1 | yes |
| quality | 1024 | uniform8 | 20 | 1.984 | 0.0009102 | 30 | 1 | yes |
| quality | 1536 | cachegen1 | 20 | 6.595 | 0.07086 | 30 | 1 | no |
| quality | 1536 | cachegen2 | 20 | 5.191 | 0.02966 | 30 | 1 | no |
| quality | 1536 | cachegen3 | 20 | 4.087 | 0.00254 | 30 | 1 | yes |
| quality | 1536 | raw | 20 | 1 | 0 | 30 | 1 | yes |
| quality | 1536 | uniform4 | 20 | 3.938 | 0.03142 | 30 | 1 | no |
| quality | 1536 | uniform8 | 20 | 1.984 | -0.0002245 | 30 | 1 | yes |

## Transfer and time to first token

Timings use a paced local socket and exclude offline encoding. Warmups are excluded. Contexts are unique inputs; observations include timing repeats. Median and P95 describe all timing observations and are not confidence bounds across independent contexts. Speedups are ratios of medians against text or uniform8. A quality label refers to the separate size/quality sample at the same length; each baseline comparison also requires that baseline to pass the descriptive tolerance. unmatched rows cannot support a claim of speedup at preserved quality. Details: [ttft.csv](ttft.csv).

| Job | Tokens | Gbps | Method | Contexts | Observations | Median s | P95 s | Text speedup | Quality label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ttft | 512 | 0.1 | cachegen2 | 3 | 30 | 0.2502 | 0.2638 | 0.2852 | unmatched |
| ttft | 512 | 0.1 | cachegen3 | 3 | 30 | 0.3002 | 0.3163 | 0.2377 | meets_descriptive_threshold |
| ttft | 512 | 0.1 | text | 3 | 30 | 0.07136 | 0.1389 | 1 | meets_descriptive_threshold |
| ttft | 512 | 0.1 | uniform8 | 3 | 30 | 0.4997 | 0.5115 | 0.1428 | meets_descriptive_threshold |
| ttft | 512 | 1 | cachegen2 | 3 | 30 | 0.05913 | 0.07381 | 0.8086 | unmatched |
| ttft | 512 | 1 | cachegen3 | 3 | 30 | 0.06369 | 0.08394 | 0.7507 | meets_descriptive_threshold |
| ttft | 512 | 1 | text | 3 | 30 | 0.04781 | 0.05476 | 1 | meets_descriptive_threshold |
| ttft | 512 | 1 | uniform8 | 3 | 30 | 0.08079 | 0.09281 | 0.5918 | meets_descriptive_threshold |
| ttft | 512 | 10 | cachegen2 | 3 | 30 | 0.04436 | 0.06107 | 1.035 | unmatched |
| ttft | 512 | 10 | cachegen3 | 3 | 30 | 0.04744 | 0.0599 | 0.9673 | meets_descriptive_threshold |
| ttft | 512 | 10 | text | 3 | 30 | 0.04589 | 0.052 | 1 | meets_descriptive_threshold |
| ttft | 512 | 10 | uniform8 | 3 | 30 | 0.05089 | 0.0649 | 0.9018 | meets_descriptive_threshold |
| ttft | 1024 | 0.1 | cachegen2 | 3 | 30 | 0.4258 | 0.4384 | 0.39 | meets_descriptive_threshold |
| ttft | 1024 | 0.1 | cachegen3 | 3 | 30 | 0.5215 | 0.6473 | 0.3184 | meets_descriptive_threshold |
| ttft | 1024 | 0.1 | text | 3 | 30 | 0.1661 | 0.2715 | 1 | meets_descriptive_threshold |
| ttft | 1024 | 0.1 | uniform8 | 3 | 30 | 0.972 | 1.042 | 0.1708 | meets_descriptive_threshold |
| ttft | 1024 | 1 | cachegen2 | 3 | 30 | 0.08011 | 0.094 | 1.093 | meets_descriptive_threshold |
| ttft | 1024 | 1 | cachegen3 | 3 | 30 | 0.09128 | 0.1071 | 0.9592 | meets_descriptive_threshold |
| ttft | 1024 | 1 | text | 3 | 30 | 0.08756 | 0.09647 | 1 | meets_descriptive_threshold |
| ttft | 1024 | 1 | uniform8 | 3 | 30 | 0.1298 | 0.1471 | 0.6747 | meets_descriptive_threshold |
| ttft | 1024 | 10 | cachegen2 | 3 | 30 | 0.05219 | 0.07221 | 1.708 | meets_descriptive_threshold |
| ttft | 1024 | 10 | cachegen3 | 3 | 30 | 0.05557 | 0.07113 | 1.604 | meets_descriptive_threshold |
| ttft | 1024 | 10 | text | 3 | 30 | 0.08916 | 0.0994 | 1 | meets_descriptive_threshold |
| ttft | 1024 | 10 | uniform8 | 3 | 30 | 0.06266 | 0.07067 | 1.423 | meets_descriptive_threshold |
| ttft | 1536 | 0.1 | cachegen2 | 3 | 30 | 0.5882 | 0.7525 | 0.3124 | unmatched |
| ttft | 1536 | 0.1 | cachegen3 | 3 | 30 | 0.7357 | 0.913 | 0.2498 | meets_descriptive_threshold |
| ttft | 1536 | 0.1 | text | 3 | 30 | 0.1838 | 0.3058 | 1 | meets_descriptive_threshold |
| ttft | 1536 | 0.1 | uniform8 | 3 | 30 | 1.443 | 1.587 | 0.1274 | meets_descriptive_threshold |
| ttft | 1536 | 1 | cachegen2 | 3 | 30 | 0.09978 | 0.1147 | 1.349 | unmatched |
| ttft | 1536 | 1 | cachegen3 | 3 | 30 | 0.1147 | 0.1257 | 1.173 | meets_descriptive_threshold |
| ttft | 1536 | 1 | text | 3 | 30 | 0.1346 | 0.1435 | 1 | meets_descriptive_threshold |
| ttft | 1536 | 1 | uniform8 | 3 | 30 | 0.1801 | 0.1973 | 0.7473 | meets_descriptive_threshold |
| ttft | 1536 | 10 | cachegen2 | 3 | 30 | 0.06437 | 0.07962 | 2.11 | unmatched |
| ttft | 1536 | 10 | cachegen3 | 3 | 30 | 0.06596 | 0.07937 | 2.059 | meets_descriptive_threshold |
| ttft | 1536 | 10 | text | 3 | 30 | 0.1358 | 0.1438 | 1 | meets_descriptive_threshold |
| ttft | 1536 | 10 | uniform8 | 3 | 30 | 0.0752 | 0.09176 | 1.806 | meets_descriptive_threshold |

## Bandwidth adaptation

TTFT is trace-estimated transfer plus measured local restoration and first-token work; there is no live network. Text fallback replays original token IDs; UTF-8 length is a size proxy without receive-side retokenization. QA pairs deduplicate timing repeats by question and trace; traces on the same question are not independent QA samples. Violation rates retain all timing observations. adaptive_static is the former equal-per-chunk-budget diagnostic baseline; adaptive uses remaining budget and a first-token reserve, forecasting remaining costs from current bandwidth and offline per-chunk calibration. This causal heuristic is not the paper's online feedback scheduler. Details: [adaptive.csv](adaptive.csv).

| Job | Strategy | Contexts | Observations | SLO miss rate | Median estimated s | Question/trace pairs | QA retention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive-scale-1-slo-0.2 | adaptive | 5 | 75 | 0 | 0.056 | 25 | 1 |
| adaptive-scale-1-slo-0.2 | adaptive_static | 5 | 75 | 0 | 0.05692 | 25 | 1 |
| adaptive-scale-1-slo-0.2 | fixed1 | 5 | 75 | 0 | 0.05167 | 25 | 1 |
| adaptive-scale-1-slo-0.2 | fixed2 | 5 | 75 | 0 | 0.0542 | 25 | 1 |
| adaptive-scale-1-slo-0.2 | fixed3 | 5 | 75 | 0 | 0.05562 | 25 | 1 |
| adaptive-scale-1-slo-0.2 | text | 5 | 75 | 0.02667 | 0.1616 | 25 | 1 |
| adaptive-scale-1-slo-0.5 | adaptive | 5 | 75 | 0 | 0.06562 | 25 | 1 |
| adaptive-scale-1-slo-0.5 | adaptive_static | 5 | 75 | 0 | 0.06652 | 25 | 1 |
| adaptive-scale-1-slo-0.5 | fixed1 | 5 | 75 | 0 | 0.05814 | 25 | 1 |
| adaptive-scale-1-slo-0.5 | fixed2 | 5 | 75 | 0 | 0.0643 | 25 | 1 |
| adaptive-scale-1-slo-0.5 | fixed3 | 5 | 75 | 0 | 0.0649 | 25 | 1 |
| adaptive-scale-1-slo-0.5 | text | 5 | 75 | 0 | 0.1931 | 25 | 1 |
| adaptive-scale-0.01-slo-0.2 | adaptive | 5 | 75 | 0.3333 | 0.1909 | 25 | 1 |
| adaptive-scale-0.01-slo-0.2 | adaptive_static | 5 | 75 | 0.36 | 0.1914 | 25 | 1 |
| adaptive-scale-0.01-slo-0.2 | fixed1 | 5 | 75 | 1 | 0.7914 | 25 | 1 |
| adaptive-scale-0.01-slo-0.2 | fixed2 | 5 | 75 | 1 | 1.149 | 25 | 1 |
| adaptive-scale-0.01-slo-0.2 | fixed3 | 5 | 75 | 1 | 1.329 | 25 | 1 |
| adaptive-scale-0.01-slo-0.2 | text | 5 | 75 | 0.3333 | 0.1924 | 25 | 1 |
| adaptive-scale-0.01-slo-0.5 | adaptive | 5 | 75 | 0.04 | 0.4138 | 25 | 1 |
| adaptive-scale-0.01-slo-0.5 | adaptive_static | 5 | 75 | 0 | 0.267 | 25 | 1 |
| adaptive-scale-0.01-slo-0.5 | fixed1 | 5 | 75 | 1 | 0.7913 | 25 | 1 |
| adaptive-scale-0.01-slo-0.5 | fixed2 | 5 | 75 | 1 | 1.146 | 25 | 1 |
| adaptive-scale-0.01-slo-0.5 | fixed3 | 5 | 75 | 1 | 1.333 | 25 | 1 |
| adaptive-scale-0.01-slo-0.5 | text | 5 | 75 | 0 | 0.1937 | 25 | 1 |

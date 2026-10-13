# Small-scale CacheGen experiment report

**Run purpose:** `pipeline-validation-only`.

**Pipeline validation only: this batch cannot support formal experiment conclusions.**

Completed jobs: 4/4. Source: [manifest.json](manifest.json).

This evaluates the repository codec on TinyLlama, using its exploratory quantization schedules. The codec lacks the paper's anchor/delta stage; single-host transfer and offline adaptation do not reproduce the paper's distributed streaming system.

## Paired size and quality

Descriptive tolerances at each context length: mean paired NLL increase ≤ 0.02; QA retention on questions answered correctly by raw KV ≥ 90%. A pass requires both natural continuation contexts and at least one raw-correct question. This small sample does not establish statistical noninferiority. Missing method answers count as failures.

NLL scores teacher-forced continuation tokens 2..N, omitting the first held-out token; QA uses synthetic inserted facts. Compression ratio is paired raw byte sum / method byte sum. These are quality proxies, not the paper's benchmark scores. Details: [quality.csv](quality.csv).

| Job | Tokens | Method | Contexts | Size ratio | ΔNLL | Raw-correct QA | QA retention | Descriptive pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quality | 512 | cachegen1 | 1 | 5.931 | 0.1721 | 3 | 1 | no |
| quality | 512 | cachegen2 | 1 | 4.349 | 0.004071 | 3 | 1 | yes |
| quality | 512 | cachegen3 | 1 | 3.556 | -0.004393 | 3 | 1 | yes |
| quality | 512 | raw | 1 | 1 | 0 | 3 | 1 | yes |
| quality | 512 | uniform4 | 1 | 3.937 | 0.01481 | 3 | 1 | yes |
| quality | 512 | uniform8 | 1 | 1.984 | 0.000779 | 3 | 1 | yes |

## Transfer and time to first token

Timings use a paced local socket and exclude offline encoding. Warmups are excluded. Contexts are unique inputs; observations include timing repeats. Median and P95 describe all timing observations and are not confidence bounds across independent contexts. Speedups are ratios of medians against text or uniform8. A quality label refers to the separate size/quality sample at the same length; each baseline comparison also requires that baseline to pass the descriptive tolerance. unmatched rows cannot support a claim of speedup at preserved quality. Details: [ttft.csv](ttft.csv).

| Job | Tokens | Gbps | Method | Contexts | Observations | Median s | P95 s | Text speedup | Quality label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ttft | 512 | 1 | cachegen2 | 1 | 2 | 0.05782 | 0.06086 | 0.7973 | meets_descriptive_threshold |
| ttft | 512 | 1 | cachegen3 | 1 | 2 | 0.06375 | 0.07121 | 0.7231 | meets_descriptive_threshold |
| ttft | 512 | 1 | text | 1 | 2 | 0.0461 | 0.04869 | 1 | meets_descriptive_threshold |
| ttft | 512 | 1 | uniform8 | 1 | 2 | 0.07816 | 0.07818 | 0.5898 | meets_descriptive_threshold |

## Bandwidth adaptation

TTFT is trace-estimated transfer plus measured local restoration and first-token work; there is no live network. Text fallback replays original token IDs; UTF-8 length is a size proxy without receive-side retokenization. QA pairs deduplicate timing repeats by question and trace; traces on the same question are not independent QA samples. Violation rates retain all timing observations. adaptive_static is the former equal-per-chunk-budget diagnostic baseline; adaptive uses remaining budget and a first-token reserve, forecasting remaining costs from current bandwidth and offline per-chunk calibration. This causal heuristic is not the paper's online feedback scheduler. Details: [adaptive.csv](adaptive.csv).

| Job | Strategy | Contexts | Observations | SLO miss rate | Median estimated s | Question/trace pairs | QA retention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive-scale-1-slo-0.2 | adaptive | 1 | 2 | 0 | 0.05909 | 1 | 1 |
| adaptive-scale-1-slo-0.2 | adaptive_static | 1 | 2 | 0 | 0.05874 | 1 | 1 |
| adaptive-scale-1-slo-0.2 | fixed1 | 1 | 2 | 0 | 0.04528 | 1 | 1 |
| adaptive-scale-1-slo-0.2 | fixed2 | 1 | 2 | 0 | 0.05402 | 1 | 1 |
| adaptive-scale-1-slo-0.2 | fixed3 | 1 | 2 | 0 | 0.0589 | 1 | 1 |
| adaptive-scale-1-slo-0.2 | text | 1 | 2 | 0 | 0.07813 | 1 | 1 |
| adaptive-scale-0.01-slo-0.2 | adaptive | 1 | 2 | 0 | 0.09063 | 1 | 1 |
| adaptive-scale-0.01-slo-0.2 | adaptive_static | 1 | 2 | 0 | 0.08652 | 1 | 1 |
| adaptive-scale-0.01-slo-0.2 | fixed1 | 1 | 2 | 1 | 1.733 | 1 | 1 |
| adaptive-scale-0.01-slo-0.2 | fixed2 | 1 | 2 | 1 | 2.55 | 1 | 1 |
| adaptive-scale-0.01-slo-0.2 | fixed3 | 1 | 2 | 1 | 2.987 | 1 | 1 |
| adaptive-scale-0.01-slo-0.2 | text | 1 | 2 | 0 | 0.09132 | 1 | 1 |

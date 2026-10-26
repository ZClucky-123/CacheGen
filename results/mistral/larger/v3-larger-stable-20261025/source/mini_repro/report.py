"""Generate paired quality and timing reports without GPU dependencies."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def summarize_quality(data, max_nll_delta=0.02, max_qa_drop=0.10):
    """Apply descriptive tolerances to paired measurements at each length."""
    if not math.isfinite(max_nll_delta) or max_nll_delta < 0:
        raise ValueError("NLL tolerance must be finite and nonnegative")
    if not math.isfinite(max_qa_drop) or not 0 <= max_qa_drop <= 1:
        raise ValueError("QA drop tolerance must be between zero and one")
    natural, questions, methods = defaultdict(dict), defaultdict(dict), defaultdict(set)
    for field, destination in (("continuation", natural), ("qa", questions)):
        for index, case in enumerate(data.get(field, [])):
            tokens = case["context_tokens"]
            rows = {row["method"]: row for row in case["methods"]}
            if "raw" not in rows:
                raise ValueError(f"quality case {case.get('case_id', index)} lacks paired raw baseline")
            destination[tokens][case.get("case_id", str(index))] = rows
            methods[tokens].update(rows)
    result = []
    for tokens in sorted(methods):
        raw_correct = [rows for rows in questions[tokens].values() if rows["raw"]["correct"]]
        for method in sorted(methods[tokens]):
            pairs = [(rows["raw"], rows[method]) for rows in natural[tokens].values() if method in rows]
            nll_delta = mean(row["nll"] - raw["nll"] for raw, row in pairs) if pairs else None
            byte_sum = sum(row["bytes"] for _, row in pairs)
            retention = (sum(bool(rows.get(method, {}).get("correct", False)) for rows in raw_correct)
                         / len(raw_correct)) if raw_correct else None
            complete_pairs = len(pairs) == len(natural[tokens])
            passed = bool(pairs and raw_correct and complete_pairs and nll_delta <= max_nll_delta
                          and retention >= 1 - max_qa_drop)
            result.append({
                "context_tokens": tokens, "method": method, "continuation_contexts": len(pairs),
                "mean_nll": mean(row["nll"] for _, row in pairs) if pairs else None,
                "mean_nll_delta": nll_delta, "mean_bytes": byte_sum / len(pairs) if pairs else None,
                "compression_ratio": sum(raw["bytes"] for raw, _ in pairs) / byte_sum if byte_sum else None,
                "qa_contexts": len(questions[tokens]), "raw_correct_contexts": len(raw_correct),
                "qa_retention_on_raw_correct": retention, "meets_descriptive_threshold": passed,
            })
    return result


def _p95(values):
    ordered = sorted(values)
    rank = (len(ordered) - 1) * 0.95
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _ttft_rows(data, quality):
    grouped = defaultdict(list)
    for index, case in enumerate(data.get("cases", [])):
        for row in case["measurements"]:
            if row.get("phase", "measure") != "measure":
                continue
            key = (case["context_tokens"], row["bandwidth_gbps"], row["method"])
            grouped[key].append((case.get("case_id", str(index)), row))
    medians = {key: median(row["ttft_wall_seconds"] for _, row in entries)
               for key, entries in grouped.items()}
    result = []
    for (tokens, bandwidth, method), entries in sorted(grouped.items()):
        values = [row["ttft_wall_seconds"] for _, row in entries]
        midpoint = median(values)
        quality_method = "raw" if method == "text" else method
        qualified = quality.get((tokens, quality_method), False)
        text = medians.get((tokens, bandwidth, "text"))
        uniform = medians.get((tokens, bandwidth, "uniform8"))
        result.append({
            "context_tokens": tokens, "bandwidth_gbps": bandwidth, "method": method,
            "contexts": len({case_id for case_id, _ in entries}), "observations": len(values),
            "median_seconds": midpoint, "p95_seconds": _p95(values),
            "quality_comparison": "meets_descriptive_threshold" if qualified else "unmatched",
            "speedup_vs_text": text / midpoint if text is not None and midpoint > 0 else None,
            "text_speedup_quality_comparison": "meets_descriptive_threshold"
            if qualified and quality.get((tokens, "raw"), False) else "unmatched",
            "speedup_vs_uniform8": uniform / midpoint if uniform is not None and midpoint > 0 else None,
            "uniform8_speedup_quality_comparison": "meets_descriptive_threshold"
            if qualified and quality.get((tokens, "uniform8"), False) else "unmatched",
            "mean_transfer_seconds": mean(row["transfer_seconds"] for _, row in entries)
            if all("transfer_seconds" in row for _, row in entries) else None,
            "mean_effective_bandwidth_gbps":
            mean(row.get("realized_bandwidth_gbps", row.get("effective_bandwidth_gbps")) for _, row in entries)
            if all("realized_bandwidth_gbps" in row or "effective_bandwidth_gbps" in row
                   for _, row in entries) else None,
        })
    return result


def _adaptive_rows(data):
    grouped = defaultdict(list)
    for index, case in enumerate(data.get("cases", [])):
        for row in case["measurements"]:
            if row.get("phase", "measure") == "measure":
                grouped[row["strategy"]].append((case.get("case_id", str(index)), case, row))
    parameters = data.get("parameters", {})
    result = []
    for strategy, entries in sorted(grouped.items()):
        pairs = defaultdict(list)
        for case_id, case, row in entries:
            pairs[(case_id, row.get("trace_id", 0))].append((case, row))
        raw_pairs = [values for values in pairs.values() if values[0][0].get("raw_correct", False)]
        values = [row["estimated_ttft_seconds"] for _, _, row in entries]
        violations = sum(bool(row["slo_violation"]) for _, _, row in entries)
        result.append({
            "context_tokens": parameters.get("length"),
            "bandwidth_scale": parameters.get("bandwidth_scale"),
            "slo_seconds": parameters.get("slo_seconds", parameters.get("slo")),
            "strategy": strategy, "contexts": len({case_id for case_id, _, _ in entries}),
            "observations": len(entries), "slo_violations": violations,
            "slo_violation_rate": violations / len(entries),
            "median_estimated_ttft_seconds": median(values), "p95_estimated_ttft_seconds": _p95(values),
            "qa_case_trace_pairs": len(pairs),
            "qa_correct_pairs": sum(all(row["correct"] for _, row in items) for items in pairs.values()),
            "raw_correct_case_trace_pairs": len(raw_pairs),
            "qa_retention_on_raw_correct_pairs":
            sum(all(row["correct"] for _, row in items) for items in raw_pairs) / len(raw_pairs)
            if raw_pairs else None,
        })
    return result


def _write_csv(path, rows):
    fields = list(rows[0]) if rows else ["job"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|")


def _table(rows, columns):
    if not rows:
        return "No completed measurements.\n"
    header = "| " + " | ".join(title for _, title in columns) + " |\n"
    divider = "| " + " | ".join("---" for _ in columns) + " |\n"
    body = "".join("| " + " | ".join(_format(row[key]) for key, _ in columns) + " |\n" for row in rows)
    return header + divider + body


def generate_report(run_dir, max_nll_delta=0.02, max_qa_drop=0.10):
    """Read completed manifest jobs and write CSV tables plus a readable report."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    purpose = manifest.get("profile", {}).get("purpose", "unspecified")
    jobs = manifest["jobs"]
    completed = []
    for job in jobs:
        if job.get("status") == "completed":
            data = json.loads((run_dir / job["result"]).read_text(encoding="utf-8"))
            completed.append((job, data))
    quality_rows, ttft_rows, adaptive_rows = [], [], []
    quality_gate = {}
    for job, data in completed:
        if job["kind"] == "quality":
            rows = summarize_quality(data, max_nll_delta, max_qa_drop)
            for row in rows:
                key = (row["context_tokens"], row["method"])
                quality_gate[key] = quality_gate.get(key, True) and row["meets_descriptive_threshold"]
            quality_rows.extend(dict(job=job["name"], **row) for row in rows)
    for job, data in completed:
        if job["kind"] == "ttft":
            ttft_rows.extend(dict(job=job["name"], **row) for row in _ttft_rows(data, quality_gate))
        elif job["kind"] == "adaptive":
            adaptive_rows.extend(dict(job=job["name"], **row) for row in _adaptive_rows(data))
    for filename, rows in (("quality.csv", quality_rows), ("ttft.csv", ttft_rows),
                           ("adaptive.csv", adaptive_rows)):
        _write_csv(run_dir / filename, rows)
    report = [
        "# Small-scale CacheGen experiment report\n",
        f"**Run purpose:** `{purpose}`.\n",
        f"Completed jobs: {len(completed)}/{len(jobs)}. Source: [manifest.json](manifest.json).\n",
        "This evaluates the repository codec on TinyLlama, using its exploratory quantization schedules. "
        "The codec lacks the paper's anchor/delta stage; single-host transfer and offline adaptation "
        "do not reproduce the paper's distributed streaming system.\n",
        "## Paired size and quality\n",
        f"Descriptive tolerances at each context length: mean paired NLL increase ≤ {max_nll_delta:g}; "
        f"QA retention on questions answered correctly by raw KV ≥ {1 - max_qa_drop:.0%}. "
        "A pass requires both natural continuation contexts and at least one raw-correct question. "
        "This small sample does not establish statistical noninferiority. Missing method answers count as failures.\n",
        "NLL scores teacher-forced continuation tokens 2..N, omitting the first held-out token; "
        "QA uses synthetic inserted facts. Compression ratio is paired raw byte sum / method byte sum. "
        "These are quality proxies, not the paper's benchmark scores. Details: [quality.csv](quality.csv).\n",
        _table(quality_rows, [("job", "Job"), ("context_tokens", "Tokens"), ("method", "Method"),
                              ("continuation_contexts", "Contexts"), ("compression_ratio", "Size ratio"),
                              ("mean_nll_delta", "ΔNLL"), ("raw_correct_contexts", "Raw-correct QA"),
                              ("qa_retention_on_raw_correct", "QA retention"),
                              ("meets_descriptive_threshold", "Descriptive pass")]),
        "## Transfer and time to first token\n",
        "Timings use a paced local socket and exclude offline encoding. Warmups are excluded. "
        "Contexts are unique inputs; observations include timing repeats. Median and P95 describe all timing "
        "observations and are not confidence bounds across independent contexts. Speedups are ratios of medians "
        "against text or uniform8. A quality label refers to the separate size/quality sample at the same length; "
        "each baseline comparison also requires that baseline to pass the descriptive tolerance. "
        "unmatched rows cannot support a claim of speedup at preserved quality. Details: [ttft.csv](ttft.csv).\n",
        _table(ttft_rows, [("job", "Job"), ("context_tokens", "Tokens"), ("bandwidth_gbps", "Gbps"),
                           ("method", "Method"), ("contexts", "Contexts"), ("observations", "Observations"),
                           ("median_seconds", "Median s"), ("p95_seconds", "P95 s"),
                           ("speedup_vs_text", "Text speedup"), ("quality_comparison", "Quality label")]),
        "## Bandwidth adaptation\n",
        "TTFT is trace-estimated transfer plus measured local restoration and first-token work; "
        "there is no live network. Text fallback replays original token IDs; UTF-8 length is a size proxy "
        "without receive-side retokenization. QA pairs deduplicate timing repeats by question and trace; traces on the "
        "same question are not independent QA samples. Violation rates retain all timing observations. "
        "adaptive_static is the former equal-per-chunk-budget diagnostic baseline; adaptive uses remaining "
        "budget and a first-token reserve, forecasting remaining costs from current bandwidth and offline "
        "per-chunk calibration. This causal heuristic is not the paper's online feedback scheduler. "
        "Details: [adaptive.csv](adaptive.csv).\n",
        _table(adaptive_rows, [("job", "Job"), ("strategy", "Strategy"), ("contexts", "Contexts"),
                               ("observations", "Observations"), ("slo_violation_rate", "SLO miss rate"),
                               ("median_estimated_ttft_seconds", "Median estimated s"),
                               ("qa_case_trace_pairs", "Question/trace pairs"),
                               ("qa_retention_on_raw_correct_pairs", "QA retention")]),
    ]
    if purpose == "pipeline-validation-only":
        report.insert(2, "**Pipeline validation only: this batch cannot support formal experiment conclusions.**\n")
    (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return {name: str(run_dir / filename) for name, filename in
            (("report", "report.md"), ("quality", "quality.csv"), ("ttft", "ttft.csv"), ("adaptive", "adaptive.csv"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    thresholds = manifest.get("profile", {}).get("quality_tolerances", {})
    paths = generate_report(args.run_dir, **thresholds)
    print(paths["report"])


if __name__ == "__main__":
    main()

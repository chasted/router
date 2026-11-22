"""Strict paired-round statistics; never infer a deployment benefit from mocks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from workload import digest, number


def percentile(values, q, *, signed=False):
    """Linear interpolation, equivalent to NumPy's default percentile method."""
    if not values:
        return None
    values = sorted(
        number(value, "sample", -math.inf if signed else 0) for value in values
    )
    index = (len(values) - 1) * q / 100
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def paired_interval(values, seed, samples=2000):
    if len(values) < 3:
        return None
    rng = random.Random(seed)
    means = [
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples)
    ]
    return [percentile(means, 2.5, signed=True), percentile(means, 97.5, signed=True)]


def local_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing or escaping evidence file: {relative}")
    return path


def read_records(root, run, trace):
    path = local_file(root, run["records"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != run["records_sha256"]:
        raise ValueError("request evidence hash changed")
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    expected = {row["id"]: row for row in trace}
    if len(records) != len(trace) or len({r["id"] for r in records}) != len(trace):
        raise ValueError("missing or duplicated request records")
    for record in records:
        row = expected.get(record["id"])
        if row is None or record["request_sha256"] != digest(row["body"]):
            raise ValueError("request identity/body differs from frozen trace")
        if record["planned_ms"] != row["offset_s"] * 1000:
            raise ValueError("arrival schedule differs from frozen trace")
        if record["status"] not in (
            "completed",
            "http_error",
            "timeout",
            "stream_error",
            "client_error",
        ):
            raise ValueError("unsent, cancelled, or unknown request status")
        sent = number(record["sent_ms"], "sent_ms")
        end = number(record["finished_ms"], "finished_ms", sent)
        lag = number(record["client_lag_ms"], "client_lag_ms")
        if not math.isclose(lag, sent - record["planned_ms"], abs_tol=1e-6):
            raise ValueError("client lag contradicts timestamps")
        first = record["first_content_ms"]
        if first is not None:
            number(first, "first_content_ms", sent)
            if first > end or not math.isclose(
                number(record["ttft_ms"], "ttft_ms"), first - sent, abs_tol=1e-6
            ):
                raise ValueError("TTFT contradicts timestamps")
        elif record["ttft_ms"] is not None:
            raise ValueError("TTFT without first content")
        if record["status"] == "completed" and (
            first is None or record["http_status"] != 200
        ):
            raise ValueError(
                "completed request lacks first content or successful HTTP status"
            )
    return records


def summarize(records, wall_ms):
    ttfts = [r["ttft_ms"] for r in records if r["ttft_ms"] is not None]
    complete = [r for r in records if r["status"] == "completed"]
    return {
        "planned": len(records),
        "completed": len(complete),
        "failed": len(records) - len(complete),
        "ttft_samples": len(ttfts),
        "failure_rate": 1 - len(complete) / len(records),
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p95_ms": percentile(ttfts, 95),
        "completion_p95_ms": percentile(
            [r["finished_ms"] - r["sent_ms"] for r in complete], 95
        ),
        "client_lag_p95_ms": percentile([r["client_lag_ms"] for r in records], 95),
        "client_lag_max_ms": max(r["client_lag_ms"] for r in records),
        "completed_requests_per_s": len(complete) * 1000 / wall_ms,
    }


def build(plan, trace, rounds, root):
    expected = [
        (i, arm)
        for i in range(1, plan["repeats"] + 1)
        for arm in (("baseline", "candidate") if i % 2 else ("candidate", "baseline"))
    ]
    if [(r["iteration"], r["arm"]) for r in rounds] != expected:
        raise ValueError("missing, duplicated, reordered, or extra paired rounds")
    groups = sorted({(r["scenario"], r["length_group"]) for r in trace})
    table, violations, indexed = [], [], {}
    for run in rounds:
        if run.get("complete") is not True:
            raise ValueError("round did not complete its lifecycle")
        telemetry = local_file(root, run["telemetry_samples"])
        if (
            hashlib.sha256(telemetry.read_bytes()).hexdigest()
            != run["telemetry_sha256"]
        ):
            raise ValueError("telemetry index hash changed")
        samples = json.loads(telemetry.read_text())
        if not samples:
            raise ValueError("missing telemetry samples")
        for sample in samples:
            if "raw_file" in sample:
                raw = local_file(telemetry.parent, sample["raw_file"])
                if hashlib.sha256(raw.read_bytes()).hexdigest() != sample["sha256"]:
                    raise ValueError("raw telemetry hash changed")
        records = read_records(root, run, trace)
        wall = number(run["wall_ms"], "wall_ms", 1e-12)
        if wall < max(r["finished_ms"] for r in records):
            raise ValueError("round duration contradicts request timestamps")
        for scenario, length in groups:
            ids = {
                r["id"]
                for r in trace
                if (r["scenario"], r["length_group"]) == (scenario, length)
            }
            stats = summarize([r for r in records if r["id"] in ids], wall)
            entry = {
                "iteration": run["iteration"],
                "arm": run["arm"],
                "scenario": scenario,
                "length_group": length,
                "cache_state": plan["cache_state"],
                **stats,
                "peak_client_backlog": run["peak_client_backlog"],
            }
            table.append(entry)
            indexed[run["iteration"], run["arm"], scenario, length] = entry
            prefix = f"round {run['iteration']} {run['arm']} {scenario}/{length}"
            if stats["failure_rate"] > plan["budgets"]["max_failure_rate"]:
                violations.append(prefix + ": failure budget exceeded")
            if stats["client_lag_max_ms"] > plan["budgets"]["max_client_lag_ms"]:
                violations.append(prefix + ": client fell behind planned arrivals")
        if run.get("telemetry_errors") or any(
            sample.get("error") for sample in samples
        ):
            violations.append(
                f"round {run['iteration']} {run['arm']}: incomplete telemetry"
            )
    comparisons = []
    for scenario, length in groups:
        pairs = []
        for iteration in range(1, plan["repeats"] + 1):
            baseline = indexed[iteration, "baseline", scenario, length]["ttft_p95_ms"]
            candidate = indexed[iteration, "candidate", scenario, length]["ttft_p95_ms"]
            if baseline is None or candidate is None or baseline <= 0:
                violations.append(
                    f"round {iteration} {scenario}/{length}: no usable TTFT denominator"
                )
                continue
            pairs.append(
                {
                    "iteration": iteration,
                    "order": "AB" if iteration % 2 else "BA",
                    "baseline_p95_ms": baseline,
                    "candidate_p95_ms": candidate,
                    "improvement_ms": baseline - candidate,
                    "improvement_pct": (baseline - candidate) / baseline * 100,
                }
            )
        values = [p["improvement_pct"] for p in pairs]
        mean = statistics.mean(values) if values else None
        if mean is not None and mean < -plan["budgets"]["max_ttft_regression_pct"]:
            violations.append(
                f"{scenario}/{length}: mean paired TTFT regression exceeds budget"
            )
        comparisons.append(
            {
                "scenario": scenario,
                "length_group": length,
                "pairs": pairs,
                "mean_improvement_ms": statistics.mean(
                    p["improvement_ms"] for p in pairs
                )
                if pairs
                else None,
                "mean_improvement_pct": mean,
                "paired_bootstrap_95_pct": paired_interval(values, plan["seed"]),
                "interval_unit": "independent paired rounds, not individual requests",
                "order_means_pct": {
                    order: statistics.mean(
                        p["improvement_pct"] for p in pairs if p["order"] == order
                    )
                    for order in ("AB", "BA")
                    if any(p["order"] == order for p in pairs)
                },
            }
        )
    coverage = {}
    for stage in ("prefill", "decode"):
        counts = [
            r.get("overrides", {}).get(stage) for r in rounds if r["arm"] == "candidate"
        ]
        required = stage in plan["require_overrides"]
        if required and (
            any(n is None for n in counts) or sum(n or 0 for n in counts) == 0
        ):
            violations.append(
                f"{stage}: pressure override not exercised or missing counter evidence"
            )
        coverage[stage] = {
            "required": required,
            "observed": None if any(n is None for n in counts) else sum(counts),
        }
    return {
        "schema_version": 1,
        "status": "functional_valid" if not violations else "invalid_measurement",
        "evidence_level": plan["evidence_level"],
        "backend_transport": plan["backend_transport"],
        "comparison": plan["comparison"],
        "performance_accepted": False,
        "violations": violations,
        "override_coverage": coverage,
        "rounds": table,
        "comparisons": comparisons,
        "limitations": [
            "Mock timings never establish deployment or NPU benefit.",
            "Intervals require >=3 independently reset paired rounds; fewer yields null.",
            "TTFT includes failed requests that emitted content; failure denominator is all planned requests.",
            "Stream chunks are not tokens; token ITL and backend phase timings are not inferred.",
            "Binary/config/URL mappings are caller-supplied, not remote process attestation.",
        ],
    }


def write(directory, result):
    directory = Path(directory)
    (directory / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (directory / "rounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rounds"][0]))
        writer.writeheader()
        writer.writerows(result["rounds"])
    lines = [
        "# HTTP PD experiment report",
        "",
        f"Status: {result['status']}",
        f"Evidence: {result['evidence_level']}; backend: {result['backend_transport']}",
        "Performance acceptance: **not assessed**.",
        "",
        "TTFT: HTTP call start to first nonempty generated content, in milliseconds.",
        "Percentiles use linear interpolation; positive (baseline-candidate)/baseline means improvement.",
        "",
        "| Scenario / input group | Mean paired difference (ms) | Mean improvement (%) | 95% paired-round interval (%) |",
        "|---|---:|---:|---|",
    ]
    for c in result["comparisons"]:
        lines.append(
            f"| {c['scenario']} / {c['length_group']} | {c['mean_improvement_ms']} | {c['mean_improvement_pct']} | {c['paired_bootstrap_95_pct']} |"
        )
    lines += [
        "",
        "See `rounds.csv` for every arm's raw statistics, sample and failure denominators.",
        "",
    ]
    lines += [f"- {item}" for item in result["violations"] + result["limitations"]]
    (directory / "report.md").write_text("\n".join(lines) + "\n")

"""Real Rust HTTP router + finite-capacity simulated engines, paired off/on runs."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid
from urllib.request import ProxyHandler, build_opener

from capacity_sim import MODEL, NAMES, PROFILES
from experiment import file_hash, write_json
import report
import workload

MODES = {
    "p": ("on", "off"),
    "d": ("off", "on"),
    "both": ("on", "on"),
    "aa": ("off", "off"),
}


def ports(count):
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [s.getsockname()[1] for s in sockets]
    finally:
        for sock in sockets:
            sock.close()


def trace(count, rate, seed):
    rows = workload.generate(
        count,
        rate,
        seed,
        MODEL,
        short_words=64,
        long_words=256,
        output_tokens=16,
        scenario="capacity_pressure",
    )
    # A startup burst followed by sustained arrivals; exact schedule frozen in JSONL.
    for i, row in enumerate(rows):
        row["offset_s"] = max(0, i - 7) / rate
        row["body"]["user"] = row["id"]
    return rows


def create_plan(directory, args, base, epoch, router_port, metrics_port, profile, mode):
    directory.mkdir(parents=True)
    script = Path(__file__).with_name("capacity_sim.py")
    rows = trace(args.num_prompts, args.request_rate, args.seed)
    workload.write(directory / "input.jsonl", rows)
    router = f"http://127.0.0.1:{router_port}"
    metrics = f"http://127.0.0.1:{metrics_port}/metrics"
    topology, mappings = [], {s: [] for s in ("prefill", "decode")}
    for name in NAMES:
        stage = "prefill" if name[0] == "p" else "decode"
        mapping = {
            "inference_endpoint": base + "/" + name,
            "dp_rank": None,
            "metrics_url": base + "/" + name + "/pressure-metrics",
            "usage": {
                "name": "vllm:kv_cache_usage_perc",
                "labels": {"engine": name, "model_name": MODEL},
            },
            "identity": {"name": "router_kv_identity", "labels": {"engine": name}},
            "expected_engine_id": name,
            "expected_instance_epoch": epoch,
            "expected_layout_fingerprint": "sim-layout-128",
        }
        mappings[stage].append(mapping)
        topology.append(
            {
                "stage": stage,
                "endpoint": mapping["inference_endpoint"],
                "dp_rank": None,
                "metrics_url": mapping["metrics_url"],
                "usage": mapping["usage"],
                "identity": mapping["identity"],
                "engine_id": name,
                "instance_epoch": epoch,
                "layout_fingerprint": "sim-layout-128",
            }
        )
    arms = {}
    for arm, modes in [("baseline", ("off", "off")), ("candidate", MODES[mode])]:
        stage_modes = dict(zip(("prefill", "decode"), modes))
        pressure = {
            s: {
                "mode": m,
                "high": 0.85,
                "low": 0.70,
                "poll_interval_ms": 20,
                "timeout_ms": 200,
                "max_age_ms": 1000,
                "max_skew_ms": 300,
                "mappings": mappings[s],
            }
            for s, m in stage_modes.items()
        }
        argv = [
            str(args.router_binary),
            "--host",
            "127.0.0.1",
            "--port",
            str(router_port),
            "--prometheus-port",
            str(metrics_port),
            "--vllm-pd-disaggregation",
            "--prefill",
            base + "/p0",
            "--prefill",
            base + "/p1",
            "--decode",
            base + "/d0",
            "--decode",
            base + "/d1",
            "--policy",
            "cache_aware",
            "--prefill-policy",
            "cache_aware",
            "--decode-policy",
            "cache_aware",
            "--pd-pressure-config",
            json.dumps(pressure),
            "--disable-retries",
            "--disable-circuit-breaker",
            "--worker-startup-timeout-secs",
            "15",
            "--request-timeout-secs",
            "30",
            "--max-concurrent-requests",
            "256",
            "--cache-threshold",
            "0.3",
            "--balance-abs-threshold",
            "64",
            "--balance-rel-threshold",
            "1.5",
        ]
        config = directory / f"{arm}-config.json"
        write_json(
            config,
            {
                "pd_pressure": pressure,
                "launch_argv": argv,
                "policy": {
                    "type": "cache_aware",
                    "cache_threshold": 0.3,
                    "balance_abs_threshold": 64,
                    "balance_rel_threshold": 1.5,
                },
            },
        )
        control = [sys.executable, str(script)]
        arms[arm] = {
            "url": router,
            "binary": str(args.router_binary),
            "source_repo": str(args.repo),
            "config": str(config),
            "modes": stage_modes,
            "topology": topology,
            "start": [argv],
            "ready_urls": [router + "/health", metrics],
            "reset": [control + ["reset", "--base", base]],
            "warmup": [
                control
                + [
                    "prime",
                    "--base",
                    base,
                    "--router",
                    router,
                    "--trace",
                    str(directory / "input.jsonl"),
                    "--profile",
                    profile,
                ]
            ],
            "stop": [
                control
                + [
                    "audit",
                    "--base",
                    base,
                    "--output",
                    "{artifact_dir}/backend-audit.json",
                ]
            ],
            "telemetry": [{"name": "router", "stage": "router", "url": metrics}]
            + [
                {"name": w["engine_id"], "stage": w["stage"], "url": w["metrics_url"]}
                for w in topology
            ],
            "override_counters": {
                s: {
                    "source": "router",
                    "metric": "vllm_router_pd_pressure_overrides_total",
                    "labels": {"stage": s},
                }
                for s, m in stage_modes.items()
                if m == "on"
            },
        }
    hot_stages = {
        "low": (),
        "all_hot": (),
        "p_hot": ("prefill",),
        "d_hot": ("decode",),
        "both_hot": ("prefill", "decode"),
        "recovery": ("prefill", "decode"),
        "signal_fault": ("prefill", "decode"),
    }[profile]
    plan = {
        "schema_version": 1,
        "evidence_level": "mock",
        "backend_transport": "http",
        "comparison": "aa" if mode == "aa" else "candidate",
        "cache_state": "warm",
        "repeats": args.repeat,
        "seed": args.seed,
        "concurrency": args.stress_concurrency,
        "request_timeout_s": 30,
        "ready_timeout_s": 20,
        "sample_interval_s": 0.05,
        "budgets": {
            "max_failure_rate": 0,
            "max_client_lag_ms": 100,
            "max_ttft_regression_pct": 15,
        },
        "hardware": {
            "kind": "CPU simulation",
            "description": "real Rust HTTP router; finite 128-block pools, 8 slots; no accelerator",
        },
        "model": {
            "id": MODEL,
            "revision": "simulation-v1",
            "tokenizer_revision": "sha256-word-id-v1",
            "chat_template": "text-words-v1",
        },
        "connector": {
            "name": "mock",
            "version": "simulation-v1",
            "mode": "native HTTP metadata handoff; no physical KV transfer",
        },
        "require_overrides": [
            s for s in hot_stages if arms["candidate"]["modes"][s] == "on"
        ],
        "trace": str(directory / "input.jsonl"),
        "arms": arms,
        "simulation": {
            "profile": profile,
            "mode": mode,
            "assert_fault_coverage": profile == "signal_fault"
            and rows[-1]["offset_s"] >= 1.5,
            "backend_epoch": epoch,
            "capacity_blocks": 128,
            "hot_background_blocks": 116,
            "low_background_blocks": 16,
            "request_blocks": "2 + ceil(input_words / 64) + 1",
            "slots": 8,
            "prefill_seconds": "0.03 + (input_words - cached_prefix_words) * 0.0005",
            "cached_prefix_words": "0 on first prefix, else min(32, input_words)",
            "decode_seconds_per_chunk": 0.01,
            "transfer_seconds": "0.005 + input_words * 0.00002",
        },
    }
    write_json(directory / "plan.json", plan)
    return directory / "plan.json"


def metric_rows(text, name):
    rows = []
    for line in text.splitlines():
        match = re.fullmatch(re.escape(name) + r"\{(.*)\}\s+([0-9.eE+\-]+)", line)
        if match:
            labels = {
                k: json.loads('"' + v + '"')
                for k, v in re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', match[1])
            }
            rows.append((labels, float(match[2])))
    return rows


def router_audit(root, modes, planned):
    samples = json.loads((root / "workers/samples.json").read_text())
    selected = [s for s in samples if s["source"] == "router"]
    raw = []
    for sample in (selected[0], selected[-1]):
        path = report.local_file(root / "workers", sample["raw_file"])
        if file_hash(path) != sample["sha256"]:
            raise ValueError("router metric evidence changed")
        raw.append(path.read_text())
    for name in (
        "vllm_router_pd_prefill_occupancy",
        "vllm_router_pd_pending_decode_requests",
        "vllm_router_pd_active_decode_requests",
    ):
        rows = metric_rows(raw[-1], name)
        if not rows or any(value != 0 for _, value in rows):
            raise ValueError("actual router occupancy did not drain: " + name)
    reasons = {}
    for stage, mode in modes.items():
        counts = []
        for text in raw:
            counts.append(
                {
                    labels["reason"]: value
                    for labels, value in metric_rows(
                        text, "vllm_router_pd_pressure_decisions_total"
                    )
                    if labels.get("stage") == stage and labels.get("mode") == mode
                }
            )
        reasons[stage] = {
            reason: counts[1].get(reason, 0) - counts[0].get(reason, 0)
            for reason in counts[1]
        }
        if sum(reasons[stage].values()) != (0 if mode == "off" else planned):
            raise ValueError(
                "actual stage decisions do not match the measured request count"
            )
    return reasons


def reconcile(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    plan = json.loads((directory / "plan.json").read_text())
    rows = workload.load(directory / "trace.jsonl")
    expected = {r["id"] for r in rows}
    result = []
    for run in manifest["rounds"]:
        root = directory / f"round-{run['iteration']}-{run['arm']}"
        data = json.loads((root / "backend-audit.json").read_text())
        records = data["records"]
        if len(records) != 2 * len(rows) or any(
            p["used"] or p["active"] or p["waiting"] for p in data["pools"].values()
        ):
            raise ValueError("missing requests or leaked simulation resources")
        stages = {}
        for stage, prefix in [("prefill", "p"), ("decode", "d")]:
            items = [r for r in records if r["worker"].startswith(prefix)]
            if (
                len(items) != len(rows)
                or {r["id"] for r in items} != expected
                or any(r["status"] != "completed" for r in items)
            ):
                raise ValueError("stage identities/count/completions do not reconcile")
            if any(p["peak_used"] > p["capacity"] for p in data["pools"].values()):
                raise ValueError("simulation exceeded capacity")
            stages[stage] = {
                "dispatches": dict(Counter(r["worker"] for r in items)),
                "queue_p95_ms": report.percentile([r["queue_ms"] for r in items], 95),
                "overrides": run.get("overrides", {}).get(stage, 0),
            }
        prefill = {r["id"]: r["worker"] for r in records if r["worker"][0] == "p"}
        if any(
            r["prefill_worker"] != prefill[r["id"]]
            for r in records
            if r["worker"][0] == "d"
        ):
            raise ValueError("actual P/D handoff mismatch")
        client = report.read_records(directory, run, rows)
        stats = report.summarize(client, run["wall_ms"])
        reasons = router_audit(root, plan["arms"][run["arm"]]["modes"], len(rows))
        if (
            plan["simulation"]["profile"] == "signal_fault"
            and run["arm"] == "candidate"
            and plan["arms"]["candidate"]["modes"]["prefill"] == "on"
            and plan["simulation"].get("assert_fault_coverage", False)
            and reasons["prefill"].get("identity_mismatch", 0) <= 0
        ):
            raise ValueError("identity fault was not observed by the real router")
        result.append(
            {
                "iteration": run["iteration"],
                "arm": run["arm"],
                "profile": plan["simulation"]["profile"],
                "mode": plan["simulation"]["mode"],
                "stages": stages,
                "pressure_reasons": reasons,
                "router_final_occupancy": 0,
                **stats,
                "events": data["events"],
                "pools": data["pools"],
                "audit_sha256": file_hash(root / "backend-audit.json"),
            }
        )
    write_json(directory / "reconciled.json", result)
    return result


def suite(recorder, args):
    if args.repeat < 2:
        raise ValueError("stage 4 requires at least two paired repetitions")
    chosen_profiles = args.simulation_profile or list(PROFILES)
    chosen_modes = args.simulation_modes or ["p", "d", "both"]
    backend_port, router_port, metrics_port = ports(3)
    base, epoch = f"http://127.0.0.1:{backend_port}", str(uuid.uuid4())
    backend = [
        sys.executable,
        str(Path(__file__).with_name("capacity_sim.py")),
        "serve",
        "--base",
        base,
        "--port",
        str(backend_port),
        "--epoch",
        epoch,
    ]
    recorder.data.update(
        simulation_backend=backend,
        router_binary_sha256=file_hash(args.router_binary),
        performance_accepted=False,
    )
    recorder.save()
    all_results, experiments = [], []
    with (recorder.directory / "backend.log").open("wb") as log:
        process = subprocess.Popen(
            backend, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic() + 10
            while True:
                try:
                    with opener.open(base + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise ValueError("simulation backend did not become ready")
                time.sleep(0.05)
            for profile in chosen_profiles:
                for mode in chosen_modes:
                    name = f"{profile}-{mode}"
                    plan = create_plan(
                        recorder.directory / "plans" / name,
                        args,
                        base,
                        epoch,
                        router_port,
                        metrics_port,
                        profile,
                        mode,
                    )
                    output = recorder.directory / name
                    code, _ = recorder.run(
                        name,
                        [
                            sys.executable,
                            str(Path(__file__).with_name("run.py")),
                            "experiment",
                            "--stage",
                            "3",
                            "--base-ref",
                            args.base_ref,
                            "--repo",
                            str(args.repo),
                            "--out-dir",
                            str(output),
                            "--plan",
                            str(plan),
                            "--timeout",
                            str(args.timeout),
                        ],
                        timeout=args.timeout,
                    )
                    child = json.loads((output / "manifest.json").read_text())
                    if child["status"] not in (
                        "functional_valid",
                        "invalid_measurement",
                    ):
                        raise ValueError(f"{name} experiment failed: {child}")
                    results = reconcile(output)
                    all_results.extend(results)
                    experiments.append(
                        {
                            "name": name,
                            "exit_code": code,
                            "status": child["status"],
                            "report": str(output / "report.json"),
                            "rounds": len(results),
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "simulation": name,
                                "status": child["status"],
                                "rounds": len(results),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    write_json(
                        recorder.directory / "simulated-results.json",
                        {"experiments": experiments, "rounds": all_results},
                    )
        finally:
            os.killpg(process.pid, signal.SIGTERM) if process.poll() is None else None
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            recorder.data["backend_exit_code"] = process.returncode
            recorder.save()
    lines = [
        "# Real-router capacity simulation",
        "",
        "Measured simulation only; no NPU/model performance acceptance.",
        "",
        "| Scenario | Candidate | Baseline p95 ms (round mean) | Candidate p95 ms (round mean) | Paired improvement % | P/D overrides | Completed/planned |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for entry in experiments:
        selected = [
            r for r in all_results if f"{r['profile']}-{r['mode']}" == entry["name"]
        ]
        a = [r for r in selected if r["arm"] == "baseline"]
        b = [r for r in selected if r["arm"] == "candidate"]
        improvements = [
            (x["ttft_p95_ms"] - y["ttft_p95_ms"]) / x["ttft_p95_ms"] * 100
            for x, y in zip(a, b)
        ]
        counts = [
            sum(r["stages"][s]["overrides"] for r in b) for s in ("prefill", "decode")
        ]
        lines.append(
            f"| {b[0]['profile']} | {b[0]['mode']} | {statistics.mean(r['ttft_p95_ms'] for r in a):.3f} | {statistics.mean(r['ttft_p95_ms'] for r in b):.3f} | {statistics.mean(improvements):.2f} | {counts[0]}/{counts[1]} | {sum(r['completed'] for r in selected)}/{sum(r['planned'] for r in selected)} |"
        )
    (recorder.directory / "simulated-results.md").write_text("\n".join(lines) + "\n")
    recorder.data["experiments"] = experiments
    passed = all(e["exit_code"] == 0 for e in experiments)
    return recorder.finish(
        "passed" if passed else "completed_with_budget_violations",
        0 if passed else 1,
        "Real router with simulated capacity/queues. See raw paired reports and backend audit; no NPU acceptance.",
    )

"""CPU-only tooling fixture, not a vLLM engine or production router implementation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import socket
import sys
import urllib.request

from aiohttp import web

import workload
from experiment import write_json


def application(config, fault=None, delay=0.001):
    pressure = config.get("pd_pressure", {})
    modes = {s: pressure.get(s, {}).get("mode", "off") for s in ("prefill", "decode")}
    state = {
        "requests": 0,
        "completed": 0,
        "handoffs": 0,
        "pairs": [],
        "modes": modes,
        "prefill": [0, 0],
        "decode": [0, 0],
        "proposals": {"prefill": 0, "decode": 0},
    }

    async def health(request):
        return web.Response(text="ready")

    async def reset(request):
        state.update(
            requests=0,
            completed=0,
            handoffs=0,
            pairs=[],
            prefill=[0, 0],
            decode=[0, 0],
            proposals={"prefill": 0, "decode": 0},
        )
        return web.json_response({"reset": True})

    async def audit(request):
        return web.json_response(state)

    async def metrics(request):
        if fault == "metrics_error":
            return web.Response(status=503)
        lines = [
            f"pd_tool_requests_total {state['requests']}",
            f"pd_tool_completed_total {state['completed']}",
        ]
        for stage in ("prefill", "decode"):
            lines += [
                f'pd_tool_overrides_total{{stage="{stage}"}} 0',
                f'pd_tool_proposals_total{{stage="{stage}"}} {state["proposals"][stage]}',
            ]
            for index in range(2):
                lines.append(
                    f'pd_tool_dispatch_total{{stage="{stage}",worker="{index}"}} {state[stage][index]}'
                )
                lines.append(
                    f'vllm:kv_cache_usage_perc{{stage="{stage}",engine="{index}"}} {0.9 if index == 0 else 0.6}'
                )
                prefix = "p" if stage == "prefill" else "d"
                endpoint = f"{request.scheme}://{request.host}/{prefix}{index}"
                lines.append(
                    f'pd_tool_identity{{stage="{stage}",engine="{index}",endpoint="{endpoint}",dp_rank="aggregate",engine_id="{prefix}{index}",instance_epoch="mock-boot",layout_fingerprint="mock-layout"}} 1'
                )
        return web.Response(text="\n".join(lines) + "\n")

    async def chat(request):
        body = await request.json()
        state["requests"] += 1
        if fault == "http_error":
            return web.Response(status=503)
        # In-memory P then D handoff: deliberately no GPU or KV-transfer claim.
        state["prefill"][0] += 1
        transfer = {
            "prefill_worker": 0,
            "request_id": request.headers.get("x-request-id", "warmup"),
        }
        await asyncio.sleep(delay)
        assert transfer["prefill_worker"] == 0
        state["decode"][0] += 1
        state["handoffs"] += 1
        state["pairs"].append(
            {"request_id": transfer["request_id"], "prefill": 0, "decode": 0}
        )
        for stage in ("prefill", "decode"):
            if modes[stage] == "shadow":
                state["proposals"][stage] += 1
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})

        async def event(value):
            await response.write(("data: " + json.dumps(value) + "\n\n").encode())

        try:
            await response.prepare(request)
            await event({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
            await asyncio.sleep(delay)
            if fault == "malformed":
                await response.write(b"data: {invalid json}\n\n")
            elif fault != "role_only":
                await event({"choices": [{"index": 0, "delta": {"content": "first"}}]})
                await asyncio.sleep(delay)
                await event(
                    {"choices": [{"index": 0, "delta": {"content": " second"}}]}
                )
                await event(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": len(
                                body["messages"][0]["content"].split()
                            ),
                            "completion_tokens": 2,
                        },
                    }
                )
            if fault != "partial":
                await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            state["completed"] += 1
        except (ConnectionResetError, RuntimeError):
            pass
        return response

    app = web.Application()
    app.add_routes(
        [
            web.get("/health", health),
            web.post("/reset", reset),
            web.get("/audit", audit),
            web.get("/metrics", metrics),
            web.post("/v1/chat/completions", chat),
        ]
    )
    return app


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def fixture_plan(
    directory,
    repo,
    *,
    comparison="aa",
    count=8,
    concurrency=4,
    fault=None,
    delay=0.001,
    repeats=2,
):
    directory.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    trace = directory / "input.jsonl"
    workload.write(trace, workload.generate(count, 500, 17, "pd-tool-mock", 4, 16, 4))
    plan = {
        "schema_version": 1,
        "evidence_level": "mock",
        "backend_transport": "http",
        "comparison": comparison,
        "cache_state": "warm",
        "repeats": repeats,
        "seed": 17,
        "concurrency": concurrency,
        "request_timeout_s": 2,
        "ready_timeout_s": 10,
        "sample_interval_s": 0.01,
        "trace": str(trace),
        "require_overrides": [],
        "hardware": {
            "kind": "cpu_mock",
            "description": "CPU HTTP tooling fixture; no accelerator",
        },
        "model": {
            "id": "pd-tool-mock",
            "revision": "fixture-v1",
            "tokenizer_revision": "word-count-fixture",
            "chat_template": "text-only fixture, not a model tokenizer",
        },
        "connector": {"name": "in-memory-mock", "version": "1", "mode": "sequential"},
        "budgets": {
            "max_failure_rate": 0,
            "max_client_lag_ms": 10000,
            "max_ttft_regression_pct": 1000000,
        },
        "arms": {},
    }
    off = directory / "off.json"
    shadow = directory / "shadow.json"
    write_json(
        off, {"pd_pressure": {"prefill": {"mode": "off"}, "decode": {"mode": "off"}}}
    )
    write_json(
        shadow,
        {"pd_pressure": {"prefill": {"mode": "shadow"}, "decode": {"mode": "shadow"}}},
    )
    for name in ("baseline", "candidate"):
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        config = shadow if name == "candidate" and comparison == "off_shadow" else off
        mode = "shadow" if config == shadow else "off"
        start = [
            sys.executable,
            str(script),
            "serve",
            "--port",
            str(port),
            "--config",
            str(config),
            "--delay",
            str(delay),
        ]
        if fault and name == "candidate":
            start += ["--fault", fault]
        plan["arms"][name] = {
            "url": url,
            "binary": sys.executable,
            "config": str(config),
            "source_repo": str(repo),
            "modes": {"prefill": mode, "decode": mode},
            "topology": [
                {
                    "stage": stage,
                    "endpoint": f"{url}/{prefix}{index}",
                    "dp_rank": None,
                    "engine_id": f"{prefix}{index}",
                    "instance_epoch": "mock-boot",
                    "layout_fingerprint": "mock-layout",
                    "metrics_url": url + "/metrics",
                    "usage": {
                        "name": "vllm:kv_cache_usage_perc",
                        "labels": {"stage": stage, "engine": str(index)},
                    },
                    "identity": {
                        "name": "pd_tool_identity",
                        "labels": {"stage": stage, "engine": str(index)},
                    },
                }
                for stage, prefix in (("prefill", "p"), ("decode", "d"))
                for index in range(2)
            ],
            "start": [start],
            "reset": [
                [
                    sys.executable,
                    str(script),
                    "control",
                    "--url",
                    url,
                    "--action",
                    "reset",
                ]
            ],
            "warmup": [
                [
                    sys.executable,
                    str(script),
                    "control",
                    "--url",
                    url,
                    "--action",
                    "warmup",
                ]
            ],
            "stop": [
                [
                    sys.executable,
                    str(script),
                    "control",
                    "--url",
                    url,
                    "--action",
                    "audit",
                    "--output",
                    "{artifact_dir}/audit.json",
                ]
            ],
            "ready_urls": [url + "/health"],
            "telemetry": [
                {"name": "router", "stage": "router", "url": url + "/metrics"}
            ],
            "override_counters": {
                s: {
                    "source": "router",
                    "metric": "pd_tool_overrides_total",
                    "labels": {"stage": s},
                }
                for s in ("prefill", "decode")
            },
        }
    path = directory / "plan.json"
    write_json(path, plan)
    return path


def selftest(recorder, args):
    count = args.stress_rounds * args.stress_concurrency
    outcomes = []
    for repeat in range(1, args.repeat + 1):
        for comparison in ("aa", "off_shadow"):
            label = f"selftest-{repeat}-{comparison}"
            plan = fixture_plan(
                recorder.directory / (label + "-input"),
                args.repo,
                comparison=comparison,
                count=count,
                concurrency=args.stress_concurrency,
            )
            output = recorder.directory / label
            code, _ = recorder.run(
                label,
                [
                    sys.executable,
                    str(Path(__file__).with_name("run.py")),
                    "experiment",
                    "--stage",
                    "3",
                    "--repo",
                    str(args.repo),
                    "--base-ref",
                    args.base_ref,
                    "--plan",
                    str(plan),
                    "--out-dir",
                    str(output),
                    "--timeout",
                    str(args.timeout),
                ],
            )
            if code:
                return recorder.finish(
                    "failed", 1, f"{label}: inspect child experiment evidence"
                )
            result = json.loads((output / "report.json").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
            if result["status"] != "functional_valid" or len(manifest["rounds"]) != 4:
                raise ValueError("selftest did not produce all paired rounds")
            for run in manifest["rounds"]:
                audit = json.loads(
                    (
                        output / f"round-{run['iteration']}-{run['arm']}" / "audit.json"
                    ).read_text()
                )
                expected_mode = (
                    "shadow"
                    if comparison == "off_shadow" and run["arm"] == "candidate"
                    else "off"
                )
                if any(
                    audit[key] != count + 1
                    for key in ("requests", "completed", "handoffs")
                ):
                    raise ValueError(
                        "mock requests, warmup and handoffs do not reconcile"
                    )
                if audit["modes"] != {
                    "prefill": expected_mode,
                    "decode": expected_mode,
                } or any(audit[s] != [count + 1, 0] for s in ("prefill", "decode")):
                    raise ValueError("mock stage modes/actual targets changed")
                if any(not p.get("cleaned_up") for p in run["processes"]):
                    raise ValueError("owned process cleanup is missing")
                if run["overrides"] != {"prefill": 0, "decode": 0}:
                    raise ValueError(
                        "off/shadow mock unexpectedly overrode actual target"
                    )
            outcomes.append(
                {
                    "repeat": repeat,
                    "comparison": comparison,
                    "rounds": 4,
                    "measured_requests": 4 * count,
                    "status": result["status"],
                }
            )
    recorder.data["selftest_results"] = outcomes
    return recorder.finish(
        "passed",
        0,
        "PR3 CPU tooling selftest only; production router covered by stage 2 separately",
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    server = sub.add_parser("serve")
    server.add_argument("--port", type=int, required=True)
    server.add_argument("--config", type=Path, required=True)
    server.add_argument(
        "--fault",
        choices=("http_error", "partial", "role_only", "malformed", "metrics_error"),
    )
    server.add_argument("--delay", type=float, default=0.001)
    control = sub.add_parser("control")
    control.add_argument("--url", required=True)
    control.add_argument(
        "--action", choices=("reset", "warmup", "audit"), required=True
    )
    control.add_argument("--output", type=Path)
    generate = sub.add_parser("plan")
    generate.add_argument("--directory", type=Path, required=True)
    generate.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3]
    )
    generate.add_argument("--comparison", choices=("aa", "off_shadow"), default="aa")
    generate.add_argument("--count", type=int, default=16)
    generate.add_argument("--concurrency", type=int, default=4)
    args = cli.parse_args()
    if args.command == "serve":
        web.run_app(
            application(json.loads(args.config.read_text()), args.fault, args.delay),
            host="127.0.0.1",
            port=args.port,
            print=None,
        )
    elif args.command == "plan":
        print(
            fixture_plan(
                args.directory.resolve(),
                args.repo.resolve(),
                comparison=args.comparison,
                count=args.count,
                concurrency=args.concurrency,
            )
        )
    else:
        data = None
        path = "/" + args.action
        if args.action == "warmup":
            path = "/v1/chat/completions"
            data = json.dumps(
                workload.generate(1, 1, 0, "pd-tool-mock")[0]["body"]
            ).encode()
        elif args.action == "reset":
            data = b"{}"
        request = urllib.request.Request(
            args.url + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read()
        if args.output:
            args.output.write_bytes(body)
        print(json.dumps({"action": args.action, "bytes": len(body)}))


if __name__ == "__main__":
    main()

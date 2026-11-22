"""Explicit deployment lifecycle and paired HTTP PD measurement protocol."""

from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import aiohttp

import report
import workload
from replay import replay

ARMS = ("baseline", "candidate")
STAGES = ("prefill", "decode")


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plan(path, url_validator):
    path = path.resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1:
        raise ValueError("expected plan schema_version 1")
    if plan.get("evidence_level") not in ("mock", "deployment"):
        raise ValueError("evidence_level must be mock or deployment")
    if plan.get("backend_transport") != "http":
        raise ValueError(
            "PR3 replay is HTTP PD; run gRPC compatibility separately with stress --stage 2"
        )
    if plan.get("comparison") not in ("aa", "off_shadow", "candidate"):
        raise ValueError("unknown comparison")
    if plan.get("cache_state") not in ("cold", "warm"):
        raise ValueError("declare cold or warm cache_state")
    workload.integer(plan.get("repeats"), "repeats", 2)
    workload.integer(plan.get("seed"), "seed", 0)
    workload.integer(plan.get("concurrency"), "concurrency")
    for name in ("request_timeout_s", "ready_timeout_s", "sample_interval_s"):
        workload.number(plan.get(name), name, 0.001)
    budgets = plan.get("budgets", {})
    for name in ("max_failure_rate", "max_client_lag_ms", "max_ttft_regression_pct"):
        workload.number(budgets.get(name), name)
    if budgets["max_failure_rate"] > 1:
        raise ValueError("failure budget must be a fraction in [0,1]")
    for name, fields in (
        ("hardware", ("kind", "description")),
        ("model", ("id", "revision", "tokenizer_revision", "chat_template")),
        ("connector", ("name", "version", "mode")),
    ):
        if not all(
            isinstance(plan.get(name, {}).get(field), str) and plan[name][field].strip()
            for field in fields
        ):
            raise ValueError(f"explicit {name} metadata is required")
    if plan["evidence_level"] == "deployment" and plan["connector"]["name"].lower() in (
        "unknown",
        "mock",
        "none",
    ):
        raise ValueError("deployment requires an identified KV connector")
    required = plan.get("require_overrides")
    if (
        not isinstance(required, list)
        or len(required) != len(set(required))
        or any(s not in STAGES for s in required)
    ):
        raise ValueError("require_overrides must be a unique list of stages")

    def existing(value, directory=False):
        if not isinstance(value, str) or not value:
            raise ValueError("explicit artifact path is required")
        resolved = (path.parent / value).resolve()
        if not (resolved.is_dir() if directory else resolved.is_file()):
            raise ValueError(f"artifact does not exist: {resolved}")
        return str(resolved)

    def commands(value):
        if not isinstance(value, list) or not all(
            isinstance(argv, list)
            and argv
            and all(isinstance(token, str) and token for token in argv)
            for argv in value
        ):
            raise ValueError("hooks must be lists of argv arrays, never shell strings")

    plan["trace"] = existing(plan.get("trace"))
    rows = workload.load(plan["trace"])
    if any(r["body"]["model"] != plan["model"]["id"] for r in rows):
        raise ValueError("trace model differs from declared model")
    if set(plan.get("arms", {})) != set(ARMS):
        raise ValueError("exactly baseline and candidate arms are required")
    for name, arm in plan["arms"].items():
        arm["url"] = url_validator(arm["url"])
        for key in ("binary", "config"):
            arm[key] = existing(arm.get(key))
        arm["source_repo"] = existing(arm.get("source_repo"), directory=True)
        if set(arm.get("modes", {})) != set(STAGES) or any(
            arm["modes"][s] not in ("off", "shadow", "on") for s in STAGES
        ):
            raise ValueError("declare both stage modes")
        config = json.loads(Path(arm["config"]).read_text())
        pressure = config.get("pd_pressure") or {}
        if any(
            pressure.get(s, {}).get("mode", "off") != arm["modes"][s] for s in STAGES
        ):
            raise ValueError(
                "declared modes disagree with captured RouterConfig pd_pressure"
            )
        for stage in STAGES:
            settings = pressure.get(stage, {})
            low = workload.number(settings.get("low", 0.7), "low")
            high = workload.number(settings.get("high", 0.85), "high")
            if not low < high <= 1:
                raise ValueError("pressure thresholds require 0 <= low < high <= 1")
        topology = arm.get("topology", [])
        if not isinstance(topology, list) or {w.get("stage") for w in topology} != set(
            STAGES
        ):
            raise ValueError("topology must identify both P and D workers")
        identities = set()
        for worker in topology:
            worker["endpoint"] = url_validator(worker["endpoint"])
            for field in ("engine_id", "instance_epoch", "layout_fingerprint"):
                if not isinstance(worker.get(field), str) or not worker[field]:
                    raise ValueError(f"worker {field} is required")
            worker["metrics_url"] = url_validator(worker["metrics_url"])
            for field in ("usage", "identity"):
                selector = worker.get(field, {})
                if not re.fullmatch(
                    r"[a-zA-Z_:][a-zA-Z0-9_:]*", selector.get("name", "")
                ) or not isinstance(selector.get("labels"), dict):
                    raise ValueError(
                        "worker metrics need explicit usage and identity selectors"
                    )
                if not all(
                    isinstance(k, str) and isinstance(v, str)
                    for k, v in selector["labels"].items()
                ):
                    raise ValueError("metric selector labels must be strings")
            key = (worker["stage"], worker["endpoint"], worker.get("dp_rank"))
            if key in identities:
                raise ValueError("duplicate topology worker")
            identities.add(key)
        for hook in ("start", "reset", "warmup", "stop"):
            commands(arm.get(hook))
        if arm["start"] and os.name != "posix":
            raise ValueError("managed process groups require Linux/WSL")
        if plan["cache_state"] == "cold" and arm["warmup"]:
            raise ValueError("cold measurements must not run a cache-warming workload")
        if plan["cache_state"] == "warm" and not arm["warmup"]:
            raise ValueError("warm measurements need an explicit warmup command")
        if not arm["reset"]:
            raise ValueError(
                "every arm needs an explicit reset command; its real semantics remain caller-verified"
            )
        if not arm.get("ready_urls"):
            raise ValueError("ready_urls are required")
        arm["ready_urls"] = [url_validator(url) for url in arm["ready_urls"]]
        telemetry = arm.get("telemetry")
        if not isinstance(telemetry, list) or not telemetry:
            raise ValueError("explicit telemetry endpoints are required")
        names = set()
        for source in telemetry:
            if (
                not re.fullmatch(r"[a-zA-Z0-9_-]+", source.get("name", ""))
                or source["name"] in names
            ):
                raise ValueError("telemetry names must be unique safe file names")
            names.add(source["name"])
            source["url"] = url_validator(source["url"])
            if source.get("stage") not in (*STAGES, "router"):
                raise ValueError("telemetry stage must be prefill, decode, or router")
        for stage, selector in arm.get("override_counters", {}).items():
            if (
                stage not in STAGES
                or selector.get("source") not in names
                or not isinstance(selector.get("labels"), dict)
            ):
                raise ValueError("invalid explicit override counter selector")
            if not re.fullmatch(
                r"[a-zA-Z_:][a-zA-Z0-9_:]*", selector.get("metric", "")
            ):
                raise ValueError("invalid counter metric")
        if any(w["metrics_url"] not in {s["url"] for s in telemetry} for w in topology):
            raise ValueError("every worker metrics URL must have a telemetry source")
        if name == "candidate" and any(arm["modes"][s] != "on" for s in required):
            raise ValueError("required actual overrides need candidate on mode")
    if plan["comparison"] == "aa":
        if any(
            file_hash(plan["arms"]["baseline"][key])
            != file_hash(plan["arms"]["candidate"][key])
            for key in ("binary", "config")
        ):
            raise ValueError("A/A requires identical binary and configuration bytes")
    if plan["comparison"] == "off_shadow":
        if (
            any(m != "off" for m in plan["arms"]["baseline"]["modes"].values())
            or not all(
                m in ("off", "shadow")
                for m in plan["arms"]["candidate"]["modes"].values()
            )
            or "shadow" not in plan["arms"]["candidate"]["modes"].values()
        ):
            raise ValueError(
                "off_shadow requires off baseline and at least one shadow candidate stage"
            )
    return plan, rows


def source_fingerprint(repo):
    names = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ]
        )
        .decode()
        .split("\0")
    )
    contents = {}
    for name in sorted(set(filter(None, names))):
        path = (Path(repo) / name).resolve()
        if not path.is_relative_to(Path(repo).resolve()):
            raise ValueError("source path escapes repository")
        contents[name] = file_hash(path) if path.is_file() else None
    return workload.digest(contents)


def snapshot_arm(recorder, name, arm):
    before = source_fingerprint(arm["source_repo"])
    root = recorder.directory / "sources" / name
    root.mkdir(parents=True)

    def git(label, *args):
        code, output = recorder.run(
            f"{name}-{label}", ["git", "-C", arm["source_repo"], *args], 60
        )
        if code:
            raise ValueError(f"{name} source provenance failed: {label}")
        return output

    revision = git("head", "rev-parse", "--verify", "HEAD^{commit}").strip()
    patch = git("patch", "diff", "--binary", "HEAD")
    (root / "working-tree.patch").write_text(patch, encoding="utf-8")
    git("status", "status", "--porcelain=v1", "-uall")
    untracked = git("untracked", "ls-files", "--others", "--exclude-standard", "-z")
    hashes = {}
    allowed = ("src/", "tests/", "scripts/benchmarks/pd_pressure/", "docs/")
    for relative in filter(None, untracked.split("\0")):
        if not relative.startswith(allowed) and relative not in (
            "Cargo.toml",
            "Cargo.lock",
            "build.rs",
        ):
            raise ValueError(
                f"unarchived untracked file in arm source: {relative}; use an isolated source tree"
            )
        source = (Path(arm["source_repo"]) / relative).resolve()
        if (
            not source.is_relative_to(Path(arm["source_repo"]).resolve())
            or not source.is_file()
        ):
            raise ValueError("source snapshot escapes repository or is not a file")
        dest = root / "untracked" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        hashes[relative] = file_hash(dest)
    shutil.copyfile(arm["config"], root / "router-config.json")
    stat = Path(arm["binary"]).stat()
    if source_fingerprint(arm["source_repo"]) != before:
        raise ValueError("source changed while capturing evidence")
    return {
        "revision": revision,
        "source_fingerprint": before,
        "patch_sha256": file_hash(root / "working-tree.patch"),
        "untracked_sha256": hashes,
        "binary_sha256": file_hash(arm["binary"]),
        "binary_size_bytes": stat.st_size,
        "binary_mtime_ns": stat.st_mtime_ns,
        "config_sha256": file_hash(root / "router-config.json"),
        "remote_mapping_attested": False,
    }


def counter(text, selector):
    found = []
    pattern = re.compile(
        r"^" + re.escape(selector["metric"]) + r"(?:\{(.*?)\})?\s+(\S+)(?:\s+\S+)?$"
    )
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if not match:
            continue
        labels = {}
        raw = match[1] or ""
        pos = 0
        for item in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)=("(?:\\.|[^"\\])*")', raw):
            if raw[pos : item.start()].strip(", ") or item[1] in labels:
                raise ValueError("ambiguous counter labels")
            labels[item[1]] = json.loads(item[2])
            pos = item.end()
        if raw[pos:].strip(", "):
            raise ValueError("invalid counter labels")
        if all(labels.get(k) == v for k, v in selector["labels"].items()):
            found.append(workload.number(float(match[2]), "counter"))
    if len(found) != 1:
        raise ValueError("override counter selector must match exactly one series")
    return found[0]


class Sampler:
    def __init__(self, directory, sources, interval):
        self.directory, self.sources, self.interval = directory, sources, interval
        directory.mkdir()
        self.samples, self.errors, self.first, self.last = [], [], {}, {}
        self.start = time.monotonic()
        self.stop = asyncio.Event()

    async def sample(self, session):
        for source in self.sources:
            item = {
                "source": source["name"],
                "stage": source["stage"],
                "started_ms": (time.monotonic() - self.start) * 1000,
            }
            try:
                async with session.get(
                    source["url"], allow_redirects=False
                ) as response:
                    item["http_status"] = response.status
                    if response.status != 200:
                        raise ValueError("telemetry HTTP failure")
                    raw = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        raw.extend(chunk)
                        if len(raw) > 1024 * 1024:
                            raise ValueError("telemetry response too large")
                    text = raw.decode("utf-8")
                    file = (
                        self.directory
                        / f"{len(self.samples):05d}-{source['name']}.prom"
                    )
                    file.write_bytes(raw)
                    item.update(raw_file=file.name, sha256=file_hash(file))
                    self.first.setdefault(source["name"], text)
                    self.last[source["name"]] = text
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ValueError,
                UnicodeError,
            ) as exc:
                item["error"] = type(exc).__name__
                self.errors.append(copy.deepcopy(item))
            item["ended_ms"] = (time.monotonic() - self.start) * 1000
            self.samples.append(item)
            write_json(self.directory / "samples.json", self.samples)

    async def loop(self, session):
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), self.interval)
            except asyncio.TimeoutError:
                await self.sample(session)


async def run_arm(recorder, plan, rows, iteration, name):
    arm = plan["arms"][name]
    directory = recorder.directory / f"round-{iteration}-{name}"
    directory.mkdir()
    run = {"iteration": iteration, "arm": name, "complete": False, "processes": []}
    recorder.data.setdefault("rounds", []).append(run)
    recorder.save()
    owned, streams, failure = [], [], None

    def expand(argv):
        return [
            token.replace("{artifact_dir}", str(directory))
            .replace("{arm}", name)
            .replace("{iteration}", str(iteration))
            for token in argv
        ]

    async def hooks(hook):
        for index, argv in enumerate(arm[hook]):
            code, _ = await asyncio.to_thread(
                recorder.run, f"{iteration}-{name}-{hook}-{index}", expand(argv)
            )
            if code:
                raise ValueError(f"{name} {hook} hook exited {code}")

    try:
        for index, argv in enumerate(arm["start"]):
            argv = expand(argv)
            log = directory / f"process-{index}.log"
            stream = log.open("wb")
            streams.append(stream)
            proc = subprocess.Popen(
                argv,
                cwd=recorder.repo,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            owned.append(proc)
            run["processes"].append(
                {
                    "argv": argv,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "pid": proc.pid,
                    "log": str(log.relative_to(recorder.directory)),
                }
            )
            recorder.save()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=2), trust_env=False
        ) as session:
            deadline = time.monotonic() + plan["ready_timeout_s"]
            for url in arm["ready_urls"]:
                while True:
                    if any(p.poll() is not None for p in owned):
                        raise ValueError(
                            "owned deployment process exited before readiness"
                        )
                    try:
                        async with session.get(url, allow_redirects=False) as response:
                            if response.status == 200:
                                break
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise ValueError("deployment readiness timed out")
                    await asyncio.sleep(0.05)
            await hooks("reset")
            await hooks("warmup")
            sampler = Sampler(
                directory / "workers", arm["telemetry"], plan["sample_interval_s"]
            )
            await sampler.sample(session)
            sampling = asyncio.create_task(sampler.loop(session))
            try:
                result = await asyncio.wait_for(
                    replay(
                        rows,
                        arm["url"],
                        directory / "requests.jsonl",
                        concurrency=plan["concurrency"],
                        timeout_s=plan["request_timeout_s"],
                    ),
                    timeout=recorder.timeout,
                )
            finally:
                sampler.stop.set()
                await sampling
                await sampler.sample(session)
            if any(p.poll() is not None for p in owned):
                raise ValueError("owned process exited during measurement")
            overrides = {}
            for stage, selector in arm.get("override_counters", {}).items():
                source = selector["source"]
                try:
                    delta = counter(sampler.last[source], selector) - counter(
                        sampler.first[source], selector
                    )
                    if delta < 0:
                        raise ValueError("counter reset during measured interval")
                    overrides[stage] = delta
                except (ValueError, KeyError) as exc:
                    sampler.errors.append({"stage": stage, "error": str(exc)})
            run.update(
                result,
                records=str(
                    (directory / "requests.jsonl").relative_to(recorder.directory)
                ),
                records_sha256=file_hash(directory / "requests.jsonl"),
                overrides=overrides,
                telemetry_samples=str(
                    (sampler.directory / "samples.json").relative_to(recorder.directory)
                ),
                telemetry_sha256=file_hash(sampler.directory / "samples.json"),
                telemetry_errors=sampler.errors,
            )
    except BaseException as exc:
        failure = exc
        run["failure"] = {"type": type(exc).__name__, "detail": str(exc)}
        raise
    finally:
        # Cleanup only process groups started by this run, even on timeout/interrupt.
        try:
            await hooks("stop")
        except BaseException as exc:
            run["cleanup_error"] = {"type": type(exc).__name__, "detail": str(exc)}
            if failure is None:
                raise
        finally:
            for proc, record in reversed(list(zip(owned, run["processes"]))):
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.to_thread(proc.wait, timeout=3)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.to_thread(proc.wait)
                record["exit_code"] = proc.returncode
                record["cleaned_up"] = True
            for stream in streams:
                stream.close()
            recorder.save()
    run["complete"] = True
    recorder.save()


def execute(recorder, plan_path, url_validator):
    plan, rows = load_plan(plan_path, url_validator)
    write_json(recorder.directory / "plan.json", plan)
    workload.write(recorder.directory / "trace.jsonl", rows)
    write_json(
        recorder.directory / "topology.json",
        {a: plan["arms"][a]["topology"] for a in ARMS},
    )
    recorder.data.update(
        plan_sha256=workload.digest(plan),
        trace_sha256=workload.digest(rows),
        aiohttp_version=aiohttp.__version__,
        evidence_level=plan["evidence_level"],
    )
    sources = {
        name: snapshot_arm(recorder, name, arm) for name, arm in plan["arms"].items()
    }
    recorder.data["arm_sources"] = sources
    recorder.save()
    if plan["comparison"] == "aa" and any(
        sources["baseline"][key] != sources["candidate"][key]
        for key in ("revision", "patch_sha256", "untracked_sha256")
    ):
        raise ValueError("A/A source trees differ")
    for iteration in range(1, plan["repeats"] + 1):
        for name in ARMS if iteration % 2 else tuple(reversed(ARMS)):
            print(f"PR3 round {iteration}/{plan['repeats']} {name}", flush=True)
            asyncio.run(run_arm(recorder, plan, rows, iteration, name))
    for name, arm in plan["arms"].items():
        if (
            source_fingerprint(arm["source_repo"])
            != sources[name]["source_fingerprint"]
        ):
            raise ValueError("source changed during experiment")
        if any(
            file_hash(arm[key]) != sources[name][key + "_sha256"]
            for key in ("binary", "config")
        ):
            raise ValueError("binary or configuration changed during experiment")
    result = report.build(plan, rows, recorder.data["rounds"], recorder.directory)
    report.write(recorder.directory, result)
    recorder.data["performance_accepted"] = False
    return recorder.finish(
        result["status"],
        0 if not result["violations"] else 1,
        "See report.json, rounds.csv and raw evidence; no performance acceptance",
    )


def regenerate(recorder, source):
    source = source.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    plan = json.loads((source / "plan.json").read_text())
    rows = workload.load(source / "trace.jsonl")
    if manifest.get("plan_sha256") != workload.digest(plan) or manifest.get(
        "trace_sha256"
    ) != workload.digest(rows):
        raise ValueError("frozen plan/trace hash mismatch")
    if manifest.get("status") not in ("functional_valid", "invalid_measurement"):
        raise ValueError("source experiment did not finish")
    result = report.build(plan, rows, manifest["rounds"], source)
    report.write(recorder.directory, result)
    recorder.data["source_evidence"] = str(source)
    return recorder.finish(result["status"], 0 if not result["violations"] else 1)

"""PR3 functional contracts: real local HTTP, deterministic statistics, and faults."""

import asyncio
import contextlib
import copy
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from aiohttp import web

sys.path.insert(0, str(Path(__file__).parents[1]))
import experiment
import mock_pd
import replay
import report
import run
import workload


@contextlib.asynccontextmanager
async def server(app):
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    site = web.SockSite(runner, sock)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


class Workloads(unittest.TestCase):
    def test_fixture_plan_does_not_overwrite_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "off.json"
            sentinel.write_text("keep existing configuration")
            with self.assertRaises(FileExistsError):
                mock_pd.fixture_plan(Path(tmp), Path(tmp))
            self.assertEqual(sentinel.read_text(), "keep existing configuration")

    def test_seed_reproduces_body_prefix_and_arrivals(self):
        a = workload.generate(8, 4, 17, "model", arrival="poisson")
        self.assertEqual(a, workload.generate(8, 4, 17, "model", arrival="poisson"))
        self.assertNotEqual(a, workload.generate(8, 4, 18, "model", arrival="poisson"))
        self.assertEqual(a[0]["prefix_group"], a[4]["prefix_group"])
        self.assertEqual({r["length_group"] for r in a}, {"short", "long"})
        self.assertEqual(a[0]["body"]["max_tokens"], 16)
        self.assertNotIn("input_tokens", a[0])

    def test_constant_and_burst_arrivals_are_explicit(self):
        self.assertEqual(
            [r["offset_s"] for r in workload.generate(3, 2, 0, "m")], [0, 0.5, 1]
        )
        self.assertEqual(
            [r["offset_s"] for r in workload.generate(3, 2, 0, "m", arrival="burst")],
            [0, 0, 0],
        )

    def test_rejects_empty_duplicate_reordered_nonfinite_and_nonstreaming(self):
        rows = workload.generate(2, 1, 0, "m")
        cases = [[], [rows[0], rows[0]], list(reversed(rows))]
        for field, value in (("offset_s", float("nan")), ("id", "")):
            bad = copy.deepcopy(rows)
            bad[0][field] = value
            cases.append(bad)
        for field, value in (
            ("stream", False),
            ("max_tokens", True),
            ("n", 2),
            ("messages", []),
        ):
            bad = copy.deepcopy(rows)
            bad[0]["body"][field] = value
            cases.append(bad)
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                workload.validate(bad)

    def test_trace_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            rows = workload.generate(2, 1, 0, "m")
            workload.write(path, rows)
            self.assertEqual(workload.load(path), rows)
            with self.assertRaises(FileExistsError):
                workload.write(path, rows)


class Replay(unittest.IsolatedAsyncioTestCase):
    async def exercise(
        self,
        fault=None,
        *,
        delay=0.001,
        count=4,
        concurrency=4,
        timeout=1,
        arrival="burst",
    ):
        rows = workload.generate(count, 500, 0, "pd-tool-mock", arrival=arrival)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "requests.jsonl"
            async with server(mock_pd.application({}, fault, delay)) as url:
                result = await replay.replay(
                    rows, url, output, concurrency=concurrency, timeout_s=timeout
                )
            records = [json.loads(line) for line in output.read_text().splitlines()]
        return result, records

    async def test_first_content_not_role_chunk_and_all_requests_complete(self):
        result, records = await self.exercise(delay=0.01)
        self.assertEqual(result["completed"], 4)
        for r in records:
            self.assertEqual(r["content_chunks"], 2)
            self.assertGreaterEqual(
                r["ttft_ms"], 15
            )  # Two fixture delays precede first content.
            self.assertEqual(len(r["inter_chunk_ms"]), 1)
            self.assertEqual(r["usage"]["completion_tokens"], 2)
            self.assertAlmostEqual(r["ttft_ms"], r["first_content_ms"] - r["sent_ms"])

    async def test_partial_stream_retains_first_content_but_fails_completion(self):
        result, records = await self.exercise("partial")
        self.assertEqual(result["completed"], 0)
        self.assertTrue(
            all(
                r["status"] == "stream_error" and r["ttft_ms"] is not None
                for r in records
            )
        )

    async def test_role_only_and_malformed_streams_are_not_success(self):
        for fault in ("role_only", "malformed"):
            with self.subTest(fault=fault):
                result, records = await self.exercise(fault)
                self.assertEqual(result["completed"], 0)
                self.assertTrue(all(r["first_content_ms"] is None for r in records))

    async def test_http_failure_retains_full_request_denominator(self):
        result, records = await self.exercise("http_error")
        self.assertEqual(result["requests"], 4)
        self.assertEqual(result["completed"], 0)
        self.assertTrue(
            all(
                r["http_status"] == 503 and r["status"] == "http_error" for r in records
            )
        )

    async def test_backlog_does_not_drop_or_reschedule_arrivals(self):
        result, records = await self.exercise(count=6, concurrency=1, delay=0.02)
        self.assertEqual(result["completed"], 6)
        self.assertGreaterEqual(result["peak_client_backlog"], 4)
        self.assertTrue(all(r["planned_ms"] == 0 for r in records))
        self.assertGreater(max(r["client_lag_ms"] for r in records), 150)

    async def test_total_request_timeout_is_recorded(self):
        result, records = await self.exercise(delay=0.05, timeout=0.01)
        self.assertEqual(result["completed"], 0)
        self.assertTrue(all(r["status"] == "timeout" for r in records))

    async def test_redirect_is_not_followed(self):
        calls = []

        async def redirect(request):
            return web.Response(status=302, headers={"Location": "/redirected"})

        async def redirected(request):
            calls.append(True)
            return web.Response(text="unexpected")

        app = web.Application()
        app.add_routes(
            [
                web.post("/v1/chat/completions", redirect),
                web.get("/redirected", redirected),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack():
            async with server(app) as url:
                output = Path(tmp) / "r.jsonl"
                await replay.replay(
                    workload.generate(1, 1, 0, "m"),
                    url,
                    output,
                    concurrency=1,
                    timeout_s=1,
                )
                self.assertEqual(json.loads(output.read_text())["status"], "http_error")
        self.assertEqual(calls, [])

    async def test_cancel_preserves_unsent_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "r.jsonl"
            async with server(mock_pd.application({}, delay=0.1)) as url:
                task = asyncio.create_task(
                    replay.replay(
                        workload.generate(4, 1, 0, "m"),
                        url,
                        output,
                        concurrency=1,
                        timeout_s=2,
                    )
                )
                await asyncio.sleep(0.02)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 4)
            self.assertTrue(all(r["status"] == "cancelled" for r in records))


def statistics_fixture(root, repeats=3):
    rows = workload.generate(4, 1, 17, "m", arrival="burst")
    for row in rows:
        row["length_group"] = "fixed"
    plan = {
        "repeats": repeats,
        "seed": 17,
        "cache_state": "warm",
        "evidence_level": "mock",
        "backend_transport": "http",
        "comparison": "candidate",
        "require_overrides": [],
        "budgets": {
            "max_failure_rate": 0,
            "max_client_lag_ms": 10,
            "max_ttft_regression_pct": 5,
        },
    }
    runs = []
    for iteration in range(1, repeats + 1):
        for arm in (
            ("baseline", "candidate") if iteration % 2 else ("candidate", "baseline")
        ):
            records = []
            for i, row in enumerate(rows):
                ttft = (i + 1) * (10 if arm == "baseline" else 5)
                records.append(
                    {
                        "id": row["id"],
                        "request_sha256": workload.digest(row["body"]),
                        "planned_ms": 0,
                        "sent_ms": 1,
                        "client_lag_ms": 1,
                        "first_content_ms": 1 + ttft,
                        "ttft_ms": ttft,
                        "finished_ms": 10 + ttft,
                        "status": "completed",
                        "http_status": 200,
                    }
                )
            file = root / f"{iteration}-{arm}.jsonl"
            file.write_text("".join(json.dumps(r) + "\n" for r in records))
            raw = root / f"{iteration}-{arm}.prom"
            raw.write_text("counter 0\n")
            telemetry = root / f"{iteration}-{arm}-samples.json"
            telemetry.write_text(
                json.dumps(
                    [{"raw_file": raw.name, "sha256": experiment.file_hash(raw)}]
                )
            )
            runs.append(
                {
                    "iteration": iteration,
                    "arm": arm,
                    "complete": True,
                    "records": file.name,
                    "records_sha256": experiment.file_hash(file),
                    "wall_ms": 100,
                    "telemetry_samples": telemetry.name,
                    "telemetry_sha256": experiment.file_hash(telemetry),
                    "peak_client_backlog": 0,
                    "telemetry_errors": [],
                    "overrides": {"prefill": 0, "decode": 0},
                }
            )
    return plan, rows, runs


def edit_records(root, run, edit):
    file = root / run["records"]
    records = [
        json.loads(line) for line in file.read_text().splitlines() if line.strip()
    ]
    edit(records)
    file.write_text("".join(json.dumps(r) + "\n" for r in records))
    run["records_sha256"] = experiment.file_hash(file)


class Statistics(unittest.TestCase):
    def test_exact_p95_percentage_denominator_and_round_interval(self):
        self.assertAlmostEqual(report.percentile([10, 20, 30, 40], 95), 38.5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, rows, runs = statistics_fixture(root)
            result = report.build(plan, rows, runs, root)
            comparison = result["comparisons"][0]
            self.assertEqual(comparison["mean_improvement_ms"], 19.25)
            self.assertEqual(comparison["mean_improvement_pct"], 50)
            self.assertEqual(comparison["paired_bootstrap_95_pct"], [50, 50])
            self.assertEqual(comparison["order_means_pct"], {"AB": 50, "BA": 50})
            self.assertFalse(result["performance_accepted"])
            report.write(root, result)
            self.assertIn("not assessed", (root / "report.md").read_text())

    def test_two_rounds_have_no_confidence_interval(self):
        self.assertIsNone(report.paired_interval([1, 2], 17))

    def test_regression_intervals_allow_negative_improvements(self):
        self.assertEqual(report.paired_interval([-10, -10, -10], 17), [-10, -10])

    def test_changed_raw_metrics_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, rows, runs = statistics_fixture(root)
            (root / "1-baseline.prom").write_text("counter 99\n")
            with self.assertRaisesRegex(ValueError, "raw telemetry hash"):
                report.build(plan, rows, runs, root)

    def test_failures_after_first_content_do_not_disappear(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, rows, runs = statistics_fixture(root)
            edit_records(root, runs[0], lambda r: r[3].update(status="stream_error"))
            result = report.build(plan, rows, runs, root)
            first = result["rounds"][0]
            self.assertEqual(
                (
                    first["planned"],
                    first["completed"],
                    first["failed"],
                    first["ttft_samples"],
                ),
                (4, 3, 1, 4),
            )
            self.assertEqual(first["ttft_p95_ms"], 38.5)
            self.assertEqual(first["failure_rate"], 0.25)
            self.assertEqual(result["status"], "invalid_measurement")

    def test_no_first_content_is_not_a_zero_ttft_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, rows, runs = statistics_fixture(root)
            edit_records(
                root,
                runs[0],
                lambda r: r[0].update(
                    status="timeout", first_content_ms=None, ttft_ms=None
                ),
            )
            result = report.build(plan, rows, runs, root)
            self.assertEqual(result["rounds"][0]["ttft_samples"], 3)
            self.assertEqual(result["rounds"][0]["planned"], 4)

    def test_missing_extra_duplicate_or_reordered_rounds_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, rows, runs = statistics_fixture(root)
            for wrong in (
                runs[:-1],
                runs + [runs[0]],
                [runs[0]] * len(runs),
                list(reversed(runs)),
            ):
                with self.subTest(wrong=wrong), self.assertRaisesRegex(
                    ValueError, "rounds"
                ):
                    report.build(plan, rows, wrong, root)

    def test_zero_missing_duplicate_requests_rejected_even_with_new_hash(self):
        for mutate in (
            lambda r: r.clear(),
            lambda r: r.pop(),
            lambda r: r.__setitem__(1, r[0]),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan, rows, runs = statistics_fixture(root)
                edit_records(root, runs[0], mutate)
                with self.assertRaisesRegex(ValueError, "request records"):
                    report.build(plan, rows, runs, root)

    def test_changed_evidence_or_timestamp_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, rows, runs = statistics_fixture(root)
            path = root / runs[0]["records"]
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                report.build(plan, rows, runs, root)
            runs[0]["records_sha256"] = experiment.file_hash(path)
            edit_records(root, runs[0], lambda r: r[0].update(ttft_ms=999))
            with self.assertRaisesRegex(ValueError, "TTFT"):
                report.build(plan, rows, runs, root)

    def test_client_lag_and_telemetry_failure_invalidate_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, rows, runs = statistics_fixture(root)
            plan["budgets"]["max_client_lag_ms"] = 0
            runs[1]["telemetry_errors"] = [{"error": "missing"}]
            result = report.build(plan, rows, runs, root)
            self.assertTrue(any("fell behind" in v for v in result["violations"]))
            self.assertTrue(any("telemetry" in v for v in result["violations"]))

    def test_zero_override_allowed_for_shadow_but_not_required_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, rows, runs = statistics_fixture(root)
            self.assertEqual(
                report.build(plan, rows, runs, root)["status"], "functional_valid"
            )
            plan["require_overrides"] = ["prefill"]
            result = report.build(plan, rows, runs, root)
            self.assertTrue(any("not exercised" in v for v in result["violations"]))

    def test_nan_infinity_and_negative_samples_rejected(self):
        for sample in (float("nan"), float("inf"), -1, True):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                report.percentile([sample], 95)


class Configuration(unittest.TestCase):
    def test_source_patch_and_untracked_content_are_replayable_and_changes_detected(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            (repo / "src").mkdir(parents=True)
            tracked = repo / "src/lib.rs"
            tracked.write_text("before\n")
            for argv in (
                ["init", "-q"],
                ["add", "."],
                [
                    "-c",
                    "user.name=fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
            ):
                subprocess.run(
                    ["git", *argv], cwd=repo, check=True, capture_output=True
                )
            tracked.write_text("after\n")
            (repo / "src/new.rs").write_text("untracked content\n")
            config = root / "config.json"
            config.write_text("{}")
            recorder = run.Recorder(root / "evidence", repo, 5)
            arm = {
                "source_repo": str(repo),
                "config": str(config),
                "binary": sys.executable,
            }
            captured = experiment.snapshot_arm(recorder, "baseline", arm)
            archive = root / "evidence/sources/baseline"
            self.assertIn("+after", (archive / "working-tree.patch").read_text())
            self.assertEqual(
                (archive / "untracked/src/new.rs").read_text(), "untracked content\n"
            )
            self.assertEqual(
                experiment.source_fingerprint(repo), captured["source_fingerprint"]
            )
            tracked.write_text("changed during experiment\n")
            self.assertNotEqual(
                experiment.source_fingerprint(repo), captured["source_fingerprint"]
            )

    def test_serve_missing_required_flags_never_executes_a_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = [
                "serve",
                "--stage",
                "3",
                "--repo",
                tmp,
                "--base-ref",
                "HEAD",
                "--out-dir",
                str(Path(tmp) / "out"),
                "--vllm",
                sys.executable,
                "--model",
                "m",
                "--baseline-url",
                "http://localhost:1",
                "--candidate-url",
                "http://localhost:2",
                "--baseline-binary",
                sys.executable,
                "--candidate-binary",
                sys.executable,
            ]
            with mock.patch.object(run, "provenance"), mock.patch.object(
                run.Recorder,
                "run",
                side_effect=[(0, "usage: bench"), (0, "usage: serve --model")],
            ) as execute:
                self.assertEqual(run.main(args), 3)
            self.assertEqual(execute.call_count, 2)
            self.assertTrue(
                all("--help" in call.args[1] for call in execute.call_args_list)
            )

    def test_manifest_requires_explicit_modes_and_complete_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = mock_pd.fixture_plan(Path(tmp) / "input", Path(tmp))
            plan, rows = experiment.load_plan(path, run.base_url)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(plan["arms"]["baseline"]["topology"]), 4)
            invalid = copy.deepcopy(plan)
            invalid["arms"]["candidate"]["modes"]["prefill"] = "shadow"
            experiment.write_json(path, invalid)
            with self.assertRaisesRegex(ValueError, "modes disagree"):
                experiment.load_plan(path, run.base_url)

    def test_ambiguous_protocol_budgets_and_hooks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = mock_pd.fixture_plan(Path(tmp) / "input", Path(tmp))
            original = json.loads(path.read_text())
            for field, value in (
                ("backend_transport", "grpc"),
                ("repeats", 0),
                ("require_overrides", ["prefill"]),
                ("cache_state", "cold"),
                ("sample_interval_s", float("nan")),
            ):
                invalid = copy.deepcopy(original)
                invalid[field] = value
                path.write_text(json.dumps(invalid))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    experiment.load_plan(path, run.base_url)
            invalid = copy.deepcopy(original)
            invalid["arms"]["baseline"]["reset"] = "curl http://server/reset"
            experiment.write_json(path, invalid)
            with self.assertRaisesRegex(ValueError, "argv"):
                experiment.load_plan(path, run.base_url)

    def test_counter_selector_rejects_missing_duplicate_and_nonfinite(self):
        selector = {"metric": "counter", "labels": {"stage": "prefill"}}
        self.assertEqual(
            experiment.counter('counter{stage="prefill",extra="value"} 3\n', selector),
            3,
        )
        for text in (
            'counter{stage="decode"} 3',
            'counter{stage="prefill"} NaN',
            'counter{stage="prefill"} -1',
            'counter{stage="prefill"} 1\ncounter{stage="prefill"} 2',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                experiment.counter(text, selector)


@unittest.skipUnless(
    sys.platform.startswith("linux"), "managed lifecycle requires Linux/WSL"
)
class Lifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_start_ready_reset_warm_sample_stop_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = mock_pd.fixture_plan(root / "input", root, count=4)
            plan, rows = experiment.load_plan(path, run.base_url)
            recorder = run.Recorder(root / "out", root, 5)
            await experiment.run_arm(recorder, plan, rows, 1, "baseline")
            saved = recorder.data["rounds"][0]
            self.assertTrue(saved["complete"])
            self.assertTrue(all(p["cleaned_up"] for p in saved["processes"]))
            audit = json.loads((root / "out/round-1-baseline/audit.json").read_text())
            self.assertEqual(
                (audit["requests"], audit["completed"], audit["handoffs"]), (5, 5, 5)
            )
            self.assertEqual(saved["overrides"], {"prefill": 0, "decode": 0})
            self.assertEqual(saved["telemetry_errors"], [])

    async def test_early_process_exit_keeps_original_error_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = mock_pd.fixture_plan(root / "input", root, count=2)
            plan, rows = experiment.load_plan(path, run.base_url)
            plan["arms"]["baseline"]["start"] = [
                [sys.executable, "-c", "raise SystemExit(7)"]
            ]
            recorder = run.Recorder(root / "out", root, 2)
            with self.assertRaisesRegex(ValueError, "before readiness"):
                await experiment.run_arm(recorder, plan, rows, 1, "baseline")
            saved = recorder.data["rounds"][0]
            self.assertFalse(saved["complete"])
            self.assertEqual(saved["processes"][0]["exit_code"], 7)
            self.assertTrue(saved["processes"][0]["cleaned_up"])
            self.assertIn("cleanup_error", saved)

    async def test_telemetry_fault_is_preserved_without_discarding_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = mock_pd.fixture_plan(
                root / "input", root, fault="metrics_error", count=2
            )
            plan, rows = experiment.load_plan(path, run.base_url)
            recorder = run.Recorder(root / "out", root, 5)
            await experiment.run_arm(recorder, plan, rows, 1, "candidate")
            saved = recorder.data["rounds"][0]
            self.assertEqual(saved["completed"], 2)
            self.assertTrue(saved["telemetry_errors"])
            self.assertTrue(saved["processes"][0]["cleaned_up"])


if __name__ == "__main__":
    unittest.main()

"""Validate finite-capacity pressure, queue/release semantics and experiment inputs."""

import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capacity_sim
import experiment
import high_pressure
import run
import workload


class CapacityTests(unittest.IsolatedAsyncioTestCase):
    async def test_pressure_is_allocated_capacity_and_waiting_does_not_allocate(self):
        pool = capacity_sim.Pool()
        await pool.set_background(116)
        await pool.acquire(8)
        blocked = asyncio.create_task(pool.acquire(8))
        await asyncio.sleep(0)
        self.assertEqual((pool.used, pool.active, pool.waiting), (8, 1, 1))
        self.assertEqual((pool.used + pool.background) / pool.capacity, 124 / 128)
        await pool.release(8)
        await asyncio.wait_for(blocked, 1)
        self.assertEqual((pool.used, pool.active, pool.waiting), (8, 1, 0))
        await pool.release(8)
        self.assertEqual((pool.used, pool.active, pool.waiting), (0, 0, 0))
        self.assertLessEqual(pool.peak_used, pool.capacity)

    async def test_cancelled_waiter_and_slot_limit_do_not_leak(self):
        pool = capacity_sim.Pool(slots=1)
        await pool.acquire(1)
        task = asyncio.create_task(pool.acquire(1))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual((pool.used, pool.active, pool.waiting), (1, 1, 0))
        await pool.release(1)
        with self.assertRaises(ValueError):
            await pool.acquire(129)

    async def test_profile_recovery_releases_background_but_keeps_live_allocations(
        self,
    ):
        sim = capacity_sim.Simulation("http://127.0.0.1:1", "epoch")
        await sim.profile_start("recovery")
        await sim.pools["p0"].acquire(8)
        await sim.phase_task
        self.assertEqual(sim.pools["p0"].background, 16)
        self.assertEqual(sim.pools["p0"].used, 8)
        self.assertEqual(sim.events[-1]["event"], "background_released")
        await sim.pools["p0"].release(8)

    async def test_invalid_identity_does_not_change_capacity_or_health(self):
        sim = capacity_sim.Simulation("http://127.0.0.1:1", "epoch")
        await sim.profile_start("signal_fault")
        request = SimpleNamespace(match_info={"name": "p1"})
        sim.identity_fault = True
        response = await sim.metrics(request)
        self.assertIn('instance_epoch="wrong-epoch"', response.text)
        self.assertIn(" 0.125\n", response.text)
        self.assertEqual(sim.pools["p1"].background, 16)
        await sim.profile_start("low")
        await sim.phase_task
        self.assertFalse(sim.identity_fault)

    async def test_busy_reset_is_rejected(self):
        sim = capacity_sim.Simulation("http://127.0.0.1:1", "epoch")
        await sim.pools["p0"].acquire(8)
        with self.assertRaisesRegex(ValueError, "busy"):
            await sim.profile_start("low")
        await sim.pools["p0"].release(8)


class PlanTests(unittest.TestCase):
    def test_actual_router_occupancy_and_decision_counts_are_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "workers").mkdir()
            gauges = "".join(
                f'{name}{{worker="p0",rank="aggregate"}} 0\n'
                for name in (
                    "vllm_router_pd_prefill_occupancy",
                    "vllm_router_pd_pending_decode_requests",
                    "vllm_router_pd_active_decode_requests",
                )
            )
            counter = 'vllm_router_pd_pressure_decisions_total{stage="prefill",mode="on",reason="override"} '
            snapshots = []
            for i, count in enumerate([0, 4]):
                path = root / "workers" / f"{i}.prom"
                path.write_text(gauges + counter + str(count) + "\n")
                snapshots.append(
                    {
                        "source": "router",
                        "raw_file": path.name,
                        "sha256": experiment.file_hash(path),
                    }
                )
            (root / "workers/samples.json").write_text(json.dumps(snapshots))
            modes = {"prefill": "on", "decode": "off"}
            self.assertEqual(
                high_pressure.router_audit(root, modes, 4)["prefill"]["override"], 4
            )
            with self.assertRaisesRegex(ValueError, "decisions"):
                high_pressure.router_audit(root, modes, 5)
            path.write_text(
                path.read_text().replace('rank="aggregate"} 0', 'rank="aggregate"} 1')
            )
            snapshots[-1]["sha256"] = experiment.file_hash(path)
            (root / "workers/samples.json").write_text(json.dumps(snapshots))
            with self.assertRaisesRegex(ValueError, "occupancy"):
                high_pressure.router_audit(root, modes, 4)

    def test_profiles_use_identical_input_and_service_parameters_across_arms(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(
                num_prompts=16,
                request_rate=32,
                seed=17,
                repeat=2,
                stress_concurrency=32,
                router_binary=Path(sys.executable),
                repo=root,
            )
            path = high_pressure.create_plan(
                root / "plan",
                args,
                "http://127.0.0.1:8000",
                "epoch",
                8001,
                8002,
                "both_hot",
                "both",
            )
            plan, rows = experiment.load_plan(path, run.base_url)
            self.assertEqual(plan["require_overrides"], ["prefill", "decode"])
            self.assertEqual(plan["evidence_level"], "mock")
            a, b = plan["arms"]["baseline"], plan["arms"]["candidate"]
            self.assertEqual(a["warmup"], b["warmup"])
            self.assertEqual(a["topology"], b["topology"])
            self.assertEqual(a["binary"], b["binary"])
            self.assertEqual(a["modes"], {"prefill": "off", "decode": "off"})
            self.assertEqual(b["modes"], {"prefill": "on", "decode": "on"})
            self.assertEqual(rows, high_pressure.trace(16, 32, 17))
            self.assertEqual([r["offset_s"] for r in rows[:8]], [0] * 8)
            self.assertTrue(all(r["id"] == r["body"]["user"] for r in rows))

    def test_audit_reconciliation_rejects_missing_backend_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workload.write(root / "trace.jsonl", high_pressure.trace(2, 32, 17))
            (root / "manifest.json").write_text(
                json.dumps({"rounds": [{"iteration": 1, "arm": "baseline"}]})
            )
            (root / "plan.json").write_text("{}")
            target = root / "round-1-baseline"
            target.mkdir()
            (target / "backend-audit.json").write_text(
                json.dumps({"records": [], "pools": {}})
            )
            with self.assertRaisesRegex(ValueError, "missing requests"):
                high_pressure.reconcile(root)


if __name__ == "__main__":
    unittest.main()

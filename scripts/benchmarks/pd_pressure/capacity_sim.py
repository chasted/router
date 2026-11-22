"""Finite-capacity HTTP PD engines for the REAL router; no model/accelerator.

All arms run the same service rules. KV pressure is allocated simulation blocks,
not an injected arm-specific response delay or a claim about physical vLLM KV.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from aiohttp import web
import aiohttp

MODEL = "capacity-sim"
NAMES = ("p0", "p1", "d0", "d1")
PROFILES = ("low", "p_hot", "d_hot", "both_hot", "all_hot", "recovery", "signal_fault")


def words(body):
    if "messages" in body:
        return " ".join(m["content"] for m in body["messages"]).split()
    return str(body.get("prompt", "")).split()


def tokens(body):
    return (
        [1]
        + [
            int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "little")
            for w in words(body)
        ]
        + [2]
    )


class Pool:
    def __init__(self, capacity=128, slots=8):
        self.capacity, self.slots = capacity, slots
        self.background = self.used = self.active = self.waiting = 0
        self.peak_used = self.peak_waiting = 0
        self.condition = asyncio.Condition()

    async def acquire(self, blocks):
        if not 0 < blocks <= self.capacity:
            raise ValueError("request exceeds simulation capacity")
        async with self.condition:
            self.waiting += 1
            self.peak_waiting = max(self.peak_waiting, self.waiting)
            try:
                await self.condition.wait_for(
                    lambda: (
                        self.used + self.background + blocks <= self.capacity
                        and self.active < self.slots
                    )
                )
                self.used += blocks
                self.active += 1
                self.peak_used = max(self.peak_used, self.used + self.background)
            finally:
                self.waiting -= 1

    async def release(self, blocks):
        async with self.condition:
            self.used -= blocks
            self.active -= 1
            assert self.used >= 0 and self.active >= 0
            self.condition.notify_all()

    async def set_background(self, blocks):
        async with self.condition:
            if not 0 <= blocks <= self.capacity - self.used:
                raise ValueError("background allocation exceeds available capacity")
            self.background = blocks
            self.peak_used = max(self.peak_used, self.used + blocks)
            self.condition.notify_all()

    def snapshot(self):
        return {
            k: getattr(self, k)
            for k in (
                "capacity",
                "slots",
                "background",
                "used",
                "active",
                "waiting",
                "peak_used",
                "peak_waiting",
            )
        }


class Simulation:
    def __init__(self, base, epoch):
        self.base, self.epoch = base, epoch
        self.pools = {name: Pool() for name in NAMES}
        self.cache = {name: set() for name in NAMES}
        self.records, self.handoffs, self.events = [], {}, []
        self.profile = "low"
        self.started = time.monotonic()
        self.phase_task = None
        self.identity_fault = False

    def elapsed(self):
        return (time.monotonic() - self.started) * 1000

    async def profile_start(self, profile):
        if profile not in PROFILES:
            raise ValueError("unknown profile")
        if any(p.active or p.waiting for p in self.pools.values()):
            raise ValueError("cannot reset a busy simulation")
        if self.phase_task:
            self.phase_task.cancel()
            await asyncio.gather(self.phase_task, return_exceptions=True)
        self.profile, self.started = profile, time.monotonic()
        self.records, self.handoffs, self.events = [], {}, []
        self.identity_fault = False
        hot = {
            "low": (),
            "p_hot": ("p0",),
            "d_hot": ("d0",),
            "both_hot": ("p0", "d0"),
            "all_hot": NAMES,
            "recovery": ("p0", "d0"),
            "signal_fault": ("p0", "d0"),
        }[profile]
        for name, pool in self.pools.items():
            pool.peak_used = pool.peak_waiting = 0
            await pool.set_background(116 if name in hot else 16)
        self.events.append(
            {"at_ms": self.elapsed(), "event": "profile_start", "profile": profile}
        )

        async def phases():
            if profile == "recovery":
                await asyncio.sleep(1)
                for pool in self.pools.values():
                    await pool.set_background(16)
                self.events.append(
                    {"at_ms": self.elapsed(), "event": "background_released"}
                )
            if profile == "signal_fault":
                await asyncio.sleep(0.6)
                self.identity_fault = True
                self.events.append(
                    {"at_ms": self.elapsed(), "event": "p1_identity_invalid"}
                )
                await asyncio.sleep(0.8)
                self.identity_fault = False
                self.events.append(
                    {"at_ms": self.elapsed(), "event": "p1_identity_recovered"}
                )

        self.phase_task = asyncio.create_task(phases())

    async def metrics(self, request):
        name = request.match_info["name"]
        pool, stage = self.pools[name], "prefill" if name[0] == "p" else "decode"
        labels = f'engine="{name}",model_name="{MODEL}"'
        epoch = "wrong-epoch" if self.identity_fault and name == "p1" else self.epoch
        text = (
            f"vllm:kv_cache_usage_perc{{{labels}}} {(pool.used + pool.background) / pool.capacity}\n"
            f'router_kv_identity{{{labels},stage="{stage}",endpoint="{self.base}/{name}",dp_rank="aggregate",engine_id="{name}",instance_epoch="{epoch}",layout_fingerprint="sim-layout-128"}} 1\n'
        )
        for field, value in pool.snapshot().items():
            text += f"sim_{field}{{{labels}}} {value}\n"
        return web.Response(text=text)

    async def generate(self, request):
        name = request.match_info["name"]
        body = await request.json()
        identifier = body.get("user") or request.headers.get("x-request-id")
        text = words(body)
        pool = self.pools[name]
        blocks = 2 + math.ceil(len(text) / 64) + 1
        record = {
            "id": identifier,
            "worker": name,
            "arrived_ms": self.elapsed(),
            "blocks": blocks,
            "input_words": len(text),
            "status": "queued",
            "first_content_ms": None,
        }
        self.records.append(record)
        acquired = False
        try:
            await pool.acquire(blocks)
            acquired = True
            record.update(started_ms=self.elapsed(), status="running")
            record["queue_ms"] = record["started_ms"] - record["arrived_ms"]
            if name[0] == "p":
                prefix = text[0] if text else "empty"
                hit = min(32, len(text)) if prefix in self.cache[name] else 0
                service_s = 0.03 + (len(text) - hit) * 0.0005
                record.update(cache_hit_words=hit, service_s=service_s)
                await asyncio.sleep(service_s)
                self.cache[name].add(prefix)
                self.handoffs[identifier] = name
                record["status"] = "completed"
                return web.json_response(
                    {
                        "id": identifier,
                        "model": MODEL,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": ""},
                                "finish_reason": "length",
                            }
                        ],
                        "kv_transfer_params": {
                            "remote_engine_id": name,
                            "remote_block_ids": [[7]],
                            "remote_host": "127.0.0.1",
                            "remote_port": 5600,
                            "tp_size": 1,
                            "do_remote_prefill": True,
                            "do_remote_decode": False,
                        },
                    }
                )
            source = body.get("kv_transfer_params", {}).get("remote_engine_id")
            if source != self.handoffs.get(identifier):
                raise ValueError("decode received an incorrect prefill handoff")
            record["prefill_worker"] = source
            await asyncio.sleep(0.005 + len(text) * 0.00002)
            output = body.get("max_tokens", 16)
            if not body.get("stream"):
                await asyncio.sleep(output * 0.01)
                record.update(status="completed", first_content_ms=self.elapsed())
                return web.json_response(
                    {
                        "model": MODEL,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "length",
                            }
                        ],
                    }
                )
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for index in range(output):
                await asyncio.sleep(0.01)
                await response.write(
                    (
                        "data: "
                        + json.dumps(
                            {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": "x"},
                                        "finish_reason": None,
                                    }
                                ]
                            }
                        )
                        + "\n\n"
                    ).encode()
                )
                if index == 0:
                    record["first_content_ms"] = self.elapsed()
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            record["status"] = "completed"
            return response
        except BaseException as exc:
            record.update(status="failed", error=type(exc).__name__)
            raise
        finally:
            if acquired:
                await pool.release(blocks)
            record["finished_ms"] = self.elapsed()

    async def control(self, request):
        body = await request.json()
        if body["action"] == "reset":
            await self.profile_start("low")
            self.cache = {name: set() for name in NAMES}
        elif body["action"] == "load":
            await self.profile_start(body["profile"])
        else:
            raise web.HTTPBadRequest(text="unknown action")
        return web.json_response({"ok": True})

    async def audit(self, request):
        return web.json_response(
            {
                "profile": self.profile,
                "epoch": self.epoch,
                "events": self.events,
                "records": self.records,
                "pools": {n: p.snapshot() for n, p in self.pools.items()},
                "dispatches": dict(Counter(r["worker"] for r in self.records)),
            }
        )


def application(base, epoch):
    sim = Simulation(base, epoch)
    app = web.Application()

    async def health(request):
        return web.Response(text="ok")

    async def models(request):
        return web.json_response({"data": [{"id": MODEL, "parent": None}]})

    async def model_info(request):
        return web.json_response({"model_path": MODEL, "is_generation": True})

    async def server_info(request):
        return web.json_response(
            {
                "model_id": MODEL,
                "instance_id": request.match_info["name"],
                "block_size": 16,
                "vllm_config": {
                    "model_config": {"model": MODEL, "max_model_len": 4096},
                    "cache_config": {"block_size": 16},
                    "parallel_config": {
                        "tensor_parallel_size": 1,
                        "data_parallel_size": 1,
                    },
                },
            }
        )

    async def tokenize(request):
        ids = tokens(await request.json())
        return web.json_response(
            {"tokens": ids, "count": len(ids), "max_model_len": 4096}
        )

    async def empty(request):
        return web.json_response({})

    app.router.add_get("/health", health)
    app.router.add_post("/control", sim.control)
    app.router.add_get("/audit", sim.audit)
    for path, handler in (
        ("health", health),
        ("v1/models", models),
        ("get_model_info", model_info),
        ("get_server_info", server_info),
        ("kv_event_sources", empty),
        ("metrics", empty),
        ("pressure-metrics", sim.metrics),
    ):
        app.router.add_get("/{name}/" + path, handler)
    app.router.add_post("/{name}/tokenize", tokenize)
    app.router.add_post("/{name}/v1/chat/completions", sim.generate)

    async def close(app):
        if sim.phase_task:
            sim.phase_task.cancel()
            await asyncio.gather(sim.phase_task, return_exceptions=True)

    app.on_cleanup.append(close)
    return app, sim


async def control(args):
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30), trust_env=False
    ) as client:
        if args.action == "reset":
            async with client.post(
                args.base + "/control", json={"action": "reset"}
            ) as response:
                response.raise_for_status()
        elif args.action == "prime":
            import workload

            for row in workload.load(args.trace)[:8]:
                body = dict(row["body"], stream=False, user="warm-" + row["id"])
                async with client.post(
                    args.router + "/v1/chat/completions", json=body
                ) as response:
                    if response.status != 200:
                        raise ValueError(
                            f"warmup failed {response.status}: {(await response.text())[:500]}"
                        )
                    await response.read()
            async with client.post(
                args.base + "/control", json={"action": "load", "profile": args.profile}
            ) as response:
                response.raise_for_status()
            # Polling interval is 20 ms; the recorded profile start includes this
            # explicit settling period. Actual observations remain in telemetry.
            await asyncio.sleep(0.15)
        elif args.action == "audit":
            async with client.get(args.base + "/audit") as response:
                response.raise_for_status()
                data = await response.json()
            Path(args.output).write_text(json.dumps(data, indent=2) + "\n")
            if any(
                p["active"] or p["waiting"] or p["used"] for p in data["pools"].values()
            ):
                raise ValueError("simulation foreground resources did not drain")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("serve", "reset", "prime", "audit"))
    cli.add_argument("--base", required=True)
    cli.add_argument("--epoch", default="simulation-epoch")
    cli.add_argument("--port", type=int)
    cli.add_argument("--router")
    cli.add_argument("--trace")
    cli.add_argument("--profile", choices=PROFILES)
    cli.add_argument("--output")
    args = cli.parse_args()
    if args.action == "serve":
        web.run_app(
            application(args.base, args.epoch)[0],
            host="127.0.0.1",
            port=args.port,
            print=None,
        )
    else:
        asyncio.run(control(args))


if __name__ == "__main__":
    main()

"""Open-loop HTTP chat replay. Requires the project's aiohttp dependency."""

from __future__ import annotations

import asyncio
import json
import time

import aiohttp

from workload import digest


async def replay(rows, url, output, *, concurrency, timeout_s, headers=None):
    """Keep every planned request, including failures and unsent cancellations.

    sent_ms is the start of the HTTP client call, including connection setup;
    it is not a packet capture timestamp. TTFT starts at this same boundary.
    """
    start = time.monotonic()

    def elapsed():
        return (time.monotonic() - start) * 1000

    gate = asyncio.Semaphore(concurrency)
    waiting, peak_waiting = 0, 0
    records = [
        {
            "id": row["id"],
            "request_sha256": digest(row["body"]),
            "planned_ms": row["offset_s"] * 1000,
            "status": "not_sent",
            "queued_ms": None,
            "sent_ms": None,
            "finished_ms": None,
            "first_content_ms": None,
            "last_content_ms": None,
            "ttft_ms": None,
            "arrival_ttft_ms": None,
            "client_lag_ms": None,
            "http_status": None,
            "content_chunks": 0,
            "inter_chunk_ms": [],
            "usage": None,
            "backlog_on_arrival": None,
        }
        for row in rows
    ]

    async def request(session, row, record):
        nonlocal waiting, peak_waiting
        try:
            await asyncio.sleep(max(0, row["offset_s"] - (time.monotonic() - start)))
            record["queued_ms"] = elapsed()
            record["backlog_on_arrival"] = waiting
            waiting += 1
            if gate.locked():
                peak_waiting = max(peak_waiting, waiting)
            try:
                async with gate:
                    waiting -= 1
                    record["sent_ms"] = elapsed()
                    record["client_lag_ms"] = record["sent_ms"] - record["planned_ms"]
                    async with session.post(
                        url + "/v1/chat/completions",
                        json=row["body"],
                        headers={**(headers or {}), "x-request-id": row["id"]},
                        allow_redirects=False,
                    ) as response:
                        record["http_status"] = response.status
                        if response.status != 200:
                            record["status"] = "http_error"
                            return
                        if "text/event-stream" not in response.headers.get(
                            "Content-Type", ""
                        ):
                            raise ValueError("expected text/event-stream")
                        data, size, done = [], 0, False
                        async for raw in response.content:
                            size += len(raw)
                            if size > 16 * 1024 * 1024:
                                raise ValueError("stream exceeds 16 MiB limit")
                            line = raw.decode("utf-8").rstrip("\r\n")
                            if line.startswith("data:"):
                                data.append(line[5:].lstrip(" "))
                                continue
                            if line or not data:
                                continue
                            payload, data = "\n".join(data), []
                            if payload == "[DONE]":
                                done = True
                                break
                            event = json.loads(payload)
                            if not isinstance(event, dict) or event.get("error"):
                                raise ValueError("error or invalid SSE event")
                            if event.get("usage") is not None:
                                record["usage"] = event["usage"]
                            choices = event.get("choices", [])
                            if not isinstance(choices, list):
                                raise ValueError("invalid SSE choices")
                            for choice in choices:
                                if choice.get("index", 0) != 0:
                                    raise ValueError("multiple choices are unsupported")
                                delta = choice.get("delta") or {}
                                content = (
                                    delta.get("content")
                                    or delta.get("reasoning_content")
                                    or delta.get("reasoning")
                                )
                                if content is not None and not isinstance(content, str):
                                    raise ValueError("non-text SSE content")
                                if content:
                                    now = elapsed()
                                    if record["first_content_ms"] is None:
                                        record["first_content_ms"] = now
                                        record["ttft_ms"] = now - record["sent_ms"]
                                        record["arrival_ttft_ms"] = (
                                            now - record["planned_ms"]
                                        )
                                    else:
                                        record["inter_chunk_ms"].append(
                                            now - record["last_content_ms"]
                                        )
                                    record["last_content_ms"] = now
                                    record["content_chunks"] += 1
                        if not done or record["first_content_ms"] is None:
                            raise ValueError(
                                "stream ended without DONE or generated content"
                            )
                        record["status"] = "completed"
            finally:
                if record["sent_ms"] is None:
                    waiting -= 1
        except asyncio.TimeoutError:
            record["status"] = "timeout"
        except (
            aiohttp.ClientError,
            ValueError,
            UnicodeError,
            AttributeError,
            TypeError,
        ) as exc:
            record["status"] = (
                "stream_error" if record["http_status"] == 200 else "client_error"
            )
            record["error_type"] = type(
                exc
            ).__name__  # Do not retain generated content or auth.
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        finally:
            peak_waiting = max(peak_waiting, waiting)
            record["finished_ms"] = elapsed()

    tasks = []
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_s),
            connector=aiohttp.TCPConnector(limit=0),
            trust_env=False,
        ) as session:
            tasks = [
                asyncio.create_task(request(session, row, record))
                for row, record in zip(rows, records)
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        with output.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    return {
        "requests": len(rows),
        "completed": sum(r["status"] == "completed" for r in records),
        "peak_client_backlog": peak_waiting,
        "wall_ms": elapsed(),
    }

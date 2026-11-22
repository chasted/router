"""Fixed text workloads. Word counts are deliberately not called token counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def number(value, name, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def validate(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("trace must contain at least one request")
    ids, previous = set(), -1
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each trace record must be an object")
        for field in ("id", "scenario", "length_group", "prefix_group"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"trace {field} must be a nonempty string")
        if row["id"] in ids:
            raise ValueError("duplicate trace request id")
        ids.add(row["id"])
        offset = number(row.get("offset_s"), "offset_s")
        if offset < previous:
            raise ValueError("trace arrival times must be nondecreasing")
        previous = offset
        body = row.get("body")
        if not isinstance(body, dict) or body.get("stream") is not True:
            raise ValueError("trace requires streaming request bodies")
        if not isinstance(body.get("model"), str) or not body["model"]:
            raise ValueError("trace requires an explicit model")
        integer(body.get("max_tokens"), "max_tokens")
        if body.get("n", 1) != 1 or body.get("tools"):
            raise ValueError("PR3 replay supports one text choice and no tools")
        messages = body.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or not all(
                isinstance(m, dict)
                and isinstance(m.get("content"), str)
                and m.get("role") in ("system", "user", "assistant")
                for m in messages
            )
        ):
            raise ValueError("trace requires text chat messages")
        digest(body)  # Also rejects NaN and non-JSON inputs.
    return rows


def load(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return validate(rows)


def generate(
    count,
    rate,
    seed,
    model,
    short_words=32,
    long_words=256,
    output_tokens=16,
    scenario="functional",
    arrival="constant",
):
    integer(count, "count")
    number(rate, "rate", 1e-12)
    integer(short_words, "short_words")
    integer(long_words, "long_words")
    integer(output_tokens, "output_tokens")
    if arrival not in ("constant", "poisson", "burst"):
        raise ValueError("unknown arrival pattern")
    rng, offset, rows = random.Random(seed), 0.0, []
    for index in range(count):
        if index and arrival != "burst":
            offset += 1 / rate if arrival == "constant" else rng.expovariate(rate)
        group = "short" if index % 2 == 0 else "long"
        prefix = str(index % 4)
        words = short_words if group == "short" else long_words
        text = " ".join([f"prefix-{prefix}"] * words) + f" request-{index}"
        rows.append(
            {
                "id": f"request-{index}",
                "offset_s": offset,
                "scenario": scenario,
                "length_group": group,
                "prefix_group": prefix,
                "input_words": words + 1,
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": text}],
                    "max_tokens": output_tokens,
                    "stream": True,
                    "temperature": 0,
                    "seed": seed,
                    "stream_options": {"include_usage": True},
                },
            }
        )
    return validate(rows)


def write(path, rows):
    validate(rows)
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--model", required=True)
    cli.add_argument("--count", type=int, default=64)
    cli.add_argument("--rate", type=float, default=8)
    cli.add_argument("--seed", type=int, default=17)
    cli.add_argument("--short-words", type=int, default=32)
    cli.add_argument("--long-words", type=int, default=256)
    cli.add_argument("--output-tokens", type=int, default=16)
    cli.add_argument("--scenario", default="functional")
    cli.add_argument(
        "--arrival", choices=("constant", "poisson", "burst"), default="constant"
    )
    args = cli.parse_args()
    try:
        rows = generate(
            args.count,
            args.rate,
            args.seed,
            args.model,
            args.short_words,
            args.long_words,
            args.output_tokens,
            args.scenario,
            args.arrival,
        )
        write(args.output, rows)
    except (OSError, ValueError) as exc:
        cli.error(str(exc))
    print(
        json.dumps(
            {"trace": str(args.output), "requests": len(rows), "sha256": digest(rows)}
        )
    )


if __name__ == "__main__":
    main()

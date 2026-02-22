#!/usr/bin/env python3
"""Warn-only benchmark comparing nanobind vs fast_metal on core cases."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from natten_mlx import get_backend, na1d, na2d, na3d, set_backend


def _bench(fn, *, warmup: int, trials: int, rounds: int, trim_head: int) -> dict[str, float]:
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    round_medians: list[float] = []
    for _ in range(rounds):
        for _ in range(warmup):
            x = fn()
            mx.eval(x)

        times: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            x = fn()
            mx.eval(x)
            times.append((time.perf_counter() - t0) * 1000.0)

        trimmed = times[trim_head:] if trim_head < len(times) else times[-1:]
        round_medians.append(float(statistics.median(trimmed)))

    return {
        "median_ms": float(statistics.median(round_medians)),
        "round_median_ms": [float(x) for x in round_medians],
        "stdev_ms": 0.0 if len(round_medians) <= 1 else float(statistics.pstdev(round_medians)),
    }


def _run_backend(
    backend: str,
    fn,
    *,
    warmup: int,
    trials: int,
    rounds: int,
    trim_head: int,
) -> dict[str, float]:
    prev = get_backend()
    try:
        set_backend(backend)
        return _bench(fn, warmup=warmup, trials=trials, rounds=rounds, trim_head=trim_head)
    finally:
        set_backend(prev)


def _make_cases():
    q1 = mx.random.normal((1, 512, 8, 64)).astype(mx.float16)
    k1 = mx.random.normal((1, 512, 8, 64)).astype(mx.float16)
    v1 = mx.random.normal((1, 512, 8, 64)).astype(mx.float16)
    q1d = mx.random.normal((1, 2048, 8, 16)).astype(mx.float16)
    k1d = mx.random.normal((1, 2048, 8, 16)).astype(mx.float16)
    v1d = mx.random.normal((1, 2048, 8, 16)).astype(mx.float16)
    q2 = mx.random.normal((1, 24, 24, 8, 16)).astype(mx.float16)
    k2 = mx.random.normal((1, 24, 24, 8, 16)).astype(mx.float16)
    v2 = mx.random.normal((1, 24, 24, 8, 16)).astype(mx.float16)
    q3 = mx.random.normal((1, 10, 12, 14, 4, 16)).astype(mx.float16)
    k3 = mx.random.normal((1, 10, 12, 14, 4, 16)).astype(mx.float16)
    v3 = mx.random.normal((1, 10, 12, 14, 4, 16)).astype(mx.float16)
    return [
        {
            "name": "na1d_causal_k9",
            "fn": lambda: na1d(q1, k1, v1, kernel_size=9, stride=1, dilation=1, is_causal=True, scale=0.5),
        },
        {
            "name": "na1d_decode_causal_k9",
            "fn": lambda: na1d(q1d, k1d, v1d, kernel_size=9, stride=1, dilation=1, is_causal=True, scale=0.5),
        },
        {
            "name": "na2d_causal_k9",
            "fn": lambda: na2d(
                q2,
                k2,
                v2,
                kernel_size=(9, 9),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=(True, False),
                scale=0.5,
            ),
        },
        {
            "name": "na3d_causal_k3",
            "fn": lambda: na3d(
                q3,
                k3,
                v3,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                dilation=(1, 1, 1),
                is_causal=(True, False, False),
                scale=0.5,
            ),
        },
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Warn-only nanobind vs fast_metal benchmark compare")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--trim-head", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("benchmarks/nanobind-vs-fast.json"))
    p.add_argument("--github-annotations", action="store_true")
    p.add_argument("--warn-threshold", type=float, default=0.97)
    args = p.parse_args()

    cases = _make_cases()
    out: dict[str, object] = {"cases": {}, "warn_threshold": args.warn_threshold}
    for case in cases:
        fast = _run_backend(
            "fast_metal",
            case["fn"],
            warmup=args.warmup,
            trials=args.trials,
            rounds=args.rounds,
            trim_head=args.trim_head,
        )
        nb = _run_backend(
            "nanobind",
            case["fn"],
            warmup=args.warmup,
            trials=args.trials,
            rounds=args.rounds,
            trim_head=args.trim_head,
        )
        ratio = float(fast["median_ms"] / max(nb["median_ms"], 1e-12))
        rec = {"fast_metal": fast, "nanobind": nb, "ratio_fast_over_nanobind": ratio}
        out["cases"][case["name"]] = rec
        if ratio < args.warn_threshold:
            msg = (
                f"nanobind slower than fast_metal on {case['name']}: "
                f"fast={fast['median_ms']:.3f}ms nanobind={nb['median_ms']:.3f}ms ratio={ratio:.3f}"
            )
            if args.github_annotations:
                print(f"::warning::{msg}")
            else:
                print(f"WARNING: {msg}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

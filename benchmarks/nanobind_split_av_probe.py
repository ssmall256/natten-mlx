#!/usr/bin/env python3
"""Targeted split-AV benchmark probe for nanobind tuning experiments.

This script is intentionally focused on split AV forward/backward kernels so we
can evaluate threadgroup-tiling experiments by shape without touching guardrail
cases.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone

import mlx.core as mx

from natten_mlx import set_backend
from natten_mlx._core import ops


def _bench(fn, *, warmup: int, trials: int) -> dict[str, float]:
    for _ in range(warmup):
        mx.eval(fn())
    times: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        mx.eval(fn())
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms": float(statistics.median(times)),
        "mean_ms": float(statistics.mean(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "stdev_ms": 0.0 if len(times) <= 1 else float(statistics.pstdev(times)),
    }


def _build_cases():
    attn2 = mx.random.normal((1, 32, 32, 8, 49))
    v2 = mx.random.normal((1, 32, 32, 8, 32))
    go2 = mx.random.normal((1, 32, 32, 8, 32))

    attn3 = mx.random.normal((1, 10, 12, 14, 4, 27))
    v3 = mx.random.normal((1, 10, 12, 14, 4, 16))
    go3 = mx.random.normal((1, 10, 12, 14, 4, 16))

    return [
        {
            "name": "split_av2d_forward_k7_u1d1_noncausal",
            "fn": lambda: ops.na2d_av_forward(
                attn2, v2, (7, 7), (1, 1), (1, 1), (False, False)
            ),
        },
        {
            "name": "split_av2d_backward_grad_v_k7_u1d1_noncausal",
            "fn": lambda: ops.na2d_av_backward(
                attn2, v2, go2, (7, 7), (1, 1), (1, 1), (False, False)
            )[1],
        },
        {
            "name": "split_av3d_forward_k3_u1d1_noncausal",
            "fn": lambda: ops.na3d_av_forward(
                attn3, v3, (3, 3, 3), (1, 1, 1), (1, 1, 1), (False, False, False)
            ),
        },
        {
            "name": "split_av3d_backward_grad_v_k3_u1d1_noncausal",
            "fn": lambda: ops.na3d_av_backward(
                attn3, v3, go3, (3, 3, 3), (1, 1, 1), (1, 1, 1), (False, False, False)
            )[1],
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted split-AV benchmark probe.")
    parser.add_argument("--backend", default="nanobind", choices=["nanobind", "fast_metal", "pure"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    set_backend(args.backend)
    cases = _build_cases()
    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "warmup": args.warmup,
        "trials": args.trials,
        "results": {case["name"]: _bench(case["fn"], warmup=args.warmup, trials=args.trials) for case in cases},
    }
    text = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

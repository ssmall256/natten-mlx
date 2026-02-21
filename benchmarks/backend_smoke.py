#!/usr/bin/env python3
"""Backend smoke benchmarks for natten-mlx.

This script is intentionally lightweight for CI:
- Produces stable-enough median timings per backend/case.
- Writes a JSON artifact.
- Emits non-failing warnings when accelerated tiers regress versus pure.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

from natten_mlx import na1d, na2d, na3d, set_backend


def _stats(times_ms: list[float]) -> dict[str, float]:
    stdev = 0.0 if len(times_ms) <= 1 else float(statistics.pstdev(times_ms))
    return {
        "mean_ms": float(statistics.mean(times_ms)),
        "median_ms": float(statistics.median(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
        "stdev_ms": float(stdev),
    }


def _bench(fn, *, warmup: int, trials: int, trim_head: int) -> dict[str, float]:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)

    times_ms: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    trimmed = times_ms[trim_head:] if trim_head < len(times_ms) else times_ms[-1:]
    trimmed_stats = _stats(trimmed)
    raw_stats = _stats(times_ms)
    return {
        **trimmed_stats,
        "raw_mean_ms": raw_stats["mean_ms"],
        "raw_median_ms": raw_stats["median_ms"],
        "raw_min_ms": raw_stats["min_ms"],
        "raw_max_ms": raw_stats["max_ms"],
        "raw_stdev_ms": raw_stats["stdev_ms"],
        "trim_head": int(trim_head),
        "raw_trials": int(len(times_ms)),
        "trimmed_trials": int(len(trimmed)),
    }


def _build_cases():
    q1 = mx.random.normal((2, 128, 8, 32))
    k1 = mx.random.normal((2, 128, 8, 32))
    v1 = mx.random.normal((2, 128, 8, 32))

    q2 = mx.random.normal((1, 24, 24, 8, 32))
    k2 = mx.random.normal((1, 24, 24, 8, 32))
    v2 = mx.random.normal((1, 24, 24, 8, 32))
    q3 = mx.random.normal((1, 10, 12, 14, 4, 16))
    k3 = mx.random.normal((1, 10, 12, 14, 4, 16))
    v3 = mx.random.normal((1, 10, 12, 14, 4, 16))

    return [
        {
            "name": "na1d_fused_k7",
            "accelerated_expected": True,
            "fn": lambda: na1d(
                q1,
                k1,
                v1,
                kernel_size=7,
                stride=1,
                dilation=1,
                is_causal=False,
                scale=0.5,
            ),
        },
        {
            "name": "na1d_fused_k7_causal",
            "accelerated_expected": True,
            "fn": lambda: na1d(
                q1,
                k1,
                v1,
                kernel_size=7,
                stride=1,
                dilation=1,
                is_causal=True,
                scale=0.5,
            ),
        },
        {
            "name": "na2d_fused_k7",
            "accelerated_expected": True,
            "fn": lambda: na2d(
                q2,
                k2,
                v2,
                kernel_size=(7, 7),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=(False, False),
                scale=0.5,
            ),
        },
        {
            "name": "na1d_stride2_causal",
            "accelerated_expected": True,
            "fn": lambda: na1d(
                q1,
                k1,
                v1,
                kernel_size=5,
                stride=2,
                dilation=2,
                is_causal=True,
                scale=0.5,
            ),
        },
        {
            "name": "na2d_stride_causal",
            "accelerated_expected": True,
            "fn": lambda: na2d(
                q2,
                k2,
                v2,
                kernel_size=(3, 3),
                stride=(2, 2),
                dilation=(2, 2),
                is_causal=(True, False),
                scale=0.5,
            ),
        },
        {
            "name": "na3d_fused_k3_causal",
            "accelerated_expected": True,
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


def _emit_warnings(payload: dict, warn_threshold: float, github_warnings: bool) -> None:
    pure = payload["results"].get("pure", {})
    for backend, cases in payload["results"].items():
        if backend == "pure":
            continue
        for case in payload["cases"]:
            case_name = case["name"]
            if not case["accelerated_expected"]:
                continue
            pure_median = pure.get(case_name, {}).get("median_ms")
            backend_median = cases.get(case_name, {}).get("median_ms")
            if pure_median is None or backend_median is None:
                continue
            ratio = backend_median / pure_median
            if ratio <= warn_threshold:
                continue
            message = (
                f"Perf drift: backend={backend} case={case_name} median={backend_median:.3f}ms "
                f"vs pure={pure_median:.3f}ms ({ratio:.2f}x slower; threshold={warn_threshold:.2f}x)"
            )
            if github_warnings:
                print(f"::warning::{message}")
            else:
                print(f"WARNING: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run natten-mlx backend smoke benchmarks.")
    parser.add_argument(
        "--backends",
        default="pure,fast_metal,nanobind",
        help="Comma-separated backends to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per case.")
    parser.add_argument("--trials", type=int, default=8, help="Measured iterations per case.")
    parser.add_argument(
        "--trim-head",
        type=int,
        default=2,
        help="Drop this many measured trials from the head before reporting stats.",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=1.25,
        help="Emit warning when accelerated median exceeds this ratio vs pure.",
    )
    parser.add_argument(
        "--github-warnings",
        action="store_true",
        help="Emit warnings in GitHub Actions annotation format.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/backend-smoke.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()
    if args.trim_head < 0:
        raise SystemExit("--trim-head must be >= 0")

    cases = _build_cases()
    backend_names = [x.strip() for x in args.backends.split(",") if x.strip()]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "warmup": args.warmup,
        "trials": args.trials,
        "trim_head": args.trim_head,
        "warn_threshold": args.warn_threshold,
        "cases": [
            {
                "name": case["name"],
                "accelerated_expected": case["accelerated_expected"],
            }
            for case in cases
        ],
        "results": {},
    }

    try:
        for backend in backend_names:
            set_backend(backend)
            payload["results"][backend] = {}
            for case in cases:
                payload["results"][backend][case["name"]] = _bench(
                    case["fn"],
                    warmup=args.warmup,
                    trials=args.trials,
                    trim_head=args.trim_head,
                )
    finally:
        set_backend("auto")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    _emit_warnings(payload, warn_threshold=args.warn_threshold, github_warnings=args.github_warnings)


if __name__ == "__main__":
    main()

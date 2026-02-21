#!/usr/bin/env python3
"""Backward-performance guardrail for accelerated natten-mlx backends.

Fails when fast_metal or nanobind backward medians regress too close to pure.
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
        "stdev_ms": stdev,
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


def _bench_rounds(fn, *, warmup: int, trials: int, trim_head: int, rounds: int) -> dict[str, float]:
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    round_results = [
        _bench(fn, warmup=warmup, trials=trials, trim_head=trim_head) for _ in range(rounds)
    ]
    round_medians = [float(r["median_ms"]) for r in round_results]
    round_raw_medians = [float(r["raw_median_ms"]) for r in round_results]
    stdev_round_median = (
        0.0 if len(round_medians) <= 1 else float(statistics.pstdev(round_medians))
    )
    stdev_round_raw_median = (
        0.0 if len(round_raw_medians) <= 1 else float(statistics.pstdev(round_raw_medians))
    )
    return {
        "mean_ms": float(statistics.median([float(r["mean_ms"]) for r in round_results])),
        "median_ms": float(statistics.median(round_medians)),
        "min_ms": float(min(float(r["min_ms"]) for r in round_results)),
        "max_ms": float(max(float(r["max_ms"]) for r in round_results)),
        "stdev_ms": stdev_round_median,
        "raw_mean_ms": float(
            statistics.median([float(r["raw_mean_ms"]) for r in round_results])
        ),
        "raw_median_ms": float(statistics.median(round_raw_medians)),
        "raw_min_ms": float(min(float(r["raw_min_ms"]) for r in round_results)),
        "raw_max_ms": float(max(float(r["raw_max_ms"]) for r in round_results)),
        "raw_stdev_ms": stdev_round_raw_median,
        "trim_head": int(trim_head),
        "raw_trials": int(sum(int(r["raw_trials"]) for r in round_results)),
        "trimmed_trials": int(sum(int(r["trimmed_trials"]) for r in round_results)),
        "rounds": int(rounds),
        "round_median_ms": [float(x) for x in round_medians],
        "round_raw_median_ms": [float(x) for x in round_raw_medians],
    }


def _build_cases():
    q1 = mx.random.normal((2, 256, 8, 32))
    k1 = mx.random.normal((2, 256, 8, 32))
    v1 = mx.random.normal((2, 256, 8, 32))
    q1_decode_causal = mx.random.normal((1, 512, 8, 64))
    k1_decode_causal = mx.random.normal((1, 512, 8, 64))
    v1_decode_causal = mx.random.normal((1, 512, 8, 64))
    q1_decode_long = mx.random.normal((1, 1024, 8, 64))
    k1_decode_long = mx.random.normal((1, 1024, 8, 64))
    v1_decode_long = mx.random.normal((1, 1024, 8, 64))

    q2 = mx.random.normal((1, 32, 32, 8, 32))
    k2 = mx.random.normal((1, 32, 32, 8, 32))
    v2 = mx.random.normal((1, 32, 32, 8, 32))

    q3 = mx.random.normal((1, 10, 12, 14, 4, 16))
    k3 = mx.random.normal((1, 10, 12, 14, 4, 16))
    v3 = mx.random.normal((1, 10, 12, 14, 4, 16))

    def _bw_1d():
        def loss_fn(q_in):
            return mx.sum(
                na1d(
                    q_in,
                    k1,
                    v1,
                    kernel_size=7,
                    stride=1,
                    dilation=1,
                    is_causal=False,
                    scale=0.5,
                )
            )

        return mx.grad(loss_fn)(q1)

    def _bw_2d():
        def loss_fn(q_in):
            return mx.sum(
                na2d(
                    q_in,
                    k2,
                    v2,
                    kernel_size=(7, 7),
                    stride=(1, 1),
                    dilation=(1, 1),
                    is_causal=(False, False),
                    scale=0.5,
                )
            )

        return mx.grad(loss_fn)(q2)

    def _bw_1d_decode_causal():
        def loss_fn(q_in):
            return mx.sum(
                na1d(
                    q_in,
                    k1_decode_causal,
                    v1_decode_causal,
                    kernel_size=9,
                    stride=1,
                    dilation=2,
                    is_causal=True,
                    scale=0.5,
                )
            )

        return mx.grad(loss_fn)(q1_decode_causal)

    def _bw_1d_decode_long_noncausal():
        def loss_fn(q_in):
            return mx.sum(
                na1d(
                    q_in,
                    k1_decode_long,
                    v1_decode_long,
                    kernel_size=7,
                    stride=1,
                    dilation=1,
                    is_causal=False,
                    scale=0.5,
                )
            )

        return mx.grad(loss_fn)(q1_decode_long)

    def _bw_3d():
        def loss_fn(q_in):
            return mx.sum(
                na3d(
                    q_in,
                    k3,
                    v3,
                    kernel_size=(3, 3, 3),
                    stride=(1, 1, 1),
                    dilation=(1, 1, 1),
                    is_causal=(False, False, False),
                    scale=0.5,
                )
            )

        return mx.grad(loss_fn)(q3)

    return [
        {"name": "na1d_k7_s1_d1_noncausal", "backward": _bw_1d},
        {"name": "na1d_k9_s1_d2_causal_L512_D64", "backward": _bw_1d_decode_causal},
        {"name": "na1d_k7_s1_d1_noncausal_L1024_D64", "backward": _bw_1d_decode_long_noncausal},
        {"name": "na2d_k7x7_s1_d1_noncausal", "backward": _bw_2d},
        {"name": "na3d_k3x3x3_s1_d1_noncausal", "backward": _bw_3d},
    ]


def _format_speedup(speedup: float) -> str:
    return f"{speedup:.2f}x"


def _emit_violation(
    *,
    backend: str,
    case: str,
    median_ms: float,
    pure_median_ms: float,
    speedup: float,
    min_speedup: float,
    github_annotations: bool,
) -> None:
    message = (
        f"Backward perf guardrail failed: backend={backend} case={case} "
        f"median={median_ms:.3f}ms pure={pure_median_ms:.3f}ms "
        f"speedup={_format_speedup(speedup)} required>={min_speedup:.2f}x"
    )
    if github_annotations:
        print(f"::error::{message}")
    else:
        print(f"ERROR: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check backward performance guardrails.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Repeat each backend/case benchmark this many times and aggregate with median-of-medians.",
    )
    parser.add_argument(
        "--trim-head",
        type=int,
        default=2,
        help="Drop this many measured trials from the head before reporting stats.",
    )
    parser.add_argument("--min-speedup", type=float, default=1.20)
    parser.add_argument("--output", default="benchmarks/backward-guardrail.json")
    parser.add_argument("--github-annotations", action="store_true")
    args = parser.parse_args()
    if args.trim_head < 0:
        raise SystemExit("--trim-head must be >= 0")
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")

    cases = _build_cases()
    backends = ["pure", "fast_metal", "nanobind"]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "warmup": args.warmup,
        "trials": args.trials,
        "rounds": args.rounds,
        "trim_head": args.trim_head,
        "min_speedup": args.min_speedup,
        "cases": [c["name"] for c in cases],
        "results": {backend: {} for backend in backends},
    }

    try:
        for backend in backends:
            set_backend(backend)
            for case in cases:
                payload["results"][backend][case["name"]] = _bench_rounds(
                    case["backward"],
                    warmup=args.warmup,
                    trials=args.trials,
                    trim_head=args.trim_head,
                    rounds=args.rounds,
                )
    finally:
        set_backend("auto")

    violations: list[tuple[str, str, float, float, float]] = []
    for backend in ("fast_metal", "nanobind"):
        for case in cases:
            name = case["name"]
            pure_median = payload["results"]["pure"][name]["median_ms"]
            backend_median = payload["results"][backend][name]["median_ms"]
            speedup = pure_median / backend_median if backend_median > 0.0 else 0.0
            payload["results"][backend][name]["speedup_vs_pure"] = speedup
            if speedup < args.min_speedup:
                violations.append((backend, name, backend_median, pure_median, speedup))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    if violations:
        for backend, case, median_ms, pure_median_ms, speedup in violations:
            _emit_violation(
                backend=backend,
                case=case,
                median_ms=median_ms,
                pure_median_ms=pure_median_ms,
                speedup=speedup,
                min_speedup=args.min_speedup,
                github_annotations=args.github_annotations,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Forward-performance guardrail for causal low-precision configurations."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

from natten_mlx import na1d, na2d, na2d_av, na2d_qk, na3d, na3d_av, na3d_qk, set_backend


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
    stdev_round_median = 0.0 if len(round_medians) <= 1 else float(statistics.pstdev(round_medians))
    stdev_round_raw_median = (
        0.0 if len(round_raw_medians) <= 1 else float(statistics.pstdev(round_raw_medians))
    )
    return {
        "mean_ms": float(statistics.median([float(r["mean_ms"]) for r in round_results])),
        "median_ms": float(statistics.median(round_medians)),
        "min_ms": float(min(float(r["min_ms"]) for r in round_results)),
        "max_ms": float(max(float(r["max_ms"]) for r in round_results)),
        "stdev_ms": stdev_round_median,
        "raw_mean_ms": float(statistics.median([float(r["raw_mean_ms"]) for r in round_results])),
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
    dtypes = [("fp16", mx.float16)]
    bf16 = getattr(mx, "bfloat16", None)
    if bf16 is not None:
        dtypes.append(("bf16", bf16))

    cases: list[dict] = []
    for dtype_name, dtype in dtypes:
        q1 = mx.random.normal((1, 512, 8, 64)).astype(dtype)
        k1 = mx.random.normal((1, 512, 8, 64)).astype(dtype)
        v1 = mx.random.normal((1, 512, 8, 64)).astype(dtype)
        q1_long = mx.random.normal((1, 2048, 8, 16)).astype(dtype)
        k1_long = mx.random.normal((1, 2048, 8, 16)).astype(dtype)
        v1_long = mx.random.normal((1, 2048, 8, 16)).astype(dtype)

        q2 = mx.random.normal((1, 24, 24, 8, 16)).astype(dtype)
        k2 = mx.random.normal((1, 24, 24, 8, 16)).astype(dtype)
        v2 = mx.random.normal((1, 24, 24, 8, 16)).astype(dtype)
        q2_split = mx.random.normal((1, 20, 18, 8, 16)).astype(dtype)
        k2_split = mx.random.normal((1, 20, 18, 8, 16)).astype(dtype)
        v2_split = mx.random.normal((1, 20, 18, 8, 16)).astype(dtype)

        q3 = mx.random.normal((1, 10, 12, 14, 4, 16)).astype(dtype)
        k3 = mx.random.normal((1, 10, 12, 14, 4, 16)).astype(dtype)
        v3 = mx.random.normal((1, 10, 12, 14, 4, 16)).astype(dtype)
        q3_split = mx.random.normal((1, 8, 10, 12, 4, 16)).astype(dtype)
        k3_split = mx.random.normal((1, 8, 10, 12, 4, 16)).astype(dtype)
        v3_split = mx.random.normal((1, 8, 10, 12, 4, 16)).astype(dtype)

        cases.extend(
            [
                {
                    "name": f"na1d_k9_s1_d1_causal_L512_D64_{dtype_name}",
                    "forward": lambda q=q1, k=k1, v=v1: na1d(
                        q,
                        k,
                        v,
                        kernel_size=9,
                        stride=1,
                        dilation=1,
                        is_causal=True,
                        scale=0.5,
                    ),
                    "min_speedup": 1.10,
                },
                {
                    "name": f"na1d_k9_s1_d1_causal_decode_L2048_D16_{dtype_name}",
                    "forward": lambda q=q1_long, k=k1_long, v=v1_long: na1d(
                        q,
                        k,
                        v,
                        kernel_size=9,
                        stride=1,
                        dilation=1,
                        is_causal=True,
                        scale=0.5,
                    ),
                    "min_speedup": 1.05,
                },
                {
                    "name": f"na2d_k9x9_s1_d1_causal_h_24x24_D16_{dtype_name}",
                    "forward": lambda q=q2, k=k2, v=v2: na2d(
                        q,
                        k,
                        v,
                        kernel_size=(9, 9),
                        stride=(1, 1),
                        dilation=(1, 1),
                        is_causal=(True, False),
                            scale=0.5,
                        ),
                    "min_speedup": 1.10,
                },
                {
                    "name": f"na2d_k7x7_s2x1_d1x2_causal_h_strided_20x18_D16_{dtype_name}",
                    "forward": lambda q=q2_split, k=k2_split, v=v2_split: na2d(
                        q,
                        k,
                        v,
                        kernel_size=(7, 7),
                        stride=(2, 1),
                        dilation=(1, 2),
                        is_causal=(True, False),
                        scale=0.5,
                    ),
                    "min_speedup": 1.05,
                },
                {
                    "name": f"na2d_split_k7x7_s2x1_d1x2_causal_h_20x18_D16_{dtype_name}",
                    "forward": lambda q=q2_split, k=k2_split, v=v2_split: na2d_av(
                        mx.softmax(
                            na2d_qk(
                                q,
                                k,
                                kernel_size=(7, 7),
                                stride=(2, 1),
                                dilation=(1, 2),
                                is_causal=(True, False),
                                scale=0.5,
                            ),
                            axis=-1,
                        ),
                        v,
                        kernel_size=(7, 7),
                        stride=(2, 1),
                        dilation=(1, 2),
                        is_causal=(True, False),
                    ),
                    "min_speedup": 1.05,
                },
                {
                    "name": f"na3d_k3x3x3_s1_d1_causal_d_10x12x14_D16_{dtype_name}",
                    "forward": lambda q=q3, k=k3, v=v3: na3d(
                        q,
                        k,
                        v,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        dilation=(1, 1, 1),
                        is_causal=(True, False, False),
                            scale=0.5,
                        ),
                    "min_speedup": 1.10,
                },
                {
                    "name": f"na3d_split_k3x3x3_s2x1x1_d1x1x2_causal_d_8x10x12_D16_{dtype_name}",
                    "forward": lambda q=q3_split, k=k3_split, v=v3_split: na3d_av(
                        mx.softmax(
                            na3d_qk(
                                q,
                                k,
                                kernel_size=(3, 3, 3),
                                stride=(2, 1, 1),
                                dilation=(1, 1, 2),
                                is_causal=(True, False, False),
                                scale=0.5,
                            ),
                            axis=-1,
                        ),
                        v,
                        kernel_size=(3, 3, 3),
                        stride=(2, 1, 1),
                        dilation=(1, 1, 2),
                        is_causal=(True, False, False),
                    ),
                    "min_speedup": 1.05,
                },
            ]
        )
    return cases


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
        f"Forward perf guardrail failed: backend={backend} case={case} "
        f"median={median_ms:.3f}ms pure={pure_median_ms:.3f}ms "
        f"speedup={_format_speedup(speedup)} required>={min_speedup:.2f}x"
    )
    if github_annotations:
        print(f"::error::{message}")
    else:
        print(f"ERROR: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check forward performance guardrails.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Repeat each backend/case benchmark this many times and aggregate with median-of-medians.",
    )
    parser.add_argument(
        "--trim-head",
        type=int,
        default=2,
        help="Drop this many measured trials from the head before reporting stats.",
    )
    parser.add_argument("--min-speedup", type=float, default=1.10)
    parser.add_argument("--output", default="benchmarks/forward-guardrail.json")
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
        "case_min_speedup": {c["name"]: float(c.get("min_speedup", args.min_speedup)) for c in cases},
        "results": {backend: {} for backend in backends},
    }

    try:
        for backend in backends:
            set_backend(backend)
            for case in cases:
                payload["results"][backend][case["name"]] = _bench_rounds(
                    case["forward"],
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
            case_min_speedup = float(case.get("min_speedup", args.min_speedup))
            payload["results"][backend][name]["required_min_speedup"] = case_min_speedup
            if speedup < case_min_speedup:
                violations.append((backend, name, backend_median, pure_median, speedup, case_min_speedup))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    if violations:
        for backend, case, median_ms, pure_median_ms, speedup, min_speedup in violations:
            _emit_violation(
                backend=backend,
                case=case,
                median_ms=median_ms,
                pure_median_ms=pure_median_ms,
                speedup=speedup,
                min_speedup=min_speedup,
                github_annotations=args.github_annotations,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

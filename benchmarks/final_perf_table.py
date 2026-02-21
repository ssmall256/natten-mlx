#!/usr/bin/env python3
"""Generate final backend performance table for natten-mlx.

Benchmarks forward and backward medians for pure, fast_metal, and nanobind.
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


def _bench(fn, *, warmup: int, trials: int) -> dict[str, float]:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)

    times_ms: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    stdev = 0.0 if len(times_ms) <= 1 else float(statistics.pstdev(times_ms))
    return {
        "mean_ms": float(statistics.mean(times_ms)),
        "median_ms": float(statistics.median(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
        "stdev_ms": stdev,
    }


def _build_cases():
    q1 = mx.random.normal((2, 256, 8, 32))
    k1 = mx.random.normal((2, 256, 8, 32))
    v1 = mx.random.normal((2, 256, 8, 32))

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
        {
            "name": "na1d_k7_s1_d1_noncausal",
            "forward": lambda: na1d(
                q1, k1, v1, kernel_size=7, stride=1, dilation=1, is_causal=False, scale=0.5
            ),
            "backward": _bw_1d,
        },
        {
            "name": "na2d_k7x7_s1_d1_noncausal",
            "forward": lambda: na2d(
                q2,
                k2,
                v2,
                kernel_size=(7, 7),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=(False, False),
                scale=0.5,
            ),
            "backward": _bw_2d,
        },
        {
            "name": "na3d_k3x3x3_s1_d1_noncausal",
            "forward": lambda: na3d(
                q3,
                k3,
                v3,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                dilation=(1, 1, 1),
                is_causal=(False, False, False),
                scale=0.5,
            ),
            "backward": _bw_3d,
        },
    ]


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _ratio(base: float, other: float) -> str:
    if other <= 0.0:
        return "n/a"
    return f"{base / other:.2f}x"


def _to_markdown(payload: dict) -> str:
    lines = []
    lines.append("| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for case in payload["cases"]:
        case_name = case["name"]
        for direction in ("forward", "backward"):
            p = payload["results"]["pure"][case_name][direction]["median_ms"]
            f = payload["results"]["fast_metal"][case_name][direction]["median_ms"]
            n = payload["results"]["nanobind"][case_name][direction]["median_ms"]
            lines.append(
                f"| `{case_name}` | `{direction}` | {_fmt(p)} | {_fmt(f)} | {_fmt(n)} | {_ratio(p, f)} | {_ratio(p, n)} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final perf table for natten-mlx backends.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output-json", default="benchmarks/final-perf.json")
    parser.add_argument("--output-md", default="benchmarks/final-perf.md")
    args = parser.parse_args()

    cases = _build_cases()
    backends = ["pure", "fast_metal", "nanobind"]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "warmup": args.warmup,
        "trials": args.trials,
        "cases": [{"name": c["name"]} for c in cases],
        "results": {b: {} for b in backends},
    }

    try:
        for backend in backends:
            set_backend(backend)
            for case in cases:
                payload["results"][backend][case["name"]] = {
                    "forward": _bench(case["forward"], warmup=args.warmup, trials=args.trials),
                    "backward": _bench(case["backward"], warmup=args.warmup, trials=args.trials),
                }
    finally:
        set_backend("auto")

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(_to_markdown(payload) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    print()
    print(_to_markdown(payload))


if __name__ == "__main__":
    main()

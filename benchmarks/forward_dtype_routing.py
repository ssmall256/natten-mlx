#!/usr/bin/env python3
"""Benchmark native low-precision vs forced-fp32 forward routing."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from natten_mlx import na2d, set_backend
from natten_mlx._core import fast_metal


def _bench_rounds(fn, *, warmup: int, trials: int, trim_head: int, rounds: int) -> tuple[float, list[float]]:
    medians: list[float] = []
    for _ in range(rounds):
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
        medians.append(float(statistics.median(trimmed)))
    return float(statistics.median(medians)), medians


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward dtype routing benchmark.")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--trials", type=int, default=36)
    parser.add_argument("--trim-head", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", default="benchmarks/forward-dtype-routing.json")
    args = parser.parse_args()

    bf16 = getattr(mx, "bfloat16", None)
    if bf16 is None:
        raise SystemExit("bfloat16 unavailable on this runtime")

    set_backend("fast_metal")
    try:
        # Include the benchmark-validated route case and neighboring controls.
        cases = [
            {"name": "route_case_k9_d16_24_causal_h", "shape": (1, 24, 24, 8, 16), "k": 9, "causal": (True, False)},
            {"name": "control_k7_d16_24_causal_h", "shape": (1, 24, 24, 8, 16), "k": 7, "causal": (True, False)},
            {"name": "control_k9_d16_32_causal_h", "shape": (1, 32, 32, 8, 16), "k": 9, "causal": (True, False)},
            {"name": "control_k9_d16_24_causal_hw", "shape": (1, 24, 24, 8, 16), "k": 9, "causal": (True, True)},
        ]
        payload = {
            "backend": "fast_metal",
            "dtype": "bfloat16",
            "warmup": args.warmup,
            "trials": args.trials,
            "trim_head": args.trim_head,
            "rounds": args.rounds,
            "cases": [],
        }
        for case in cases:
            shape = case["shape"]
            q = mx.random.normal(shape).astype(bf16)
            k = mx.random.normal(shape).astype(bf16)
            v = mx.random.normal(shape).astype(bf16)
            ksz = case["k"]
            causal = case["causal"]
            native = lambda: na2d(
                q,
                k,
                v,
                kernel_size=(ksz, ksz),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=causal,
                scale=0.5,
            )
            forced = lambda: na2d(
                q.astype(mx.float32),
                k.astype(mx.float32),
                v.astype(mx.float32),
                kernel_size=(ksz, ksz),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=causal,
                scale=0.5,
            ).astype(bf16)
            route_flag = fast_metal._ENABLE_FORWARD_LOWP_FP32_ROUTE
            try:
                fast_metal._ENABLE_FORWARD_LOWP_FP32_ROUTE = False
                native_ms, native_round_medians = _bench_rounds(
                    native,
                    warmup=args.warmup,
                    trials=args.trials,
                    trim_head=args.trim_head,
                    rounds=args.rounds,
                )
                fast_metal._ENABLE_FORWARD_LOWP_FP32_ROUTE = True
                forced_ms, forced_round_medians = _bench_rounds(
                    forced,
                    warmup=args.warmup,
                    trials=args.trials,
                    trim_head=args.trim_head,
                    rounds=args.rounds,
                )
            finally:
                fast_metal._ENABLE_FORWARD_LOWP_FP32_ROUTE = route_flag
            payload["cases"].append(
                {
                    "name": case["name"],
                    "native_ms": native_ms,
                    "native_round_medians_ms": native_round_medians,
                    "forced_fp32_ms": forced_ms,
                    "forced_fp32_round_medians_ms": forced_round_medians,
                    "native_over_forced": native_ms / forced_ms if forced_ms > 0.0 else 0.0,
                }
            )
    finally:
        set_backend("auto")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Warn-only benchmark comparing nanobind vs fast_metal on backward core cases."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from natten_mlx import get_backend, na1d, na2d, na3d, set_backend
from natten_mlx._core import ops


def _bench(fn, *, warmup: int, trials: int, rounds: int, trim_head: int) -> dict[str, float]:
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    round_medians: list[float] = []
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
        round_medians.append(float(statistics.median(trimmed)))

    return {
        "median_ms": float(statistics.median(round_medians)),
        "round_median_ms": [float(x) for x in round_medians],
        "stdev_ms": 0.0 if len(round_medians) <= 1 else float(statistics.pstdev(round_medians)),
    }


def _run_backend(backend: str, fn, *, warmup: int, trials: int, rounds: int, trim_head: int) -> dict[str, float]:
    prev = get_backend()
    try:
        set_backend(backend)
        return _bench(fn, warmup=warmup, trials=trials, rounds=rounds, trim_head=trim_head)
    finally:
        set_backend(prev)


def _make_cases():
    q1 = mx.random.normal((2, 256, 8, 32))
    k1 = mx.random.normal((2, 256, 8, 32))
    v1 = mx.random.normal((2, 256, 8, 32))
    q2 = mx.random.normal((1, 32, 32, 8, 32))
    k2 = mx.random.normal((1, 32, 32, 8, 32))
    v2 = mx.random.normal((1, 32, 32, 8, 32))
    q3 = mx.random.normal((1, 10, 12, 14, 4, 16))
    k3 = mx.random.normal((1, 10, 12, 14, 4, 16))
    v3 = mx.random.normal((1, 10, 12, 14, 4, 16))
    grad_attn_2d_split = mx.random.normal((1, 32, 32, 8, 49))
    grad_attn_3d_split = mx.random.normal((1, 10, 12, 14, 4, 27))

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

    def _bw_split_2d_qk_grad_k_hotspot():
        _grad_q, grad_k = ops.na2d_qk_backward(
            q2,
            k2,
            grad_attn_2d_split,
            (7, 7),
            (1, 1),
            (1, 1),
            (False, False),
            0.5,
        )
        return grad_k

    def _bw_split_3d_qk_grad_k_hotspot():
        _grad_q, grad_k = ops.na3d_qk_backward(
            q3,
            k3,
            grad_attn_3d_split,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            (False, False, False),
            0.5,
        )
        return grad_k

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

    return [
        {"name": "na1d_k7_s1_d1_noncausal", "fn": _bw_1d},
        {"name": "na2d_k7x7_s1_d1_noncausal", "fn": _bw_2d},
        {"name": "na3d_k3x3x3_s1_d1_noncausal", "fn": _bw_3d},
        {"name": "split_na2d_qk_grad_k_k7_s1_d1_noncausal", "fn": _bw_split_2d_qk_grad_k_hotspot},
        {"name": "split_na3d_qk_grad_k_k3_s1_d1_noncausal", "fn": _bw_split_3d_qk_grad_k_hotspot},
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Warn-only nanobind vs fast_metal backward benchmark compare")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--trim-head", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("benchmarks/nanobind-vs-fast-backward.json"))
    p.add_argument("--github-annotations", action="store_true")
    p.add_argument("--warn-threshold", type=float, default=0.80)
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
                f"nanobind backward slower than fast_metal on {case['name']}: "
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


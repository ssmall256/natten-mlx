#!/usr/bin/env python3
"""Tune nanobind backward mode + threadgroup candidates and emit recommendations."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx

from natten_mlx import na1d, na2d, na3d, set_backend
from natten_mlx._core import ops


def _bench(fn, *, warmup: int, trials: int, rounds: int, trim_head: int) -> float:
    round_medians: list[float] = []
    for _ in range(rounds):
        for _ in range(warmup):
            out = fn()
            mx.eval(out)
        times: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            out = fn()
            mx.eval(out)
            times.append((time.perf_counter() - t0) * 1000.0)
        trimmed = times[trim_head:] if trim_head < len(times) else times[-1:]
        round_medians.append(float(statistics.median(trimmed)))
    return float(statistics.median(round_medians))


def _token_band(tokens: int) -> str:
    if tokens <= 256:
        return "tiny"
    if tokens <= 1024:
        return "small"
    if tokens <= 4096:
        return "medium"
    return "large"


def _head_dim_band(head_dim: int) -> str:
    if head_dim <= 16:
        return "d16"
    if head_dim <= 32:
        return "d32"
    return "d64p"


def _kernel_band(kernel_size: int) -> str:
    if kernel_size <= 5:
        return "k_small"
    if kernel_size <= 9:
        return "k_mid"
    return "k_large"


def main() -> None:
    p = argparse.ArgumentParser(description="Nanobind backward tuner")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--trim-head", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("benchmarks/nanobind-backward-tuning.json"))
    args = p.parse_args()

    q1 = mx.random.normal((2, 256, 8, 32))
    k1 = mx.random.normal((2, 256, 8, 32))
    v1 = mx.random.normal((2, 256, 8, 32))
    q2 = mx.random.normal((1, 32, 32, 8, 32))
    k2 = mx.random.normal((1, 32, 32, 8, 32))
    v2 = mx.random.normal((1, 32, 32, 8, 32))
    q3 = mx.random.normal((1, 10, 12, 14, 4, 16))
    k3 = mx.random.normal((1, 10, 12, 14, 4, 16))
    v3 = mx.random.normal((1, 10, 12, 14, 4, 16))
    grad_attn_2d = mx.random.normal((1, 32, 32, 8, 49))
    grad_attn_3d = mx.random.normal((1, 10, 12, 14, 4, 27))

    cases = [
        {
            "name": "na1d_k7_s1_d1_noncausal",
            "op": "na1d_qk_backward",
            "tokens": 256,
            "head_dim": 32,
            "kernel_size": 7,
            "fn": lambda: mx.grad(lambda x: mx.sum(na1d(x, k1, v1, kernel_size=7, stride=1, dilation=1, is_causal=False, scale=0.5)))(q1),
        },
        {
            "name": "na2d_k7x7_s1_d1_noncausal",
            "op": "na2d_qk_backward",
            "tokens": 32 * 32,
            "head_dim": 32,
            "kernel_size": 7,
            "fn": lambda: mx.grad(lambda x: mx.sum(na2d(x, k2, v2, kernel_size=(7, 7), stride=(1, 1), dilation=(1, 1), is_causal=(False, False), scale=0.5)))(q2),
        },
        {
            "name": "na3d_k3x3x3_s1_d1_noncausal",
            "op": "na3d_qk_backward",
            "tokens": 10 * 12 * 14,
            "head_dim": 16,
            "kernel_size": 3,
            "fn": lambda: mx.grad(lambda x: mx.sum(na3d(x, k3, v3, kernel_size=(3, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1), is_causal=(False, False, False), scale=0.5)))(q3),
        },
        {
            "name": "split_na2d_qk_grad_k_k7_s1_d1_noncausal",
            "op": "na2d_qk_backward",
            "tokens": 32 * 32,
            "head_dim": 32,
            "kernel_size": 7,
            "fn": lambda: ops.na2d_qk_backward(q2, k2, grad_attn_2d, (7, 7), (1, 1), (1, 1), (False, False), 0.5)[1],
        },
        {
            "name": "split_na3d_qk_grad_k_k3_s1_d1_noncausal",
            "op": "na3d_qk_backward",
            "tokens": 10 * 12 * 14,
            "head_dim": 16,
            "kernel_size": 3,
            "fn": lambda: ops.na3d_qk_backward(q3, k3, grad_attn_3d, (3, 3, 3), (1, 1, 1), (1, 1, 1), (False, False, False), 0.5)[1],
        },
    ]

    modes = ["atomic", "tiled"]
    tg_candidates = [(96, 1, 1), (128, 1, 1), (160, 1, 1), (192, 1, 1), (256, 1, 1)]

    set_backend("nanobind")
    payload: dict[str, object] = {"cases": {}}
    recommended_modes: dict[tuple[str, str, str, str, str, str, str], str] = {}
    recommended_tg: dict[tuple[str, str, str, str, str, str, str], tuple[int, int, int]] = {}

    try:
        for case in cases:
            results = []
            best = None
            for mode in modes:
                os.environ["NATTEN_NANOBIND_BWD_MODE"] = mode
                for tg in tg_candidates:
                    os.environ["NATTEN_NANOBIND_BWD_TG_OVERRIDE"] = f"{tg[0]},{tg[1]},{tg[2]}"
                    median_ms = _bench(
                        case["fn"],
                        warmup=args.warmup,
                        trials=args.trials,
                        rounds=args.rounds,
                        trim_head=args.trim_head,
                    )
                    rec = {
                        "mode": mode,
                        "threadgroup": tg,
                        "median_ms": median_ms,
                    }
                    results.append(rec)
                    if best is None or median_ms < best["median_ms"]:
                        best = rec

            assert best is not None
            payload["cases"][case["name"]] = {
                "best": best,
                "results": results,
            }

            key = (
                str(case["op"]),
                "fp32",
                _token_band(int(case["tokens"])),
                _head_dim_band(int(case["head_dim"])),
                _kernel_band(int(case["kernel_size"])),
                "c0",
                "s1",
            )
            recommended_modes[key] = str(best["mode"])
            recommended_tg[key] = tuple(best["threadgroup"])

        payload["recommended_mode_table"] = {
            str(k): v for k, v in sorted(recommended_modes.items(), key=lambda x: str(x[0]))
        }
        payload["recommended_threadgroup_table"] = {
            str(k): list(v) for k, v in sorted(recommended_tg.items(), key=lambda x: str(x[0]))
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        os.environ.pop("NATTEN_NANOBIND_BWD_MODE", None)
        os.environ.pop("NATTEN_NANOBIND_BWD_TG_OVERRIDE", None)
        set_backend("auto")


if __name__ == "__main__":
    main()

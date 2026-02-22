#!/usr/bin/env python3
"""Forward tuner for fused softmax strategy and launch threadgroups.

This script sweeps candidate launch/threadgroup settings and softmax strategy
variants, then emits:
1) a JSON artifact with raw timing data, and
2) a deterministic Python table module consumed by fast_metal dispatch.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from natten_mlx import na2d, na2d_av, na2d_qk, na3d, na3d_av, na3d_qk, set_backend
from natten_mlx._core import fast_metal


def _bench_rounds(fn, *, warmup: int, trials: int, trim_head: int, rounds: int) -> float:
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
    return float(statistics.median(round_medians))


def _build_cases(dtype):
    q2 = mx.random.normal((1, 24, 24, 8, 16)).astype(dtype)
    k2 = mx.random.normal((1, 24, 24, 8, 16)).astype(dtype)
    v2 = mx.random.normal((1, 24, 24, 8, 16)).astype(dtype)
    q3 = mx.random.normal((1, 10, 12, 14, 4, 16)).astype(dtype)
    k3 = mx.random.normal((1, 10, 12, 14, 4, 16)).astype(dtype)
    v3 = mx.random.normal((1, 10, 12, 14, 4, 16)).astype(dtype)
    q2s = mx.random.normal((1, 20, 18, 8, 16)).astype(dtype)
    k2s = mx.random.normal((1, 20, 18, 8, 16)).astype(dtype)
    v2s = mx.random.normal((1, 20, 18, 8, 16)).astype(dtype)
    q3s = mx.random.normal((1, 8, 10, 12, 4, 16)).astype(dtype)
    k3s = mx.random.normal((1, 8, 10, 12, 4, 16)).astype(dtype)
    v3s = mx.random.normal((1, 8, 10, 12, 4, 16)).astype(dtype)
    return [
        {
            "name": "na2d_fused_causal",
            "op": "na2d_fused",
            "fn": lambda: na2d(
                q2,
                k2,
                v2,
                kernel_size=(7, 7),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=(True, False),
                scale=0.5,
            ),
            "shape": (24, 24),
            "kernel_size": 7,
            "head_dim": 16,
            "causal_rank": 1,
            "stride_unit": True,
        },
        {
            "name": "na3d_fused_causal",
            "op": "na3d_fused",
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
            "shape": (10, 12, 14),
            "kernel_size": 3,
            "head_dim": 16,
            "causal_rank": 1,
            "stride_unit": True,
        },
        {
            "name": "na2d_av_split_causal_strided",
            "op": "na2d_av_split",
            "fn": lambda: na2d_av(
                mx.softmax(
                    na2d_qk(
                        q2s,
                        k2s,
                        kernel_size=(7, 7),
                        stride=(2, 1),
                        dilation=(1, 2),
                        is_causal=(True, False),
                        scale=0.5,
                    ),
                    axis=-1,
                ),
                v2s,
                kernel_size=(7, 7),
                stride=(2, 1),
                dilation=(1, 2),
                is_causal=(True, False),
            ),
            "shape": (20, 18),
            "kernel_size": 7,
            "head_dim": 16,
            "causal_rank": 1,
            "stride_unit": False,
        },
        {
            "name": "na3d_av_split_causal_strided",
            "op": "na3d_av_split",
            "fn": lambda: na3d_av(
                mx.softmax(
                    na3d_qk(
                        q3s,
                        k3s,
                        kernel_size=(3, 3, 3),
                        stride=(2, 1, 1),
                        dilation=(1, 1, 2),
                        is_causal=(True, False, False),
                        scale=0.5,
                    ),
                    axis=-1,
                ),
                v3s,
                kernel_size=(3, 3, 3),
                stride=(2, 1, 1),
                dilation=(1, 1, 2),
                is_causal=(True, False, False),
            ),
            "shape": (8, 10, 12),
            "kernel_size": 3,
            "head_dim": 16,
            "causal_rank": 1,
            "stride_unit": False,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune forward launch + softmax strategy tables.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--trim-head", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", default="benchmarks/forward-tuning.json")
    parser.add_argument("--output-py", default="src/natten_mlx/_core/_forward_tuning.py")
    args = parser.parse_args()

    dtype = mx.float16
    if getattr(mx, "bfloat16", None) is not None:
        dtype = mx.bfloat16

    threadgroup_candidates = [
        (8, 8, 1),
        (16, 8, 1),
        (16, 4, 1),
        (32, 4, 1),
    ]
    softmax_candidates = ["stored", "recompute"]

    cases = _build_cases(dtype)
    payload: dict[str, object] = {
        "warmup": args.warmup,
        "trials": args.trials,
        "trim_head": args.trim_head,
        "rounds": args.rounds,
        "dtype": str(dtype),
        "cases": {},
    }

    family = fast_metal._gpu_family_key()
    threadgroup_table: dict[tuple[str, tuple[str, str, str, str, bool]], tuple[int, int, int]] = {}
    softmax_table: dict[tuple[str, tuple[str, str, str, str, bool]], str] = {}

    old_override = fast_metal._FORWARD_SOFTMAX_STRATEGY_OVERRIDE
    old_lookup_tg = fast_metal._lookup_threadgroup
    try:
        set_backend("fast_metal")
        for case in cases:
            results: list[dict[str, object]] = []
            best = None
            for strategy in (softmax_candidates if case["op"] in {"na2d_fused", "na3d_fused"} else ["stored"]):
                fast_metal._FORWARD_SOFTMAX_STRATEGY_OVERRIDE = strategy
                for tg in threadgroup_candidates:
                    fast_metal._lookup_threadgroup = lambda **_kwargs: tg
                    ms = _bench_rounds(
                        case["fn"],
                        warmup=args.warmup,
                        trials=args.trials,
                        trim_head=args.trim_head,
                        rounds=args.rounds,
                    )
                    rec = {"strategy": strategy, "threadgroup": tg, "median_ms": ms}
                    results.append(rec)
                    if best is None or ms < best["median_ms"]:
                        best = rec
            assert best is not None
            payload["cases"][case["name"]] = {"best": best, "results": results}

            tokens = int(np.prod(case["shape"]))
            band_key = (
                fast_metal._token_band(tokens),
                fast_metal._head_dim_band(case["head_dim"]),
                fast_metal._kernel_band(case["kernel_size"]),
                fast_metal._causal_rank_band(case["causal_rank"]),
                bool(case["stride_unit"]),
            )
            threadgroup_table[(case["op"], band_key)] = tuple(best["threadgroup"])
            if case["op"] in {"na2d_fused", "na3d_fused"}:
                softmax_table[(case["op"], ("lowp",) + band_key[0:1] + band_key[2:])] = str(best["strategy"])
    finally:
        fast_metal._FORWARD_SOFTMAX_STRATEGY_OVERRIDE = old_override
        fast_metal._lookup_threadgroup = old_lookup_tg
        set_backend("auto")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Emit deterministic Python tables.
    py_out = Path(args.output_py)
    py_out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '"""Locked forward tuning tables used by fast_metal dispatch."""',
        "",
        "from __future__ import annotations",
        "",
        "FORWARD_THREADGROUP_TABLE = {",
        f'    "{family}": {{',
    ]
    for (op, key), tg in sorted(threadgroup_table.items(), key=lambda x: str(x[0])):
        lines.append(f'        ({op!r}, {key!r}): {tuple(int(v) for v in tg)!r},')
    lines.extend(
        [
            "    },",
            '    "apple_unknown": {},',
            "}",
            "",
            "FORWARD_SOFTMAX_STRATEGY_TABLE = {",
            f'    "{family}": {{',
        ]
    )
    for (op, key), strategy in sorted(softmax_table.items(), key=lambda x: str(x[0])):
        lines.append(f'        ({op!r}, {key!r}): {strategy!r},')
    lines.extend(
        [
            "    },",
            '    "apple_unknown": {},',
            "}",
            "",
        ]
    )
    py_out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

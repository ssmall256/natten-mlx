#!/usr/bin/env python3
"""Tune nanobind fused forward launch + softmax strategy and emit locked tables."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from natten_mlx import na1d, na2d, na3d, set_backend


def _bench(fn, *, warmup: int, trials: int) -> float:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    times: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(times))


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


def _causal_rank_band(causal_rank: int) -> str:
    if causal_rank <= 0:
        return "c0"
    if causal_rank == 1:
        return "c1"
    return "c2p"


def main() -> None:
    p = argparse.ArgumentParser(description="Nanobind launch tuner")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--output", type=Path, default=Path("benchmarks/nanobind-tuning.json"))
    p.add_argument(
        "--output-py",
        type=Path,
        default=Path("src/natten_mlx/_core/_nanobind_tuning.py"),
    )
    args = p.parse_args()

    dtype = mx.float16
    q1 = mx.random.normal((1, 512, 8, 64)).astype(dtype)
    k1 = mx.random.normal((1, 512, 8, 64)).astype(dtype)
    v1 = mx.random.normal((1, 512, 8, 64)).astype(dtype)
    q2 = mx.random.normal((1, 24, 24, 8, 16)).astype(dtype)
    k2 = mx.random.normal((1, 24, 24, 8, 16)).astype(dtype)
    v2 = mx.random.normal((1, 24, 24, 8, 16)).astype(dtype)
    q3 = mx.random.normal((1, 10, 12, 14, 4, 16)).astype(dtype)
    k3 = mx.random.normal((1, 10, 12, 14, 4, 16)).astype(dtype)
    v3 = mx.random.normal((1, 10, 12, 14, 4, 16)).astype(dtype)

    cases = [
        ("na1d_fused", 512, 64, 9, 1, True, lambda: na1d(q1, k1, v1, kernel_size=9, stride=1, dilation=1, is_causal=True, scale=0.5)),
        ("na2d_fused", 24 * 24, 16, 9, 1, True, lambda: na2d(q2, k2, v2, kernel_size=(9, 9), stride=(1, 1), dilation=(1, 1), is_causal=(True, False), scale=0.5)),
        ("na3d_fused", 10 * 12 * 14, 16, 3, 1, True, lambda: na3d(q3, k3, v3, kernel_size=(3, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1), is_causal=(True, False, False), scale=0.5)),
    ]
    tg_candidates = {
        "na1d_fused": [(64, 1, 1), (128, 1, 1), (256, 1, 1)],
        "na2d_fused": [(8, 8, 1), (16, 8, 1), (16, 4, 1)],
        "na3d_fused": [(8, 8, 1), (8, 4, 1), (16, 4, 1)],
    }
    strategy_candidates = ["stored", "recompute"]

    prev = set_backend("nanobind")
    try:
        payload: dict[str, object] = {"cases": {}}
        # Runtime reads these override env vars if set.
        import os

        threadgroup_table: dict[tuple[str, str, str, str, str, str, bool], tuple[int, int, int]] = {}
        strategy_table: dict[tuple[str, str, str, str, str, bool], str] = {}
        for op, tokens, head_dim, kernel_size, causal_rank, stride_unit, fn in cases:
            best = None
            results = []
            for tg in tg_candidates[op]:
                os.environ["NATTEN_NANOBIND_TG_OVERRIDE"] = ",".join(str(x) for x in tg)
                for strategy in (strategy_candidates if op in {"na2d_fused", "na3d_fused"} else ["stored"]):
                    os.environ["NATTEN_NANOBIND_SOFTMAX_OVERRIDE"] = strategy
                    ms = _bench(fn, warmup=args.warmup, trials=args.trials)
                    rec = {"threadgroup": tg, "strategy": strategy, "median_ms": ms}
                    results.append(rec)
                    if best is None or ms < best["median_ms"]:
                        best = rec
            assert best is not None
            payload["cases"][op] = {"best": best, "results": results}
            key = (
                op,
                "lowp",
                _token_band(tokens),
                _head_dim_band(head_dim),
                _kernel_band(kernel_size),
                _causal_rank_band(causal_rank),
                stride_unit,
            )
            threadgroup_table[key] = tuple(best["threadgroup"])
            strategy_key = (
                op,
                "lowp",
                _token_band(tokens),
                _kernel_band(kernel_size),
                _causal_rank_band(causal_rank),
                stride_unit,
            )
            strategy_table[strategy_key] = str(best["strategy"])

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        lines = [
            '"""Locked tuning tables for nanobind native C++/Metal runtime."""',
            "",
            "from __future__ import annotations",
            "",
            "import os",
            "import platform",
            "",
            "def _is_apple_silicon() -> bool:",
            '    return platform.system().lower() == "darwin" and "arm64" in platform.machine().lower()',
            "",
            "def gpu_family_key() -> str:",
            '    override = os.getenv("NATTEN_MLX_GPU_FAMILY", "").strip()',
            "    if override:",
            "        return override",
            "    if _is_apple_silicon():",
            '        return "apple_silicon"',
            '    return "apple_unknown"',
            "",
            "def token_band(tokens: int) -> str:",
            "    if tokens <= 256: return 'tiny'",
            "    if tokens <= 1024: return 'small'",
            "    if tokens <= 4096: return 'medium'",
            "    return 'large'",
            "",
            "def head_dim_band(head_dim: int) -> str:",
            "    if head_dim <= 16: return 'd16'",
            "    if head_dim <= 32: return 'd32'",
            "    return 'd64p'",
            "",
            "def kernel_band(kernel_size: int) -> str:",
            "    if kernel_size <= 5: return 'k_small'",
            "    if kernel_size <= 9: return 'k_mid'",
            "    return 'k_large'",
            "",
            "def causal_rank_band(causal_rank: int) -> str:",
            "    if causal_rank <= 0: return 'c0'",
            "    if causal_rank == 1: return 'c1'",
            "    return 'c2p'",
            "",
            "def dtype_class(dtype_tag: str) -> str:",
            "    return 'lowp' if dtype_tag in {'fp16', 'bf16'} else 'fp32'",
            "",
            "NANOBIND_THREADGROUP_TABLE = {",
            "    'apple_silicon': {",
        ]
        for key, tg in sorted(threadgroup_table.items(), key=lambda x: str(x[0])):
            lines.append(f"        {key!r}: {tg!r},")
        lines.extend(
            [
                "    },",
                "    'apple_unknown': {},",
                "}",
                "",
                "NANOBIND_SOFTMAX_STRATEGY_TABLE = {",
                "    'apple_silicon': {",
            ]
        )
        for key, strategy in sorted(strategy_table.items(), key=lambda x: str(x[0])):
            lines.append(f"        {key!r}: {strategy!r},")
        lines.extend(
            [
                "    },",
                "    'apple_unknown': {},",
                "}",
                "",
                "def lookup_threadgroup(*, op: str, dtype_tag: str, tokens: int, head_dim: int, kernel_size: int, causal_rank: int, stride_unit: bool):",
                "    table = NANOBIND_THREADGROUP_TABLE.get(gpu_family_key()) or NANOBIND_THREADGROUP_TABLE['apple_unknown']",
                "    key = (op, dtype_class(dtype_tag), token_band(tokens), head_dim_band(head_dim), kernel_band(kernel_size), causal_rank_band(causal_rank), bool(stride_unit))",
                "    return table.get(key)",
                "",
                "def choose_softmax_strategy(*, op: str, dtype_tag: str, tokens: int, kernel_size: int, causal_rank: int, stride_unit: bool) -> str:",
                "    table = NANOBIND_SOFTMAX_STRATEGY_TABLE.get(gpu_family_key()) or NANOBIND_SOFTMAX_STRATEGY_TABLE['apple_unknown']",
                "    key = (op, dtype_class(dtype_tag), token_band(tokens), kernel_band(kernel_size), causal_rank_band(causal_rank), bool(stride_unit))",
                "    v = table.get(key, 'stored')",
                "    return v if v in {'stored', 'recompute'} else 'stored'",
                "",
            ]
        )
        args.output_py.write_text("\n".join(lines), encoding="utf-8")
    finally:
        import os

        os.environ.pop("NATTEN_NANOBIND_TG_OVERRIDE", None)
        os.environ.pop("NATTEN_NANOBIND_SOFTMAX_OVERRIDE", None)
        set_backend("auto")


if __name__ == "__main__":
    main()

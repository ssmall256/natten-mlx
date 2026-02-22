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

    def _make_1d_case(name: str, *, causal: bool) -> dict:
        def _forward():
            return na1d(
                q1,
                k1,
                v1,
                kernel_size=7,
                stride=1,
                dilation=1,
                is_causal=causal,
                scale=0.5,
            )

        def _backward():
            def loss_fn(q_in):
                return mx.sum(
                    na1d(
                        q_in,
                        k1,
                        v1,
                        kernel_size=7,
                        stride=1,
                        dilation=1,
                        is_causal=causal,
                        scale=0.5,
                    )
                )

            return mx.grad(loss_fn)(q1)

        return {"name": name, "forward": _forward, "backward": _backward}

    def _make_2d_case(name: str, *, causal_h: bool, causal_w: bool) -> dict:
        causal = (causal_h, causal_w)

        def _forward():
            return na2d(
                q2,
                k2,
                v2,
                kernel_size=(7, 7),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=causal,
                scale=0.5,
            )

        def _backward():
            def loss_fn(q_in):
                return mx.sum(
                    na2d(
                        q_in,
                        k2,
                        v2,
                        kernel_size=(7, 7),
                        stride=(1, 1),
                        dilation=(1, 1),
                        is_causal=causal,
                        scale=0.5,
                    )
                )

            return mx.grad(loss_fn)(q2)

        return {"name": name, "forward": _forward, "backward": _backward}

    def _make_3d_case(name: str, *, causal_d: bool, causal_h: bool, causal_w: bool) -> dict:
        causal = (causal_d, causal_h, causal_w)

        def _forward():
            return na3d(
                q3,
                k3,
                v3,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                dilation=(1, 1, 1),
                is_causal=causal,
                scale=0.5,
            )

        def _backward():
            def loss_fn(q_in):
                return mx.sum(
                    na3d(
                        q_in,
                        k3,
                        v3,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        dilation=(1, 1, 1),
                        is_causal=causal,
                        scale=0.5,
                    )
                )

            return mx.grad(loss_fn)(q3)

        return {"name": name, "forward": _forward, "backward": _backward}

    return [
        _make_1d_case("na1d_k7_s1_d1_noncausal", causal=False),
        _make_1d_case("na1d_k7_s1_d1_causal", causal=True),
        _make_2d_case("na2d_k7x7_s1_d1_noncausal", causal_h=False, causal_w=False),
        _make_2d_case("na2d_k7x7_s1_d1_causal_h", causal_h=True, causal_w=False),
        _make_3d_case("na3d_k3x3x3_s1_d1_noncausal", causal_d=False, causal_h=False, causal_w=False),
        _make_3d_case("na3d_k3x3x3_s1_d1_causal_d", causal_d=True, causal_h=False, causal_w=False),
    ]


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _ratio(base: float, other: float) -> str:
    if other <= 0.0:
        return "n/a"
    return f"{base / other:.2f}x"


def _to_markdown(payload: dict) -> str:
    lines = []
    lines.append("| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal vs pure | nanobind vs pure | nanobind vs fast_metal |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for case in payload["cases"]:
        case_name = case["name"]
        for direction in ("forward", "backward"):
            p = payload["results"]["pure"][case_name][direction]["median_ms"]
            f = payload["results"]["fast_metal"][case_name][direction]["median_ms"]
            n = payload["results"]["nanobind"][case_name][direction]["median_ms"]
            lines.append(
                f"| `{case_name}` | `{direction}` | {_fmt(p)} | {_fmt(f)} | {_fmt(n)} | {_ratio(p, f)} | {_ratio(p, n)} | {_ratio(f, n)} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final perf table for natten-mlx backends.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--trim-head",
        type=int,
        default=2,
        help="Drop this many measured trials from the head before reporting stats.",
    )
    parser.add_argument("--output-json", default="benchmarks/final-perf.json")
    parser.add_argument("--output-md", default="benchmarks/final-perf.md")
    args = parser.parse_args()
    if args.trim_head < 0:
        raise SystemExit("--trim-head must be >= 0")

    cases = _build_cases()
    backends = ["pure", "fast_metal", "nanobind"]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "warmup": args.warmup,
        "trials": args.trials,
        "trim_head": args.trim_head,
        "cases": [{"name": c["name"]} for c in cases],
        "results": {b: {} for b in backends},
    }

    try:
        for backend in backends:
            set_backend(backend)
            for case in cases:
                payload["results"][backend][case["name"]] = {
                    "forward": _bench(
                        case["forward"],
                        warmup=args.warmup,
                        trials=args.trials,
                        trim_head=args.trim_head,
                    ),
                    "backward": _bench(
                        case["backward"],
                        warmup=args.warmup,
                        trials=args.trials,
                        trim_head=args.trim_head,
                    ),
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

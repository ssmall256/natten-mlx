# natten-mlx

`natten-mlx` brings neighborhood attention (NATTEN-style) to Apple's MLX framework.

## What It Is

- 1D, 2D, and 3D neighborhood attention functional ops for MLX arrays.
- `mlx.nn.Module` wrappers for `NeighborhoodAttention1D`, `NeighborhoodAttention2D`, and `NeighborhoodAttention3D`.
- Backend tiers with runtime dispatch:
  - Tier 0: pure MLX (implemented)
  - Tier 1: fast Metal kernels (fused + split forward/backward paths with pure fallback)
  - Tier 2: nanobind backend (in-tree implementation with optional external extension override)
- Compatibility shims for historical NATTEN API eras (`v014`, `v015`, `v017`, `v020`).

## Installation

```bash
pip install natten-mlx
```

## Quick Start (Modern API)

```python
import mlx.core as mx
from natten_mlx import na1d, NeighborhoodAttention1D

B, L, H, D = 2, 64, 4, 32
q = mx.random.normal((B, L, H, D))
k = mx.random.normal((B, L, H, D))
v = mx.random.normal((B, L, H, D))

out = na1d(q, k, v, kernel_size=7, stride=1, dilation=1, is_causal=False)

x = mx.random.normal((B, L, H * D))
layer = NeighborhoodAttention1D(embed_dim=H * D, num_heads=H, kernel_size=7)
y = layer(x)
```

3D API is also available:

```python
import mlx.core as mx
from natten_mlx import na3d, NeighborhoodAttention3D

q = mx.random.normal((1, 8, 10, 12, 4, 16))
out = na3d(q, q, q, kernel_size=(3, 3, 3))

layer3d = NeighborhoodAttention3D(embed_dim=64, num_heads=4, kernel_size=(3, 3, 3))
x3d = mx.random.normal((1, 8, 10, 12, 64))
y3d = layer3d(x3d)
```

## Compat Mode

```python
import natten_mlx.compat.v014 as natten

layer = natten.NeighborhoodAttention1D(dim=128, kernel_size=7, num_heads=4)
```

Compat shims preserve API names and signatures where possible, but tensor types are `mlx.core.array`, not `torch.Tensor`.

## Semantics Notes

- Parameter validation follows strict NATTEN-style coverage constraints: `dilation * kernel_size <= input_size` per spatial dimension.
- `attn_drop` is supported in:
  - Modern modules: `NeighborhoodAttention1D`, `NeighborhoodAttention2D`, `NeighborhoodAttention3D`
  - v0.14 compat modules: `natten_mlx.compat.v014.NeighborhoodAttention1D`, `NeighborhoodAttention2D`
- When `attn_drop > 0`, modules take the split `qk -> softmax -> dropout -> av` path; otherwise they use fused `na1d` / `na2d` / `na3d`.
- Split `qk/av` kernels in modern modules are stride-aware and causal-aware, so dropout path now supports strided and causal configurations.

## Upstream Parity

- `tests/test_upstream_parity.py` compares v0.14 functional outputs against official `natten==0.14.6`.
- Upstream parity suite also includes split-path gradient parity checks (1D/2D) against official v0.14.
- Required CI job: `upstream-parity-required` in `.github/workflows/backend-matrix.yml`.
- Local run:

```bash
uv sync --extra dev
uv pip install numpy "torch==2.3.1"
uv pip install --no-build-isolation "natten==0.14.6"
NATTEN_UPSTREAM_PARITY=1 uv run python -m pytest tests/test_upstream_parity.py -q
```

## Backend Matrix CI

- Backend matrix tests are gated in `.github/workflows/backend-matrix.yml` with forced:
  - `NATTEN_BACKEND=pure`
  - `NATTEN_BACKEND=fast_metal`
  - `NATTEN_BACKEND=nanobind`
- Includes a required upstream parity job (`NATTEN_UPSTREAM_PARITY=1`) in the same workflow.
- Includes benchmark smoke run with JSON artifact upload and non-failing perf warnings.
- Includes a required backward perf guardrail (fast backends must maintain minimum speedup vs pure).
  - Required CI gate uses sequential median-of-medians aggregation (`--rounds 3`) for stability.
  - Guardrail covers baseline 1D/2D/3D plus decode-like 1D cases (causal and long non-causal).
- Includes a required forward perf guardrail for causal low-precision (`float16`/`bfloat16`) cases.
- Includes low-precision backend parity coverage (`tests/test_low_precision_backend_parity.py`) for `float16` and `bfloat16` (when available), with explicit tolerance thresholds and causal forward cases.
- Local benchmark smoke run:

```bash
uv run python benchmarks/backend_smoke.py --output benchmarks/backend-smoke.json --github-warnings
```

- Local backward perf guardrail run:

```bash
uv run python benchmarks/backward_perf_guardrail.py --output benchmarks/backward-guardrail.json --min-speedup 1.20
```

- Local forward perf guardrail run:

```bash
uv run python benchmarks/forward_perf_guardrail.py --output benchmarks/forward-guardrail.json --min-speedup 1.10
```

## Final Performance Table

Snapshot generated from this repo on:
- Generated at (UTC): `2026-02-22T15:17:00.763383+00:00`
- Platform: `macOS-26.3-arm64-arm-64bit`
- Python: `3.11.11`
- Command:

```bash
uv run python benchmarks/final_perf_table.py --warmup 5 --trials 25 --output-json benchmarks/final-perf.json --output-md benchmarks/final-perf.md
```

Benchmarks report trimmed statistics by default (`--trim-head 2`) to reduce cold-trial noise; JSON artifacts also retain full raw metrics (`raw_*`).

Median latency table (ms, lower is better; includes both noncausal and causal configurations):

| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.838 | 0.173 | 0.174 | 4.85x | 4.80x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.919 | 0.279 | 0.649 | 3.29x | 1.42x |
| `na1d_k7_s1_d1_causal` | `forward` | 0.703 | 0.172 | 0.175 | 4.09x | 4.01x |
| `na1d_k7_s1_d1_causal` | `backward` | 0.874 | 0.272 | 0.591 | 3.21x | 1.48x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.722 | 0.260 | 0.294 | 6.63x | 5.86x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 1.905 | 0.469 | 0.905 | 4.07x | 2.10x |
| `na2d_k7x7_s1_d1_causal_h` | `forward` | 1.421 | 0.260 | 0.392 | 5.47x | 3.62x |
| `na2d_k7x7_s1_d1_causal_h` | `backward` | 1.870 | 0.479 | 1.808 | 3.90x | 1.03x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.834 | 0.184 | 0.286 | 4.53x | 2.91x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 0.981 | 0.316 | 0.697 | 3.10x | 1.41x |
| `na3d_k3x3x3_s1_d1_causal_d` | `forward` | 0.862 | 0.171 | 0.177 | 5.05x | 4.86x |
| `na3d_k3x3x3_s1_d1_causal_d` | `backward` | 0.946 | 0.326 | 0.765 | 2.90x | 1.24x |

Raw artifacts are written to:
- `benchmarks/final-perf.json`
- `benchmarks/final-perf.md`

## natten-mlx vs natten-mps

- Use `natten-mlx` for MLX-native projects.
- Use `natten-mps` for PyTorch + MPS projects.

## Support Matrix

```python
import natten_mlx
print(natten_mlx.get_support_matrix())
```

Current design targets three tiers:
- Metal kernels (nanobind tier): supported via in-tree nanobind backend implementation, with optional override to an external extension.
- MLX fast Metal kernels: fused and split forward/backward paths for covered configurations, with automatic fallback.
- Pure MLX: full semantic coverage baseline.

Nanobind tier resolution order:
1. `NATTEN_MLX_NANOBIND_MODULE` override (if set)
2. compiled in-tree extension: `natten_mlx._core._nanobind_ext`
3. in-tree Python fallback: `natten_mlx._core._nanobind_impl`

Nanobind availability semantics:
- `natten_mlx.has_nanobind()` reports compiled-extension availability only.
- If compiled extension is unavailable and backend is explicitly set to `nanobind`,
  in-tree fallback still runs for correctness.

To build the in-tree nanobind extension locally:

```bash
uv pip install nanobind
NATTEN_MLX_BUILD_NANOBIND=1 uv pip install --no-build-isolation -e .
```

Audit provenance for this synthesis: `BACKEND_SYNTHESIS.md`.

Backward support across backends uses explicit backend backward entrypoints for fused and split paths, with pure fallback as safety.

## Limitations

- Fast Metal split acceleration eligibility is strict:
  - `stride>=1` and `dilation>=1` on each active spatial axis, with causal and non-causal supported.
  - Kernel shape must match operator dimensionality: odd `K` (1D), square odd `(K, K)` (2D), cubic odd `(K, K, K)` (3D).
- Fast Metal fused acceleration eligibility follows the same per-axis stride/dilation and causal rules, with the same odd/square/cubic kernel-shape requirement by dimensionality.
- Unsupported accelerated configurations fall back to pure backend for correctness.

## Runtime Notes

- Nanobind tier uses its own in-tree Metal-kernel implementation path with pure fallback safety; it does not delegate to `fast_metal`.
- Compiled nanobind extension Stage B runs `na1d` / `na2d` / `na3d` forward as fused-first from C++ entrypoints, with fallback chain `fused -> split -> pure`.
- Compiled nanobind extension Stage B runs `na1d` / `na2d` / `na3d` backward as fused-first staged pipeline (`qk -> softmax grad -> qk/av backward`) in C++ entrypoints, with fallback chain `fused -> split -> pure`.
- `NATTEN_NANOBIND_NATIVE_RUNTIME` controls compiled nanobind runtime routing: default native C++/Metal path is on; set `NATTEN_NANOBIND_NATIVE_RUNTIME=0` to force Python bridge mode for debugging/regression bisects.
- Fast Metal forward no longer forces `float32` materialization for `float16`/`bfloat16` inputs; kernels accumulate in `float32` and outputs are cast back to input dtype.
- Fast Metal low-precision dtype routing stays native by default; optional shape-aware fp32 fallback can be enabled with `NATTEN_MLX_FORWARD_LOWP_FP32_ROUTE=1` (currently a narrow `bfloat16` 2D fused causal/K9 small-shape rule).
- Fast Metal vec4 forward kernels now cover split `na1d_qk`/`na1d_av`, split `na2d_qk`, split `na3d_qk`, and fused `na1d`/`na2d`/`na3d` when `head_dim % 4 == 0` (including causal paths).
- Fused 2D/3D kernels now use packed per-neighbor linear indices and in-place softmax weight storage to reduce per-thread local-memory footprint.
- 2D/3D split `qk_backward` now uses inverse-map `grad_k` kernels (matching the 1D strategy) to avoid reverse-search hotspot behavior.
- The required backward guardrail includes dedicated split `qk_backward` 2D/3D `grad_k` hotspot cases.
- Experimental AV-backward fusion kernels (`NATTEN_MLX_AV_BWD_FUSION=1`) now use inverse-map `grad_v` accumulation, but default remains split AV backward because fused is not consistently faster across representative 1D/2D/3D shapes.
- MLX lazy evaluation applies; this package does not force evaluation.

## License

MIT

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
- Generated at (UTC): `2026-02-22T20:18:49.989965+00:00`
- Platform: `macOS-26.3-arm64-arm-64bit`
- Python: `3.11.8`
- Command:

```bash
uv run python benchmarks/final_perf_table.py --warmup 10 --trials 25 --output-json benchmarks/final-perf.json --output-md benchmarks/final-perf.md
```

Benchmarks report trimmed statistics by default (`--trim-head 2`) to reduce cold-trial noise; JSON artifacts also retain full raw metrics (`raw_*`).

Median latency table (ms, lower is better; includes both noncausal and causal configurations):

| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal vs pure | nanobind vs pure | nanobind vs fast_metal |
|---|---:|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.824 | 0.165 | 0.155 | 5.01x | 5.32x | 1.06x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.461 | 0.308 | 0.186 | 1.50x | 2.47x | 1.65x |
| `na1d_k7_s1_d1_causal` | `forward` | 0.306 | 0.185 | 0.141 | 1.65x | 2.17x | 1.31x |
| `na1d_k7_s1_d1_causal` | `backward` | 0.359 | 0.284 | 0.188 | 1.26x | 1.91x | 1.51x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.397 | 0.279 | 0.248 | 5.00x | 5.64x | 1.13x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 1.789 | 0.478 | 0.324 | 3.74x | 5.52x | 1.47x |
| `na2d_k7x7_s1_d1_causal_h` | `forward` | 1.386 | 0.252 | 0.250 | 5.50x | 5.55x | 1.01x |
| `na2d_k7x7_s1_d1_causal_h` | `backward` | 1.780 | 0.495 | 0.317 | 3.60x | 5.61x | 1.56x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.826 | 0.203 | 0.180 | 4.06x | 4.60x | 1.13x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 0.966 | 0.327 | 0.214 | 2.95x | 4.51x | 1.53x |
| `na3d_k3x3x3_s1_d1_causal_d` | `forward` | 0.823 | 0.186 | 0.175 | 4.43x | 4.69x | 1.06x |
| `na3d_k3x3x3_s1_d1_causal_d` | `backward` | 0.946 | 0.339 | 0.206 | 2.79x | 4.58x | 1.64x |

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

Backward support across backends uses v2 primitive backward with pure fallback as safety.

## Limitations

- Metal kernel acceleration requires odd kernel sizes within dimension-specific limits (1D: K≤63, 2D: K≤13, 3D: K≤7).
- `stride>=1` and `dilation>=1` on each active spatial axis, with causal and non-causal supported.
- Unsupported configurations fall back to pure backend for correctness.

## Runtime Notes

- Nanobind tier uses v2 `mx::Primitive` subclasses for lazy MLX graph integration with pure fallback; it does not delegate to `fast_metal`.
- Dispatch chain: v2 primitive (Metal kernels) → pure MLX fallback.
- `NATTEN_NANOBIND_DISABLE_V2=1` bypasses v2 primitives and falls back to pure.
- MLX lazy evaluation applies; this package does not force evaluation.

## Acknowledgments

This project implements the neighborhood attention mechanism introduced by [NATTEN](https://github.com/SHI-Labs/NATTEN) (SHI-Labs), ported to Apple's MLX framework with custom Metal kernels. The original NATTEN library and the research behind it are by Ali Hassani, Steven Walton, Humphrey Shi, and collaborators.

If you use neighborhood attention in research, please cite the original papers:

- Hassani et al., "Neighborhood Attention Transformer" (CVPR 2023)
- Hassani & Shi, "Dilated Neighborhood Attention Transformer" (2022)
- Hassani et al., "Faster Neighborhood Attention" (NeurIPS 2024)

## License

MIT — see [LICENSE](LICENSE) for details.

NATTEN (the original PyTorch library) is also MIT-licensed.

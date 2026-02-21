# natten-mlx

`natten-mlx` brings neighborhood attention (NATTEN-style) to Apple's MLX framework.

## What It Is

- 1D and 2D neighborhood attention functional ops for MLX arrays.
- 3D neighborhood attention functional ops for MLX arrays.
- `mlx.nn.Module` wrappers for `NeighborhoodAttention1D`, `NeighborhoodAttention2D`, and `NeighborhoodAttention3D`.
- Backend tiers with runtime dispatch:
  - Tier 0: pure MLX (implemented)
  - Tier 1: fast Metal kernels (fused + split forward paths with pure fallback)
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
  - Modern modules: `NeighborhoodAttention1D`, `NeighborhoodAttention2D`
  - v0.14 compat modules: `natten_mlx.compat.v014.NeighborhoodAttention1D`, `NeighborhoodAttention2D`
- When `attn_drop > 0`, modules take the split `qk -> softmax -> dropout -> av` path; otherwise they use fused `na1d` / `na2d`.
- Split `qk/av` kernels in modern modules are stride-aware and causal-aware, so dropout path now supports strided and causal configurations.

## Upstream Parity

- `tests/test_upstream_parity.py` compares v0.14 functional outputs against official `natten==0.14.6`.
- Upstream parity suite also includes split-path gradient parity checks (1D/2D) against official v0.14.
- CI workflow: `.github/workflows/upstream-parity.yml`.
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
- Includes benchmark smoke run with JSON artifact upload and non-failing perf warnings.
- Local benchmark smoke run:

```bash
uv run python benchmarks/backend_smoke.py --output benchmarks/backend-smoke.json --github-warnings
```

## Final Performance Table

Snapshot generated from this repo on:
- Platform: `macOS-26.3-arm64-arm-64bit`
- Python: `3.11.11`
- Command:

```bash
uv run python benchmarks/final_perf_table.py --warmup 5 --trials 25 --output-json benchmarks/final-perf.json --output-md benchmarks/final-perf.md
```

Median latency table (ms, lower is better):

| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.738 | 0.214 | 0.215 | 3.44x | 3.44x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 1.006 | 0.694 | 0.716 | 1.45x | 1.40x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.739 | 0.682 | 0.686 | 2.55x | 2.54x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 1.994 | 2.387 | 2.399 | 0.84x | 0.83x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.890 | 0.280 | 0.293 | 3.18x | 3.04x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 1.011 | 2.030 | 2.120 | 0.50x | 0.48x |

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
- MLX fast Metal kernels: fused and split forward paths for common configurations, with automatic fallback.
- Pure MLX: full semantic coverage baseline.

Nanobind tier resolution order:
1. `NATTEN_MLX_NANOBIND_MODULE` override (if set)
2. compiled in-tree extension: `natten_mlx._core._nanobind_ext`
3. in-tree Python fallback: `natten_mlx._core._nanobind_impl`

To build the in-tree nanobind extension locally:

```bash
uv pip install nanobind
NATTEN_MLX_BUILD_NANOBIND=1 uv pip install --no-build-isolation -e .
```

Audit provenance for this synthesis: `BACKEND_SYNTHESIS.md`.

Backward support across backends uses explicit backend backward entrypoints for fused and split paths, with pure fallback as safety.

## Limitations

- Fast Metal fused 3D is not implemented; 3D uses split acceleration where eligible.
- Fast Metal split acceleration eligibility is strict:
  - 1D split: odd `K`, `stride>=1`, `dilation>=1`, causal/non-causal supported.
  - 2D split: square odd `K`, per-axis `stride>=1`, equal dilations, per-axis causal/non-causal supported.
  - 3D split: cubic odd `K`, per-axis `stride>=1`, equal dilations, per-axis causal/non-causal supported.
- Fast Metal fused acceleration eligibility:
  - 1D fused: odd `K`, `stride>=1`, `dilation>=1`, causal/non-causal supported.
  - 2D fused: square odd `K`, per-axis `stride>=1`, per-axis `dilation>=1`, per-axis causal/non-causal supported.
- Unsupported accelerated configurations fall back to pure backend for correctness.
- Nanobind tier delegates to fast-metal where available (same exact eligibility constraints), otherwise pure fallback.
- MLX lazy evaluation applies; this package does not force evaluation.

## License

MIT

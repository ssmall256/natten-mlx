# natten-mlx

`natten-mlx` brings neighborhood attention (NATTEN-style) to Apple's MLX framework.

## What It Is

- 1D and 2D neighborhood attention functional ops for MLX arrays.
- `mlx.nn.Module` wrappers for `NeighborhoodAttention1D` and `NeighborhoodAttention2D`.
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

Audit provenance for this synthesis: `BACKEND_SYNTHESIS.md`.

Backward support for end-to-end `na1d` / `na2d` is preserved across backends by using pure-semantic custom VJP when accelerated backends are active.

## Limitations

- No 3D neighborhood attention yet.
- Fast Metal fused acceleration currently targets non-causal, stride-1, K in `{3,5,7}` (2D additionally requires square kernel and equal dilations). Other configurations use pure backend for correctness.
- Nanobind tier delegates to fast-metal where available (same fused-path constraints), otherwise pure fallback.
- MLX lazy evaluation applies; this package does not force evaluation.

## License

MIT

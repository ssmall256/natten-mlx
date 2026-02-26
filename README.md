# natten-mlx

Neighborhood attention ([NATTEN](https://github.com/SHI-Labs/NATTEN)) for Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx).

## Why this exists

Upstream NATTEN provides CUDA kernels but dropped macOS support after v0.14. If you work in the MLX ecosystem on Apple Silicon — for training, fine-tuning, or inference — there was no GPU-accelerated neighborhood attention.

`natten-mlx` fills that gap with hand-tuned Metal kernels and compiled nanobind primitives for MLX, covering 1D, 2D, and 3D neighborhood attention with full backward pass support. For PyTorch MPS workflows, see the sibling project [natten-mps](https://github.com/ssmall256/natten-mps).

[Installation](#installation) | [Quick Start](#quick-start) | [Features](#features) | [Backends](#backends) | [Performance](#performance) | [Limitations](#limitations) | [Acknowledgments](#acknowledgments) | [License](#license)

## Installation

```bash
pip install natten-mlx
```

Requires Python 3.10+ and MLX >= 0.30.3 (Apple Silicon / macOS only).

To build the optional nanobind extension (for Tier 2 performance):

```bash
uv pip install nanobind
NATTEN_MLX_BUILD_NANOBIND=1 uv pip install --no-build-isolation -e .
```

## Quick Start

### Functional API

```python
import mlx.core as mx
from natten_mlx import na1d

B, L, H, D = 2, 64, 4, 32
q = mx.random.normal((B, L, H, D))
k = mx.random.normal((B, L, H, D))
v = mx.random.normal((B, L, H, D))

out = na1d(q, k, v, kernel_size=7)
```

### Module API

```python
from natten_mlx import NeighborhoodAttention1D

layer = NeighborhoodAttention1D(embed_dim=128, num_heads=4, kernel_size=7)
x = mx.random.normal((2, 64, 128))
y = layer(x)
```

### Split QK/AV

For models that need access to attention weights (e.g., for dropout or visualization):

```python
from natten_mlx import na1d_qk, na1d_av

logits = na1d_qk(q, k, kernel_size=7, scale=D ** -0.5)  # [B, L, H, K]
attn = mx.softmax(logits, axis=-1)
out = na1d_av(attn, v, kernel_size=7)                     # [B, L, H, D]
```

### 2D and 3D

```python
from natten_mlx import na2d, na3d

# 2D: [B, H, W, heads, head_dim]
out_2d = na2d(q2d, k2d, v2d, kernel_size=7)

# 3D: [B, D, H, W, heads, head_dim]
out_3d = na3d(q3d, k3d, v3d, kernel_size=(3, 3, 3))
```

### Variable-length attention

```python
import mlx.core as mx
from natten_mlx import na1d_varlen, na2d_varlen

# 1D: padded batch with per-sample lengths
q = mx.random.normal((3, 128, 4, 32))   # B=3, L_max=128
k = mx.random.normal((3, 128, 4, 32))
v = mx.random.normal((3, 128, 4, 32))
seq_lens = mx.array([128, 96, 64])
out = na1d_varlen(q, k, v, kernel_size=7, seq_lens=seq_lens)

# 2D: padded batch with per-sample (H, W)
q2d = mx.random.normal((2, 32, 32, 4, 32))  # B=2, H_max=32, W_max=32
k2d = mx.random.normal((2, 32, 32, 4, 32))
v2d = mx.random.normal((2, 32, 32, 4, 32))
spatial_sizes = mx.array([[32, 32], [24, 20]])
out2d = na2d_varlen(q2d, k2d, v2d, kernel_size=7, spatial_sizes=spatial_sizes)
```

## Features

- **1D, 2D, and 3D** neighborhood attention ops with full dilation, stride, and causal support.
- **Variable-length (varlen)** attention — padded batches with per-sample spatial sizes, Metal-accelerated for all ranks.
- **`nn.Module` wrappers**: `NeighborhoodAttention1D`, `2D`, `3D` — drop-in attention layers.
- **Three backend tiers** with automatic dispatch:
  - **Pure MLX** — full coverage baseline, no compilation needed.
  - **Fast Metal** — hand-tuned Metal kernels for forward and backward.
  - **Nanobind** — compiled C++/Metal primitives via `mx::Primitive` for maximum performance.
- **float32, float16, and bfloat16** support.
- **GQA / MQA** — grouped-query and multi-query attention via `num_kv_heads` (nn modules) or mismatched Q/KV head counts (functional API).
- **`return_lse`** — return log-sum-exp alongside output for gradient checkpointing and attention merging.
- **`additional_keys` / `additional_values`** — prepend extra global tokens that every query attends to.
- **`merge_attentions`** — numerically stable sigmoid-based merge of multiple attention outputs.
- **FMHA fast path** — auto-dispatches to `mx.fast.scaled_dot_product_attention` when kernel covers the full spatial extent.
- **Extras** — model-specific fused Metal kernels (e.g., `extras.allin1` for DiNAT with fused QK+RPB).
- **Compat shims** for historical NATTEN API versions (`v014`, `v015`, `v017`, `v020`).

### New features usage

```python
import mlx.core as mx
from natten_mlx import na1d, merge_attentions

# GQA: 8 query heads, 2 KV heads
q = mx.random.normal((1, 128, 8, 32))
k = mx.random.normal((1, 128, 2, 32))
v = mx.random.normal((1, 128, 2, 32))
out = na1d(q, k, v, kernel_size=7)

# return_lse for merging
out1, lse1 = na1d(q, k, v, kernel_size=7, return_lse=True)
out2, lse2 = na1d(q, k, v, kernel_size=7, return_lse=True)
merged, merged_lse = merge_attentions([out1, out2], [lse1, lse2])

# Additional global tokens
add_k = mx.random.normal((1, 4, 2, 32))
add_v = mx.random.normal((1, 4, 2, 32))
out = na1d(q, k, v, kernel_size=7, additional_keys=add_k, additional_values=add_v)

# GQA via nn module
from natten_mlx import NeighborhoodAttention1D
layer = NeighborhoodAttention1D(embed_dim=256, num_heads=8, kernel_size=7, num_kv_heads=2)
```

## Extras: Model-Specific Fused Kernels

The `extras/` namespace provides model-specific optimized kernels that go beyond the core NATTEN ops. These fuse operations like QK + relative position bias (RPB) into single Metal kernel dispatches for maximum performance.

### `extras.allin1` — DiNAT Attention

Fused QK+RPB and AV kernels tuned for [DiNAT](https://arxiv.org/abs/2209.15001) models (k=3/5/7, D=12). Used by [all-in-one-mlx](https://github.com/ssmall256/all-in-one-mlx).

```python
from natten_mlx.extras.allin1 import na1d_qk_rpb, na1d_av_fused

# Fused QK + relative position bias in one Metal kernel dispatch
logits = na1d_qk_rpb(q, k, rpb, kernel_size=5, dilation=2, scale=0.288)
attn = mx.softmax(logits, axis=-1)
out = na1d_av_fused(attn, v, kernel_size=5, dilation=2)
```

Layout is spatial-first `[B, L, H, D]` — transposition to the internal heads-first layout is handled automatically.

The same `extras/` pattern exists in `natten-mps` for PyTorch MPS projects, with identical function signatures:

```python
# Same API, different backend:
from natten_mlx.extras.allin1 import na1d_qk_rpb  # MLX
from natten_mps.extras.allin1 import na1d_qk_rpb   # PyTorch MPS
```

## Compat Mode

API shims for code written against historical NATTEN versions:

```python
import natten_mlx.compat.v014 as natten

layer = natten.NeighborhoodAttention1D(dim=128, kernel_size=7, num_heads=4)
```

Tensor types are `mlx.core.array`, not `torch.Tensor`. Compat shims cover `v014`, `v015`, `v017`, and `v020`.

## Backends

```python
import natten_mlx

print(natten_mlx.get_support_matrix())
print(natten_mlx.has_nanobind())  # True if compiled extension is available
```

Backend dispatch order: **nanobind** (compiled Metal primitives) > **fast_metal** (MLX Metal kernels) > **pure** (MLX ops). Unsupported configurations fall back automatically.

Override with environment variables:

```bash
NATTEN_BACKEND=pure           # Force pure MLX
NATTEN_BACKEND=fast_metal     # Force fast Metal
NATTEN_BACKEND=nanobind       # Force nanobind
```

## Performance

Median latency (ms, lower is better) on Apple Silicon:

| Case | Direction | pure | fast_metal | nanobind | speedup vs pure |
|------|-----------|-----:|-----------:|---------:|----------------:|
| 1D k=7 noncausal | fwd | 0.82 | 0.17 | **0.16** | **5.3x** |
| 1D k=7 noncausal | bwd | 0.46 | 0.31 | **0.19** | **2.5x** |
| 2D k=7 noncausal | fwd | 1.40 | 0.28 | **0.25** | **5.6x** |
| 2D k=7 noncausal | bwd | 1.79 | 0.48 | **0.32** | **5.5x** |
| 3D k=3 noncausal | fwd | 0.83 | 0.20 | **0.18** | **4.6x** |
| 3D k=3 noncausal | bwd | 0.97 | 0.33 | **0.21** | **4.5x** |

Full benchmarks with causal configurations: `benchmarks/final-perf.md`.

### Cross-framework: natten-mlx vs natten-mps

Apple Silicon (M-series), fp32, B=1 H=4 D=32, Metal-accelerated:

| Config | natten-mlx fwd | natten-mps fwd | natten-mlx bwd | natten-mps bwd |
|---|---|---|---|---|
| 1D L=256 K=7 | 0.21 ms | 0.25 ms | 0.14 ms | 0.39 ms |
| 1D L=1024 K=7 | 0.27 ms | 0.40 ms | 0.26 ms | 0.63 ms |
| 2D 32×32 K=7 | 0.65 ms | 0.88 ms | 1.02 ms | 1.62 ms |
| 2D 64×64 K=7 | 1.13 ms | 1.32 ms | 0.97 ms | 1.55 ms |
| 2D 32×32 K=7 causal | 0.29 ms | 0.37 ms | 0.31 ms | 0.49 ms |
| 3D 16³ K=3 | 0.43 ms | 0.55 ms | 0.50 ms | 0.89 ms |

MLX's compiled Metal primitives have lower dispatch overhead than PyTorch MPS, giving a consistent 1.2–1.5× forward advantage and 1.6–2.8× backward advantage. Both are orders of magnitude faster than pure-framework baselines.

### Variable-length (varlen) attention

Metal-accelerated varlen forward, fp32:

| Config | natten-mlx | natten-mps | MLX speedup |
|---|---|---|---|
| varlen 1D B=4 L=128 K=7 | 0.53 ms | 1.74 ms | 3.3× |
| varlen 1D B=4 L=256 K=7 | 0.51 ms | 1.74 ms | 3.4× |
| varlen 2D B=2 16×16 K=3 | 0.82 ms | 2.39 ms | 2.9× |
| varlen 2D B=2 32×32 K=7 | 1.23 ms | 3.79 ms | 3.1× |
| varlen 3D B=2 8³ K=3 | 1.55 ms | 3.82 ms | 2.5× |

Both projects now support GPU-accelerated varlen for all ranks (1D/2D/3D). Backward pass uses per-sample autograd re-differentiation through the standard Metal-accelerated `na*d` kernels.

### Apple Silicon vs CUDA GPUs — backward pass

NATTEN's CUDA backward pass has known performance issues for 3D and large 2D workloads. Apple Silicon backward passes are competitive with — and sometimes faster than — datacenter GPUs:

| Config | natten-mlx bwd | A100 CUDA bwd |
|---|---|---|
| 3D 32³ K=3 | 5.7 ms | 458 ms (default) / 11.8 ms (KV-parallel) |

The 3D backward advantage comes from our optimized gradient kernels. NATTEN's default CUDA 3D backward has a known O(n³·K³) scaling issue; even with the KV-parallel fix, Apple Silicon matches A100 performance. See NATTEN GitHub issues [#157](https://github.com/SHI-Labs/NATTEN/issues/157) (A100/H100 3D backward) and [#161](https://github.com/SHI-Labs/NATTEN/issues/161) (A40 2D/3D backward) for the CUDA reference numbers.

## Limitations

- Metal kernel acceleration requires odd kernel sizes (1D: K≤63, 2D: K≤13, 3D: K≤7).
- Unsupported kernel sizes or configurations fall back to pure MLX.
- macOS only (Apple Silicon required for Metal backends).

## natten-mlx vs natten-mps

| | natten-mlx | natten-mps |
|---|---|---|
| Framework | MLX | PyTorch |
| Device | Apple Silicon (Metal) | Apple Silicon (MPS) |
| Use when | MLX-native projects | PyTorch + MPS projects |

Both packages provide the same `extras/` namespace for model-specific kernels.

## Development

```bash
# Run tests
uv sync --extra dev
uv run python -m pytest tests/ -q

# Run benchmarks
uv run python benchmarks/final_perf_table.py --warmup 10 --trials 25

# Upstream parity (requires torch + natten)
uv pip install numpy "torch==2.3.1"
uv pip install --no-build-isolation "natten==0.14.6"
NATTEN_UPSTREAM_PARITY=1 uv run python -m pytest tests/test_upstream_parity.py -q
```

CI runs backend matrix tests, upstream parity checks, and performance guardrails on every push. See `.github/workflows/backend-matrix.yml`.

## Acknowledgments

This project implements the neighborhood attention mechanism introduced by [NATTEN](https://github.com/SHI-Labs/NATTEN) (SHI-Labs), ported to Apple's MLX framework with custom Metal kernels. The original NATTEN library and the research behind it are by Ali Hassani, Steven Walton, Humphrey Shi, and collaborators.

If you use neighborhood attention in research, please cite the original papers:

- Hassani et al., "Neighborhood Attention Transformer" (CVPR 2023)
- Hassani & Shi, "Dilated Neighborhood Attention Transformer" (2022)
- Hassani et al., "Faster Neighborhood Attention" (NeurIPS 2024)

## License

MIT — see [LICENSE](LICENSE) for details.

NATTEN (the original PyTorch library) is also MIT-licensed.

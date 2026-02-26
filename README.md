# natten-mlx

Neighborhood Attention ([NATTEN](https://github.com/SHI-Labs/NATTEN)) for Apple Silicon — built on [MLX](https://github.com/ml-explore/mlx).

> **Disclaimer (unofficial):** This is an independent, unofficial implementation/port for Apple Silicon.  
> **Not affiliated with** SHI-Labs or the upstream NATTEN project.

Neighborhood Attention was introduced by the NATTEN authors. If you use Neighborhood Attention in research, please cite the original papers (see [Acknowledgments](#acknowledgments)).

> **v0.x** — API may change between minor versions. Pin your dependency for production use.

---

## Why this exists

Upstream NATTEN is CUDA-focused and targets NVIDIA GPUs. In the MLX ecosystem on Apple Silicon, there isn’t an official GPU-accelerated Neighborhood Attention implementation to drop in.

**natten-mlx** fills that gap with:
- **MLX-native ops** (pure MLX baseline)
- **hand-tuned Metal kernels** for fast acceleration
- an optional **nanobind** extension for the lowest dispatch overhead and best latency

For PyTorch + MPS workflows, see the sibling project: **[natten-mps](https://github.com/ssmall256/natten-mps)**.

**Jump to:** [Installation](#installation) | [Quick start](#quick-start) | [Features](#features) | [Backends](#backends) | [Performance](#performance) | [Limitations](#limitations) | [Acknowledgments](#acknowledgments)

---

## Use natten-mlx if…

- You’re building in **MLX**
- You want **Metal acceleration** for 1D/2D/3D neighborhood attention
- You want the option of a compiled path (**nanobind**) for best latency

---

## Installation

```bash
pip install natten-mlx
```

Requirements:
- Python 3.10+
- MLX >= 0.30.3
- Apple Silicon / macOS (Metal backends)

### Optional: build the nanobind extension

The nanobind tier is optional. It provides compiled MLX primitives with the lowest dispatch overhead.

```bash
uv pip install nanobind
NATTEN_MLX_BUILD_NANOBIND=1 uv pip install --no-build-isolation -e .
```

---

## Quick start

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
import mlx.core as mx
from natten_mlx import NeighborhoodAttention1D

layer = NeighborhoodAttention1D(embed_dim=128, num_heads=4, kernel_size=7)
x = mx.random.normal((2, 64, 128))  # [B, L, C]
y = layer(x)
```

### Split QK / AV (access attention weights)

```python
import mlx.core as mx
from natten_mlx import na1d_qk, na1d_av

B, L, H, D = 2, 64, 4, 32
q = mx.random.normal((B, L, H, D))
k = mx.random.normal((B, L, H, D))
v = mx.random.normal((B, L, H, D))

logits = na1d_qk(q, k, kernel_size=7, scale=D ** -0.5)  # [B, L, H, K]
attn = mx.softmax(logits, axis=-1)
out = na1d_av(attn, v, kernel_size=7)                   # [B, L, H, D]
```

### 2D and 3D

```python
from natten_mlx import na2d, na3d

# 2D: [B, H, W, heads, head_dim]
out_2d = na2d(q2d, k2d, v2d, kernel_size=7)

# 3D: [B, D, H, W, heads, head_dim]
out_3d = na3d(q3d, k3d, v3d, kernel_size=(3, 3, 3))
```

---

## Features

Core:
- **1D / 2D / 3D** neighborhood attention (fused and split QK/AV ops)
- **Causal masking** with per-axis control
- **Stride** for downsampling
- **Non-uniform kernels** — per-axis kernel sizes and dilations for 2D/3D

Batching / advanced:
- **Variable-length (varlen)** attention — padded batches with per-sample spatial sizes
- **GQA / MQA** (`num_kv_heads`)
- **additional_keys / additional_values** — prepend extra global tokens that every query attends to
- **merge_attentions** — numerically stable sigmoid-based merge of multiple attention outputs
- **return_lse** for stable merges and downstream composition
- **FMHA fast path** — dispatches to `mx.fast.scaled_dot_product_attention` when the kernel covers the full spatial extent

Extras:
- **`extras/`** namespace for model-specific fused kernels (e.g., DiNAT-style fused QK+RPB paths)

Compatibility:
- **Compat shims** for historical NATTEN API versions (`v014`, `v015`, `v017`, `v020`)

---

## Extras: Model-Specific Fused Kernels

The `extras/` namespace provides model-specific optimized kernels that go beyond the core NATTEN ops. These fuse operations like QK + relative position bias (RPB) into single Metal kernel dispatches for maximum performance.

### `extras.allin1` — DiNAT Attention

Fused QK+RPB and AV kernels tuned for [DiNAT](https://arxiv.org/abs/2209.15001) models (k=3/5/7, D=12).

```python
from natten_mlx.extras.allin1 import na1d_qk_rpb, na1d_av_fused

logits = na1d_qk_rpb(q, k, rpb, kernel_size=5, dilation=2, scale=0.288)
attn = mx.softmax(logits, axis=-1)
out = na1d_av_fused(attn, v, kernel_size=5, dilation=2)
```

The same `extras/` pattern exists in `natten-mps` for PyTorch MPS projects, with identical function signatures.

---

## Compat mode

API shims for code written against historical NATTEN versions:

```python
import natten_mlx.compat.v014 as natten

layer = natten.NeighborhoodAttention1D(dim=128, kernel_size=7, num_heads=4)
```

Note: tensor types are `mlx.core.array` (not `torch.Tensor`).

---

## Backends

```python
import natten_mlx

print(natten_mlx.get_support_matrix())
print(natten_mlx.has_nanobind())  # True if compiled extension is available
```

Backend dispatch order:
**nanobind** (compiled MLX primitives) > **fast_metal** (MLX Metal kernels) > **pure** (MLX ops)

Override with environment variables:

```bash
NATTEN_BACKEND=pure           # Force pure MLX
NATTEN_BACKEND=fast_metal     # Force fast Metal
NATTEN_BACKEND=nanobind       # Force nanobind (requires extension)
```

---

## Performance

Median latency (ms, lower is better) on Apple Silicon:

| Case | Direction | pure | fast_metal | nanobind | speedup vs pure |
|------|-----------|-----:|-----------:|---------:|----------------:|
| 1D k=7 noncausal | fwd | 0.82 | 0.17 | **0.16** | **5.3×** |
| 1D k=7 noncausal | bwd | 0.46 | 0.31 | **0.19** | **2.5×** |
| 2D k=7 noncausal | fwd | 1.40 | 0.28 | **0.25** | **5.6×** |
| 2D k=7 noncausal | bwd | 1.79 | 0.48 | **0.32** | **5.5×** |
| 3D k=3 noncausal | fwd | 0.83 | 0.20 | **0.18** | **4.6×** |
| 3D k=3 noncausal | bwd | 0.97 | 0.33 | **0.21** | **4.5×** |

Full benchmarks (including causal): `benchmarks/final-perf.md`.

### Cross-framework: natten-mlx vs natten-mps

Apple Silicon (M-series), fp32, B=1 H=4 D=32, Metal-accelerated:

| Config | natten-mlx fwd | natten-mps fwd | natten-mlx bwd | natten-mps bwd |
|---|---:|---:|---:|---:|
| 1D L=256 K=7 | 0.21 ms | 0.25 ms | 0.14 ms | 0.39 ms |
| 1D L=1024 K=7 | 0.27 ms | 0.40 ms | 0.26 ms | 0.63 ms |
| 2D 32×32 K=7 | 0.65 ms | 0.88 ms | 1.02 ms | 1.62 ms |
| 2D 64×64 K=7 | 1.13 ms | 1.32 ms | 0.97 ms | 1.55 ms |
| 2D 32×32 K=7 causal | 0.29 ms | 0.37 ms | 0.31 ms | 0.49 ms |
| 3D 16³ K=3 | 0.43 ms | 0.55 ms | 0.50 ms | 0.89 ms |

MLX’s compiled primitives generally have lower dispatch overhead than PyTorch MPS, so natten-mlx is often faster at the same shapes. Both are dramatically faster than pure-framework baselines.

### Notes on CUDA backward performance (context)

For certain large 2D/3D configurations, upstream CUDA backward performance has been actively discussed in NATTEN issue threads. The CUDA numbers referenced in this repository are sourced from upstream GitHub issues (e.g. [#157](https://github.com/SHI-Labs/NATTEN/issues/157), [#161](https://github.com/SHI-Labs/NATTEN/issues/161)) and are **not** intended as a controlled, apples-to-apples benchmark. They are included as context for shapes where backward performance has been publicly investigated.

### Methodology

All timings on **Apple M4 Max**, macOS 26.3, Python 3.11, MLX 0.30.6, float32. Each kernel is warmed up for 5 iterations, then timed for 20 trials; the first 2 are trimmed and the reported value is the **median** of the remaining 18. `mx.eval()` is called before each measurement to ensure synchronization. Reproduce with `python benchmarks/final_perf_table.py`.

---

## Limitations

- **Odd kernel sizes only** for neighborhood attention (this matches upstream NATTEN’s neighborhood half-width formulation).
- Metal kernel acceleration requires odd kernel sizes with the following caps:
  - 1D: K ≤ 63
  - 2D: K ≤ 13
  - 3D: K ≤ 7
- Unsupported kernel sizes or configurations fall back automatically (to a supported backend or `pure`).
- **Supported dtypes:** float32, float16, and bfloat16. Low-precision dtypes may be upcast to float32 for certain ops.
- macOS only (Apple Silicon required for Metal backends).

---

## natten-mlx vs natten-mps

| | natten-mlx | natten-mps |
|---|---|---|
| Framework | MLX | PyTorch |
| Device | Apple Silicon (Metal) | Apple Silicon (MPS) |
| Best for | MLX-native projects | PyTorch + MPS projects |

Both packages provide a matching `extras/` namespace for model-specific fused kernels.

---

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

---

## Acknowledgments

This project implements Neighborhood Attention as introduced by the upstream [NATTEN](https://github.com/SHI-Labs/NATTEN) project (SHI-Labs). The original NATTEN library and research are by Ali Hassani, Steven Walton, Humphrey Shi, and collaborators.

If you use Neighborhood Attention in research, please cite the original papers:

- Hassani et al., **Neighborhood Attention Transformer** (CVPR 2023)
- Hassani & Shi, **Dilated Neighborhood Attention Transformer** (2022)
- Hassani et al., **Faster Neighborhood Attention** (NeurIPS 2024)

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{hassani2023neighborhood,
  title   = {Neighborhood Attention Transformer},
  author  = {Hassani, Ali and Walton, Steven and Li, Jiachen and Li, Shen and Shi, Humphrey},
  booktitle = {CVPR},
  year    = {2023}
}

@article{hassani2022dilated,
  title   = {Dilated Neighborhood Attention Transformer},
  author  = {Hassani, Ali and Shi, Humphrey},
  journal = {arXiv preprint arXiv:2209.15001},
  year    = {2022}
}

@inproceedings{hassani2024faster,
  title   = {Faster Neighborhood Attention: Reducing the O(n^2) Cost of Self Attention at the Threadblock Level},
  author  = {Hassani, Ali and Ke, Wen-Mei and Gong, Jiaming and Walton, Steven and Shi, Humphrey},
  booktitle = {NeurIPS},
  year    = {2024}
}
```
</details>

---

## License

MIT — see [LICENSE](LICENSE) for details.  
Upstream NATTEN is also MIT-licensed.

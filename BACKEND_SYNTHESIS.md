# Backend Synthesis Audit

This document records the cross-repo synthesis work for `natten-mlx` backend design.

## Scope

Target support matrix:

1. Metal kernels (with fusion)
2. MLX fast Metal kernels (with fusion)
3. Pure MLX

And for each tier:

1. Forward support
2. Backward support

## Candidate Repositories Audited

1. `/Users/sam/Code/natten-mlx older attempt`
2. `/Users/sam/Code/d3rm-mlx/mlx_natten`
3. `/Users/sam/Code/natten_metal`
4. `/Users/sam/Code/natten_metal_gem`
5. `/Users/sam/Code/natten_metal_candidates/alternate3-natten-metal-gem`
6. `/Users/sam/Code/natten-mps`
7. `/Users/sam/Code/natten-mlx` (current)

## High-Signal Findings

### 1) Metal kernels (PyTorch MPS extension family)

Relevant files:

1. `/Users/sam/Code/natten_metal/natten_metal_bindings.mm`
2. `/Users/sam/Code/natten_metal/natten_metal.mm`
3. `/Users/sam/Code/natten_metal_gem/natten_metal_bindings.mm`
4. `/Users/sam/Code/natten_metal_gem/natten_metal.mm`
5. `/Users/sam/Code/natten_metal_candidates/alternate3-natten-metal-gem/natten_metal_bindings.mm`
6. `/Users/sam/Code/natten_metal_candidates/alternate3-natten-metal-gem/natten_metal.mm`
7. `/Users/sam/Code/natten_metal_candidates/alternate3-natten-metal-gem/natten_kernels.metal`

Observed capabilities:

1. 1D/2D QK and AV forward kernels.
2. 1D backward kernels for QK and AV in multiple variants.
3. RPB forward kernels (`*_qk_rpb`).
4. Fused forward kernels present in advanced variants:
5. 2D `qk+softmax` fused specialization (`K=3`).
6. 2D `qkv` fully fused specialization (`K=3`).
7. 1D `qkv` fused variants in gem/alt3 code paths.

### 2) MLX fast Metal kernels (MX fast kernel family)

Relevant files:

1. `/Users/sam/Code/natten-mlx older attempt/src/natten_mlx/ops.py`
2. `/Users/sam/Code/natten-mlx older attempt/src/natten_mlx/ops_shift.py`
3. `/Users/sam/Code/natten-mlx older attempt/src/natten_mlx/d3rm_fused/ops.py`
4. `/Users/sam/Code/natten-mlx older attempt/src/natten_mlx/compat/functional_metal.py`
5. `/Users/sam/Code/d3rm-mlx/mlx_natten/ops.py`

Observed capabilities:

1. 2D fused `qk+softmax+av` fast paths for `K in {3,5,7}`.
2. Optional fp16 fused variants.
3. Split 1D/2D QK and AV kernels.
4. Extensive experimental backward kernel set in `compat/functional_metal.py`.

### 3) Pure backend baseline

Relevant files:

1. `/Users/sam/Code/natten-mlx/src/natten_mlx/_core/pure.py`
2. `/Users/sam/Code/natten-mlx/src/natten_mlx/autograd/na1d.py`
3. `/Users/sam/Code/natten-mlx/src/natten_mlx/autograd/na2d.py`

Observed capabilities:

1. Full semantics coverage for stride/dilation/causal and split/fused API behavior.
2. Stable parity-tested baseline.
3. Reliable differentiability using standard MLX ops.

## Synthesis Applied to Official Repo

### Implemented

1. Added fast-MLX split kernel sources and wrappers.
2. Added fast-MLX fused kernel sources and wrappers for 1D/2D.
3. Kept pure fallback for unsupported fast-kernel configurations.
4. Added in-tree nanobind backend implementation with optional external module override:
5. `/Users/sam/Code/natten-mlx/src/natten_mlx/_core/_nanobind_impl.py`
6. Added explicit backend capability API:
7. `/Users/sam/Code/natten-mlx/src/natten_mlx/support_matrix.py`
8. Hardened backward behavior:
9. Custom VJP now uses pure semantics for gradient evaluation when accelerated backends are active.
10. Added tests for support matrix API.

### Deliberately Deferred

1. Porting the full experimental backward-kernel suite from old `functional_metal.py` (high complexity and regression risk).
2. Directly embedding Torch MPS extension code into MLX package.

## Current Matrix in This Repo

1. `pure`:
2. Forward: full
3. Backward: full
4. Fusion: no

1. `fast_metal`:
2. Forward: yes (fused + split on supported configs, fallback otherwise)
3. Backward: yes for end-to-end `na1d`/`na2d` via pure-semantic custom VJP
4. Fusion: yes (supported config subset)

1. `nanobind`:
2. Forward: full (in-tree implementation delegates to fast_metal/pure)
3. Backward: yes for end-to-end `na1d`/`na2d` via pure-semantic custom VJP
4. Fusion: yes when delegated fast_metal fast path is eligible; otherwise fallback

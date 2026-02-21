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
6. Added true nanobind C++ extension target with forward/backward bindings:
7. `/Users/sam/Code/natten-mlx/src/natten_mlx/_core/_nanobind_ext.cpp`
6. Added explicit backend capability API:
7. `/Users/sam/Code/natten-mlx/src/natten_mlx/support_matrix.py`
8. Hardened backward behavior:
9. Added explicit backend backward entrypoints for fused and split 1D/2D/3D paths.
10. Added Metal split-backward kernels for all gradient components (`grad_q`, `grad_k`, `grad_attn`, `grad_v`) across 1D/2D/3D.
11. Custom VJP now routes to backend backward when available, with pure fallback safety.
12. Added backend gradient parity coverage (backend vs pure and upstream v0.14 split reference).
13. Added tests for support matrix API.
14. Added required upstream parity gate to backend CI workflow.
15. Added required backward perf guardrail CI gate (`benchmarks/backward_perf_guardrail.py`).
16. Added experimental fused AV-backward Metal kernels for 1D/2D/3D, with default runtime path kept on split kernels due current benchmark results.
17. Optimized split AV-backward `grad_v` kernels with inverse-map edge packing (cached per-edge index bases) to reduce index decode overhead.
18. Added adaptive compressed index types for cached inverse-map edge bases (`uint16` when safe, `int32` fallback) to cut split-backward index memory traffic.
19. Hardened required backward perf gating with median-of-medians aggregation (`benchmarks/backward_perf_guardrail.py --rounds`) and sequential CI ordering.
20. Added 1D split `grad_v` vec4-style backward specialization (4 channels per thread) with dedicated launch tuning.
21. Added 2D/3D split `grad_v` vec4-style backward specializations (4 channels per thread) with dimension-specific launch tuning.
22. Added 1D vec4 selection heuristic guard for small-shape corner cases (`D=32, L<=128`) where scalar can be more stable.
23. Expanded required backward perf guardrail coverage with decode-like 1D cases (causal and long non-causal) so 1D backward tuning gains are CI-enforced.
24. Replaced 2D/3D split `qk_backward` `grad_k` reverse-search kernels with inverse-map kernels to remove the major hotspot and restore strong split-backward scaling.
25. Expanded required backward perf guardrail coverage with explicit split `qk_backward` 2D/3D `grad_k` hotspot cases.
26. Optimized experimental AV-backward fused kernels to use inverse-map `grad_v` accumulation (removing dense reverse-search in fused mode).
27. Expanded vec4 forward coverage to split `na2d_qk` and split `na3d_qk` Metal kernels (in addition to prior 1D split vec4 paths).
28. Expanded vec4 forward coverage to fused `na2d` and fused `na3d`, and added split `na1d_av` vec4 kernel path.

### Deliberately Deferred

1. Enabling AV-backward fused kernels by default pending consistently positive cross-shape results (after inverse-map optimization, fused remains mixed vs split on benchmark hardware).

## Current Matrix in This Repo

1. `pure`:
2. Forward: full
3. Backward: full
4. Fusion: no
5. Notes: full 1D/2D/3D semantic coverage and fallback baseline

1. `fast_metal`:
2. Forward: yes (fused + split on supported configs, fallback otherwise)
3. Backward: yes (backend backward entrypoints for fused/split, pure fallback for unsupported/error paths)
4. Fusion:
5. 1D: yes (odd K, stride>=1, dilation>=1, causal/non-causal)
6. 2D: yes (square odd K, per-axis stride/dilation >= 1, per-axis causal/non-causal)
7. 3D: yes (cubic odd K, per-axis stride/dilation >= 1, per-axis causal/non-causal)
8. Split acceleration eligibility:
9. 1D: odd K, stride>=1, dilation>=1, causal/non-causal
10. 2D: square odd K, stride>=1 per axis, per-axis dilation>=1, per-axis causal/non-causal
11. 3D: cubic odd K, stride>=1 per axis, per-axis dilation>=1, per-axis causal/non-causal

1. `nanobind`:
2. Forward: full (compiled extension when available; otherwise in-tree fallback delegates to fast_metal/pure)
3. Backward: yes (compiled extension and fallback both expose backend backward entrypoints)
4. Fusion: matches fast_metal when delegated; otherwise pure fallback
5. Constraints: exactly the same fused/split eligibility constraints as fast_metal when delegated

## Final Perf Snapshot

Generated on `2026-02-21` using:
`uv run python benchmarks/final_perf_table.py --warmup 5 --trials 25`
(`--trim-head 2` default trimmed reporting with `raw_*` metrics retained in JSON output)

| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.615 | 0.211 | 0.201 | 2.92x | 3.06x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.483 | 0.302 | 0.303 | 1.60x | 1.59x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.691 | 0.667 | 0.664 | 2.53x | 2.55x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 1.875 | 0.558 | 0.561 | 3.36x | 3.34x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.862 | 0.305 | 0.302 | 2.82x | 2.85x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 0.959 | 0.370 | 0.374 | 2.59x | 2.57x |

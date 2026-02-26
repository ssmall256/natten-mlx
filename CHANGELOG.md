# Changelog

All notable changes to natten-mlx are documented here.

## [0.3.0] — 2026-02-25

### Added
- Variable-length (varlen) attention for 1D, 2D, and 3D with Metal acceleration.
- `extras.allin1` namespace for DiNAT fused QK+RPB and AV Metal kernels.
- `return_lse` parameter for log-sum-exp output.
- `merge_attentions` for numerically stable attention merging.
- GQA / MQA support via mismatched head counts and `num_kv_heads` in nn modules.
- `additional_keys` / `additional_values` for global token prepending.
- FMHA fast path (auto-dispatch to `mx.fast.scaled_dot_product_attention`).
- Fused SIMD backward kernels for improved backward pass performance.

## [0.2.0] — 2026-02-25

### Added
- v2 Metal primitives with fused softmax backward and function constants.
- vec4 forward and backward optimization passes for all dimensions.
- Causal forward kernel optimizations for 2D and 3D.
- Low-precision (float16, bfloat16) forward parity with full-precision paths.
- Native nanobind Metal runtime with compiled C++ primitives.

## [0.1.0] — 2026-02-24

### Added
- Initial release with pure MLX, fast Metal, and nanobind backend tiers.
- 1D, 2D, and 3D neighborhood attention (fused and split QK/AV).
- Metal backward kernels with scatter-accumulate dispatch.
- Causal masking with per-axis control.
- Strided output for downsampling.
- Non-uniform per-axis kernel sizes and dilations.
- Compatibility shims for NATTEN v0.14, v0.15, v0.17, and v0.20.
- Upstream parity tests against NATTEN v0.14.6.
- Backend matrix CI with performance guardrails.
- Automatic backend selection (nanobind > fast_metal > pure).

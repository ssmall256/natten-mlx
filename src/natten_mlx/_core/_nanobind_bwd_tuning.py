"""Locked tuning tables for nanobind native backward runtime."""

from __future__ import annotations

import os
import platform


def _is_apple_silicon() -> bool:
    return platform.system().lower() == "darwin" and "arm64" in platform.machine().lower()


def gpu_family_key() -> str:
    override = os.getenv("NATTEN_MLX_GPU_FAMILY", "").strip()
    if override:
        return override
    if _is_apple_silicon():
        return "apple_silicon"
    return "apple_unknown"


def token_band(tokens: int) -> str:
    if tokens <= 0:
        return "unknown"
    if tokens <= 256:
        return "tiny"
    if tokens <= 1024:
        return "small"
    if tokens <= 4096:
        return "medium"
    return "large"


def head_dim_band(head_dim: int) -> str:
    if head_dim <= 0:
        return "unknown"
    if head_dim <= 16:
        return "d16"
    if head_dim <= 32:
        return "d32"
    return "d64p"


def kernel_band(kernel_size: int) -> str:
    if kernel_size <= 0:
        return "unknown"
    if kernel_size <= 5:
        return "k_small"
    if kernel_size <= 9:
        return "k_mid"
    return "k_large"


def causal_rank_band(causal_rank: int) -> str:
    if causal_rank < 0:
        return "unknown"
    if causal_rank <= 0:
        return "c0"
    if causal_rank == 1:
        return "c1"
    return "c2p"


def dtype_class(dtype_tag: str) -> str:
    return "lowp" if dtype_tag in {"fp16", "bf16"} else "fp32"


# Backward mode: "atomic" (default stable path), "tiled" (experimental gather/direct path).
NANOBIND_BWD_MODE_TABLE = {
    "apple_silicon": {
        ("na1d_qk_backward", "fp32", "any", "any", "k_mid", "c0", "s1"): "tiled",
        ("na1d_av_backward", "fp32", "any", "any", "k_mid", "c0", "s1"): "tiled",
        ("na1d_fused_backward_qk", "fp32", "any", "any", "k_mid", "c0", "s1"): "tiled",
        ("na1d_fused_backward_v", "fp32", "any", "any", "k_mid", "c0", "s1"): "tiled",
        ("na1d_qk_backward", "fp32", "any", "any", "k_mid", "c1", "s1"): "tiled",
        ("na1d_av_backward", "fp32", "any", "any", "k_mid", "c1", "s1"): "tiled",
        ("na1d_fused_backward_qk", "fp32", "any", "any", "k_mid", "c1", "s1"): "tiled",
        ("na1d_fused_backward_v", "fp32", "any", "any", "k_mid", "c1", "s1"): "tiled",
        ("na2d_qk_backward", "fp32", "any", "any", "k_mid", "c0", "s1"): "tiled",
        ("na2d_av_backward", "fp32", "any", "any", "k_mid", "c0", "s1"): "tiled",
        ("na3d_qk_backward", "fp32", "any", "any", "k_small", "c0", "s1"): "tiled",
        ("na3d_av_backward", "fp32", "any", "any", "k_small", "c0", "s1"): "tiled",
        ("na2d_fused_backward_qk", "fp32", "any", "any", "k_mid", "c0", "s1"): "tiled",
        ("na2d_fused_backward_v", "fp32", "any", "any", "k_mid", "c0", "s1"): "tiled",
        ("na3d_fused_backward_qk", "fp32", "any", "any", "k_small", "c0", "s1"): "tiled",
        ("na3d_fused_backward_v", "fp32", "any", "any", "k_small", "c0", "s1"): "tiled",
        # Fused 2D/3D direct kernels are currently best in tiny bands; larger
        # bands favor atomic fused paths on Apple Silicon.
        ("na2d_fused_backward_qk", "fp32", "small", "any", "k_mid", "c0", "s1"): "atomic",
        ("na2d_fused_backward_qk", "fp32", "medium", "any", "k_mid", "c0", "s1"): "atomic",
        ("na2d_fused_backward_qk", "fp32", "large", "any", "k_mid", "c0", "s1"): "atomic",
        ("na2d_fused_backward_v", "fp32", "small", "any", "k_mid", "c0", "s1"): "atomic",
        ("na2d_fused_backward_v", "fp32", "medium", "any", "k_mid", "c0", "s1"): "atomic",
        ("na2d_fused_backward_v", "fp32", "large", "any", "k_mid", "c0", "s1"): "atomic",
        ("na3d_fused_backward_qk", "fp32", "small", "any", "k_small", "c0", "s1"): "atomic",
        ("na3d_fused_backward_qk", "fp32", "medium", "any", "k_small", "c0", "s1"): "atomic",
        ("na3d_fused_backward_qk", "fp32", "large", "any", "k_small", "c0", "s1"): "atomic",
        ("na3d_fused_backward_v", "fp32", "small", "any", "k_small", "c0", "s1"): "atomic",
        ("na3d_fused_backward_v", "fp32", "medium", "any", "k_small", "c0", "s1"): "atomic",
        ("na3d_fused_backward_v", "fp32", "large", "any", "k_small", "c0", "s1"): "atomic",
        ("na2d_qk_backward", "fp32", "any", "any", "any", "any", "any"): "atomic",
        ("na2d_av_backward", "fp32", "any", "any", "any", "any", "any"): "atomic",
        ("na3d_qk_backward", "fp32", "any", "any", "any", "any", "any"): "atomic",
        ("na3d_av_backward", "fp32", "any", "any", "any", "any", "any"): "atomic",
        ("na2d_fused_backward_qk", "fp32", "any", "any", "any", "any", "any"): "atomic",
        ("na2d_fused_backward_v", "fp32", "any", "any", "any", "any", "any"): "atomic",
        ("na3d_fused_backward_qk", "fp32", "any", "any", "any", "any", "any"): "atomic",
        ("na3d_fused_backward_v", "fp32", "any", "any", "any", "any", "any"): "atomic",
        ("na1d_qk_backward", "fp32", "any", "any", "any", "any", "any"): "atomic",
        ("na1d_av_backward", "fp32", "any", "any", "any", "any", "any"): "atomic",
        ("na1d_fused_backward_qk", "fp32", "any", "any", "any", "any", "any"): "atomic",
        ("na1d_fused_backward_v", "fp32", "any", "any", "any", "any", "any"): "atomic",
    },
    "apple_unknown": {},
}

# Optional per-kernel launch override table for backward kernels.
NANOBIND_BWD_THREADGROUP_TABLE = {
    "apple_silicon": {
        ("na2d_fused_bwd_attn_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na3d_fused_bwd_attn_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na1d_fused_bwd_attn_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na1d_qk_bwd_k_direct_u1d1_nc_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na1d_qk_bwd_k_direct_u1d1_nc_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na1d_qk_bwd_k_direct_s1_causal_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na1d_qk_bwd_k_direct_s1_causal_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na1d_av_bwd_v_direct_u1d1_nc_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na1d_av_bwd_v_direct_u1d1_nc_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na1d_av_bwd_v_direct_s1_causal_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na1d_av_bwd_v_direct_s1_causal_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na1d_fused_bwd_q_softmax_s1_fp32", "fp32", "any", "any", "any", "any", "any"): (128, 1, 1),
        ("na1d_fused_bwd_q_softmax_s1_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (128, 1, 1),
        (
            "na1d_fused_bwd_kv_softmax_direct_s1_causal_fp32",
            "fp32",
            "any",
            "any",
            "any",
            "any",
            "any",
        ): (256, 1, 1),
        (
            "na1d_fused_bwd_kv_softmax_direct_s1_causal_vec4_fp32",
            "fp32",
            "any",
            "any",
            "any",
            "any",
            "any",
        ): (256, 1, 1),
        ("na1d_fused_bwd_kv_softmax_direct_u1d1_nc_fp32", "fp32", "any", "any", "any", "any", "any"): (
            128,
            1,
            1,
        ),
        (
            "na1d_fused_bwd_kv_softmax_direct_u1d1_nc_vec4_fp32",
            "fp32",
            "any",
            "any",
            "any",
            "any",
            "any",
        ): (128, 1, 1),
        ("na2d_qk_bwd_q_fp32", "fp32", "any", "any", "any", "any", "any"): (64, 1, 1),
        ("na2d_qk_bwd_q_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (192, 1, 1),
        ("na2d_qk_bwd_k_direct_u1d1_nc_fp32", "fp32", "any", "any", "any", "any", "any"): (192, 1, 1),
        ("na2d_qk_bwd_k_direct_u1d1_nc_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (192, 1, 1),
        ("na2d_qk_bwd_k_direct_u1d1_nc_k7_fp32", "fp32", "any", "any", "any", "any", "any"): (192, 1, 1),
        ("na2d_qk_bwd_k_direct_u1d1_nc_k7_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (192, 1, 1),
        ("na2d_av_bwd_v_direct_u1d1_nc_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na2d_av_bwd_v_direct_u1d1_nc_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na2d_fused_bwd_qkv_softmax_tiled_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na2d_fused_bwd_kv_softmax_tiled_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na2d_fused_bwd_kv_softmax_tiled_k3_fp32", "fp32", "any", "any", "any", "any", "any"): (128, 1, 1),
        ("na2d_fused_bwd_kv_softmax_tiled_k5_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na2d_fused_bwd_kv_softmax_tiled_k7_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na2d_fused_bwd_kv_softmax_tiled_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na2d_fused_bwd_kv_softmax_tiled_k3_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (128, 1, 1),
        ("na2d_fused_bwd_kv_softmax_tiled_k5_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na2d_fused_bwd_kv_softmax_tiled_k7_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (224, 1, 1),
        ("na2d_fused_bwd_q_softmax_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na2d_fused_bwd_q_softmax_u1d1_nc_fp32", "fp32", "any", "any", "any", "any", "any"): (
            256,
            1,
            1,
        ),
        ("na2d_fused_bwd_q_softmax_u1d1_nc_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (
            224,
            1,
            1,
        ),
        ("na3d_fused_bwd_q_softmax_k3_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na3d_fused_bwd_q_softmax_k5_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na3d_fused_bwd_q_softmax_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na3d_fused_bwd_q_softmax_k3_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na3d_fused_bwd_q_softmax_k5_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na3d_fused_bwd_qkv_softmax_tiled_fp32", "fp32", "any", "any", "any", "any", "any"): (128, 1, 1),
        ("na3d_fused_bwd_qkv_softmax_tiled_k3_fp32", "fp32", "any", "any", "any", "any", "any"): (192, 1, 1),
        ("na3d_fused_bwd_qkv_softmax_tiled_k5_fp32", "fp32", "any", "any", "any", "any", "any"): (128, 1, 1),
        ("na3d_fused_bwd_qkv_softmax_tiled_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na3d_fused_bwd_qkv_softmax_tiled_k3_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na3d_fused_bwd_qkv_softmax_tiled_k5_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (192, 1, 1),
        ("na3d_fused_bwd_kv_softmax_tiled_fp32", "fp32", "any", "any", "any", "any", "any"): (128, 1, 1),
        ("na3d_fused_bwd_kv_softmax_tiled_k3_fp32", "fp32", "any", "any", "any", "any", "any"): (192, 1, 1),
        ("na3d_fused_bwd_kv_softmax_tiled_k5_fp32", "fp32", "any", "any", "any", "any", "any"): (128, 1, 1),
        ("na3d_fused_bwd_kv_softmax_tiled_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na3d_fused_bwd_kv_softmax_tiled_k3_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na3d_fused_bwd_kv_softmax_tiled_k5_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (192, 1, 1),
        ("na3d_qk_bwd_k_direct_u1d1_nc_k3_fp32", "fp32", "any", "any", "any", "any", "any"): (192, 1, 1),
        ("na3d_qk_bwd_k_direct_u1d1_nc_k3_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (192, 1, 1),
        ("na3d_qk_bwd_k_direct_u1d1_nc_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
        ("na3d_av_bwd_v_direct_u1d1_nc_vec4_fp32", "fp32", "any", "any", "any", "any", "any"): (256, 1, 1),
    },
    "apple_unknown": {},
}


def _lookup_with_fallback(table: dict, key: tuple):
    # Exact match first.
    if key in table:
        return table[key]
    # Fallback hierarchy from specific shape bands to op-level defaults.
    op, dtype, tb, hb, kb, cb, sb = key
    fallbacks = [
        (op, dtype, tb, hb, kb, cb, sb),
        (op, dtype, tb, "any", kb, cb, sb),
        (op, dtype, "any", hb, kb, cb, sb),
        (op, dtype, "any", "any", kb, cb, sb),
        (op, dtype, "any", "any", "any", cb, sb),
        (op, dtype, "any", "any", "any", "any", sb),
        (op, dtype, "any", "any", "any", "any", "any"),
    ]
    for candidate in fallbacks:
        value = table.get(candidate)
        if value is not None:
            return value
    return None


def lookup_backward_mode(
    *,
    op: str,
    dtype_tag: str,
    tokens: int,
    head_dim: int,
    kernel_size: int,
    causal_rank: int,
    stride_unit: bool,
) -> str:
    table = NANOBIND_BWD_MODE_TABLE.get(gpu_family_key()) or NANOBIND_BWD_MODE_TABLE["apple_unknown"]
    key = (
        op,
        dtype_class(dtype_tag),
        token_band(tokens),
        head_dim_band(head_dim),
        kernel_band(kernel_size),
        causal_rank_band(causal_rank),
        "s1" if bool(stride_unit) else "sN",
    )
    mode = _lookup_with_fallback(table, key)
    if mode not in {"atomic", "simple", "tiled"}:
        return "atomic"
    return "atomic" if mode == "simple" else mode


def lookup_backward_threadgroup(
    *,
    op: str,
    dtype_tag: str,
    tokens: int,
    head_dim: int,
    kernel_size: int,
    causal_rank: int,
    stride_unit: bool,
):
    table = NANOBIND_BWD_THREADGROUP_TABLE.get(gpu_family_key()) or NANOBIND_BWD_THREADGROUP_TABLE["apple_unknown"]
    key = (
        op,
        dtype_class(dtype_tag),
        token_band(tokens),
        head_dim_band(head_dim),
        kernel_band(kernel_size),
        causal_rank_band(causal_rank),
        "s1" if bool(stride_unit) else "sN",
    )
    tg = _lookup_with_fallback(table, key)
    if tg is None:
        return None
    if not isinstance(tg, tuple) or len(tg) != 3:
        return None
    return (max(1, int(tg[0])), max(1, int(tg[1])), max(1, int(tg[2])))

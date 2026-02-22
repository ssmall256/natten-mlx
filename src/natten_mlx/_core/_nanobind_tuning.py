"""Locked tuning tables for nanobind native C++/Metal runtime."""

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
    if tokens <= 256: return 'tiny'
    if tokens <= 1024: return 'small'
    if tokens <= 4096: return 'medium'
    return 'large'

def head_dim_band(head_dim: int) -> str:
    if head_dim <= 16: return 'd16'
    if head_dim <= 32: return 'd32'
    return 'd64p'

def kernel_band(kernel_size: int) -> str:
    if kernel_size <= 5: return 'k_small'
    if kernel_size <= 9: return 'k_mid'
    return 'k_large'

def causal_rank_band(causal_rank: int) -> str:
    if causal_rank <= 0: return 'c0'
    if causal_rank == 1: return 'c1'
    return 'c2p'

def dtype_class(dtype_tag: str) -> str:
    return 'lowp' if dtype_tag in {'fp16', 'bf16'} else 'fp32'

NANOBIND_THREADGROUP_TABLE = {
    'apple_silicon': {
        ('na1d_fused', 'lowp', 'small', 'd64p', 'k_mid', 'c1', True): (128, 1, 1),
        ('na2d_fused', 'lowp', 'small', 'd16', 'k_mid', 'c1', True): (8, 8, 1),
        ('na3d_fused', 'lowp', 'medium', 'd16', 'k_small', 'c1', True): (8, 8, 1),
    },
    'apple_unknown': {},
}

NANOBIND_SOFTMAX_STRATEGY_TABLE = {
    'apple_silicon': {
        ('na1d_fused', 'lowp', 'small', 'k_mid', 'c1', True): 'stored',
        ('na2d_fused', 'lowp', 'small', 'k_mid', 'c1', True): 'stored',
        ('na3d_fused', 'lowp', 'medium', 'k_small', 'c1', True): 'stored',
    },
    'apple_unknown': {},
}

def lookup_threadgroup(*, op: str, dtype_tag: str, tokens: int, head_dim: int, kernel_size: int, causal_rank: int, stride_unit: bool):
    table = NANOBIND_THREADGROUP_TABLE.get(gpu_family_key()) or NANOBIND_THREADGROUP_TABLE['apple_unknown']
    key = (op, dtype_class(dtype_tag), token_band(tokens), head_dim_band(head_dim), kernel_band(kernel_size), causal_rank_band(causal_rank), bool(stride_unit))
    return table.get(key)

def choose_softmax_strategy(*, op: str, dtype_tag: str, tokens: int, kernel_size: int, causal_rank: int, stride_unit: bool) -> str:
    table = NANOBIND_SOFTMAX_STRATEGY_TABLE.get(gpu_family_key()) or NANOBIND_SOFTMAX_STRATEGY_TABLE['apple_unknown']
    key = (op, dtype_class(dtype_tag), token_band(tokens), kernel_band(kernel_size), causal_rank_band(causal_rank), bool(stride_unit))
    v = table.get(key, 'stored')
    return v if v in {'stored', 'recompute'} else 'stored'

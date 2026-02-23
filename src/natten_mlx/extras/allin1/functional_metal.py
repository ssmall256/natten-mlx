"""
HIGH-PERFORMANCE NATTEN Functional API Compatibility Layer

Uses partially-fused Metal kernels for maximum performance while maintaining
API compatibility with NATTEN functional API.

Performance: Nearly as fast as fully-fused kernels, with only the unavoidable
intermediate attention scores write required by the API.
"""

import mlx.core as mx
import os
from typing import Union, Tuple, Optional, Any, Dict
import numpy as np
from collections import OrderedDict

from .reference_impl import get_window_start, get_window_end, get_pb_start

from .kernels import (
    NATTEN_1D_K3_QKRPB_SOURCE,
    NATTEN_1D_K3_AV_SOURCE,
    NATTEN_1D_K5_QKRPB_SOURCE,
    NATTEN_1D_K5_AV_SOURCE,
    NATTEN_1D_K7_QKRPB_SOURCE,
    NATTEN_1D_K7_AV_SOURCE,
    NATTEN_1D_K5_QKRPB_FAST_D12_SOURCE,
    NATTEN_1D_K5_AV_FAST_D12_SOURCE,
    NATTEN_K3_QKRPB_SOURCE,
    NATTEN_K3_AV_SOURCE,
    NATTEN_K5_QKRPB_SOURCE,
    NATTEN_K5_AV_SOURCE,
    NATTEN_K7_QKRPB_SOURCE,
    NATTEN_K7_AV_SOURCE,
    NATTEN_K3_QK_BWD_DQ_SOURCE,
    NATTEN_K3_QK_BWD_DK_SOURCE,
    NATTEN_K3_QK_BWD_DRPB_SOURCE,
    NATTEN_K3_AV_BWD_DATTN_SOURCE,
    NATTEN_K3_AV_BWD_DVAL_SOURCE,
    NATTEN_K5_QK_BWD_DQ_SOURCE,
    NATTEN_K5_QK_BWD_DK_SOURCE,
    NATTEN_K5_QK_BWD_DRPB_SOURCE,
    NATTEN_K5_AV_BWD_DATTN_SOURCE,
    NATTEN_K5_AV_BWD_DVAL_SOURCE,
    NATTEN_K7_QK_BWD_DQ_SOURCE,
    NATTEN_K7_QK_BWD_DK_SOURCE,
    NATTEN_K7_QK_BWD_DRPB_SOURCE,
    NATTEN_K7_AV_BWD_DATTN_SOURCE,
    NATTEN_K7_AV_BWD_DVAL_SOURCE,
    NATTEN_K3_AV_BWD_DATTN_TG_T8_SOURCE,
    NATTEN_K5_AV_BWD_DATTN_TG_T8_SOURCE,
    NATTEN_K7_AV_BWD_DATTN_TG_T8_SOURCE,
    NATTEN_K3_AV_BWD_DATTN_SPLIT_TG_SOURCE,
    NATTEN_K5_AV_BWD_DATTN_SPLIT_TG_SOURCE,
    NATTEN_K7_AV_BWD_DATTN_SPLIT_TG_SOURCE,
    NATTEN_K3_AV_BWD_DATTN_SPLIT_TG_T8_SOURCE,
    NATTEN_K5_AV_BWD_DATTN_SPLIT_TG_T8_SOURCE,
    NATTEN_K7_AV_BWD_DATTN_SPLIT_TG_T8_SOURCE,
    NATTEN_K3_AV_BWD_FUSED_SPLIT_TG_SOURCE,
    NATTEN_K5_AV_BWD_FUSED_SPLIT_TG_SOURCE,
    NATTEN_K7_AV_BWD_FUSED_SPLIT_TG_SOURCE,
    NATTEN_K3_AV_BWD_FUSED_SPLIT_TG_T8_SOURCE,
    NATTEN_K5_AV_BWD_FUSED_SPLIT_TG_T8_SOURCE,
    NATTEN_K7_AV_BWD_FUSED_SPLIT_TG_T8_SOURCE,
    NATTEN_1D_K3_QK_BWD_DQ_SOURCE,
    NATTEN_1D_K3_QK_BWD_DK_SOURCE,
    NATTEN_1D_K3_QK_BWD_DRPB_SOURCE,
    NATTEN_1D_K3_AV_BWD_DATTN_SOURCE,
    NATTEN_1D_K3_AV_BWD_DVAL_SOURCE,
    NATTEN_1D_K5_QK_BWD_DQ_SOURCE,
    NATTEN_1D_K5_QK_BWD_DK_SOURCE,
    NATTEN_1D_K5_QK_BWD_DRPB_SOURCE,
    NATTEN_1D_K5_AV_BWD_DATTN_SOURCE,
    NATTEN_1D_K5_AV_BWD_DVAL_SOURCE,
    NATTEN_1D_K7_QK_BWD_DQ_SOURCE,
    NATTEN_1D_K7_QK_BWD_DK_SOURCE,
    NATTEN_1D_K7_QK_BWD_DRPB_SOURCE,
    NATTEN_1D_K7_AV_BWD_DATTN_SOURCE,
    NATTEN_1D_K7_AV_BWD_DVAL_SOURCE,
    NATTEN_1D_K3_QK_BWD_DQ_TG_SOURCE,
    NATTEN_1D_K5_QK_BWD_DQ_TG_SOURCE,
    NATTEN_1D_K7_QK_BWD_DQ_TG_SOURCE,
    NATTEN_1D_K3_QK_BWD_DQ_TG_V4_SOURCE,
    NATTEN_1D_K5_QK_BWD_DQ_TG_V4_SOURCE,
    NATTEN_1D_K7_QK_BWD_DQ_TG_V4_SOURCE,
    NATTEN_1D_K3_AV_BWD_DATTN_TG_SOURCE,
    NATTEN_1D_K5_AV_BWD_DATTN_TG_SOURCE,
    NATTEN_1D_K7_AV_BWD_DATTN_TG_SOURCE,
    NATTEN_1D_K3_AV_BWD_DATTN_TG_V4_SOURCE,
    NATTEN_1D_K5_AV_BWD_DATTN_TG_V4_SOURCE,
    NATTEN_1D_K7_AV_BWD_DATTN_TG_V4_SOURCE,
    NATTEN_1D_K3_AV_BWD_DVAL_FAST_V4_SOURCE,
    NATTEN_1D_K5_AV_BWD_DVAL_FAST_V4_SOURCE,
    NATTEN_1D_K7_AV_BWD_DVAL_FAST_V4_SOURCE,
    NATTEN_K3_QK_BWD_DK_FAST_SOURCE,
    NATTEN_K5_QK_BWD_DK_FAST_SOURCE,
    NATTEN_K7_QK_BWD_DK_FAST_SOURCE,
    NATTEN_K3_QK_BWD_DK_FAST_V4_SOURCE,
    NATTEN_K5_QK_BWD_DK_FAST_V4_SOURCE,
    NATTEN_K7_QK_BWD_DK_FAST_V4_SOURCE,
    NATTEN_K3_AV_BWD_DVAL_FAST_SOURCE,
    NATTEN_K5_AV_BWD_DVAL_FAST_SOURCE,
    NATTEN_K7_AV_BWD_DVAL_FAST_SOURCE,
    NATTEN_1D_K3_QK_BWD_DK_FAST_SOURCE,
    NATTEN_1D_K5_QK_BWD_DK_FAST_SOURCE,
    NATTEN_1D_K7_QK_BWD_DK_FAST_SOURCE,
    NATTEN_1D_K3_QK_BWD_DK_FAST_V4_SOURCE,
    NATTEN_1D_K5_QK_BWD_DK_FAST_V4_SOURCE,
    NATTEN_1D_K7_QK_BWD_DK_FAST_V4_SOURCE,
    NATTEN_1D_K3_AV_BWD_DVAL_FAST_SOURCE,
    NATTEN_1D_K5_AV_BWD_DVAL_FAST_SOURCE,
    NATTEN_1D_K7_AV_BWD_DVAL_FAST_SOURCE,
    NATTEN_K3_QK_BWD_DRPB_FAST_SOURCE,
    NATTEN_K5_QK_BWD_DRPB_FAST_SOURCE,
    NATTEN_K7_QK_BWD_DRPB_FAST_SOURCE,
    NATTEN_K3_QK_BWD_DRPB_FAST_U2_SOURCE,
    NATTEN_K5_QK_BWD_DRPB_FAST_U2_SOURCE,
    NATTEN_K7_QK_BWD_DRPB_FAST_U2_SOURCE,
    NATTEN_K3_QK_BWD_DRPB_FAST_V4_SOURCE,
    NATTEN_K5_QK_BWD_DRPB_FAST_V4_SOURCE,
    NATTEN_K7_QK_BWD_DRPB_FAST_V4_SOURCE,
    NATTEN_K3_QK_BWD_DRPB_FAST_SPLIT_SOURCE,
    NATTEN_K5_QK_BWD_DRPB_FAST_SPLIT_SOURCE,
    NATTEN_K7_QK_BWD_DRPB_FAST_SPLIT_SOURCE,
    NATTEN_1D_K3_QK_BWD_DRPB_FAST_SOURCE,
    NATTEN_1D_K5_QK_BWD_DRPB_FAST_SOURCE,
    NATTEN_1D_K7_QK_BWD_DRPB_FAST_SOURCE,
    NATTEN_K3_QK_BWD_DQ_TG_SOURCE,
    NATTEN_K5_QK_BWD_DQ_TG_SOURCE,
    NATTEN_K7_QK_BWD_DQ_TG_SOURCE,
    NATTEN_K3_AV_BWD_DATTN_TG_SOURCE,
    NATTEN_K5_AV_BWD_DATTN_TG_SOURCE,
    NATTEN_K7_AV_BWD_DATTN_TG_SOURCE,
    NATTEN_K3_AV_BWD_DVAL_TG_SOURCE,
    NATTEN_K5_AV_BWD_DVAL_TG_SOURCE,
    NATTEN_K7_AV_BWD_DVAL_TG_SOURCE,
    NATTEN_K3_AV_BWD_FUSED_TG_SOURCE,
    NATTEN_K5_AV_BWD_FUSED_TG_SOURCE,
    NATTEN_K7_AV_BWD_FUSED_TG_SOURCE,
    NATTEN_K3_AV_BWD_FUSED_TG_T8_SOURCE,
    NATTEN_K5_AV_BWD_FUSED_TG_T8_SOURCE,
    NATTEN_K7_AV_BWD_FUSED_TG_T8_SOURCE,
    NATTEN_K3_AV_BWD_FUSED_TG_UNROLL2_SOURCE,
    NATTEN_K5_AV_BWD_FUSED_TG_UNROLL2_SOURCE,
    NATTEN_K7_AV_BWD_FUSED_TG_UNROLL2_SOURCE,
    NATTEN_K3_AV_BWD_FUSED_TG_T8_UNROLL2_SOURCE,
    NATTEN_K5_AV_BWD_FUSED_TG_T8_UNROLL2_SOURCE,
    NATTEN_K7_AV_BWD_FUSED_TG_T8_UNROLL2_SOURCE,
    _av_bwd_fused_unroll2_source,
)
def get_threadgroup_for_shape(kernel_size, Ht, W, dtype, device=None):
    """Simple threadgroup heuristic for Metal kernels."""
    max_dim = max(Ht, W)
    if max_dim <= 48:
        return (min(32, W), min(1, Ht), 1)
    elif max_dim <= 96:
        return (min(32, W), min(2, Ht), 1)
    else:
        return (min(32, W), min(4, Ht), 1)

try:
    from ..d3rm_fused import natten_fused_k3, natten_fused_k5, natten_fused_k7
    _HAS_D3RM_FUSED = True
except Exception:
    _HAS_D3RM_FUSED = False
    natten_fused_k3 = natten_fused_k5 = natten_fused_k7 = None

def _d3rm_fused_enabled() -> bool:
    return os.environ.get("NATTEN_MLX_USE_D3RM_FUSED", "1") != "0" and _HAS_D3RM_FUSED

def get_d3rm_fused_status() -> Dict[str, Any]:
    return {
        "available": _HAS_D3RM_FUSED,
        "enabled": _d3rm_fused_enabled(),
    }


# Kernel caches
_qkrpb_kernel_cache = {}
_av_kernel_cache = {}
_qkrpb_1d_kernel_cache = {}
_av_1d_kernel_cache = {}
_qkrpb_bwd_dq_cache = {}
_qkrpb_bwd_dk_cache = {}
_qkrpb_bwd_drpb_cache = {}
_av_bwd_dattn_cache = {}
_av_bwd_dval_cache = {}
_qkrpb_1d_bwd_dq_cache = {}
_qkrpb_1d_bwd_dk_cache = {}
_qkrpb_1d_bwd_drpb_cache = {}
_av_1d_bwd_dattn_cache = {}
_av_1d_bwd_dval_cache = {}
_qkrpb_bwd_dk_fast_cache = {}
_av_bwd_dval_fast_cache = {}
_av_bwd_dval_tg_cache = {}
_av_bwd_fused_tg_cache = {}
_av_bwd_fused_split_tg_cache = {}
_LAST_METAL_BWD_STATS: Optional[Dict[str, float]] = None
_LAST_METAL_BWD_CONFIG: Optional[Dict[str, int]] = None


def get_last_metal_bwd_stats() -> Optional[Dict[str, float]]:
    return _LAST_METAL_BWD_STATS


def get_last_metal_bwd_config() -> Optional[Dict[str, int]]:
    return _LAST_METAL_BWD_CONFIG


def _strip_vec4_loads(source: str) -> str:
    # Replace vectorized float4 pointer loads with scalar loads.
    vec_to_scalar = [
        (
            "        if (d0 + 3 < dim) {\n"
            "            tmp = *((device const float4*)(value + v_base));\n"
            "        } else {\n"
            "            if (d0 + 0 < dim) tmp.x = value[v_base + 0];\n"
            "            if (d0 + 1 < dim) tmp.y = value[v_base + 1];\n"
            "            if (d0 + 2 < dim) tmp.z = value[v_base + 2];\n"
            "            if (d0 + 3 < dim) tmp.w = value[v_base + 3];\n"
            "        }\n",
            "        if (d0 + 0 < dim) tmp.x = value[v_base + 0];\n"
            "        if (d0 + 1 < dim) tmp.y = value[v_base + 1];\n"
            "        if (d0 + 2 < dim) tmp.z = value[v_base + 2];\n"
            "        if (d0 + 3 < dim) tmp.w = value[v_base + 3];\n",
        ),
        (
            "        if (d0 + 3 < dim) {\n"
            "            tmpd = *((device const float4*)(d_out + do_base));\n"
            "        } else {\n"
            "            if (d0 + 0 < dim) tmpd.x = d_out[do_base + 0];\n"
            "            if (d0 + 1 < dim) tmpd.y = d_out[do_base + 1];\n"
            "            if (d0 + 2 < dim) tmpd.z = d_out[do_base + 2];\n"
            "            if (d0 + 3 < dim) tmpd.w = d_out[do_base + 3];\n"
            "        }\n",
            "        if (d0 + 0 < dim) tmpd.x = d_out[do_base + 0];\n"
            "        if (d0 + 1 < dim) tmpd.y = d_out[do_base + 1];\n"
            "        if (d0 + 2 < dim) tmpd.z = d_out[do_base + 2];\n"
            "        if (d0 + 3 < dim) tmpd.w = d_out[do_base + 3];\n",
        ),
        (
            "            if (d0 + 3 < dim) {\n"
            "                dout = *((device const float4*)(d_out + do_base));\n"
            "            } else {\n"
            "                if (d0 + 0 < dim) dout.x = d_out[do_base + 0];\n"
            "                if (d0 + 1 < dim) dout.y = d_out[do_base + 1];\n"
            "                if (d0 + 2 < dim) dout.z = d_out[do_base + 2];\n"
            "                if (d0 + 3 < dim) dout.w = d_out[do_base + 3];\n"
            "            }\n",
            "            if (d0 + 0 < dim) dout.x = d_out[do_base + 0];\n"
            "            if (d0 + 1 < dim) dout.y = d_out[do_base + 1];\n"
            "            if (d0 + 2 < dim) dout.z = d_out[do_base + 2];\n"
            "            if (d0 + 3 < dim) dout.w = d_out[do_base + 3];\n",
        ),
        (
            "                        if (d0 + 3 < dim) {\n"
            "                            tmp2 = *((device const float4*)(value + v_base2));\n"
            "                        } else {\n"
            "                            if (d0 + 0 < dim) tmp2.x = value[v_base2 + 0];\n"
            "                            if (d0 + 1 < dim) tmp2.y = value[v_base2 + 1];\n"
            "                            if (d0 + 2 < dim) tmp2.z = value[v_base2 + 2];\n"
            "                            if (d0 + 3 < dim) tmp2.w = value[v_base2 + 3];\n"
            "                        }\n",
            "                        if (d0 + 0 < dim) tmp2.x = value[v_base2 + 0];\n"
            "                        if (d0 + 1 < dim) tmp2.y = value[v_base2 + 1];\n"
            "                        if (d0 + 2 < dim) tmp2.z = value[v_base2 + 2];\n"
            "                        if (d0 + 3 < dim) tmp2.w = value[v_base2 + 3];\n",
        ),
        (
            "                        if (d0 + 3 < dim) {\n"
            "                            tmp2 = *((device const float4*)(d_out + do_base2));\n"
            "                        } else {\n"
            "                            if (d0 + 0 < dim) tmp2.x = d_out[do_base2 + 0];\n"
            "                            if (d0 + 1 < dim) tmp2.y = d_out[do_base2 + 1];\n"
            "                            if (d0 + 2 < dim) tmp2.z = d_out[do_base2 + 2];\n"
            "                            if (d0 + 3 < dim) tmp2.w = d_out[do_base2 + 3];\n"
            "                        }\n",
            "                        if (d0 + 0 < dim) tmp2.x = d_out[do_base2 + 0];\n"
            "                        if (d0 + 1 < dim) tmp2.y = d_out[do_base2 + 1];\n"
            "                        if (d0 + 2 < dim) tmp2.z = d_out[do_base2 + 2];\n"
            "                        if (d0 + 3 < dim) tmp2.w = d_out[do_base2 + 3];\n",
        ),
    ]
    out = source
    for before, after in vec_to_scalar:
        out = out.replace(before, after)
    return out
_qkrpb_1d_bwd_dk_fast_cache = {}
_av_1d_bwd_dval_fast_cache = {}
_qkrpb_bwd_drpb_fast_cache = {}
_qkrpb_1d_bwd_drpb_fast_cache = {}
_qkrpb_bwd_drpb_fast_u2_cache = {}
_qkrpb_bwd_drpb_fast_v4_cache = {}
_qkrpb_bwd_drpb_fast_split_cache = {}
_qkrpb_1d_qkrpb_fast_d12_cache = {}
_qkrpb_1d_av_fast_d12_cache = {}
_qkrpb_bwd_dq_tg_cache = {}
_av_bwd_dattn_tg_cache = {}
_av_bwd_dattn_split_tg_cache = {}
_qkrpb_1d_bwd_dq_tg_cache = {}
_qkrpb_1d_bwd_dq_tg_v4_cache = {}
_av_1d_bwd_dattn_tg_cache = {}
_av_1d_bwd_dval_fast_v4_cache = {}
_av_1d_bwd_dattn_tg_v4_cache = {}
_qkrpb_1d_bwd_dk_fast_v4_cache = {}
_qkrpb_bwd_dk_fast_v4_cache = {}
_DRPB_CACHE_MAX = 16
_drpb_1d_precompute_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_drpb_2d_precompute_cache: "OrderedDict[tuple, tuple]" = OrderedDict()


def _lru_get(cache: "OrderedDict[tuple, tuple]", key):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _lru_set(cache: "OrderedDict[tuple, tuple]", key, value):
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > _DRPB_CACHE_MAX:
        cache.popitem(last=False)


def _get_or_compile_qkrpb_kernel(kernel_size: int):
    """Get or compile QK+RPB Metal kernel."""
    if kernel_size not in _qkrpb_kernel_cache:
        if kernel_size == 3:
            source = NATTEN_K3_QKRPB_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_QKRPB_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_QKRPB_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")

        _qkrpb_kernel_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_qkrpb_k{kernel_size}",
            input_names=["query", "key", "rpb", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )

    return _qkrpb_kernel_cache[kernel_size]


def _get_or_compile_av_kernel(kernel_size: int):
    """Get or compile AV Metal kernel."""
    if kernel_size not in _av_kernel_cache:
        if kernel_size == 3:
            source = NATTEN_K3_AV_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_AV_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_AV_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")

        _av_kernel_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_av_k{kernel_size}",
            input_names=["attention_probs", "value", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )

    return _av_kernel_cache[kernel_size]


def _get_or_compile_qkrpb_kernel_1d(kernel_size: int):
    """Get or compile 1D QK+RPB Metal kernel."""
    if kernel_size not in _qkrpb_1d_kernel_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_QKRPB_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_QKRPB_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_QKRPB_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")

        _qkrpb_1d_kernel_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten1d_qkrpb_k{kernel_size}",
            input_names=["query", "key", "rpb", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )

    return _qkrpb_1d_kernel_cache[kernel_size]


def _get_or_compile_qkrpb_kernel_1d_fast_d12(kernel_size: int):
    if kernel_size != 5:
        raise ValueError("Fast D=12 1D QKRPB kernel only supports K=5.")
    if kernel_size not in _qkrpb_1d_qkrpb_fast_d12_cache:
        _qkrpb_1d_qkrpb_fast_d12_cache[kernel_size] = mx.fast.metal_kernel(
            name="natten1d_qkrpb_fast_d12_k5",
            input_names=["query", "key", "rpb", "pi_arr", "ni_arr", "ei_arr", "dilation_param"],
            output_names=["out"],
            source=NATTEN_1D_K5_QKRPB_FAST_D12_SOURCE,
            ensure_row_contiguous=True,
        )
    return _qkrpb_1d_qkrpb_fast_d12_cache[kernel_size]


def _get_or_compile_av_kernel_1d(kernel_size: int):
    """Get or compile 1D AV Metal kernel."""
    if kernel_size not in _av_1d_kernel_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_AV_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_AV_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_AV_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")

        _av_1d_kernel_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten1d_av_k{kernel_size}",
            input_names=["attention_probs", "value", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )

    return _av_1d_kernel_cache[kernel_size]


def _get_or_compile_av_kernel_1d_fast_d12(kernel_size: int):
    if kernel_size != 5:
        raise ValueError("Fast D=12 1D AV kernel only supports K=5.")
    if kernel_size not in _qkrpb_1d_av_fast_d12_cache:
        _qkrpb_1d_av_fast_d12_cache[kernel_size] = mx.fast.metal_kernel(
            name="natten1d_av_fast_d12_k5",
            input_names=["attention_probs", "value", "ni_arr", "ei_arr", "dilation_param"],
            output_names=["out"],
            source=NATTEN_1D_K5_AV_FAST_D12_SOURCE,
            ensure_row_contiguous=True,
        )
    return _qkrpb_1d_av_fast_d12_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_dq_kernel(kernel_size: int):
    if kernel_size not in _qkrpb_bwd_dq_cache:
        if kernel_size == 3:
            source = NATTEN_K3_QK_BWD_DQ_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_QK_BWD_DQ_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_QK_BWD_DQ_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_bwd_dq_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_qk_bwd_dq_k{kernel_size}",
            input_names=["query", "key", "d_attn", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_bwd_dq_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_dq_kernel_tg(kernel_size: int):
    if kernel_size not in _qkrpb_bwd_dq_tg_cache:
        if kernel_size == 3:
            source = NATTEN_K3_QK_BWD_DQ_TG_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_QK_BWD_DQ_TG_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_QK_BWD_DQ_TG_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_bwd_dq_tg_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_qk_bwd_dq_tg_k{kernel_size}",
            input_names=["query", "key", "d_attn", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_bwd_dq_tg_cache[kernel_size]

def _apply_1d_dq_tile(source: str, tile: int) -> str:
    return source.replace("TILE = 64", f"TILE = {tile}")

def _get_or_compile_qkrpb_bwd_dq_kernel_1d_tg(kernel_size: int, tile: int = 64):
    key = (kernel_size, tile)
    if key not in _qkrpb_1d_bwd_dq_tg_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_QK_BWD_DQ_TG_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_QK_BWD_DQ_TG_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_QK_BWD_DQ_TG_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        if tile != 64:
            source = _apply_1d_dq_tile(source, tile)
        _qkrpb_1d_bwd_dq_tg_cache[key] = mx.fast.metal_kernel(
            name=f"natten1d_qk_bwd_dq_tg_k{kernel_size}_t{tile}",
            input_names=["query", "key", "d_attn", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_1d_bwd_dq_tg_cache[key]

def _get_or_compile_qkrpb_bwd_dq_kernel_1d_tg_v4(kernel_size: int, tile: int = 64):
    key = (kernel_size, tile)
    if key not in _qkrpb_1d_bwd_dq_tg_v4_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_QK_BWD_DQ_TG_V4_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_QK_BWD_DQ_TG_V4_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_QK_BWD_DQ_TG_V4_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        if tile != 64:
            source = _apply_1d_dq_tile(source, tile)
        _qkrpb_1d_bwd_dq_tg_v4_cache[key] = mx.fast.metal_kernel(
            name=f"natten1d_qk_bwd_dq_tg_v4_k{kernel_size}_t{tile}",
            input_names=["query", "key", "d_attn", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_1d_bwd_dq_tg_v4_cache[key]

def _get_or_compile_qkrpb_bwd_dk_kernel(kernel_size: int):
    if kernel_size not in _qkrpb_bwd_dk_cache:
        if kernel_size == 3:
            source = NATTEN_K3_QK_BWD_DK_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_QK_BWD_DK_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_QK_BWD_DK_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_bwd_dk_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_qk_bwd_dk_k{kernel_size}",
            input_names=["query", "key", "d_attn", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_bwd_dk_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_dk_kernel_fast(kernel_size: int):
    if kernel_size not in _qkrpb_bwd_dk_fast_cache:
        if kernel_size == 3:
            source = NATTEN_K3_QK_BWD_DK_FAST_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_QK_BWD_DK_FAST_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_QK_BWD_DK_FAST_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_bwd_dk_fast_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_qk_bwd_dk_fast_k{kernel_size}",
            input_names=["query", "key", "d_attn", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_bwd_dk_fast_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_dk_kernel_fast_v4(kernel_size: int):
    if kernel_size not in _qkrpb_bwd_dk_fast_v4_cache:
        if kernel_size == 3:
            source = NATTEN_K3_QK_BWD_DK_FAST_V4_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_QK_BWD_DK_FAST_V4_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_QK_BWD_DK_FAST_V4_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_bwd_dk_fast_v4_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_qk_bwd_dk_fast_v4_k{kernel_size}",
            input_names=["query", "key", "d_attn", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_bwd_dk_fast_v4_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_drpb_kernel(kernel_size: int):
    if kernel_size not in _qkrpb_bwd_drpb_cache:
        if kernel_size == 3:
            source = NATTEN_K3_QK_BWD_DRPB_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_QK_BWD_DRPB_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_QK_BWD_DRPB_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_bwd_drpb_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_qk_bwd_drpb_k{kernel_size}",
            input_names=["query", "d_attn", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_bwd_drpb_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_drpb_kernel_fast(kernel_size: int):
    if kernel_size not in _qkrpb_bwd_drpb_fast_cache:
        if kernel_size == 3:
            source = NATTEN_K3_QK_BWD_DRPB_FAST_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_QK_BWD_DRPB_FAST_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_QK_BWD_DRPB_FAST_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_bwd_drpb_fast_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_qk_bwd_drpb_fast_k{kernel_size}",
            input_names=["query", "d_attn", "pi_arr", "ni_arr", "ei_arr", "pj_arr", "nj_arr", "ej_arr", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_bwd_drpb_fast_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_drpb_kernel_fast_u2(kernel_size: int):
    if kernel_size not in _qkrpb_bwd_drpb_fast_u2_cache:
        if kernel_size == 3:
            source = NATTEN_K3_QK_BWD_DRPB_FAST_U2_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_QK_BWD_DRPB_FAST_U2_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_QK_BWD_DRPB_FAST_U2_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_bwd_drpb_fast_u2_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_qk_bwd_drpb_fast_u2_k{kernel_size}",
            input_names=["query", "d_attn", "pi_arr", "ni_arr", "ei_arr", "pj_arr", "nj_arr", "ej_arr", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_bwd_drpb_fast_u2_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_drpb_kernel_fast_v4(kernel_size: int):
    if kernel_size not in _qkrpb_bwd_drpb_fast_v4_cache:
        if kernel_size == 3:
            source = NATTEN_K3_QK_BWD_DRPB_FAST_V4_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_QK_BWD_DRPB_FAST_V4_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_QK_BWD_DRPB_FAST_V4_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_bwd_drpb_fast_v4_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_qk_bwd_drpb_fast_v4_k{kernel_size}",
            input_names=["query", "d_attn", "pi_arr", "ni_arr", "ei_arr", "pj_arr", "nj_arr", "ej_arr", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_bwd_drpb_fast_v4_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_drpb_kernel_fast_split(kernel_size: int):
    if kernel_size not in _qkrpb_bwd_drpb_fast_split_cache:
        if kernel_size == 3:
            source = NATTEN_K3_QK_BWD_DRPB_FAST_SPLIT_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_QK_BWD_DRPB_FAST_SPLIT_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_QK_BWD_DRPB_FAST_SPLIT_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_bwd_drpb_fast_split_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_qk_bwd_drpb_fast_split_k{kernel_size}",
            input_names=["query", "d_attn", "pi_arr", "ni_arr", "ei_arr", "pj_arr", "nj_arr", "ej_arr", "dilation_param", "split_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_bwd_drpb_fast_split_cache[kernel_size]

def _get_or_compile_av_bwd_dattn_kernel(kernel_size: int):
    if kernel_size not in _av_bwd_dattn_cache:
        if kernel_size == 3:
            source = NATTEN_K3_AV_BWD_DATTN_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_AV_BWD_DATTN_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_AV_BWD_DATTN_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _av_bwd_dattn_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_av_bwd_dattn_k{kernel_size}",
            input_names=["d_out", "value", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_bwd_dattn_cache[kernel_size]

def _get_or_compile_av_bwd_dattn_kernel_tg(kernel_size: int, tile: int = 16):
    key = (kernel_size, tile)
    if key not in _av_bwd_dattn_tg_cache:
        if kernel_size == 3:
            source = NATTEN_K3_AV_BWD_DATTN_TG_T8_SOURCE if tile == 8 else NATTEN_K3_AV_BWD_DATTN_TG_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_AV_BWD_DATTN_TG_T8_SOURCE if tile == 8 else NATTEN_K5_AV_BWD_DATTN_TG_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_AV_BWD_DATTN_TG_T8_SOURCE if tile == 8 else NATTEN_K7_AV_BWD_DATTN_TG_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _av_bwd_dattn_tg_cache[key] = mx.fast.metal_kernel(
            name=f"natten2d_av_bwd_dattn_tg_k{kernel_size}_t{tile}",
            input_names=["d_out", "value", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_bwd_dattn_tg_cache[key]

def _get_or_compile_av_bwd_dattn_split_kernel_tg(kernel_size: int, tile: int = 16):
    key = (kernel_size, tile)
    if key not in _av_bwd_dattn_split_tg_cache:
        if kernel_size == 3:
            source = NATTEN_K3_AV_BWD_DATTN_SPLIT_TG_T8_SOURCE if tile == 8 else NATTEN_K3_AV_BWD_DATTN_SPLIT_TG_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_AV_BWD_DATTN_SPLIT_TG_T8_SOURCE if tile == 8 else NATTEN_K5_AV_BWD_DATTN_SPLIT_TG_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_AV_BWD_DATTN_SPLIT_TG_T8_SOURCE if tile == 8 else NATTEN_K7_AV_BWD_DATTN_SPLIT_TG_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _av_bwd_dattn_split_tg_cache[key] = mx.fast.metal_kernel(
            name=f"natten2d_av_bwd_dattn_split_tg_k{kernel_size}_t{tile}",
            input_names=["d_out", "value", "dilation_param", "split_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_bwd_dattn_split_tg_cache[key]

def _apply_1d_dattn_tile(source: str, tile: int) -> str:
    return source.replace("TILE = 64", f"TILE = {tile}")

def _get_or_compile_av_bwd_dattn_kernel_1d_tg(kernel_size: int, tile: int = 64):
    key = (kernel_size, tile)
    if key not in _av_1d_bwd_dattn_tg_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_AV_BWD_DATTN_TG_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_AV_BWD_DATTN_TG_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_AV_BWD_DATTN_TG_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        if tile != 64:
            source = _apply_1d_dattn_tile(source, tile)
        _av_1d_bwd_dattn_tg_cache[key] = mx.fast.metal_kernel(
            name=f"natten1d_av_bwd_dattn_tg_k{kernel_size}_t{tile}",
            input_names=["d_out", "value", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_1d_bwd_dattn_tg_cache[key]

def _get_or_compile_av_bwd_dattn_kernel_1d_tg_v4(kernel_size: int, tile: int = 64):
    key = (kernel_size, tile)
    if key not in _av_1d_bwd_dattn_tg_v4_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_AV_BWD_DATTN_TG_V4_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_AV_BWD_DATTN_TG_V4_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_AV_BWD_DATTN_TG_V4_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        if tile != 64:
            source = _apply_1d_dattn_tile(source, tile)
        _av_1d_bwd_dattn_tg_v4_cache[key] = mx.fast.metal_kernel(
            name=f"natten1d_av_bwd_dattn_tg_v4_k{kernel_size}_t{tile}",
            input_names=["d_out", "value", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_1d_bwd_dattn_tg_v4_cache[key]

def _get_or_compile_av_bwd_dval_kernel(kernel_size: int):
    if kernel_size not in _av_bwd_dval_cache:
        if kernel_size == 3:
            source = NATTEN_K3_AV_BWD_DVAL_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_AV_BWD_DVAL_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_AV_BWD_DVAL_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _av_bwd_dval_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_av_bwd_dval_k{kernel_size}",
            input_names=["d_out", "attn", "value", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_bwd_dval_cache[kernel_size]

def _get_or_compile_av_bwd_dval_kernel_fast(kernel_size: int):
    if kernel_size not in _av_bwd_dval_fast_cache:
        if kernel_size == 3:
            source = NATTEN_K3_AV_BWD_DVAL_FAST_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_AV_BWD_DVAL_FAST_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_AV_BWD_DVAL_FAST_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _av_bwd_dval_fast_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_av_bwd_dval_fast_k{kernel_size}",
            input_names=["d_out", "attn", "value", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_bwd_dval_fast_cache[kernel_size]

def _get_or_compile_av_bwd_dval_kernel_tg(kernel_size: int):
    if kernel_size not in _av_bwd_dval_tg_cache:
        if kernel_size == 3:
            source = NATTEN_K3_AV_BWD_DVAL_TG_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_AV_BWD_DVAL_TG_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_AV_BWD_DVAL_TG_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _av_bwd_dval_tg_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten2d_av_bwd_dval_tg_k{kernel_size}",
            input_names=["d_out", "attn", "value", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_bwd_dval_tg_cache[kernel_size]

def _apply_fused_tile(source: str, tile_h: int, tile_w: int) -> str:
    source = source.replace("TILE_H = 16", f"TILE_H = {tile_h}")
    source = source.replace("TILE_W = 16", f"TILE_W = {tile_w}")
    return source

def _get_or_compile_av_bwd_fused_kernel_tg(kernel_size: int, tile_h: int, tile_w: int, vec4: bool, unroll2: bool):
    key = (kernel_size, tile_h, tile_w, vec4, unroll2)
    if key not in _av_bwd_fused_tg_cache:
        if kernel_size == 3:
            source = NATTEN_K3_AV_BWD_FUSED_TG_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_AV_BWD_FUSED_TG_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_AV_BWD_FUSED_TG_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        if unroll2:
            source = _av_bwd_fused_unroll2_source(source)
        if tile_h != 16 or tile_w != 16:
            source = _apply_fused_tile(source, tile_h, tile_w)
        if not vec4:
            source = _strip_vec4_loads(source)
        _av_bwd_fused_tg_cache[key] = mx.fast.metal_kernel(
            name=f"natten2d_av_bwd_fused_tg_k{kernel_size}_th{tile_h}_tw{tile_w}_v{1 if vec4 else 0}",
            input_names=["d_out", "value", "attn", "dilation_param"],
            output_names=["out_attn", "out_val"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_bwd_fused_tg_cache[key]

def _get_or_compile_av_bwd_fused_split_kernel_tg(kernel_size: int, tile: int, vec4: bool):
    key = (kernel_size, tile, vec4)
    if key not in _av_bwd_fused_split_tg_cache:
        if kernel_size == 3:
            source = NATTEN_K3_AV_BWD_FUSED_SPLIT_TG_T8_SOURCE if tile == 8 else NATTEN_K3_AV_BWD_FUSED_SPLIT_TG_SOURCE
        elif kernel_size == 5:
            source = NATTEN_K5_AV_BWD_FUSED_SPLIT_TG_T8_SOURCE if tile == 8 else NATTEN_K5_AV_BWD_FUSED_SPLIT_TG_SOURCE
        elif kernel_size == 7:
            source = NATTEN_K7_AV_BWD_FUSED_SPLIT_TG_T8_SOURCE if tile == 8 else NATTEN_K7_AV_BWD_FUSED_SPLIT_TG_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        if not vec4:
            source = _strip_vec4_loads(source)
        _av_bwd_fused_split_tg_cache[key] = mx.fast.metal_kernel(
            name=f"natten2d_av_bwd_fused_split_tg_k{kernel_size}_t{tile}_v{1 if vec4 else 0}",
            input_names=["d_out", "value", "attn", "dilation_param", "split_param"],
            output_names=["out_attn", "out_val"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_bwd_fused_split_tg_cache[key]

def _get_or_compile_qkrpb_bwd_dq_kernel_1d(kernel_size: int):
    if kernel_size not in _qkrpb_1d_bwd_dq_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_QK_BWD_DQ_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_QK_BWD_DQ_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_QK_BWD_DQ_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_1d_bwd_dq_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten1d_qk_bwd_dq_k{kernel_size}",
            input_names=["query", "key", "d_attn", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_1d_bwd_dq_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_dk_kernel_1d(kernel_size: int):
    if kernel_size not in _qkrpb_1d_bwd_dk_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_QK_BWD_DK_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_QK_BWD_DK_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_QK_BWD_DK_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_1d_bwd_dk_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten1d_qk_bwd_dk_k{kernel_size}",
            input_names=["query", "key", "d_attn", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_1d_bwd_dk_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_dk_kernel_1d_fast(kernel_size: int):
    if kernel_size not in _qkrpb_1d_bwd_dk_fast_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_QK_BWD_DK_FAST_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_QK_BWD_DK_FAST_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_QK_BWD_DK_FAST_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_1d_bwd_dk_fast_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten1d_qk_bwd_dk_fast_k{kernel_size}",
            input_names=["query", "key", "d_attn", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_1d_bwd_dk_fast_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_dk_kernel_1d_fast_v4(kernel_size: int):
    if kernel_size not in _qkrpb_1d_bwd_dk_fast_v4_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_QK_BWD_DK_FAST_V4_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_QK_BWD_DK_FAST_V4_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_QK_BWD_DK_FAST_V4_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_1d_bwd_dk_fast_v4_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten1d_qk_bwd_dk_fast_v4_k{kernel_size}",
            input_names=["query", "key", "d_attn", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_1d_bwd_dk_fast_v4_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_drpb_kernel_1d(kernel_size: int):
    if kernel_size not in _qkrpb_1d_bwd_drpb_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_QK_BWD_DRPB_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_QK_BWD_DRPB_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_QK_BWD_DRPB_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_1d_bwd_drpb_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten1d_qk_bwd_drpb_k{kernel_size}",
            input_names=["query", "d_attn", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_1d_bwd_drpb_cache[kernel_size]

def _get_or_compile_qkrpb_bwd_drpb_kernel_1d_fast(kernel_size: int):
    if kernel_size not in _qkrpb_1d_bwd_drpb_fast_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_QK_BWD_DRPB_FAST_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_QK_BWD_DRPB_FAST_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_QK_BWD_DRPB_FAST_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _qkrpb_1d_bwd_drpb_fast_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten1d_qk_bwd_drpb_fast_k{kernel_size}",
            input_names=["query", "d_attn", "pi_arr", "ni_arr", "ei_arr", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _qkrpb_1d_bwd_drpb_fast_cache[kernel_size]

def _get_or_compile_av_bwd_dattn_kernel_1d(kernel_size: int):
    if kernel_size not in _av_1d_bwd_dattn_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_AV_BWD_DATTN_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_AV_BWD_DATTN_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_AV_BWD_DATTN_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _av_1d_bwd_dattn_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten1d_av_bwd_dattn_k{kernel_size}",
            input_names=["d_out", "value", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_1d_bwd_dattn_cache[kernel_size]

def _get_or_compile_av_bwd_dval_kernel_1d(kernel_size: int):
    if kernel_size not in _av_1d_bwd_dval_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_AV_BWD_DVAL_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_AV_BWD_DVAL_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_AV_BWD_DVAL_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _av_1d_bwd_dval_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten1d_av_bwd_dval_k{kernel_size}",
            input_names=["d_out", "attn", "value", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_1d_bwd_dval_cache[kernel_size]

def _get_or_compile_av_bwd_dval_kernel_1d_fast(kernel_size: int):
    if kernel_size not in _av_1d_bwd_dval_fast_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_AV_BWD_DVAL_FAST_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_AV_BWD_DVAL_FAST_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_AV_BWD_DVAL_FAST_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _av_1d_bwd_dval_fast_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten1d_av_bwd_dval_fast_k{kernel_size}",
            input_names=["d_out", "attn", "value", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_1d_bwd_dval_fast_cache[kernel_size]

def _get_or_compile_av_bwd_dval_kernel_1d_fast_v4(kernel_size: int):
    if kernel_size not in _av_1d_bwd_dval_fast_v4_cache:
        if kernel_size == 3:
            source = NATTEN_1D_K3_AV_BWD_DVAL_FAST_V4_SOURCE
        elif kernel_size == 5:
            source = NATTEN_1D_K5_AV_BWD_DVAL_FAST_V4_SOURCE
        elif kernel_size == 7:
            source = NATTEN_1D_K7_AV_BWD_DVAL_FAST_V4_SOURCE
        else:
            raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3, 5, or 7.")
        _av_1d_bwd_dval_fast_v4_cache[kernel_size] = mx.fast.metal_kernel(
            name=f"natten1d_av_bwd_dval_fast_v4_k{kernel_size}",
            input_names=["d_out", "attn", "value", "dilation_param"],
            output_names=["out"],
            source=source,
            ensure_row_contiguous=True,
        )
    return _av_1d_bwd_dval_fast_v4_cache[kernel_size]

def _torch_to_mlx(tensor):
    """Convert PyTorch tensor to MLX array."""
    if tensor is None:
        return None
    return mx.array(tensor.detach().cpu().numpy())


def _mlx_to_torch(array, reference_tensor=None):
    """Convert MLX array to PyTorch tensor."""
    try:
        import torch
    except ImportError:
        return array

    np_array = np.array(array)
    result = torch.from_numpy(np_array)

    if reference_tensor is not None:
        result = result.to(dtype=reference_tensor.dtype, device=reference_tensor.device)

    return result


def _is_torch_tensor(x):
    """Check if input is a PyTorch tensor."""
    if x is None:
        return False
    return hasattr(x, '__module__') and 'torch' in x.__module__


def _check_args_against_dim(length: int, kernel_size: int, dilation: int, axis_name: str) -> None:
    if kernel_size * dilation > length:
        raise ValueError(
            f"Invalid NATTEN args on {axis_name}: kernel_size * dilation must be <= axis length. "
            f"Got kernel_size={kernel_size}, dilation={dilation}, {axis_name}={length}."
        )


def _natten2dqkrpb_metal(
    query: mx.array,
    key: mx.array,
    rpb: Optional[mx.array],
    kernel_size: int,
    dilation: int,
) -> mx.array:
    """
    2D NATTEN QK+RPB using fused Metal kernel.

    Returns attention scores BEFORE softmax.
    """
    B, H, height, width, D = query.shape
    L = kernel_size * kernel_size

    # Get compiled kernel
    kernel = _get_or_compile_qkrpb_kernel(kernel_size)

    # Get threadgroup size
    threads = get_threadgroup_for_shape(kernel_size, height, width, 'fp32')

    # Output shape: [B, H, height, width, L]
    output_shape = (B, H, height, width, L)

    # Prepare dilation parameter
    dilation_param = mx.array([dilation], dtype=mx.int32)

    # Call Metal kernel
    outputs = kernel(
        inputs=[query, key, rpb, dilation_param],
        grid=(width, height, B * H),
        threadgroup=threads,
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )

    return outputs[0]


def _natten2dav_metal(
    attention_probs: mx.array,
    value: mx.array,
    kernel_size: int,
    dilation: int,
) -> mx.array:
    """
    2D NATTEN AV using fused Metal kernel.

    Applies softmaxed attention to values.
    """
    B, H, height, width, _ = attention_probs.shape
    _, _, _, _, D = value.shape

    # Get compiled kernel
    kernel = _get_or_compile_av_kernel(kernel_size)

    # Get threadgroup size
    threads = get_threadgroup_for_shape(kernel_size, height, width, 'fp32')

    # Output shape: [B, H, height, width, D]
    output_shape = (B, H, height, width, D)

    # Prepare dilation parameter
    dilation_param = mx.array([dilation], dtype=mx.int32)

    # Call Metal kernel
    outputs = kernel(
        inputs=[attention_probs, value, dilation_param],
        grid=(width, height, B * H),
        threadgroup=threads,
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )

    return outputs[0]


def _threadgroup_1d(length: int):
    return (min(256, int(length)), 1, 1)


def _natten1dqkrpb_metal_backward(query: mx.array, key: mx.array, rpb: Optional[mx.array], d_attn: mx.array,
                                  kernel_size: int, dilation: int):
    B, H, L, D = query.shape
    dilation_param = mx.array([dilation], dtype=mx.int32)
    use_fast = os.environ.get("NATTEN_METAL_BWD_FAST", "0") == "1"
    use_tg = use_fast and dilation == 1 and os.environ.get("NATTEN_METAL_BWD_1D_TG", "1") == "1"
    qk_v4_env = os.environ.get("NATTEN_METAL_BWD_1D_QK_V4", "").strip()
    dq_v4_env = os.environ.get("NATTEN_METAL_BWD_1D_QK_DQ_V4", "").strip()
    dk_v4_env = os.environ.get("NATTEN_METAL_BWD_1D_QK_DK_V4", "").strip()
    if qk_v4_env in ("0", "1"):
        use_dq_v4 = qk_v4_env == "1"
        use_dk_v4 = qk_v4_env == "1"
    else:
        if dq_v4_env in ("0", "1"):
            use_dq_v4 = dq_v4_env == "1"
        else:
            # Default: dq v4 on when vectorization is safe
            use_dq_v4 = (D % 4 == 0)
        if dk_v4_env in ("0", "1"):
            use_dk_v4 = dk_v4_env == "1"
        else:
            # Default: dk v4 on when vectorization is safe
            use_dk_v4 = (D % 4 == 0)
    dq_tile_env = os.environ.get("NATTEN_METAL_BWD_1D_DQ_TG_TILE", "").strip()
    if dq_tile_env in ("64", "128"):
        dq_tile = int(dq_tile_env)
    else:
        dq_tile = 128
    if use_tg and use_dq_v4:
        dq_kernel = _get_or_compile_qkrpb_bwd_dq_kernel_1d_tg_v4(kernel_size, dq_tile)
    else:
        dq_kernel = _get_or_compile_qkrpb_bwd_dq_kernel_1d_tg(kernel_size, dq_tile) if use_tg else _get_or_compile_qkrpb_bwd_dq_kernel_1d(kernel_size)
    if use_fast and use_dk_v4:
        dk_kernel = _get_or_compile_qkrpb_bwd_dk_kernel_1d_fast_v4(kernel_size)
    else:
        dk_kernel = _get_or_compile_qkrpb_bwd_dk_kernel_1d_fast(kernel_size) if use_fast else _get_or_compile_qkrpb_bwd_dk_kernel_1d(kernel_size)
    do_profile = os.environ.get("NATTEN_METAL_BWD_PROFILE", "0") == "1"
    prof_stats = {}
    if do_profile:
        import time
    if use_tg:
        nh = kernel_size // 2
        tg = (min(dq_tile + 2 * nh, 256), 1, 1)
        groups = (L + dq_tile - 1) // dq_tile
        grid = (groups * tg[0], 1, B * H)
        if do_profile:
            prof_stats["tile"] = dq_tile
            prof_stats["K"] = kernel_size
            prof_stats["L"] = L
            prof_stats["D"] = D
        t0 = time.perf_counter() if do_profile else None
        outputs = dq_kernel(
            inputs=[query, key, d_attn, dilation_param],
            grid=grid,
            threadgroup=tg,
            output_shapes=[query.shape],
            output_dtypes=[mx.float32],
        )
        if do_profile:
            mx.eval(outputs[0])
            prof_stats["dq_tg_ms"] = (time.perf_counter() - t0) * 1000
    else:
        t0 = time.perf_counter() if do_profile else None
        outputs = dq_kernel(
            inputs=[query, key, d_attn, dilation_param],
            grid=(L, 1, B * H),
            threadgroup=_threadgroup_1d(L),
            output_shapes=[query.shape],
            output_dtypes=[mx.float32],
        )
        if do_profile:
            mx.eval(outputs[0])
            prof_stats["dq_ms"] = (time.perf_counter() - t0) * 1000
    d_query = outputs[0]
    t0 = time.perf_counter() if do_profile else None
    outputs = dk_kernel(
        inputs=[query, key, d_attn, dilation_param],
        grid=(L, 1, B * H),
        threadgroup=_threadgroup_1d(L),
        output_shapes=[key.shape],
        output_dtypes=[mx.float32],
    )
    if do_profile:
        mx.eval(outputs[0])
        prof_stats["dk_ms"] = (time.perf_counter() - t0) * 1000
    d_key = outputs[0]
    d_rpb = None
    if rpb is not None:
        use_fast = os.environ.get("NATTEN_METAL_BWD_FAST", "0") == "1"
        if use_fast:
            cache_key = (L, kernel_size, dilation)
            cached = _lru_get(_drpb_1d_precompute_cache, cache_key)
            if cached is None:
                pi_list = [get_pb_start(i, L, kernel_size, kernel_size // 2, dilation) for i in range(L)]
                ni_list = [get_window_start(i, L, kernel_size, kernel_size // 2, dilation) for i in range(L)]
                ei_list = [get_window_end(ni_list[i], L, kernel_size, dilation) for i in range(L)]
                pi_arr = mx.array(pi_list, dtype=mx.int32)
                ni_arr = mx.array(ni_list, dtype=mx.int32)
                ei_arr = mx.array(ei_list, dtype=mx.int32)
                _lru_set(_drpb_1d_precompute_cache, cache_key, (pi_arr, ni_arr, ei_arr))
            else:
                pi_arr, ni_arr, ei_arr = cached
            drpb_kernel = _get_or_compile_qkrpb_bwd_drpb_kernel_1d_fast(kernel_size)
            outputs = drpb_kernel(
                inputs=[query, d_attn, pi_arr, ni_arr, ei_arr, dilation_param],
                grid=(rpb.shape[1], 1, H),
                threadgroup=_threadgroup_1d(rpb.shape[1]),
                output_shapes=[rpb.shape],
                output_dtypes=[mx.float32],
            )
            if do_profile:
                mx.eval(outputs[0])
                prof_stats["drpb_fast_ms"] = (time.perf_counter() - t0) * 1000
            d_rpb = outputs[0]
        else:
            drpb_kernel = _get_or_compile_qkrpb_bwd_drpb_kernel_1d(kernel_size)
            t0 = time.perf_counter() if do_profile else None
            outputs = drpb_kernel(
                inputs=[query, d_attn, dilation_param],
                grid=(rpb.shape[1], 1, H),
                threadgroup=_threadgroup_1d(rpb.shape[1]),
                output_shapes=[rpb.shape],
                output_dtypes=[mx.float32],
            )
            if do_profile:
                mx.eval(outputs[0])
                prof_stats["drpb_ms"] = (time.perf_counter() - t0) * 1000
            d_rpb = outputs[0]
    if do_profile:
        import json
        print(f"[metal bwd] stats {json.dumps(prof_stats, sort_keys=True)}")
    return d_query, d_key, d_rpb


def _natten1dav_metal_backward(attn: mx.array, value: mx.array, d_out: mx.array, kernel_size: int, dilation: int):
    B, H, L, D = value.shape
    dilation_param = mx.array([dilation], dtype=mx.int32)
    use_fast = os.environ.get("NATTEN_METAL_BWD_FAST", "0") == "1"
    use_tg = use_fast and dilation == 1 and os.environ.get("NATTEN_METAL_BWD_1D_TG", "1") == "1"
    tile_env = os.environ.get("NATTEN_METAL_BWD_1D_DATTN_TG_TILE", "").strip()
    if tile_env in ("64", "128"):
        dattn_tile = int(tile_env)
    else:
        dattn_tile = 64
    dattn_kernel = _get_or_compile_av_bwd_dattn_kernel_1d_tg(kernel_size, dattn_tile) if use_tg else _get_or_compile_av_bwd_dattn_kernel_1d(kernel_size)
    use_v4_env = os.environ.get("NATTEN_METAL_BWD_1D_DVAL_V4", "").strip()
    if use_v4_env in ("0", "1"):
        use_v4 = use_v4_env == "1"
    else:
        use_v4 = (D % 4 == 0)
    if use_fast:
        dval_kernel = _get_or_compile_av_bwd_dval_kernel_1d_fast_v4(kernel_size) if use_v4 else _get_or_compile_av_bwd_dval_kernel_1d_fast(kernel_size)
    else:
        dval_kernel = _get_or_compile_av_bwd_dval_kernel_1d(kernel_size)
    dattn_v4_env = os.environ.get("NATTEN_METAL_BWD_1D_DATTN_V4", "").strip()
    if dattn_v4_env in ("0", "1"):
        use_dattn_v4 = dattn_v4_env == "1"
    else:
        use_dattn_v4 = (D % 4 == 0)
    if use_tg and use_dattn_v4:
        dattn_kernel = _get_or_compile_av_bwd_dattn_kernel_1d_tg_v4(kernel_size, dattn_tile)
    do_profile = os.environ.get("NATTEN_METAL_BWD_PROFILE", "0") == "1"
    prof_stats = {}
    if do_profile:
        import time
    if use_tg:
        nh = kernel_size // 2
        tg = (min(dattn_tile + 2 * nh, 256), 1, 1)
        groups = (L + dattn_tile - 1) // dattn_tile
        grid = (groups * tg[0], 1, B * H)
        t0 = time.perf_counter() if do_profile else None
        outputs = dattn_kernel(
            inputs=[d_out, value, dilation_param],
            grid=grid,
            threadgroup=tg,
            output_shapes=[attn.shape],
            output_dtypes=[mx.float32],
        )
        if do_profile:
            mx.eval(outputs[0])
            prof_stats["d_attn_tg_ms"] = (time.perf_counter() - t0) * 1000
    else:
        t0 = time.perf_counter() if do_profile else None
        outputs = dattn_kernel(
            inputs=[d_out, value, dilation_param],
            grid=(L, 1, B * H),
            threadgroup=_threadgroup_1d(L),
            output_shapes=[attn.shape],
            output_dtypes=[mx.float32],
        )
        if do_profile:
            mx.eval(outputs[0])
            prof_stats["d_attn_ms"] = (time.perf_counter() - t0) * 1000
    d_attn = outputs[0]
    t0 = time.perf_counter() if do_profile else None
    outputs = dval_kernel(
        inputs=[d_out, attn, value, dilation_param],
        grid=(L, 1, B * H),
        threadgroup=_threadgroup_1d(L),
        output_shapes=[value.shape],
        output_dtypes=[mx.float32],
    )
    if do_profile:
        mx.eval(outputs[0])
        prof_stats["d_val_ms"] = (time.perf_counter() - t0) * 1000
    d_value = outputs[0]
    if do_profile:
        import json
        print(f"[metal bwd] stats {json.dumps(prof_stats, sort_keys=True)}")
    return d_attn, d_value


def _natten2dqkrpb_metal_backward(query: mx.array, key: mx.array, rpb: Optional[mx.array], d_attn: mx.array,
                                  kernel_size: int, dilation: int):
    B, H, height, width, D = query.shape
    dilation_param = mx.array([dilation], dtype=mx.int32)
    threads = get_threadgroup_for_shape(kernel_size, height, width, 'fp32')
    use_fast = os.environ.get("NATTEN_METAL_BWD_FAST", "0") == "1"
    use_tg = use_fast and dilation == 1
    dq_kernel = _get_or_compile_qkrpb_bwd_dq_kernel_tg(kernel_size) if use_tg else _get_or_compile_qkrpb_bwd_dq_kernel(kernel_size)
    dk_v4_env = os.environ.get("NATTEN_METAL_BWD_2D_DK_V4", "").strip()
    if dk_v4_env in ("0", "1"):
        use_dk_v4_2d = dk_v4_env == "1"
    else:
        use_dk_v4_2d = (D % 4 == 0)
    if use_fast and use_dk_v4_2d:
        dk_kernel = _get_or_compile_qkrpb_bwd_dk_kernel_fast_v4(kernel_size)
    else:
        dk_kernel = _get_or_compile_qkrpb_bwd_dk_kernel_fast(kernel_size) if use_fast else _get_or_compile_qkrpb_bwd_dk_kernel(kernel_size)
    do_profile = os.environ.get("NATTEN_METAL_BWD_PROFILE", "0") == "1"
    prof_stats = {}
    if do_profile:
        import time
    if use_tg:
        tg = (min(16 + 2 * (kernel_size // 2), 24), min(16 + 2 * (kernel_size // 2), 24), 1)
        groups_x = (width + 15) // 16
        groups_y = (height + 15) // 16
        grid = (groups_x * tg[0], groups_y * tg[1], B * H)
        t0 = time.perf_counter() if do_profile else None
        outputs = dq_kernel(
            inputs=[query, key, d_attn, dilation_param],
            grid=grid,
            threadgroup=tg,
            output_shapes=[query.shape],
            output_dtypes=[mx.float32],
        )
        if do_profile:
            mx.eval(outputs[0])
            prof_stats["dq_tg_ms"] = (time.perf_counter() - t0) * 1000
    else:
        t0 = time.perf_counter() if do_profile else None
        outputs = dq_kernel(
            inputs=[query, key, d_attn, dilation_param],
            grid=(width, height, B * H),
            threadgroup=threads,
            output_shapes=[query.shape],
            output_dtypes=[mx.float32],
        )
        if do_profile:
            mx.eval(outputs[0])
            prof_stats["dq_ms"] = (time.perf_counter() - t0) * 1000
    d_query = outputs[0]
    t0 = time.perf_counter() if do_profile else None
    outputs = dk_kernel(
        inputs=[query, key, d_attn, dilation_param],
        grid=(width, height, B * H),
        threadgroup=threads,
        output_shapes=[key.shape],
        output_dtypes=[mx.float32],
    )
    if do_profile:
        mx.eval(outputs[0])
        prof_stats["dk_ms"] = (time.perf_counter() - t0) * 1000
    d_key = outputs[0]
    d_rpb = None
    if rpb is not None:
        use_fast = os.environ.get("NATTEN_METAL_BWD_FAST", "0") == "1"
        if use_fast:
            cache_key = (height, width, kernel_size, dilation)
            cached = _lru_get(_drpb_2d_precompute_cache, cache_key)
            if cached is None:
                pi_list = [get_pb_start(i, height, kernel_size, kernel_size // 2, dilation) for i in range(height)]
                ni_list = [get_window_start(i, height, kernel_size, kernel_size // 2, dilation) for i in range(height)]
                ei_list = [get_window_end(ni_list[i], height, kernel_size, dilation) for i in range(height)]
                pj_list = [get_pb_start(j, width, kernel_size, kernel_size // 2, dilation) for j in range(width)]
                nj_list = [get_window_start(j, width, kernel_size, kernel_size // 2, dilation) for j in range(width)]
                ej_list = [get_window_end(nj_list[j], width, kernel_size, dilation) for j in range(width)]
                pi_arr = mx.array(pi_list, dtype=mx.int32)
                ni_arr = mx.array(ni_list, dtype=mx.int32)
                ei_arr = mx.array(ei_list, dtype=mx.int32)
                pj_arr = mx.array(pj_list, dtype=mx.int32)
                nj_arr = mx.array(nj_list, dtype=mx.int32)
                ej_arr = mx.array(ej_list, dtype=mx.int32)
                _lru_set(_drpb_2d_precompute_cache, cache_key, (pi_arr, ni_arr, ei_arr, pj_arr, nj_arr, ej_arr))
            else:
                pi_arr, ni_arr, ei_arr, pj_arr, nj_arr, ej_arr = cached
            drpb_v4_env = os.environ.get("NATTEN_METAL_BWD_2D_DRPB_V4", "").strip()
            if drpb_v4_env in ("0", "1"):
                use_drpb_v4 = drpb_v4_env == "1"
            else:
                use_drpb_v4 = False
            drpb_u2_env = os.environ.get("NATTEN_METAL_BWD_2D_DRPB_U2", "").strip()
            if drpb_u2_env in ("0", "1"):
                use_drpb_u2 = drpb_u2_env == "1"
            else:
                use_drpb_u2 = True
            split_env_raw = os.environ.get("NATTEN_METAL_BWD_2D_DRPB_SPLIT", "").strip()
            split_env = split_env_raw == "1"
            split_n_env = os.environ.get("NATTEN_METAL_BWD_2D_DRPB_NSPLIT", "").strip()
            auto_split = (kernel_size == 7) or (height * width >= 96 * 96)
            if split_n_env.isdigit():
                split_n = int(split_n_env)
            else:
                if height * width >= 96 * 96 and width >= 4096:
                    split_n = 4
                elif kernel_size == 7 and height * width >= 96 * 96:
                    split_n = 4
                elif auto_split:
                    split_n = 2
                else:
                    split_n = 1
            use_split = split_env or (split_env_raw == "" and auto_split)
            if use_split and split_n > 1:
                drpb_kernel = _get_or_compile_qkrpb_bwd_drpb_kernel_fast_split(kernel_size)
                t0 = time.perf_counter() if do_profile else None
                d_rpb = None
                step = (height + split_n - 1) // split_n
                for s in range(split_n):
                    start = s * step
                    end = min(height, start + step)
                    if start >= end:
                        continue
                    split_param = mx.array([start, end], dtype=mx.int32)
                    outputs = drpb_kernel(
                        inputs=[query, d_attn, pi_arr, ni_arr, ei_arr, pj_arr, nj_arr, ej_arr, dilation_param, split_param],
                        grid=(rpb.shape[2], rpb.shape[1], H),
                        threadgroup=(min(16, rpb.shape[2]), min(16, rpb.shape[1]), 1),
                        output_shapes=[rpb.shape],
                        output_dtypes=[mx.float32],
                    )
                    if d_rpb is None:
                        d_rpb = outputs[0]
                    else:
                        d_rpb = d_rpb + outputs[0]
                if do_profile:
                    mx.eval(d_rpb)
                    prof_stats["drpb_fast_ms"] = (time.perf_counter() - t0) * 1000
            else:
                if use_drpb_v4:
                    drpb_kernel = _get_or_compile_qkrpb_bwd_drpb_kernel_fast_v4(kernel_size)
                elif use_drpb_u2:
                    drpb_kernel = _get_or_compile_qkrpb_bwd_drpb_kernel_fast_u2(kernel_size)
                else:
                    drpb_kernel = _get_or_compile_qkrpb_bwd_drpb_kernel_fast(kernel_size)
                t0 = time.perf_counter() if do_profile else None
                outputs = drpb_kernel(
                    inputs=[query, d_attn, pi_arr, ni_arr, ei_arr, pj_arr, nj_arr, ej_arr, dilation_param],
                    grid=(rpb.shape[2], rpb.shape[1], H),
                    threadgroup=(min(16, rpb.shape[2]), min(16, rpb.shape[1]), 1),
                    output_shapes=[rpb.shape],
                    output_dtypes=[mx.float32],
                )
                if do_profile:
                    mx.eval(outputs[0])
                    prof_stats["drpb_fast_ms"] = (time.perf_counter() - t0) * 1000
                d_rpb = outputs[0]
        else:
            drpb_kernel = _get_or_compile_qkrpb_bwd_drpb_kernel(kernel_size)
            t0 = time.perf_counter() if do_profile else None
            outputs = drpb_kernel(
                inputs=[query, d_attn, dilation_param],
                grid=(rpb.shape[2], rpb.shape[1], H),
                threadgroup=(min(16, rpb.shape[2]), min(16, rpb.shape[1]), 1),
                output_shapes=[rpb.shape],
                output_dtypes=[mx.float32],
            )
            if do_profile:
                mx.eval(outputs[0])
                prof_stats["drpb_ms"] = (time.perf_counter() - t0) * 1000
            d_rpb = outputs[0]
    if do_profile:
        import json
        print(f"[metal bwd] stats {json.dumps(prof_stats, sort_keys=True)}")
    return d_query, d_key, d_rpb


def _natten2dav_metal_backward(attn: mx.array, value: mx.array, d_out: mx.array, kernel_size: int, dilation: int):
    global _LAST_METAL_BWD_STATS, _LAST_METAL_BWD_CONFIG
    B, H, height, width, D = value.shape
    dilation_param = mx.array([dilation], dtype=mx.int32)
    threads = get_threadgroup_for_shape(kernel_size, height, width, 'fp32')
    use_fast = os.environ.get("NATTEN_METAL_BWD_FAST", "0") == "1"
    use_tg = use_fast and dilation == 1
    # NOTE: 2D AV backward uses a fused TG kernel by default for correctness + speed.
    # TG d_attn is kept off by default due to known edge-tile correctness issues.
    # To debug unfused paths, set NATTEN_METAL_BWD_ALLOW_UNFUSED=1 and NATTEN_METAL_BWD_FUSED_AV=0.
    allow_unfused = os.environ.get("NATTEN_METAL_BWD_ALLOW_UNFUSED", "0") == "1"
    fused_env = os.environ.get("NATTEN_METAL_BWD_FUSED_AV", "1") == "1"
    use_fused = use_tg and (fused_env or not allow_unfused)
    use_splitkeys = (
        use_fused
        and kernel_size == 7
        and os.environ.get("NATTEN_METAL_BWD_SPLITKEYS", "0") == "1"
    )
    splitkeys_splits = int(os.environ.get("NATTEN_METAL_BWD_SPLITKEYS_NSPLIT", "2"))
    use_split_dattn = (
        use_tg
        and kernel_size == 7
        and os.environ.get("NATTEN_METAL_BWD_SPLIT_DATTN", "0") == "1"
    )
    split_dattn_splits = int(os.environ.get("NATTEN_METAL_BWD_SPLIT_DATTN_NSPLIT", "2"))
    k7_split = os.environ.get("NATTEN_METAL_BWD_K7_SPLIT", "1") == "1"
    use_k7_split = use_fused and kernel_size == 7 and k7_split
    use_tg_dattn = use_tg and os.environ.get("NATTEN_METAL_BWD_DATTN_TG", "0") == "1"
    do_profile = os.environ.get("NATTEN_METAL_BWD_PROFILE", "0") == "1"
    prof_stats = {}
    if do_profile:
        import time
    if use_k7_split:
        use_tg_dattn = True
        use_tg_dval = True
        use_fused = False
        use_splitkeys = False
        use_split_dattn = False
    if use_split_dattn:
        use_fused = False
        use_splitkeys = False
        use_k7_split = False
    dattn_tile_env = os.environ.get("NATTEN_METAL_BWD_DATTN_TG_TILE", "").strip()
    if dattn_tile_env in ("8", "16"):
        dattn_tile = int(dattn_tile_env)
    else:
        dattn_tile = 8 if use_k7_split else 16
    dattn_kernel = _get_or_compile_av_bwd_dattn_kernel_tg(kernel_size, dattn_tile) if use_tg_dattn else _get_or_compile_av_bwd_dattn_kernel(kernel_size)
    use_tg_dval = use_tg and os.environ.get("NATTEN_METAL_BWD_DVAL_TG", "1") == "1"
    if use_k7_split:
        use_tg_dval = True
    dval_kernel = _get_or_compile_av_bwd_dval_kernel_tg(kernel_size) if use_tg_dval else (_get_or_compile_av_bwd_dval_kernel_fast(kernel_size) if use_fast else _get_or_compile_av_bwd_dval_kernel(kernel_size))
    if use_fused:
        tile_env = os.environ.get("NATTEN_METAL_BWD_FUSED_TILE", "").strip()
        tile_h_env = os.environ.get("NATTEN_METAL_BWD_FUSED_TILE_H", "").strip()
        tile_w_env = os.environ.get("NATTEN_METAL_BWD_FUSED_TILE_W", "").strip()
        tile_env_set = tile_env in ("8", "16", "4", "32")
        if tile_env_set:
            tile_h = int(tile_env)
            tile_w = int(tile_env)
        else:
            # Heuristic from M4 profiling:
            # - K=3 prefers TILE=16
            # - K=5 prefers TILE=8
            # - K=7 prefers TILE_H=8, TILE_W=16
            if kernel_size == 7:
                tile_h = 8
                tile_w = 16
            else:
                tile_h = 16 if kernel_size <= 3 else 8
                tile_w = tile_h
        tile_h_env_set = tile_h_env in ("8", "16", "4")
        tile_w_env_set = tile_w_env in ("8", "16", "32")
        if tile_h_env_set:
            tile_h = int(tile_h_env)
        if tile_w_env_set:
            tile_w = int(tile_w_env)
        vec_env = os.environ.get("NATTEN_METAL_BWD_FUSED_VEC4", "").strip()
        if vec_env in ("0", "1"):
            vec4 = vec_env == "1"
        else:
            # Heuristic: vec4 helps K<=5, hurts K=7 on M4
            vec4 = kernel_size <= 5
        if kernel_size == 7:
            k7_vec4_env = os.environ.get("NATTEN_METAL_BWD_K7_VEC4", "").strip()
            if k7_vec4_env in ("0", "1"):
                vec4 = k7_vec4_env == "1"
        k7_force = os.environ.get("NATTEN_METAL_BWD_K7_FORCE_T8V0", "0") == "1"
        if k7_force and kernel_size == 7:
            tile_h = 8
            tile_w = 8
            vec4 = False
        # Extreme-width heuristic: short height, very large width
        if (not tile_env_set) and (not tile_h_env_set) and (not tile_w_env_set):
            if height <= 8 and width >= 4096 and kernel_size == 5:
                tile_h = 4
                tile_w = 16
        if use_splitkeys and tile_h != tile_w:
            tile_w = tile_h
        tg_base_y = min(tile_h + 2 * (kernel_size // 2), 24)
        tg_base_x = min(tile_w + 2 * (kernel_size // 2), 24)
        tg = (tg_base_x, tg_base_y, 1)
        groups_x = (width + (tile_w - 1)) // tile_w
        groups_y = (height + (tile_h - 1)) // tile_h
        grid = (groups_x * tg[0], groups_y * tg[1], B * H)
        unroll2_env = os.environ.get("NATTEN_METAL_BWD_FUSED_UNROLL2", "").strip()
        if unroll2_env in ("0", "1"):
            unroll2 = unroll2_env == "1"
        else:
            unroll2 = (kernel_size == 5 and D == 12)
        if use_splitkeys:
            split_kernel = _get_or_compile_av_bwd_fused_split_kernel_tg(kernel_size, tile_h, vec4)
            L = kernel_size * kernel_size
            splits = max(1, splitkeys_splits)
            split_len = (L + splits - 1) // splits
            t0 = time.perf_counter() if do_profile else None
            attn_parts = []
            d_value = None
            for split_idx in range(splits):
                start = split_idx * split_len
                if start >= L:
                    break
                length = min(split_len, L - start)
                split_param = mx.array([start, length], dtype=mx.int32)
                outputs = split_kernel(
                    inputs=[d_out, value, attn, dilation_param, split_param],
                    grid=grid,
                    threadgroup=tg,
                    output_shapes=[(B, H, height, width, length), value.shape],
                    output_dtypes=[mx.float32, mx.float32],
                )
                attn_parts.append(outputs[0])
                if d_value is None:
                    d_value = outputs[1]
                else:
                    d_value = d_value + outputs[1]
            d_attn = mx.concatenate(attn_parts, axis=-1)
            if do_profile:
                mx.eval(d_value)
                prof_stats["d_attn_dval_split_ms"] = (time.perf_counter() - t0) * 1000
            _LAST_METAL_BWD_STATS = dict(prof_stats)
            _LAST_METAL_BWD_CONFIG = {
                "tile_h": tile_h,
                "tile_w": tile_w,
                "vec4": 1 if vec4 else 0,
                "unroll2": 1 if unroll2 else 0,
                "K": kernel_size,
                "H": height,
                "W": width,
                "D": D,
                "k7_split": False,
                "splitkeys": True,
                "splitkeys_splits": splits,
            }
            return d_attn, d_value
        fused_kernel = _get_or_compile_av_bwd_fused_kernel_tg(kernel_size, tile_h, tile_w, vec4, unroll2)
        t0 = time.perf_counter() if do_profile else None
        outputs = fused_kernel(
            inputs=[d_out, value, attn, dilation_param],
            grid=grid,
            threadgroup=tg,
            output_shapes=[attn.shape, value.shape],
            output_dtypes=[mx.float32, mx.float32],
        )
        if do_profile:
            mx.eval(outputs[0])
            prof_stats["d_attn_dval_fused_ms"] = (time.perf_counter() - t0) * 1000
        d_attn = outputs[0]
        d_value = outputs[1]
        if do_profile:
            _LAST_METAL_BWD_STATS = dict(prof_stats)
            _LAST_METAL_BWD_CONFIG = {
                "tile_h": tile_h,
                "tile_w": tile_w,
                "vec4": 1 if vec4 else 0,
                "K": kernel_size,
                "H": height,
                "W": width,
                "D": D,
                "k7_split": False,
            }
        return d_attn, d_value
    if use_split_dattn:
        nh = kernel_size // 2
        tg_base = min(dattn_tile + 2 * nh, 24)
        tg = (tg_base, tg_base, 1)
        groups_x = (width + (dattn_tile - 1)) // dattn_tile
        groups_y = (height + (dattn_tile - 1)) // dattn_tile
        grid = (groups_x * tg[0], groups_y * tg[1], B * H)
        split_kernel = _get_or_compile_av_bwd_dattn_split_kernel_tg(kernel_size, dattn_tile)
        L = kernel_size * kernel_size
        splits = max(1, split_dattn_splits)
        split_len = (L + splits - 1) // splits
        t0 = time.perf_counter() if do_profile else None
        attn_parts = []
        for split_idx in range(splits):
            start = split_idx * split_len
            if start >= L:
                break
            length = min(split_len, L - start)
            split_param = mx.array([start, length], dtype=mx.int32)
            outputs = split_kernel(
                inputs=[d_out, value, dilation_param, split_param],
                grid=grid,
                threadgroup=tg,
                output_shapes=[(B, H, height, width, length)],
                output_dtypes=[mx.float32],
            )
            attn_parts.append(outputs[0])
        d_attn = mx.concatenate(attn_parts, axis=-1)
        if do_profile:
            mx.eval(d_attn)
            prof_stats["d_attn_split_ms"] = (time.perf_counter() - t0) * 1000
        # d_val path
        t0 = time.perf_counter() if do_profile else None
        if use_tg_dval:
            tg = (min(16 + 2 * (kernel_size // 2), 24), min(16 + 2 * (kernel_size // 2), 24), 1)
            groups_x = (width + 15) // 16
            groups_y = (height + 15) // 16
            grid = (groups_x * tg[0], groups_y * tg[1], B * H)
            outputs = dval_kernel(
                inputs=[d_out, attn, value, dilation_param],
                grid=grid,
                threadgroup=tg,
                output_shapes=[value.shape],
                output_dtypes=[mx.float32],
            )
        else:
            outputs = dval_kernel(
                inputs=[d_out, attn, value, dilation_param],
                grid=(width, height, B * H),
                threadgroup=threads,
                output_shapes=[value.shape],
                output_dtypes=[mx.float32],
            )
        if do_profile:
            mx.eval(outputs[0])
            prof_stats["d_val_ms"] = (time.perf_counter() - t0) * 1000
            _LAST_METAL_BWD_STATS = dict(prof_stats)
            _LAST_METAL_BWD_CONFIG = {
                "tile": -1,
                "vec4": -1,
                "K": kernel_size,
                "H": height,
                "W": width,
                "D": D,
                "k7_split": False,
                "split_dattn": True,
                "split_dattn_splits": splits,
            }
        d_value = outputs[0]
        return d_attn, d_value
    if use_tg_dattn:
        nh = kernel_size // 2
        tg_base = min(dattn_tile + 2 * nh, 24)
        tg = (tg_base, tg_base, 1)
        groups_x = (width + (dattn_tile - 1)) // dattn_tile
        groups_y = (height + (dattn_tile - 1)) // dattn_tile
        grid = (groups_x * tg[0], groups_y * tg[1], B * H)
        t0 = time.perf_counter() if do_profile else None
        outputs = dattn_kernel(
            inputs=[d_out, value, dilation_param],
            grid=grid,
            threadgroup=tg,
            output_shapes=[attn.shape],
            output_dtypes=[mx.float32],
        )
        if do_profile:
            mx.eval(outputs[0])
            prof_stats["d_attn_tg_ms"] = (time.perf_counter() - t0) * 1000
    else:
        t0 = time.perf_counter() if do_profile else None
        outputs = dattn_kernel(
            inputs=[d_out, value, dilation_param],
            grid=(width, height, B * H),
            threadgroup=threads,
            output_shapes=[attn.shape],
            output_dtypes=[mx.float32],
        )
        if do_profile:
            mx.eval(outputs[0])
            prof_stats["d_attn_ms"] = (time.perf_counter() - t0) * 1000
    d_attn = outputs[0]
    t0 = time.perf_counter() if do_profile else None
    if use_tg_dval:
        tg = (min(16 + 2 * (kernel_size // 2), 24), min(16 + 2 * (kernel_size // 2), 24), 1)
        groups_x = (width + 15) // 16
        groups_y = (height + 15) // 16
        grid = (groups_x * tg[0], groups_y * tg[1], B * H)
        outputs = dval_kernel(
            inputs=[d_out, attn, value, dilation_param],
            grid=grid,
            threadgroup=tg,
            output_shapes=[value.shape],
            output_dtypes=[mx.float32],
        )
    else:
        outputs = dval_kernel(
            inputs=[d_out, attn, value, dilation_param],
            grid=(width, height, B * H),
            threadgroup=threads,
            output_shapes=[value.shape],
            output_dtypes=[mx.float32],
        )
    if do_profile:
        mx.eval(outputs[0])
        prof_stats["d_val_ms"] = (time.perf_counter() - t0) * 1000
    d_value = outputs[0]
    if do_profile:
        # Unfused path: record config without tile/vec4.
        prof_stats["tile"] = -1
        prof_stats["vec4"] = -1
        prof_stats["K"] = kernel_size
        prof_stats["H"] = height
        prof_stats["W"] = width
        prof_stats["D"] = D
        _LAST_METAL_BWD_STATS = dict(prof_stats)
        _LAST_METAL_BWD_CONFIG = {
            "tile": -1,
            "vec4": -1,
            "K": kernel_size,
            "H": height,
            "W": width,
            "D": D,
            "k7_split": use_k7_split,
        }
    return d_attn, d_value


def _natten1dqkrpb_metal(
    query: mx.array,
    key: mx.array,
    rpb: Optional[mx.array],
    kernel_size: int,
    dilation: int,
) -> mx.array:
    """1D NATTEN QK+RPB using fused Metal kernel."""
    B, H, L, D = query.shape
    K = kernel_size
    output_shape = (B, H, L, K)
    dilation_param = mx.array([dilation], dtype=mx.int32)

    use_fast_d12_env = os.environ.get("NATTEN_METAL_1D_D12_FAST", "").strip()
    if use_fast_d12_env in ("0", "1"):
        use_fast_d12 = use_fast_d12_env == "1"
    else:
        use_fast_d12 = True

    use_fast_d12 = use_fast_d12 and (kernel_size == 5) and (D == 12) and (rpb is not None)
    if use_fast_d12 and dilation > 1:
        cache_key = (L, kernel_size, dilation)
        cached = _lru_get(_drpb_1d_precompute_cache, cache_key)
        if cached is None:
            pi_list = [get_pb_start(i, L, kernel_size, kernel_size // 2, dilation) for i in range(L)]
            ni_list = [get_window_start(i, L, kernel_size, kernel_size // 2, dilation) for i in range(L)]
            ei_list = [get_window_end(ni_list[i], L, kernel_size, dilation) for i in range(L)]
            pi_arr = mx.array(pi_list, dtype=mx.int32)
            ni_arr = mx.array(ni_list, dtype=mx.int32)
            ei_arr = mx.array(ei_list, dtype=mx.int32)
            _lru_set(_drpb_1d_precompute_cache, cache_key, (pi_arr, ni_arr, ei_arr))
        else:
            pi_arr, ni_arr, ei_arr = cached
        kernel = _get_or_compile_qkrpb_kernel_1d_fast_d12(kernel_size)
        threads = get_threadgroup_for_shape(kernel_size, 1, L, 'fp32')
        outputs = kernel(
            inputs=[query, key, rpb, pi_arr, ni_arr, ei_arr, dilation_param],
            grid=(L, 1, B * H),
            threadgroup=threads,
            output_shapes=[output_shape],
            output_dtypes=[mx.float32],
        )
        return outputs[0]

    kernel = _get_or_compile_qkrpb_kernel_1d(kernel_size)
    threads = get_threadgroup_for_shape(kernel_size, 1, L, 'fp32')

    outputs = kernel(
        inputs=[query, key, rpb, dilation_param],
        grid=(L, 1, B * H),
        threadgroup=threads,
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )

    return outputs[0]


def _natten1dav_metal(
    attention_probs: mx.array,
    value: mx.array,
    kernel_size: int,
    dilation: int,
) -> mx.array:
    """1D NATTEN AV using fused Metal kernel."""
    B, H, L, K = attention_probs.shape
    _, _, _, D = value.shape
    output_shape = (B, H, L, D)
    dilation_param = mx.array([dilation], dtype=mx.int32)

    use_fast_d12_env = os.environ.get("NATTEN_METAL_1D_D12_FAST", "").strip()
    if use_fast_d12_env in ("0", "1"):
        use_fast_d12 = use_fast_d12_env == "1"
    else:
        use_fast_d12 = True

    use_fast_d12 = use_fast_d12 and (kernel_size == 5) and (D == 12)
    if use_fast_d12 and dilation > 1:
        cache_key = (L, kernel_size, dilation)
        cached = _lru_get(_drpb_1d_precompute_cache, cache_key)
        if cached is None:
            pi_list = [get_pb_start(i, L, kernel_size, kernel_size // 2, dilation) for i in range(L)]
            ni_list = [get_window_start(i, L, kernel_size, kernel_size // 2, dilation) for i in range(L)]
            ei_list = [get_window_end(ni_list[i], L, kernel_size, dilation) for i in range(L)]
            pi_arr = mx.array(pi_list, dtype=mx.int32)
            ni_arr = mx.array(ni_list, dtype=mx.int32)
            ei_arr = mx.array(ei_list, dtype=mx.int32)
            _lru_set(_drpb_1d_precompute_cache, cache_key, (pi_arr, ni_arr, ei_arr))
        else:
            pi_arr, ni_arr, ei_arr = cached
        kernel = _get_or_compile_av_kernel_1d_fast_d12(kernel_size)
        threads = get_threadgroup_for_shape(kernel_size, 1, L, 'fp32')
        outputs = kernel(
            inputs=[attention_probs, value, ni_arr, ei_arr, dilation_param],
            grid=(L, 1, B * H),
            threadgroup=threads,
            output_shapes=[output_shape],
            output_dtypes=[mx.float32],
        )
        return outputs[0]

    kernel = _get_or_compile_av_kernel_1d(kernel_size)
    threads = get_threadgroup_for_shape(kernel_size, 1, L, 'fp32')

    outputs = kernel(
        inputs=[attention_probs, value, dilation_param],
        grid=(L, 1, B * H),
        threadgroup=threads,
        output_shapes=[output_shape],
        output_dtypes=[mx.float32],
    )

    return outputs[0]


# Public API with automatic PyTorch conversion
def natten1dqkrpb(query: Any, key: Any, rpb: Optional[Any], kernel_size: int, dilation: int) -> Any:
    """1D NATTEN QK+RPB (returns scores BEFORE softmax)."""
    is_torch = _is_torch_tensor(query)
    if is_torch:
        ref = query
        query = _torch_to_mlx(query)
        key = _torch_to_mlx(key)
        rpb = _torch_to_mlx(rpb) if rpb is not None else None

    _check_args_against_dim(int(query.shape[2]), int(kernel_size), int(dilation), "length")

    result = _natten1dqkrpb_metal(query, key, rpb, kernel_size, dilation)
    return _mlx_to_torch(result, ref) if is_torch else result


def natten1dav(attention_probs: Any, value: Any, kernel_size: int, dilation: int) -> Any:
    """1D NATTEN AV (applies softmaxed attention to values)."""
    is_torch = _is_torch_tensor(attention_probs)
    if is_torch:
        ref = attention_probs
        attention_probs = _torch_to_mlx(attention_probs)
        value = _torch_to_mlx(value)

    _check_args_against_dim(int(value.shape[2]), int(kernel_size), int(dilation), "length")

    result = _natten1dav_metal(attention_probs, value, kernel_size, dilation)
    return _mlx_to_torch(result, ref) if is_torch else result


def natten2dqkrpb(
    query: Any,
    key: Any,
    rpb: Optional[Any],
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]],
) -> Any:
    """2D NATTEN QK+RPB (returns scores BEFORE softmax). Uses fused Metal kernel."""
    # Only support square kernels and uniform dilation for now
    if isinstance(kernel_size, tuple):
        if kernel_size[0] != kernel_size[1]:
            raise ValueError("Only square kernels supported")
        kernel_size = kernel_size[0]

    if isinstance(dilation, tuple):
        if dilation[0] != dilation[1]:
            raise ValueError("Only uniform dilation supported")
        dilation = dilation[0]

    _check_args_against_dim(int(query.shape[2]), int(kernel_size), int(dilation), "height")
    _check_args_against_dim(int(query.shape[3]), int(kernel_size), int(dilation), "width")

    is_torch = _is_torch_tensor(query)
    if is_torch:
        ref = query
        query = _torch_to_mlx(query)
        key = _torch_to_mlx(key)
        rpb = _torch_to_mlx(rpb) if rpb is not None else None

    result = _natten2dqkrpb_metal(query, key, rpb, kernel_size, dilation)
    return _mlx_to_torch(result, ref) if is_torch else result


def natten2dav(
    attention_probs: Any,
    value: Any,
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]],
) -> Any:
    """2D NATTEN AV (applies softmaxed attention to values). Uses fused Metal kernel."""
    # Only support square kernels and uniform dilation for now
    if isinstance(kernel_size, tuple):
        if kernel_size[0] != kernel_size[1]:
            raise ValueError("Only square kernels supported")
        kernel_size = kernel_size[0]

    if isinstance(dilation, tuple):
        if dilation[0] != dilation[1]:
            raise ValueError("Only uniform dilation supported")
        dilation = dilation[0]

    _check_args_against_dim(int(value.shape[2]), int(kernel_size), int(dilation), "height")
    _check_args_against_dim(int(value.shape[3]), int(kernel_size), int(dilation), "width")

    is_torch = _is_torch_tensor(attention_probs)
    if is_torch:
        ref = attention_probs
        attention_probs = _torch_to_mlx(attention_probs)
        value = _torch_to_mlx(value)

    result = _natten2dav_metal(attention_probs, value, kernel_size, dilation)
    return _mlx_to_torch(result, ref) if is_torch else result


def natten2d_qkv_fused(
    query: Any,
    key: Any,
    value: Any,
    rpb: Optional[Any],
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]],
) -> Any:
    """2D fused QKV (QK+Softmax+AV) when vendored fused kernels are available."""
    if isinstance(kernel_size, tuple):
        if kernel_size[0] != kernel_size[1]:
            raise ValueError("Only square kernels supported")
        kernel_size = kernel_size[0]

    if isinstance(dilation, tuple):
        if dilation[0] != dilation[1]:
            raise ValueError("Only uniform dilation supported")
        dilation = dilation[0]

    _check_args_against_dim(int(query.shape[2]), int(kernel_size), int(dilation), "height")
    _check_args_against_dim(int(query.shape[3]), int(kernel_size), int(dilation), "width")

    is_torch = _is_torch_tensor(query)
    if is_torch:
        ref = query
        query = _torch_to_mlx(query)
        key = _torch_to_mlx(key)
        value = _torch_to_mlx(value)
        rpb = _torch_to_mlx(rpb) if rpb is not None else None

    if _d3rm_fused_enabled() and kernel_size in (3, 5, 7):
        fused_map = {3: natten_fused_k3, 5: natten_fused_k5, 7: natten_fused_k7}
        out = fused_map[kernel_size](query, key, value, rpb, dilation=dilation, boundary_mode="shift")
    else:
        # Fallback to unfused QK + softmax + AV
        attn = _natten2dqkrpb_metal(query, key, rpb, kernel_size, dilation)
        attn = mx.softmax(attn, axis=-1)
        out = _natten2dav_metal(attn, value, kernel_size, dilation)

    return _mlx_to_torch(out, ref) if is_torch else out


def natten1dqkrpb_backward(
    d_attn: Any,
    query: Any,
    key: Any,
    rpb: Optional[Any],
    kernel_size: int,
    dilation: int,
) -> Tuple[Any, Any, Optional[Any]]:
    is_torch = _is_torch_tensor(query)
    if is_torch:
        ref = query
        query = _torch_to_mlx(query)
        key = _torch_to_mlx(key)
        d_attn = _torch_to_mlx(d_attn)
        rpb = _torch_to_mlx(rpb) if rpb is not None else None
    d_q, d_k, d_rpb = _natten1dqkrpb_metal_backward(query, key, rpb, d_attn, kernel_size, dilation)
    if is_torch:
        d_q = _mlx_to_torch(d_q, ref)
        d_k = _mlx_to_torch(d_k, ref)
        d_rpb = _mlx_to_torch(d_rpb, rpb) if d_rpb is not None else None
    return d_q, d_k, d_rpb


def natten1dav_backward(
    d_out: Any,
    attention_probs: Any,
    value: Any,
    kernel_size: int,
    dilation: int,
) -> Tuple[Any, Any]:
    is_torch = _is_torch_tensor(attention_probs)
    if is_torch:
        ref = attention_probs
        ref_value = value
        attention_probs = _torch_to_mlx(attention_probs)
        value = _torch_to_mlx(value)
        d_out = _torch_to_mlx(d_out)
    d_attn, d_val = _natten1dav_metal_backward(attention_probs, value, d_out, kernel_size, dilation)
    if is_torch:
        d_attn = _mlx_to_torch(d_attn, ref)
        d_val = _mlx_to_torch(d_val, ref_value)
    return d_attn, d_val


def natten2dqkrpb_backward(
    d_attn: Any,
    query: Any,
    key: Any,
    rpb: Optional[Any],
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]],
) -> Tuple[Any, Any, Optional[Any]]:
    if isinstance(kernel_size, tuple):
        if kernel_size[0] != kernel_size[1]:
            raise ValueError("Only square kernels supported")
        kernel_size = kernel_size[0]
    if isinstance(dilation, tuple):
        if dilation[0] != dilation[1]:
            raise ValueError("Only uniform dilation supported")
        dilation = dilation[0]
    is_torch = _is_torch_tensor(query)
    if is_torch:
        ref = query
        query = _torch_to_mlx(query)
        key = _torch_to_mlx(key)
        d_attn = _torch_to_mlx(d_attn)
        rpb = _torch_to_mlx(rpb) if rpb is not None else None
    d_q, d_k, d_rpb = _natten2dqkrpb_metal_backward(query, key, rpb, d_attn, kernel_size, dilation)
    if is_torch:
        d_q = _mlx_to_torch(d_q, ref)
        d_k = _mlx_to_torch(d_k, ref)
        d_rpb = _mlx_to_torch(d_rpb, rpb) if d_rpb is not None else None
    return d_q, d_k, d_rpb


def natten2dav_backward(
    d_out: Any,
    attention_probs: Any,
    value: Any,
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]],
) -> Tuple[Any, Any]:
    if isinstance(kernel_size, tuple):
        if kernel_size[0] != kernel_size[1]:
            raise ValueError("Only square kernels supported")
        kernel_size = kernel_size[0]
    if isinstance(dilation, tuple):
        if dilation[0] != dilation[1]:
            raise ValueError("Only uniform dilation supported")
        dilation = dilation[0]
    is_torch = _is_torch_tensor(attention_probs)
    if is_torch:
        ref = attention_probs
        ref_value = value
        attention_probs = _torch_to_mlx(attention_probs)
        value = _torch_to_mlx(value)
        d_out = _torch_to_mlx(d_out)
    d_attn, d_val = _natten2dav_metal_backward(attention_probs, value, d_out, kernel_size, dilation)
    if is_torch:
        d_attn = _mlx_to_torch(d_attn, ref)
        d_val = _mlx_to_torch(d_val, ref_value)
    return d_attn, d_val


__all__ = [
    'natten1dqkrpb', 'natten1dav', 'natten2dqkrpb', 'natten2dav',
    'natten1dqkrpb_backward', 'natten1dav_backward',
    'natten2dqkrpb_backward', 'natten2dav_backward',
    'natten2d_qkv_fused',
    'get_d3rm_fused_status',
]

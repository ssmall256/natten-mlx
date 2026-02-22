"""Locked forward tuning tables used by fast_metal dispatch.

These are deterministic defaults generated/updated by benchmarks/forward_tuner.py.
"""

from __future__ import annotations

# Threadgroup launch tables keyed by GPU family, then operation and shape bands.
# Shape key:
# (token_band, head_dim_band, kernel_band, causal_rank_band, stride_unit)
FORWARD_THREADGROUP_TABLE: dict[str, dict[tuple[str, tuple[str, str, str, str, bool]], tuple[int, int, int]]] = {
    "apple_silicon": {
        ("na2d_fused", ("tiny", "d16", "k_small", "c0", True)): (8, 8, 1),
        ("na2d_fused", ("small", "d16", "k_mid", "c1", True)): (16, 8, 1),
        ("na2d_fused", ("medium", "d16", "k_mid", "c1", True)): (16, 8, 1),
        ("na3d_fused", ("small", "d16", "k_small", "c1", True)): (8, 8, 1),
        ("na3d_fused", ("medium", "d16", "k_small", "c1", True)): (16, 8, 1),
        ("na2d_av_split", ("small", "d16", "k_mid", "c1", True)): (16, 8, 1),
        ("na3d_av_split", ("small", "d16", "k_small", "c1", True)): (8, 8, 1),
    },
    "apple_unknown": {},
}


# Softmax strategy tables keyed by GPU family and op family.
# Key:
# (dtype_class, token_band, kernel_band, causal_rank_band, stride_unit)
FORWARD_SOFTMAX_STRATEGY_TABLE: dict[str, dict[tuple[str, tuple[str, str, str, str, bool]], str]] = {
    "apple_silicon": {
        ("na2d_fused", ("lowp", "small", "k_mid", "c1", True)): "recompute",
        ("na3d_fused", ("lowp", "small", "k_small", "c1", True)): "recompute",
    },
    "apple_unknown": {},
}


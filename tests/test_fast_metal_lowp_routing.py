import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx._core import fast_metal


def test_lowp_fp32_route_is_strict_and_deterministic():
    bf16 = getattr(mx, "bfloat16", None)
    if bf16 is None:
        pytest.skip("bfloat16 unavailable")

    old = fast_metal._ENABLE_FORWARD_LOWP_FP32_ROUTE
    fast_metal._ENABLE_FORWARD_LOWP_FP32_ROUTE = True
    try:
        should_route = fast_metal._should_force_fp32_lowp_forward(
            op="na2d_fused",
            dtype=bf16,
            kernel_size=9,
            head_dim=16,
            spatial_shape=(24, 24),
            stride=(1, 1),
            dilation=(1, 1),
            causal=(1, 0),
        )
        assert should_route is True

        # Keep route narrow to avoid broad low-precision regressions.
        assert (
            fast_metal._should_force_fp32_lowp_forward(
                op="na2d_fused",
                dtype=bf16,
                kernel_size=7,
                head_dim=16,
                spatial_shape=(24, 24),
                stride=(1, 1),
                dilation=(1, 1),
                causal=(1, 0),
            )
            is False
        )
        assert (
            fast_metal._should_force_fp32_lowp_forward(
                op="na2d_fused",
                dtype=mx.float16,
                kernel_size=9,
                head_dim=16,
                spatial_shape=(24, 24),
                stride=(1, 1),
                dilation=(1, 1),
                causal=(1, 0),
            )
            is False
        )
        assert (
            fast_metal._should_force_fp32_lowp_forward(
                op="na2d_fused",
                dtype=bf16,
                kernel_size=9,
                head_dim=16,
                spatial_shape=(32, 32),
                stride=(1, 1),
                dilation=(1, 1),
                causal=(1, 0),
            )
            is False
        )
        assert (
            fast_metal._should_force_fp32_lowp_forward(
                op="na2d_fused",
                dtype=bf16,
                kernel_size=9,
                head_dim=16,
                spatial_shape=(24, 24),
                stride=(1, 1),
                dilation=(1, 1),
                causal=(1, 1),
            )
            is False
        )
    finally:
        fast_metal._ENABLE_FORWARD_LOWP_FP32_ROUTE = old

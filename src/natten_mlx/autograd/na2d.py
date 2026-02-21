"""Autograd helpers for 2D neighborhood attention."""

from __future__ import annotations

import mlx.core as mx

from natten_mlx._core import ops


if hasattr(mx, "custom_function"):

    @mx.custom_function
    def _na2d_custom(
        q,
        k,
        v,
        kernel_size_tuple,
        stride_tuple,
        dilation_tuple,
        is_causal_tuple,
        scale_float,
    ):
        return ops.na2d_forward(
            q,
            k,
            v,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
            scale_float,
        )

    @_na2d_custom.vjp
    def _na2d_vjp(primals, cotangent, output):
        q, k, v, kernel_size_tuple, stride_tuple, dilation_tuple, is_causal_tuple, scale_float = primals
        grad_q, grad_k, grad_v = ops.na2d_backward(
            q,
            k,
            v,
            cotangent,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
            scale_float,
        )
        return (grad_q, grad_k, grad_v, None, None, None, None, None)

else:
    _na2d_custom = None


def na2d_with_grad(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    """Autograd-capable entrypoint for 2D NA."""
    backend = ops.get_backend()
    if backend == "pure" or _na2d_custom is None:
        return ops.na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)
    return _na2d_custom(q, k, v, kernel_size, stride, dilation, is_causal, scale)

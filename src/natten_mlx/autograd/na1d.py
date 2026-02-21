"""Autograd helpers for 1D neighborhood attention."""

from __future__ import annotations

import mlx.core as mx

from natten_mlx._core import ops


if hasattr(mx, "custom_function"):

    @mx.custom_function
    def _na1d_custom(
        q,
        k,
        v,
        kernel_size_tuple,
        stride_tuple,
        dilation_tuple,
        is_causal_tuple,
        scale_float,
    ):
        return ops.na1d_forward(
            q,
            k,
            v,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
            scale_float,
        )

    @_na1d_custom.vjp
    def _na1d_vjp(primals, cotangent, output):
        q, k, v, kernel_size_tuple, stride_tuple, dilation_tuple, is_causal_tuple, scale_float = primals
        grad_q, grad_k, grad_v = ops.na1d_backward(
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
    _na1d_custom = None


def na1d_with_grad(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    """Autograd-capable entrypoint for 1D NA.

    Pure backend uses direct MLX primitives; accelerated backends can opt into
    custom VJP path when available.
    """
    backend = ops.get_backend()
    if backend == "pure" or _na1d_custom is None:
        return ops.na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)
    return _na1d_custom(q, k, v, kernel_size, stride, dilation, is_causal, scale)

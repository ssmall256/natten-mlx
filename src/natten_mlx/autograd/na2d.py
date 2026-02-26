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
        try:
            from natten_mlx._core import fast_metal
            if fast_metal._can_use_fused_simd(q.shape[-1]):
                grad_q, grad_k, grad_v = fast_metal.na2d_fused_simd_backward(
                    q, k, v, cotangent, output,
                    kernel_size_tuple, stride_tuple, dilation_tuple,
                    is_causal_tuple, scale_float,
                )
                return (grad_q, grad_k, grad_v, None, None, None, None, None)
        except Exception:
            pass
        grad_q, grad_k, grad_v = ops.na2d_backward(
            q, k, v, cotangent,
            kernel_size_tuple, stride_tuple, dilation_tuple,
            is_causal_tuple, scale_float,
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


if hasattr(mx, "custom_function"):

    @mx.custom_function
    def _na2d_varlen_custom(q, k, v, spatial_sizes, kernel_size_tuple, dilation_tuple, scale_float):
        try:
            from natten_mlx._core import fast_metal
            if fast_metal.is_available():
                return fast_metal.na2d_varlen_forward(
                    q, k, v, spatial_sizes, kernel_size_tuple, dilation_tuple, scale_float)
        except Exception:
            pass
        from natten_mlx._core import pure
        return pure.na2d_varlen_forward(q, k, v, spatial_sizes, kernel_size_tuple, dilation_tuple, scale_float)

    @_na2d_varlen_custom.vjp
    def _na2d_varlen_vjp(primals, cotangent, output):
        q, k, v, spatial_sizes, kernel_size_tuple, dilation_tuple, scale_float = primals
        from natten_mlx._core import pure
        grad_q, grad_k, grad_v = pure.na2d_varlen_backward(
            q, k, v, cotangent, spatial_sizes, kernel_size_tuple, dilation_tuple, scale_float,
        )
        return (grad_q, grad_k, grad_v, None, None, None, None)

else:
    _na2d_varlen_custom = None


def na2d_varlen_with_grad(q, k, v, spatial_sizes, kernel_size, dilation, scale):
    """Autograd-capable entrypoint for variable-length 2D NA."""
    if _na2d_varlen_custom is None:
        from natten_mlx._core import pure
        return pure.na2d_varlen_forward(q, k, v, spatial_sizes, kernel_size, dilation, scale)
    return _na2d_varlen_custom(q, k, v, spatial_sizes, kernel_size, dilation, scale)

"""Autograd helpers for split neighborhood attention ops."""

from __future__ import annotations

import mlx.core as mx

from natten_mlx._core import ops


if hasattr(mx, "custom_function"):

    @mx.custom_function
    def _na1d_qk_custom(
        q,
        k,
        kernel_size_tuple,
        stride_tuple,
        dilation_tuple,
        is_causal_tuple,
        scale_float,
    ):
        return ops.na1d_qk_forward(
            q,
            k,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
            scale_float,
        )

    @_na1d_qk_custom.vjp
    def _na1d_qk_vjp(primals, cotangent, output):
        q, k, kernel_size_tuple, stride_tuple, dilation_tuple, is_causal_tuple, scale_float = primals
        grad_q, grad_k = ops.na1d_qk_backward(
            q,
            k,
            cotangent,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
            scale_float,
        )
        return (grad_q, grad_k, None, None, None, None, None)

    @mx.custom_function
    def _na1d_av_custom(
        attn,
        v,
        kernel_size_tuple,
        stride_tuple,
        dilation_tuple,
        is_causal_tuple,
    ):
        return ops.na1d_av_forward(
            attn,
            v,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
        )

    @_na1d_av_custom.vjp
    def _na1d_av_vjp(primals, cotangent, output):
        attn, v, kernel_size_tuple, stride_tuple, dilation_tuple, is_causal_tuple = primals
        grad_attn, grad_v = ops.na1d_av_backward(
            attn,
            v,
            cotangent,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
        )
        return (grad_attn, grad_v, None, None, None, None)

    @mx.custom_function
    def _na2d_qk_custom(
        q,
        k,
        kernel_size_tuple,
        stride_tuple,
        dilation_tuple,
        is_causal_tuple,
        scale_float,
    ):
        return ops.na2d_qk_forward(
            q,
            k,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
            scale_float,
        )

    @_na2d_qk_custom.vjp
    def _na2d_qk_vjp(primals, cotangent, output):
        q, k, kernel_size_tuple, stride_tuple, dilation_tuple, is_causal_tuple, scale_float = primals
        grad_q, grad_k = ops.na2d_qk_backward(
            q,
            k,
            cotangent,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
            scale_float,
        )
        return (grad_q, grad_k, None, None, None, None, None)

    @mx.custom_function
    def _na2d_av_custom(
        attn,
        v,
        kernel_size_tuple,
        stride_tuple,
        dilation_tuple,
        is_causal_tuple,
    ):
        return ops.na2d_av_forward(
            attn,
            v,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
        )

    @_na2d_av_custom.vjp
    def _na2d_av_vjp(primals, cotangent, output):
        attn, v, kernel_size_tuple, stride_tuple, dilation_tuple, is_causal_tuple = primals
        grad_attn, grad_v = ops.na2d_av_backward(
            attn,
            v,
            cotangent,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
        )
        return (grad_attn, grad_v, None, None, None, None)

    @mx.custom_function
    def _na3d_qk_custom(
        q,
        k,
        kernel_size_tuple,
        stride_tuple,
        dilation_tuple,
        is_causal_tuple,
        scale_float,
    ):
        return ops.na3d_qk_forward(
            q,
            k,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
            scale_float,
        )

    @_na3d_qk_custom.vjp
    def _na3d_qk_vjp(primals, cotangent, output):
        q, k, kernel_size_tuple, stride_tuple, dilation_tuple, is_causal_tuple, scale_float = primals
        grad_q, grad_k = ops.na3d_qk_backward(
            q,
            k,
            cotangent,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
            scale_float,
        )
        return (grad_q, grad_k, None, None, None, None, None)

    @mx.custom_function
    def _na3d_av_custom(
        attn,
        v,
        kernel_size_tuple,
        stride_tuple,
        dilation_tuple,
        is_causal_tuple,
    ):
        return ops.na3d_av_forward(
            attn,
            v,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
        )

    @_na3d_av_custom.vjp
    def _na3d_av_vjp(primals, cotangent, output):
        attn, v, kernel_size_tuple, stride_tuple, dilation_tuple, is_causal_tuple = primals
        grad_attn, grad_v = ops.na3d_av_backward(
            attn,
            v,
            cotangent,
            kernel_size_tuple,
            stride_tuple,
            dilation_tuple,
            is_causal_tuple,
        )
        return (grad_attn, grad_v, None, None, None, None)

else:
    _na1d_qk_custom = None
    _na1d_av_custom = None
    _na2d_qk_custom = None
    _na2d_av_custom = None
    _na3d_qk_custom = None
    _na3d_av_custom = None


def na1d_qk_with_grad(q, k, kernel_size, stride, dilation, is_causal, scale):
    backend = ops.get_backend()
    if backend == "pure" or _na1d_qk_custom is None:
        return ops.na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)
    return _na1d_qk_custom(q, k, kernel_size, stride, dilation, is_causal, scale)


def na1d_av_with_grad(attn, v, kernel_size, stride, dilation, is_causal):
    backend = ops.get_backend()
    if backend == "pure" or _na1d_av_custom is None:
        return ops.na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)
    return _na1d_av_custom(attn, v, kernel_size, stride, dilation, is_causal)


def na2d_qk_with_grad(q, k, kernel_size, stride, dilation, is_causal, scale):
    backend = ops.get_backend()
    if backend == "pure" or _na2d_qk_custom is None:
        return ops.na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)
    return _na2d_qk_custom(q, k, kernel_size, stride, dilation, is_causal, scale)


def na2d_av_with_grad(attn, v, kernel_size, stride, dilation, is_causal):
    backend = ops.get_backend()
    if backend == "pure" or _na2d_av_custom is None:
        return ops.na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)
    return _na2d_av_custom(attn, v, kernel_size, stride, dilation, is_causal)


def na3d_qk_with_grad(q, k, kernel_size, stride, dilation, is_causal, scale):
    backend = ops.get_backend()
    if backend == "pure" or _na3d_qk_custom is None:
        return ops.na3d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)
    return _na3d_qk_custom(q, k, kernel_size, stride, dilation, is_causal, scale)


def na3d_av_with_grad(attn, v, kernel_size, stride, dilation, is_causal):
    backend = ops.get_backend()
    if backend == "pure" or _na3d_av_custom is None:
        return ops.na3d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)
    return _na3d_av_custom(attn, v, kernel_size, stride, dilation, is_causal)

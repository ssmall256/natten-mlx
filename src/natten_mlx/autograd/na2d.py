"""Autograd helpers for 2D neighborhood attention."""

from __future__ import annotations

import mlx.core as mx

from natten_mlx._core import ops, pure


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

        def forward_fn(q_in, k_in, v_in):
            backend_name = ops.get_backend()
            fn = pure.na2d_forward if backend_name != "pure" else ops.na2d_forward
            return fn(
                q_in,
                k_in,
                v_in,
                kernel_size_tuple,
                stride_tuple,
                dilation_tuple,
                is_causal_tuple,
                scale_float,
            )

        _, vjp_fn = mx.vjp(forward_fn, q, k, v)
        grad_q, grad_k, grad_v = vjp_fn(cotangent)
        return (grad_q, grad_k, grad_v, None, None, None, None, None)

else:
    _na2d_custom = None


def na2d_with_grad(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    """Autograd-capable entrypoint for 2D NA."""
    backend = ops.get_backend()
    if backend == "pure" or _na2d_custom is None:
        return ops.na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)
    return _na2d_custom(q, k, v, kernel_size, stride, dilation, is_causal, scale)

"""Attention merging for natten-mlx.

Merges multiple attention outputs that share the same query, as if their
key/value contexts had been concatenated. Uses a numerically stable
sigmoid-based formulation from ring-flash-attention.

Typical usage:
    out_a, lse_a = na1d(q, k_a, v_a, kernel_size=7, return_lse=True)
    out_b, lse_b = na1d(q, k_b, v_b, kernel_size=7, return_lse=True)
    merged_out, merged_lse = merge_attentions([out_a, out_b], [lse_a, lse_b])
"""
from __future__ import annotations

from typing import List, Tuple

import mlx.core as mx
import mlx.nn


def merge_attentions(
    outputs: List[mx.array],
    lse_tensors: List[mx.array],
) -> Tuple[mx.array, mx.array]:
    """Merge multiple attention outputs sharing the same query.

    Takes attention outputs and their logsumexp tensors and merges them
    as if their key/value contexts had been concatenated.

    Args:
        outputs: List of attention output arrays, each ``[B, ..spatial.., H, D]``.
        lse_tensors: List of logsumexp arrays, each ``[B, ..spatial.., H]``.

    Returns:
        ``(merged_output, merged_lse)``
    """
    if len(outputs) < 2:
        raise ValueError("merge_attentions expects at least two outputs.")
    if len(outputs) != len(lse_tensors):
        raise ValueError("Number of outputs and LSE tensors must match.")

    ref_shape = outputs[0].shape
    for i, (o, l) in enumerate(zip(outputs, lse_tensors)):
        if o.shape != ref_shape:
            raise ValueError(
                f"Output {i} shape {o.shape} does not match output 0 shape {ref_shape}."
            )
        expected_lse_shape = ref_shape[:-1]
        if l.shape != expected_lse_shape:
            raise ValueError(
                f"LSE {i} shape {l.shape} does not match expected {expected_lse_shape}."
            )

    lse_list = [mx.expand_dims(lse, axis=-1).astype(mx.float32) for lse in lse_tensors]
    out_list = [o.astype(mx.float32) for o in outputs]

    # Sigmoid-based merge (ref: ring-flash-attention)
    output = out_list[0] - mx.sigmoid(lse_list[1] - lse_list[0]) * (
        out_list[0] - out_list[1]
    )
    logsumexp = lse_list[0] - mlx.nn.log_sigmoid(lse_list[0] - lse_list[1])

    for i in range(2, len(out_list)):
        output = output - mx.sigmoid(lse_list[i] - logsumexp) * (
            output - out_list[i]
        )
        logsumexp = logsumexp - mlx.nn.log_sigmoid(logsumexp - lse_list[i])

    return output.astype(outputs[0].dtype), mx.squeeze(logsumexp, axis=-1)


__all__ = ["merge_attentions"]

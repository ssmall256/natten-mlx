"""Window helper formulas matching upstream NATTEN boundary semantics."""

from __future__ import annotations

import numpy as np


def get_window_start(
    index: int,
    length: int,
    kernel_size: int,
    neighborhood_size: int,
    dilation: int,
) -> int:
    """Compute non-causal window start with NATTEN phase alignment."""
    if dilation <= 1:
        return max(index - neighborhood_size, 0) + (
            (index + neighborhood_size >= length)
            * (length - index - neighborhood_size - 1)
        )

    ni = index - neighborhood_size * dilation
    if ni < 0:
        return index % dilation

    if index + neighborhood_size * dilation >= length:
        imodd = index % dilation
        a = (length // dilation) * dilation
        b = length - a
        if imodd < b:
            return length - b + imodd - 2 * neighborhood_size * dilation
        return a + imodd - kernel_size * dilation

    return ni


def get_window_end(start_index: int, length: int, kernel_size: int, dilation: int) -> int:
    """Compute half-open window end index."""
    return min(length, start_index + kernel_size * dilation)


def get_pb_start(
    index: int,
    length: int,
    kernel_size: int,
    neighborhood_size: int,
    dilation: int,
) -> int:
    """Compute RPB start index with NATTEN coupling at boundaries."""
    if dilation <= 1:
        return neighborhood_size + (
            (index < neighborhood_size) * (neighborhood_size - index)
        ) + (
            (index + neighborhood_size >= length)
            * (length - index - 1 - neighborhood_size)
        )

    if index - neighborhood_size * dilation < 0:
        return kernel_size - 1 - (index // dilation)
    if index + neighborhood_size * dilation >= length:
        return (length - index - 1) // dilation
    return neighborhood_size


def compute_window_start_end(
    query_positions: np.ndarray,
    length: int,
    kernel_size: int,
    dilation: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized starts/ends for a list of query positions."""
    neighborhood_size = kernel_size // 2
    starts = np.array(
        [
            get_window_start(
                int(idx), length, kernel_size, neighborhood_size, dilation
            )
            for idx in query_positions
        ],
        dtype=np.int32,
    )
    ends = np.array(
        [get_window_end(int(st), length, kernel_size, dilation) for st in starts],
        dtype=np.int32,
    )
    return starts, ends


def compute_pb_start(
    query_positions: np.ndarray,
    length: int,
    kernel_size: int,
    dilation: int,
) -> np.ndarray:
    """Vectorized pb-start indices for a list of query positions."""
    neighborhood_size = kernel_size // 2
    return np.array(
        [
            get_pb_start(int(idx), length, kernel_size, neighborhood_size, dilation)
            for idx in query_positions
        ],
        dtype=np.int32,
    )


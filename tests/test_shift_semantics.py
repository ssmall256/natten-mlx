import numpy as np

from natten_mlx.utils.window import (
    compute_pb_start,
    compute_window_start_end,
    get_pb_start,
    get_window_start,
)


def test_phase_alignment_dilated_shift_windows():
    length = 64
    kernel_size = 7
    dilation = 4

    qpos = np.arange(length, dtype=np.int32)
    starts, ends = compute_window_start_end(qpos, length, kernel_size, dilation)

    for index, start, end in zip(qpos, starts, ends):
        for ki in range(kernel_size):
            key_pos = int(start + ki * dilation)
            if key_pos < end:
                assert key_pos % dilation == int(index % dilation)


def test_shift_window_bounds_and_size_constraints():
    length = 64
    kernel_size = 7
    dilation = 4

    qpos = np.arange(length, dtype=np.int32)
    starts, ends = compute_window_start_end(qpos, length, kernel_size, dilation)

    for start, end in zip(starts, ends):
        assert 0 <= int(start) < length
        assert 0 < int(end) <= length


def test_pb_start_stays_within_valid_range():
    length = 64
    kernel_size = 7
    dilation = 4

    qpos = np.arange(length, dtype=np.int32)
    pb = compute_pb_start(qpos, length, kernel_size, dilation)

    for pi in pb:
        for ki in range(kernel_size):
            idx = int(pi + ki)
            assert 0 <= idx <= (2 * kernel_size - 2)


def test_dilation1_right_boundary_matches_natten_formula():
    length = 32
    kernel_size = 3
    neighborhood_size = 1
    dilation = 1

    index = 31
    ni = get_window_start(index, length, kernel_size, neighborhood_size, dilation)
    pi = get_pb_start(index, length, kernel_size, neighborhood_size, dilation)

    assert ni == 29
    assert pi == 0

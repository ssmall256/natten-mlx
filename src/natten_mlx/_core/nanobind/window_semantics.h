#pragma once

#include <utility>

namespace natten_mlx::nanobind_window {

int get_window_start(
    int index,
    int length,
    int kernel_size,
    int neighborhood_size,
    int dilation);

int get_window_end(int start_index, int length, int kernel_size, int dilation);

int get_pb_start(
    int index,
    int length,
    int kernel_size,
    int neighborhood_size,
    int dilation);

std::pair<int, int> get_window_start_end(
    int index,
    int length,
    int kernel_size,
    int dilation);

}  // namespace natten_mlx::nanobind_window


#include "window_semantics.h"

#include <algorithm>

namespace natten_mlx::nanobind_window {

int get_window_start(
    int index,
    int length,
    int kernel_size,
    int neighborhood_size,
    int dilation) {
    if (dilation <= 1) {
        return std::max(index - neighborhood_size, 0) +
            ((index + neighborhood_size >= length)
                 ? (length - index - neighborhood_size - 1)
                 : 0);
    }

    int ni = index - neighborhood_size * dilation;
    if (ni < 0) {
        return index % dilation;
    }

    if (index + neighborhood_size * dilation >= length) {
        int imod = index % dilation;
        int a = (length / dilation) * dilation;
        int b = length - a;
        if (imod < b) {
            return length - b + imod - 2 * neighborhood_size * dilation;
        }
        return a + imod - kernel_size * dilation;
    }

    return ni;
}

int get_window_end(int start_index, int length, int kernel_size, int dilation) {
    return std::min(length, start_index + kernel_size * dilation);
}

int get_pb_start(
    int index,
    int length,
    int kernel_size,
    int neighborhood_size,
    int dilation) {
    if (dilation <= 1) {
        return neighborhood_size +
            ((index < neighborhood_size) ? (neighborhood_size - index) : 0) +
            ((index + neighborhood_size >= length)
                 ? (length - index - 1 - neighborhood_size)
                 : 0);
    }

    if (index - neighborhood_size * dilation < 0) {
        return kernel_size - 1 - (index / dilation);
    }
    if (index + neighborhood_size * dilation >= length) {
        return (length - index - 1) / dilation;
    }
    return neighborhood_size;
}

std::pair<int, int> get_window_start_end(
    int index,
    int length,
    int kernel_size,
    int dilation) {
    int neighborhood_size = kernel_size / 2;
    int start = get_window_start(index, length, kernel_size, neighborhood_size, dilation);
    int end = get_window_end(start, length, kernel_size, dilation);
    return {start, end};
}

}  // namespace natten_mlx::nanobind_window


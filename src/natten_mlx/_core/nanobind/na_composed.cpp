#include "na_composed.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <nanobind/stl/tuple.h>

#include "na_split_forward.h"
#include "na1d_primitive.h"
#include "na2d_primitive.h"
#include "na3d_primitive.h"
#include "na1d_bwd_primitive.h"
#include "na2d_bwd_primitive.h"
#include "na3d_bwd_primitive.h"
#include "py_dispatch.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace {

bool is_sequence(const nb::object& obj) {
    return nb::isinstance<nb::tuple>(obj) || nb::isinstance<nb::list>(obj);
}

int scalar_or_index_int(const nb::object& obj, size_t idx) {
    if (!is_sequence(obj)) {
        return nb::cast<int>(obj);
    }
    nb::sequence seq = nb::cast<nb::sequence>(obj);
    if (idx >= static_cast<size_t>(nb::len(seq))) {
        throw std::runtime_error("invalid parameter rank");
    }
    return nb::cast<int>(seq[idx]);
}

bool scalar_or_index_bool(const nb::object& obj, size_t idx) {
    if (!is_sequence(obj)) {
        return nb::cast<bool>(obj);
    }
    nb::sequence seq = nb::cast<nb::sequence>(obj);
    if (idx >= static_cast<size_t>(nb::len(seq))) {
        throw std::runtime_error("invalid parameter rank");
    }
    return nb::cast<bool>(seq[idx]);
}

}  // namespace

namespace natten_mlx::nanobind_composed {

nb::object na1d_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    // Try v2 primitive path first
    {
        const char* v2_env = std::getenv("NATTEN_NANOBIND_DISABLE_V2");
        bool v2_disabled = v2_env != nullptr && std::string(v2_env) == "1";
        if (!v2_disabled) {
            try {
                mx::array q_arr = nb::cast<mx::array>(q);
                int ks = scalar_or_index_int(kernel_size, 0);
                int s = scalar_or_index_int(stride, 0);
                int d = scalar_or_index_int(dilation, 0);
                bool c = scalar_or_index_bool(is_causal, 0);
                int head_dim = q_arr.shape(3);
                bool eligible = (ks > 0) && (ks % 2 == 1) && (ks <= 63) &&
                    s >= 1 && d >= 1;
                if (eligible) {
                    mx::array k_arr = nb::cast<mx::array>(k);
                    mx::array v_arr = nb::cast<mx::array>(v);
                    float sc;
                    if (scale.is_none()) {
                        sc = std::pow(static_cast<float>(head_dim), -0.5f);
                    } else {
                        sc = nb::cast<float>(scale);
                    }
                    mx::array out = natten_mlx::na1d_fused_forward_v2(
                        q_arr, k_arr, v_arr, ks, s, d, c, sc);
                    return nb::cast(out);
                }
            } catch (const std::exception& e) {
                std::cerr << "natten_mlx: v2 na1d_forward fell back to pure: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "natten_mlx: v2 na1d_forward fell back to pure (unknown error)" << std::endl;
            }
        }
    }
    return natten_mlx::nanobind_backend::call_pure(
        "na1d_forward", q, k, v, kernel_size, stride, dilation, is_causal, scale);
}

nb::object na2d_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    // Try v2 primitive path first
    {
        const char* v2_env = std::getenv("NATTEN_NANOBIND_DISABLE_V2");
        bool v2_disabled = v2_env != nullptr && std::string(v2_env) == "1";
        if (!v2_disabled) {
            try {
                mx::array q_arr = nb::cast<mx::array>(q);
                mx::array k_arr = nb::cast<mx::array>(k);
                mx::array v_arr = nb::cast<mx::array>(v);
                int ks = scalar_or_index_int(kernel_size, 0);
                int sh = scalar_or_index_int(stride, 0);
                int sw = scalar_or_index_int(stride, 1);
                int dh = scalar_or_index_int(dilation, 0);
                int dw = scalar_or_index_int(dilation, 1);
                bool ch = scalar_or_index_bool(is_causal, 0);
                bool cw = scalar_or_index_bool(is_causal, 1);
                int head_dim = q_arr.shape(4);
                bool eligible = (ks > 0) && (ks % 2 == 1) && (ks <= 13) &&
                    sh >= 1 && sw >= 1 && dh >= 1 && dw >= 1;
                if (eligible) {
                    float sc;
                    if (scale.is_none()) {
                        sc = std::pow(static_cast<float>(head_dim), -0.5f);
                    } else {
                        sc = nb::cast<float>(scale);
                    }
                    mx::array out = natten_mlx::na2d_fused_forward_v2(
                        q_arr, k_arr, v_arr,
                        ks, sh, sw, dh, dw, ch, cw, sc);
                    return nb::cast(out);
                }
            } catch (const std::exception& e) {
                std::cerr << "natten_mlx: v2 na2d_forward fell back to pure: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "natten_mlx: v2 na2d_forward fell back to pure (unknown error)" << std::endl;
            }
        }
    }
    return natten_mlx::nanobind_backend::call_pure(
        "na2d_forward", q, k, v, kernel_size, stride, dilation, is_causal, scale);
}

nb::object na3d_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    // Try v2 primitive path first
    {
        const char* v2_env = std::getenv("NATTEN_NANOBIND_DISABLE_V2");
        bool v2_disabled = v2_env != nullptr && std::string(v2_env) == "1";
        if (!v2_disabled) {
            try {
                mx::array q_arr = nb::cast<mx::array>(q);
                int ks = scalar_or_index_int(kernel_size, 0);
                int sd = scalar_or_index_int(stride, 0);
                int sh = scalar_or_index_int(stride, 1);
                int sw = scalar_or_index_int(stride, 2);
                int dd = scalar_or_index_int(dilation, 0);
                int dh = scalar_or_index_int(dilation, 1);
                int dw = scalar_or_index_int(dilation, 2);
                bool cd = scalar_or_index_bool(is_causal, 0);
                bool ch = scalar_or_index_bool(is_causal, 1);
                bool cw = scalar_or_index_bool(is_causal, 2);
                int head_dim = q_arr.shape(5);
                bool eligible = (ks > 0) && (ks % 2 == 1) && (ks <= 7) &&
                    sd >= 1 && sh >= 1 && sw >= 1 &&
                    dd >= 1 && dh >= 1 && dw >= 1;
                if (eligible) {
                    mx::array k_arr = nb::cast<mx::array>(k);
                    mx::array v_arr = nb::cast<mx::array>(v);
                    float sc;
                    if (scale.is_none()) {
                        sc = std::pow(static_cast<float>(head_dim), -0.5f);
                    } else {
                        sc = nb::cast<float>(scale);
                    }
                    mx::array out = natten_mlx::na3d_fused_forward_v2(
                        q_arr, k_arr, v_arr, ks,
                        sd, sh, sw, dd, dh, dw, cd, ch, cw, sc);
                    return nb::cast(out);
                }
            } catch (const std::exception& e) {
                std::cerr << "natten_mlx: v2 na3d_forward fell back to pure: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "natten_mlx: v2 na3d_forward fell back to pure (unknown error)" << std::endl;
            }
        }
    }
    return natten_mlx::nanobind_backend::call_pure(
        "na3d_forward", q, k, v, kernel_size, stride, dilation, is_causal, scale);
}

nb::object na1d_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    // Try v2 primitive backward path first
    {
        const char* v2_env = std::getenv("NATTEN_NANOBIND_DISABLE_V2");
        bool v2_disabled = v2_env != nullptr && std::string(v2_env) == "1";
        if (!v2_disabled) {
            try {
                mx::array q_arr = nb::cast<mx::array>(q);
                mx::array k_arr = nb::cast<mx::array>(k);
                mx::array v_arr = nb::cast<mx::array>(v);
                mx::array go_arr = nb::cast<mx::array>(grad_out);
                int ks = scalar_or_index_int(kernel_size, 0);
                int s = scalar_or_index_int(stride, 0);
                int d = scalar_or_index_int(dilation, 0);
                bool c = scalar_or_index_bool(is_causal, 0);
                int head_dim = q_arr.shape(3);
                bool eligible = (ks > 0) && (ks % 2 == 1) && (ks <= 63) &&
                    s >= 1 && d >= 1;
                if (eligible) {
                    float sc;
                    if (scale.is_none()) {
                        sc = std::pow(static_cast<float>(head_dim), -0.5f);
                    } else {
                        sc = nb::cast<float>(scale);
                    }
                    auto grads = natten_mlx::na1d_backward_v2(
                        q_arr, k_arr, v_arr, go_arr,
                        ks, s, d, c, sc);
                    return nb::cast(nb::make_tuple(
                        nb::cast(grads[0]), nb::cast(grads[1]), nb::cast(grads[2])));
                }
            } catch (const std::exception& e) {
                std::cerr << "natten_mlx: v2 na1d_backward fell back to pure: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "natten_mlx: v2 na1d_backward fell back to pure (unknown error)" << std::endl;
            }
        }
    }
    return natten_mlx::nanobind_backend::call_pure(
        "na1d_backward", q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
}

nb::object na2d_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    // Try v2 primitive backward path first
    {
        const char* v2_env = std::getenv("NATTEN_NANOBIND_DISABLE_V2");
        bool v2_disabled = v2_env != nullptr && std::string(v2_env) == "1";
        if (!v2_disabled) {
            try {
                mx::array q_arr = nb::cast<mx::array>(q);
                mx::array k_arr = nb::cast<mx::array>(k);
                mx::array v_arr = nb::cast<mx::array>(v);
                mx::array go_arr = nb::cast<mx::array>(grad_out);
                int ks = scalar_or_index_int(kernel_size, 0);
                int sh = scalar_or_index_int(stride, 0);
                int sw = scalar_or_index_int(stride, 1);
                int dh = scalar_or_index_int(dilation, 0);
                int dw = scalar_or_index_int(dilation, 1);
                bool ch = scalar_or_index_bool(is_causal, 0);
                bool cw = scalar_or_index_bool(is_causal, 1);
                int head_dim = q_arr.shape(4);
                bool eligible = (ks > 0) && (ks % 2 == 1) && (ks <= 13) &&
                    sh >= 1 && sw >= 1 && dh >= 1 && dw >= 1;
                if (eligible) {
                    float sc;
                    if (scale.is_none()) {
                        sc = std::pow(static_cast<float>(head_dim), -0.5f);
                    } else {
                        sc = nb::cast<float>(scale);
                    }
                    auto grads = natten_mlx::na2d_backward_v2(
                        q_arr, k_arr, v_arr, go_arr,
                        ks, sh, sw, dh, dw, ch, cw, sc);
                    return nb::cast(nb::make_tuple(
                        nb::cast(grads[0]), nb::cast(grads[1]), nb::cast(grads[2])));
                }
            } catch (const std::exception& e) {
                std::cerr << "natten_mlx: v2 na2d_backward fell back to pure: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "natten_mlx: v2 na2d_backward fell back to pure (unknown error)" << std::endl;
            }
        }
    }
    return natten_mlx::nanobind_backend::call_pure(
        "na2d_backward", q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
}

nb::object na3d_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    // Try v2 primitive backward path first
    {
        const char* v2_env = std::getenv("NATTEN_NANOBIND_DISABLE_V2");
        bool v2_disabled = v2_env != nullptr && std::string(v2_env) == "1";
        if (!v2_disabled) {
            try {
                mx::array q_arr = nb::cast<mx::array>(q);
                mx::array k_arr = nb::cast<mx::array>(k);
                mx::array v_arr = nb::cast<mx::array>(v);
                mx::array go_arr = nb::cast<mx::array>(grad_out);
                int ks = scalar_or_index_int(kernel_size, 0);
                int sd = scalar_or_index_int(stride, 0);
                int sh = scalar_or_index_int(stride, 1);
                int sw = scalar_or_index_int(stride, 2);
                int dd = scalar_or_index_int(dilation, 0);
                int dh = scalar_or_index_int(dilation, 1);
                int dw = scalar_or_index_int(dilation, 2);
                bool cd = scalar_or_index_bool(is_causal, 0);
                bool ch = scalar_or_index_bool(is_causal, 1);
                bool cw = scalar_or_index_bool(is_causal, 2);
                int head_dim = q_arr.shape(5);
                bool eligible = (ks > 0) && (ks % 2 == 1) && (ks <= 7) &&
                    sd >= 1 && sh >= 1 && sw >= 1 &&
                    dd >= 1 && dh >= 1 && dw >= 1;
                if (eligible) {
                    float sc;
                    if (scale.is_none()) {
                        sc = std::pow(static_cast<float>(head_dim), -0.5f);
                    } else {
                        sc = nb::cast<float>(scale);
                    }
                    auto grads = natten_mlx::na3d_backward_v2(
                        q_arr, k_arr, v_arr, go_arr,
                        ks, sd, sh, sw, dd, dh, dw, cd, ch, cw, sc);
                    return nb::cast(nb::make_tuple(
                        nb::cast(grads[0]), nb::cast(grads[1]), nb::cast(grads[2])));
                }
            } catch (const std::exception& e) {
                std::cerr << "natten_mlx: v2 na3d_backward fell back to pure: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "natten_mlx: v2 na3d_backward fell back to pure (unknown error)" << std::endl;
            }
        }
    }
    return natten_mlx::nanobind_backend::call_pure(
        "na3d_backward", q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
}

}  // namespace natten_mlx::nanobind_composed

#include "na_composed.h"

#include <nanobind/stl/tuple.h>

#include "metal_runtime.h"
#include "na_fused_backward.h"
#include "na_fused_forward.h"
#include "na_split_backward.h"
#include "na_split_forward.h"
#include "py_dispatch.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

nb::object softmax_last_dim(const nb::object& x) {
    return natten_mlx::nanobind_backend::mx_module().attr("softmax")(x, "axis"_a = -1);
}

nb::object grad_logits_from_softmax(
    const nb::object& attn,
    const nb::object& grad_attn) {
    nb::object mx = natten_mlx::nanobind_backend::mx_module();
    nb::object prod = mx.attr("multiply")(grad_attn, attn);
    nb::object inner = mx.attr("sum")(prod, "axis"_a = -1, "keepdims"_a = true);
    nb::object centered = mx.attr("subtract")(grad_attn, inner);
    return mx.attr("multiply")(attn, centered);
}

nb::object backward_composed(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale,
    nb::object (*qk_forward)(
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&),
    nb::object (*av_forward)(
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&),
    nb::object (*qk_backward)(
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&),
    nb::object (*av_backward)(
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&)) {
    nb::object logits = qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale);
    nb::object attn = softmax_last_dim(logits);
    nb::tuple av_bw = nb::cast<nb::tuple>(
        av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal));
    nb::object grad_attn = av_bw[0];
    nb::object grad_v = av_bw[1];
    nb::object grad_logits = grad_logits_from_softmax(attn, grad_attn);

    nb::tuple qk_bw = nb::cast<nb::tuple>(
        qk_backward(q, k, grad_logits, kernel_size, stride, dilation, is_causal, scale));
    nb::object grad_q = qk_bw[0];
    nb::object grad_k = qk_bw[1];
    return nb::make_tuple(grad_q, grad_k, grad_v);
}

nb::object forward_split_composed(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale,
    nb::object (*qk_forward)(
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&),
    nb::object (*av_forward)(
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&)) {
    nb::object logits = qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale);
    nb::object attn = softmax_last_dim(logits);
    return av_forward(attn, v, kernel_size, stride, dilation, is_causal);
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
    try {
        nb::object out = natten_mlx::nanobind_fused_forward::na1d_forward(
            q, k, v, kernel_size, stride, dilation, is_causal, scale);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_forward", "fused");
        return out;
    } catch (...) {
    }
    try {
        nb::object out = forward_split_composed(
            q,
            k,
            v,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            natten_mlx::nanobind_split_forward::na1d_qk_forward,
            natten_mlx::nanobind_split_forward::na1d_av_forward);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_forward", "split");
        return out;
    } catch (...) {
    }
    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_forward", "pure");
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
    try {
        nb::object out = natten_mlx::nanobind_fused_forward::na2d_forward(
            q, k, v, kernel_size, stride, dilation, is_causal, scale);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_forward", "fused");
        return out;
    } catch (...) {
    }
    try {
        nb::object out = forward_split_composed(
            q,
            k,
            v,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            natten_mlx::nanobind_split_forward::na2d_qk_forward,
            natten_mlx::nanobind_split_forward::na2d_av_forward);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_forward", "split");
        return out;
    } catch (...) {
    }
    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_forward", "pure");
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
    try {
        nb::object out = natten_mlx::nanobind_fused_forward::na3d_forward(
            q, k, v, kernel_size, stride, dilation, is_causal, scale);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_forward", "fused");
        return out;
    } catch (...) {
    }
    try {
        nb::object out = forward_split_composed(
            q,
            k,
            v,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            natten_mlx::nanobind_split_forward::na3d_qk_forward,
            natten_mlx::nanobind_split_forward::na3d_av_forward);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_forward", "split");
        return out;
    } catch (...) {
    }
    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_forward", "pure");
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
    try {
        nb::object out = natten_mlx::nanobind_fused_backward::na1d_backward(
            q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_backward", "fused");
        return out;
    } catch (...) {
    }
    try {
        nb::object out = backward_composed(
            q,
            k,
            v,
            grad_out,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            natten_mlx::nanobind_split_forward::na1d_qk_forward,
            natten_mlx::nanobind_split_forward::na1d_av_forward,
            natten_mlx::nanobind_split_backward::na1d_qk_backward,
            natten_mlx::nanobind_split_backward::na1d_av_backward);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_backward", "split");
        return out;
    } catch (...) {
    }
    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_backward", "pure");
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
    try {
        nb::object out = natten_mlx::nanobind_fused_backward::na2d_backward(
            q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_backward", "fused");
        return out;
    } catch (...) {
    }
    try {
        nb::object out = backward_composed(
            q,
            k,
            v,
            grad_out,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            natten_mlx::nanobind_split_forward::na2d_qk_forward,
            natten_mlx::nanobind_split_forward::na2d_av_forward,
            natten_mlx::nanobind_split_backward::na2d_qk_backward,
            natten_mlx::nanobind_split_backward::na2d_av_backward);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_backward", "split");
        return out;
    } catch (...) {
    }
    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_backward", "pure");
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
    try {
        nb::object out = natten_mlx::nanobind_fused_backward::na3d_backward(
            q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_backward", "fused");
        return out;
    } catch (...) {
    }
    try {
        nb::object out = backward_composed(
            q,
            k,
            v,
            grad_out,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            natten_mlx::nanobind_split_forward::na3d_qk_forward,
            natten_mlx::nanobind_split_forward::na3d_av_forward,
            natten_mlx::nanobind_split_backward::na3d_qk_backward,
            natten_mlx::nanobind_split_backward::na3d_av_backward);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_backward", "split");
        return out;
    } catch (...) {
    }
    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_backward", "pure");
    return natten_mlx::nanobind_backend::call_pure(
        "na3d_backward", q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
}

}  // namespace natten_mlx::nanobind_composed

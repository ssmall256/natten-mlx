#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace natten_mlx::nanobind_split_backward {

nb::object na1d_qk_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_attn,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale);

nb::object na1d_av_backward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal);

nb::object na2d_qk_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_attn,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale);

nb::object na2d_av_backward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal);

nb::object na3d_qk_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_attn,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale);

nb::object na3d_av_backward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal);

}  // namespace natten_mlx::nanobind_split_backward


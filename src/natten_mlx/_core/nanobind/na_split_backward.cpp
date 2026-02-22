#include "na_split_backward.h"

#include "py_dispatch.h"

namespace natten_mlx::nanobind_split_backward {

nb::object na1d_qk_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_attn,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    return natten_mlx::nanobind_backend::call_pure(
        "na1d_qk_backward", q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale);
}

nb::object na1d_av_backward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
    return natten_mlx::nanobind_backend::call_pure(
        "na1d_av_backward", attn, v, grad_out, kernel_size, stride, dilation, is_causal);
}

nb::object na2d_qk_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_attn,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    return natten_mlx::nanobind_backend::call_pure(
        "na2d_qk_backward", q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale);
}

nb::object na2d_av_backward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
    return natten_mlx::nanobind_backend::call_pure(
        "na2d_av_backward", attn, v, grad_out, kernel_size, stride, dilation, is_causal);
}

nb::object na3d_qk_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_attn,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    return natten_mlx::nanobind_backend::call_pure(
        "na3d_qk_backward", q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale);
}

nb::object na3d_av_backward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
    return natten_mlx::nanobind_backend::call_pure(
        "na3d_av_backward", attn, v, grad_out, kernel_size, stride, dilation, is_causal);
}

}  // namespace natten_mlx::nanobind_split_backward

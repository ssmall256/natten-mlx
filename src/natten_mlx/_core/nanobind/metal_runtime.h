#pragma once

#include <string>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace natten_mlx::nanobind_metal_runtime {

bool supports_1d_fused(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation);
bool supports_2d_fused(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation);
bool supports_3d_fused(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation);

bool supports_1d_split(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal);
bool supports_2d_split(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal);
bool supports_3d_split(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal);

nb::object na1d_qk_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale);
nb::object na1d_av_forward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal);
nb::object na2d_qk_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale);
nb::object na2d_av_forward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal);
nb::object na3d_qk_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale);
nb::object na3d_av_forward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal);

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

nb::object na1d_fused_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale);
nb::object na2d_fused_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale);
nb::object na3d_fused_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale);

nb::object na1d_fused_backward_attn(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale);
nb::object na1d_fused_backward_qk(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_logits,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale);
nb::object na1d_fused_backward_v(
    const nb::object& attn,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal);

nb::object na2d_fused_backward_qk(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_logits,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale);
nb::object na2d_fused_backward_v(
    const nb::object& attn,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal);

nb::object na3d_fused_backward_qk(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_logits,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale);
nb::object na3d_fused_backward_v(
    const nb::object& attn,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal);

void debug_set_last_route(const std::string& op, const std::string& route);
std::string debug_get_last_route(const std::string& op);
void debug_clear_last_routes();
void debug_force_fused_failure(bool enabled);
void debug_force_split_failure(bool enabled);
bool debug_forced_fused_failure();
bool debug_forced_split_failure();

void debug_inc_python_bridge_calls();
int debug_get_python_bridge_calls();
void debug_clear_python_bridge_calls();

}  // namespace natten_mlx::nanobind_metal_runtime

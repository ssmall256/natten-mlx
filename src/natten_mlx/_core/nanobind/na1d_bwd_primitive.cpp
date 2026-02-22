#include "nanobind/na1d_bwd_primitive.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>

#include <dlfcn.h>

#include <mlx/allocator.h>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/stream.h>
#include <mlx/backend/metal/device.h>

namespace mx = mlx::core;

namespace natten_mlx {

namespace {

struct NA1DParamsV2 {
  int B;
  int L;
  int H;
  int D;
  int K;
  int S;
  int DIL;
  int CAUSAL;
  float SCALE;
};

std::mutex& v2_1d_bwd_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, MTL::ComputePipelineState*>& v2_1d_bwd_cache() {
  static std::unordered_map<std::string, MTL::ComputePipelineState*> c;
  return c;
}

std::string v2_1d_bwd_dir() {
  static std::string dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&v2_1d_bwd_dir), &info)) {
      return std::string(".");
    }
    std::string path(info.dli_fname);
    auto pos = path.find_last_of('/');
    return (pos == std::string::npos) ? std::string(".") : path.substr(0, pos);
  }();
  return dir;
}

MTL::ComputePipelineState* get_1d_bwd_kernel(const std::string& name, int kernel_size) {
  std::string cache_key = name + "_k" + std::to_string(kernel_size);
  {
    std::lock_guard<std::mutex> lock(v2_1d_bwd_mutex());
    auto it = v2_1d_bwd_cache().find(cache_key);
    if (it != v2_1d_bwd_cache().end()) return it->second;
  }
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* lib = dev.get_library("natten_nb", v2_1d_bwd_dir());
  if (!lib) throw std::runtime_error("Failed to load natten_nb metallib");
  mx::metal::MTLFCList fc = {
      {&kernel_size, MTL::DataType::DataTypeInt, 0}};
  auto* k = dev.get_kernel(name, lib, cache_key, fc);
  if (!k) throw std::runtime_error("Failed to resolve 1D bwd kernel: " + name);
  {
    std::lock_guard<std::mutex> lock(v2_1d_bwd_mutex());
    v2_1d_bwd_cache()[cache_key] = k;
  }
  return k;
}

std::string dtype_suffix(mx::Dtype dtype) {
  if (dtype == mx::float32) return "fp32";
  if (dtype == mx::float16) return "fp16";
  if (dtype == mx::bfloat16) return "bf16";
  return "fp32";
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

}  // namespace

// ---- NA1DBwdAttn ----

void NA1DBwdAttn::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& grad_out = inputs[3];

  int B = q.shape(0), L = q.shape(1);
  int H = q.shape(2), D = q.shape(3);
  int out_len = ceil_div(L, stride_);

  mx::Dtype input_dtype = q.dtype();
  bool vec4 = use_vec4_ && (D % 4 == 0);
  std::string suffix = dtype_suffix(input_dtype);
  std::string kname = vec4
      ? ("na1d_bwd_attn_v2_vec4_" + suffix)
      : ("na1d_bwd_attn_v2_" + suffix);

  NA1DParamsV2 params{B, L, H, D, kernel_size_, stride_, dilation_, causal_, scale_};

  // Both outputs are fp32: [B, out_len, H, K]
  size_t out_bytes = static_cast<size_t>(B) * out_len * H * kernel_size_ * sizeof(float);
  outputs[0].set_data(mx::allocator::malloc(out_bytes));
  outputs[1].set_data(mx::allocator::malloc(out_bytes));

  auto* kernel_fn = get_1d_bwd_kernel(kname, kernel_size_);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto& enc = dev.get_command_encoder(stream().index);

  enc.set_compute_pipeline_state(kernel_fn);
  enc.set_input_array(q, 0);
  enc.set_input_array(k, 1);
  enc.set_input_array(v, 2);
  enc.set_input_array(grad_out, 3);
  enc.set_output_array(outputs[0], 4);
  enc.set_output_array(outputs[1], 5);
  enc.set_bytes(params, 6);

  size_t gx = static_cast<size_t>(out_len);
  size_t gy = static_cast<size_t>(B * H);
  enc.dispatch_threads(
      MTL::Size(gx, gy, 1),
      MTL::Size(std::min<size_t>(256, gx), 1, 1));
}

// ---- NA1DBwdGradQ ----

void NA1DBwdGradQ::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& gl = inputs[0];
  auto& key = inputs[1];

  int B = key.shape(0), L = key.shape(1);
  int H = key.shape(2), D = key.shape(3);

  std::string suffix = dtype_suffix(out_dtype_);
  std::string kname = "na1d_bwd_grad_q_v2_" + suffix;

  NA1DParamsV2 params{B, L, H, D, kernel_size_, stride_, dilation_, causal_, scale_};

  size_t out_bytes = static_cast<size_t>(B) * L * H * D * mx::size_of(out_dtype_);
  outputs[0].set_data(mx::allocator::malloc(out_bytes));

  auto* kernel_fn = get_1d_bwd_kernel(kname, kernel_size_);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto& enc = dev.get_command_encoder(stream().index);

  enc.set_compute_pipeline_state(kernel_fn);
  enc.set_input_array(gl, 0);
  enc.set_input_array(key, 1);
  enc.set_output_array(outputs[0], 2);
  enc.set_bytes(params, 3);

  size_t gx = static_cast<size_t>(L);
  size_t gy = static_cast<size_t>(B * H);
  enc.dispatch_threads(
      MTL::Size(gx, gy, 1),
      MTL::Size(std::min<size_t>(256, gx), 1, 1));
}

// ---- NA1DBwdGradK ----

void NA1DBwdGradK::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& gl = inputs[0];
  auto& query = inputs[1];

  int B = query.shape(0), L = query.shape(1);
  int H = query.shape(2), D = query.shape(3);

  std::string suffix = dtype_suffix(out_dtype_);
  std::string kname = "na1d_bwd_grad_k_v2_" + suffix;

  NA1DParamsV2 params{B, L, H, D, kernel_size_, stride_, dilation_, causal_, scale_};

  size_t out_bytes = static_cast<size_t>(B) * L * H * D * mx::size_of(out_dtype_);
  outputs[0].set_data(mx::allocator::malloc(out_bytes));

  auto* kernel_fn = get_1d_bwd_kernel(kname, kernel_size_);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto& enc = dev.get_command_encoder(stream().index);

  enc.set_compute_pipeline_state(kernel_fn);
  enc.set_input_array(gl, 0);
  enc.set_input_array(query, 1);
  enc.set_output_array(outputs[0], 2);
  enc.set_bytes(params, 3);

  size_t gx = static_cast<size_t>(L);
  size_t gy = static_cast<size_t>(B * H);
  enc.dispatch_threads(
      MTL::Size(gx, gy, 1),
      MTL::Size(std::min<size_t>(256, gx), 1, 1));
}

// ---- NA1DBwdGradV ----

void NA1DBwdGradV::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& attn = inputs[0];
  auto& grad_out = inputs[1];

  int B = attn.shape(0);
  int H = attn.shape(2);
  int D = grad_out.shape(3);

  std::string suffix = dtype_suffix(out_dtype_);
  std::string kname = "na1d_bwd_grad_v_v2_" + suffix;

  NA1DParamsV2 params{B, IL_, H, D, kernel_size_, stride_, dilation_, causal_, 1.0f};

  size_t out_bytes = static_cast<size_t>(B) * IL_ * H * D * mx::size_of(out_dtype_);
  outputs[0].set_data(mx::allocator::malloc(out_bytes));

  auto* kernel_fn = get_1d_bwd_kernel(kname, kernel_size_);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto& enc = dev.get_command_encoder(stream().index);

  enc.set_compute_pipeline_state(kernel_fn);
  enc.set_input_array(attn, 0);
  enc.set_input_array(grad_out, 1);
  enc.set_output_array(outputs[0], 2);
  enc.set_bytes(params, 3);

  size_t gx = static_cast<size_t>(IL_);
  size_t gy = static_cast<size_t>(B * H);
  enc.dispatch_threads(
      MTL::Size(gx, gy, 1),
      MTL::Size(std::min<size_t>(256, gx), 1, 1));
}

// ---- Factory: na1d_backward_v2 ----

std::vector<mx::array> na1d_backward_v2(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& grad_out,
    int kernel_size,
    int stride, int dilation,
    bool causal,
    float scale,
    mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);

  int B = q.shape(0), L = q.shape(1);
  int H = q.shape(2), D = q.shape(3);
  int out_len = (L + stride - 1) / stride;
  int c = causal ? 1 : 0;
  bool use_vec4 = (D % 4 == 0);

  mx::Dtype orig_dtype = q.dtype();
  mx::Dtype compute_dtype = orig_dtype;
  if (compute_dtype != mx::float32 && compute_dtype != mx::float16 &&
      compute_dtype != mx::bfloat16) {
    compute_dtype = mx::float32;
  }

  auto qc = mx::contiguous(q, false, stream);
  auto kc = mx::contiguous(k, false, stream);
  auto vc = mx::contiguous(v, false, stream);
  auto goc = mx::contiguous(grad_out, false, stream);

  if (qc.dtype() != compute_dtype) {
    qc = mx::astype(qc, compute_dtype, stream);
    kc = mx::astype(kc, compute_dtype, stream);
    vc = mx::astype(vc, compute_dtype, stream);
    goc = mx::astype(goc, compute_dtype, stream);
  }

  // Stage 1: fused attn recompute + grad_logits (both fp32)
  auto attn_prim = std::make_shared<NA1DBwdAttn>(
      stream, kernel_size, stride, dilation, c, scale, use_vec4);

  mx::Shape attn_shape = {B, out_len, H, kernel_size};
  auto outputs_attn = mx::array::make_arrays(
      {attn_shape, attn_shape},
      {mx::float32, mx::float32},
      std::move(attn_prim),
      {qc, kc, vc, goc});
  auto attn = outputs_attn[0];
  auto grad_logits = outputs_attn[1];

  // Stage 3: grad_q and grad_k
  auto kf = (kc.dtype() == mx::float32) ? kc : mx::astype(kc, mx::float32, stream);
  auto qf = (qc.dtype() == mx::float32) ? qc : mx::astype(qc, mx::float32, stream);

  auto grad_q_prim = std::make_shared<NA1DBwdGradQ>(
      stream, kernel_size, stride, dilation, c, scale, compute_dtype);
  mx::Shape qk_shape = {B, L, H, D};
  auto grad_q = mx::array(qk_shape, compute_dtype, std::move(grad_q_prim), {grad_logits, kf});

  auto grad_k_prim = std::make_shared<NA1DBwdGradK>(
      stream, kernel_size, stride, dilation, c, scale, compute_dtype);
  auto grad_k = mx::array(qk_shape, compute_dtype, std::move(grad_k_prim), {grad_logits, qf});

  // Stage 4: grad_v
  auto grad_v_prim = std::make_shared<NA1DBwdGradV>(
      stream, kernel_size, stride, dilation, c, L, compute_dtype);
  auto grad_v = mx::array(qk_shape, compute_dtype, std::move(grad_v_prim), {attn, goc});

  if (compute_dtype != orig_dtype) {
    grad_q = mx::astype(grad_q, orig_dtype, stream);
    grad_k = mx::astype(grad_k, orig_dtype, stream);
    grad_v = mx::astype(grad_v, orig_dtype, stream);
  }

  return {grad_q, grad_k, grad_v};
}

}  // namespace natten_mlx

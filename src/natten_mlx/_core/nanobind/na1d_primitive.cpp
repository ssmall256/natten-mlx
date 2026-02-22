#include "nanobind/na1d_primitive.h"

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

std::mutex& v2_1d_kernel_cache_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, MTL::ComputePipelineState*>& v2_1d_kernel_cache() {
  static std::unordered_map<std::string, MTL::ComputePipelineState*> cache;
  return cache;
}

std::string v2_1d_binary_dir() {
  static std::string dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&v2_1d_binary_dir), &info)) {
      return std::string(".");
    }
    std::string path(info.dli_fname);
    auto pos = path.find_last_of('/');
    return (pos == std::string::npos) ? std::string(".") : path.substr(0, pos);
  }();
  return dir;
}

MTL::ComputePipelineState* get_v2_1d_kernel(const std::string& name) {
  {
    std::lock_guard<std::mutex> lock(v2_1d_kernel_cache_mutex());
    auto it = v2_1d_kernel_cache().find(name);
    if (it != v2_1d_kernel_cache().end()) {
      return it->second;
    }
  }
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* lib = dev.get_library("natten_nb", v2_1d_binary_dir());
  if (!lib) {
    throw std::runtime_error("Failed to load natten_nb metallib for v2 1D kernels");
  }
  auto* kernel = dev.get_kernel(name, lib);
  if (!kernel) {
    throw std::runtime_error("Failed to resolve v2 1D kernel: " + name);
  }
  {
    std::lock_guard<std::mutex> lock(v2_1d_kernel_cache_mutex());
    v2_1d_kernel_cache()[name] = kernel;
  }
  return kernel;
}

std::string dtype_suffix(mx::Dtype dtype) {
  if (dtype == mx::float32) return "fp32";
  if (dtype == mx::float16) return "fp16";
  if (dtype == mx::bfloat16) return "bf16";
  return "fp32";
}

}  // namespace

void NA1DFusedForward::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];

  int B = q.shape(0);
  int L = q.shape(1);
  int H = q.shape(2);
  int D = q.shape(3);
  int out_len = (L + stride_ - 1) / stride_;

  mx::Dtype compute_dtype = q.dtype();
  bool vec4 = use_vec4_ && (D % 4 == 0);
  std::string suffix = dtype_suffix(compute_dtype);
  std::string kname = vec4
      ? ("na1d_fused_v2_stored_vec4_" + suffix)
      : ("na1d_fused_v2_stored_" + suffix);

  NA1DParamsV2 params{B, L, H, D, kernel_size_, stride_, dilation_, causal_, scale_};

  size_t out_bytes = static_cast<size_t>(B) * out_len * H * D *
      mx::size_of(compute_dtype);
  outputs[0].set_data(mx::allocator::malloc(out_bytes));

  auto* kernel_fn = get_v2_1d_kernel(kname);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto& enc = dev.get_command_encoder(stream().index);

  enc.set_compute_pipeline_state(kernel_fn);
  enc.set_input_array(q, 0);
  enc.set_input_array(k, 1);
  enc.set_input_array(v, 2);
  enc.set_output_array(outputs[0], 3);
  enc.set_bytes(params, 4);

  // 2D grid: (out_len, B*H)
  size_t gx = static_cast<size_t>(out_len);
  size_t gy = static_cast<size_t>(B * H);
  size_t tgx = std::min<size_t>(256, gx);
  size_t tgy = 1;

  enc.dispatch_threads(
      MTL::Size(gx, gy, 1),
      MTL::Size(tgx, tgy, 1));
}

mx::array na1d_fused_forward_v2(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    int kernel_size,
    int stride,
    int dilation,
    bool causal,
    float scale,
    mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);

  int D = q.shape(q.ndim() - 1);
  bool use_vec4 = (D % 4 == 0);

  int B = q.shape(0);
  int L = q.shape(1);
  int H = q.shape(2);
  int out_len = (L + stride - 1) / stride;

  mx::Dtype compute_dtype = q.dtype();
  if (compute_dtype != mx::float32 && compute_dtype != mx::float16 &&
      compute_dtype != mx::bfloat16) {
    compute_dtype = mx::float32;
  }

  auto qc = mx::contiguous(q, false, stream);
  auto kc = mx::contiguous(k, false, stream);
  auto vc = mx::contiguous(v, false, stream);

  if (qc.dtype() != compute_dtype) {
    qc = mx::astype(qc, compute_dtype, stream);
    kc = mx::astype(kc, compute_dtype, stream);
    vc = mx::astype(vc, compute_dtype, stream);
  }

  auto prim = std::make_shared<NA1DFusedForward>(
      stream, kernel_size, stride, dilation,
      causal ? 1 : 0, scale, use_vec4);

  mx::Shape out_shape = {B, out_len, H, D};
  auto out = mx::array(out_shape, compute_dtype, std::move(prim), {qc, kc, vc});

  if (out.dtype() != q.dtype()) {
    out = mx::astype(out, q.dtype(), stream);
  }

  return out;
}

}  // namespace natten_mlx

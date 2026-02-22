#include "metal_runtime.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/stream.h>
#include <mlx/transforms.h>
#include <mlx/backend/metal/device.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace mx = mlx::core;
using namespace nb::literals;
namespace {

struct NA1DParams {
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

struct NA2DParams {
  int B;
  int IH;
  int IW;
  int H;
  int D;
  int K;
  int SH;
  int SW;
  int DH;
  int DW;
  int CH;
  int CW;
  float SCALE;
};

struct NA3DParams {
  int B;
  int ID;
  int IH;
  int IW;
  int H;
  int D;
  int K;
  int SD;
  int SH;
  int SW;
  int DD;
  int DH;
  int DW;
  int CD;
  int CH;
  int CW;
  float SCALE;
};

struct Clear1Params {
  uint32_t N;
};

struct Clear2Params {
  uint32_t N0;
  uint32_t N1;
};

struct Clear3Params {
  uint32_t N0;
  uint32_t N1;
  uint32_t N2;
};

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

int first_kernel_size(const nb::object& kernel_size) {
  return scalar_or_index_int(kernel_size, 0);
}

bool valid_kernel(int k) {
  return k > 0 && (k % 2 == 1);
}

bool valid_stride_1d(const nb::object& stride) {
  return scalar_or_index_int(stride, 0) >= 1;
}

bool valid_dilation_1d(const nb::object& dilation) {
  return scalar_or_index_int(dilation, 0) >= 1;
}

bool valid_stride_2d(const nb::object& stride) {
  return scalar_or_index_int(stride, 0) >= 1 && scalar_or_index_int(stride, 1) >= 1;
}

bool valid_dilation_2d(const nb::object& dilation) {
  return scalar_or_index_int(dilation, 0) >= 1 && scalar_or_index_int(dilation, 1) >= 1;
}

bool valid_stride_3d(const nb::object& stride) {
  return scalar_or_index_int(stride, 0) >= 1 && scalar_or_index_int(stride, 1) >= 1 &&
      scalar_or_index_int(stride, 2) >= 1;
}

bool valid_dilation_3d(const nb::object& dilation) {
  return scalar_or_index_int(dilation, 0) >= 1 && scalar_or_index_int(dilation, 1) >= 1 &&
      scalar_or_index_int(dilation, 2) >= 1;
}

bool square_kernel_2d(const nb::object& kernel_size) {
  return scalar_or_index_int(kernel_size, 0) == scalar_or_index_int(kernel_size, 1);
}

bool cubic_kernel_3d(const nb::object& kernel_size) {
  int k0 = scalar_or_index_int(kernel_size, 0);
  return k0 == scalar_or_index_int(kernel_size, 1) && k0 == scalar_or_index_int(kernel_size, 2);
}

int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

mx::Shape to_shape(const std::vector<int>& dims) {
  mx::Shape s;
  s.reserve(dims.size());
  for (int d : dims) {
    s.push_back(static_cast<mx::ShapeElem>(d));
  }
  return s;
}

size_t numel(const mx::Shape& shape) {
  size_t n = 1;
  for (auto d : shape) {
    n *= static_cast<size_t>(std::max<int>(d, 1));
  }
  return n;
}

mx::array to_float32(const mx::array& x) {
  if (x.dtype() == mx::float32) {
    return x;
  }
  return mx::astype(x, mx::float32);
}

mx::array cast_to_dtype(const mx::array& x, mx::Dtype dtype) {
  if (x.dtype() == dtype) {
    return x;
  }
  return mx::astype(x, dtype);
}

mx::array as_array(const nb::object& obj) {
  return nb::cast<mx::array>(obj);
}

float resolve_scale(const nb::object& scale, int head_dim) {
  if (scale.is_none()) {
    return std::pow(static_cast<float>(head_dim), -0.5f);
  }
  return nb::cast<float>(scale);
}

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to resolve current binary path");
    }
    std::string path(info.dli_fname);
    auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
      return std::string(".");
    }
    return path.substr(0, pos);
  }();
  return binary_dir;
}

std::mutex& route_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, std::string>& route_map() {
  static std::unordered_map<std::string, std::string> m;
  return m;
}

std::unordered_map<std::string, std::string>& kernel_map() {
  static std::unordered_map<std::string, std::string> m;
  return m;
}

bool& force_fused_failure_flag() {
  static bool v = false;
  return v;
}

bool& force_split_failure_flag() {
  static bool v = false;
  return v;
}

int& python_bridge_calls() {
  static int v = 0;
  return v;
}

std::mutex& kernel_cache_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, MTL::ComputePipelineState*>& kernel_cache() {
  static std::unordered_map<std::string, MTL::ComputePipelineState*> cache;
  return cache;
}

std::mutex& tuning_mutex() {
  static std::mutex m;
  return m;
}

bool& tuning_loaded() {
  static bool loaded = false;
  return loaded;
}

nb::object& threadgroup_lookup_fn() {
  static nb::object fn;
  return fn;
}

nb::object& softmax_strategy_fn() {
  static nb::object fn;
  return fn;
}

nb::object& bwd_mode_lookup_fn() {
  static nb::object fn;
  return fn;
}

nb::object& bwd_threadgroup_lookup_fn() {
  static nb::object fn;
  return fn;
}

std::mutex& launch_metrics_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, double>& launch_metrics_ms_total() {
  static std::unordered_map<std::string, double> m;
  return m;
}

std::unordered_map<std::string, int>& launch_metrics_count() {
  static std::unordered_map<std::string, int> m;
  return m;
}

struct LaunchConfig {
  MTL::Size grid;
  MTL::Size threadgroup;
};

struct VariantConfig {
  std::string dtype_tag;
  bool vec4;
};

struct SplitForwardVariant {
  std::string dtype_tag;
  bool native_lowp;
};

struct TuningQueryKey {
  std::string op;
  std::string dtype_tag;
  std::string gpu_family_override;
  int tokens;
  int head_dim;
  int kernel_size;
  int causal_rank;
  bool stride_unit;

  bool operator==(const TuningQueryKey& other) const {
    return op == other.op && dtype_tag == other.dtype_tag &&
        gpu_family_override == other.gpu_family_override && tokens == other.tokens &&
        head_dim == other.head_dim && kernel_size == other.kernel_size &&
        causal_rank == other.causal_rank && stride_unit == other.stride_unit;
  }
};

struct TuningQueryKeyHash {
  size_t operator()(const TuningQueryKey& k) const {
    size_t h = std::hash<std::string>{}(k.op);
    auto mix = [&](size_t v) { h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2); };
    mix(std::hash<std::string>{}(k.dtype_tag));
    mix(std::hash<std::string>{}(k.gpu_family_override));
    mix(std::hash<int>{}(k.tokens));
    mix(std::hash<int>{}(k.head_dim));
    mix(std::hash<int>{}(k.kernel_size));
    mix(std::hash<int>{}(k.causal_rank));
    mix(std::hash<bool>{}(k.stride_unit));
    return h;
  }
};

std::mutex& tuning_cache_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<
    TuningQueryKey,
    std::optional<std::tuple<size_t, size_t, size_t>>,
    TuningQueryKeyHash>&
forward_threadgroup_tuning_cache() {
  static std::unordered_map<
      TuningQueryKey,
      std::optional<std::tuple<size_t, size_t, size_t>>,
      TuningQueryKeyHash>
      cache;
  return cache;
}

std::unordered_map<
    TuningQueryKey,
    std::optional<std::tuple<size_t, size_t, size_t>>,
    TuningQueryKeyHash>&
backward_threadgroup_tuning_cache() {
  static std::unordered_map<
      TuningQueryKey,
      std::optional<std::tuple<size_t, size_t, size_t>>,
      TuningQueryKeyHash>
      cache;
  return cache;
}

std::unordered_map<TuningQueryKey, std::string, TuningQueryKeyHash>& softmax_strategy_cache() {
  static std::unordered_map<TuningQueryKey, std::string, TuningQueryKeyHash> cache;
  return cache;
}

std::unordered_map<TuningQueryKey, std::string, TuningQueryKeyHash>& backward_mode_cache() {
  static std::unordered_map<TuningQueryKey, std::string, TuningQueryKeyHash> cache;
  return cache;
}

std::string gpu_family_override_key() {
  const char* v = std::getenv("NATTEN_MLX_GPU_FAMILY");
  return v == nullptr ? std::string() : std::string(v);
}

void ensure_tuning_loaded();

inline int causal_rank_1d(int c0) {
  return c0 ? 1 : 0;
}

inline int causal_rank_2d(int c0, int c1) {
  return (c0 ? 1 : 0) + (c1 ? 1 : 0);
}

inline int causal_rank_3d(int c0, int c1, int c2) {
  return (c0 ? 1 : 0) + (c1 ? 1 : 0) + (c2 ? 1 : 0);
}

inline bool launch_metrics_enabled() {
  const char* value = std::getenv("NATTEN_NANOBIND_LAUNCH_METRICS");
  return value != nullptr && std::string(value) == "1";
}

inline bool sync_launch_outputs_enabled() {
  const char* value = std::getenv("NATTEN_NANOBIND_SYNC_LAUNCH");
  if (value == nullptr) {
    return false;
  }
  std::string mode(value);
  return mode == "1" || mode == "true" || mode == "on";
}

inline std::string parse_bwd_mode_env() {
  const char* value = std::getenv("NATTEN_NANOBIND_BWD_MODE");
  if (value == nullptr) {
    return "auto";
  }
  std::string mode(value);
  if (mode == "atomic" || mode == "simple" || mode == "tiled" || mode == "auto") {
    return mode;
  }
  return "auto";
}

inline std::string parse_qkv_stage_mode_env() {
  const char* value = std::getenv("NATTEN_NANOBIND_QKV_STAGE_MODE");
  if (value == nullptr) {
    return "auto";
  }
  std::string mode(value);
  if (mode == "atomic" || mode == "tiled" || mode == "auto") {
    return mode;
  }
  return "auto";
}

inline std::string parse_qkv_tiled_layout_env() {
  const char* value = std::getenv("NATTEN_NANOBIND_QKV_TILED_LAYOUT");
  if (value == nullptr) {
    return "auto";
  }
  std::string mode(value);
  if (mode == "auto" || mode == "split" || mode == "single") {
    return mode;
  }
  return "auto";
}

inline std::optional<size_t> parse_positive_tg_env(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return std::nullopt;
  }
  try {
    int parsed = std::stoi(std::string(value));
    if (parsed > 0) {
      return static_cast<size_t>(parsed);
    }
  } catch (...) {
  }
  return std::nullopt;
}

inline std::string choose_na3d_qkv_tiled_kernel(int kernel_size, bool use_vec4) {
  if (kernel_size == 3) {
    return use_vec4 ? "na3d_fused_bwd_qkv_softmax_tiled_k3_vec4_fp32"
                    : "na3d_fused_bwd_qkv_softmax_tiled_k3_fp32";
  }
  if (kernel_size == 5) {
    return use_vec4 ? "na3d_fused_bwd_qkv_softmax_tiled_k5_vec4_fp32"
                    : "na3d_fused_bwd_qkv_softmax_tiled_k5_fp32";
  }
  return use_vec4 ? "na3d_fused_bwd_qkv_softmax_tiled_vec4_fp32"
                  : "na3d_fused_bwd_qkv_softmax_tiled_fp32";
}

inline std::string choose_na2d_kv_tiled_kernel(int kernel_size, bool use_vec4) {
  if (kernel_size == 3) {
    return use_vec4 ? "na2d_fused_bwd_kv_softmax_tiled_k3_vec4_fp32"
                    : "na2d_fused_bwd_kv_softmax_tiled_k3_fp32";
  }
  if (kernel_size == 5) {
    return use_vec4 ? "na2d_fused_bwd_kv_softmax_tiled_k5_vec4_fp32"
                    : "na2d_fused_bwd_kv_softmax_tiled_k5_fp32";
  }
  if (kernel_size == 7) {
    return use_vec4 ? "na2d_fused_bwd_kv_softmax_tiled_k7_vec4_fp32"
                    : "na2d_fused_bwd_kv_softmax_tiled_k7_fp32";
  }
  return use_vec4 ? "na2d_fused_bwd_kv_softmax_tiled_vec4_fp32"
                  : "na2d_fused_bwd_kv_softmax_tiled_fp32";
}

inline std::string choose_na2d_q_softmax_kernel(bool use_vec4, bool unit_stride_noncausal = false) {
  if (unit_stride_noncausal) {
    return use_vec4 ? "na2d_fused_bwd_q_softmax_u1d1_nc_vec4_fp32"
                    : "na2d_fused_bwd_q_softmax_u1d1_nc_fp32";
  }
  return use_vec4 ? "na2d_fused_bwd_q_softmax_vec4_fp32" : "na2d_fused_bwd_q_softmax_fp32";
}

inline std::string choose_na3d_q_softmax_kernel(int kernel_size, bool use_vec4) {
  if (!use_vec4) {
    if (kernel_size == 3) {
      return "na3d_fused_bwd_q_softmax_k3_fp32";
    }
    if (kernel_size == 5) {
      return "na3d_fused_bwd_q_softmax_k5_fp32";
    }
    return "na3d_fused_bwd_q_softmax_fp32";
  }
  if (kernel_size == 3) {
    return "na3d_fused_bwd_q_softmax_k3_vec4_fp32";
  }
  if (kernel_size == 5) {
    return "na3d_fused_bwd_q_softmax_k5_vec4_fp32";
  }
  return "na3d_fused_bwd_q_softmax_vec4_fp32";
}

inline std::string choose_na3d_fused_k_direct_softmax_kernel(int kernel_size, bool use_vec4) {
  if (kernel_size == 3) {
    return use_vec4 ? "na3d_fused_bwd_k_direct_softmax_u1d1_nc_k3_vec4_fp32"
                    : "na3d_fused_bwd_k_direct_softmax_u1d1_nc_k3_fp32";
  }
  return use_vec4 ? "na3d_fused_bwd_k_direct_softmax_u1d1_nc_vec4_fp32"
                  : "na3d_fused_bwd_k_direct_softmax_u1d1_nc_fp32";
}

inline std::string choose_na3d_kv_tiled_kernel(int kernel_size, bool use_vec4) {
  if (kernel_size == 3) {
    return use_vec4 ? "na3d_fused_bwd_kv_softmax_tiled_k3_vec4_fp32"
                    : "na3d_fused_bwd_kv_softmax_tiled_k3_fp32";
  }
  if (kernel_size == 5) {
    return use_vec4 ? "na3d_fused_bwd_kv_softmax_tiled_k5_vec4_fp32"
                    : "na3d_fused_bwd_kv_softmax_tiled_k5_fp32";
  }
  return use_vec4 ? "na3d_fused_bwd_kv_softmax_tiled_vec4_fp32"
                  : "na3d_fused_bwd_kv_softmax_tiled_fp32";
}

inline std::string choose_na2d_qk_bwd_k_direct_kernel(int kernel_size, bool use_vec4) {
  if (kernel_size == 7) {
    return use_vec4 ? "na2d_qk_bwd_k_direct_u1d1_nc_k7_vec4_fp32"
                    : "na2d_qk_bwd_k_direct_u1d1_nc_k7_fp32";
  }
  return use_vec4 ? "na2d_qk_bwd_k_direct_u1d1_nc_vec4_fp32"
                  : "na2d_qk_bwd_k_direct_u1d1_nc_fp32";
}

inline std::string choose_na3d_qk_bwd_k_direct_kernel(int kernel_size, bool use_vec4) {
  if (kernel_size == 3) {
    return use_vec4 ? "na3d_qk_bwd_k_direct_u1d1_nc_k3_vec4_fp32"
                    : "na3d_qk_bwd_k_direct_u1d1_nc_k3_fp32";
  }
  return use_vec4 ? "na3d_qk_bwd_k_direct_u1d1_nc_vec4_fp32"
                  : "na3d_qk_bwd_k_direct_u1d1_nc_fp32";
}

inline bool fused_backward_direct_enabled(int ndim) {
  const char* dim_env = nullptr;
  if (ndim == 2) {
    dim_env = std::getenv("NATTEN_NANOBIND_FUSED_BWD_DIRECT_2D");
  } else if (ndim == 3) {
    dim_env = std::getenv("NATTEN_NANOBIND_FUSED_BWD_DIRECT_3D");
  }
  if (dim_env != nullptr) {
    std::string mode(dim_env);
    return mode == "1" || mode == "true" || mode == "on";
  }
  const char* global_env = std::getenv("NATTEN_NANOBIND_FUSED_BWD_DIRECT");
  if (global_env == nullptr) {
    return true;
  }
  std::string mode(global_env);
  return mode == "1" || mode == "true" || mode == "on";
}

inline bool direct_nonatomic_enabled(
    const std::string& op,
    const std::string& dtype_tag,
    int tokens,
    int head_dim,
    int kernel_size,
    int causal_rank,
    bool stride_unit) {
  std::string env_mode = parse_bwd_mode_env();
  if (env_mode == "atomic" || env_mode == "simple") {
    return false;
  }
  if (env_mode == "tiled") {
    return true;
  }

  const char* value = std::getenv("NATTEN_NANOBIND_ENABLE_DIRECT_NONATOMIC");
  if (value != nullptr && std::string(value) == "1") {
    return true;
  }

  ensure_tuning_loaded();
  nb::object fn = bwd_mode_lookup_fn();
  if (fn.is_none()) {
    return false;
  }
  TuningQueryKey key{
      op,
      dtype_tag,
      gpu_family_override_key(),
      tokens,
      head_dim,
      kernel_size,
      causal_rank,
      stride_unit};
  {
    std::lock_guard<std::mutex> lock(tuning_cache_mutex());
    auto it = backward_mode_cache().find(key);
    if (it != backward_mode_cache().end()) {
      return it->second == "tiled";
    }
  }
  std::string mode = "atomic";
  try {
    mode = nb::cast<std::string>(fn(
        "op"_a = op,
        "dtype_tag"_a = dtype_tag,
        "tokens"_a = tokens,
        "head_dim"_a = head_dim,
        "kernel_size"_a = kernel_size,
        "causal_rank"_a = causal_rank,
        "stride_unit"_a = stride_unit));
  } catch (...) {
    mode = "atomic";
  }
  {
    std::lock_guard<std::mutex> lock(tuning_cache_mutex());
    backward_mode_cache()[key] = mode;
  }
  return mode == "tiled";
}

inline bool can_use_direct_nonatomic_1d(
    const std::string& op,
    const std::string& dtype_tag,
    int head_dim,
    int l,
    int k,
    int s,
    int dil,
    int c) {
  int tokens = l;
  bool stride_unit = (s == 1);
  int causal_rank = causal_rank_1d(c);
  return direct_nonatomic_enabled(op, dtype_tag, tokens, head_dim, k, causal_rank, stride_unit) &&
      s == 1 && dil == 1 && c == 0 && l >= k && k <= 31;
}

inline bool can_use_direct_causal_1d(
    const std::string& op,
    const std::string& dtype_tag,
    int head_dim,
    int l,
    int k,
    int s,
    int dil,
    int c) {
  int tokens = l;
  bool stride_unit = (s == 1);
  int causal_rank = causal_rank_1d(c);
  return direct_nonatomic_enabled(op, dtype_tag, tokens, head_dim, k, causal_rank, stride_unit) &&
      s == 1 && dil >= 1 && c == 1 && l >= 1 && k <= 31;
}

inline std::string choose_na1d_qk_bwd_k_direct_kernel(int causal, bool use_vec4) {
  if (causal != 0) {
    return use_vec4 ? "na1d_qk_bwd_k_direct_s1_causal_vec4_fp32"
                    : "na1d_qk_bwd_k_direct_s1_causal_fp32";
  }
  return use_vec4 ? "na1d_qk_bwd_k_direct_u1d1_nc_vec4_fp32"
                  : "na1d_qk_bwd_k_direct_u1d1_nc_fp32";
}

inline std::string choose_na1d_av_bwd_v_direct_kernel(int causal, bool use_vec4) {
  if (causal != 0) {
    return use_vec4 ? "na1d_av_bwd_v_direct_s1_causal_vec4_fp32"
                    : "na1d_av_bwd_v_direct_s1_causal_fp32";
  }
  return use_vec4 ? "na1d_av_bwd_v_direct_u1d1_nc_vec4_fp32"
                  : "na1d_av_bwd_v_direct_u1d1_nc_fp32";
}

inline bool can_use_direct_nonatomic_2d(
    const std::string& op,
    const std::string& dtype_tag,
    int head_dim,
    int ih,
    int iw,
    int k,
    int sh,
    int sw,
    int dh,
    int dw,
    int ch,
    int cw) {
  // Direct kernels use small fixed candidate buffers; keep this path to the
  // common small-k regime to avoid truncating candidates for very large K.
  int tokens = ih * iw;
  bool stride_unit = (sh == 1 && sw == 1);
  int causal_rank = causal_rank_2d(ch, cw);
  return direct_nonatomic_enabled(op, dtype_tag, tokens, head_dim, k, causal_rank, stride_unit) &&
      sh == 1 && sw == 1 && dh == 1 && dw == 1 && ch == 0 && cw == 0 && ih >= k &&
      iw >= k && k <= 31;
}

inline bool can_use_direct_nonatomic_3d(
    const std::string& op,
    const std::string& dtype_tag,
    int head_dim,
    int id,
    int ih,
    int iw,
    int k,
    int sd,
    int sh,
    int sw,
    int dd,
    int dh,
    int dw,
    int cd,
    int ch,
    int cw) {
  int tokens = id * ih * iw;
  bool stride_unit = (sd == 1 && sh == 1 && sw == 1);
  int causal_rank = causal_rank_3d(cd, ch, cw);
  return direct_nonatomic_enabled(op, dtype_tag, tokens, head_dim, k, causal_rank, stride_unit) &&
      sd == 1 && sh == 1 && sw == 1 && dd == 1 && dh == 1 && dw == 1 && cd == 0 &&
      ch == 0 && cw == 0 && id >= k && ih >= k && iw >= k && k <= 31;
}

inline bool force_direct_nonatomic_3d_hotshape(
    int head_dim,
    int id,
    int ih,
    int iw,
    int k,
    int sd,
    int sh,
    int sw,
    int dd,
    int dh,
    int dw,
    int cd,
    int ch,
    int cw) {
  std::string env_mode = parse_bwd_mode_env();
  if (env_mode == "atomic" || env_mode == "simple") {
    return false;
  }
  if (const char* qkv_mode = std::getenv("NATTEN_NANOBIND_QKV_STAGE_MODE")) {
    std::string mode(qkv_mode);
    if (mode == "tiled") {
      return false;
    }
  }
  bool noncausal = (cd == 0 && ch == 0 && cw == 0);
  bool unit_step = (sd == 1 && sh == 1 && sw == 1 && dd == 1 && dh == 1 && dw == 1);
  int tokens = id * ih * iw;
  return noncausal && unit_step && (k == 3 || k == 5) && tokens >= 768 && head_dim <= 64;
}

inline bool prefer_tiled_qkv_stage_2d(
    int head_dim,
    int ih,
    int iw,
    int k,
    int sh,
    int sw,
    int dh,
    int dw,
    int ch,
    int cw) {
  bool tiled_eligible = sh == 1 && sw == 1 && dh == 1 && dw == 1 && ch == 0 && cw == 0 &&
      ih >= k && iw >= k && k <= 31;
  if (!tiled_eligible) {
    return false;
  }
  std::string qkv_mode = parse_qkv_stage_mode_env();
  if (qkv_mode == "atomic") {
    return false;
  }
  if (qkv_mode == "tiled") {
    return true;
  }
  int tokens = ih * iw;
  return tokens >= 384 || (tokens >= 256 && head_dim >= 32);
}

inline bool prefer_tiled_qkv_stage_3d(
    int head_dim,
    int id,
    int ih,
    int iw,
    int k,
    int sd,
    int sh,
    int sw,
    int dd,
    int dh,
    int dw,
    int cd,
    int ch,
    int cw) {
  bool tiled_eligible = sd == 1 && sh == 1 && sw == 1 && dd == 1 && dh == 1 && dw == 1 &&
      cd == 0 && ch == 0 && cw == 0 && id >= k && ih >= k && iw >= k && k <= 31;
  if (!tiled_eligible) {
    return false;
  }
  std::string qkv_mode = parse_qkv_stage_mode_env();
  if (qkv_mode == "atomic") {
    return false;
  }
  if (qkv_mode == "tiled") {
    return true;
  }
  int tokens = id * ih * iw;
  return tokens >= 640 || (tokens >= 384 && head_dim >= 32);
}

inline bool direct_vec4_eligible(int head_dim) {
  return head_dim >= 16 && (head_dim % 4 == 0);
}

void ensure_tuning_loaded() {
  std::lock_guard<std::mutex> lock(tuning_mutex());
  if (tuning_loaded()) {
    return;
  }
  try {
    auto mod = nb::module_::import_("natten_mlx._core._nanobind_tuning");
    threadgroup_lookup_fn() = mod.attr("lookup_threadgroup");
    softmax_strategy_fn() = mod.attr("choose_softmax_strategy");
  } catch (...) {
    threadgroup_lookup_fn() = nb::none();
    softmax_strategy_fn() = nb::none();
  }
  try {
    auto mod = nb::module_::import_("natten_mlx._core._nanobind_bwd_tuning");
    bwd_mode_lookup_fn() = mod.attr("lookup_backward_mode");
    bwd_threadgroup_lookup_fn() = mod.attr("lookup_backward_threadgroup");
  } catch (...) {
    bwd_mode_lookup_fn() = nb::none();
    bwd_threadgroup_lookup_fn() = nb::none();
  }
  tuning_loaded() = true;
}

std::optional<std::tuple<size_t, size_t, size_t>> lookup_backward_threadgroup_from_tuning(
    const std::string& op,
    const std::string& dtype_tag,
    int tokens,
    int head_dim,
    int kernel_size,
    int causal_rank,
    bool stride_unit) {
  if (const char* v = std::getenv("NATTEN_NANOBIND_BWD_TG_OVERRIDE")) {
    std::string s(v);
    size_t c1 = s.find(',');
    size_t c2 = (c1 == std::string::npos) ? std::string::npos : s.find(',', c1 + 1);
    if (c1 != std::string::npos && c2 != std::string::npos) {
      try {
        int x = std::stoi(s.substr(0, c1));
        int y = std::stoi(s.substr(c1 + 1, c2 - c1 - 1));
        int z = std::stoi(s.substr(c2 + 1));
        return std::make_tuple(
            static_cast<size_t>(std::max(1, x)),
            static_cast<size_t>(std::max(1, y)),
            static_cast<size_t>(std::max(1, z)));
      } catch (...) {
      }
    }
  }

  TuningQueryKey key{
      op,
      dtype_tag,
      gpu_family_override_key(),
      tokens,
      head_dim,
      kernel_size,
      causal_rank,
      stride_unit};
  {
    std::lock_guard<std::mutex> lock(tuning_cache_mutex());
    auto it = backward_threadgroup_tuning_cache().find(key);
    if (it != backward_threadgroup_tuning_cache().end()) {
      return it->second;
    }
  }

  ensure_tuning_loaded();
  nb::object fn = bwd_threadgroup_lookup_fn();
  if (fn.is_none()) {
    std::lock_guard<std::mutex> lock(tuning_cache_mutex());
    backward_threadgroup_tuning_cache()[key] = std::nullopt;
    return std::nullopt;
  }
  std::optional<std::tuple<size_t, size_t, size_t>> value = std::nullopt;
  try {
    nb::object result = fn(
        "op"_a = op,
        "dtype_tag"_a = dtype_tag,
        "tokens"_a = tokens,
        "head_dim"_a = head_dim,
        "kernel_size"_a = kernel_size,
        "causal_rank"_a = causal_rank,
        "stride_unit"_a = stride_unit);
    if (!result.is_none()) {
      nb::tuple tg = nb::cast<nb::tuple>(result);
      if (nb::len(tg) == 3) {
        size_t x = static_cast<size_t>(std::max(1, nb::cast<int>(tg[0])));
        size_t y = static_cast<size_t>(std::max(1, nb::cast<int>(tg[1])));
        size_t z = static_cast<size_t>(std::max(1, nb::cast<int>(tg[2])));
        value = std::make_tuple(x, y, z);
      }
    }
  } catch (...) {
    value = std::nullopt;
  }
  {
    std::lock_guard<std::mutex> lock(tuning_cache_mutex());
    backward_threadgroup_tuning_cache()[key] = value;
  }
  return value;
}

std::optional<std::tuple<size_t, size_t, size_t>> lookup_threadgroup_from_tuning(
    const std::string& op,
    const std::string& dtype_tag,
    int tokens,
    int head_dim,
    int kernel_size,
    int causal_rank,
    bool stride_unit) {
  if (const char* v = std::getenv("NATTEN_NANOBIND_TG_OVERRIDE")) {
    std::string s(v);
    size_t c1 = s.find(',');
    size_t c2 = (c1 == std::string::npos) ? std::string::npos : s.find(',', c1 + 1);
    if (c1 != std::string::npos && c2 != std::string::npos) {
      try {
        int x = std::stoi(s.substr(0, c1));
        int y = std::stoi(s.substr(c1 + 1, c2 - c1 - 1));
        int z = std::stoi(s.substr(c2 + 1));
        return std::make_tuple(
            static_cast<size_t>(std::max(1, x)),
            static_cast<size_t>(std::max(1, y)),
            static_cast<size_t>(std::max(1, z)));
      } catch (...) {
      }
    }
  }
  TuningQueryKey key{
      op,
      dtype_tag,
      gpu_family_override_key(),
      tokens,
      head_dim,
      kernel_size,
      causal_rank,
      stride_unit};
  {
    std::lock_guard<std::mutex> lock(tuning_cache_mutex());
    auto it = forward_threadgroup_tuning_cache().find(key);
    if (it != forward_threadgroup_tuning_cache().end()) {
      return it->second;
    }
  }
  ensure_tuning_loaded();
  nb::object fn = threadgroup_lookup_fn();
  if (fn.is_none()) {
    std::lock_guard<std::mutex> lock(tuning_cache_mutex());
    forward_threadgroup_tuning_cache()[key] = std::nullopt;
    return std::nullopt;
  }
  std::optional<std::tuple<size_t, size_t, size_t>> value = std::nullopt;
  try {
    nb::object result = fn(
        "op"_a = op,
        "dtype_tag"_a = dtype_tag,
        "tokens"_a = tokens,
        "head_dim"_a = head_dim,
        "kernel_size"_a = kernel_size,
        "causal_rank"_a = causal_rank,
        "stride_unit"_a = stride_unit);
    if (!result.is_none()) {
      nb::tuple tg = nb::cast<nb::tuple>(result);
      if (nb::len(tg) == 3) {
        size_t x = static_cast<size_t>(std::max(1, nb::cast<int>(tg[0])));
        size_t y = static_cast<size_t>(std::max(1, nb::cast<int>(tg[1])));
        size_t z = static_cast<size_t>(std::max(1, nb::cast<int>(tg[2])));
        value = std::make_tuple(x, y, z);
      }
    }
  } catch (...) {
    value = std::nullopt;
  }
  {
    std::lock_guard<std::mutex> lock(tuning_cache_mutex());
    forward_threadgroup_tuning_cache()[key] = value;
  }
  return value;
}

std::string choose_softmax_strategy_from_tuning(
    const std::string& op,
    const std::string& dtype_tag,
    int tokens,
    int kernel_size,
    int causal_rank,
    bool stride_unit) {
  if (const char* v = std::getenv("NATTEN_NANOBIND_SOFTMAX_OVERRIDE")) {
    std::string s(v);
    if (s == "stored" || s == "recompute") {
      return s;
    }
  }
  TuningQueryKey key{
      op,
      dtype_tag,
      gpu_family_override_key(),
      tokens,
      -1,
      kernel_size,
      causal_rank,
      stride_unit};
  {
    std::lock_guard<std::mutex> lock(tuning_cache_mutex());
    auto it = softmax_strategy_cache().find(key);
    if (it != softmax_strategy_cache().end()) {
      return it->second;
    }
  }
  ensure_tuning_loaded();
  nb::object fn = softmax_strategy_fn();
  if (fn.is_none()) {
    std::lock_guard<std::mutex> lock(tuning_cache_mutex());
    softmax_strategy_cache()[key] = "stored";
    return "stored";
  }
  std::string strategy = "stored";
  try {
    std::string s = nb::cast<std::string>(fn(
        "op"_a = op,
        "dtype_tag"_a = dtype_tag,
        "tokens"_a = tokens,
        "kernel_size"_a = kernel_size,
        "causal_rank"_a = causal_rank,
        "stride_unit"_a = stride_unit));
    if (s == "recompute") {
      strategy = s;
    }
  } catch (...) {
  }
  {
    std::lock_guard<std::mutex> lock(tuning_cache_mutex());
    softmax_strategy_cache()[key] = strategy;
  }
  return strategy;
}

enum class FusedForwardMode {
  Auto,
  Fused,
  Split,
};

inline FusedForwardMode parse_fused_forward_mode_env(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return FusedForwardMode::Auto;
  }
  std::string mode(value);
  if (mode == "fused") {
    return FusedForwardMode::Fused;
  }
  if (mode == "split") {
    return FusedForwardMode::Split;
  }
  return FusedForwardMode::Auto;
}

inline bool prefer_split_composed_fwd_2d(
    int ih,
    int iw,
    int k,
    int sh,
    int sw,
    int dh,
    int dw,
    int ch,
    int cw) {
  FusedForwardMode mode = parse_fused_forward_mode_env("NATTEN_NANOBIND_FUSED_FWD_2D_MODE");
  if (mode == FusedForwardMode::Fused) {
    return false;
  }
  if (mode == FusedForwardMode::Split) {
    return true;
  }
  const bool noncausal = (ch == 0 && cw == 0);
  const bool unit_step = (sh == 1 && sw == 1 && dh == 1 && dw == 1);
  const int tokens = ih * iw;
  return noncausal && unit_step && k >= 7 && tokens >= 512;
}

inline bool prefer_split_composed_fwd_3d(
    int id,
    int ih,
    int iw,
    int k,
    int sd,
    int sh,
    int sw,
    int dd,
    int dh,
    int dw,
    int cd,
    int ch,
    int cw) {
  FusedForwardMode mode = parse_fused_forward_mode_env("NATTEN_NANOBIND_FUSED_FWD_3D_MODE");
  if (mode == FusedForwardMode::Fused) {
    return false;
  }
  if (mode == FusedForwardMode::Split) {
    return true;
  }
  const bool noncausal = (cd == 0 && ch == 0 && cw == 0);
  const bool unit_step = (sd == 1 && sh == 1 && sw == 1 && dd == 1 && dh == 1 && dw == 1);
  const int tokens = id * ih * iw;
  return noncausal && unit_step && k >= 3 && tokens >= 1024;
}

void throw_if_forced_split_failure() {
  if (natten_mlx::nanobind_metal_runtime::debug_forced_split_failure()) {
    throw std::runtime_error("forced split failure");
  }
}

void throw_if_forced_fused_failure() {
  if (natten_mlx::nanobind_metal_runtime::debug_forced_fused_failure()) {
    throw std::runtime_error("forced fused failure");
  }
}

MTL::ComputePipelineState* get_kernel(const std::string& name) {
  {
    std::lock_guard<std::mutex> lock(kernel_cache_mutex());
    auto it = kernel_cache().find(name);
    if (it != kernel_cache().end()) {
      return it->second;
    }
  }
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* lib = dev.get_library("natten_nb", current_binary_dir());
  if (lib == nullptr) {
    throw std::runtime_error("failed to load natten_nb metallib");
  }
  auto* kernel = dev.get_kernel(name, lib);
  if (kernel == nullptr) {
    throw std::runtime_error("failed to resolve kernel: " + name);
  }
  {
    std::lock_guard<std::mutex> lock(kernel_cache_mutex());
    kernel_cache()[name] = kernel;
  }
  return kernel;
}

LaunchConfig linear_launch(size_t threads, size_t tg_x = 256) {
  threads = std::max<size_t>(1, threads);
  size_t x = std::max<size_t>(1, std::min(tg_x, threads));
  return LaunchConfig{MTL::Size(threads, 1, 1), MTL::Size(x, 1, 1)};
}

inline bool is_backward_kernel_name(const std::string& kernel_name) {
  return kernel_name.find("_bwd_") != std::string::npos ||
      kernel_name.find("_backward") != std::string::npos;
}

LaunchConfig maybe_override_backward_launch(
    const std::string& kernel_name,
    size_t threads,
    const LaunchConfig& fallback) {
  if (!is_backward_kernel_name(kernel_name)) {
    return fallback;
  }
  const bool env_bwd_tg_override = std::getenv("NATTEN_NANOBIND_BWD_TG_OVERRIDE") != nullptr;
  // High-priority kernel-specific overrides for known hotspots. Keep env
  // override behavior intact for active tuning sessions.
  const bool is_3d_split_q_vec4 = kernel_name == "na3d_qk_bwd_q_vec4_fp32";
  const bool is_3d_split_k_direct_nc =
      kernel_name.find("na3d_qk_bwd_k_direct_u1d1_nc") != std::string::npos;
  const bool is_3d_split_v_direct_nc =
      kernel_name.find("na3d_av_bwd_v_direct_u1d1_nc") != std::string::npos;
  const bool is_3d_split_attn_vec4 = kernel_name == "na3d_av_bwd_attn_vec4_fp32";
  const bool is_3d_fused_q_softmax_k =
      kernel_name == "na3d_fused_bwd_q_softmax_k3_fp32" ||
      kernel_name == "na3d_fused_bwd_q_softmax_k3_vec4_fp32" ||
      kernel_name == "na3d_fused_bwd_q_softmax_k5_fp32" ||
      kernel_name == "na3d_fused_bwd_q_softmax_k5_vec4_fp32" ||
      kernel_name == "na3d_fused_bwd_q_softmax_token_k3_vec4_fp32";
  const bool is_3d_fused_k_direct_softmax =
      kernel_name.find("na3d_fused_bwd_k_direct_softmax_u1d1_nc") != std::string::npos;
  const bool is_3d_fused_kv_tiled =
      kernel_name.find("na3d_fused_bwd_kv_softmax_tiled") != std::string::npos;
  const bool is_3d_fused_qkv_tiled =
      kernel_name.find("na3d_fused_bwd_qkv_softmax_tiled") != std::string::npos;
  const bool is_2d_fused_q_softmax =
      kernel_name.find("na2d_fused_bwd_q_softmax") != std::string::npos;
  const bool is_2d_fused_kv_tiled =
      kernel_name.find("na2d_fused_bwd_kv_softmax_tiled") != std::string::npos;
  const bool is_2d_fused_qkv_tiled =
      kernel_name.find("na2d_fused_bwd_qkv_softmax_tiled") != std::string::npos;
  const bool is_1d_fused_q =
      kernel_name == "na1d_fused_bwd_q_softmax_s1_fp32" ||
      kernel_name == "na1d_fused_bwd_q_softmax_s1_vec4_fp32" ||
      kernel_name == "na1d_fused_bwd_q_softmax_direct_s1_causal_fp32" ||
      kernel_name == "na1d_fused_bwd_q_softmax_direct_s1_causal_vec4_fp32" ||
      kernel_name == "na1d_fused_bwd_q_softmax_direct_s1_causal_k9_fp32" ||
      kernel_name == "na1d_fused_bwd_q_softmax_direct_s1_causal_k9_vec4_fp32" ||
      kernel_name == "na1d_fused_bwd_q_softmax_direct_s1_causal_k9_token_vec4_fp32";
  const bool is_1d_fused_attn_causal_k9 =
      kernel_name == "na1d_fused_bwd_attn_s1_causal_k9_fp32" ||
      kernel_name == "na1d_fused_bwd_attn_s1_causal_k9_vec4_fp32";
  const bool is_1d_fused_kv_direct =
      kernel_name.find("na1d_fused_bwd_kv_softmax_direct_") != std::string::npos;
  const bool is_1d_fused_kv_direct_causal_k9 =
      kernel_name.find("na1d_fused_bwd_kv_softmax_direct_s1_causal_k9") != std::string::npos;
  const bool is_1d_split_k_direct_nc =
      kernel_name.find("na1d_qk_bwd_k_direct_u1d1_nc") != std::string::npos;
  const bool is_1d_split_v_direct_nc =
      kernel_name.find("na1d_av_bwd_v_direct_u1d1_nc") != std::string::npos;
  const bool is_1d_split_k_direct_causal =
      kernel_name.find("na1d_qk_bwd_k_direct_s1_causal") != std::string::npos;
  const bool is_1d_split_v_direct_causal =
      kernel_name.find("na1d_av_bwd_v_direct_s1_causal") != std::string::npos;

  if (!env_bwd_tg_override) {
    if (is_3d_split_q_vec4 || is_3d_split_k_direct_nc || is_3d_split_v_direct_nc ||
        is_3d_split_attn_vec4) {
      size_t tg = parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_3D_SPLIT").value_or(224);
      return linear_launch(threads, tg);
    }
    if (is_3d_fused_q_softmax_k) {
      if (kernel_name == "na3d_fused_bwd_q_softmax_token_k3_vec4_fp32") {
        size_t tg =
            parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_3D_FUSED_Q_TOKEN").value_or(128);
        return linear_launch(threads, tg);
      }
      size_t tg = parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_3D_FUSED_Q").value_or(288);
      return linear_launch(threads, tg);
    }
    if (is_3d_fused_k_direct_softmax || is_3d_fused_kv_tiled || is_3d_fused_qkv_tiled) {
      size_t tg = parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_3D_FUSED_KV").value_or(192);
      return linear_launch(threads, tg);
    }
    if (is_2d_fused_q_softmax) {
      size_t tg = parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_2D_FUSED_Q").value_or(192);
      return linear_launch(threads, tg);
    }
    if (is_2d_fused_kv_tiled || is_2d_fused_qkv_tiled) {
      size_t tg = parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_2D_FUSED_KV").value_or(256);
      return linear_launch(threads, tg);
    }
  }

  auto tg = lookup_backward_threadgroup_from_tuning(
      kernel_name,
      "fp32",
      static_cast<int>(threads),
      -1,
      -1,
      0,
      true);
  if (!tg.has_value()) {
    if (is_1d_fused_q || is_1d_fused_kv_direct || is_1d_split_k_direct_nc ||
        is_1d_split_v_direct_nc || is_1d_split_k_direct_causal || is_1d_split_v_direct_causal) {
      const bool causal_1d =
          kernel_name.find("_s1_causal_") != std::string::npos ||
          kernel_name.find("_direct_s1_causal") != std::string::npos;
      if (causal_1d) {
        if (is_1d_fused_attn_causal_k9) {
          size_t causal_attn =
              parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_1D_CAUSAL_ATTN").value_or(256);
          return linear_launch(threads, causal_attn);
        }
        size_t causal_q = parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_1D_CAUSAL_Q").value_or(192);
        size_t causal_kv =
            parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_1D_CAUSAL_KV").value_or(256);
        size_t causal_split =
            parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_1D_CAUSAL_SPLIT_KV").value_or(192);
        if (kernel_name == "na1d_fused_bwd_q_softmax_direct_s1_causal_k9_token_vec4_fp32") {
          size_t causal_q_token =
              parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_1D_CAUSAL_Q_TOKEN").value_or(256);
          return linear_launch(threads, causal_q_token);
        }
        if (is_1d_fused_kv_direct_causal_k9) {
          size_t causal_kv_k9 =
              parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_1D_CAUSAL_KV_K9").value_or(320);
          return linear_launch(threads, causal_kv_k9);
        }
        if (is_1d_fused_q) {
          return linear_launch(threads, causal_q);
        }
        if (is_1d_fused_kv_direct) {
          return linear_launch(threads, causal_kv);
        }
        if (is_1d_split_k_direct_causal || is_1d_split_v_direct_causal) {
          return linear_launch(threads, causal_split);
        }
      }
      return linear_launch(threads, 128);
    }
    if (is_3d_split_q_vec4 || is_3d_split_k_direct_nc || is_3d_split_v_direct_nc ||
        is_3d_split_attn_vec4) {
      size_t tg_split = parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_3D_SPLIT").value_or(224);
      return linear_launch(threads, tg_split);
    }
    if (is_3d_fused_q_softmax_k) {
      if (kernel_name == "na3d_fused_bwd_q_softmax_token_k3_vec4_fp32") {
        size_t tg_token =
            parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_3D_FUSED_Q_TOKEN").value_or(128);
        return linear_launch(threads, tg_token);
      }
      size_t tg_q = parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_3D_FUSED_Q").value_or(288);
      return linear_launch(threads, tg_q);
    }
    if (is_3d_fused_k_direct_softmax || is_3d_fused_kv_tiled || is_3d_fused_qkv_tiled) {
      size_t tg_kv = parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_3D_FUSED_KV").value_or(192);
      return linear_launch(threads, tg_kv);
    }
    if (is_2d_fused_q_softmax) {
      size_t tg_q2 = parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_2D_FUSED_Q").value_or(192);
      return linear_launch(threads, tg_q2);
    }
    if (is_2d_fused_kv_tiled || is_2d_fused_qkv_tiled) {
      size_t tg_kv2 = parse_positive_tg_env("NATTEN_NANOBIND_BWD_TG_2D_FUSED_KV").value_or(256);
      return linear_launch(threads, tg_kv2);
    }
    return fallback;
  }
  size_t tx = std::max<size_t>(1, std::get<0>(*tg));
  size_t ty = std::max<size_t>(1, std::get<1>(*tg));
  size_t tz = std::max<size_t>(1, std::get<2>(*tg));
  return LaunchConfig{MTL::Size(std::max<size_t>(1, threads), 1, 1), MTL::Size(tx, ty, tz)};
}

LaunchConfig maybe_override_launch(
    const std::string& kernel_name,
    size_t threads,
    const LaunchConfig& fallback) {
  if (is_backward_kernel_name(kernel_name)) {
    return maybe_override_backward_launch(kernel_name, threads, fallback);
  }
  const bool env_tg_override = std::getenv("NATTEN_NANOBIND_TG_OVERRIDE") != nullptr;
  const bool is_3d_split_av_k3 =
      kernel_name == "na3d_av_k3_fp32" || kernel_name == "na3d_av_k3_vec4_fp32";
  const bool is_1d_fwd_causal_k9d_vec4 = kernel_name == "na1d_fused_causal_k9d_vec4_fp32";
  if (!env_tg_override && is_1d_fwd_causal_k9d_vec4) {
    size_t tg = parse_positive_tg_env("NATTEN_NANOBIND_FWD_TG_1D_CAUSAL_K9D").value_or(256);
    return linear_launch(threads, tg);
  }
  if (!env_tg_override && is_3d_split_av_k3) {
    return linear_launch(threads, 160);
  }
  return fallback;
}

void maybe_record_launch_metrics(
    const std::string& kernel_name,
    std::chrono::steady_clock::time_point start_time) {
  if (!launch_metrics_enabled()) {
    return;
  }
  double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start_time)
                  .count();
  std::lock_guard<std::mutex> lock(launch_metrics_mutex());
  launch_metrics_ms_total()[kernel_name] += ms;
  launch_metrics_count()[kernel_name] += 1;
}

void clear_output_f32(mx::array& out, size_t out_elems) {
  if (out_elems == 0) {
    return;
  }
  auto stream = mx::default_stream(mx::Device::gpu);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* clear_kernel = get_kernel("natten_clear_f32");
  auto clear_launch = linear_launch(out_elems);
  auto& enc = dev.get_command_encoder(stream.index);
  enc.set_compute_pipeline_state(clear_kernel);
  enc.set_output_array(out, 0);
  Clear1Params clear_params{static_cast<uint32_t>(out_elems)};
  enc.set_bytes(clear_params, 1);
  enc.dispatch_threads(clear_launch.grid, clear_launch.threadgroup);
  dev.end_encoding(stream.index);
}

void clear_output2_f32(mx::array& out0, size_t out0_elems, mx::array& out1, size_t out1_elems) {
  size_t threads = std::max(out0_elems, out1_elems);
  if (threads == 0) {
    return;
  }
  auto stream = mx::default_stream(mx::Device::gpu);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* clear_kernel = get_kernel("natten_clear2_f32");
  auto clear_launch = linear_launch(threads);
  auto& enc = dev.get_command_encoder(stream.index);
  enc.set_compute_pipeline_state(clear_kernel);
  enc.set_output_array(out0, 0);
  enc.set_output_array(out1, 1);
  Clear2Params clear_params{static_cast<uint32_t>(out0_elems), static_cast<uint32_t>(out1_elems)};
  enc.set_bytes(clear_params, 2);
  enc.dispatch_threads(clear_launch.grid, clear_launch.threadgroup);
  dev.end_encoding(stream.index);
}

void clear_output3_f32(
    mx::array& out0,
    size_t out0_elems,
    mx::array& out1,
    size_t out1_elems,
    mx::array& out2,
    size_t out2_elems) {
  size_t threads = std::max(out0_elems, std::max(out1_elems, out2_elems));
  if (threads == 0) {
    return;
  }
  auto stream = mx::default_stream(mx::Device::gpu);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* clear_kernel = get_kernel("natten_clear3_f32");
  auto clear_launch = linear_launch(threads);
  auto& enc = dev.get_command_encoder(stream.index);
  enc.set_compute_pipeline_state(clear_kernel);
  enc.set_output_array(out0, 0);
  enc.set_output_array(out1, 1);
  enc.set_output_array(out2, 2);
  Clear3Params clear_params{
      static_cast<uint32_t>(out0_elems),
      static_cast<uint32_t>(out1_elems),
      static_cast<uint32_t>(out2_elems)};
  enc.set_bytes(clear_params, 3);
  enc.dispatch_threads(clear_launch.grid, clear_launch.threadgroup);
  dev.end_encoding(stream.index);
}

LaunchConfig make_launch(
    size_t grid_x,
    size_t grid_y,
    size_t grid_z,
    size_t tg_x,
    size_t tg_y,
    size_t tg_z) {
  return LaunchConfig{
      MTL::Size(std::max<size_t>(1, grid_x), std::max<size_t>(1, grid_y), std::max<size_t>(1, grid_z)),
      MTL::Size(std::max<size_t>(1, tg_x), std::max<size_t>(1, tg_y), std::max<size_t>(1, tg_z))};
}

VariantConfig fused_forward_variant(const mx::array& x, bool prefer_vec4) {
  size_t rank = x.shape().size();
  int head_dim = x.shape(static_cast<int>(rank) - 1);
  bool vec4 = prefer_vec4 && (head_dim % 4 == 0);
  const char* mode_env = std::getenv("NATTEN_NANOBIND_FWD_DTYPE_MODE");
  std::string mode = mode_env == nullptr ? "auto" : std::string(mode_env);

  if (mode == "fp32") {
    return {"fp32", vec4};
  }

  if (x.dtype() == mx::float16) {
    // Keep low-precision forward native by default for better memory bandwidth.
    return {"fp16", vec4};
  }
  if (x.dtype() == mx::bfloat16) {
    return {"bf16", vec4};
  }
  if (x.dtype() == mx::float32 || mode == "native") {
    return {"fp32", vec4};
  }
  return {"fp32", false};
}

SplitForwardVariant split_forward_variant(const mx::array& x) {
  const char* mode_env = std::getenv("NATTEN_NANOBIND_SPLIT_FWD_DTYPE_MODE");
  std::string mode = mode_env == nullptr ? "auto" : std::string(mode_env);
  if (mode == "fp32") {
    return {"fp32", false};
  }
  if (x.dtype() == mx::float16) {
    return {"fp16", true};
  }
  if (x.dtype() == mx::bfloat16) {
    return {"bf16", true};
  }
  return {"fp32", false};
}

void debug_set_last_kernel(const std::string& op, const std::string& kernel_name) {
  std::lock_guard<std::mutex> lock(route_mutex());
  kernel_map()[op] = kernel_name;
}

template <typename Params>
mx::array launch_one_cfg(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const Params& params,
    const mx::Shape& out_shape,
    const LaunchConfig& launch,
    bool accumulation_output,
    mx::Dtype out_dtype = mx::float32);

template <typename Params>
mx::array launch_one(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const Params& params,
    const mx::Shape& out_shape,
    bool accumulation_output,
    size_t thread_count = 0,
    mx::Dtype out_dtype = mx::float32) {
  size_t threads = (thread_count == 0) ? numel(out_shape) : thread_count;
  LaunchConfig launch = linear_launch(threads);
  launch = maybe_override_launch(kernel_name, threads, launch);
  return launch_one_cfg(
      kernel_name,
      inputs,
      params,
      out_shape,
      launch,
      accumulation_output,
      out_dtype);
}

template <typename Params>
mx::array launch_one_cfg(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const Params& params,
    const mx::Shape& out_shape,
    const LaunchConfig& launch,
    bool accumulation_output,
    mx::Dtype out_dtype) {
  auto stream = mx::default_stream(mx::Device::gpu);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* kernel = get_kernel(kernel_name);
  std::vector<mx::array> realized_inputs = inputs;
  mx::eval(realized_inputs);
  auto launch_start = std::chrono::steady_clock::now();

  size_t out_elems = numel(out_shape);
  mx::array out(mx::allocator::malloc(out_elems * mx::size_of(out_dtype)), out_shape, out_dtype);
  if (accumulation_output) {
    if (out_dtype != mx::float32) {
      throw std::runtime_error("accumulation outputs require float32 output dtype");
    }
    clear_output_f32(out, out_elems);
  }

  auto& enc = dev.get_command_encoder(stream.index);
  enc.set_compute_pipeline_state(kernel);
  int arg = 0;
  for (const auto& in : realized_inputs) {
    enc.set_input_array(in, arg++);
  }
  enc.set_output_array(out, arg++);
  enc.set_bytes(params, arg++);

  enc.dispatch_threads(launch.grid, launch.threadgroup);
  dev.end_encoding(stream.index);
  if (dev.command_buffer_needs_commit(stream.index)) {
    dev.commit_command_buffer(stream.index);
  }
  if (sync_launch_outputs_enabled()) {
    mx::array out_copy = mx::copy(out, stream);
    mx::eval({out_copy});
    maybe_record_launch_metrics(kernel_name, launch_start);
    return out_copy;
  }
  mx::array out_copy = mx::copy(out, stream);
  maybe_record_launch_metrics(kernel_name, launch_start);
  return out_copy;
}

template <typename Params>
nb::tuple launch_two_cfg(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const Params& params,
    const mx::Shape& out_shape0,
    const mx::Shape& out_shape1,
    const LaunchConfig& launch,
    bool accumulation_output);

template <typename Params>
nb::tuple launch_two(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const Params& params,
    const mx::Shape& out_shape0,
    const mx::Shape& out_shape1,
    bool accumulation_output,
    size_t thread_count = 0) {
  size_t threads =
      (thread_count == 0) ? std::max(numel(out_shape0), numel(out_shape1)) : thread_count;
  LaunchConfig launch = linear_launch(threads);
  launch = maybe_override_launch(kernel_name, threads, launch);
  return launch_two_cfg(
      kernel_name,
      inputs,
      params,
      out_shape0,
      out_shape1,
      launch,
      accumulation_output);
}

template <typename Params>
nb::tuple launch_two_cfg(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const Params& params,
    const mx::Shape& out_shape0,
    const mx::Shape& out_shape1,
    const LaunchConfig& launch,
    bool accumulation_output) {
  auto stream = mx::default_stream(mx::Device::gpu);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* kernel = get_kernel(kernel_name);
  std::vector<mx::array> realized_inputs = inputs;
  mx::eval(realized_inputs);
  auto launch_start = std::chrono::steady_clock::now();

  size_t out0_elems = numel(out_shape0);
  size_t out1_elems = numel(out_shape1);
  mx::array out0(mx::allocator::malloc(out0_elems * mx::size_of(mx::float32)), out_shape0, mx::float32);
  mx::array out1(mx::allocator::malloc(out1_elems * mx::size_of(mx::float32)), out_shape1, mx::float32);
  if (accumulation_output) {
    clear_output2_f32(out0, out0_elems, out1, out1_elems);
  }

  auto& enc = dev.get_command_encoder(stream.index);
  enc.set_compute_pipeline_state(kernel);
  int arg = 0;
  for (const auto& in : realized_inputs) {
    enc.set_input_array(in, arg++);
  }
  enc.set_output_array(out0, arg++);
  enc.set_output_array(out1, arg++);
  enc.set_bytes(params, arg++);

  enc.dispatch_threads(launch.grid, launch.threadgroup);
  dev.end_encoding(stream.index);
  if (dev.command_buffer_needs_commit(stream.index)) {
    dev.commit_command_buffer(stream.index);
  }
  if (sync_launch_outputs_enabled()) {
    mx::array out0_copy = mx::copy(out0, stream);
    mx::array out1_copy = mx::copy(out1, stream);
    mx::eval({out0_copy, out1_copy});
    maybe_record_launch_metrics(kernel_name, launch_start);
    return nb::make_tuple(out0_copy, out1_copy);
  }
  mx::array out0_copy = mx::copy(out0, stream);
  mx::array out1_copy = mx::copy(out1, stream);
  maybe_record_launch_metrics(kernel_name, launch_start);
  return nb::make_tuple(out0_copy, out1_copy);
}

template <typename Params>
nb::tuple launch_three_cfg(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const Params& params,
    const mx::Shape& out_shape0,
    const mx::Shape& out_shape1,
    const mx::Shape& out_shape2,
    const LaunchConfig& launch,
    bool accumulation_output);

template <typename Params>
nb::tuple launch_three(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const Params& params,
    const mx::Shape& out_shape0,
    const mx::Shape& out_shape1,
    const mx::Shape& out_shape2,
    bool accumulation_output,
    size_t thread_count = 0) {
  size_t threads = thread_count == 0
      ? std::max(numel(out_shape0), std::max(numel(out_shape1), numel(out_shape2)))
      : thread_count;
  LaunchConfig launch = linear_launch(threads);
  launch = maybe_override_backward_launch(kernel_name, threads, launch);
  return launch_three_cfg(
      kernel_name,
      inputs,
      params,
      out_shape0,
      out_shape1,
      out_shape2,
      launch,
      accumulation_output);
}

template <typename Params>
nb::tuple launch_three_cfg(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const Params& params,
    const mx::Shape& out_shape0,
    const mx::Shape& out_shape1,
    const mx::Shape& out_shape2,
    const LaunchConfig& launch,
    bool accumulation_output) {
  auto stream = mx::default_stream(mx::Device::gpu);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* kernel = get_kernel(kernel_name);
  std::vector<mx::array> realized_inputs = inputs;
  mx::eval(realized_inputs);
  auto launch_start = std::chrono::steady_clock::now();

  size_t out0_elems = numel(out_shape0);
  size_t out1_elems = numel(out_shape1);
  size_t out2_elems = numel(out_shape2);
  mx::array out0(mx::allocator::malloc(out0_elems * mx::size_of(mx::float32)), out_shape0, mx::float32);
  mx::array out1(mx::allocator::malloc(out1_elems * mx::size_of(mx::float32)), out_shape1, mx::float32);
  mx::array out2(mx::allocator::malloc(out2_elems * mx::size_of(mx::float32)), out_shape2, mx::float32);
  if (accumulation_output) {
    clear_output3_f32(out0, out0_elems, out1, out1_elems, out2, out2_elems);
  }

  auto& enc = dev.get_command_encoder(stream.index);
  enc.set_compute_pipeline_state(kernel);
  int arg = 0;
  for (const auto& in : realized_inputs) {
    enc.set_input_array(in, arg++);
  }
  enc.set_output_array(out0, arg++);
  enc.set_output_array(out1, arg++);
  enc.set_output_array(out2, arg++);
  enc.set_bytes(params, arg++);

  enc.dispatch_threads(launch.grid, launch.threadgroup);
  dev.end_encoding(stream.index);
  if (dev.command_buffer_needs_commit(stream.index)) {
    dev.commit_command_buffer(stream.index);
  }
  if (sync_launch_outputs_enabled()) {
    mx::array out0_copy = mx::copy(out0, stream);
    mx::array out1_copy = mx::copy(out1, stream);
    mx::array out2_copy = mx::copy(out2, stream);
    mx::eval({out0_copy, out1_copy, out2_copy});
    maybe_record_launch_metrics(kernel_name, launch_start);
    return nb::make_tuple(out0_copy, out1_copy, out2_copy);
  }
  mx::array out0_copy = mx::copy(out0, stream);
  mx::array out1_copy = mx::copy(out1, stream);
  mx::array out2_copy = mx::copy(out2, stream);
  maybe_record_launch_metrics(kernel_name, launch_start);
  return nb::make_tuple(out0_copy, out1_copy, out2_copy);
}

}  // namespace

namespace natten_mlx::nanobind_metal_runtime {

bool supports_1d_fused(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation) {
  return valid_kernel(first_kernel_size(kernel_size)) && valid_stride_1d(stride) &&
      valid_dilation_1d(dilation);
}

bool supports_2d_fused(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation) {
  return square_kernel_2d(kernel_size) && valid_kernel(first_kernel_size(kernel_size)) &&
      valid_stride_2d(stride) && valid_dilation_2d(dilation);
}

bool supports_3d_fused(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation) {
  return cubic_kernel_3d(kernel_size) && valid_kernel(first_kernel_size(kernel_size)) &&
      valid_stride_3d(stride) && valid_dilation_3d(dilation);
}

bool supports_1d_split(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object&) {
  return supports_1d_fused(kernel_size, stride, dilation);
}

bool supports_2d_split(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object&) {
  return supports_2d_fused(kernel_size, stride, dilation);
}

bool supports_3d_split(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object&) {
  return supports_3d_fused(kernel_size, stride, dilation);
}

nb::object na1d_qk_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_split_failure();
  if (!supports_1d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 1D QK unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  SplitForwardVariant variant = split_forward_variant(q_arr);
  mx::array qx = q_arr;
  mx::array kx = k_arr;
  if (!variant.native_lowp) {
    qx = to_float32(q_arr);
    kx = to_float32(k_arr);
  } else {
    mx::Dtype target = q_arr.dtype();
    if (k_arr.dtype() != target) {
      kx = cast_to_dtype(k_arr, target);
    }
  }

  int B = qx.shape(0);
  int L = qx.shape(1);
  int H = qx.shape(2);
  int D = qx.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int out_l = ceil_div(L, S);

  NA1DParams p{B, L, H, D, K, S, Dil, C, resolve_scale(scale, D)};
  bool use_vec4 =
      !variant.native_lowp && direct_vec4_eligible(D) && S == 1 && Dil == 1 && C == 0;
  auto out = [&]() -> mx::array {
    if (use_vec4) {
      try {
        return launch_one("na1d_qk_vec4_fp32", {qx, kx}, p, to_shape({B, out_l, H, K}), false);
      } catch (...) {
      }
    }
    std::string kname = std::string("na1d_qk_") + variant.dtype_tag;
    try {
      return launch_one(
          kname,
          {qx, kx},
          p,
          to_shape({B, out_l, H, K}),
          false,
          0,
          variant.native_lowp ? q_arr.dtype() : mx::float32);
    } catch (...) {
      if (!variant.native_lowp) {
        throw;
      }
      qx = to_float32(q_arr);
      kx = to_float32(k_arr);
      return launch_one("na1d_qk_fp32", {qx, kx}, p, to_shape({B, out_l, H, K}), false);
    }
  }();
  return nb::cast(cast_to_dtype(out, q_arr.dtype()));
}

nb::object na1d_av_forward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  throw_if_forced_split_failure();
  if (!supports_1d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 1D AV unsupported configuration");
  }

  auto attn_arr = as_array(attn);
  auto v_arr = as_array(v);
  SplitForwardVariant variant = split_forward_variant(v_arr);
  mx::array af = attn_arr;
  mx::array vf = v_arr;
  if (!variant.native_lowp) {
    af = to_float32(attn_arr);
    vf = to_float32(v_arr);
  } else {
    mx::Dtype target = v_arr.dtype();
    if (attn_arr.dtype() != target) {
      af = cast_to_dtype(attn_arr, target);
    }
  }

  int B = vf.shape(0);
  int L = vf.shape(1);
  int H = vf.shape(2);
  int D = vf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int out_l = af.shape(1);

  NA1DParams p{B, L, H, D, K, S, Dil, C, 1.0f};
  bool use_vec4 =
      !variant.native_lowp && direct_vec4_eligible(D) && S == 1 && Dil == 1 && C == 0;
  auto out_shape = to_shape({B, out_l, H, D});
  auto out = [&]() -> mx::array {
    if (use_vec4) {
      try {
        return launch_one(
            "na1d_av_vec4_fp32",
            {af, vf},
            p,
            out_shape,
            false,
            std::max<size_t>(1, numel(out_shape) / 4));
      } catch (...) {
      }
    }
    std::string kname = std::string("na1d_av_") + variant.dtype_tag;
    try {
      return launch_one(
          kname,
          {af, vf},
          p,
          out_shape,
          false,
          0,
          variant.native_lowp ? v_arr.dtype() : mx::float32);
    } catch (...) {
      if (!variant.native_lowp) {
        throw;
      }
      af = to_float32(attn_arr);
      vf = to_float32(v_arr);
      return launch_one("na1d_av_fp32", {af, vf}, p, out_shape, false);
    }
  }();
  return nb::cast(cast_to_dtype(out, v_arr.dtype()));
}

nb::object na2d_qk_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_split_failure();
  if (!supports_2d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 2D QK unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  SplitForwardVariant variant = split_forward_variant(q_arr);
  mx::array qf = q_arr;
  mx::array kf = k_arr;
  if (!variant.native_lowp) {
    qf = to_float32(q_arr);
    kf = to_float32(k_arr);
  } else {
    mx::Dtype target = q_arr.dtype();
    if (k_arr.dtype() != target) {
      kf = cast_to_dtype(k_arr, target);
    }
  }

  int B = qf.shape(0);
  int IH = qf.shape(1);
  int IW = qf.shape(2);
  int H = qf.shape(3);
  int D = qf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int out_h = ceil_div(IH, SH);
  int out_w = ceil_div(IW, SW);

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, resolve_scale(scale, D)};
  bool use_vec4_lowp =
      variant.native_lowp && q_arr.dtype() == mx::float16 && direct_vec4_eligible(D);
  bool use_vec4_fp32 = !variant.native_lowp && direct_vec4_eligible(D) && SH == 1 && SW == 1 &&
      DH == 1 && DW == 1 && CH == 0 && CW == 0;
  auto out = [&]() -> mx::array {
    if (use_vec4_lowp) {
      try {
        return launch_one(
            std::string("na2d_qk_vec4_") + variant.dtype_tag,
            {qf, kf},
            p,
            to_shape({B, out_h, out_w, H, K * K}),
            false,
            0,
            q_arr.dtype());
      } catch (...) {
      }
    }
    if (use_vec4_fp32) {
      try {
        return launch_one(
            "na2d_qk_vec4_fp32", {qf, kf}, p, to_shape({B, out_h, out_w, H, K * K}), false);
      } catch (...) {
      }
    }
    std::string kname = std::string("na2d_qk_") + variant.dtype_tag;
    try {
      return launch_one(
          kname,
          {qf, kf},
          p,
          to_shape({B, out_h, out_w, H, K * K}),
          false,
          0,
          variant.native_lowp ? q_arr.dtype() : mx::float32);
    } catch (...) {
      if (!variant.native_lowp) {
        throw;
      }
      qf = to_float32(q_arr);
      kf = to_float32(k_arr);
      return launch_one(
          "na2d_qk_fp32", {qf, kf}, p, to_shape({B, out_h, out_w, H, K * K}), false);
    }
  }();
  return nb::cast(cast_to_dtype(out, q_arr.dtype()));
}

nb::object na2d_av_forward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  throw_if_forced_split_failure();
  if (!supports_2d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 2D AV unsupported configuration");
  }

  auto attn_arr = as_array(attn);
  auto v_arr = as_array(v);
  SplitForwardVariant variant = split_forward_variant(v_arr);
  mx::array af = attn_arr;
  mx::array vf = v_arr;
  if (!variant.native_lowp) {
    af = to_float32(attn_arr);
    vf = to_float32(v_arr);
  } else {
    mx::Dtype target = v_arr.dtype();
    if (attn_arr.dtype() != target) {
      af = cast_to_dtype(attn_arr, target);
    }
  }

  int B = vf.shape(0);
  int IH = vf.shape(1);
  int IW = vf.shape(2);
  int H = vf.shape(3);
  int D = vf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;

  int out_h = af.shape(1);
  int out_w = af.shape(2);

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, 1.0f};
  bool use_vec4_lowp =
      variant.native_lowp && v_arr.dtype() == mx::float16 && direct_vec4_eligible(D);
  bool use_vec4_fp32 = !variant.native_lowp && direct_vec4_eligible(D) && SH == 1 && SW == 1 &&
      DH == 1 && DW == 1 && CH == 0 && CW == 0;
  auto out_shape = to_shape({B, out_h, out_w, H, D});
  auto out = [&]() -> mx::array {
    if (use_vec4_lowp) {
      try {
        return launch_one(
            std::string("na2d_av_vec4_") + variant.dtype_tag,
            {af, vf},
            p,
            out_shape,
            false,
            std::max<size_t>(1, numel(out_shape) / 4),
            v_arr.dtype());
      } catch (...) {
      }
    }
    if (use_vec4_fp32) {
      try {
        return launch_one(
            "na2d_av_vec4_fp32",
            {af, vf},
            p,
            out_shape,
            false,
            std::max<size_t>(1, numel(out_shape) / 4));
      } catch (...) {
      }
    }
    std::string kname = std::string("na2d_av_") + variant.dtype_tag;
    try {
      return launch_one(
          kname,
          {af, vf},
          p,
          out_shape,
          false,
          0,
          variant.native_lowp ? v_arr.dtype() : mx::float32);
    } catch (...) {
      if (!variant.native_lowp) {
        throw;
      }
      af = to_float32(attn_arr);
      vf = to_float32(v_arr);
      return launch_one("na2d_av_fp32", {af, vf}, p, out_shape, false);
    }
  }();
  return nb::cast(cast_to_dtype(out, v_arr.dtype()));
}

nb::object na3d_qk_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_split_failure();
  if (!supports_3d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 3D QK unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  SplitForwardVariant variant = split_forward_variant(q_arr);
  mx::array qf = q_arr;
  mx::array kf = k_arr;
  if (!variant.native_lowp) {
    qf = to_float32(q_arr);
    kf = to_float32(k_arr);
  } else {
    mx::Dtype target = q_arr.dtype();
    if (k_arr.dtype() != target) {
      kf = cast_to_dtype(k_arr, target);
    }
  }

  int B = qf.shape(0);
  int ID = qf.shape(1);
  int IH = qf.shape(2);
  int IW = qf.shape(3);
  int H = qf.shape(4);
  int D = qf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;
  int out_d = ceil_div(ID, SD);
  int out_h = ceil_div(IH, SH);
  int out_w = ceil_div(IW, SW);

  NA3DParams p{
      B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, resolve_scale(scale, D)};
  bool use_vec4 = !variant.native_lowp && direct_vec4_eligible(D) && SD == 1 && SH == 1 &&
      SW == 1 && DD == 1 &&
      DH == 1 && DW == 1 && CD == 0 && CH == 0 && CW == 0;
  auto out = [&]() -> mx::array {
    if (use_vec4) {
      try {
        return launch_one(
            "na3d_qk_vec4_fp32",
            {qf, kf},
            p,
            to_shape({B, out_d, out_h, out_w, H, K * K * K}),
            false);
      } catch (...) {
      }
    }
    std::string kname = std::string("na3d_qk_") + variant.dtype_tag;
    try {
      return launch_one(
          kname,
          {qf, kf},
          p,
          to_shape({B, out_d, out_h, out_w, H, K * K * K}),
          false,
          0,
          variant.native_lowp ? q_arr.dtype() : mx::float32);
    } catch (...) {
      if (!variant.native_lowp) {
        throw;
      }
      qf = to_float32(q_arr);
      kf = to_float32(k_arr);
      return launch_one(
          "na3d_qk_fp32",
          {qf, kf},
          p,
          to_shape({B, out_d, out_h, out_w, H, K * K * K}),
          false);
    }
  }();
  return nb::cast(cast_to_dtype(out, q_arr.dtype()));
}

nb::object na3d_av_forward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  throw_if_forced_split_failure();
  if (!supports_3d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 3D AV unsupported configuration");
  }

  auto attn_arr = as_array(attn);
  auto v_arr = as_array(v);
  SplitForwardVariant variant = split_forward_variant(v_arr);
  mx::array af = attn_arr;
  mx::array vf = v_arr;
  if (!variant.native_lowp) {
    af = to_float32(attn_arr);
    vf = to_float32(v_arr);
  } else {
    mx::Dtype target = v_arr.dtype();
    if (attn_arr.dtype() != target) {
      af = cast_to_dtype(attn_arr, target);
    }
  }

  int B = vf.shape(0);
  int ID = vf.shape(1);
  int IH = vf.shape(2);
  int IW = vf.shape(3);
  int H = vf.shape(4);
  int D = vf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;

  int out_d = af.shape(1);
  int out_h = af.shape(2);
  int out_w = af.shape(3);

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, 1.0f};
  bool use_vec4 = !variant.native_lowp && direct_vec4_eligible(D) && SD == 1 && SH == 1 &&
      SW == 1 && DD == 1 &&
      DH == 1 && DW == 1 && CD == 0 && CH == 0 && CW == 0;
  bool use_k3_fast = !variant.native_lowp && K == 3 && SD == 1 && SH == 1 && SW == 1 && DD == 1 &&
      DH == 1 && DW == 1 && CD == 0 && CH == 0 && CW == 0;
  auto out_shape = to_shape({B, out_d, out_h, out_w, H, D});
  std::string av_kernel = std::string("na3d_av_") + variant.dtype_tag;
  auto out = [&]() -> mx::array {
    if (use_vec4) {
      if (use_k3_fast) {
        try {
          av_kernel = "na3d_av_k3_vec4_fp32";
          return launch_one(
              "na3d_av_k3_vec4_fp32",
              {af, vf},
              p,
              out_shape,
              false,
              std::max<size_t>(1, numel(out_shape) / 4));
        } catch (...) {
        }
      }
      try {
        av_kernel = "na3d_av_vec4_fp32";
        return launch_one(
            "na3d_av_vec4_fp32",
            {af, vf},
            p,
            out_shape,
            false,
            std::max<size_t>(1, numel(out_shape) / 4));
      } catch (...) {
      }
    }
    std::string kname = std::string("na3d_av_") + variant.dtype_tag;
    try {
      av_kernel = kname;
      return launch_one(
          kname,
          {af, vf},
          p,
          out_shape,
          false,
          0,
          variant.native_lowp ? v_arr.dtype() : mx::float32);
    } catch (...) {
      if (!variant.native_lowp) {
        if (use_k3_fast) {
          av_kernel = "na3d_av_k3_fp32";
          return launch_one("na3d_av_k3_fp32", {af, vf}, p, out_shape, false);
        }
        throw;
      }
      af = to_float32(attn_arr);
      vf = to_float32(v_arr);
      av_kernel = "na3d_av_fp32";
      return launch_one("na3d_av_fp32", {af, vf}, p, out_shape, false);
    }
  }();
  debug_set_last_kernel("na3d_av_forward", av_kernel);
  return nb::cast(cast_to_dtype(out, v_arr.dtype()));
}

nb::object na1d_qk_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_attn,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_split_failure();
  if (!supports_1d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 1D QK backward unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto g_arr = as_array(grad_attn);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto gf = to_float32(g_arr);

  int B = qf.shape(0);
  int L = qf.shape(1);
  int H = qf.shape(2);
  int D = qf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int out_l = ceil_div(L, S);

  NA1DParams p{B, L, H, D, K, S, Dil, C, resolve_scale(scale, D)};
  auto grad_q = launch_one("na1d_qk_bwd_q_fp32", {gf, kf}, p, qf.shape(), false);
  bool use_direct = can_use_direct_nonatomic_1d("na1d_qk_backward", "fp32", D, L, K, S, Dil, C) ||
      can_use_direct_causal_1d("na1d_qk_backward", "fp32", D, L, K, S, Dil, C);
  bool use_vec4 = use_direct && direct_vec4_eligible(D);
  std::string grad_k_kernel = "na1d_qk_bwd_k_accum_fp32";
  mx::array grad_k = [&]() {
    if (use_direct) {
      grad_k_kernel = choose_na1d_qk_bwd_k_direct_kernel(C, use_vec4);
      return launch_one(
          grad_k_kernel,
          {gf, qf},
          p,
          kf.shape(),
          false,
          use_vec4 ? (numel(kf.shape()) / 4) : numel(kf.shape()));
    }
    size_t grad_k_threads = static_cast<size_t>(B) * static_cast<size_t>(out_l) *
        static_cast<size_t>(H) * static_cast<size_t>(K) * static_cast<size_t>(D);
    return launch_one("na1d_qk_bwd_k_accum_fp32", {gf, qf}, p, kf.shape(), true, grad_k_threads);
  }();
  debug_set_last_kernel("na1d_qk_backward", grad_k_kernel);
  return nb::make_tuple(cast_to_dtype(grad_q, q_arr.dtype()), cast_to_dtype(grad_k, k_arr.dtype()));
}

nb::object na1d_av_backward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  throw_if_forced_split_failure();
  if (!supports_1d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 1D AV backward unsupported configuration");
  }

  auto a_arr = as_array(attn);
  auto v_arr = as_array(v);
  auto go_arr = as_array(grad_out);
  auto af = to_float32(a_arr);
  auto vf = to_float32(v_arr);
  auto gof = to_float32(go_arr);

  int B = vf.shape(0);
  int L = vf.shape(1);
  int H = vf.shape(2);
  int D = vf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int out_l = af.shape(1);

  NA1DParams p{B, L, H, D, K, S, Dil, C, 1.0f};
  auto grad_attn = launch_one("na1d_av_bwd_attn_fp32", {gof, vf}, p, af.shape(), false);
  bool use_direct = can_use_direct_nonatomic_1d("na1d_av_backward", "fp32", D, L, K, S, Dil, C) ||
      can_use_direct_causal_1d("na1d_av_backward", "fp32", D, L, K, S, Dil, C);
  bool use_vec4 = use_direct && direct_vec4_eligible(D);
  std::string grad_v_kernel = "na1d_av_bwd_v_accum_fp32";
  mx::array grad_v = [&]() {
    if (use_direct) {
      grad_v_kernel = choose_na1d_av_bwd_v_direct_kernel(C, use_vec4);
      return launch_one(
          grad_v_kernel,
          {af, gof},
          p,
          vf.shape(),
          false,
          use_vec4 ? (numel(vf.shape()) / 4) : numel(vf.shape()));
    }
    size_t grad_v_threads = static_cast<size_t>(B) * static_cast<size_t>(out_l) *
        static_cast<size_t>(H) * static_cast<size_t>(K) * static_cast<size_t>(D);
    return launch_one("na1d_av_bwd_v_accum_fp32", {af, gof}, p, vf.shape(), true, grad_v_threads);
  }();
  debug_set_last_kernel("na1d_av_backward", grad_v_kernel);
  return nb::make_tuple(cast_to_dtype(grad_attn, a_arr.dtype()), cast_to_dtype(grad_v, v_arr.dtype()));
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
  throw_if_forced_split_failure();
  if (!supports_2d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 2D QK backward unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto g_arr = as_array(grad_attn);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto gf = to_float32(g_arr);

  int B = qf.shape(0);
  int IH = qf.shape(1);
  int IW = qf.shape(2);
  int H = qf.shape(3);
  int D = qf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int out_h = ceil_div(IH, SH);
  int out_w = ceil_div(IW, SW);

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, resolve_scale(scale, D)};
  bool use_vec4_q = direct_vec4_eligible(D);
  if (const char* v = std::getenv("NATTEN_NANOBIND_QK_BWD_Q_VEC4")) {
    use_vec4_q = use_vec4_q && (std::string(v) == "1");
  } else {
    use_vec4_q = false;
  }
  std::string grad_q_kernel = use_vec4_q ? "na2d_qk_bwd_q_vec4_fp32" : "na2d_qk_bwd_q_fp32";
  auto grad_q = launch_one(
      grad_q_kernel,
      {gf, kf},
      p,
      qf.shape(),
      false,
      use_vec4_q ? (numel(qf.shape()) / 4) : 0);
  bool use_direct =
      can_use_direct_nonatomic_2d("na2d_qk_backward", "fp32", D, IH, IW, K, SH, SW, DH, DW, CH, CW);
  bool use_vec4 = use_direct && direct_vec4_eligible(D);
  std::string grad_k_kernel = "na2d_qk_bwd_k_accum_fp32";
  mx::array grad_k = [&]() {
    if (use_direct) {
      grad_k_kernel = choose_na2d_qk_bwd_k_direct_kernel(K, use_vec4);
      return launch_one(
          grad_k_kernel,
          {gf, qf},
          p,
          kf.shape(),
          false,
          use_vec4 ? (numel(kf.shape()) / 4) : numel(kf.shape()));
    }
    size_t grad_k_threads = static_cast<size_t>(B) * static_cast<size_t>(out_h) *
        static_cast<size_t>(out_w) * static_cast<size_t>(H) * static_cast<size_t>(K) *
        static_cast<size_t>(K) * static_cast<size_t>(D);
    return launch_one("na2d_qk_bwd_k_accum_fp32", {gf, qf}, p, kf.shape(), true, grad_k_threads);
  }();
  debug_set_last_kernel("na2d_qk_backward_grad_q", grad_q_kernel);
  debug_set_last_kernel("na2d_qk_backward", grad_k_kernel);
  return nb::make_tuple(cast_to_dtype(grad_q, q_arr.dtype()), cast_to_dtype(grad_k, k_arr.dtype()));
}

nb::object na2d_av_backward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  throw_if_forced_split_failure();
  if (!supports_2d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 2D AV backward unsupported configuration");
  }

  auto a_arr = as_array(attn);
  auto v_arr = as_array(v);
  auto go_arr = as_array(grad_out);
  auto af = to_float32(a_arr);
  auto vf = to_float32(v_arr);
  auto gof = to_float32(go_arr);

  int B = vf.shape(0);
  int IH = vf.shape(1);
  int IW = vf.shape(2);
  int H = vf.shape(3);
  int D = vf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int out_h = af.shape(1);
  int out_w = af.shape(2);

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, 1.0f};
  auto grad_attn = launch_one("na2d_av_bwd_attn_fp32", {gof, vf}, p, af.shape(), false);
  bool use_direct =
      can_use_direct_nonatomic_2d("na2d_av_backward", "fp32", D, IH, IW, K, SH, SW, DH, DW, CH, CW);
  bool use_vec4 = use_direct && direct_vec4_eligible(D);
  std::string grad_v_kernel = "na2d_av_bwd_v_accum_fp32";
  mx::array grad_v = [&]() {
    if (use_direct) {
      grad_v_kernel = use_vec4 ? "na2d_av_bwd_v_direct_u1d1_nc_vec4_fp32"
                               : "na2d_av_bwd_v_direct_u1d1_nc_fp32";
      return launch_one(
          grad_v_kernel,
          {af, gof},
          p,
          vf.shape(),
          false,
          use_vec4 ? (numel(vf.shape()) / 4) : numel(vf.shape()));
    }
    size_t grad_v_threads = static_cast<size_t>(B) * static_cast<size_t>(out_h) *
        static_cast<size_t>(out_w) * static_cast<size_t>(H) * static_cast<size_t>(K) *
        static_cast<size_t>(K) * static_cast<size_t>(D);
    return launch_one("na2d_av_bwd_v_accum_fp32", {af, gof}, p, vf.shape(), true, grad_v_threads);
  }();
  debug_set_last_kernel("na2d_av_backward", grad_v_kernel);
  return nb::make_tuple(cast_to_dtype(grad_attn, a_arr.dtype()), cast_to_dtype(grad_v, v_arr.dtype()));
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
  throw_if_forced_split_failure();
  if (!supports_3d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 3D QK backward unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto g_arr = as_array(grad_attn);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto gf = to_float32(g_arr);

  int B = qf.shape(0);
  int ID = qf.shape(1);
  int IH = qf.shape(2);
  int IW = qf.shape(3);
  int H = qf.shape(4);
  int D = qf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;
  int out_d = ceil_div(ID, SD);
  int out_h = ceil_div(IH, SH);
  int out_w = ceil_div(IW, SW);

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, resolve_scale(scale, D)};
  bool use_vec4_q = direct_vec4_eligible(D);
  std::string grad_q_kernel = use_vec4_q ? "na3d_qk_bwd_q_vec4_fp32" : "na3d_qk_bwd_q_fp32";
  auto grad_q = launch_one(
      grad_q_kernel,
      {gf, kf},
      p,
      qf.shape(),
          false,
          use_vec4_q ? (numel(qf.shape()) / 4) : numel(qf.shape()));
  bool use_direct = can_use_direct_nonatomic_3d(
      "na3d_qk_backward", "fp32", D, ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW);
  use_direct = use_direct || force_direct_nonatomic_3d_hotshape(
                               D, ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW);
  bool use_vec4 = use_direct && direct_vec4_eligible(D);
  std::string grad_k_kernel = "na3d_qk_bwd_k_accum_fp32";
  mx::array grad_k = [&]() {
    if (use_direct) {
      grad_k_kernel = choose_na3d_qk_bwd_k_direct_kernel(K, use_vec4);
      return launch_one(
          grad_k_kernel,
          {gf, qf},
          p,
          kf.shape(),
          false,
          use_vec4 ? (numel(kf.shape()) / 4) : numel(kf.shape()));
    }
    size_t grad_k_threads = static_cast<size_t>(B) * static_cast<size_t>(out_d) *
        static_cast<size_t>(out_h) * static_cast<size_t>(out_w) * static_cast<size_t>(H) *
        static_cast<size_t>(K) * static_cast<size_t>(K) * static_cast<size_t>(K) *
        static_cast<size_t>(D);
    return launch_one("na3d_qk_bwd_k_accum_fp32", {gf, qf}, p, kf.shape(), true, grad_k_threads);
  }();
  debug_set_last_kernel("na3d_qk_backward_grad_q", grad_q_kernel);
  debug_set_last_kernel("na3d_qk_backward", grad_k_kernel);
  return nb::make_tuple(cast_to_dtype(grad_q, q_arr.dtype()), cast_to_dtype(grad_k, k_arr.dtype()));
}

nb::object na3d_av_backward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  throw_if_forced_split_failure();
  if (!supports_3d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 3D AV backward unsupported configuration");
  }

  auto a_arr = as_array(attn);
  auto v_arr = as_array(v);
  auto go_arr = as_array(grad_out);
  auto af = to_float32(a_arr);
  auto vf = to_float32(v_arr);
  auto gof = to_float32(go_arr);

  int B = vf.shape(0);
  int ID = vf.shape(1);
  int IH = vf.shape(2);
  int IW = vf.shape(3);
  int H = vf.shape(4);
  int D = vf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;
  int out_d = af.shape(1);
  int out_h = af.shape(2);
  int out_w = af.shape(3);

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, 1.0f};
  bool use_vec4_attn = direct_vec4_eligible(D);
  std::string grad_attn_kernel =
      use_vec4_attn ? "na3d_av_bwd_attn_vec4_fp32" : "na3d_av_bwd_attn_fp32";
  auto grad_attn = launch_one(
      grad_attn_kernel,
      {gof, vf},
      p,
      af.shape(),
      false);
  bool use_direct = can_use_direct_nonatomic_3d(
      "na3d_av_backward", "fp32", D, ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW);
  use_direct = use_direct || force_direct_nonatomic_3d_hotshape(
                               D, ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW);
  bool use_vec4 = use_direct && direct_vec4_eligible(D);
  std::string grad_v_kernel = "na3d_av_bwd_v_accum_fp32";
  mx::array grad_v = [&]() {
    if (use_direct) {
      grad_v_kernel = use_vec4 ? "na3d_av_bwd_v_direct_u1d1_nc_vec4_fp32"
                               : "na3d_av_bwd_v_direct_u1d1_nc_fp32";
      return launch_one(
          grad_v_kernel,
          {af, gof},
          p,
          vf.shape(),
          false,
          use_vec4 ? (numel(vf.shape()) / 4) : numel(vf.shape()));
    }
    size_t grad_v_threads = static_cast<size_t>(B) * static_cast<size_t>(out_d) *
        static_cast<size_t>(out_h) * static_cast<size_t>(out_w) * static_cast<size_t>(H) *
        static_cast<size_t>(K) * static_cast<size_t>(K) * static_cast<size_t>(K) *
        static_cast<size_t>(D);
    grad_v_kernel = "na3d_av_bwd_v_accum_fp32";
    return launch_one("na3d_av_bwd_v_accum_fp32", {af, gof}, p, vf.shape(), true, grad_v_threads);
  }();
  debug_set_last_kernel("na3d_av_backward_grad_attn", grad_attn_kernel);
  debug_set_last_kernel("na3d_av_backward", grad_v_kernel);
  return nb::make_tuple(cast_to_dtype(grad_attn, a_arr.dtype()), cast_to_dtype(grad_v, v_arr.dtype()));
}

nb::object na1d_fused_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_fused_failure();
  if (!supports_1d_fused(kernel_size, stride, dilation)) {
    throw std::runtime_error("nanobind fused 1D unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto v_arr = as_array(v);
  VariantConfig variant = fused_forward_variant(q_arr, true);

  mx::array qx = q_arr;
  mx::array kx = k_arr;
  mx::array vx = v_arr;
  if (variant.dtype_tag == "fp32") {
    qx = to_float32(q_arr);
    kx = to_float32(k_arr);
    vx = to_float32(v_arr);
  } else {
    mx::Dtype target = q_arr.dtype();
    if (k_arr.dtype() != target) {
      kx = cast_to_dtype(k_arr, target);
    }
    if (v_arr.dtype() != target) {
      vx = cast_to_dtype(v_arr, target);
    }
  }

  int B = qx.shape(0);
  int L = qx.shape(1);
  int H = qx.shape(2);
  int D = qx.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int out_l = ceil_div(L, S);
  bool use_causal_k9_vec4 = variant.vec4 && (C == 1) && (K == 9) && (S == 1) && (Dil == 1);
  bool use_causal_k9d_vec4_fp32 =
      variant.vec4 && (variant.dtype_tag == "fp32") && (C == 1) && (K == 9) && (S == 1) && (Dil >= 1);

  NA1DParams p{B, L, H, D, K, S, Dil, C, resolve_scale(scale, D)};
  std::string generic_kname =
      std::string("na1d_fused") + (variant.vec4 ? "_vec4_" : "_") + variant.dtype_tag;
  std::string kname = use_causal_k9d_vec4_fp32
      ? "na1d_fused_causal_k9d_vec4_fp32"
      : (use_causal_k9_vec4
              ? (std::string("na1d_fused_causal_k9_vec4_") + variant.dtype_tag)
              : generic_kname);
  auto tuned = lookup_threadgroup_from_tuning(
      "na1d_fused",
      variant.dtype_tag,
      out_l,
      D,
      K,
      causal_rank_1d(C),
      (S == 1 && Dil == 1));
  size_t tgx = tuned.has_value() ? std::get<0>(*tuned) : 128;
  size_t tgy = tuned.has_value() ? std::get<1>(*tuned) : 1;
  size_t tgz = tuned.has_value() ? std::get<2>(*tuned) : 1;
  auto launch = make_launch(
      static_cast<size_t>(out_l),
      1,
      static_cast<size_t>(B * H),
      tgx,
      tgy,
      tgz);
  auto out = [&]() -> mx::array {
    try {
      return launch_one_cfg(
          kname,
          {qx, kx, vx},
          p,
          to_shape({B, out_l, H, D}),
          launch,
          false,
          (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
    } catch (...) {
      if (kname != generic_kname) {
        try {
          kname = generic_kname;
          return launch_one_cfg(
              kname,
              {qx, kx, vx},
              p,
              to_shape({B, out_l, H, D}),
              launch,
              false,
              (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
        } catch (...) {
        }
      }
      if (variant.dtype_tag == "fp32") {
        throw;
      }
      qx = to_float32(q_arr);
      kx = to_float32(k_arr);
      vx = to_float32(v_arr);
      variant.dtype_tag = "fp32";
      generic_kname = std::string("na1d_fused") + (variant.vec4 ? "_vec4_" : "_") + variant.dtype_tag;
      kname = use_causal_k9d_vec4_fp32
          ? "na1d_fused_causal_k9d_vec4_fp32"
          : (use_causal_k9_vec4
                  ? (std::string("na1d_fused_causal_k9_vec4_") + variant.dtype_tag)
                  : generic_kname);
      auto tuned_fp32 = lookup_threadgroup_from_tuning(
          "na1d_fused",
          variant.dtype_tag,
          out_l,
          D,
          K,
          causal_rank_1d(C),
          (S == 1 && Dil == 1));
      size_t tgx_fp32 = tuned_fp32.has_value() ? std::get<0>(*tuned_fp32) : 128;
      size_t tgy_fp32 = tuned_fp32.has_value() ? std::get<1>(*tuned_fp32) : 1;
      size_t tgz_fp32 = tuned_fp32.has_value() ? std::get<2>(*tuned_fp32) : 1;
      auto launch_fp32 = make_launch(
          static_cast<size_t>(out_l),
          1,
          static_cast<size_t>(B * H),
          tgx_fp32,
          tgy_fp32,
          tgz_fp32);
      try {
        return launch_one_cfg(
            kname,
            {qx, kx, vx},
            p,
            to_shape({B, out_l, H, D}),
            launch_fp32,
            false,
            (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
      } catch (...) {
        if (kname == generic_kname) {
          throw;
        }
        kname = generic_kname;
        return launch_one_cfg(
            kname,
            {qx, kx, vx},
            p,
            to_shape({B, out_l, H, D}),
            launch_fp32,
            false,
            (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
      }
    }
  }();
  debug_set_last_kernel("na1d_forward", kname);
  return nb::cast(cast_to_dtype(out, q_arr.dtype()));
}

nb::object na2d_fused_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_fused_failure();
  if (!supports_2d_fused(kernel_size, stride, dilation)) {
    throw std::runtime_error("nanobind fused 2D unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto v_arr = as_array(v);
  VariantConfig variant = fused_forward_variant(q_arr, true);

  mx::array qx = q_arr;
  mx::array kx = k_arr;
  mx::array vx = v_arr;
  if (variant.dtype_tag == "fp32") {
    qx = to_float32(q_arr);
    kx = to_float32(k_arr);
    vx = to_float32(v_arr);
  } else {
    mx::Dtype target = q_arr.dtype();
    if (k_arr.dtype() != target) {
      kx = cast_to_dtype(k_arr, target);
    }
    if (v_arr.dtype() != target) {
      vx = cast_to_dtype(v_arr, target);
    }
  }

  int B = qx.shape(0);
  int IH = qx.shape(1);
  int IW = qx.shape(2);
  int H = qx.shape(3);
  int D = qx.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int out_h = ceil_div(IH, SH);
  int out_w = ceil_div(IW, SW);

  if (prefer_split_composed_fwd_2d(IH, IW, K, SH, SW, DH, DW, CH, CW)) {
    throw std::runtime_error("nanobind fused 2D forward prefers split-composed path");
  }

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, resolve_scale(scale, D)};
  bool use_bf16_strided_causal_h_opt =
      (variant.dtype_tag == "bf16") && (D == 16) && (K == 7) && (SH == 2) && (SW == 1) &&
      (DH == 1) && (DW == 2) && (CH == 1) && (CW == 0);
  if (use_bf16_strided_causal_h_opt) {
    std::string kname_opt = "na2d_fused_strided_causal_h_k7d16_bf16";
    auto tuned_opt = lookup_threadgroup_from_tuning(
        "na2d_fused_strided_causal_h",
        variant.dtype_tag,
        out_h * out_w,
        D,
        K,
        causal_rank_2d(CH, CW),
        false);
    size_t tgx_opt = tuned_opt.has_value() ? std::get<0>(*tuned_opt) : 16;
    size_t tgy_opt = tuned_opt.has_value() ? std::get<1>(*tuned_opt) : 8;
    size_t tgz_opt = tuned_opt.has_value() ? std::get<2>(*tuned_opt) : 1;
    auto launch_opt = make_launch(
        static_cast<size_t>(out_w),
        static_cast<size_t>(out_h),
        static_cast<size_t>(B * H),
        tgx_opt,
        tgy_opt,
        tgz_opt);
    try {
      auto out_opt = launch_one_cfg(
          kname_opt,
          {qx, kx, vx},
          p,
          to_shape({B, out_h, out_w, H, D}),
          launch_opt,
          false,
          q_arr.dtype());
      debug_set_last_kernel("na2d_forward", kname_opt);
      return nb::cast(cast_to_dtype(out_opt, q_arr.dtype()));
    } catch (...) {
      // Fall through to the generic fused paths.
    }
  }

  std::string strategy = choose_softmax_strategy_from_tuning(
      "na2d_fused",
      variant.dtype_tag,
      out_h * out_w,
      K,
      causal_rank_2d(CH, CW),
      (SH == 1 && SW == 1 && DH == 1 && DW == 1));
  std::string kname = std::string("na2d_fused_") + strategy + (variant.vec4 ? "_vec4_" : "_") + variant.dtype_tag;
  auto tuned = lookup_threadgroup_from_tuning(
      "na2d_fused",
      variant.dtype_tag,
      out_h * out_w,
      D,
      K,
      causal_rank_2d(CH, CW),
      (SH == 1 && SW == 1 && DH == 1 && DW == 1));
  size_t tgx = tuned.has_value() ? std::get<0>(*tuned) : 16;
  size_t tgy = tuned.has_value() ? std::get<1>(*tuned) : 8;
  size_t tgz = tuned.has_value() ? std::get<2>(*tuned) : 1;
  auto launch = make_launch(
      static_cast<size_t>(out_w),
      static_cast<size_t>(out_h),
      static_cast<size_t>(B * H),
      tgx,
      tgy,
      tgz);
  auto out = [&]() -> mx::array {
    try {
      return launch_one_cfg(
          kname,
          {qx, kx, vx},
          p,
          to_shape({B, out_h, out_w, H, D}),
          launch,
          false,
          (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
    } catch (...) {
      try {
        std::string fallback =
            std::string("na2d_fused") + (variant.vec4 ? "_vec4_" : "_") + variant.dtype_tag;
        kname = fallback;
        return launch_one_cfg(
            fallback,
            {qx, kx, vx},
            p,
            to_shape({B, out_h, out_w, H, D}),
            launch,
            false,
            (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
      } catch (...) {
        if (variant.dtype_tag == "fp32") {
          throw;
        }
        qx = to_float32(q_arr);
        kx = to_float32(k_arr);
        vx = to_float32(v_arr);
        variant.dtype_tag = "fp32";
        std::string strategy_fp32 = choose_softmax_strategy_from_tuning(
            "na2d_fused",
            variant.dtype_tag,
            out_h * out_w,
            K,
            causal_rank_2d(CH, CW),
            (SH == 1 && SW == 1 && DH == 1 && DW == 1));
        kname = std::string("na2d_fused_") + strategy_fp32 + (variant.vec4 ? "_vec4_" : "_") +
            variant.dtype_tag;
        auto tuned_fp32 = lookup_threadgroup_from_tuning(
            "na2d_fused",
            variant.dtype_tag,
            out_h * out_w,
            D,
            K,
            causal_rank_2d(CH, CW),
            (SH == 1 && SW == 1 && DH == 1 && DW == 1));
        size_t tgx_fp32 = tuned_fp32.has_value() ? std::get<0>(*tuned_fp32) : 16;
        size_t tgy_fp32 = tuned_fp32.has_value() ? std::get<1>(*tuned_fp32) : 8;
        size_t tgz_fp32 = tuned_fp32.has_value() ? std::get<2>(*tuned_fp32) : 1;
        auto launch_fp32 = make_launch(
            static_cast<size_t>(out_w),
            static_cast<size_t>(out_h),
            static_cast<size_t>(B * H),
            tgx_fp32,
            tgy_fp32,
            tgz_fp32);
        try {
          return launch_one_cfg(
              kname,
              {qx, kx, vx},
              p,
              to_shape({B, out_h, out_w, H, D}),
              launch_fp32,
              false,
              (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
        } catch (...) {
          std::string fallback_fp32 =
              std::string("na2d_fused") + (variant.vec4 ? "_vec4_" : "_") + variant.dtype_tag;
          kname = fallback_fp32;
          return launch_one_cfg(
              fallback_fp32,
              {qx, kx, vx},
              p,
              to_shape({B, out_h, out_w, H, D}),
              launch_fp32,
              false,
              (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
        }
      }
    }
  }();
  debug_set_last_kernel("na2d_forward", kname);
  return nb::cast(cast_to_dtype(out, q_arr.dtype()));
}

nb::object na3d_fused_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_fused_failure();
  if (!supports_3d_fused(kernel_size, stride, dilation)) {
    throw std::runtime_error("nanobind fused 3D unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto v_arr = as_array(v);
  VariantConfig variant = fused_forward_variant(q_arr, true);

  mx::array qx = q_arr;
  mx::array kx = k_arr;
  mx::array vx = v_arr;
  if (variant.dtype_tag == "fp32") {
    qx = to_float32(q_arr);
    kx = to_float32(k_arr);
    vx = to_float32(v_arr);
  } else {
    mx::Dtype target = q_arr.dtype();
    if (k_arr.dtype() != target) {
      kx = cast_to_dtype(k_arr, target);
    }
    if (v_arr.dtype() != target) {
      vx = cast_to_dtype(v_arr, target);
    }
  }

  int B = qx.shape(0);
  int ID = qx.shape(1);
  int IH = qx.shape(2);
  int IW = qx.shape(3);
  int H = qx.shape(4);
  int D = qx.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;
  int out_d = ceil_div(ID, SD);
  int out_h = ceil_div(IH, SH);
  int out_w = ceil_div(IW, SW);
  bool use_causal_d_k3_vec4 = variant.vec4 && (K == 3) && (CD == 1) && (CH == 0) && (CW == 0) &&
      (SD == 1) && (SH == 1) && (SW == 1) && (DD == 1) && (DH == 1) && (DW == 1);

  if (prefer_split_composed_fwd_3d(ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW)) {
    throw std::runtime_error("nanobind fused 3D forward prefers split-composed path");
  }

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, resolve_scale(scale, D)};
  std::string strategy = choose_softmax_strategy_from_tuning(
      "na3d_fused",
      variant.dtype_tag,
      out_d * out_h * out_w,
      K,
      causal_rank_3d(CD, CH, CW),
      (SD == 1 && SH == 1 && SW == 1 && DD == 1 && DH == 1 && DW == 1));
  std::string strategy_kname =
      std::string("na3d_fused_") + strategy + (variant.vec4 ? "_vec4_" : "_") + variant.dtype_tag;
  std::string fallback_kname =
      std::string("na3d_fused") + (variant.vec4 ? "_vec4_" : "_") + variant.dtype_tag;
  std::string kname = use_causal_d_k3_vec4
      ? (std::string("na3d_fused_causal_d_k3_vec4_") + variant.dtype_tag)
      : strategy_kname;
  auto tuned = lookup_threadgroup_from_tuning(
      "na3d_fused",
      variant.dtype_tag,
      out_d * out_h * out_w,
      D,
      K,
      causal_rank_3d(CD, CH, CW),
      (SD == 1 && SH == 1 && SW == 1 && DD == 1 && DH == 1 && DW == 1));
  size_t tgx = tuned.has_value() ? std::get<0>(*tuned) : 8;
  size_t tgy = tuned.has_value() ? std::get<1>(*tuned) : 8;
  size_t tgz = tuned.has_value() ? std::get<2>(*tuned) : 1;
  auto launch = make_launch(
      static_cast<size_t>(out_w),
      static_cast<size_t>(out_h),
      static_cast<size_t>(B * H * out_d),
      tgx,
      tgy,
      tgz);
  auto out = [&]() -> mx::array {
    try {
      return launch_one_cfg(
          kname,
          {qx, kx, vx},
          p,
          to_shape({B, out_d, out_h, out_w, H, D}),
          launch,
          false,
          (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
    } catch (...) {
      try {
        kname = strategy_kname;
        return launch_one_cfg(
            strategy_kname,
            {qx, kx, vx},
            p,
            to_shape({B, out_d, out_h, out_w, H, D}),
            launch,
            false,
            (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
      } catch (...) {
        try {
          kname = fallback_kname;
          return launch_one_cfg(
              fallback_kname,
              {qx, kx, vx},
              p,
              to_shape({B, out_d, out_h, out_w, H, D}),
              launch,
              false,
              (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
        } catch (...) {
        }
        if (variant.dtype_tag == "fp32") {
          throw;
        }
        qx = to_float32(q_arr);
        kx = to_float32(k_arr);
        vx = to_float32(v_arr);
        variant.dtype_tag = "fp32";
        std::string strategy_fp32 = choose_softmax_strategy_from_tuning(
            "na3d_fused",
            variant.dtype_tag,
            out_d * out_h * out_w,
            K,
            causal_rank_3d(CD, CH, CW),
            (SD == 1 && SH == 1 && SW == 1 && DD == 1 && DH == 1 && DW == 1));
        strategy_kname =
            std::string("na3d_fused_") + strategy_fp32 + (variant.vec4 ? "_vec4_" : "_") +
            variant.dtype_tag;
        fallback_kname = std::string("na3d_fused") + (variant.vec4 ? "_vec4_" : "_") + variant.dtype_tag;
        kname = use_causal_d_k3_vec4
            ? (std::string("na3d_fused_causal_d_k3_vec4_") + variant.dtype_tag)
            : strategy_kname;
        auto tuned_fp32 = lookup_threadgroup_from_tuning(
            "na3d_fused",
            variant.dtype_tag,
            out_d * out_h * out_w,
            D,
            K,
            causal_rank_3d(CD, CH, CW),
            (SD == 1 && SH == 1 && SW == 1 && DD == 1 && DH == 1 && DW == 1));
        size_t tgx_fp32 = tuned_fp32.has_value() ? std::get<0>(*tuned_fp32) : 8;
        size_t tgy_fp32 = tuned_fp32.has_value() ? std::get<1>(*tuned_fp32) : 8;
        size_t tgz_fp32 = tuned_fp32.has_value() ? std::get<2>(*tuned_fp32) : 1;
        auto launch_fp32 = make_launch(
            static_cast<size_t>(out_w),
            static_cast<size_t>(out_h),
            static_cast<size_t>(B * H * out_d),
            tgx_fp32,
            tgy_fp32,
            tgz_fp32);
        try {
          return launch_one_cfg(
              kname,
              {qx, kx, vx},
              p,
              to_shape({B, out_d, out_h, out_w, H, D}),
              launch_fp32,
              false,
              (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
        } catch (...) {
          try {
            kname = strategy_kname;
            return launch_one_cfg(
                strategy_kname,
                {qx, kx, vx},
                p,
                to_shape({B, out_d, out_h, out_w, H, D}),
                launch_fp32,
                false,
                (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
          } catch (...) {
            kname = fallback_kname;
            return launch_one_cfg(
                fallback_kname,
                {qx, kx, vx},
                p,
                to_shape({B, out_d, out_h, out_w, H, D}),
                launch_fp32,
                false,
                (variant.dtype_tag == "fp32") ? mx::float32 : q_arr.dtype());
          }
        }
      }
    }
  }();
  debug_set_last_kernel("na3d_forward", kname);
  return nb::cast(cast_to_dtype(out, q_arr.dtype()));
}

nb::object na1d_fused_backward_attn(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto v_arr = as_array(v);
  auto go_arr = as_array(grad_out);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto vf = to_float32(v_arr);
  auto gof = to_float32(go_arr);

  int B = qf.shape(0);
  int L = qf.shape(1);
  int H = qf.shape(2);
  int D = qf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int out_l = ceil_div(L, S);

  NA1DParams p{B, L, H, D, K, S, Dil, C, resolve_scale(scale, D)};
  bool use_vec4 = direct_vec4_eligible(D);
  std::string attn_kernel = use_vec4 ? "na1d_fused_bwd_attn_vec4_fp32" : "na1d_fused_bwd_attn_fp32";
  if (C != 0 && S == 1 && K == 9) {
    attn_kernel = use_vec4 ? "na1d_fused_bwd_attn_s1_causal_k9_vec4_fp32"
                           : "na1d_fused_bwd_attn_s1_causal_k9_fp32";
  }
  size_t attn_threads = static_cast<size_t>(B) * static_cast<size_t>(out_l) * static_cast<size_t>(H);
  nb::tuple out = launch_two(
      attn_kernel,
      {qf, kf, vf, gof},
      p,
      to_shape({B, out_l, H, K}),
      to_shape({B, out_l, H, K}),
      false,
      attn_threads);
  debug_set_last_kernel("na1d_fused_backward_attn", attn_kernel);
  return out;
}

nb::object na2d_fused_backward_attn(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto v_arr = as_array(v);
  auto go_arr = as_array(grad_out);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto vf = to_float32(v_arr);
  auto gof = to_float32(go_arr);

  int B = qf.shape(0);
  int IH = qf.shape(1);
  int IW = qf.shape(2);
  int H = qf.shape(3);
  int D = qf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int out_h = ceil_div(IH, SH);
  int out_w = ceil_div(IW, SW);
  int k2 = K * K;

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, resolve_scale(scale, D)};
  size_t attn_threads = static_cast<size_t>(B) * static_cast<size_t>(out_h) *
      static_cast<size_t>(out_w) * static_cast<size_t>(H);
  nb::tuple out = launch_two(
      "na2d_fused_bwd_attn_fp32",
      {qf, kf, vf, gof},
      p,
      to_shape({B, out_h, out_w, H, k2}),
      to_shape({B, out_h, out_w, H, k2}),
      false,
      attn_threads);
  debug_set_last_kernel("na2d_fused_backward_attn", "na2d_fused_bwd_attn_fp32");
  return out;
}

nb::object na3d_fused_backward_attn(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto v_arr = as_array(v);
  auto go_arr = as_array(grad_out);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto vf = to_float32(v_arr);
  auto gof = to_float32(go_arr);

  int B = qf.shape(0);
  int ID = qf.shape(1);
  int IH = qf.shape(2);
  int IW = qf.shape(3);
  int H = qf.shape(4);
  int D = qf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;
  int out_d = ceil_div(ID, SD);
  int out_h = ceil_div(IH, SH);
  int out_w = ceil_div(IW, SW);
  int k3 = K * K * K;

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, resolve_scale(scale, D)};
  size_t attn_threads = static_cast<size_t>(B) * static_cast<size_t>(out_d) *
      static_cast<size_t>(out_h) * static_cast<size_t>(out_w) * static_cast<size_t>(H);
  nb::tuple out = launch_two(
      "na3d_fused_bwd_attn_fp32",
      {qf, kf, vf, gof},
      p,
      to_shape({B, out_d, out_h, out_w, H, k3}),
      to_shape({B, out_d, out_h, out_w, H, k3}),
      false,
      attn_threads);
  debug_set_last_kernel("na3d_fused_backward_attn", "na3d_fused_bwd_attn_fp32");
  return out;
}

nb::object na1d_fused_backward_qk(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_logits,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto gl_arr = as_array(grad_logits);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto glf = to_float32(gl_arr);

  int B = qf.shape(0);
  int L = qf.shape(1);
  int H = qf.shape(2);
  int D = qf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int out_l = glf.shape(1);

  NA1DParams p{B, L, H, D, K, S, Dil, C, resolve_scale(scale, D)};
  bool use_direct = can_use_direct_nonatomic_1d("na1d_fused_backward_qk", "fp32", D, L, K, S, Dil, C) ||
      can_use_direct_causal_1d("na1d_fused_backward_qk", "fp32", D, L, K, S, Dil, C);
  bool use_vec4 = use_direct && direct_vec4_eligible(D);
  std::string qk_kernel = "na1d_fused_bwd_qk_fp32";
  nb::tuple out;
  if (use_direct) {
    qk_kernel = choose_na1d_qk_bwd_k_direct_kernel(C, use_vec4);
    out = nb::make_tuple(
        launch_one("na1d_qk_bwd_q_fp32", {glf, kf}, p, qf.shape(), false),
        launch_one(
            qk_kernel,
            {glf, qf},
            p,
            kf.shape(),
            false,
            use_vec4 ? (numel(kf.shape()) / 4) : numel(kf.shape())));
  } else {
    size_t threads = static_cast<size_t>(B) * static_cast<size_t>(out_l) * static_cast<size_t>(H) *
        static_cast<size_t>(K) * static_cast<size_t>(D);
    out = launch_two(
        "na1d_fused_bwd_qk_fp32", {glf, qf, kf}, p, qf.shape(), kf.shape(), true, threads);
  }
  debug_set_last_kernel("na1d_fused_backward_qk", qk_kernel);
  mx::array gq = nb::cast<mx::array>(out[0]);
  mx::array gk = nb::cast<mx::array>(out[1]);
  return nb::make_tuple(cast_to_dtype(gq, q_arr.dtype()), cast_to_dtype(gk, k_arr.dtype()));
}

nb::object na1d_fused_backward_v(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  auto v_arr = as_array(v);
  auto a_arr = as_array(attn);
  auto go_arr = as_array(grad_out);
  auto af = to_float32(a_arr);
  auto vf = to_float32(v_arr);
  auto gof = to_float32(go_arr);

  int B = vf.shape(0);
  int out_l = gof.shape(1);
  int L = vf.shape(1);
  int H = vf.shape(2);
  int D = vf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;

  NA1DParams p{B, L, H, D, K, S, Dil, C, 1.0f};
  bool use_direct = can_use_direct_nonatomic_1d("na1d_fused_backward_v", "fp32", D, L, K, S, Dil, C) ||
      can_use_direct_causal_1d("na1d_fused_backward_v", "fp32", D, L, K, S, Dil, C);
  bool use_vec4 = use_direct && direct_vec4_eligible(D);
  std::string v_kernel = "na1d_fused_bwd_v_fp32";
  mx::array gv = [&]() {
    if (use_direct) {
      v_kernel = choose_na1d_av_bwd_v_direct_kernel(C, use_vec4);
      return launch_one(
          v_kernel,
          {af, gof},
          p,
          to_shape({B, L, H, D}),
          false,
          use_vec4 ? (numel(vf.shape()) / 4) : numel(vf.shape()));
    }
    size_t threads = static_cast<size_t>(B) * static_cast<size_t>(out_l) * static_cast<size_t>(H) *
        static_cast<size_t>(K) * static_cast<size_t>(D);
    return launch_one("na1d_fused_bwd_v_fp32", {af, gof}, p, to_shape({B, L, H, D}), true, threads);
  }();
  debug_set_last_kernel("na1d_fused_backward_v", v_kernel);
  return nb::cast(cast_to_dtype(gv, v_arr.dtype()));
}

nb::object na1d_fused_backward_qkv_from_softmax(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& attn,
    const nb::object& grad_attn,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto v_arr = as_array(v);
  auto a_arr = as_array(attn);
  auto ga_arr = as_array(grad_attn);
  auto go_arr = as_array(grad_out);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto vf = to_float32(v_arr);
  auto af = to_float32(a_arr);
  auto gaf = to_float32(ga_arr);
  auto gof = to_float32(go_arr);

  int B = qf.shape(0);
  int L = qf.shape(1);
  int H = qf.shape(2);
  int D = qf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int out_l = af.shape(1);
  int attn_k = af.shape(3);

  // Focused optimized path for 1D unit-stride shapes:
  // - causal with any dilation
  // - noncausal unit-dilation
  bool noncausal_u1d1 = (C == 0 && Dil == 1);
  bool causal_s1 = (C == 1);
  if (S != 1 || out_l != L || (!causal_s1 && !noncausal_u1d1)) {
    throw std::runtime_error(
        "na1d_fused_backward_qkv_from_softmax requires S=1 and either causal or noncausal-d1");
  }

  NA1DParams p{B, L, H, D, K, S, Dil, C, resolve_scale(scale, D)};
  bool use_vec4 = direct_vec4_eligible(D);
  size_t inner_threads = static_cast<size_t>(B) * static_cast<size_t>(L) * static_cast<size_t>(H);
  std::string inner_kernel = "na1d_fused_bwd_inner_s1_fp32";
  if (causal_s1 && attn_k == 9) {
    inner_kernel = "na1d_fused_bwd_inner_s1_causal_k9_fp32";
  }
  mx::array inner = launch_one(
      inner_kernel,
      {af, gaf},
      p,
      to_shape({B, L, H}),
      false,
      inner_threads);

  std::string q_kernel;
  bool use_token_vec4_q = false;
  if (causal_s1) {
    if (attn_k == 9) {
      use_token_vec4_q = use_vec4 && D >= 32;
      if (use_token_vec4_q) {
        q_kernel = "na1d_fused_bwd_q_softmax_direct_s1_causal_k9_token_vec4_fp32";
      } else {
        q_kernel = use_vec4 ? "na1d_fused_bwd_q_softmax_direct_s1_causal_k9_vec4_fp32"
                            : "na1d_fused_bwd_q_softmax_direct_s1_causal_k9_fp32";
      }
    } else {
      q_kernel = use_vec4 ? "na1d_fused_bwd_q_softmax_direct_s1_causal_vec4_fp32"
                          : "na1d_fused_bwd_q_softmax_direct_s1_causal_fp32";
    }
  } else {
    q_kernel = use_vec4 ? "na1d_fused_bwd_q_softmax_s1_vec4_fp32"
                        : "na1d_fused_bwd_q_softmax_s1_fp32";
  }
  std::string kv_kernel;
  if (causal_s1) {
    if (attn_k == 9) {
      kv_kernel = use_vec4 ? "na1d_fused_bwd_kv_softmax_direct_s1_causal_k9_vec4_fp32"
                           : "na1d_fused_bwd_kv_softmax_direct_s1_causal_k9_fp32";
    } else {
      kv_kernel = use_vec4 ? "na1d_fused_bwd_kv_softmax_direct_s1_causal_vec4_fp32"
                           : "na1d_fused_bwd_kv_softmax_direct_s1_causal_fp32";
    }
  } else {
    kv_kernel = use_vec4 ? "na1d_fused_bwd_kv_softmax_direct_u1d1_nc_vec4_fp32"
                         : "na1d_fused_bwd_kv_softmax_direct_u1d1_nc_fp32";
  }
  size_t q_threads = use_token_vec4_q
      ? static_cast<size_t>(B) * static_cast<size_t>(L) * static_cast<size_t>(H)
      : (use_vec4 ? (numel(qf.shape()) / 4) : numel(qf.shape()));
  size_t kv_threads = use_vec4 ? (numel(kf.shape()) / 4) : numel(kf.shape());
  mx::array gq = launch_one(q_kernel, {af, gaf, inner, kf}, p, qf.shape(), false, q_threads);
  nb::tuple kv = launch_two(
      kv_kernel,
      {af, gaf, inner, qf, gof},
      p,
      kf.shape(),
      vf.shape(),
      false,
      kv_threads);
  debug_set_last_kernel("na1d_fused_backward_qk_grad_q", q_kernel);
  debug_set_last_kernel("na1d_fused_backward_qk", kv_kernel);
  debug_set_last_kernel("na1d_fused_backward_v", kv_kernel);

  mx::array gk = nb::cast<mx::array>(kv[0]);
  mx::array gv = nb::cast<mx::array>(kv[1]);
  return nb::make_tuple(
      cast_to_dtype(gq, q_arr.dtype()),
      cast_to_dtype(gk, k_arr.dtype()),
      cast_to_dtype(gv, v_arr.dtype()));
}

nb::object na2d_fused_backward_qk(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_logits,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto gl_arr = as_array(grad_logits);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto glf = to_float32(gl_arr);

  int B = qf.shape(0);
  int IH = qf.shape(1);
  int IW = qf.shape(2);
  int H = qf.shape(3);
  int D = qf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int out_h = glf.shape(1);
  int out_w = glf.shape(2);

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, resolve_scale(scale, D)};
  bool use_vec4_q = direct_vec4_eligible(D);
  if (const char* v = std::getenv("NATTEN_NANOBIND_QK_BWD_Q_VEC4")) {
    use_vec4_q = use_vec4_q && (std::string(v) == "1");
  } else {
    use_vec4_q = false;
  }
  std::string grad_q_kernel = use_vec4_q ? "na2d_qk_bwd_q_vec4_fp32" : "na2d_qk_bwd_q_fp32";
  bool use_direct = can_use_direct_nonatomic_2d(
      "na2d_fused_backward_qk", "fp32", D, IH, IW, K, SH, SW, DH, DW, CH, CW);
  use_direct = use_direct && fused_backward_direct_enabled(2);
  bool use_vec4 = use_direct && direct_vec4_eligible(D);
  std::string qk_kernel = "na2d_fused_bwd_qk_fp32";
  nb::tuple out;
  if (use_direct) {
    qk_kernel = choose_na2d_qk_bwd_k_direct_kernel(K, use_vec4);
    out = nb::make_tuple(
        launch_one(
            grad_q_kernel,
            {glf, kf},
            p,
            qf.shape(),
            false,
            use_vec4_q ? (numel(qf.shape()) / 4) : 0),
        launch_one(
            qk_kernel,
            {glf, qf},
            p,
            kf.shape(),
            false,
            use_vec4 ? (numel(kf.shape()) / 4) : numel(kf.shape())));
  } else {
    size_t threads = static_cast<size_t>(B) * static_cast<size_t>(out_h) *
        static_cast<size_t>(out_w) * static_cast<size_t>(H) * static_cast<size_t>(K) *
        static_cast<size_t>(K) * static_cast<size_t>(D);
    out = launch_two(
        "na2d_fused_bwd_qk_fp32", {glf, qf, kf}, p, qf.shape(), kf.shape(), true, threads);
  }
  debug_set_last_kernel("na2d_fused_backward_qk_grad_q", grad_q_kernel);
  debug_set_last_kernel("na2d_fused_backward_qk", qk_kernel);
  mx::array gq = nb::cast<mx::array>(out[0]);
  mx::array gk = nb::cast<mx::array>(out[1]);
  return nb::make_tuple(cast_to_dtype(gq, q_arr.dtype()), cast_to_dtype(gk, k_arr.dtype()));
}

nb::object na2d_fused_backward_qk_from_softmax(
    const nb::object& q,
    const nb::object& k,
    const nb::object& attn,
    const nb::object& grad_attn,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto a_arr = as_array(attn);
  auto ga_arr = as_array(grad_attn);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto af = to_float32(a_arr);
  auto gaf = to_float32(ga_arr);

  int B = qf.shape(0);
  int IH = qf.shape(1);
  int IW = qf.shape(2);
  int H = qf.shape(3);
  int D = qf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int out_h = af.shape(1);
  int out_w = af.shape(2);

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, resolve_scale(scale, D)};
  bool use_vec4_q = direct_vec4_eligible(D);
  bool use_direct = can_use_direct_nonatomic_2d(
      "na2d_fused_backward_qk", "fp32", D, IH, IW, K, SH, SW, DH, DW, CH, CW);
  use_direct = use_direct && fused_backward_direct_enabled(2);
  std::string grad_q_kernel = choose_na2d_q_softmax_kernel(use_vec4_q, use_direct);
  bool use_vec4 = use_direct && direct_vec4_eligible(D);
  std::string qk_kernel = "na2d_fused_bwd_qk_softmax_fp32";
  nb::tuple out;
  if (use_direct) {
    mx::array inner = mx::sum(mx::multiply(gaf, af), -1, false);
    qk_kernel = use_vec4 ? "na2d_fused_bwd_k_direct_softmax_u1d1_nc_vec4_fp32"
                         : "na2d_fused_bwd_k_direct_softmax_u1d1_nc_fp32";
    out = nb::make_tuple(
        launch_one(
            grad_q_kernel,
            {af, gaf, inner, kf},
            p,
            qf.shape(),
            false,
            use_vec4_q ? (numel(qf.shape()) / 4) : numel(qf.shape())),
        launch_one(
            qk_kernel,
            {af, gaf, inner, qf},
            p,
            kf.shape(),
            false,
            use_vec4 ? (numel(kf.shape()) / 4) : numel(kf.shape())));
  } else {
    mx::array inner = mx::sum(mx::multiply(gaf, af), -1, false);
    size_t threads = static_cast<size_t>(B) * static_cast<size_t>(out_h) *
        static_cast<size_t>(out_w) * static_cast<size_t>(H) * static_cast<size_t>(K) *
        static_cast<size_t>(K) * static_cast<size_t>(D);
    grad_q_kernel = "na2d_fused_bwd_qk_softmax_fp32";
    out = launch_two(
        "na2d_fused_bwd_qk_softmax_fp32",
        {af, gaf, inner, qf, kf},
        p,
        qf.shape(),
        kf.shape(),
        true,
        threads);
  }
  debug_set_last_kernel("na2d_fused_backward_qk_grad_q", grad_q_kernel);
  debug_set_last_kernel("na2d_fused_backward_qk", qk_kernel);
  mx::array gq = nb::cast<mx::array>(out[0]);
  mx::array gk = nb::cast<mx::array>(out[1]);
  return nb::make_tuple(cast_to_dtype(gq, q_arr.dtype()), cast_to_dtype(gk, k_arr.dtype()));
}

nb::object na2d_fused_backward_qkv_from_softmax(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& attn,
    const nb::object& grad_attn,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto v_arr = as_array(v);
  auto a_arr = as_array(attn);
  auto ga_arr = as_array(grad_attn);
  auto go_arr = as_array(grad_out);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto af = to_float32(a_arr);
  auto gaf = to_float32(ga_arr);
  auto gof = to_float32(go_arr);

  int B = qf.shape(0);
  int IH = qf.shape(1);
  int IW = qf.shape(2);
  int H = qf.shape(3);
  int D = qf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int out_h = af.shape(1);
  int out_w = af.shape(2);
  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, resolve_scale(scale, D)};

  bool use_direct = can_use_direct_nonatomic_2d(
      "na2d_fused_backward_qk", "fp32", D, IH, IW, K, SH, SW, DH, DW, CH, CW);
  use_direct = use_direct && fused_backward_direct_enabled(2);
  bool tiled_eligible = SH == 1 && SW == 1 && DH == 1 && DW == 1 && CH == 0 && CW == 0 &&
      IH >= K && IW >= K && K <= 31;
  std::string qkv_mode = parse_qkv_stage_mode_env();
  bool use_tiled = prefer_tiled_qkv_stage_2d(D, IH, IW, K, SH, SW, DH, DW, CH, CW);
  bool use_vec4 = use_tiled && direct_vec4_eligible(D);
  if (qkv_mode == "tiled" && !tiled_eligible) {
    throw std::runtime_error(
        "na2d_fused_backward_qkv_from_softmax tiled mode requires noncausal unit-stride unit-dilation and K<=31");
  }
  if (use_tiled) {
    mx::array inner = mx::sum(mx::multiply(gaf, af), -1, false);
    std::string q_kernel = choose_na2d_q_softmax_kernel(use_vec4, true);
    std::string kv_kernel = choose_na2d_kv_tiled_kernel(K, use_vec4);
    size_t q_threads = use_vec4 ? (numel(qf.shape()) / 4) : numel(qf.shape());
    size_t kv_threads = use_vec4 ? (numel(kf.shape()) / 4) : numel(kf.shape());
    mx::array gq = launch_one(q_kernel, {af, gaf, inner, kf}, p, qf.shape(), false, q_threads);
    nb::tuple kv = launch_two(
        kv_kernel,
        {af, gaf, inner, qf, gof},
        p,
        kf.shape(),
        to_shape({B, IH, IW, H, D}),
        false,
        kv_threads);
    debug_set_last_kernel("na2d_fused_backward_qk_grad_q", q_kernel);
    debug_set_last_kernel("na2d_fused_backward_qk", kv_kernel);
    debug_set_last_kernel("na2d_fused_backward_v", kv_kernel);
    mx::array gk = nb::cast<mx::array>(kv[0]);
    mx::array gv = nb::cast<mx::array>(kv[1]);
    return nb::make_tuple(
        cast_to_dtype(gq, q_arr.dtype()),
        cast_to_dtype(gk, k_arr.dtype()),
        cast_to_dtype(gv, v_arr.dtype()));
  }

  if (use_direct) {
    nb::tuple qk = nb::cast<nb::tuple>(na2d_fused_backward_qk_from_softmax(
        q, k, attn, grad_attn, kernel_size, stride, dilation, is_causal, scale));
    nb::object gv = na2d_fused_backward_v(attn, v, grad_out, kernel_size, stride, dilation, is_causal);
    return nb::make_tuple(qk[0], qk[1], gv);
  }

  mx::array inner = mx::sum(mx::multiply(gaf, af), -1, false);
  std::string qkv_kernel = "na2d_fused_bwd_qkv_softmax_fp32";
  size_t threads = static_cast<size_t>(B) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w) *
      static_cast<size_t>(H) * static_cast<size_t>(K) * static_cast<size_t>(K) *
      static_cast<size_t>(D);
  nb::tuple out = launch_three(
      qkv_kernel,
      {af, gaf, inner, qf, kf, gof},
      p,
      qf.shape(),
      kf.shape(),
      to_shape({B, IH, IW, H, D}),
      true,
      threads);
  debug_set_last_kernel("na2d_fused_backward_qk_grad_q", qkv_kernel);
  debug_set_last_kernel("na2d_fused_backward_qk", qkv_kernel);
  debug_set_last_kernel("na2d_fused_backward_v", qkv_kernel);
  mx::array gq = nb::cast<mx::array>(out[0]);
  mx::array gk = nb::cast<mx::array>(out[1]);
  mx::array gv = nb::cast<mx::array>(out[2]);
  return nb::make_tuple(
      cast_to_dtype(gq, q_arr.dtype()),
      cast_to_dtype(gk, k_arr.dtype()),
      cast_to_dtype(gv, v_arr.dtype()));
}

nb::object na2d_fused_backward_v(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  auto v_arr = as_array(v);
  auto go_arr = as_array(grad_out);
  auto af = to_float32(as_array(attn));
  auto vf = to_float32(v_arr);
  auto gof = to_float32(go_arr);

  int B = vf.shape(0);
  int out_h = gof.shape(1);
  int out_w = gof.shape(2);
  int IH = vf.shape(1);
  int IW = vf.shape(2);
  int H = vf.shape(3);
  int D = vf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, 1.0f};
  bool use_direct = can_use_direct_nonatomic_2d(
      "na2d_fused_backward_v", "fp32", D, IH, IW, K, SH, SW, DH, DW, CH, CW);
  use_direct = use_direct && fused_backward_direct_enabled(2);
  bool use_vec4 = use_direct && direct_vec4_eligible(D);
  std::string v_kernel = "na2d_fused_bwd_v_fp32";
  mx::array gv = [&]() {
    if (use_direct) {
      v_kernel = use_vec4 ? "na2d_av_bwd_v_direct_u1d1_nc_vec4_fp32"
                          : "na2d_av_bwd_v_direct_u1d1_nc_fp32";
      return launch_one(
          v_kernel,
          {af, gof},
          p,
          to_shape({B, IH, IW, H, D}),
          false,
          use_vec4 ? (numel(vf.shape()) / 4) : numel(vf.shape()));
    }
    size_t threads = static_cast<size_t>(B) * static_cast<size_t>(out_h) *
        static_cast<size_t>(out_w) * static_cast<size_t>(H) * static_cast<size_t>(K) *
        static_cast<size_t>(K) * static_cast<size_t>(D);
    return launch_one("na2d_fused_bwd_v_fp32", {af, gof}, p, to_shape({B, IH, IW, H, D}), true, threads);
  }();
  debug_set_last_kernel("na2d_fused_backward_v", v_kernel);
  return nb::cast(cast_to_dtype(gv, v_arr.dtype()));
}

nb::object na3d_fused_backward_qk(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_logits,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto gl_arr = as_array(grad_logits);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto glf = to_float32(gl_arr);

  int B = qf.shape(0);
  int ID = qf.shape(1);
  int IH = qf.shape(2);
  int IW = qf.shape(3);
  int H = qf.shape(4);
  int D = qf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;
  int out_d = glf.shape(1);
  int out_h = glf.shape(2);
  int out_w = glf.shape(3);

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, resolve_scale(scale, D)};
  bool use_direct = can_use_direct_nonatomic_3d(
      "na3d_fused_backward_qk", "fp32", D, ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW);
  use_direct = (use_direct || force_direct_nonatomic_3d_hotshape(
                                  D, ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW)) &&
      fused_backward_direct_enabled(3);
  bool use_vec4 = use_direct && direct_vec4_eligible(D);
  std::string qk_kernel = "na3d_fused_bwd_qk_fp32";
  nb::tuple out;
  if (use_direct) {
    qk_kernel = choose_na3d_qk_bwd_k_direct_kernel(K, use_vec4);
    out = nb::make_tuple(
        launch_one("na3d_qk_bwd_q_fp32", {glf, kf}, p, qf.shape(), false),
        launch_one(
            qk_kernel,
            {glf, qf},
            p,
            kf.shape(),
            false,
            use_vec4 ? (numel(kf.shape()) / 4) : numel(kf.shape())));
  } else {
    size_t threads = static_cast<size_t>(B) * static_cast<size_t>(out_d) *
        static_cast<size_t>(out_h) * static_cast<size_t>(out_w) * static_cast<size_t>(H) *
        static_cast<size_t>(K) * static_cast<size_t>(K) * static_cast<size_t>(K) *
        static_cast<size_t>(D);
    out = launch_two(
        "na3d_fused_bwd_qk_fp32", {glf, qf, kf}, p, qf.shape(), kf.shape(), true, threads);
  }
  debug_set_last_kernel("na3d_fused_backward_qk", qk_kernel);
  mx::array gq = nb::cast<mx::array>(out[0]);
  mx::array gk = nb::cast<mx::array>(out[1]);
  return nb::make_tuple(cast_to_dtype(gq, q_arr.dtype()), cast_to_dtype(gk, k_arr.dtype()));
}

nb::object na3d_fused_backward_qk_from_softmax(
    const nb::object& q,
    const nb::object& k,
    const nb::object& attn,
    const nb::object& grad_attn,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto a_arr = as_array(attn);
  auto ga_arr = as_array(grad_attn);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto af = to_float32(a_arr);
  auto gaf = to_float32(ga_arr);

  int B = qf.shape(0);
  int ID = qf.shape(1);
  int IH = qf.shape(2);
  int IW = qf.shape(3);
  int H = qf.shape(4);
  int D = qf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;
  int out_d = af.shape(1);
  int out_h = af.shape(2);
  int out_w = af.shape(3);

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, resolve_scale(scale, D)};
  bool use_direct = can_use_direct_nonatomic_3d(
      "na3d_fused_backward_qk", "fp32", D, ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW);
  use_direct = (use_direct || force_direct_nonatomic_3d_hotshape(
                                  D, ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW)) &&
      fused_backward_direct_enabled(3);
  bool use_vec4_q = direct_vec4_eligible(D);
  bool use_q_token_vec4 = use_vec4_q && K == 3 && SD == 1 && SH == 1 && SW == 1 && DD == 1 &&
      DH == 1 && DW == 1 && CD == 0 && CH == 0 && CW == 0 && ID >= K && IH >= K && IW >= K;
  std::string qk_kernel = "na3d_fused_bwd_qk_softmax_fp32";
  std::string grad_q_kernel =
      use_q_token_vec4 ? "na3d_fused_bwd_q_softmax_token_k3_vec4_fp32"
                       : choose_na3d_q_softmax_kernel(K, use_vec4_q);
  nb::tuple out;
  if (use_direct) {
    mx::array inner = mx::sum(mx::multiply(gaf, af), -1, false);
    bool use_vec4_k = direct_vec4_eligible(D);
    qk_kernel = choose_na3d_fused_k_direct_softmax_kernel(K, use_vec4_k);
    size_t qk_threads = use_vec4_k ? std::max<size_t>(1, numel(kf.shape()) / 4) : numel(kf.shape());
    size_t q_threads = use_q_token_vec4
        ? static_cast<size_t>(B) * static_cast<size_t>(ID) * static_cast<size_t>(IH) *
            static_cast<size_t>(IW) * static_cast<size_t>(H)
        : (use_vec4_q ? (numel(qf.shape()) / 4) : numel(qf.shape()));
    out = nb::make_tuple(
        launch_one(
            grad_q_kernel,
            {af, gaf, inner, kf},
            p,
            qf.shape(),
            false,
            q_threads),
        launch_one(
            qk_kernel,
            {af, gaf, inner, qf},
            p,
            kf.shape(),
            false,
            qk_threads));
  } else {
    mx::array inner = mx::sum(mx::multiply(gaf, af), -1, false);
    size_t threads = static_cast<size_t>(B) * static_cast<size_t>(out_d) *
        static_cast<size_t>(out_h) * static_cast<size_t>(out_w) * static_cast<size_t>(H) *
        static_cast<size_t>(K) * static_cast<size_t>(K) * static_cast<size_t>(K) *
        static_cast<size_t>(D);
    grad_q_kernel = "na3d_fused_bwd_qk_softmax_fp32";
    out = launch_two(
        "na3d_fused_bwd_qk_softmax_fp32",
        {af, gaf, inner, qf, kf},
        p,
        qf.shape(),
        kf.shape(),
        true,
        threads);
  }
  debug_set_last_kernel("na3d_fused_backward_qk_grad_q", grad_q_kernel);
  debug_set_last_kernel("na3d_fused_backward_qk", qk_kernel);
  mx::array gq = nb::cast<mx::array>(out[0]);
  mx::array gk = nb::cast<mx::array>(out[1]);
  return nb::make_tuple(cast_to_dtype(gq, q_arr.dtype()), cast_to_dtype(gk, k_arr.dtype()));
}

nb::object na3d_fused_backward_qkv_from_softmax(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& attn,
    const nb::object& grad_attn,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto v_arr = as_array(v);
  auto a_arr = as_array(attn);
  auto ga_arr = as_array(grad_attn);
  auto go_arr = as_array(grad_out);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto af = to_float32(a_arr);
  auto gaf = to_float32(ga_arr);
  auto gof = to_float32(go_arr);

  int B = qf.shape(0);
  int ID = qf.shape(1);
  int IH = qf.shape(2);
  int IW = qf.shape(3);
  int H = qf.shape(4);
  int D = qf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;
  int out_d = af.shape(1);
  int out_h = af.shape(2);
  int out_w = af.shape(3);
  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, resolve_scale(scale, D)};

  bool use_direct = can_use_direct_nonatomic_3d(
      "na3d_fused_backward_qk", "fp32", D, ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW);
  use_direct = (use_direct || force_direct_nonatomic_3d_hotshape(
                                  D, ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW)) &&
      fused_backward_direct_enabled(3);
  bool tiled_eligible = SD == 1 && SH == 1 && SW == 1 && DD == 1 && DH == 1 && DW == 1 &&
      CD == 0 && CH == 0 && CW == 0 && ID >= K && IH >= K && IW >= K && K <= 31;
  std::string qkv_mode = parse_qkv_stage_mode_env();
  bool use_tiled =
      prefer_tiled_qkv_stage_3d(D, ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW);
  bool use_vec4 = use_tiled && direct_vec4_eligible(D);
  if (qkv_mode == "tiled" && !tiled_eligible) {
    throw std::runtime_error(
        "na3d_fused_backward_qkv_from_softmax tiled mode requires noncausal unit-stride unit-dilation and K<=31");
  }
  if (use_tiled) {
    std::string tiled_layout = parse_qkv_tiled_layout_env();
    bool use_single_tiled = false;
    if (tiled_layout == "single") {
      use_single_tiled = true;
    } else if (tiled_layout == "split") {
      use_single_tiled = false;
    } else {
      use_single_tiled = false;
    }
    mx::array inner = mx::sum(mx::multiply(gaf, af), -1, false);
    if (use_single_tiled) {
      std::string qkv_kernel = choose_na3d_qkv_tiled_kernel(K, use_vec4);
      size_t threads = use_vec4 ? (numel(kf.shape()) / 4) : numel(kf.shape());
      nb::tuple out = launch_three(
          qkv_kernel,
          {af, gaf, inner, qf, kf, gof},
          p,
          qf.shape(),
          kf.shape(),
          to_shape({B, ID, IH, IW, H, D}),
          false,
          threads);
      debug_set_last_kernel("na3d_fused_backward_qk_grad_q", qkv_kernel);
      debug_set_last_kernel("na3d_fused_backward_qk", qkv_kernel);
      debug_set_last_kernel("na3d_fused_backward_v", qkv_kernel);
      mx::array gq = nb::cast<mx::array>(out[0]);
      mx::array gk = nb::cast<mx::array>(out[1]);
      mx::array gv = nb::cast<mx::array>(out[2]);
      return nb::make_tuple(
          cast_to_dtype(gq, q_arr.dtype()),
          cast_to_dtype(gk, k_arr.dtype()),
          cast_to_dtype(gv, v_arr.dtype()));
    }
    bool use_q_token_vec4 = use_vec4 && K == 3 && SD == 1 && SH == 1 && SW == 1 && DD == 1 &&
        DH == 1 && DW == 1 && CD == 0 && CH == 0 && CW == 0 && ID >= K && IH >= K && IW >= K;
    std::string q_kernel = use_q_token_vec4
        ? "na3d_fused_bwd_q_softmax_token_k3_vec4_fp32"
        : choose_na3d_q_softmax_kernel(K, use_vec4);
    std::string kv_kernel = choose_na3d_kv_tiled_kernel(K, use_vec4);
    size_t q_threads = use_q_token_vec4
        ? static_cast<size_t>(B) * static_cast<size_t>(ID) * static_cast<size_t>(IH) *
            static_cast<size_t>(IW) * static_cast<size_t>(H)
        : (use_vec4 ? (numel(qf.shape()) / 4) : numel(qf.shape()));
    size_t kv_threads = use_vec4 ? (numel(kf.shape()) / 4) : numel(kf.shape());
    mx::array gq = launch_one(q_kernel, {af, gaf, inner, kf}, p, qf.shape(), false, q_threads);
    nb::tuple kv = launch_two(
        kv_kernel,
        {af, gaf, inner, qf, gof},
        p,
        kf.shape(),
        to_shape({B, ID, IH, IW, H, D}),
        false,
        kv_threads);
    debug_set_last_kernel("na3d_fused_backward_qk_grad_q", q_kernel);
    debug_set_last_kernel("na3d_fused_backward_qk", kv_kernel);
    debug_set_last_kernel("na3d_fused_backward_v", kv_kernel);
    mx::array gk = nb::cast<mx::array>(kv[0]);
    mx::array gv = nb::cast<mx::array>(kv[1]);
    return nb::make_tuple(
        cast_to_dtype(gq, q_arr.dtype()),
        cast_to_dtype(gk, k_arr.dtype()),
        cast_to_dtype(gv, v_arr.dtype()));
  }

  if (use_direct) {
    nb::tuple qk = nb::cast<nb::tuple>(na3d_fused_backward_qk_from_softmax(
        q, k, attn, grad_attn, kernel_size, stride, dilation, is_causal, scale));
    nb::object gv = na3d_fused_backward_v(attn, v, grad_out, kernel_size, stride, dilation, is_causal);
    return nb::make_tuple(qk[0], qk[1], gv);
  }

  mx::array inner = mx::sum(mx::multiply(gaf, af), -1, false);
  std::string qkv_kernel = "na3d_fused_bwd_qkv_softmax_fp32";
  size_t threads = static_cast<size_t>(B) * static_cast<size_t>(out_d) * static_cast<size_t>(out_h) *
      static_cast<size_t>(out_w) * static_cast<size_t>(H) * static_cast<size_t>(K) *
      static_cast<size_t>(K) * static_cast<size_t>(K) * static_cast<size_t>(D);
  nb::tuple out = launch_three(
      qkv_kernel,
      {af, gaf, inner, qf, kf, gof},
      p,
      qf.shape(),
      kf.shape(),
      to_shape({B, ID, IH, IW, H, D}),
      true,
      threads);
  debug_set_last_kernel("na3d_fused_backward_qk_grad_q", qkv_kernel);
  debug_set_last_kernel("na3d_fused_backward_qk", qkv_kernel);
  debug_set_last_kernel("na3d_fused_backward_v", qkv_kernel);
  mx::array gq = nb::cast<mx::array>(out[0]);
  mx::array gk = nb::cast<mx::array>(out[1]);
  mx::array gv = nb::cast<mx::array>(out[2]);
  return nb::make_tuple(
      cast_to_dtype(gq, q_arr.dtype()),
      cast_to_dtype(gk, k_arr.dtype()),
      cast_to_dtype(gv, v_arr.dtype()));
}

nb::object na3d_fused_backward_v(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  auto v_arr = as_array(v);
  auto go_arr = as_array(grad_out);
  auto af = to_float32(as_array(attn));
  auto vf = to_float32(v_arr);
  auto gof = to_float32(go_arr);

  int B = vf.shape(0);
  int out_d = gof.shape(1);
  int out_h = gof.shape(2);
  int out_w = gof.shape(3);
  int ID = vf.shape(1);
  int IH = vf.shape(2);
  int IW = vf.shape(3);
  int H = vf.shape(4);
  int D = vf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, 1.0f};
  bool use_direct = can_use_direct_nonatomic_3d(
      "na3d_fused_backward_v", "fp32", D, ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW);
  use_direct = (use_direct || force_direct_nonatomic_3d_hotshape(
                                  D, ID, IH, IW, K, SD, SH, SW, DD, DH, DW, CD, CH, CW)) &&
      fused_backward_direct_enabled(3);
  bool use_vec4 = use_direct && direct_vec4_eligible(D);
  std::string v_kernel = "na3d_fused_bwd_v_fp32";
  mx::array gv = [&]() {
    if (use_direct) {
      v_kernel = use_vec4 ? "na3d_av_bwd_v_direct_u1d1_nc_vec4_fp32"
                          : "na3d_av_bwd_v_direct_u1d1_nc_fp32";
      return launch_one(
          v_kernel,
          {af, gof},
          p,
          to_shape({B, ID, IH, IW, H, D}),
          false,
          use_vec4 ? (numel(vf.shape()) / 4) : numel(vf.shape()));
    }
    size_t threads = static_cast<size_t>(B) * static_cast<size_t>(out_d) *
        static_cast<size_t>(out_h) * static_cast<size_t>(out_w) * static_cast<size_t>(H) *
        static_cast<size_t>(K) * static_cast<size_t>(K) * static_cast<size_t>(K) *
        static_cast<size_t>(D);
    return launch_one(
        "na3d_fused_bwd_v_fp32", {af, gof}, p, to_shape({B, ID, IH, IW, H, D}), true, threads);
  }();
  debug_set_last_kernel("na3d_fused_backward_v", v_kernel);
  return nb::cast(cast_to_dtype(gv, v_arr.dtype()));
}

void debug_set_last_route(const std::string& op, const std::string& route) {
  std::lock_guard<std::mutex> lock(route_mutex());
  route_map()[op] = route;
}

std::string debug_get_last_route(const std::string& op) {
  std::lock_guard<std::mutex> lock(route_mutex());
  auto it = route_map().find(op);
  return (it == route_map().end()) ? std::string() : it->second;
}

void debug_clear_last_routes() {
  std::lock_guard<std::mutex> lock(route_mutex());
  route_map().clear();
}

std::string debug_get_last_kernel(const std::string& op) {
  std::lock_guard<std::mutex> lock(route_mutex());
  auto it = kernel_map().find(op);
  return (it == kernel_map().end()) ? std::string() : it->second;
}

void debug_clear_last_kernels() {
  std::lock_guard<std::mutex> lock(route_mutex());
  kernel_map().clear();
}

void debug_force_fused_failure(bool enabled) {
  force_fused_failure_flag() = enabled;
}

void debug_force_split_failure(bool enabled) {
  force_split_failure_flag() = enabled;
}

bool debug_forced_fused_failure() {
  return force_fused_failure_flag();
}

bool debug_forced_split_failure() {
  return force_split_failure_flag();
}

void debug_inc_python_bridge_calls() {
  python_bridge_calls() += 1;
}

int debug_get_python_bridge_calls() {
  return python_bridge_calls();
}

void debug_clear_python_bridge_calls() {
  python_bridge_calls() = 0;
}

nb::dict debug_get_launch_metrics() {
  std::lock_guard<std::mutex> lock(launch_metrics_mutex());
  nb::dict out;
  for (const auto& it : launch_metrics_count()) {
    const std::string& kernel = it.first;
    int count = std::max(1, it.second);
    double total_ms = launch_metrics_ms_total()[kernel];
    nb::dict rec;
    rec["count"] = nb::int_(count);
    rec["total_ms"] = nb::float_(total_ms);
    rec["mean_ms"] = nb::float_(total_ms / static_cast<double>(count));
    out[nb::str(kernel.c_str())] = rec;
  }
  return out;
}

void debug_clear_launch_metrics() {
  std::lock_guard<std::mutex> lock(launch_metrics_mutex());
  launch_metrics_count().clear();
  launch_metrics_ms_total().clear();
}

}  // namespace natten_mlx::nanobind_metal_runtime

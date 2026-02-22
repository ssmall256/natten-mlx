#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

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

inline int natten_get_window_start(
    int index,
    int length,
    int kernel_size,
    int neighborhood_size,
    int dilation) {
  if (dilation <= 1) {
    return max(index - neighborhood_size, 0) +
        ((index + neighborhood_size >= length) ? (length - index - neighborhood_size - 1) : 0);
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

template <typename T>
inline float na_dot_1d(
    device const T* query,
    device const T* key,
    int q_base,
    int k_base,
    int dim,
    float scale) {
  float acc = 0.0f;
  for (int d = 0; d < dim; ++d) {
    acc += (float)query[q_base + d] * (float)key[k_base + d];
  }
  return acc * scale;
}

template <typename T>
inline float na_dot_2d(
    device const T* query,
    device const T* key,
    int q_base,
    int k_base,
    int dim,
    float scale) {
  float acc = 0.0f;
  for (int d = 0; d < dim; ++d) {
    acc += (float)query[q_base + d] * (float)key[k_base + d];
  }
  return acc * scale;
}

template <typename T>
inline float na_dot_3d(
    device const T* query,
    device const T* key,
    int q_base,
    int k_base,
    int dim,
    float scale) {
  float acc = 0.0f;
  for (int d = 0; d < dim; ++d) {
    acc += (float)query[q_base + d] * (float)key[k_base + d];
  }
  return acc * scale;
}

template <typename T>
inline void na1d_fused_impl(
    device const T* query,
    device const T* key,
    device const T* value,
    device T* out,
    constant NA1DParams& p,
    uint3 gid) {
  int out_len = (p.L + p.S - 1) / p.S;
  if ((int)gid.x >= out_len || (int)gid.z >= p.B * p.H) {
    return;
  }

  int bh = (int)gid.z;
  int b = bh / p.H;
  int h = bh - b * p.H;
  int oq = (int)gid.x;
  int qidx = oq * p.S;
  if (qidx >= p.L) {
    return;
  }

  int q_base = (((b * p.L + qidx) * p.H + h) * p.D);
  int out_base = (((b * out_len + oq) * p.H + h) * p.D);
  int nh = p.K / 2;

  float max_score = -INFINITY;
  int valid_count = 0;

  for (int kk = 0; kk < p.K; ++kk) {
    int kidx = p.CAUSAL
        ? (qidx - kk * p.DIL)
        : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kk * p.DIL);
    if (kidx < 0 || kidx >= p.L) {
      continue;
    }
    int k_base = (((b * p.L + kidx) * p.H + h) * p.D);
    float s = na_dot_1d(query, key, q_base, k_base, p.D, p.SCALE);
    max_score = max(max_score, s);
    valid_count += 1;
  }

  if (valid_count == 0) {
    for (int d = 0; d < p.D; ++d) {
      out[out_base + d] = (T)0.0f;
    }
    return;
  }

  float sum_exp = 0.0f;
  for (int kk = 0; kk < p.K; ++kk) {
    int kidx = p.CAUSAL
        ? (qidx - kk * p.DIL)
        : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kk * p.DIL);
    if (kidx < 0 || kidx >= p.L) {
      continue;
    }
    int k_base = (((b * p.L + kidx) * p.H + h) * p.D);
    float s = na_dot_1d(query, key, q_base, k_base, p.D, p.SCALE);
    sum_exp += exp(s - max_score);
  }

  for (int d = 0; d < p.D; ++d) {
    out[out_base + d] = (T)0.0f;
  }
  for (int kk = 0; kk < p.K; ++kk) {
    int kidx = p.CAUSAL
        ? (qidx - kk * p.DIL)
        : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kk * p.DIL);
    if (kidx < 0 || kidx >= p.L) {
      continue;
    }
    int k_base = (((b * p.L + kidx) * p.H + h) * p.D);
    float s = na_dot_1d(query, key, q_base, k_base, p.D, p.SCALE);
    float w = exp(s - max_score) / sum_exp;
    int v_base = (((b * p.L + kidx) * p.H + h) * p.D);
    for (int d = 0; d < p.D; ++d) {
      out[out_base + d] += (T)(w * (float)value[v_base + d]);
    }
  }
}

template <typename T>
inline void na2d_fused_impl(
    device const T* query,
    device const T* key,
    device const T* value,
    device T* out,
    constant NA2DParams& p,
    uint3 gid) {
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;
  if ((int)gid.x >= out_w || (int)gid.y >= out_h || (int)gid.z >= p.B * p.H) {
    return;
  }

  int bh = (int)gid.z;
  int b = bh / p.H;
  int h = bh - b * p.H;
  int oh = (int)gid.y;
  int ow = (int)gid.x;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    return;
  }

  int q_base = ((((b * p.IH + qh) * p.IW + qw) * p.H + h) * p.D);
  int out_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D);
  int nh = p.K / 2;

  float max_score = -INFINITY;
  int valid_count = 0;
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = p.CH
        ? (qh - kh * p.DH)
        : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW
          ? (qw - kw * p.DW)
          : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      int k_base = ((((b * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
      float s = na_dot_2d(query, key, q_base, k_base, p.D, p.SCALE);
      max_score = max(max_score, s);
      valid_count += 1;
    }
  }

  if (valid_count == 0) {
    for (int d = 0; d < p.D; ++d) {
      out[out_base + d] = (T)0.0f;
    }
    return;
  }

  float sum_exp = 0.0f;
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = p.CH
        ? (qh - kh * p.DH)
        : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW
          ? (qw - kw * p.DW)
          : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      int k_base = ((((b * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
      float s = na_dot_2d(query, key, q_base, k_base, p.D, p.SCALE);
      sum_exp += exp(s - max_score);
    }
  }

  for (int d = 0; d < p.D; ++d) {
    out[out_base + d] = (T)0.0f;
  }
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = p.CH
        ? (qh - kh * p.DH)
        : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW
          ? (qw - kw * p.DW)
          : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      int k_base = ((((b * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
      float s = na_dot_2d(query, key, q_base, k_base, p.D, p.SCALE);
      float w = exp(s - max_score) / sum_exp;
      int v_base = ((((b * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
      for (int d = 0; d < p.D; ++d) {
        out[out_base + d] += (T)(w * (float)value[v_base + d]);
      }
    }
  }
}

template <typename T>
inline void na3d_fused_impl(
    device const T* query,
    device const T* key,
    device const T* value,
    device T* out,
    constant NA3DParams& p,
    uint3 gid) {
  int out_d = (p.ID + p.SD - 1) / p.SD;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;
  if ((int)gid.x >= out_w || (int)gid.y >= out_h || (int)gid.z >= p.B * p.H * out_d) {
    return;
  }

  int z = (int)gid.z;
  int od = z % out_d;
  int bh = z / out_d;
  int b = bh / p.H;
  int h = bh - b * p.H;
  int oh = (int)gid.y;
  int ow = (int)gid.x;
  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) {
    return;
  }

  int q_base = (((((b * p.ID + qd) * p.IH + qh) * p.IW + qw) * p.H + h) * p.D);
  int out_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D);
  int nh = p.K / 2;

  float max_score = -INFINITY;
  int valid_count = 0;
  for (int kd = 0; kd < p.K; ++kd) {
    int id = p.CD
        ? (qd - kd * p.DD)
        : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH
          ? (qh - kh * p.DH)
          : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW
            ? (qw - kw * p.DW)
            : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int k_base = (((((b * p.ID + id) * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
        float s = na_dot_3d(query, key, q_base, k_base, p.D, p.SCALE);
        max_score = max(max_score, s);
        valid_count += 1;
      }
    }
  }

  if (valid_count == 0) {
    for (int d = 0; d < p.D; ++d) {
      out[out_base + d] = (T)0.0f;
    }
    return;
  }

  float sum_exp = 0.0f;
  for (int kd = 0; kd < p.K; ++kd) {
    int id = p.CD
        ? (qd - kd * p.DD)
        : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH
          ? (qh - kh * p.DH)
          : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW
            ? (qw - kw * p.DW)
            : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int k_base = (((((b * p.ID + id) * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
        float s = na_dot_3d(query, key, q_base, k_base, p.D, p.SCALE);
        sum_exp += exp(s - max_score);
      }
    }
  }

  for (int d = 0; d < p.D; ++d) {
    out[out_base + d] = (T)0.0f;
  }
  for (int kd = 0; kd < p.K; ++kd) {
    int id = p.CD
        ? (qd - kd * p.DD)
        : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH
          ? (qh - kh * p.DH)
          : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW
            ? (qw - kw * p.DW)
            : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int k_base = (((((b * p.ID + id) * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
        float s = na_dot_3d(query, key, q_base, k_base, p.D, p.SCALE);
        float w = exp(s - max_score) / sum_exp;
        int v_base = (((((b * p.ID + id) * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
        for (int d = 0; d < p.D; ++d) {
          out[out_base + d] += (T)(w * (float)value[v_base + d]);
        }
      }
    }
  }
}

template <typename T>
[[kernel]] void na1d_fused_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA1DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na1d_fused_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na1d_fused_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA1DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na1d_fused_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na2d_fused_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA2DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na2d_fused_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na2d_fused_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA2DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na2d_fused_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na3d_fused_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA3DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na3d_fused_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na3d_fused_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA3DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na3d_fused_impl(query, key, value, out, p, gid);
}

#define INSTANTIATE_1D(name, type)                                                         \
  template [[host_name("na1d_fused_" #name)]] [[kernel]] void na1d_fused_kernel<type>(    \
      device const type * query [[buffer(0)]],                                             \
      device const type * key [[buffer(1)]],                                               \
      device const type * value [[buffer(2)]],                                             \
      device type * out [[buffer(3)]],                                                     \
      constant NA1DParams & p [[buffer(4)]],                                               \
      uint3 gid [[thread_position_in_grid]]);                                              \
  template [[host_name("na1d_fused_vec4_" #name)]] [[kernel]] void na1d_fused_vec4_kernel<type>( \
      device const type * query [[buffer(0)]],                                             \
      device const type * key [[buffer(1)]],                                               \
      device const type * value [[buffer(2)]],                                             \
      device type * out [[buffer(3)]],                                                     \
      constant NA1DParams & p [[buffer(4)]],                                               \
      uint3 gid [[thread_position_in_grid]]);

#define INSTANTIATE_2D(name, type)                                                         \
  template [[host_name("na2d_fused_" #name)]] [[kernel]] void na2d_fused_kernel<type>(    \
      device const type * query [[buffer(0)]],                                             \
      device const type * key [[buffer(1)]],                                               \
      device const type * value [[buffer(2)]],                                             \
      device type * out [[buffer(3)]],                                                     \
      constant NA2DParams & p [[buffer(4)]],                                               \
      uint3 gid [[thread_position_in_grid]]);                                              \
  template [[host_name("na2d_fused_vec4_" #name)]] [[kernel]] void na2d_fused_vec4_kernel<type>( \
      device const type * query [[buffer(0)]],                                             \
      device const type * key [[buffer(1)]],                                               \
      device const type * value [[buffer(2)]],                                             \
      device type * out [[buffer(3)]],                                                     \
      constant NA2DParams & p [[buffer(4)]],                                               \
      uint3 gid [[thread_position_in_grid]]);

#define INSTANTIATE_3D(name, type)                                                         \
  template [[host_name("na3d_fused_" #name)]] [[kernel]] void na3d_fused_kernel<type>(    \
      device const type * query [[buffer(0)]],                                             \
      device const type * key [[buffer(1)]],                                               \
      device const type * value [[buffer(2)]],                                             \
      device type * out [[buffer(3)]],                                                     \
      constant NA3DParams & p [[buffer(4)]],                                               \
      uint3 gid [[thread_position_in_grid]]);                                              \
  template [[host_name("na3d_fused_vec4_" #name)]] [[kernel]] void na3d_fused_vec4_kernel<type>( \
      device const type * query [[buffer(0)]],                                             \
      device const type * key [[buffer(1)]],                                               \
      device const type * value [[buffer(2)]],                                             \
      device type * out [[buffer(3)]],                                                     \
      constant NA3DParams & p [[buffer(4)]],                                               \
      uint3 gid [[thread_position_in_grid]]);

INSTANTIATE_1D(fp16, half);
INSTANTIATE_1D(bf16, bfloat16_t);
INSTANTIATE_1D(fp32, float);

INSTANTIATE_2D(fp16, half);
INSTANTIATE_2D(bf16, bfloat16_t);
INSTANTIATE_2D(fp32, float);

INSTANTIATE_3D(fp16, half);
INSTANTIATE_3D(bf16, bfloat16_t);
INSTANTIATE_3D(fp32, float);

inline int nb_ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

inline int nb_base_1d(int b, int i, int h, int d, int L, int H, int D) {
  return (((b * L + i) * H + h) * D + d);
}

inline int nb_base_2d(int b, int i, int j, int h, int d, int IH, int IW, int H, int D) {
  return ((((b * IH + i) * IW + j) * H + h) * D + d);
}

inline int nb_base_3d(int b, int z, int i, int j, int h, int d, int ID, int IH, int IW, int H, int D) {
  return (((((b * ID + z) * IH + i) * IW + j) * H + h) * D + d);
}

[[kernel]] void na1d_fused_bwd_attn_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* grad_out [[buffer(3)]],
    device float* attn [[buffer(4)]],
    device float* grad_attn [[buffer(5)]],
    constant NA1DParams& p [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = nb_ceil_div(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * out_l * p.H * p.K;
  if (idx >= total) {
    return;
  }

  int kpos = idx % p.K;
  int t = idx / p.K;
  int h = t % p.H;
  t /= p.H;
  int oq = t % out_l;
  int b = t / out_l;

  int qidx = oq * p.S;
  if (qidx >= p.L) {
    attn[idx] = 0.0f;
    grad_attn[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int kidx = p.CAUSAL ? (qidx - kpos * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);

  float max_score = -INFINITY;
  for (int kk = 0; kk < p.K; ++kk) {
    int k2 = p.CAUSAL ? (qidx - kk * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kk * p.DIL);
    if (k2 < 0 || k2 >= p.L) {
      continue;
    }
    float s = 0.0f;
    int q_base = nb_base_1d(b, qidx, h, 0, p.L, p.H, p.D);
    int k_base = nb_base_1d(b, k2, h, 0, p.L, p.H, p.D);
    for (int d = 0; d < p.D; ++d) {
      s += query[q_base + d] * key[k_base + d];
    }
    s *= p.SCALE;
    max_score = max(max_score, s);
  }

  float sum_exp = 0.0f;
  for (int kk = 0; kk < p.K; ++kk) {
    int k2 = p.CAUSAL ? (qidx - kk * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kk * p.DIL);
    if (k2 < 0 || k2 >= p.L) {
      continue;
    }
    float s = 0.0f;
    int q_base = nb_base_1d(b, qidx, h, 0, p.L, p.H, p.D);
    int k_base = nb_base_1d(b, k2, h, 0, p.L, p.H, p.D);
    for (int d = 0; d < p.D; ++d) {
      s += query[q_base + d] * key[k_base + d];
    }
    s *= p.SCALE;
    sum_exp += exp(s - max_score);
  }

  if (kidx < 0 || kidx >= p.L || sum_exp <= 0.0f) {
    attn[idx] = 0.0f;
    grad_attn[idx] = 0.0f;
    return;
  }

  float s_this = 0.0f;
  int q_base = nb_base_1d(b, qidx, h, 0, p.L, p.H, p.D);
  int k_base = nb_base_1d(b, kidx, h, 0, p.L, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    s_this += query[q_base + d] * key[k_base + d];
  }
  s_this *= p.SCALE;
  float w = exp(s_this - max_score) / sum_exp;
  attn[idx] = w;

  float ga = 0.0f;
  int go_base = (((b * out_l + oq) * p.H + h) * p.D);
  int v_base = nb_base_1d(b, kidx, h, 0, p.L, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    ga += grad_out[go_base + d] * value[v_base + d];
  }
  grad_attn[idx] = ga;
}

[[kernel]] void na1d_fused_bwd_qk_fp32(
    device const float* grad_logits [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device const float* key [[buffer(2)]],
    device atomic<float>* grad_q [[buffer(3)]],
    device atomic<float>* grad_k [[buffer(4)]],
    constant NA1DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = nb_ceil_div(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * out_l * p.H * p.K * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int kpos = t % p.K;
  t /= p.K;
  int h = t % p.H;
  t /= p.H;
  int oq = t % out_l;
  int b = t / out_l;

  int qidx = oq * p.S;
  if (qidx >= p.L) {
    return;
  }
  int nh = p.K / 2;
  int kidx = p.CAUSAL ? (qidx - kpos * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
  if (kidx < 0 || kidx >= p.L) {
    return;
  }

  int gl_idx = (((b * out_l + oq) * p.H + h) * p.K + kpos);
  float g = grad_logits[gl_idx] * p.SCALE;
  atomic_fetch_add_explicit(&grad_q[nb_base_1d(b, qidx, h, d, p.L, p.H, p.D)], g * key[nb_base_1d(b, kidx, h, d, p.L, p.H, p.D)], memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_k[nb_base_1d(b, kidx, h, d, p.L, p.H, p.D)], g * query[nb_base_1d(b, qidx, h, d, p.L, p.H, p.D)], memory_order_relaxed);
}

[[kernel]] void na1d_fused_bwd_v_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device atomic<float>* grad_v [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = nb_ceil_div(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * out_l * p.H * p.K * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int kpos = t % p.K;
  t /= p.K;
  int h = t % p.H;
  t /= p.H;
  int oq = t % out_l;
  int b = t / out_l;

  int qidx = oq * p.S;
  if (qidx >= p.L) {
    return;
  }
  int nh = p.K / 2;
  int kidx = p.CAUSAL ? (qidx - kpos * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
  if (kidx < 0 || kidx >= p.L) {
    return;
  }

  int aidx = (((b * out_l + oq) * p.H + h) * p.K + kpos);
  int goidx = (((b * out_l + oq) * p.H + h) * p.D + d);
  float g = attn[aidx] * grad_out[goidx];
  atomic_fetch_add_explicit(&grad_v[nb_base_1d(b, kidx, h, d, p.L, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na2d_fused_bwd_qk_fp32(
    device const float* grad_logits [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device const float* key [[buffer(2)]],
    device atomic<float>* grad_q [[buffer(3)]],
    device atomic<float>* grad_k [[buffer(4)]],
    constant NA2DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = nb_ceil_div(p.IH, p.SH);
  int out_w = nb_ceil_div(p.IW, p.SW);
  int k2 = p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * k2 * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int kpos = t % k2;
  int kh = kpos / p.K;
  int kw = kpos % p.K;
  t /= k2;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    return;
  }
  int nh = p.K / 2;
  int ih = p.CH ? (qh - kh * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - kw * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int gl_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * k2 + kpos);
  float g = grad_logits[gl_idx] * p.SCALE;
  atomic_fetch_add_explicit(&grad_q[nb_base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D)], g * key[nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)], memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_k[nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)], g * query[nb_base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D)], memory_order_relaxed);
}

[[kernel]] void na2d_fused_bwd_v_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device atomic<float>* grad_v [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = nb_ceil_div(p.IH, p.SH);
  int out_w = nb_ceil_div(p.IW, p.SW);
  int k2 = p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * k2 * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int kpos = t % k2;
  int kh = kpos / p.K;
  int kw = kpos % p.K;
  t /= k2;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    return;
  }
  int nh = p.K / 2;
  int ih = p.CH ? (qh - kh * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - kw * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int a_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * k2 + kpos);
  int go_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D + d);
  float g = attn[a_idx] * grad_out[go_idx];
  atomic_fetch_add_explicit(&grad_v[nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na3d_fused_bwd_qk_fp32(
    device const float* grad_logits [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device const float* key [[buffer(2)]],
    device atomic<float>* grad_q [[buffer(3)]],
    device atomic<float>* grad_k [[buffer(4)]],
    constant NA3DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = nb_ceil_div(p.ID, p.SD);
  int out_h = nb_ceil_div(p.IH, p.SH);
  int out_w = nb_ceil_div(p.IW, p.SW);
  int k3 = p.K * p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * k3 * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int kpos = t % k3;
  int kd = kpos / (p.K * p.K);
  int rem = kpos % (p.K * p.K);
  int kh = rem / p.K;
  int kw = rem % p.K;
  t /= k3;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  t /= out_h;
  int od = t % out_d;
  int b = t / out_d;

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) {
    return;
  }
  int nh = p.K / 2;
  int id = p.CD ? (qd - kd * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - kh * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - kw * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int gl_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3 + kpos);
  float g = grad_logits[gl_idx] * p.SCALE;
  atomic_fetch_add_explicit(&grad_q[nb_base_3d(b, qd, qh, qw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], g * key[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_k[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], g * query[nb_base_3d(b, qd, qh, qw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], memory_order_relaxed);
}

[[kernel]] void na3d_fused_bwd_v_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device atomic<float>* grad_v [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = nb_ceil_div(p.ID, p.SD);
  int out_h = nb_ceil_div(p.IH, p.SH);
  int out_w = nb_ceil_div(p.IW, p.SW);
  int k3 = p.K * p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * k3 * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int kpos = t % k3;
  int kd = kpos / (p.K * p.K);
  int rem = kpos % (p.K * p.K);
  int kh = rem / p.K;
  int kw = rem % p.K;
  t /= k3;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  t /= out_h;
  int od = t % out_d;
  int b = t / out_d;

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) {
    return;
  }
  int nh = p.K / 2;
  int id = p.CD ? (qd - kd * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - kh * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - kw * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int a_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3 + kpos);
  int go_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D + d);
  float g = attn[a_idx] * grad_out[go_idx];
  atomic_fetch_add_explicit(&grad_v[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], g, memory_order_relaxed);
}

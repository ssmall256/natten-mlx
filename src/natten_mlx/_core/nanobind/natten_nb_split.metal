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

struct Clear1Params {
  uint N;
};

struct Clear2Params {
  uint N0;
  uint N1;
};

struct Clear3Params {
  uint N0;
  uint N1;
  uint N2;
};

[[kernel]] void natten_clear_f32(
    device float* out [[buffer(0)]],
    constant Clear1Params& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid < p.N) {
    out[tid] = 0.0f;
  }
}

[[kernel]] void natten_clear2_f32(
    device float* out0 [[buffer(0)]],
    device float* out1 [[buffer(1)]],
    constant Clear2Params& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid < p.N0) {
    out0[tid] = 0.0f;
  }
  if (tid < p.N1) {
    out1[tid] = 0.0f;
  }
}

[[kernel]] void natten_clear3_f32(
    device float* out0 [[buffer(0)]],
    device float* out1 [[buffer(1)]],
    device float* out2 [[buffer(2)]],
    constant Clear3Params& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid < p.N0) {
    out0[tid] = 0.0f;
  }
  if (tid < p.N1) {
    out1[tid] = 0.0f;
  }
  if (tid < p.N2) {
    out2[tid] = 0.0f;
  }
}

inline int ceil_div_int(int a, int b) {
  return (a + b - 1) / b;
}

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

inline bool valid_1d(int idx, int length) {
  return idx >= 0 && idx < length;
}

inline int base_1d(int b, int i, int h, int d, int L, int H, int D) {
  return (((b * L + i) * H + h) * D + d);
}

inline int base_2d(int b, int i, int j, int h, int d, int IH, int IW, int H, int D) {
  return ((((b * IH + i) * IW + j) * H + h) * D + d);
}

inline int base_3d(int b, int z, int i, int j, int h, int d, int ID, int IH, int IW, int H, int D) {
  return (((((b * ID + z) * IH + i) * IW + j) * H + h) * D + d);
}

inline int start_noncausal_u1(int q, int length, int k, int nh) {
  if (q < nh) {
    return 0;
  }
  if (q + nh >= length) {
    return max(0, length - k);
  }
  return q - nh;
}

template <typename T>
struct lowp_vec4_type;

template <>
struct lowp_vec4_type<half> {
  using type = half4;
};

template <>
struct lowp_vec4_type<bfloat16_t> {
  using type = bfloat4;
};

template <typename T>
[[kernel]] void na1d_qk_lowp_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
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
    out[idx] = (T)(-INFINITY);
    return;
  }

  int nh = p.K / 2;
  int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
  if (!valid_1d(kidx, p.L)) {
    out[idx] = (T)(-INFINITY);
    return;
  }

  float acc = 0.0f;
  int q_base = base_1d(b, qidx, h, 0, p.L, p.H, p.D);
  int k_base = base_1d(b, kidx, h, 0, p.L, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += (float)query[q_base + d] * (float)key[k_base + d];
  }
  out[idx] = (T)(acc * p.SCALE);
}

template <typename T>
[[kernel]] void na1d_av_lowp_kernel(
    device const T* attn [[buffer(0)]],
    device const T* value [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * out_l * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int oq = t % out_l;
  int b = t / out_l;

  int qidx = oq * p.S;
  if (qidx >= p.L) {
    out[idx] = (T)0.0f;
    return;
  }

  int nh = p.K / 2;
  float acc = 0.0f;
  int attn_base = (((b * out_l + oq) * p.H + h) * p.K);
  for (int kpos = 0; kpos < p.K; ++kpos) {
    int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL)
                        : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
    if (!valid_1d(kidx, p.L)) {
      continue;
    }
    float w = (float)attn[attn_base + kpos];
    acc += w * (float)value[base_1d(b, kidx, h, d, p.L, p.H, p.D)];
  }
  out[idx] = (T)acc;
}

template <typename T>
[[kernel]] void na2d_qk_lowp_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k2 = p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * k2;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k2;
  int kh = kpos / p.K;
  int kw = kpos % p.K;
  int t = idx / k2;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    out[idx] = (T)(-INFINITY);
    return;
  }

  int nh = p.K / 2;
  int h_start = p.CH ? 0 : natten_get_window_start(qh, p.IH, p.K, nh, p.DH);
  int w_start = p.CW ? 0 : natten_get_window_start(qw, p.IW, p.K, nh, p.DW);
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH) : (h_start + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW) : (w_start + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    out[idx] = (T)(-INFINITY);
    return;
  }

  float acc = 0.0f;
  int q_base = base_2d(b, qh, qw, h, 0, p.IH, p.IW, p.H, p.D);
  int k_base = base_2d(b, ih, iw, h, 0, p.IH, p.IW, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += (float)query[q_base + d] * (float)key[k_base + d];
  }
  out[idx] = (T)(acc * p.SCALE);
}

template <typename T>
[[kernel]] void na2d_qk_lowp_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k2 = p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * k2;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k2;
  int kh = kpos / p.K;
  int kw = kpos % p.K;
  int t = idx / k2;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    out[idx] = (T)(-INFINITY);
    return;
  }

  int nh = p.K / 2;
  int h_start = p.CH ? 0 : natten_get_window_start(qh, p.IH, p.K, nh, p.DH);
  int w_start = p.CW ? 0 : natten_get_window_start(qw, p.IW, p.K, nh, p.DW);
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH) : (h_start + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW) : (w_start + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    out[idx] = (T)(-INFINITY);
    return;
  }

  float acc = 0.0f;
  int q_base = base_2d(b, qh, qw, h, 0, p.IH, p.IW, p.H, p.D);
  int k_base = base_2d(b, ih, iw, h, 0, p.IH, p.IW, p.H, p.D);
  using lowp_vec4 = typename lowp_vec4_type<T>::type;
  for (int d = 0; d < p.D; d += 4) {
    const device lowp_vec4* q4 = reinterpret_cast<const device lowp_vec4*>(query + q_base + d);
    const device lowp_vec4* k4 = reinterpret_cast<const device lowp_vec4*>(key + k_base + d);
    acc += dot(float4(*q4), float4(*k4));
  }
  out[idx] = (T)(acc * p.SCALE);
}

template <typename T>
[[kernel]] void na2d_av_lowp_kernel(
    device const T* attn [[buffer(0)]],
    device const T* value [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    out[idx] = (T)0.0f;
    return;
  }

  int nh = p.K / 2;
  int h_start = p.CH ? 0 : natten_get_window_start(qh, p.IH, p.K, nh, p.DH);
  int w_start = p.CW ? 0 : natten_get_window_start(qw, p.IW, p.K, nh, p.DW);
  float acc = 0.0f;
  int attn_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * (p.K * p.K));
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH) : (h_start + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW) : (w_start + kw * p.DW);
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      float w = (float)attn[attn_base + kh * p.K + kw];
      acc += w * (float)value[base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)];
    }
  }
  out[idx] = (T)acc;
}

template <typename T>
[[kernel]] void na2d_av_lowp_vec4_kernel(
    device const T* attn [[buffer(0)]],
    device const T* value [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int d4_count = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * d4_count;
  if (idx >= total) {
    return;
  }

  int d4 = idx % d4_count;
  int t = idx / d4_count;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  int out_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D + d4 * 4);
  if (qh >= p.IH || qw >= p.IW) {
    out[out_base + 0] = (T)0.0f;
    out[out_base + 1] = (T)0.0f;
    out[out_base + 2] = (T)0.0f;
    out[out_base + 3] = (T)0.0f;
    return;
  }

  int nh = p.K / 2;
  int h_start = p.CH ? 0 : natten_get_window_start(qh, p.IH, p.K, nh, p.DH);
  int w_start = p.CW ? 0 : natten_get_window_start(qw, p.IW, p.K, nh, p.DW);
  float4 acc = float4(0.0f);
  using lowp_vec4 = typename lowp_vec4_type<T>::type;
  int attn_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * (p.K * p.K));
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH) : (h_start + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW) : (w_start + kw * p.DW);
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      float w = (float)attn[attn_base + kh * p.K + kw];
      int v_base = base_2d(b, ih, iw, h, d4 * 4, p.IH, p.IW, p.H, p.D);
      const device lowp_vec4* v4 = reinterpret_cast<const device lowp_vec4*>(value + v_base);
      acc += w * float4(*v4);
    }
  }
  device lowp_vec4* out4 = reinterpret_cast<device lowp_vec4*>(out + out_base);
  *out4 = lowp_vec4(acc);
}

template <typename T>
[[kernel]] void na3d_qk_lowp_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k3 = p.K * p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * k3;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k3;
  int kd = kpos / (p.K * p.K);
  int rem = kpos % (p.K * p.K);
  int kh = rem / p.K;
  int kw = rem % p.K;
  int t = idx / k3;
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
    out[idx] = (T)(-INFINITY);
    return;
  }

  int nh = p.K / 2;
  int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    out[idx] = (T)(-INFINITY);
    return;
  }

  float acc = 0.0f;
  int q_base = base_3d(b, qd, qh, qw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
  int k_base = base_3d(b, id, ih, iw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += (float)query[q_base + d] * (float)key[k_base + d];
  }
  out[idx] = (T)(acc * p.SCALE);
}

template <typename T>
[[kernel]] void na3d_av_lowp_kernel(
    device const T* attn [[buffer(0)]],
    device const T* value [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
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
    out[idx] = (T)0.0f;
    return;
  }

  int nh = p.K / 2;
  float acc = 0.0f;
  int k3 = p.K * p.K * p.K;
  int attn_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3);
  for (int kd = 0; kd < p.K; ++kd) {
    int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                  : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                    : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                      : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int kpos = (kd * p.K + kh) * p.K + kw;
        float w = (float)attn[attn_base + kpos];
        acc += w * (float)value[base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)];
      }
    }
  }
  out[idx] = (T)acc;
}

template [[host_name("na1d_qk_fp16")]] [[kernel]] void na1d_qk_lowp_kernel<half>(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na1d_qk_bf16")]] [[kernel]] void na1d_qk_lowp_kernel<bfloat16_t>(
    device const bfloat16_t* query [[buffer(0)]],
    device const bfloat16_t* key [[buffer(1)]],
    device bfloat16_t* out [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na1d_av_fp16")]] [[kernel]] void na1d_av_lowp_kernel<half>(
    device const half* attn [[buffer(0)]],
    device const half* value [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na1d_av_bf16")]] [[kernel]] void na1d_av_lowp_kernel<bfloat16_t>(
    device const bfloat16_t* attn [[buffer(0)]],
    device const bfloat16_t* value [[buffer(1)]],
    device bfloat16_t* out [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na2d_qk_fp16")]] [[kernel]] void na2d_qk_lowp_kernel<half>(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na2d_qk_bf16")]] [[kernel]] void na2d_qk_lowp_kernel<bfloat16_t>(
    device const bfloat16_t* query [[buffer(0)]],
    device const bfloat16_t* key [[buffer(1)]],
    device bfloat16_t* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na2d_qk_vec4_fp16")]] [[kernel]] void na2d_qk_lowp_vec4_kernel<half>(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na2d_qk_vec4_bf16")]] [[kernel]] void na2d_qk_lowp_vec4_kernel<bfloat16_t>(
    device const bfloat16_t* query [[buffer(0)]],
    device const bfloat16_t* key [[buffer(1)]],
    device bfloat16_t* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na2d_av_fp16")]] [[kernel]] void na2d_av_lowp_kernel<half>(
    device const half* attn [[buffer(0)]],
    device const half* value [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na2d_av_bf16")]] [[kernel]] void na2d_av_lowp_kernel<bfloat16_t>(
    device const bfloat16_t* attn [[buffer(0)]],
    device const bfloat16_t* value [[buffer(1)]],
    device bfloat16_t* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na2d_av_vec4_fp16")]] [[kernel]] void na2d_av_lowp_vec4_kernel<half>(
    device const half* attn [[buffer(0)]],
    device const half* value [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na2d_av_vec4_bf16")]] [[kernel]] void na2d_av_lowp_vec4_kernel<bfloat16_t>(
    device const bfloat16_t* attn [[buffer(0)]],
    device const bfloat16_t* value [[buffer(1)]],
    device bfloat16_t* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na3d_qk_fp16")]] [[kernel]] void na3d_qk_lowp_kernel<half>(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na3d_qk_bf16")]] [[kernel]] void na3d_qk_lowp_kernel<bfloat16_t>(
    device const bfloat16_t* query [[buffer(0)]],
    device const bfloat16_t* key [[buffer(1)]],
    device bfloat16_t* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na3d_av_fp16")]] [[kernel]] void na3d_av_lowp_kernel<half>(
    device const half* attn [[buffer(0)]],
    device const half* value [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

template [[host_name("na3d_av_bf16")]] [[kernel]] void na3d_av_lowp_kernel<bfloat16_t>(
    device const bfloat16_t* attn [[buffer(0)]],
    device const bfloat16_t* value [[buffer(1)]],
    device bfloat16_t* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]);

[[kernel]] void na1d_qk_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
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
    out[idx] = -INFINITY;
    return;
  }

  int nh = p.K / 2;
  int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
  if (!valid_1d(kidx, p.L)) {
    out[idx] = -INFINITY;
    return;
  }

  float acc = 0.0f;
  int q_base = base_1d(b, qidx, h, 0, p.L, p.H, p.D);
  int k_base = base_1d(b, kidx, h, 0, p.L, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += query[q_base + d] * key[k_base + d];
  }
  out[idx] = acc * p.SCALE;
}

[[kernel]] void na1d_qk_vec4_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
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
    out[idx] = -INFINITY;
    return;
  }

  int nh = p.K / 2;
  int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
  if (!valid_1d(kidx, p.L)) {
    out[idx] = -INFINITY;
    return;
  }

  int q_base = base_1d(b, qidx, h, 0, p.L, p.H, p.D);
  int k_base = base_1d(b, kidx, h, 0, p.L, p.H, p.D);
  int dim4 = p.D / 4;
  float acc = 0.0f;
  const device float4* q4 = reinterpret_cast<const device float4*>(query + q_base);
  const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
  for (int d4 = 0; d4 < dim4; ++d4) {
    acc += dot(q4[d4], k4[d4]);
  }
  out[idx] = acc * p.SCALE;
}

[[kernel]] void na1d_av_fp32(
    device const float* attn [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * out_l * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int oq = t % out_l;
  int b = t / out_l;

  int qidx = oq * p.S;
  if (qidx >= p.L) {
    out[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  float acc = 0.0f;
  int attn_base = (((b * out_l + oq) * p.H + h) * p.K);
  for (int kpos = 0; kpos < p.K; ++kpos) {
    int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL)
                        : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
    if (!valid_1d(kidx, p.L)) {
      continue;
    }
    float w = attn[attn_base + kpos];
    acc += w * value[base_1d(b, kidx, h, d, p.L, p.H, p.D)];
  }
  out[idx] = acc;
}

[[kernel]] void na1d_av_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * out_l * p.H * dim4;
  if (idx >= total || dim4 <= 0) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int oq = t % out_l;
  int b = t / out_l;

  int qidx = oq * p.S;
  if (qidx >= p.L) {
    int out_base = (((b * out_l + oq) * p.H + h) * p.D + d0);
    out[out_base] = 0.0f;
    out[out_base + 1] = 0.0f;
    out[out_base + 2] = 0.0f;
    out[out_base + 3] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  float4 acc = float4(0.0f);
  int attn_base = (((b * out_l + oq) * p.H + h) * p.K);
  for (int kpos = 0; kpos < p.K; ++kpos) {
    int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL)
                        : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
    if (!valid_1d(kidx, p.L)) {
      continue;
    }
    float w = attn[attn_base + kpos];
    int v_base = base_1d(b, kidx, h, 0, p.L, p.H, p.D);
    const device float4* v4 = reinterpret_cast<const device float4*>(value + v_base);
    acc += w * v4[d4];
  }
  int out_base = (((b * out_l + oq) * p.H + h) * p.D + d0);
  out[out_base] = acc.x;
  out[out_base + 1] = acc.y;
  out[out_base + 2] = acc.z;
  out[out_base + 3] = acc.w;
}

[[kernel]] void na1d_qk_bwd_q_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* grad_q [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int i = t % p.L;
  int b = t / p.L;

  if ((i % p.S) != 0) {
    grad_q[idx] = 0.0f;
    return;
  }
  int oq = i / p.S;
  if (oq >= out_l) {
    grad_q[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int attn_base = (((b * out_l + oq) * p.H + h) * p.K);
  float acc = 0.0f;
  for (int kpos = 0; kpos < p.K; ++kpos) {
    int kidx = p.CAUSAL ? (i - (p.K - 1 - kpos) * p.DIL)
                        : (natten_get_window_start(i, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
    if (!valid_1d(kidx, p.L)) {
      continue;
    }
    acc += grad_attn[attn_base + kpos] * key[base_1d(b, kidx, h, d, p.L, p.H, p.D)];
  }
  grad_q[idx] = acc * p.SCALE;
}

[[kernel]] void na1d_qk_bwd_k_accum_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device atomic<float>* grad_k [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
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
  int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
  if (!valid_1d(kidx, p.L)) {
    return;
  }

  int attn_idx = (((b * out_l + oq) * p.H + h) * p.K + kpos);
  float g = grad_attn[attn_idx] * query[base_1d(b, qidx, h, d, p.L, p.H, p.D)] * p.SCALE;
  atomic_fetch_add_explicit(&grad_k[base_1d(b, kidx, h, d, p.L, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na1d_av_bwd_attn_fp32(
    device const float* grad_out [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* grad_attn [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
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
    grad_attn[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
  if (!valid_1d(kidx, p.L)) {
    grad_attn[idx] = 0.0f;
    return;
  }

  float acc = 0.0f;
  int go_base = (((b * out_l + oq) * p.H + h) * p.D);
  int v_base = (((b * p.L + kidx) * p.H + h) * p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += grad_out[go_base + d] * value[v_base + d];
  }
  grad_attn[idx] = acc;
}

[[kernel]] void na1d_av_bwd_v_accum_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device atomic<float>* grad_v [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
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
  int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
  if (!valid_1d(kidx, p.L)) {
    return;
  }

  int attn_idx = (((b * out_l + oq) * p.H + h) * p.K + kpos);
  int go_idx = (((b * out_l + oq) * p.H + h) * p.D + d);
  float g = attn[attn_idx] * grad_out[go_idx];
  atomic_fetch_add_explicit(&grad_v[base_1d(b, kidx, h, d, p.L, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na1d_qk_bwd_k_direct_u1d1_nc_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device float* grad_k [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int kidx = t % p.L;
  int b = t / p.L;

  int nh = p.K / 2;
  int out_l = p.L;
  int q_begin = max(0, kidx - nh);
  int q_end = min(p.L - 1, kidx + nh);
  float acc = 0.0f;
  for (int qidx = q_begin; qidx <= q_end; ++qidx) {
    int start = natten_get_window_start(qidx, p.L, p.K, nh, 1);
    int kpos = kidx - start;
    if (kpos < 0 || kpos >= p.K) {
      continue;
    }
    int attn_idx = (((b * out_l + qidx) * p.H + h) * p.K + kpos);
    acc += grad_attn[attn_idx] * query[base_1d(b, qidx, h, d, p.L, p.H, p.D)] * p.SCALE;
  }
  grad_k[idx] = acc;
}

[[kernel]] void na1d_qk_bwd_k_direct_u1d1_nc_vec4_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device float* grad_k [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * dim4;
  if (idx >= total || dim4 <= 0) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int kidx = t % p.L;
  int b = t / p.L;

  int nh = p.K / 2;
  int out_l = p.L;
  int q_begin = max(0, kidx - nh);
  int q_end = min(p.L - 1, kidx + nh);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int qidx = q_begin; qidx <= q_end; ++qidx) {
    int start = natten_get_window_start(qidx, p.L, p.K, nh, 1);
    int kpos = kidx - start;
    if (kpos < 0 || kpos >= p.K) {
      continue;
    }
    int attn_idx = (((b * out_l + qidx) * p.H + h) * p.K + kpos);
    float g = grad_attn[attn_idx] * p.SCALE;
    int q_base = base_1d(b, qidx, h, d0, p.L, p.H, p.D);
    acc0 += g * query[q_base];
    acc1 += g * query[q_base + 1];
    acc2 += g * query[q_base + 2];
    acc3 += g * query[q_base + 3];
  }
  int out_base = base_1d(b, kidx, h, d0, p.L, p.H, p.D);
  grad_k[out_base] = acc0;
  grad_k[out_base + 1] = acc1;
  grad_k[out_base + 2] = acc2;
  grad_k[out_base + 3] = acc3;
}

[[kernel]] void na1d_av_bwd_v_direct_u1d1_nc_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_v [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int kidx = t % p.L;
  int b = t / p.L;

  int nh = p.K / 2;
  int out_l = p.L;
  int q_begin = max(0, kidx - nh);
  int q_end = min(p.L - 1, kidx + nh);
  float acc = 0.0f;
  for (int qidx = q_begin; qidx <= q_end; ++qidx) {
    int start = natten_get_window_start(qidx, p.L, p.K, nh, 1);
    int kpos = kidx - start;
    if (kpos < 0 || kpos >= p.K) {
      continue;
    }
    int attn_idx = (((b * out_l + qidx) * p.H + h) * p.K + kpos);
    int go_idx = (((b * out_l + qidx) * p.H + h) * p.D + d);
    acc += attn[attn_idx] * grad_out[go_idx];
  }
  grad_v[idx] = acc;
}

[[kernel]] void na1d_av_bwd_v_direct_u1d1_nc_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_v [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * dim4;
  if (idx >= total || dim4 <= 0) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int kidx = t % p.L;
  int b = t / p.L;

  int nh = p.K / 2;
  int out_l = p.L;
  int q_begin = max(0, kidx - nh);
  int q_end = min(p.L - 1, kidx + nh);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int qidx = q_begin; qidx <= q_end; ++qidx) {
    int start = natten_get_window_start(qidx, p.L, p.K, nh, 1);
    int kpos = kidx - start;
    if (kpos < 0 || kpos >= p.K) {
      continue;
    }
    int attn_idx = (((b * out_l + qidx) * p.H + h) * p.K + kpos);
    int go_idx = (((b * out_l + qidx) * p.H + h) * p.D + d0);
    float w = attn[attn_idx];
    acc0 += w * grad_out[go_idx];
    acc1 += w * grad_out[go_idx + 1];
    acc2 += w * grad_out[go_idx + 2];
    acc3 += w * grad_out[go_idx + 3];
  }
  int out_base = base_1d(b, kidx, h, d0, p.L, p.H, p.D);
  grad_v[out_base] = acc0;
  grad_v[out_base + 1] = acc1;
  grad_v[out_base + 2] = acc2;
  grad_v[out_base + 3] = acc3;
}

[[kernel]] void na1d_qk_bwd_k_direct_s1_causal_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device float* grad_k [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * p.D;
  if (idx >= total || p.S != 1 || p.CAUSAL == 0) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int kidx = t % p.L;
  int b = t / p.L;

  int out_l = p.L;
  int q_begin = kidx;
  int q_end = min(p.L - 1, kidx + (p.K - 1) * p.DIL);
  float acc = 0.0f;
  for (int qidx = q_begin; qidx <= q_end; ++qidx) {
    int delta = qidx - kidx;
    if (delta < 0 || (delta % p.DIL) != 0) {
      continue;
    }
    int step = delta / p.DIL;
    if (step < 0 || step >= p.K) {
      continue;
    }
    int kpos = p.K - 1 - step;
    int attn_idx = (((b * out_l + qidx) * p.H + h) * p.K + kpos);
    acc += grad_attn[attn_idx] * query[base_1d(b, qidx, h, d, p.L, p.H, p.D)];
  }
  grad_k[idx] = acc * p.SCALE;
}

[[kernel]] void na1d_qk_bwd_k_direct_s1_causal_vec4_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device float* grad_k [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * dim4;
  if (idx >= total || dim4 <= 0 || p.S != 1 || p.CAUSAL == 0) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int kidx = t % p.L;
  int b = t / p.L;

  int out_l = p.L;
  int q_begin = kidx;
  int q_end = min(p.L - 1, kidx + (p.K - 1) * p.DIL);
  float4 acc = float4(0.0f);
  for (int qidx = q_begin; qidx <= q_end; ++qidx) {
    int delta = qidx - kidx;
    if (delta < 0 || (delta % p.DIL) != 0) {
      continue;
    }
    int step = delta / p.DIL;
    if (step < 0 || step >= p.K) {
      continue;
    }
    int kpos = p.K - 1 - step;
    int attn_idx = (((b * out_l + qidx) * p.H + h) * p.K + kpos);
    float g = grad_attn[attn_idx];
    int q_base = base_1d(b, qidx, h, d0, p.L, p.H, p.D);
    acc.x += g * query[q_base];
    acc.y += g * query[q_base + 1];
    acc.z += g * query[q_base + 2];
    acc.w += g * query[q_base + 3];
  }
  acc *= p.SCALE;
  int out_base = base_1d(b, kidx, h, d0, p.L, p.H, p.D);
  grad_k[out_base] = acc.x;
  grad_k[out_base + 1] = acc.y;
  grad_k[out_base + 2] = acc.z;
  grad_k[out_base + 3] = acc.w;
}

[[kernel]] void na1d_av_bwd_v_direct_s1_causal_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_v [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * p.D;
  if (idx >= total || p.S != 1 || p.CAUSAL == 0) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int kidx = t % p.L;
  int b = t / p.L;

  int out_l = p.L;
  int q_begin = kidx;
  int q_end = min(p.L - 1, kidx + (p.K - 1) * p.DIL);
  float acc = 0.0f;
  for (int qidx = q_begin; qidx <= q_end; ++qidx) {
    int delta = qidx - kidx;
    if (delta < 0 || (delta % p.DIL) != 0) {
      continue;
    }
    int step = delta / p.DIL;
    if (step < 0 || step >= p.K) {
      continue;
    }
    int kpos = p.K - 1 - step;
    int attn_idx = (((b * out_l + qidx) * p.H + h) * p.K + kpos);
    int go_idx = (((b * out_l + qidx) * p.H + h) * p.D + d);
    acc += attn[attn_idx] * grad_out[go_idx];
  }
  grad_v[idx] = acc;
}

[[kernel]] void na1d_av_bwd_v_direct_s1_causal_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_v [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * dim4;
  if (idx >= total || dim4 <= 0 || p.S != 1 || p.CAUSAL == 0) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int kidx = t % p.L;
  int b = t / p.L;

  int out_l = p.L;
  int q_begin = kidx;
  int q_end = min(p.L - 1, kidx + (p.K - 1) * p.DIL);
  float4 acc = float4(0.0f);
  for (int qidx = q_begin; qidx <= q_end; ++qidx) {
    int delta = qidx - kidx;
    if (delta < 0 || (delta % p.DIL) != 0) {
      continue;
    }
    int step = delta / p.DIL;
    if (step < 0 || step >= p.K) {
      continue;
    }
    int kpos = p.K - 1 - step;
    int attn_idx = (((b * out_l + qidx) * p.H + h) * p.K + kpos);
    int go_idx = (((b * out_l + qidx) * p.H + h) * p.D + d0);
    float w = attn[attn_idx];
    acc.x += w * grad_out[go_idx];
    acc.y += w * grad_out[go_idx + 1];
    acc.z += w * grad_out[go_idx + 2];
    acc.w += w * grad_out[go_idx + 3];
  }
  int out_base = base_1d(b, kidx, h, d0, p.L, p.H, p.D);
  grad_v[out_base] = acc.x;
  grad_v[out_base + 1] = acc.y;
  grad_v[out_base + 2] = acc.z;
  grad_v[out_base + 3] = acc.w;
}

[[kernel]] void na2d_qk_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k2 = p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * k2;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k2;
  int kh = kpos / p.K;
  int kw = kpos % p.K;
  int t = idx / k2;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    out[idx] = -INFINITY;
    return;
  }

  int nh = p.K / 2;
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    out[idx] = -INFINITY;
    return;
  }

  float acc = 0.0f;
  int q_base = base_2d(b, qh, qw, h, 0, p.IH, p.IW, p.H, p.D);
  int k_base = base_2d(b, ih, iw, h, 0, p.IH, p.IW, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += query[q_base + d] * key[k_base + d];
  }
  out[idx] = acc * p.SCALE;
}

[[kernel]] void na2d_qk_vec4_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k2 = p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * k2;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k2;
  int kh = kpos / p.K;
  int kw = kpos % p.K;
  int t = idx / k2;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    out[idx] = -INFINITY;
    return;
  }

  int nh = p.K / 2;
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    out[idx] = -INFINITY;
    return;
  }

  int q_base = base_2d(b, qh, qw, h, 0, p.IH, p.IW, p.H, p.D);
  int k_base = base_2d(b, ih, iw, h, 0, p.IH, p.IW, p.H, p.D);
  int dim4 = p.D / 4;
  float acc = 0.0f;
  const device float4* q4 = reinterpret_cast<const device float4*>(query + q_base);
  const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
  for (int d4 = 0; d4 < dim4; ++d4) {
    acc += dot(q4[d4], k4[d4]);
  }
  out[idx] = acc * p.SCALE;
}

[[kernel]] void na2d_av_fp32(
    device const float* attn [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    out[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  float acc = 0.0f;
  int attn_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * (p.K * p.K));
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                  : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                    : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      float w = attn[attn_base + kh * p.K + kw];
      acc += w * value[base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)];
    }
  }
  out[idx] = acc;
}

[[kernel]] void na2d_av_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * dim4;
  if (idx >= total || dim4 <= 0) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    int out_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D + d0);
    out[out_base] = 0.0f;
    out[out_base + 1] = 0.0f;
    out[out_base + 2] = 0.0f;
    out[out_base + 3] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  float4 acc = float4(0.0f);
  int attn_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * (p.K * p.K));
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                  : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                    : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      float w = attn[attn_base + kh * p.K + kw];
      int v_base = base_2d(b, ih, iw, h, 0, p.IH, p.IW, p.H, p.D);
      const device float4* v4 = reinterpret_cast<const device float4*>(value + v_base);
      acc += w * v4[d4];
    }
  }
  int out_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D + d0);
  out[out_base] = acc.x;
  out[out_base + 1] = acc.y;
  out[out_base + 2] = acc.z;
  out[out_base + 3] = acc.w;
}

[[kernel]] void na2d_qk_bwd_q_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* grad_q [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int idx = static_cast<int>(tid);
  int total = p.B * p.IH * p.IW * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int w = t % p.IW;
  t /= p.IW;
  int i = t % p.IH;
  int b = t / p.IH;

  if ((i % p.SH) != 0 || (w % p.SW) != 0) {
    grad_q[idx] = 0.0f;
    return;
  }
  int oh = i / p.SH;
  int ow = w / p.SW;
  if (oh >= out_h || ow >= out_w) {
    grad_q[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int attn_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * (p.K * p.K));
  float acc = 0.0f;
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = p.CH ? (i - (p.K - 1 - kh) * p.DH)
                  : (natten_get_window_start(i, p.IH, p.K, nh, p.DH) + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW ? (w - (p.K - 1 - kw) * p.DW)
                    : (natten_get_window_start(w, p.IW, p.K, nh, p.DW) + kw * p.DW);
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      float g = grad_attn[attn_base + kh * p.K + kw];
      acc += g * key[base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)];
    }
  }
  grad_q[idx] = acc * p.SCALE;
}

[[kernel]] void na2d_qk_bwd_q_vec4_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* grad_q [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.D % 4 != 0) {
    return;
  }
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.IH * p.IW * p.H * dim4;
  if (idx >= total) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int w = t % p.IW;
  t /= p.IW;
  int i = t % p.IH;
  int b = t / p.IH;

  int out_base = ((((b * p.IH + i) * p.IW + w) * p.H + h) * p.D + d0);
  if ((i % p.SH) != 0 || (w % p.SW) != 0) {
    grad_q[out_base] = 0.0f;
    grad_q[out_base + 1] = 0.0f;
    grad_q[out_base + 2] = 0.0f;
    grad_q[out_base + 3] = 0.0f;
    return;
  }
  int oh = i / p.SH;
  int ow = w / p.SW;
  if (oh >= out_h || ow >= out_w) {
    grad_q[out_base] = 0.0f;
    grad_q[out_base + 1] = 0.0f;
    grad_q[out_base + 2] = 0.0f;
    grad_q[out_base + 3] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int attn_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * (p.K * p.K));
  float4 acc = float4(0.0f);
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = p.CH ? (i - (p.K - 1 - kh) * p.DH)
                  : (natten_get_window_start(i, p.IH, p.K, nh, p.DH) + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW ? (w - (p.K - 1 - kw) * p.DW)
                    : (natten_get_window_start(w, p.IW, p.K, nh, p.DW) + kw * p.DW);
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      float g = grad_attn[attn_base + kh * p.K + kw];
      int k_base = base_2d(b, ih, iw, h, 0, p.IH, p.IW, p.H, p.D);
      const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
      acc += g * k4[d4];
    }
  }
  acc *= p.SCALE;
  grad_q[out_base] = acc.x;
  grad_q[out_base + 1] = acc.y;
  grad_q[out_base + 2] = acc.z;
  grad_q[out_base + 3] = acc.w;
}

[[kernel]] void na2d_qk_bwd_k_accum_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device atomic<float>* grad_k [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
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
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int attn_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * k2 + kpos);
  float g = grad_attn[attn_idx] * query[base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D)] * p.SCALE;
  atomic_fetch_add_explicit(&grad_k[base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na2d_qk_bwd_k_direct_u1d1_nc_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device float* grad_k [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.IH * p.IW * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int iw = t % p.IW;
  t /= p.IW;
  int ih = t % p.IH;
  int b = t / p.IH;

  int nh = p.K / 2;
  int k2 = p.K * p.K;
  int qh_min = max(0, ih - (p.K - 1));
  int qh_max = min(p.IH - 1, ih + (p.K - 1));
  int qw_min = max(0, iw - (p.K - 1));
  int qw_max = min(p.IW - 1, iw + (p.K - 1));

  float acc = 0.0f;
  for (int qh = qh_min; qh <= qh_max; ++qh) {
    int hs = natten_get_window_start(qh, p.IH, p.K, nh, 1);
    int kh = ih - hs;
    if (kh < 0 || kh >= p.K) {
      continue;
    }
    for (int qw = qw_min; qw <= qw_max; ++qw) {
      int ws = natten_get_window_start(qw, p.IW, p.K, nh, 1);
      int kw = iw - ws;
      if (kw < 0 || kw >= p.K) {
        continue;
      }
      int kpos = kh * p.K + kw;
      int attn_idx = ((((b * p.IH + qh) * p.IW + qw) * p.H + h) * k2 + kpos);
      acc += grad_attn[attn_idx] * query[base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D)] *
          p.SCALE;
    }
  }
  grad_k[idx] = acc;
}

[[kernel]] void na2d_qk_bwd_k_direct_u1d1_nc_vec4_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device float* grad_k [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.IH * p.IW * p.H * dim4;
  if (idx >= total || dim4 <= 0) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int iw = t % p.IW;
  t /= p.IW;
  int ih = t % p.IH;
  int b = t / p.IH;

  int nh = p.K / 2;
  int k2 = p.K * p.K;
  int qh_min = max(0, ih - (p.K - 1));
  int qh_max = min(p.IH - 1, ih + (p.K - 1));
  int qw_min = max(0, iw - (p.K - 1));
  int qw_max = min(p.IW - 1, iw + (p.K - 1));

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int qh = qh_min; qh <= qh_max; ++qh) {
    int hs = natten_get_window_start(qh, p.IH, p.K, nh, 1);
    int kh = ih - hs;
    if (kh < 0 || kh >= p.K) {
      continue;
    }
    for (int qw = qw_min; qw <= qw_max; ++qw) {
      int ws = natten_get_window_start(qw, p.IW, p.K, nh, 1);
      int kw = iw - ws;
      if (kw < 0 || kw >= p.K) {
        continue;
      }
      int kpos = kh * p.K + kw;
      int attn_idx = ((((b * p.IH + qh) * p.IW + qw) * p.H + h) * k2 + kpos);
      int q_base = base_2d(b, qh, qw, h, d0, p.IH, p.IW, p.H, p.D);
      float g = grad_attn[attn_idx] * p.SCALE;
      acc0 += g * query[q_base];
      acc1 += g * query[q_base + 1];
      acc2 += g * query[q_base + 2];
      acc3 += g * query[q_base + 3];
    }
  }
  int out_base = base_2d(b, ih, iw, h, d0, p.IH, p.IW, p.H, p.D);
  grad_k[out_base] = acc0;
  grad_k[out_base + 1] = acc1;
  grad_k[out_base + 2] = acc2;
  grad_k[out_base + 3] = acc3;
}

template <int KFIX>
inline void na2d_qk_bwd_k_direct_u1d1_nc_k_impl(
    device const float* grad_attn,
    device const float* query,
    device float* grad_k,
    constant NA2DParams& p,
    uint tid) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.IH * p.IW * p.H * p.D;
  if (idx >= total || p.K != KFIX) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int iw = t % p.IW;
  t /= p.IW;
  int ih = t % p.IH;
  int b = t / p.IH;

  constexpr int NH = KFIX / 2;
  constexpr int K2 = KFIX * KFIX;
  int qh_min = max(0, ih - (KFIX - 1));
  int qh_max = min(p.IH - 1, ih + (KFIX - 1));
  int qw_min = max(0, iw - (KFIX - 1));
  int qw_max = min(p.IW - 1, iw + (KFIX - 1));

  float acc = 0.0f;
  for (int qh = qh_min; qh <= qh_max; ++qh) {
    int hs = natten_get_window_start(qh, p.IH, KFIX, NH, 1);
    int kh = ih - hs;
    if (kh < 0 || kh >= KFIX) {
      continue;
    }
    for (int qw = qw_min; qw <= qw_max; ++qw) {
      int ws = natten_get_window_start(qw, p.IW, KFIX, NH, 1);
      int kw = iw - ws;
      if (kw < 0 || kw >= KFIX) {
        continue;
      }
      int kpos = kh * KFIX + kw;
      int attn_idx = ((((b * p.IH + qh) * p.IW + qw) * p.H + h) * K2 + kpos);
      acc += grad_attn[attn_idx] * query[base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D)];
    }
  }
  grad_k[idx] = acc * p.SCALE;
}

template <int KFIX>
inline void na2d_qk_bwd_k_direct_u1d1_nc_vec4_k_impl(
    device const float* grad_attn,
    device const float* query,
    device float* grad_k,
    constant NA2DParams& p,
    uint tid) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.IH * p.IW * p.H * dim4;
  if (idx >= total || dim4 <= 0 || p.K != KFIX) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int iw = t % p.IW;
  t /= p.IW;
  int ih = t % p.IH;
  int b = t / p.IH;

  constexpr int NH = KFIX / 2;
  constexpr int K2 = KFIX * KFIX;
  int qh_min = max(0, ih - (KFIX - 1));
  int qh_max = min(p.IH - 1, ih + (KFIX - 1));
  int qw_min = max(0, iw - (KFIX - 1));
  int qw_max = min(p.IW - 1, iw + (KFIX - 1));

  float4 acc = float4(0.0f);
  for (int qh = qh_min; qh <= qh_max; ++qh) {
    int hs = natten_get_window_start(qh, p.IH, KFIX, NH, 1);
    int kh = ih - hs;
    if (kh < 0 || kh >= KFIX) {
      continue;
    }
    for (int qw = qw_min; qw <= qw_max; ++qw) {
      int ws = natten_get_window_start(qw, p.IW, KFIX, NH, 1);
      int kw = iw - ws;
      if (kw < 0 || kw >= KFIX) {
        continue;
      }
      int kpos = kh * KFIX + kw;
      int attn_idx = ((((b * p.IH + qh) * p.IW + qw) * p.H + h) * K2 + kpos);
      float g = grad_attn[attn_idx];
      int q_base = base_2d(b, qh, qw, h, d0, p.IH, p.IW, p.H, p.D);
      acc.x += g * query[q_base];
      acc.y += g * query[q_base + 1];
      acc.z += g * query[q_base + 2];
      acc.w += g * query[q_base + 3];
    }
  }
  acc *= p.SCALE;
  int out_base = base_2d(b, ih, iw, h, d0, p.IH, p.IW, p.H, p.D);
  grad_k[out_base] = acc.x;
  grad_k[out_base + 1] = acc.y;
  grad_k[out_base + 2] = acc.z;
  grad_k[out_base + 3] = acc.w;
}

[[kernel]] void na2d_qk_bwd_k_direct_u1d1_nc_k7_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device float* grad_k [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  na2d_qk_bwd_k_direct_u1d1_nc_k_impl<7>(grad_attn, query, grad_k, p, tid);
}

[[kernel]] void na2d_qk_bwd_k_direct_u1d1_nc_k7_vec4_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device float* grad_k [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  na2d_qk_bwd_k_direct_u1d1_nc_vec4_k_impl<7>(grad_attn, query, grad_k, p, tid);
}

[[kernel]] void na2d_av_bwd_attn_fp32(
    device const float* grad_out [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* grad_attn [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k2 = p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * k2;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k2;
  int kh = kpos / p.K;
  int kw = kpos % p.K;
  int t = idx / k2;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    grad_attn[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    grad_attn[idx] = 0.0f;
    return;
  }

  float acc = 0.0f;
  int go_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D);
  int v_base = base_2d(b, ih, iw, h, 0, p.IH, p.IW, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += grad_out[go_base + d] * value[v_base + d];
  }
  grad_attn[idx] = acc;
}

[[kernel]] void na2d_av_bwd_v_accum_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device atomic<float>* grad_v [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
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
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int attn_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * k2 + kpos);
  int go_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D + d);
  float g = attn[attn_idx] * grad_out[go_idx];
  atomic_fetch_add_explicit(&grad_v[base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na2d_av_bwd_v_direct_u1d1_nc_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_v [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.IH * p.IW * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int iw = t % p.IW;
  t /= p.IW;
  int ih = t % p.IH;
  int b = t / p.IH;

  int nh = p.K / 2;
  int k2 = p.K * p.K;
  int qh_min = max(0, ih - (p.K - 1));
  int qh_max = min(p.IH - 1, ih + (p.K - 1));
  int qw_min = max(0, iw - (p.K - 1));
  int qw_max = min(p.IW - 1, iw + (p.K - 1));

  float acc = 0.0f;
  for (int qh = qh_min; qh <= qh_max; ++qh) {
    int hs = natten_get_window_start(qh, p.IH, p.K, nh, 1);
    int kh = ih - hs;
    if (kh < 0 || kh >= p.K) {
      continue;
    }
    for (int qw = qw_min; qw <= qw_max; ++qw) {
      int ws = natten_get_window_start(qw, p.IW, p.K, nh, 1);
      int kw = iw - ws;
      if (kw < 0 || kw >= p.K) {
        continue;
      }
      int kpos = kh * p.K + kw;
      int attn_idx = ((((b * p.IH + qh) * p.IW + qw) * p.H + h) * k2 + kpos);
      int go_idx = ((((b * p.IH + qh) * p.IW + qw) * p.H + h) * p.D + d);
      acc += attn[attn_idx] * grad_out[go_idx];
    }
  }
  grad_v[idx] = acc;
}

[[kernel]] void na2d_av_bwd_v_direct_u1d1_nc_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_v [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.IH * p.IW * p.H * dim4;
  if (idx >= total || dim4 <= 0) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int iw = t % p.IW;
  t /= p.IW;
  int ih = t % p.IH;
  int b = t / p.IH;

  int nh = p.K / 2;
  int k2 = p.K * p.K;
  int qh_min = max(0, ih - (p.K - 1));
  int qh_max = min(p.IH - 1, ih + (p.K - 1));
  int qw_min = max(0, iw - (p.K - 1));
  int qw_max = min(p.IW - 1, iw + (p.K - 1));

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int qh = qh_min; qh <= qh_max; ++qh) {
    int hs = natten_get_window_start(qh, p.IH, p.K, nh, 1);
    int kh = ih - hs;
    if (kh < 0 || kh >= p.K) {
      continue;
    }
    for (int qw = qw_min; qw <= qw_max; ++qw) {
      int ws = natten_get_window_start(qw, p.IW, p.K, nh, 1);
      int kw = iw - ws;
      if (kw < 0 || kw >= p.K) {
        continue;
      }
      int kpos = kh * p.K + kw;
      int attn_idx = ((((b * p.IH + qh) * p.IW + qw) * p.H + h) * k2 + kpos);
      int go_base = ((((b * p.IH + qh) * p.IW + qw) * p.H + h) * p.D + d0);
      float a = attn[attn_idx];
      acc0 += a * grad_out[go_base];
      acc1 += a * grad_out[go_base + 1];
      acc2 += a * grad_out[go_base + 2];
      acc3 += a * grad_out[go_base + 3];
    }
  }
  int out_base = base_2d(b, ih, iw, h, d0, p.IH, p.IW, p.H, p.D);
  grad_v[out_base] = acc0;
  grad_v[out_base + 1] = acc1;
  grad_v[out_base + 2] = acc2;
  grad_v[out_base + 3] = acc3;
}

[[kernel]] void na3d_qk_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k3 = p.K * p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * k3;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k3;
  int kd = kpos / (p.K * p.K);
  int rem = kpos % (p.K * p.K);
  int kh = rem / p.K;
  int kw = rem % p.K;
  int t = idx / k3;
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
    out[idx] = -INFINITY;
    return;
  }

  int nh = p.K / 2;
  int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    out[idx] = -INFINITY;
    return;
  }

  float acc = 0.0f;
  int q_base = base_3d(b, qd, qh, qw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
  int k_base = base_3d(b, id, ih, iw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += query[q_base + d] * key[k_base + d];
  }
  out[idx] = acc * p.SCALE;
}

[[kernel]] void na3d_qk_vec4_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k3 = p.K * p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * k3;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k3;
  int kd = kpos / (p.K * p.K);
  int rem = kpos % (p.K * p.K);
  int kh = rem / p.K;
  int kw = rem % p.K;
  int t = idx / k3;
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
    out[idx] = -INFINITY;
    return;
  }

  int nh = p.K / 2;
  int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    out[idx] = -INFINITY;
    return;
  }

  int q_base = base_3d(b, qd, qh, qw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
  int k_base = base_3d(b, id, ih, iw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
  int dim4 = p.D / 4;
  float acc = 0.0f;
  const device float4* q4 = reinterpret_cast<const device float4*>(query + q_base);
  const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
  for (int d4 = 0; d4 < dim4; ++d4) {
    acc += dot(q4[d4], k4[d4]);
  }
  out[idx] = acc * p.SCALE;
}

[[kernel]] void na3d_av_fp32(
    device const float* attn [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
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
    out[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  float acc = 0.0f;
  int k3 = p.K * p.K * p.K;
  int attn_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3);
  for (int kd = 0; kd < p.K; ++kd) {
    int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                  : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                    : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                      : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int kpos = (kd * p.K + kh) * p.K + kw;
        float w = attn[attn_base + kpos];
        acc += w * value[base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)];
      }
    }
  }
  out[idx] = acc;
}

[[kernel]] void na3d_av_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * dim4;
  if (idx >= total || dim4 <= 0) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
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
    int out_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D + d0);
    out[out_base] = 0.0f;
    out[out_base + 1] = 0.0f;
    out[out_base + 2] = 0.0f;
    out[out_base + 3] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  float4 acc = float4(0.0f);
  int k3 = p.K * p.K * p.K;
  int attn_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3);
  for (int kd = 0; kd < p.K; ++kd) {
    int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                  : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                    : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                      : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int kpos = (kd * p.K + kh) * p.K + kw;
        float w = attn[attn_base + kpos];
        int v_base = base_3d(b, id, ih, iw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
        const device float4* v4 = reinterpret_cast<const device float4*>(value + v_base);
        acc += w * v4[d4];
      }
    }
  }
  int out_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D + d0);
  out[out_base] = acc.x;
  out[out_base + 1] = acc.y;
  out[out_base + 2] = acc.z;
  out[out_base + 3] = acc.w;
}

[[kernel]] void na3d_av_k3_fp32(
    device const float* attn [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  t /= out_h;
  int od = t % out_d;
  int b = t / out_d;

  int out_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D + d);
  if (p.K != 3) {
    out[out_base] = 0.0f;
    return;
  }

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) {
    out[out_base] = 0.0f;
    return;
  }

  int start_d = p.CD ? (qd - 2 * p.DD) : natten_get_window_start(qd, p.ID, 3, 1, p.DD);
  int start_h = p.CH ? (qh - 2 * p.DH) : natten_get_window_start(qh, p.IH, 3, 1, p.DH);
  int start_w = p.CW ? (qw - 2 * p.DW) : natten_get_window_start(qw, p.IW, 3, 1, p.DW);
  int attn_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * 27);
  float acc = 0.0f;
  for (int kd = 0; kd < 3; ++kd) {
    int id = start_d + kd * p.DD;
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < 3; ++kh) {
      int ih = start_h + kh * p.DH;
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < 3; ++kw) {
        int iw = start_w + kw * p.DW;
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int kpos = (kd * 3 + kh) * 3 + kw;
        float w = attn[attn_base + kpos];
        acc += w * value[base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)];
      }
    }
  }
  out[out_base] = acc;
}

[[kernel]] void na3d_av_k3_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * dim4;
  if (idx >= total || dim4 <= 0) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  t /= out_h;
  int od = t % out_d;
  int b = t / out_d;

  int out_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D + d0);
  if (p.K != 3) {
    out[out_base] = 0.0f;
    out[out_base + 1] = 0.0f;
    out[out_base + 2] = 0.0f;
    out[out_base + 3] = 0.0f;
    return;
  }

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) {
    out[out_base] = 0.0f;
    out[out_base + 1] = 0.0f;
    out[out_base + 2] = 0.0f;
    out[out_base + 3] = 0.0f;
    return;
  }

  int start_d = p.CD ? (qd - 2 * p.DD) : natten_get_window_start(qd, p.ID, 3, 1, p.DD);
  int start_h = p.CH ? (qh - 2 * p.DH) : natten_get_window_start(qh, p.IH, 3, 1, p.DH);
  int start_w = p.CW ? (qw - 2 * p.DW) : natten_get_window_start(qw, p.IW, 3, 1, p.DW);
  int attn_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * 27);
  float4 acc = float4(0.0f);
  for (int kd = 0; kd < 3; ++kd) {
    int id = start_d + kd * p.DD;
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < 3; ++kh) {
      int ih = start_h + kh * p.DH;
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < 3; ++kw) {
        int iw = start_w + kw * p.DW;
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int kpos = (kd * 3 + kh) * 3 + kw;
        float w = attn[attn_base + kpos];
        int v_base = base_3d(b, id, ih, iw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
        const device float4* v4 = reinterpret_cast<const device float4*>(value + v_base);
        acc += w * v4[0];
      }
    }
  }
  out[out_base] = acc.x;
  out[out_base + 1] = acc.y;
  out[out_base + 2] = acc.z;
  out[out_base + 3] = acc.w;
}

[[kernel]] void na3d_qk_bwd_q_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* grad_q [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int idx = static_cast<int>(tid);
  int total = p.B * p.ID * p.IH * p.IW * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int w = t % p.IW;
  t /= p.IW;
  int i = t % p.IH;
  t /= p.IH;
  int z = t % p.ID;
  int b = t / p.ID;

  if ((z % p.SD) != 0 || (i % p.SH) != 0 || (w % p.SW) != 0) {
    grad_q[idx] = 0.0f;
    return;
  }
  int od = z / p.SD;
  int oh = i / p.SH;
  int ow = w / p.SW;
  if (od >= out_d || oh >= out_h || ow >= out_w) {
    grad_q[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int k3 = p.K * p.K * p.K;
  int attn_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3);
  float acc = 0.0f;
  for (int kd = 0; kd < p.K; ++kd) {
    int id = p.CD ? (z - (p.K - 1 - kd) * p.DD)
                  : (natten_get_window_start(z, p.ID, p.K, nh, p.DD) + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH ? (i - (p.K - 1 - kh) * p.DH)
                    : (natten_get_window_start(i, p.IH, p.K, nh, p.DH) + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW ? (w - (p.K - 1 - kw) * p.DW)
                      : (natten_get_window_start(w, p.IW, p.K, nh, p.DW) + kw * p.DW);
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int kpos = (kd * p.K + kh) * p.K + kw;
        float g = grad_attn[attn_base + kpos];
        acc += g * key[base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)];
      }
    }
  }
  grad_q[idx] = acc * p.SCALE;
}

[[kernel]] void na3d_qk_bwd_q_vec4_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* grad_q [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.ID * p.IH * p.IW * p.H * dim4;
  if (idx >= total || dim4 <= 0) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int w = t % p.IW;
  t /= p.IW;
  int i = t % p.IH;
  t /= p.IH;
  int z = t % p.ID;
  int b = t / p.ID;

  int out_base = base_3d(b, z, i, w, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
  if ((z % p.SD) != 0 || (i % p.SH) != 0 || (w % p.SW) != 0) {
    grad_q[out_base] = 0.0f;
    grad_q[out_base + 1] = 0.0f;
    grad_q[out_base + 2] = 0.0f;
    grad_q[out_base + 3] = 0.0f;
    return;
  }
  int od = z / p.SD;
  int oh = i / p.SH;
  int ow = w / p.SW;
  if (od >= out_d || oh >= out_h || ow >= out_w) {
    grad_q[out_base] = 0.0f;
    grad_q[out_base + 1] = 0.0f;
    grad_q[out_base + 2] = 0.0f;
    grad_q[out_base + 3] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int k3 = p.K * p.K * p.K;
  int attn_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3);
  float4 acc = float4(0.0f);
  for (int kd = 0; kd < p.K; ++kd) {
    int id = p.CD ? (z - (p.K - 1 - kd) * p.DD)
                  : (natten_get_window_start(z, p.ID, p.K, nh, p.DD) + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH ? (i - (p.K - 1 - kh) * p.DH)
                    : (natten_get_window_start(i, p.IH, p.K, nh, p.DH) + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW ? (w - (p.K - 1 - kw) * p.DW)
                      : (natten_get_window_start(w, p.IW, p.K, nh, p.DW) + kw * p.DW);
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int kpos = (kd * p.K + kh) * p.K + kw;
        float g = grad_attn[attn_base + kpos];
        int key_base = base_3d(b, id, ih, iw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
        const device float4* k4 = reinterpret_cast<const device float4*>(key + key_base);
        acc += g * k4[0];
      }
    }
  }
  acc *= p.SCALE;
  grad_q[out_base] = acc.x;
  grad_q[out_base + 1] = acc.y;
  grad_q[out_base + 2] = acc.z;
  grad_q[out_base + 3] = acc.w;
}

[[kernel]] void na3d_qk_bwd_k_accum_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device atomic<float>* grad_k [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
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
  int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int attn_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3 + kpos);
  float g = grad_attn[attn_idx] * query[base_3d(b, qd, qh, qw, h, d, p.ID, p.IH, p.IW, p.H, p.D)] * p.SCALE;
  atomic_fetch_add_explicit(&grad_k[base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na3d_qk_bwd_k_direct_u1d1_nc_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device float* grad_k [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.ID * p.IH * p.IW * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int iw = t % p.IW;
  t /= p.IW;
  int ih = t % p.IH;
  t /= p.IH;
  int id = t % p.ID;
  int b = t / p.ID;

  int nh = p.K / 2;
  int k3 = p.K * p.K * p.K;
  int qd_min = max(0, id - (p.K - 1));
  int qd_max = min(p.ID - 1, id + (p.K - 1));
  int qh_min = max(0, ih - (p.K - 1));
  int qh_max = min(p.IH - 1, ih + (p.K - 1));
  int qw_min = max(0, iw - (p.K - 1));
  int qw_max = min(p.IW - 1, iw + (p.K - 1));

  float acc = 0.0f;
  for (int qd = qd_min; qd <= qd_max; ++qd) {
    int ds = natten_get_window_start(qd, p.ID, p.K, nh, 1);
    int kd = id - ds;
    if (kd < 0 || kd >= p.K) {
      continue;
    }
    for (int qh = qh_min; qh <= qh_max; ++qh) {
      int hs = natten_get_window_start(qh, p.IH, p.K, nh, 1);
      int kh = ih - hs;
      if (kh < 0 || kh >= p.K) {
        continue;
      }
      for (int qw = qw_min; qw <= qw_max; ++qw) {
        int ws = natten_get_window_start(qw, p.IW, p.K, nh, 1);
        int kw = iw - ws;
        if (kw < 0 || kw >= p.K) {
          continue;
        }
        int kpos = (kd * p.K + kh) * p.K + kw;
        int attn_idx = (((((b * p.ID + qd) * p.IH + qh) * p.IW + qw) * p.H + h) * k3 + kpos);
        acc += grad_attn[attn_idx] *
            query[base_3d(b, qd, qh, qw, h, d, p.ID, p.IH, p.IW, p.H, p.D)] * p.SCALE;
      }
    }
  }
  grad_k[idx] = acc;
}

[[kernel]] void na3d_qk_bwd_k_direct_u1d1_nc_vec4_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device float* grad_k [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.ID * p.IH * p.IW * p.H * dim4;
  if (idx >= total || dim4 <= 0) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int iw = t % p.IW;
  t /= p.IW;
  int ih = t % p.IH;
  t /= p.IH;
  int id = t % p.ID;
  int b = t / p.ID;

  int nh = p.K / 2;
  int k3 = p.K * p.K * p.K;
  int qd_min = max(0, id - (p.K - 1));
  int qd_max = min(p.ID - 1, id + (p.K - 1));
  int qh_min = max(0, ih - (p.K - 1));
  int qh_max = min(p.IH - 1, ih + (p.K - 1));
  int qw_min = max(0, iw - (p.K - 1));
  int qw_max = min(p.IW - 1, iw + (p.K - 1));

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int qd = qd_min; qd <= qd_max; ++qd) {
    int ds = natten_get_window_start(qd, p.ID, p.K, nh, 1);
    int kd = id - ds;
    if (kd < 0 || kd >= p.K) {
      continue;
    }
    for (int qh = qh_min; qh <= qh_max; ++qh) {
      int hs = natten_get_window_start(qh, p.IH, p.K, nh, 1);
      int kh = ih - hs;
      if (kh < 0 || kh >= p.K) {
        continue;
      }
      for (int qw = qw_min; qw <= qw_max; ++qw) {
        int ws = natten_get_window_start(qw, p.IW, p.K, nh, 1);
        int kw = iw - ws;
        if (kw < 0 || kw >= p.K) {
          continue;
        }
        int kpos = (kd * p.K + kh) * p.K + kw;
        int attn_idx = (((((b * p.ID + qd) * p.IH + qh) * p.IW + qw) * p.H + h) * k3 + kpos);
        int q_base = base_3d(b, qd, qh, qw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
        float g = grad_attn[attn_idx] * p.SCALE;
        acc0 += g * query[q_base];
        acc1 += g * query[q_base + 1];
        acc2 += g * query[q_base + 2];
        acc3 += g * query[q_base + 3];
      }
    }
  }
  int out_base = base_3d(b, id, ih, iw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
  grad_k[out_base] = acc0;
  grad_k[out_base + 1] = acc1;
  grad_k[out_base + 2] = acc2;
  grad_k[out_base + 3] = acc3;
}

template <int KFIX>
inline void na3d_qk_bwd_k_direct_u1d1_nc_k_impl(
    device const float* grad_attn,
    device const float* query,
    device float* grad_k,
    constant NA3DParams& p,
    uint tid) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.ID * p.IH * p.IW * p.H * p.D;
  if (idx >= total || p.K != KFIX) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int iw = t % p.IW;
  t /= p.IW;
  int ih = t % p.IH;
  t /= p.IH;
  int id = t % p.ID;
  int b = t / p.ID;

  constexpr int NH = KFIX / 2;
  constexpr int K3 = KFIX * KFIX * KFIX;
  int qd_min = max(0, id - (KFIX - 1));
  int qd_max = min(p.ID - 1, id + (KFIX - 1));
  int qh_min = max(0, ih - (KFIX - 1));
  int qh_max = min(p.IH - 1, ih + (KFIX - 1));
  int qw_min = max(0, iw - (KFIX - 1));
  int qw_max = min(p.IW - 1, iw + (KFIX - 1));

  float acc = 0.0f;
  for (int qd = qd_min; qd <= qd_max; ++qd) {
    int ds = natten_get_window_start(qd, p.ID, KFIX, NH, 1);
    int kd = id - ds;
    if (kd < 0 || kd >= KFIX) {
      continue;
    }
    for (int qh = qh_min; qh <= qh_max; ++qh) {
      int hs = natten_get_window_start(qh, p.IH, KFIX, NH, 1);
      int kh = ih - hs;
      if (kh < 0 || kh >= KFIX) {
        continue;
      }
      for (int qw = qw_min; qw <= qw_max; ++qw) {
        int ws = natten_get_window_start(qw, p.IW, KFIX, NH, 1);
        int kw = iw - ws;
        if (kw < 0 || kw >= KFIX) {
          continue;
        }
        int kpos = (kd * KFIX + kh) * KFIX + kw;
        int attn_idx = (((((b * p.ID + qd) * p.IH + qh) * p.IW + qw) * p.H + h) * K3 + kpos);
        acc +=
            grad_attn[attn_idx] * query[base_3d(b, qd, qh, qw, h, d, p.ID, p.IH, p.IW, p.H, p.D)];
      }
    }
  }
  grad_k[idx] = acc * p.SCALE;
}

template <int KFIX>
inline void na3d_qk_bwd_k_direct_u1d1_nc_vec4_k_impl(
    device const float* grad_attn,
    device const float* query,
    device float* grad_k,
    constant NA3DParams& p,
    uint tid) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.ID * p.IH * p.IW * p.H * dim4;
  if (idx >= total || dim4 <= 0 || p.K != KFIX) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int iw = t % p.IW;
  t /= p.IW;
  int ih = t % p.IH;
  t /= p.IH;
  int id = t % p.ID;
  int b = t / p.ID;

  constexpr int NH = KFIX / 2;
  constexpr int K3 = KFIX * KFIX * KFIX;
  int qd_min = max(0, id - (KFIX - 1));
  int qd_max = min(p.ID - 1, id + (KFIX - 1));
  int qh_min = max(0, ih - (KFIX - 1));
  int qh_max = min(p.IH - 1, ih + (KFIX - 1));
  int qw_min = max(0, iw - (KFIX - 1));
  int qw_max = min(p.IW - 1, iw + (KFIX - 1));

  float4 acc = float4(0.0f);
  for (int qd = qd_min; qd <= qd_max; ++qd) {
    int ds = natten_get_window_start(qd, p.ID, KFIX, NH, 1);
    int kd = id - ds;
    if (kd < 0 || kd >= KFIX) {
      continue;
    }
    for (int qh = qh_min; qh <= qh_max; ++qh) {
      int hs = natten_get_window_start(qh, p.IH, KFIX, NH, 1);
      int kh = ih - hs;
      if (kh < 0 || kh >= KFIX) {
        continue;
      }
      for (int qw = qw_min; qw <= qw_max; ++qw) {
        int ws = natten_get_window_start(qw, p.IW, KFIX, NH, 1);
        int kw = iw - ws;
        if (kw < 0 || kw >= KFIX) {
          continue;
        }
        int kpos = (kd * KFIX + kh) * KFIX + kw;
        int attn_idx = (((((b * p.ID + qd) * p.IH + qh) * p.IW + qw) * p.H + h) * K3 + kpos);
        float g = grad_attn[attn_idx];
        int q_base = base_3d(b, qd, qh, qw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
        acc.x += g * query[q_base];
        acc.y += g * query[q_base + 1];
        acc.z += g * query[q_base + 2];
        acc.w += g * query[q_base + 3];
      }
    }
  }
  acc *= p.SCALE;
  int out_base = base_3d(b, id, ih, iw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
  grad_k[out_base] = acc.x;
  grad_k[out_base + 1] = acc.y;
  grad_k[out_base + 2] = acc.z;
  grad_k[out_base + 3] = acc.w;
}

[[kernel]] void na3d_qk_bwd_k_direct_u1d1_nc_k3_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device float* grad_k [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  na3d_qk_bwd_k_direct_u1d1_nc_k_impl<3>(grad_attn, query, grad_k, p, tid);
}

[[kernel]] void na3d_qk_bwd_k_direct_u1d1_nc_k3_vec4_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device float* grad_k [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  na3d_qk_bwd_k_direct_u1d1_nc_vec4_k_impl<3>(grad_attn, query, grad_k, p, tid);
}

[[kernel]] void na3d_av_bwd_attn_fp32(
    device const float* grad_out [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* grad_attn [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k3 = p.K * p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * k3;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k3;
  int kd = kpos / (p.K * p.K);
  int rem = kpos % (p.K * p.K);
  int kh = rem / p.K;
  int kw = rem % p.K;
  int t = idx / k3;
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
    grad_attn[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    grad_attn[idx] = 0.0f;
    return;
  }

  float acc = 0.0f;
  int go_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D);
  int v_base = base_3d(b, id, ih, iw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += grad_out[go_base + d] * value[v_base + d];
  }
  grad_attn[idx] = acc;
}

[[kernel]] void na3d_av_bwd_attn_vec4_fp32(
    device const float* grad_out [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* grad_attn [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k3 = p.K * p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * k3;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k3;
  int kd = kpos / (p.K * p.K);
  int rem = kpos % (p.K * p.K);
  int kh = rem / p.K;
  int kw = rem % p.K;
  int t = idx / k3;
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
    grad_attn[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    grad_attn[idx] = 0.0f;
    return;
  }

  float acc = 0.0f;
  int go_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D);
  int v_base = base_3d(b, id, ih, iw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
  int dim4 = p.D / 4;
  for (int d4 = 0; d4 < dim4; ++d4) {
    int off = d4 * 4;
    const device float4* go4 = reinterpret_cast<const device float4*>(grad_out + go_base + off);
    const device float4* v4 = reinterpret_cast<const device float4*>(value + v_base + off);
    float4 prod = go4[0] * v4[0];
    acc += (prod.x + prod.y + prod.z + prod.w);
  }
  for (int d = dim4 * 4; d < p.D; ++d) {
    acc += grad_out[go_base + d] * value[v_base + d];
  }
  grad_attn[idx] = acc;
}

[[kernel]] void na3d_av_bwd_v_accum_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device atomic<float>* grad_v [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
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
  int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int attn_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3 + kpos);
  int go_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D + d);
  float g = attn[attn_idx] * grad_out[go_idx];
  atomic_fetch_add_explicit(&grad_v[base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na3d_av_bwd_v_direct_u1d1_nc_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_v [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.ID * p.IH * p.IW * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int iw = t % p.IW;
  t /= p.IW;
  int ih = t % p.IH;
  t /= p.IH;
  int id = t % p.ID;
  int b = t / p.ID;

  int nh = p.K / 2;
  int k3 = p.K * p.K * p.K;
  int qd_min = max(0, id - (p.K - 1));
  int qd_max = min(p.ID - 1, id + (p.K - 1));
  int qh_min = max(0, ih - (p.K - 1));
  int qh_max = min(p.IH - 1, ih + (p.K - 1));
  int qw_min = max(0, iw - (p.K - 1));
  int qw_max = min(p.IW - 1, iw + (p.K - 1));

  float acc = 0.0f;
  for (int qd = qd_min; qd <= qd_max; ++qd) {
    int ds = natten_get_window_start(qd, p.ID, p.K, nh, 1);
    int kd = id - ds;
    if (kd < 0 || kd >= p.K) {
      continue;
    }
    for (int qh = qh_min; qh <= qh_max; ++qh) {
      int hs = natten_get_window_start(qh, p.IH, p.K, nh, 1);
      int kh = ih - hs;
      if (kh < 0 || kh >= p.K) {
        continue;
      }
      for (int qw = qw_min; qw <= qw_max; ++qw) {
        int ws = natten_get_window_start(qw, p.IW, p.K, nh, 1);
        int kw = iw - ws;
        if (kw < 0 || kw >= p.K) {
          continue;
        }
        int kpos = (kd * p.K + kh) * p.K + kw;
        int attn_idx = (((((b * p.ID + qd) * p.IH + qh) * p.IW + qw) * p.H + h) * k3 + kpos);
        int go_idx = (((((b * p.ID + qd) * p.IH + qh) * p.IW + qw) * p.H + h) * p.D + d);
        acc += attn[attn_idx] * grad_out[go_idx];
      }
    }
  }
  grad_v[idx] = acc;
}

[[kernel]] void na3d_av_bwd_v_direct_u1d1_nc_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_v [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.ID * p.IH * p.IW * p.H * dim4;
  if (idx >= total || dim4 <= 0) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int iw = t % p.IW;
  t /= p.IW;
  int ih = t % p.IH;
  t /= p.IH;
  int id = t % p.ID;
  int b = t / p.ID;

  int nh = p.K / 2;
  int k3 = p.K * p.K * p.K;
  int qd_min = max(0, id - (p.K - 1));
  int qd_max = min(p.ID - 1, id + (p.K - 1));
  int qh_min = max(0, ih - (p.K - 1));
  int qh_max = min(p.IH - 1, ih + (p.K - 1));
  int qw_min = max(0, iw - (p.K - 1));
  int qw_max = min(p.IW - 1, iw + (p.K - 1));

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int qd = qd_min; qd <= qd_max; ++qd) {
    int ds = natten_get_window_start(qd, p.ID, p.K, nh, 1);
    int kd = id - ds;
    if (kd < 0 || kd >= p.K) {
      continue;
    }
    for (int qh = qh_min; qh <= qh_max; ++qh) {
      int hs = natten_get_window_start(qh, p.IH, p.K, nh, 1);
      int kh = ih - hs;
      if (kh < 0 || kh >= p.K) {
        continue;
      }
      for (int qw = qw_min; qw <= qw_max; ++qw) {
        int ws = natten_get_window_start(qw, p.IW, p.K, nh, 1);
        int kw = iw - ws;
        if (kw < 0 || kw >= p.K) {
          continue;
        }
        int kpos = (kd * p.K + kh) * p.K + kw;
        int attn_idx = (((((b * p.ID + qd) * p.IH + qh) * p.IW + qw) * p.H + h) * k3 + kpos);
        int go_base = (((((b * p.ID + qd) * p.IH + qh) * p.IW + qw) * p.H + h) * p.D + d0);
        float a = attn[attn_idx];
        acc0 += a * grad_out[go_base];
        acc1 += a * grad_out[go_base + 1];
        acc2 += a * grad_out[go_base + 2];
        acc3 += a * grad_out[go_base + 3];
      }
    }
  }
  int out_base = base_3d(b, id, ih, iw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
  grad_v[out_base] = acc0;
  grad_v[out_base + 1] = acc1;
  grad_v[out_base + 2] = acc2;
  grad_v[out_base + 3] = acc3;
}

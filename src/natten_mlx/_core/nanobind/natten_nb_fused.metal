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

inline float na_dot_2d_bf16_vec4(
    device const bfloat16_t* query,
    device const bfloat16_t* key,
    int q_base,
    int k_base,
    int dim,
    float scale) {
  float acc = 0.0f;
  int d4_count = dim / 4;
  for (int d4 = 0; d4 < d4_count; ++d4) {
    const device bfloat4* q4 = reinterpret_cast<const device bfloat4*>(query + q_base + d4 * 4);
    const device bfloat4* k4 = reinterpret_cast<const device bfloat4*>(key + k_base + d4 * 4);
    acc += dot(float4(*q4), float4(*k4));
  }
  for (int d = d4_count * 4; d < dim; ++d) {
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
        ? (qidx - (p.K - 1 - kk) * p.DIL)
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
        ? (qidx - (p.K - 1 - kk) * p.DIL)
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
        ? (qidx - (p.K - 1 - kk) * p.DIL)
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
        ? (qh - (p.K - 1 - kh) * p.DH)
        : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW
          ? (qw - (p.K - 1 - kw) * p.DW)
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
        ? (qh - (p.K - 1 - kh) * p.DH)
        : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW
          ? (qw - (p.K - 1 - kw) * p.DW)
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
        ? (qh - (p.K - 1 - kh) * p.DH)
        : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW
          ? (qw - (p.K - 1 - kw) * p.DW)
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
        ? (qd - (p.K - 1 - kd) * p.DD)
        : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH
          ? (qh - (p.K - 1 - kh) * p.DH)
          : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW
            ? (qw - (p.K - 1 - kw) * p.DW)
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
        ? (qd - (p.K - 1 - kd) * p.DD)
        : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH
          ? (qh - (p.K - 1 - kh) * p.DH)
          : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW
            ? (qw - (p.K - 1 - kw) * p.DW)
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
        ? (qd - (p.K - 1 - kd) * p.DD)
        : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH
          ? (qh - (p.K - 1 - kh) * p.DH)
          : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW
            ? (qw - (p.K - 1 - kw) * p.DW)
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
inline float na_dot_vec4(
    device const T* query,
    device const T* key,
    int q_base,
    int k_base,
    int dim,
    float scale) {
  int dim4 = dim / 4;
  float sum = 0.0f;
  for (int d4 = 0; d4 < dim4; ++d4) {
    int d0 = d4 * 4;
    sum += (float)query[q_base + d0] * (float)key[k_base + d0];
    sum += (float)query[q_base + d0 + 1] * (float)key[k_base + d0 + 1];
    sum += (float)query[q_base + d0 + 2] * (float)key[k_base + d0 + 2];
    sum += (float)query[q_base + d0 + 3] * (float)key[k_base + d0 + 3];
  }
  return sum * scale;
}

template <typename T>
inline void na1d_fused_vec4_impl(
    device const T* query,
    device const T* key,
    device const T* value,
    device T* out,
    constant NA1DParams& p,
    uint3 gid) {
  if ((p.D % 4) != 0) {
    na1d_fused_impl(query, key, value, out, p, gid);
    return;
  }
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
  int start = p.CAUSAL ? (qidx - (p.K - 1) * p.DIL) : natten_get_window_start(qidx, p.L, p.K, nh, p.DIL);
  int dim4 = p.D / 4;

  float max_score = -INFINITY;
  int valid_count = 0;
  for (int kk = 0; kk < p.K; ++kk) {
    int kidx = start + kk * p.DIL;
    if (kidx < 0 || kidx >= p.L) {
      continue;
    }
    int k_base = (((b * p.L + kidx) * p.H + h) * p.D);
    float s = na_dot_vec4(query, key, q_base, k_base, p.D, p.SCALE);
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
    int kidx = start + kk * p.DIL;
    if (kidx < 0 || kidx >= p.L) {
      continue;
    }
    int k_base = (((b * p.L + kidx) * p.H + h) * p.D);
    float s = na_dot_vec4(query, key, q_base, k_base, p.D, p.SCALE);
    sum_exp += exp(s - max_score);
  }
  float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;

  for (int d4 = 0; d4 < dim4; ++d4) {
    int d0 = d4 * 4;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (int kk = 0; kk < p.K; ++kk) {
      int kidx = start + kk * p.DIL;
      if (kidx < 0 || kidx >= p.L) {
        continue;
      }
      int k_base = (((b * p.L + kidx) * p.H + h) * p.D);
      float s = na_dot_vec4(query, key, q_base, k_base, p.D, p.SCALE);
      float w = exp(s - max_score) * inv_sum;
      int v_base = (((b * p.L + kidx) * p.H + h) * p.D + d0);
      acc0 += w * (float)value[v_base];
      acc1 += w * (float)value[v_base + 1];
      acc2 += w * (float)value[v_base + 2];
      acc3 += w * (float)value[v_base + 3];
    }
    out[out_base + d0] = (T)acc0;
    out[out_base + d0 + 1] = (T)acc1;
    out[out_base + d0 + 2] = (T)acc2;
    out[out_base + d0 + 3] = (T)acc3;
  }
}

template <typename T>
inline void na2d_fused_vec4_impl(
    device const T* query,
    device const T* key,
    device const T* value,
    device T* out,
    constant NA2DParams& p,
    uint3 gid) {
  if ((p.D % 4) != 0) {
    na2d_fused_impl(query, key, value, out, p, gid);
    return;
  }
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
  int h_start = p.CH ? (qh - (p.K - 1) * p.DH) : natten_get_window_start(qh, p.IH, p.K, nh, p.DH);
  int w_start = p.CW ? (qw - (p.K - 1) * p.DW) : natten_get_window_start(qw, p.IW, p.K, nh, p.DW);
  int dim4 = p.D / 4;

  float max_score = -INFINITY;
  int valid_count = 0;
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = h_start + kh * p.DH;
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = w_start + kw * p.DW;
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      int k_base = ((((b * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
      float s = na_dot_vec4(query, key, q_base, k_base, p.D, p.SCALE);
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
    int ih = h_start + kh * p.DH;
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = w_start + kw * p.DW;
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      int k_base = ((((b * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
      float s = na_dot_vec4(query, key, q_base, k_base, p.D, p.SCALE);
      sum_exp += exp(s - max_score);
    }
  }
  float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;

  for (int d4 = 0; d4 < dim4; ++d4) {
    int d0 = d4 * 4;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = h_start + kh * p.DH;
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = w_start + kw * p.DW;
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int k_base = ((((b * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
        float s = na_dot_vec4(query, key, q_base, k_base, p.D, p.SCALE);
        float w = exp(s - max_score) * inv_sum;
        int v_base = ((((b * p.IH + ih) * p.IW + iw) * p.H + h) * p.D + d0);
        acc0 += w * (float)value[v_base];
        acc1 += w * (float)value[v_base + 1];
        acc2 += w * (float)value[v_base + 2];
        acc3 += w * (float)value[v_base + 3];
      }
    }
    out[out_base + d0] = (T)acc0;
    out[out_base + d0 + 1] = (T)acc1;
    out[out_base + d0 + 2] = (T)acc2;
    out[out_base + d0 + 3] = (T)acc3;
  }
}

template <typename T>
inline void na3d_fused_vec4_impl(
    device const T* query,
    device const T* key,
    device const T* value,
    device T* out,
    constant NA3DParams& p,
    uint3 gid) {
  if ((p.D % 4) != 0) {
    na3d_fused_impl(query, key, value, out, p, gid);
    return;
  }
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
  int d_start = p.CD ? (qd - (p.K - 1) * p.DD) : natten_get_window_start(qd, p.ID, p.K, nh, p.DD);
  int h_start = p.CH ? (qh - (p.K - 1) * p.DH) : natten_get_window_start(qh, p.IH, p.K, nh, p.DH);
  int w_start = p.CW ? (qw - (p.K - 1) * p.DW) : natten_get_window_start(qw, p.IW, p.K, nh, p.DW);
  int dim4 = p.D / 4;

  float max_score = -INFINITY;
  int valid_count = 0;
  for (int kd = 0; kd < p.K; ++kd) {
    int id = d_start + kd * p.DD;
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = h_start + kh * p.DH;
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = w_start + kw * p.DW;
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int k_base = (((((b * p.ID + id) * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
        float s = na_dot_vec4(query, key, q_base, k_base, p.D, p.SCALE);
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
    int id = d_start + kd * p.DD;
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = h_start + kh * p.DH;
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = w_start + kw * p.DW;
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int k_base = (((((b * p.ID + id) * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
        float s = na_dot_vec4(query, key, q_base, k_base, p.D, p.SCALE);
        sum_exp += exp(s - max_score);
      }
    }
  }
  float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;

  for (int d4 = 0; d4 < dim4; ++d4) {
    int d0 = d4 * 4;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (int kd = 0; kd < p.K; ++kd) {
      int id = d_start + kd * p.DD;
      if (id < 0 || id >= p.ID) {
        continue;
      }
      for (int kh = 0; kh < p.K; ++kh) {
        int ih = h_start + kh * p.DH;
        if (ih < 0 || ih >= p.IH) {
          continue;
        }
        for (int kw = 0; kw < p.K; ++kw) {
          int iw = w_start + kw * p.DW;
          if (iw < 0 || iw >= p.IW) {
            continue;
          }
          int k_base = (((((b * p.ID + id) * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
          float s = na_dot_vec4(query, key, q_base, k_base, p.D, p.SCALE);
          float w = exp(s - max_score) * inv_sum;
          int v_base = (((((b * p.ID + id) * p.IH + ih) * p.IW + iw) * p.H + h) * p.D + d0);
          acc0 += w * (float)value[v_base];
          acc1 += w * (float)value[v_base + 1];
          acc2 += w * (float)value[v_base + 2];
          acc3 += w * (float)value[v_base + 3];
        }
      }
    }
    out[out_base + d0] = (T)acc0;
    out[out_base + d0 + 1] = (T)acc1;
    out[out_base + d0 + 2] = (T)acc2;
    out[out_base + d0 + 3] = (T)acc3;
  }
}

[[kernel]] void na2d_fused_strided_causal_h_bf16(
    device const bfloat16_t* query [[buffer(0)]],
    device const bfloat16_t* key [[buffer(1)]],
    device const bfloat16_t* value [[buffer(2)]],
    device bfloat16_t* out [[buffer(3)]],
    constant NA2DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;
  if ((int)gid.x >= out_w || (int)gid.y >= out_h || (int)gid.z >= p.B * p.H) {
    return;
  }
  if (p.D % 4 != 0 || p.K <= 0 || p.K > 15) {
    return;
  }
  if (!(p.CH == 1 && p.CW == 0)) {
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

  constexpr int MAX_EDGES = 15 * 15;
  float scores[MAX_EDGES];
  int key_bases[MAX_EDGES];
  int edge_count = 0;

  int q_base = ((((b * p.IH + qh) * p.IW + qw) * p.H + h) * p.D);
  int out_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D);
  int nh = p.K / 2;
  int w_start = natten_get_window_start(qw, p.IW, p.K, nh, p.DW);

  float max_score = -INFINITY;
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = qh - (p.K - 1 - kh) * p.DH;
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = w_start + kw * p.DW;
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      int k_base = ((((b * p.IH + ih) * p.IW + iw) * p.H + h) * p.D);
      float s = na_dot_2d_bf16_vec4(query, key, q_base, k_base, p.D, p.SCALE);
      scores[edge_count] = s;
      key_bases[edge_count] = k_base;
      max_score = max(max_score, s);
      edge_count += 1;
    }
  }

  if (edge_count == 0) {
    int d4_count = p.D / 4;
    for (int d4 = 0; d4 < d4_count; ++d4) {
      device bfloat4* out4 = reinterpret_cast<device bfloat4*>(out + out_base + d4 * 4);
      *out4 = bfloat4(float4(0.0f));
    }
    return;
  }

  float sum_exp = 0.0f;
  for (int i = 0; i < edge_count; ++i) {
    float w = exp(scores[i] - max_score);
    scores[i] = w;
    sum_exp += w;
  }
  float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;

  int d4_count = p.D / 4;
  for (int d4 = 0; d4 < d4_count; ++d4) {
    float4 acc = float4(0.0f);
    for (int i = 0; i < edge_count; ++i) {
      float w = scores[i] * inv_sum;
      const device bfloat4* v4 =
          reinterpret_cast<const device bfloat4*>(value + key_bases[i] + d4 * 4);
      acc += w * float4(*v4);
    }
    device bfloat4* out4 = reinterpret_cast<device bfloat4*>(out + out_base + d4 * 4);
    *out4 = bfloat4(acc);
  }
}

[[kernel]] void na2d_fused_strided_causal_h_k7d16_bf16(
    device const bfloat16_t* query [[buffer(0)]],
    device const bfloat16_t* key [[buffer(1)]],
    device const bfloat16_t* value [[buffer(2)]],
    device bfloat16_t* out [[buffer(3)]],
    constant NA2DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (!(p.K == 7 && p.D == 16 && p.SH == 2 && p.SW == 1 && p.DH == 1 && p.DW == 2 &&
          p.CH == 1 && p.CW == 0)) {
    return;
  }

  int out_h = (p.IH + 1) / 2;
  int out_w = p.IW;
  if ((int)gid.x >= out_w || (int)gid.y >= out_h || (int)gid.z >= p.B * p.H) {
    return;
  }

  int bh = (int)gid.z;
  int b = bh / p.H;
  int h = bh - b * p.H;
  int oh = (int)gid.y;
  int ow = (int)gid.x;
  int qh = oh * 2;
  int qw = ow;
  if (qh >= p.IH || qw >= p.IW) {
    return;
  }

  constexpr int MAX_EDGES = 49;
  float scores[MAX_EDGES];
  int key_bases[MAX_EDGES];
  int edge_count = 0;

  int q_base = ((((b * p.IH + qh) * p.IW + qw) * p.H + h) * 16);
  int out_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * 16);
  const device bfloat4* q40 = reinterpret_cast<const device bfloat4*>(query + q_base + 0);
  const device bfloat4* q41 = reinterpret_cast<const device bfloat4*>(query + q_base + 4);
  const device bfloat4* q42 = reinterpret_cast<const device bfloat4*>(query + q_base + 8);
  const device bfloat4* q43 = reinterpret_cast<const device bfloat4*>(query + q_base + 12);

  int w_start = natten_get_window_start(qw, p.IW, 7, 3, 2);
  float max_score = -INFINITY;
  for (int kh = 0; kh < 7; ++kh) {
    int ih = qh - (6 - kh);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < 7; ++kw) {
      int iw = w_start + kw * 2;
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      int k_base = ((((b * p.IH + ih) * p.IW + iw) * p.H + h) * 16);
      const device bfloat4* k40 = reinterpret_cast<const device bfloat4*>(key + k_base + 0);
      const device bfloat4* k41 = reinterpret_cast<const device bfloat4*>(key + k_base + 4);
      const device bfloat4* k42 = reinterpret_cast<const device bfloat4*>(key + k_base + 8);
      const device bfloat4* k43 = reinterpret_cast<const device bfloat4*>(key + k_base + 12);
      float s = (dot(float4(*q40), float4(*k40)) + dot(float4(*q41), float4(*k41)) +
                    dot(float4(*q42), float4(*k42)) + dot(float4(*q43), float4(*k43))) *
          p.SCALE;
      scores[edge_count] = s;
      key_bases[edge_count] = k_base;
      max_score = max(max_score, s);
      edge_count += 1;
    }
  }

  if (edge_count == 0) {
    device bfloat4* out40 = reinterpret_cast<device bfloat4*>(out + out_base + 0);
    device bfloat4* out41 = reinterpret_cast<device bfloat4*>(out + out_base + 4);
    device bfloat4* out42 = reinterpret_cast<device bfloat4*>(out + out_base + 8);
    device bfloat4* out43 = reinterpret_cast<device bfloat4*>(out + out_base + 12);
    *out40 = bfloat4(float4(0.0f));
    *out41 = bfloat4(float4(0.0f));
    *out42 = bfloat4(float4(0.0f));
    *out43 = bfloat4(float4(0.0f));
    return;
  }

  float sum_exp = 0.0f;
  for (int i = 0; i < edge_count; ++i) {
    float w = exp(scores[i] - max_score);
    scores[i] = w;
    sum_exp += w;
  }
  float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;

  float4 acc0 = float4(0.0f);
  float4 acc1 = float4(0.0f);
  float4 acc2 = float4(0.0f);
  float4 acc3 = float4(0.0f);
  for (int i = 0; i < edge_count; ++i) {
    float w = scores[i] * inv_sum;
    int v_base = key_bases[i];
    const device bfloat4* v40 = reinterpret_cast<const device bfloat4*>(value + v_base + 0);
    const device bfloat4* v41 = reinterpret_cast<const device bfloat4*>(value + v_base + 4);
    const device bfloat4* v42 = reinterpret_cast<const device bfloat4*>(value + v_base + 8);
    const device bfloat4* v43 = reinterpret_cast<const device bfloat4*>(value + v_base + 12);
    acc0 += w * float4(*v40);
    acc1 += w * float4(*v41);
    acc2 += w * float4(*v42);
    acc3 += w * float4(*v43);
  }

  device bfloat4* out40 = reinterpret_cast<device bfloat4*>(out + out_base + 0);
  device bfloat4* out41 = reinterpret_cast<device bfloat4*>(out + out_base + 4);
  device bfloat4* out42 = reinterpret_cast<device bfloat4*>(out + out_base + 8);
  device bfloat4* out43 = reinterpret_cast<device bfloat4*>(out + out_base + 12);
  *out40 = bfloat4(acc0);
  *out41 = bfloat4(acc1);
  *out42 = bfloat4(acc2);
  *out43 = bfloat4(acc3);
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
  na1d_fused_vec4_impl(query, key, value, out, p, gid);
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
  na2d_fused_vec4_impl(query, key, value, out, p, gid);
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
  na3d_fused_vec4_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na2d_fused_stored_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA2DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na2d_fused_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na2d_fused_recompute_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA2DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na2d_fused_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na2d_fused_stored_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA2DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na2d_fused_vec4_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na2d_fused_recompute_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA2DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na2d_fused_vec4_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na3d_fused_stored_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA3DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na3d_fused_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na3d_fused_recompute_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA3DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na3d_fused_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na3d_fused_stored_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA3DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na3d_fused_vec4_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na3d_fused_recompute_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA3DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  na3d_fused_vec4_impl(query, key, value, out, p, gid);
}

template <typename T>
[[kernel]] void na1d_fused_causal_k9_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA1DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (!(p.CAUSAL == 1 && p.K == 9 && p.S == 1 && p.DIL == 1)) {
    return;
  }
  na1d_fused_vec4_impl(query, key, value, out, p, gid);
}

[[kernel]] void na1d_fused_causal_k9d_vec4_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant NA1DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (!(p.CAUSAL == 1 && p.K == 9 && p.S == 1 && (p.D % 4) == 0 && p.DIL >= 1)) {
    return;
  }

  int out_l = p.L;
  if ((int)gid.x >= out_l || (int)gid.z >= p.B * p.H) {
    return;
  }
  int bh = (int)gid.z;
  int b = bh / p.H;
  int h = bh - b * p.H;
  int qidx = (int)gid.x;

  int q_base = (((b * p.L + qidx) * p.H + h) * p.D);
  int out_base = q_base;
  int dim4 = p.D / 4;
  const device float4* q4 = reinterpret_cast<const device float4*>(query + q_base);

  int k_begin = max(0, 8 - (qidx / p.DIL));
  float scores[9];
  int key_base[9];
  for (int i = 0; i < 9; ++i) {
    scores[i] = -INFINITY;
    key_base[i] = 0;
  }

  float max_score = -INFINITY;
  for (int kpos = k_begin; kpos < 9; ++kpos) {
    int kidx = qidx - (8 - kpos) * p.DIL;
    int kb = (((b * p.L + kidx) * p.H + h) * p.D);
    key_base[kpos] = kb;
    const device float4* k4 = reinterpret_cast<const device float4*>(key + kb);
    float s = 0.0f;
    for (int d4 = 0; d4 < dim4; ++d4) {
      s += dot(q4[d4], k4[d4]);
    }
    s *= p.SCALE;
    scores[kpos] = s;
    max_score = max(max_score, s);
  }

  if (!isfinite(max_score)) {
    int d4_count = p.D / 4;
    for (int d4 = 0; d4 < d4_count; ++d4) {
      device float4* out4 = reinterpret_cast<device float4*>(out + out_base + d4 * 4);
      *out4 = float4(0.0f);
    }
    return;
  }

  float sum_exp = 0.0f;
  for (int kpos = k_begin; kpos < 9; ++kpos) {
    float w = exp(scores[kpos] - max_score);
    scores[kpos] = w;
    sum_exp += w;
  }
  float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;

  for (int d4 = 0; d4 < dim4; ++d4) {
    float4 acc = float4(0.0f);
    for (int kpos = k_begin; kpos < 9; ++kpos) {
      const device float4* v4 = reinterpret_cast<const device float4*>(value + key_base[kpos] + d4 * 4);
      acc += (scores[kpos] * inv_sum) * (*v4);
    }
    device float4* out4 = reinterpret_cast<device float4*>(out + out_base + d4 * 4);
    *out4 = acc;
  }
}

template <typename T>
[[kernel]] void na3d_fused_causal_d_k3_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA3DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (!(p.K == 3 && p.CD == 1 && p.CH == 0 && p.CW == 0 && p.SD == 1 && p.SH == 1 && p.SW == 1 &&
          p.DD == 1 && p.DH == 1 && p.DW == 1)) {
    return;
  }
  na3d_fused_vec4_impl(query, key, value, out, p, gid);
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
      uint3 gid [[thread_position_in_grid]]);                                              \
  template [[host_name("na1d_fused_causal_k9_vec4_" #name)]] [[kernel]] void               \
      na1d_fused_causal_k9_vec4_kernel<type>(                                               \
          device const type * query [[buffer(0)]],                                          \
          device const type * key [[buffer(1)]],                                            \
          device const type * value [[buffer(2)]],                                          \
          device type * out [[buffer(3)]],                                                  \
          constant NA1DParams & p [[buffer(4)]],                                            \
          uint3 gid [[thread_position_in_grid]]);

#define INSTANTIATE_2D(name, type)                                                         \
  template [[host_name("na2d_fused_" #name)]] [[kernel]] void na2d_fused_kernel<type>(    \
      device const type * query [[buffer(0)]],                                             \
      device const type * key [[buffer(1)]],                                               \
      device const type * value [[buffer(2)]],                                             \
      device type * out [[buffer(3)]],                                                     \
      constant NA2DParams & p [[buffer(4)]],                                               \
      uint3 gid [[thread_position_in_grid]]);                                              \
  template [[host_name("na2d_fused_stored_" #name)]] [[kernel]] void na2d_fused_stored_kernel<type>( \
      device const type * query [[buffer(0)]],                                             \
      device const type * key [[buffer(1)]],                                               \
      device const type * value [[buffer(2)]],                                             \
      device type * out [[buffer(3)]],                                                     \
      constant NA2DParams & p [[buffer(4)]],                                               \
      uint3 gid [[thread_position_in_grid]]);                                              \
  template [[host_name("na2d_fused_recompute_" #name)]] [[kernel]] void na2d_fused_recompute_kernel<type>( \
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
      uint3 gid [[thread_position_in_grid]]);                                              \
  template [[host_name("na2d_fused_stored_vec4_" #name)]] [[kernel]] void na2d_fused_stored_vec4_kernel<type>( \
      device const type * query [[buffer(0)]],                                             \
      device const type * key [[buffer(1)]],                                               \
      device const type * value [[buffer(2)]],                                             \
      device type * out [[buffer(3)]],                                                     \
      constant NA2DParams & p [[buffer(4)]],                                               \
      uint3 gid [[thread_position_in_grid]]);                                              \
  template [[host_name("na2d_fused_recompute_vec4_" #name)]] [[kernel]] void na2d_fused_recompute_vec4_kernel<type>( \
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
  template [[host_name("na3d_fused_stored_" #name)]] [[kernel]] void na3d_fused_stored_kernel<type>( \
      device const type * query [[buffer(0)]],                                             \
      device const type * key [[buffer(1)]],                                               \
      device const type * value [[buffer(2)]],                                             \
      device type * out [[buffer(3)]],                                                     \
      constant NA3DParams & p [[buffer(4)]],                                               \
      uint3 gid [[thread_position_in_grid]]);                                              \
  template [[host_name("na3d_fused_recompute_" #name)]] [[kernel]] void na3d_fused_recompute_kernel<type>( \
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
      uint3 gid [[thread_position_in_grid]]);                                              \
  template [[host_name("na3d_fused_stored_vec4_" #name)]] [[kernel]] void na3d_fused_stored_vec4_kernel<type>( \
      device const type * query [[buffer(0)]],                                             \
      device const type * key [[buffer(1)]],                                               \
      device const type * value [[buffer(2)]],                                             \
      device type * out [[buffer(3)]],                                                     \
      constant NA3DParams & p [[buffer(4)]],                                               \
      uint3 gid [[thread_position_in_grid]]);                                              \
  template [[host_name("na3d_fused_recompute_vec4_" #name)]] [[kernel]] void na3d_fused_recompute_vec4_kernel<type>( \
      device const type * query [[buffer(0)]],                                             \
      device const type * key [[buffer(1)]],                                               \
      device const type * value [[buffer(2)]],                                             \
      device type * out [[buffer(3)]],                                                     \
      constant NA3DParams & p [[buffer(4)]],                                               \
      uint3 gid [[thread_position_in_grid]]);                                              \
  template [[host_name("na3d_fused_causal_d_k3_vec4_" #name)]] [[kernel]] void             \
      na3d_fused_causal_d_k3_vec4_kernel<type>(                                             \
          device const type * query [[buffer(0)]],                                          \
          device const type * key [[buffer(1)]],                                            \
          device const type * value [[buffer(2)]],                                          \
          device type * out [[buffer(3)]],                                                  \
          constant NA3DParams & p [[buffer(4)]],                                            \
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
  constexpr int MAX_K_1D_FAST = 31;
  int out_l = nb_ceil_div(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * out_l * p.H;
  if (idx >= total) {
    return;
  }

  int h = idx % p.H;
  int t = idx / p.H;
  int oq = t % out_l;
  int b = t / out_l;

  int qidx = oq * p.S;
  int attn_base = (((b * out_l + oq) * p.H + h) * p.K);
  int go_base = (((b * out_l + oq) * p.H + h) * p.D);
  if (qidx >= p.L) {
    for (int kk = 0; kk < p.K; ++kk) {
      attn[attn_base + kk] = 0.0f;
      grad_attn[attn_base + kk] = 0.0f;
    }
    return;
  }

  int nh = p.K / 2;
  int q_base = nb_base_1d(b, qidx, h, 0, p.L, p.H, p.D);
  int start = p.CAUSAL ? (qidx - (p.K - 1) * p.DIL)
                       : natten_get_window_start(qidx, p.L, p.K, nh, p.DIL);

  if (p.K <= MAX_K_1D_FAST) {
    float scores[MAX_K_1D_FAST];
    float gas[MAX_K_1D_FAST];
    uchar valid[MAX_K_1D_FAST];
    for (int kk = 0; kk < p.K; ++kk) {
      scores[kk] = 0.0f;
      gas[kk] = 0.0f;
      valid[kk] = 0;
    }

    float max_score = -INFINITY;
    for (int kk = 0; kk < p.K; ++kk) {
      int kidx = start + kk * p.DIL;
      if (kidx < 0 || kidx >= p.L) {
        continue;
      }
      int k_base = nb_base_1d(b, kidx, h, 0, p.L, p.H, p.D);
      float s = 0.0f;
      float ga = 0.0f;
      for (int d = 0; d < p.D; ++d) {
        s += query[q_base + d] * key[k_base + d];
        ga += grad_out[go_base + d] * value[k_base + d];
      }
      s *= p.SCALE;
      scores[kk] = s;
      gas[kk] = ga;
      valid[kk] = 1;
      max_score = max(max_score, s);
    }

    if (!isfinite(max_score)) {
      for (int kk = 0; kk < p.K; ++kk) {
        attn[attn_base + kk] = 0.0f;
        grad_attn[attn_base + kk] = 0.0f;
      }
      return;
    }

    float sum_exp = 0.0f;
    for (int kk = 0; kk < p.K; ++kk) {
      if (valid[kk] == 0) {
        continue;
      }
      sum_exp += exp(scores[kk] - max_score);
    }

    if (sum_exp <= 0.0f) {
      for (int kk = 0; kk < p.K; ++kk) {
        attn[attn_base + kk] = 0.0f;
        grad_attn[attn_base + kk] = 0.0f;
      }
      return;
    }

    float inv_sum = 1.0f / sum_exp;
    for (int kk = 0; kk < p.K; ++kk) {
      int out_idx = attn_base + kk;
      if (valid[kk] == 0) {
        attn[out_idx] = 0.0f;
        grad_attn[out_idx] = 0.0f;
        continue;
      }
      attn[out_idx] = exp(scores[kk] - max_score) * inv_sum;
      grad_attn[out_idx] = gas[kk];
    }
    return;
  }

  float max_score = -INFINITY;
  for (int kk = 0; kk < p.K; ++kk) {
    int k2 = start + kk * p.DIL;
    if (k2 < 0 || k2 >= p.L) {
      continue;
    }
    float s = 0.0f;
    int k_base = nb_base_1d(b, k2, h, 0, p.L, p.H, p.D);
    for (int d = 0; d < p.D; ++d) {
      s += query[q_base + d] * key[k_base + d];
    }
    s *= p.SCALE;
    max_score = max(max_score, s);
  }

  float sum_exp = 0.0f;
  for (int kk = 0; kk < p.K; ++kk) {
    int k2 = start + kk * p.DIL;
    if (k2 < 0 || k2 >= p.L) {
      continue;
    }
    float s = 0.0f;
    int k_base = nb_base_1d(b, k2, h, 0, p.L, p.H, p.D);
    for (int d = 0; d < p.D; ++d) {
      s += query[q_base + d] * key[k_base + d];
    }
    s *= p.SCALE;
    sum_exp += exp(s - max_score);
  }

  if (sum_exp <= 0.0f) {
    for (int kk = 0; kk < p.K; ++kk) {
      attn[attn_base + kk] = 0.0f;
      grad_attn[attn_base + kk] = 0.0f;
    }
    return;
  }

  float inv_sum = 1.0f / sum_exp;
  for (int kk = 0; kk < p.K; ++kk) {
    int kidx = start + kk * p.DIL;
    int out_idx = attn_base + kk;
    if (kidx < 0 || kidx >= p.L) {
      attn[out_idx] = 0.0f;
      grad_attn[out_idx] = 0.0f;
      continue;
    }
    int k_base = nb_base_1d(b, kidx, h, 0, p.L, p.H, p.D);
    float s_this = 0.0f;
    for (int d = 0; d < p.D; ++d) {
      s_this += query[q_base + d] * key[k_base + d];
    }
    s_this *= p.SCALE;
    float w = exp(s_this - max_score) * inv_sum;
    attn[out_idx] = w;

    float ga = 0.0f;
    int v_base = nb_base_1d(b, kidx, h, 0, p.L, p.H, p.D);
    for (int d = 0; d < p.D; ++d) {
      ga += grad_out[go_base + d] * value[v_base + d];
    }
    grad_attn[out_idx] = ga;
  }
}

[[kernel]] void na1d_fused_bwd_attn_vec4_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* grad_out [[buffer(3)]],
    device float* attn [[buffer(4)]],
    device float* grad_attn [[buffer(5)]],
    constant NA1DParams& p [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  constexpr int MAX_K_1D_FAST = 31;
  int dim4 = p.D / 4;
  int out_l = nb_ceil_div(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * out_l * p.H;
  if (idx >= total || dim4 <= 0 || (p.D % 4) != 0) {
    return;
  }

  int h = idx % p.H;
  int t = idx / p.H;
  int oq = t % out_l;
  int b = t / out_l;

  int qidx = oq * p.S;
  int attn_base = (((b * out_l + oq) * p.H + h) * p.K);
  int go_base = (((b * out_l + oq) * p.H + h) * p.D);
  if (qidx >= p.L) {
    for (int kk = 0; kk < p.K; ++kk) {
      attn[attn_base + kk] = 0.0f;
      grad_attn[attn_base + kk] = 0.0f;
    }
    return;
  }

  int nh = p.K / 2;
  int q_base = nb_base_1d(b, qidx, h, 0, p.L, p.H, p.D);
  int start = p.CAUSAL ? (qidx - (p.K - 1) * p.DIL)
                       : natten_get_window_start(qidx, p.L, p.K, nh, p.DIL);
  const device float4* q4 = reinterpret_cast<const device float4*>(query + q_base);
  const device float4* go4 = reinterpret_cast<const device float4*>(grad_out + go_base);

  if (p.K <= MAX_K_1D_FAST) {
    float scores[MAX_K_1D_FAST];
    float gas[MAX_K_1D_FAST];
    uchar valid[MAX_K_1D_FAST];
    for (int kk = 0; kk < p.K; ++kk) {
      scores[kk] = 0.0f;
      gas[kk] = 0.0f;
      valid[kk] = 0;
    }

    float max_score = -INFINITY;
    for (int kk = 0; kk < p.K; ++kk) {
      int kidx = start + kk * p.DIL;
      if (kidx < 0 || kidx >= p.L) {
        continue;
      }
      int k_base = nb_base_1d(b, kidx, h, 0, p.L, p.H, p.D);
      const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
      const device float4* v4 = reinterpret_cast<const device float4*>(value + k_base);
      float s = 0.0f;
      float ga = 0.0f;
      for (int d4 = 0; d4 < dim4; ++d4) {
        s += dot(q4[d4], k4[d4]);
        ga += dot(go4[d4], v4[d4]);
      }
      s *= p.SCALE;
      scores[kk] = s;
      gas[kk] = ga;
      valid[kk] = 1;
      max_score = max(max_score, s);
    }

    if (!isfinite(max_score)) {
      for (int kk = 0; kk < p.K; ++kk) {
        attn[attn_base + kk] = 0.0f;
        grad_attn[attn_base + kk] = 0.0f;
      }
      return;
    }

    float sum_exp = 0.0f;
    for (int kk = 0; kk < p.K; ++kk) {
      if (valid[kk] == 0) {
        continue;
      }
      sum_exp += exp(scores[kk] - max_score);
    }

    if (sum_exp <= 0.0f) {
      for (int kk = 0; kk < p.K; ++kk) {
        attn[attn_base + kk] = 0.0f;
        grad_attn[attn_base + kk] = 0.0f;
      }
      return;
    }

    float inv_sum = 1.0f / sum_exp;
    for (int kk = 0; kk < p.K; ++kk) {
      int out_idx = attn_base + kk;
      if (valid[kk] == 0) {
        attn[out_idx] = 0.0f;
        grad_attn[out_idx] = 0.0f;
        continue;
      }
      attn[out_idx] = exp(scores[kk] - max_score) * inv_sum;
      grad_attn[out_idx] = gas[kk];
    }
    return;
  }

  float max_score = -INFINITY;
  for (int kk = 0; kk < p.K; ++kk) {
    int k2 = start + kk * p.DIL;
    if (k2 < 0 || k2 >= p.L) {
      continue;
    }
    int k_base = nb_base_1d(b, k2, h, 0, p.L, p.H, p.D);
    const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
    float s = 0.0f;
    for (int d4 = 0; d4 < dim4; ++d4) {
      s += dot(q4[d4], k4[d4]);
    }
    s *= p.SCALE;
    max_score = max(max_score, s);
  }

  float sum_exp = 0.0f;
  for (int kk = 0; kk < p.K; ++kk) {
    int k2 = start + kk * p.DIL;
    if (k2 < 0 || k2 >= p.L) {
      continue;
    }
    int k_base = nb_base_1d(b, k2, h, 0, p.L, p.H, p.D);
    const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
    float s = 0.0f;
    for (int d4 = 0; d4 < dim4; ++d4) {
      s += dot(q4[d4], k4[d4]);
    }
    s *= p.SCALE;
    sum_exp += exp(s - max_score);
  }

  if (sum_exp <= 0.0f) {
    for (int kk = 0; kk < p.K; ++kk) {
      attn[attn_base + kk] = 0.0f;
      grad_attn[attn_base + kk] = 0.0f;
    }
    return;
  }

  float inv_sum = 1.0f / sum_exp;
  for (int kk = 0; kk < p.K; ++kk) {
    int kidx = start + kk * p.DIL;
    int out_idx = attn_base + kk;
    if (kidx < 0 || kidx >= p.L) {
      attn[out_idx] = 0.0f;
      grad_attn[out_idx] = 0.0f;
      continue;
    }
    int k_base = nb_base_1d(b, kidx, h, 0, p.L, p.H, p.D);
    const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
    const device float4* v4 = reinterpret_cast<const device float4*>(value + k_base);
    float s_this = 0.0f;
    float ga = 0.0f;
    for (int d4 = 0; d4 < dim4; ++d4) {
      s_this += dot(q4[d4], k4[d4]);
      ga += dot(go4[d4], v4[d4]);
    }
    s_this *= p.SCALE;
    float w = exp(s_this - max_score) * inv_sum;
    attn[out_idx] = w;
    grad_attn[out_idx] = ga;
  }
}

[[kernel]] void na1d_fused_bwd_attn_s1_causal_k9_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* grad_out [[buffer(3)]],
    device float* attn [[buffer(4)]],
    device float* grad_attn [[buffer(5)]],
    constant NA1DParams& p [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H;
  if (idx >= total || p.S != 1 || p.CAUSAL == 0 || p.K != 9) {
    return;
  }

  int h = idx % p.H;
  int t = idx / p.H;
  int qidx = t % p.L;
  int b = t / p.L;

  int attn_base = (((b * p.L + qidx) * p.H + h) * 9);
  int go_base = (((b * p.L + qidx) * p.H + h) * p.D);
  int q_base = nb_base_1d(b, qidx, h, 0, p.L, p.H, p.D);
  int k_begin = max(0, 8 - (qidx / p.DIL));

  float scores[9];
  float gas[9];
  for (int kk = 0; kk < 9; ++kk) {
    scores[kk] = -INFINITY;
    gas[kk] = 0.0f;
  }

  float max_score = -INFINITY;
  for (int kpos = k_begin; kpos < 9; ++kpos) {
    int kidx = qidx - (8 - kpos) * p.DIL;
    int k_base = nb_base_1d(b, kidx, h, 0, p.L, p.H, p.D);
    float s = 0.0f;
    float ga = 0.0f;
    for (int d = 0; d < p.D; ++d) {
      s += query[q_base + d] * key[k_base + d];
      ga += grad_out[go_base + d] * value[k_base + d];
    }
    s *= p.SCALE;
    scores[kpos] = s;
    gas[kpos] = ga;
    max_score = max(max_score, s);
  }

  if (!isfinite(max_score)) {
    for (int kk = 0; kk < 9; ++kk) {
      attn[attn_base + kk] = 0.0f;
      grad_attn[attn_base + kk] = 0.0f;
    }
    return;
  }

  float sum_exp = 0.0f;
  for (int kpos = k_begin; kpos < 9; ++kpos) {
    sum_exp += exp(scores[kpos] - max_score);
  }
  if (sum_exp <= 0.0f) {
    for (int kk = 0; kk < 9; ++kk) {
      attn[attn_base + kk] = 0.0f;
      grad_attn[attn_base + kk] = 0.0f;
    }
    return;
  }

  float inv_sum = 1.0f / sum_exp;
  for (int kk = 0; kk < 9; ++kk) {
    int out_idx = attn_base + kk;
    if (kk < k_begin) {
      attn[out_idx] = 0.0f;
      grad_attn[out_idx] = 0.0f;
      continue;
    }
    attn[out_idx] = exp(scores[kk] - max_score) * inv_sum;
    grad_attn[out_idx] = gas[kk];
  }
}

[[kernel]] void na1d_fused_bwd_attn_s1_causal_k9_vec4_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* grad_out [[buffer(3)]],
    device float* attn [[buffer(4)]],
    device float* grad_attn [[buffer(5)]],
    constant NA1DParams& p [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H;
  if (idx >= total || dim4 <= 0 || p.S != 1 || p.CAUSAL == 0 || p.K != 9 || (p.D % 4) != 0) {
    return;
  }

  int h = idx % p.H;
  int t = idx / p.H;
  int qidx = t % p.L;
  int b = t / p.L;

  int attn_base = (((b * p.L + qidx) * p.H + h) * 9);
  int go_base = (((b * p.L + qidx) * p.H + h) * p.D);
  int q_base = nb_base_1d(b, qidx, h, 0, p.L, p.H, p.D);
  int k_begin = max(0, 8 - (qidx / p.DIL));
  const device float4* q4 = reinterpret_cast<const device float4*>(query + q_base);
  const device float4* go4 = reinterpret_cast<const device float4*>(grad_out + go_base);

  float scores[9];
  float gas[9];
  for (int kk = 0; kk < 9; ++kk) {
    scores[kk] = -INFINITY;
    gas[kk] = 0.0f;
  }

  float max_score = -INFINITY;
  for (int kpos = k_begin; kpos < 9; ++kpos) {
    int kidx = qidx - (8 - kpos) * p.DIL;
    int k_base = nb_base_1d(b, kidx, h, 0, p.L, p.H, p.D);
    const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
    const device float4* v4 = reinterpret_cast<const device float4*>(value + k_base);
    float s = 0.0f;
    float ga = 0.0f;
    for (int d4 = 0; d4 < dim4; ++d4) {
      s += dot(q4[d4], k4[d4]);
      ga += dot(go4[d4], v4[d4]);
    }
    s *= p.SCALE;
    scores[kpos] = s;
    gas[kpos] = ga;
    max_score = max(max_score, s);
  }

  if (!isfinite(max_score)) {
    for (int kk = 0; kk < 9; ++kk) {
      attn[attn_base + kk] = 0.0f;
      grad_attn[attn_base + kk] = 0.0f;
    }
    return;
  }

  float sum_exp = 0.0f;
  for (int kpos = k_begin; kpos < 9; ++kpos) {
    sum_exp += exp(scores[kpos] - max_score);
  }
  if (sum_exp <= 0.0f) {
    for (int kk = 0; kk < 9; ++kk) {
      attn[attn_base + kk] = 0.0f;
      grad_attn[attn_base + kk] = 0.0f;
    }
    return;
  }

  float inv_sum = 1.0f / sum_exp;
  for (int kk = 0; kk < 9; ++kk) {
    int out_idx = attn_base + kk;
    if (kk < k_begin) {
      attn[out_idx] = 0.0f;
      grad_attn[out_idx] = 0.0f;
      continue;
    }
    attn[out_idx] = exp(scores[kk] - max_score) * inv_sum;
    grad_attn[out_idx] = gas[kk];
  }
}

[[kernel]] void na2d_fused_bwd_attn_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* grad_out [[buffer(3)]],
    device float* attn [[buffer(4)]],
    device float* grad_attn [[buffer(5)]],
    constant NA2DParams& p [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  constexpr int MAX_K_2D_FAST = 15;
  constexpr int MAX_K2_2D_FAST = MAX_K_2D_FAST * MAX_K_2D_FAST;
  int out_h = nb_ceil_div(p.IH, p.SH);
  int out_w = nb_ceil_div(p.IW, p.SW);
  int k2 = p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H;
  if (idx >= total) {
    return;
  }

  int h = idx % p.H;
  int t = idx / p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  int out_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * k2);
  if (qh >= p.IH || qw >= p.IW) {
    for (int kpos = 0; kpos < k2; ++kpos) {
      attn[out_base + kpos] = 0.0f;
      grad_attn[out_base + kpos] = 0.0f;
    }
    return;
  }

  int nh = p.K / 2;
  int h_start = p.CH ? (qh - (p.K - 1) * p.DH) : natten_get_window_start(qh, p.IH, p.K, nh, p.DH);
  int w_start = p.CW ? (qw - (p.K - 1) * p.DW) : natten_get_window_start(qw, p.IW, p.K, nh, p.DW);
  int q_base = nb_base_2d(b, qh, qw, h, 0, p.IH, p.IW, p.H, p.D);
  int go_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D);

  if (p.K <= MAX_K_2D_FAST) {
    float scores[MAX_K2_2D_FAST];
    float gas[MAX_K2_2D_FAST];
    uchar valid[MAX_K2_2D_FAST];
    for (int kpos = 0; kpos < k2; ++kpos) {
      scores[kpos] = 0.0f;
      gas[kpos] = 0.0f;
      valid[kpos] = 0;
    }

    float max_score = -INFINITY;
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = h_start + kh * p.DH;
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = w_start + kw * p.DW;
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int kpos = kh * p.K + kw;
        int k_base = nb_base_2d(b, ih, iw, h, 0, p.IH, p.IW, p.H, p.D);
        float s = 0.0f;
        float ga = 0.0f;
        for (int d = 0; d < p.D; ++d) {
          s += query[q_base + d] * key[k_base + d];
          ga += grad_out[go_base + d] * value[k_base + d];
        }
        s *= p.SCALE;
        scores[kpos] = s;
        gas[kpos] = ga;
        valid[kpos] = 1;
        max_score = max(max_score, s);
      }
    }

    if (!isfinite(max_score)) {
      for (int kpos = 0; kpos < k2; ++kpos) {
        attn[out_base + kpos] = 0.0f;
        grad_attn[out_base + kpos] = 0.0f;
      }
      return;
    }

    float sum_exp = 0.0f;
    for (int kpos = 0; kpos < k2; ++kpos) {
      if (valid[kpos] == 0) {
        continue;
      }
      sum_exp += exp(scores[kpos] - max_score);
    }

    if (sum_exp <= 0.0f) {
      for (int kpos = 0; kpos < k2; ++kpos) {
        attn[out_base + kpos] = 0.0f;
        grad_attn[out_base + kpos] = 0.0f;
      }
      return;
    }

    float inv_sum = 1.0f / sum_exp;
    for (int kpos = 0; kpos < k2; ++kpos) {
      int out_idx = out_base + kpos;
      if (valid[kpos] == 0) {
        attn[out_idx] = 0.0f;
        grad_attn[out_idx] = 0.0f;
        continue;
      }
      attn[out_idx] = exp(scores[kpos] - max_score) * inv_sum;
      grad_attn[out_idx] = gas[kpos];
    }
    return;
  }

  float max_score = -INFINITY;
  for (int kk_h = 0; kk_h < p.K; ++kk_h) {
    int ih2 = h_start + kk_h * p.DH;
    if (ih2 < 0 || ih2 >= p.IH) {
      continue;
    }
    for (int kk_w = 0; kk_w < p.K; ++kk_w) {
      int iw2 = w_start + kk_w * p.DW;
      if (iw2 < 0 || iw2 >= p.IW) {
        continue;
      }
      int k_base = nb_base_2d(b, ih2, iw2, h, 0, p.IH, p.IW, p.H, p.D);
      float s = 0.0f;
      for (int d = 0; d < p.D; ++d) {
        s += query[q_base + d] * key[k_base + d];
      }
      s *= p.SCALE;
      max_score = max(max_score, s);
    }
  }

  float sum_exp = 0.0f;
  for (int kk_h = 0; kk_h < p.K; ++kk_h) {
    int ih2 = h_start + kk_h * p.DH;
    if (ih2 < 0 || ih2 >= p.IH) {
      continue;
    }
    for (int kk_w = 0; kk_w < p.K; ++kk_w) {
      int iw2 = w_start + kk_w * p.DW;
      if (iw2 < 0 || iw2 >= p.IW) {
        continue;
      }
      int k_base = nb_base_2d(b, ih2, iw2, h, 0, p.IH, p.IW, p.H, p.D);
      float s = 0.0f;
      for (int d = 0; d < p.D; ++d) {
        s += query[q_base + d] * key[k_base + d];
      }
      s *= p.SCALE;
      sum_exp += exp(s - max_score);
    }
  }

  if (sum_exp <= 0.0f) {
    for (int kpos = 0; kpos < k2; ++kpos) {
      attn[out_base + kpos] = 0.0f;
      grad_attn[out_base + kpos] = 0.0f;
    }
    return;
  }

  float inv_sum = 1.0f / sum_exp;
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = h_start + kh * p.DH;
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = w_start + kw * p.DW;
      int kpos = kh * p.K + kw;
      int out_idx = out_base + kpos;
      if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
        attn[out_idx] = 0.0f;
        grad_attn[out_idx] = 0.0f;
        continue;
      }
      int k_base = nb_base_2d(b, ih, iw, h, 0, p.IH, p.IW, p.H, p.D);
      float s_this = 0.0f;
      for (int d = 0; d < p.D; ++d) {
        s_this += query[q_base + d] * key[k_base + d];
      }
      s_this *= p.SCALE;
      float w = exp(s_this - max_score) * inv_sum;
      attn[out_idx] = w;

      float ga = 0.0f;
      int v_base = nb_base_2d(b, ih, iw, h, 0, p.IH, p.IW, p.H, p.D);
      for (int d = 0; d < p.D; ++d) {
        ga += grad_out[go_base + d] * value[v_base + d];
      }
      grad_attn[out_idx] = ga;
    }
  }
}

[[kernel]] void na3d_fused_bwd_attn_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* grad_out [[buffer(3)]],
    device float* attn [[buffer(4)]],
    device float* grad_attn [[buffer(5)]],
    constant NA3DParams& p [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = nb_ceil_div(p.ID, p.SD);
  int out_h = nb_ceil_div(p.IH, p.SH);
  int out_w = nb_ceil_div(p.IW, p.SW);
  int k3 = p.K * p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H;
  if (idx >= total) {
    return;
  }

  int h = idx % p.H;
  int t = idx / p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  t /= out_h;
  int od = t % out_d;
  int b = t / out_d;

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  int out_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3);
  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) {
    for (int kpos = 0; kpos < k3; ++kpos) {
      attn[out_base + kpos] = 0.0f;
      grad_attn[out_base + kpos] = 0.0f;
    }
    return;
  }

  int nh = p.K / 2;
  int d_start = p.CD ? (qd - (p.K - 1) * p.DD) : natten_get_window_start(qd, p.ID, p.K, nh, p.DD);
  int h_start = p.CH ? (qh - (p.K - 1) * p.DH) : natten_get_window_start(qh, p.IH, p.K, nh, p.DH);
  int w_start = p.CW ? (qw - (p.K - 1) * p.DW) : natten_get_window_start(qw, p.IW, p.K, nh, p.DW);
  int q_base = nb_base_3d(b, qd, qh, qw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
  int go_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D);

  float max_score = -INFINITY;
  for (int kk_d = 0; kk_d < p.K; ++kk_d) {
    int id2 = d_start + kk_d * p.DD;
    if (id2 < 0 || id2 >= p.ID) {
      continue;
    }
    for (int kk_h = 0; kk_h < p.K; ++kk_h) {
      int ih2 = h_start + kk_h * p.DH;
      if (ih2 < 0 || ih2 >= p.IH) {
        continue;
      }
      for (int kk_w = 0; kk_w < p.K; ++kk_w) {
        int iw2 = w_start + kk_w * p.DW;
        if (iw2 < 0 || iw2 >= p.IW) {
          continue;
        }
        int k_base = nb_base_3d(b, id2, ih2, iw2, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
        float s = 0.0f;
        for (int d = 0; d < p.D; ++d) {
          s += query[q_base + d] * key[k_base + d];
        }
        s *= p.SCALE;
        max_score = max(max_score, s);
      }
    }
  }

  float sum_exp = 0.0f;
  for (int kk_d = 0; kk_d < p.K; ++kk_d) {
    int id2 = d_start + kk_d * p.DD;
    if (id2 < 0 || id2 >= p.ID) {
      continue;
    }
    for (int kk_h = 0; kk_h < p.K; ++kk_h) {
      int ih2 = h_start + kk_h * p.DH;
      if (ih2 < 0 || ih2 >= p.IH) {
        continue;
      }
      for (int kk_w = 0; kk_w < p.K; ++kk_w) {
        int iw2 = w_start + kk_w * p.DW;
        if (iw2 < 0 || iw2 >= p.IW) {
          continue;
        }
        int k_base = nb_base_3d(b, id2, ih2, iw2, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
        float s = 0.0f;
        for (int d = 0; d < p.D; ++d) {
          s += query[q_base + d] * key[k_base + d];
        }
        s *= p.SCALE;
        sum_exp += exp(s - max_score);
      }
    }
  }

  if (sum_exp <= 0.0f) {
    for (int kpos = 0; kpos < k3; ++kpos) {
      attn[out_base + kpos] = 0.0f;
      grad_attn[out_base + kpos] = 0.0f;
    }
    return;
  }

  float inv_sum = 1.0f / sum_exp;
  for (int kd = 0; kd < p.K; ++kd) {
    int id = d_start + kd * p.DD;
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = h_start + kh * p.DH;
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = w_start + kw * p.DW;
        int kpos = (kd * p.K + kh) * p.K + kw;
        int out_idx = out_base + kpos;
        if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
          attn[out_idx] = 0.0f;
          grad_attn[out_idx] = 0.0f;
          continue;
        }
        int k_base = nb_base_3d(b, id, ih, iw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
        float s_this = 0.0f;
        for (int d = 0; d < p.D; ++d) {
          s_this += query[q_base + d] * key[k_base + d];
        }
        s_this *= p.SCALE;
        float w = exp(s_this - max_score) * inv_sum;
        attn[out_idx] = w;

        float ga = 0.0f;
        int v_base = nb_base_3d(b, id, ih, iw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
        for (int d = 0; d < p.D; ++d) {
          ga += grad_out[go_base + d] * value[v_base + d];
        }
        grad_attn[out_idx] = ga;
      }
    }
  }
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
  int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL)
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
  int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
  if (kidx < 0 || kidx >= p.L) {
    return;
  }

  int aidx = (((b * out_l + oq) * p.H + h) * p.K + kpos);
  int goidx = (((b * out_l + oq) * p.H + h) * p.D + d);
  float g = attn[aidx] * grad_out[goidx];
  atomic_fetch_add_explicit(&grad_v[nb_base_1d(b, kidx, h, d, p.L, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na1d_fused_bwd_inner_s1_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device float* inner [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H;
  if (idx >= total || p.S != 1) {
    return;
  }

  int h = idx % p.H;
  int t = idx / p.H;
  int qidx = t % p.L;
  int b = t / p.L;
  int token_idx = ((b * p.L + qidx) * p.H + h);
  int a_base = token_idx * p.K;

  float acc = 0.0f;
  int nh = p.K / 2;
  int start = natten_get_window_start(qidx, p.L, p.K, nh, p.DIL);
  for (int kpos = 0; kpos < p.K; ++kpos) {
    int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL) : (start + kpos * p.DIL);
    if (kidx < 0 || kidx >= p.L) {
      continue;
    }
    int a_idx = a_base + kpos;
    acc += attn[a_idx] * grad_attn[a_idx];
  }
  inner[idx] = acc;
}

[[kernel]] void na1d_fused_bwd_inner_s1_causal_k9_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device float* inner [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H;
  if (idx >= total || p.S != 1 || p.CAUSAL == 0 || p.K != 9) {
    return;
  }

  int h = idx % p.H;
  int t = idx / p.H;
  int qidx = t % p.L;
  int b = t / p.L;
  int token_idx = ((b * p.L + qidx) * p.H + h);
  int a_base = token_idx * 9;

  float acc = 0.0f;
  int k_begin = max(0, 8 - (qidx / p.DIL));
  for (int kpos = k_begin; kpos < 9; ++kpos) {
    int a_idx = a_base + kpos;
    acc += attn[a_idx] * grad_attn[a_idx];
  }
  inner[idx] = acc;
}

[[kernel]] void na1d_fused_bwd_q_softmax_s1_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA1DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * p.D;
  if (idx >= total || p.S != 1) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int qidx = t % p.L;
  int b = t / p.L;

  int nh = p.K / 2;
  int token_idx = ((b * p.L + qidx) * p.H + h);
  int a_base = token_idx * p.K;
  float inner_val = inner[token_idx];
  float acc = 0.0f;
  int start = natten_get_window_start(qidx, p.L, p.K, nh, p.DIL);
  for (int kpos = 0; kpos < p.K; ++kpos) {
    int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL) : (start + kpos * p.DIL);
    if (kidx < 0 || kidx >= p.L) {
      continue;
    }
    int a_idx = a_base + kpos;
    float g = attn[a_idx] * (grad_attn[a_idx] - inner_val) * p.SCALE;
    acc += g * key[nb_base_1d(b, kidx, h, d, p.L, p.H, p.D)];
  }
  grad_q[idx] = acc;
}

[[kernel]] void na1d_fused_bwd_q_softmax_s1_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA1DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * dim4;
  if (idx >= total || dim4 <= 0 || p.S != 1) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int qidx = t % p.L;
  int b = t / p.L;

  int nh = p.K / 2;
  int token_idx = ((b * p.L + qidx) * p.H + h);
  int a_base = token_idx * p.K;
  float inner_val = inner[token_idx];
  float4 acc = float4(0.0f);
  int start = natten_get_window_start(qidx, p.L, p.K, nh, p.DIL);
  for (int kpos = 0; kpos < p.K; ++kpos) {
    int kidx = p.CAUSAL ? (qidx - (p.K - 1 - kpos) * p.DIL) : (start + kpos * p.DIL);
    if (kidx < 0 || kidx >= p.L) {
      continue;
    }
    int a_idx = a_base + kpos;
    float g = attn[a_idx] * (grad_attn[a_idx] - inner_val) * p.SCALE;
    int k_base = nb_base_1d(b, kidx, h, d0, p.L, p.H, p.D);
    const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
    acc += g * k4[0];
  }
  int out_base = nb_base_1d(b, qidx, h, d0, p.L, p.H, p.D);
  grad_q[out_base] = acc.x;
  grad_q[out_base + 1] = acc.y;
  grad_q[out_base + 2] = acc.z;
  grad_q[out_base + 3] = acc.w;
}

[[kernel]] void na1d_fused_bwd_q_softmax_direct_s1_causal_k9_token_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA1DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H;
  if (idx >= total || dim4 <= 0 || p.S != 1 || p.CAUSAL == 0 || p.K != 9) {
    return;
  }

  int h = idx % p.H;
  int t = idx / p.H;
  int qidx = t % p.L;
  int b = t / p.L;

  int token_idx = ((b * p.L + qidx) * p.H + h);
  int a_base = token_idx * 9;
  float inner_val = inner[token_idx];
  int k_begin = max(0, 8 - (qidx / p.DIL));
  for (int d4 = 0; d4 < dim4; ++d4) {
    int d0 = d4 * 4;
    float4 acc = float4(0.0f);
    for (int kpos = k_begin; kpos < 9; ++kpos) {
      int kidx = qidx - (8 - kpos) * p.DIL;
      int a_idx = a_base + kpos;
      float g = attn[a_idx] * (grad_attn[a_idx] - inner_val) * p.SCALE;
      int k_base = nb_base_1d(b, kidx, h, d0, p.L, p.H, p.D);
      const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
      acc += g * k4[0];
    }
    int out_base = nb_base_1d(b, qidx, h, d0, p.L, p.H, p.D);
    grad_q[out_base] = acc.x;
    grad_q[out_base + 1] = acc.y;
    grad_q[out_base + 2] = acc.z;
    grad_q[out_base + 3] = acc.w;
  }
}

[[kernel]] void na1d_fused_bwd_q_softmax_direct_s1_causal_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA1DParams& p [[buffer(5)]],
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
  int qidx = t % p.L;
  int b = t / p.L;

  int token_idx = ((b * p.L + qidx) * p.H + h);
  int a_base = token_idx * p.K;
  float inner_val = inner[token_idx];
  float acc = 0.0f;
  int min_k = p.K - 1 - (qidx / p.DIL);
  int k_begin = max(0, min_k);
  for (int kpos = k_begin; kpos < p.K; ++kpos) {
    int kidx = qidx - (p.K - 1 - kpos) * p.DIL;
    int a_idx = a_base + kpos;
    float g = attn[a_idx] * (grad_attn[a_idx] - inner_val) * p.SCALE;
    acc += g * key[nb_base_1d(b, kidx, h, d, p.L, p.H, p.D)];
  }
  grad_q[idx] = acc;
}

[[kernel]] void na1d_fused_bwd_q_softmax_direct_s1_causal_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA1DParams& p [[buffer(5)]],
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
  int qidx = t % p.L;
  int b = t / p.L;

  int token_idx = ((b * p.L + qidx) * p.H + h);
  int a_base = token_idx * p.K;
  float inner_val = inner[token_idx];
  float4 acc = float4(0.0f);
  int min_k = p.K - 1 - (qidx / p.DIL);
  int k_begin = max(0, min_k);
  for (int kpos = k_begin; kpos < p.K; ++kpos) {
    int kidx = qidx - (p.K - 1 - kpos) * p.DIL;
    int a_idx = a_base + kpos;
    float g = attn[a_idx] * (grad_attn[a_idx] - inner_val) * p.SCALE;
    int k_base = nb_base_1d(b, kidx, h, d0, p.L, p.H, p.D);
    const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
    acc += g * k4[0];
  }
  int out_base = nb_base_1d(b, qidx, h, d0, p.L, p.H, p.D);
  grad_q[out_base] = acc.x;
  grad_q[out_base + 1] = acc.y;
  grad_q[out_base + 2] = acc.z;
  grad_q[out_base + 3] = acc.w;
}

[[kernel]] void na1d_fused_bwd_q_softmax_direct_s1_causal_k9_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA1DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * p.D;
  if (idx >= total || p.S != 1 || p.CAUSAL == 0 || p.K != 9) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int qidx = t % p.L;
  int b = t / p.L;

  int token_idx = ((b * p.L + qidx) * p.H + h);
  int a_base = token_idx * 9;
  float inner_val = inner[token_idx];
  float acc = 0.0f;
  int k_begin = max(0, 8 - (qidx / p.DIL));
  for (int kpos = k_begin; kpos < 9; ++kpos) {
    int kidx = qidx - (8 - kpos) * p.DIL;
    int a_idx = a_base + kpos;
    float g = attn[a_idx] * (grad_attn[a_idx] - inner_val) * p.SCALE;
    acc += g * key[nb_base_1d(b, kidx, h, d, p.L, p.H, p.D)];
  }
  grad_q[idx] = acc;
}

[[kernel]] void na1d_fused_bwd_q_softmax_direct_s1_causal_k9_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA1DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * dim4;
  if (idx >= total || dim4 <= 0 || p.S != 1 || p.CAUSAL == 0 || p.K != 9) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int qidx = t % p.L;
  int b = t / p.L;

  int token_idx = ((b * p.L + qidx) * p.H + h);
  int a_base = token_idx * 9;
  float inner_val = inner[token_idx];
  float4 acc = float4(0.0f);
  int k_begin = max(0, 8 - (qidx / p.DIL));
  for (int kpos = k_begin; kpos < 9; ++kpos) {
    int kidx = qidx - (8 - kpos) * p.DIL;
    int a_idx = a_base + kpos;
    float g = attn[a_idx] * (grad_attn[a_idx] - inner_val) * p.SCALE;
    int k_base = nb_base_1d(b, kidx, h, d0, p.L, p.H, p.D);
    const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
    acc += g * k4[0];
  }
  int out_base = nb_base_1d(b, qidx, h, d0, p.L, p.H, p.D);
  grad_q[out_base] = acc.x;
  grad_q[out_base + 1] = acc.y;
  grad_q[out_base + 2] = acc.z;
  grad_q[out_base + 3] = acc.w;
}

[[kernel]] void na1d_fused_bwd_kv_softmax_direct_s1_causal_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA1DParams& p [[buffer(7)]],
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

  float acc_k = 0.0f;
  float acc_v = 0.0f;
  for (int kpos = 0; kpos < p.K; ++kpos) {
    int qidx = kidx + (p.K - 1 - kpos) * p.DIL;
    if (qidx < 0 || qidx >= p.L) {
      continue;
    }
    int token_idx = ((b * p.L + qidx) * p.H + h);
    int a_idx = token_idx * p.K + kpos;
    float a = attn[a_idx];
    float g = a * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;
    acc_k += g * query[nb_base_1d(b, qidx, h, d, p.L, p.H, p.D)];
    acc_v += a * grad_out[token_idx * p.D + d];
  }
  int out_base = nb_base_1d(b, kidx, h, d, p.L, p.H, p.D);
  grad_k[out_base] = acc_k;
  grad_v[out_base] = acc_v;
}

[[kernel]] void na1d_fused_bwd_kv_softmax_direct_s1_causal_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA1DParams& p [[buffer(7)]],
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

  float4 acc_k = float4(0.0f);
  float4 acc_v = float4(0.0f);
  for (int kpos = 0; kpos < p.K; ++kpos) {
    int qidx = kidx + (p.K - 1 - kpos) * p.DIL;
    if (qidx < 0 || qidx >= p.L) {
      continue;
    }
    int token_idx = ((b * p.L + qidx) * p.H + h);
    int a_idx = token_idx * p.K + kpos;
    float a = attn[a_idx];
    float g = a * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;

    int q_base = nb_base_1d(b, qidx, h, d0, p.L, p.H, p.D);
    const device float4* q4 = reinterpret_cast<const device float4*>(query + q_base);
    const device float4* go4 = reinterpret_cast<const device float4*>(grad_out + token_idx * p.D + d0);
    acc_k += g * q4[0];
    acc_v += a * go4[0];
  }
  int out_base = nb_base_1d(b, kidx, h, d0, p.L, p.H, p.D);
  grad_k[out_base] = acc_k.x;
  grad_k[out_base + 1] = acc_k.y;
  grad_k[out_base + 2] = acc_k.z;
  grad_k[out_base + 3] = acc_k.w;
  grad_v[out_base] = acc_v.x;
  grad_v[out_base + 1] = acc_v.y;
  grad_v[out_base + 2] = acc_v.z;
  grad_v[out_base + 3] = acc_v.w;
}

[[kernel]] void na1d_fused_bwd_kv_softmax_direct_s1_causal_k9_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA1DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * p.D;
  if (idx >= total || p.S != 1 || p.CAUSAL == 0 || p.K != 9) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int kidx = t % p.L;
  int b = t / p.L;

  int k_begin = max(0, 8 - ((p.L - 1 - kidx) / p.DIL));
  int qidx = kidx + (8 - k_begin) * p.DIL;
  float acc_k = 0.0f;
  float acc_v = 0.0f;
  for (int kpos = k_begin; kpos < 9; ++kpos, qidx -= p.DIL) {
    int token_idx = ((b * p.L + qidx) * p.H + h);
    int a_idx = token_idx * 9 + kpos;
    float a = attn[a_idx];
    float g = a * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;
    acc_k += g * query[nb_base_1d(b, qidx, h, d, p.L, p.H, p.D)];
    acc_v += a * grad_out[token_idx * p.D + d];
  }
  int out_base = nb_base_1d(b, kidx, h, d, p.L, p.H, p.D);
  grad_k[out_base] = acc_k;
  grad_v[out_base] = acc_v;
}

[[kernel]] void na1d_fused_bwd_kv_softmax_direct_s1_causal_k9_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA1DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * dim4;
  if (idx >= total || dim4 <= 0 || p.S != 1 || p.CAUSAL == 0 || p.K != 9) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int kidx = t % p.L;
  int b = t / p.L;

  int k_begin = max(0, 8 - ((p.L - 1 - kidx) / p.DIL));
  int qidx = kidx + (8 - k_begin) * p.DIL;
  float4 acc_k = float4(0.0f);
  float4 acc_v = float4(0.0f);
  for (int kpos = k_begin; kpos < 9; ++kpos, qidx -= p.DIL) {
    int token_idx = ((b * p.L + qidx) * p.H + h);
    int a_idx = token_idx * 9 + kpos;
    float a = attn[a_idx];
    float g = a * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;

    int q_base = nb_base_1d(b, qidx, h, d0, p.L, p.H, p.D);
    const device float4* q4 = reinterpret_cast<const device float4*>(query + q_base);
    const device float4* go4 = reinterpret_cast<const device float4*>(grad_out + token_idx * p.D + d0);
    acc_k += g * q4[0];
    acc_v += a * go4[0];
  }
  int out_base = nb_base_1d(b, kidx, h, d0, p.L, p.H, p.D);
  grad_k[out_base] = acc_k.x;
  grad_k[out_base + 1] = acc_k.y;
  grad_k[out_base + 2] = acc_k.z;
  grad_k[out_base + 3] = acc_k.w;
  grad_v[out_base] = acc_v.x;
  grad_v[out_base + 1] = acc_v.y;
  grad_v[out_base + 2] = acc_v.z;
  grad_v[out_base + 3] = acc_v.w;
}

[[kernel]] void na1d_fused_bwd_kv_softmax_direct_u1d1_nc_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA1DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * p.D;
  if (idx >= total || p.S != 1 || p.CAUSAL != 0 || p.DIL != 1) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int kidx = t % p.L;
  int b = t / p.L;

  int radius = p.K - 1;
  int q_begin = max(0, kidx - radius);
  int q_end = min(p.L - 1, kidx + radius);
  float acc_k = 0.0f;
  float acc_v = 0.0f;
  for (int qidx = q_begin; qidx <= q_end; ++qidx) {
    int start = natten_get_window_start(qidx, p.L, p.K, p.K / 2, 1);
    int kpos = kidx - start;
    if (kpos < 0 || kpos >= p.K) {
      continue;
    }
    int token_idx = ((b * p.L + qidx) * p.H + h);
    int a_idx = token_idx * p.K + kpos;
    float a = attn[a_idx];
    float g = a * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;
    acc_k += g * query[nb_base_1d(b, qidx, h, d, p.L, p.H, p.D)];
    acc_v += a * grad_out[token_idx * p.D + d];
  }
  int out_base = nb_base_1d(b, kidx, h, d, p.L, p.H, p.D);
  grad_k[out_base] = acc_k;
  grad_v[out_base] = acc_v;
}

[[kernel]] void na1d_fused_bwd_kv_softmax_direct_u1d1_nc_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA1DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * dim4;
  if (idx >= total || dim4 <= 0 || p.S != 1 || p.CAUSAL != 0 || p.DIL != 1) {
    return;
  }

  int d4 = idx % dim4;
  int d0 = d4 * 4;
  int t = idx / dim4;
  int h = t % p.H;
  t /= p.H;
  int kidx = t % p.L;
  int b = t / p.L;

  int radius = p.K - 1;
  int q_begin = max(0, kidx - radius);
  int q_end = min(p.L - 1, kidx + radius);
  float4 acc_k = float4(0.0f);
  float4 acc_v = float4(0.0f);
  for (int qidx = q_begin; qidx <= q_end; ++qidx) {
    int start = natten_get_window_start(qidx, p.L, p.K, p.K / 2, 1);
    int kpos = kidx - start;
    if (kpos < 0 || kpos >= p.K) {
      continue;
    }
    int token_idx = ((b * p.L + qidx) * p.H + h);
    int a_idx = token_idx * p.K + kpos;
    float a = attn[a_idx];
    float g = a * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;

    int q_base = nb_base_1d(b, qidx, h, d0, p.L, p.H, p.D);
    const device float4* q4 = reinterpret_cast<const device float4*>(query + q_base);
    const device float4* go4 = reinterpret_cast<const device float4*>(grad_out + token_idx * p.D + d0);
    acc_k += g * q4[0];
    acc_v += a * go4[0];
  }
  int out_base = nb_base_1d(b, kidx, h, d0, p.L, p.H, p.D);
  grad_k[out_base] = acc_k.x;
  grad_k[out_base + 1] = acc_k.y;
  grad_k[out_base + 2] = acc_k.z;
  grad_k[out_base + 3] = acc_k.w;
  grad_v[out_base] = acc_v.x;
  grad_v[out_base + 1] = acc_v.y;
  grad_v[out_base + 2] = acc_v.z;
  grad_v[out_base + 3] = acc_v.w;
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
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int gl_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * k2 + kpos);
  float g = grad_logits[gl_idx] * p.SCALE;
  atomic_fetch_add_explicit(&grad_q[nb_base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D)], g * key[nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)], memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_k[nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)], g * query[nb_base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D)], memory_order_relaxed);
}

[[kernel]] void na2d_fused_bwd_qk_softmax_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* key [[buffer(4)]],
    device atomic<float>* grad_q [[buffer(5)]],
    device atomic<float>* grad_k [[buffer(6)]],
    constant NA2DParams& p [[buffer(7)]],
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
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int a_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * k2 + kpos);
  int inner_idx = (((b * out_h + oh) * out_w + ow) * p.H + h);
  float g = attn[a_idx] * (grad_attn[a_idx] - inner[inner_idx]) * p.SCALE;
  atomic_fetch_add_explicit(
      &grad_q[nb_base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D)],
      g * key[nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)],
      memory_order_relaxed);
  atomic_fetch_add_explicit(
      &grad_k[nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)],
      g * query[nb_base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D)],
      memory_order_relaxed);
}

[[kernel]] void na2d_fused_bwd_qkv_softmax_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* key [[buffer(4)]],
    device const float* grad_out [[buffer(5)]],
    device atomic<float>* grad_q [[buffer(6)]],
    device atomic<float>* grad_k [[buffer(7)]],
    device atomic<float>* grad_v [[buffer(8)]],
    constant NA2DParams& p [[buffer(9)]],
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
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int a_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * k2 + kpos);
  int inner_idx = (((b * out_h + oh) * out_w + ow) * p.H + h);
  float g = attn[a_idx] * (grad_attn[a_idx] - inner[inner_idx]) * p.SCALE;
  atomic_fetch_add_explicit(
      &grad_q[nb_base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D)],
      g * key[nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)],
      memory_order_relaxed);
  atomic_fetch_add_explicit(
      &grad_k[nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)],
      g * query[nb_base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D)],
      memory_order_relaxed);

  int go_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D + d);
  float gv = attn[a_idx] * grad_out[go_idx];
  atomic_fetch_add_explicit(
      &grad_v[nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)], gv, memory_order_relaxed);
}

[[kernel]] void na2d_fused_bwd_qkv_softmax_tiled_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* key [[buffer(4)]],
    device const float* grad_out [[buffer(5)]],
    device atomic<float>* grad_q [[buffer(6)]],
    device float* grad_k [[buffer(7)]],
    device float* grad_v [[buffer(8)]],
    constant NA2DParams& p [[buffer(9)]],
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
  int out_h = p.IH;
  int out_w = p.IW;
  int base_h = p.IH - p.K;
  int base_w = p.IW - p.K;
  int top_h_end = min(nh, p.IH);
  int top_w_end = min(nh, p.IW);
  int bot_h_begin = max(0, p.IH - nh);
  int bot_w_begin = max(0, p.IW - nh);
  int k2 = p.K * p.K;

  int out_base = nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D);
  float key_val = key[out_base];
  float acc_k = 0.0f;
  float acc_v = 0.0f;
  for (int kh = 0; kh < p.K; ++kh) {
    int oh_list[32];
    int n_oh = 0;
    if (ih == kh) {
      for (int oh = 0; oh < top_h_end && n_oh < 32; ++oh) {
        oh_list[n_oh++] = oh;
      }
    }
    int oh_mid = ih - kh + nh;
    if (oh_mid >= nh && oh_mid < p.IH - nh && n_oh < 32) {
      oh_list[n_oh++] = oh_mid;
    }
    if (ih == base_h + kh) {
      for (int oh = bot_h_begin; oh < p.IH && n_oh < 32; ++oh) {
        oh_list[n_oh++] = oh;
      }
    }
    if (n_oh == 0) {
      continue;
    }

    for (int kw = 0; kw < p.K; ++kw) {
      int ow_list[32];
      int n_ow = 0;
      if (iw == kw) {
        for (int ow = 0; ow < top_w_end && n_ow < 32; ++ow) {
          ow_list[n_ow++] = ow;
        }
      }
      int ow_mid = iw - kw + nh;
      if (ow_mid >= nh && ow_mid < p.IW - nh && n_ow < 32) {
        ow_list[n_ow++] = ow_mid;
      }
      if (iw == base_w + kw) {
        for (int ow = bot_w_begin; ow < p.IW && n_ow < 32; ++ow) {
          ow_list[n_ow++] = ow;
        }
      }
      if (n_ow == 0) {
        continue;
      }

      int kpos = kh * p.K + kw;
      for (int i_oh = 0; i_oh < n_oh; ++i_oh) {
        int oh = oh_list[i_oh];
        int qh = oh;
        if (qh < 0 || qh >= p.IH) {
          continue;
        }
        for (int i_ow = 0; i_ow < n_ow; ++i_ow) {
          int ow = ow_list[i_ow];
          int qw = ow;
          if (qw < 0 || qw >= p.IW) {
            continue;
          }
          int token_idx = (((b * out_h + oh) * out_w + ow) * p.H + h);
          int a_idx = token_idx * k2 + kpos;
          float a = attn[a_idx];
          float g = a * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;
          int q_base = nb_base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D);
          acc_k += g * query[q_base];
          acc_v += a * grad_out[token_idx * p.D + d];
          atomic_fetch_add_explicit(&grad_q[q_base], g * key_val, memory_order_relaxed);
        }
      }
    }
  }
  grad_k[out_base] = acc_k;
  grad_v[out_base] = acc_v;
}

template <int LIST_CAP>
inline int nb2d_collect_out_candidates(
    int coord,
    int k_idx,
    int nh,
    int limit,
    int base,
    int top_end,
    int bot_begin,
    thread int* out_list) {
  int n = 0;
  if (coord == k_idx) {
    for (int o = 0; o < top_end && n < LIST_CAP; ++o) {
      out_list[n++] = o;
    }
  }
  int mid = coord - k_idx + nh;
  if (mid >= nh && mid < limit - nh && n < LIST_CAP) {
    out_list[n++] = mid;
  }
  if (coord == base + k_idx) {
    for (int o = bot_begin; o < limit && n < LIST_CAP; ++o) {
      out_list[n++] = o;
    }
  }
  return n;
}

template <int LIST_CAP>
inline void na2d_fused_bwd_kv_softmax_tiled_scalar_impl(
    device const float* attn,
    device const float* grad_attn,
    device const float* inner,
    device const float* query,
    device const float* grad_out,
    device float* grad_k,
    device float* grad_v,
    constant NA2DParams& p,
    uint tid) {
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
  int out_h = p.IH;
  int out_w = p.IW;
  int base_h = p.IH - p.K;
  int base_w = p.IW - p.K;
  int top_h_end = min(nh, p.IH);
  int top_w_end = min(nh, p.IW);
  int bot_h_begin = max(0, p.IH - nh);
  int bot_w_begin = max(0, p.IW - nh);
  int k2 = p.K * p.K;

  int out_base = nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D);
  float acc_k = 0.0f;
  float acc_v = 0.0f;
  for (int kh = 0; kh < p.K; ++kh) {
    int oh_list[LIST_CAP];
    int n_oh = nb2d_collect_out_candidates<LIST_CAP>(
        ih, kh, nh, p.IH, base_h, top_h_end, bot_h_begin, oh_list);
    if (n_oh == 0) {
      continue;
    }

    for (int kw = 0; kw < p.K; ++kw) {
      int ow_list[LIST_CAP];
      int n_ow = nb2d_collect_out_candidates<LIST_CAP>(
          iw, kw, nh, p.IW, base_w, top_w_end, bot_w_begin, ow_list);
      if (n_ow == 0) {
        continue;
      }

      int kpos = kh * p.K + kw;
      for (int i_oh = 0; i_oh < n_oh; ++i_oh) {
        int qh = oh_list[i_oh];
        if (qh < 0 || qh >= p.IH) {
          continue;
        }
        for (int i_ow = 0; i_ow < n_ow; ++i_ow) {
          int qw = ow_list[i_ow];
          if (qw < 0 || qw >= p.IW) {
            continue;
          }
          int token_idx = (((b * out_h + qh) * out_w + qw) * p.H + h);
          int a_idx = token_idx * k2 + kpos;
          float a = attn[a_idx];
          float g = a * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;
          int q_base = nb_base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D);
          acc_k += g * query[q_base];
          acc_v += a * grad_out[token_idx * p.D + d];
        }
      }
    }
  }
  grad_k[out_base] = acc_k;
  grad_v[out_base] = acc_v;
}

template <int LIST_CAP>
inline void na2d_fused_bwd_kv_softmax_tiled_vec4_impl(
    device const float* attn,
    device const float* grad_attn,
    device const float* inner,
    device const float* query,
    device const float* grad_out,
    device float* grad_k,
    device float* grad_v,
    constant NA2DParams& p,
    uint tid) {
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
  int out_h = p.IH;
  int out_w = p.IW;
  int base_h = p.IH - p.K;
  int base_w = p.IW - p.K;
  int top_h_end = min(nh, p.IH);
  int top_w_end = min(nh, p.IW);
  int bot_h_begin = max(0, p.IH - nh);
  int bot_w_begin = max(0, p.IW - nh);
  int k2 = p.K * p.K;

  int out_base = nb_base_2d(b, ih, iw, h, d0, p.IH, p.IW, p.H, p.D);
  float4 acc_k = float4(0.0f);
  float4 acc_v = float4(0.0f);
  for (int kh = 0; kh < p.K; ++kh) {
    int oh_list[LIST_CAP];
    int n_oh = nb2d_collect_out_candidates<LIST_CAP>(
        ih, kh, nh, p.IH, base_h, top_h_end, bot_h_begin, oh_list);
    if (n_oh == 0) {
      continue;
    }

    for (int kw = 0; kw < p.K; ++kw) {
      int ow_list[LIST_CAP];
      int n_ow = nb2d_collect_out_candidates<LIST_CAP>(
          iw, kw, nh, p.IW, base_w, top_w_end, bot_w_begin, ow_list);
      if (n_ow == 0) {
        continue;
      }

      int kpos = kh * p.K + kw;
      for (int i_oh = 0; i_oh < n_oh; ++i_oh) {
        int qh = oh_list[i_oh];
        if (qh < 0 || qh >= p.IH) {
          continue;
        }
        for (int i_ow = 0; i_ow < n_ow; ++i_ow) {
          int qw = ow_list[i_ow];
          if (qw < 0 || qw >= p.IW) {
            continue;
          }
          int token_idx = (((b * out_h + qh) * out_w + qw) * p.H + h);
          int a_idx = token_idx * k2 + kpos;
          float a = attn[a_idx];
          float g = a * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;
          int q_base = nb_base_2d(b, qh, qw, h, d0, p.IH, p.IW, p.H, p.D);
          int go_base = token_idx * p.D + d0;
          const device float4* q4 = reinterpret_cast<const device float4*>(query + q_base);
          const device float4* go4 = reinterpret_cast<const device float4*>(grad_out + go_base);
          acc_k += g * q4[0];
          acc_v += a * go4[0];
        }
      }
    }
  }
  grad_k[out_base] = acc_k.x;
  grad_k[out_base + 1] = acc_k.y;
  grad_k[out_base + 2] = acc_k.z;
  grad_k[out_base + 3] = acc_k.w;
  grad_v[out_base] = acc_v.x;
  grad_v[out_base + 1] = acc_v.y;
  grad_v[out_base + 2] = acc_v.z;
  grad_v[out_base + 3] = acc_v.w;
}

[[kernel]] void na2d_fused_bwd_kv_softmax_tiled_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA2DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  na2d_fused_bwd_kv_softmax_tiled_scalar_impl<32>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na2d_fused_bwd_kv_softmax_tiled_k3_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA2DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 3) {
    return;
  }
  na2d_fused_bwd_kv_softmax_tiled_scalar_impl<4>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na2d_fused_bwd_kv_softmax_tiled_k5_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA2DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 5) {
    return;
  }
  na2d_fused_bwd_kv_softmax_tiled_scalar_impl<8>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na2d_fused_bwd_kv_softmax_tiled_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA2DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  na2d_fused_bwd_kv_softmax_tiled_vec4_impl<32>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na2d_fused_bwd_kv_softmax_tiled_k3_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA2DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 3) {
    return;
  }
  na2d_fused_bwd_kv_softmax_tiled_vec4_impl<4>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na2d_fused_bwd_kv_softmax_tiled_k5_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA2DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 5) {
    return;
  }
  na2d_fused_bwd_kv_softmax_tiled_vec4_impl<8>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na2d_fused_bwd_kv_softmax_tiled_k7_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA2DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 7) {
    return;
  }
  na2d_fused_bwd_kv_softmax_tiled_scalar_impl<8>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na2d_fused_bwd_kv_softmax_tiled_k7_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA2DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 7) {
    return;
  }
  na2d_fused_bwd_kv_softmax_tiled_vec4_impl<8>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na2d_fused_bwd_q_softmax_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA2DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = nb_ceil_div(p.IH, p.SH);
  int out_w = nb_ceil_div(p.IW, p.SW);
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
  int k2 = p.K * p.K;
  int token_idx = (((b * out_h + oh) * out_w + ow) * p.H + h);
  int attn_base = token_idx * k2;
  float inner_val = inner[token_idx];
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
      int kpos = kh * p.K + kw;
      int a_idx = attn_base + kpos;
      float g = attn[a_idx] * (grad_attn[a_idx] - inner_val);
      acc += g * key[nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)];
    }
  }
  grad_q[idx] = acc * p.SCALE;
}

[[kernel]] void na2d_fused_bwd_q_softmax_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA2DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int out_h = nb_ceil_div(p.IH, p.SH);
  int out_w = nb_ceil_div(p.IW, p.SW);
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
  int w = t % p.IW;
  t /= p.IW;
  int i = t % p.IH;
  int b = t / p.IH;

  int out_base = nb_base_2d(b, i, w, h, d0, p.IH, p.IW, p.H, p.D);
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
  int h_start_nc = natten_get_window_start(i, p.IH, p.K, nh, p.DH);
  int w_start_nc = natten_get_window_start(w, p.IW, p.K, nh, p.DW);
  int k2 = p.K * p.K;
  int token_idx = (((b * out_h + oh) * out_w + ow) * p.H + h);
  int attn_base = token_idx * k2;
  float inner_val = inner[token_idx];
  float4 acc = float4(0.0f);
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = p.CH ? (i - (p.K - 1 - kh) * p.DH)
                  : (h_start_nc + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW ? (w - (p.K - 1 - kw) * p.DW)
                    : (w_start_nc + kw * p.DW);
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      int kpos = kh * p.K + kw;
      int a_idx = attn_base + kpos;
      float g = attn[a_idx] * (grad_attn[a_idx] - inner_val);
      int k_base = nb_base_2d(b, ih, iw, h, d0, p.IH, p.IW, p.H, p.D);
      const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
      acc += g * k4[0];
    }
  }
  acc *= p.SCALE;
  grad_q[out_base] = acc.x;
  grad_q[out_base + 1] = acc.y;
  grad_q[out_base + 2] = acc.z;
  grad_q[out_base + 3] = acc.w;
}

[[kernel]] void na2d_fused_bwd_q_softmax_u1d1_nc_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA2DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int idx = static_cast<int>(tid);
  int total = p.B * p.IH * p.IW * p.H * p.D;
  if (idx >= total || p.SH != 1 || p.SW != 1 || p.DH != 1 || p.DW != 1 || p.CH != 0 || p.CW != 0) {
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

  int nh = p.K / 2;
  int start_h = natten_get_window_start(i, p.IH, p.K, nh, 1);
  int start_w = natten_get_window_start(w, p.IW, p.K, nh, 1);
  int k2 = p.K * p.K;
  int token_idx = (((b * p.IH + i) * p.IW + w) * p.H + h);
  int attn_base = token_idx * k2;
  float inner_val = inner[token_idx];
  float acc = 0.0f;
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = start_h + kh;
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = start_w + kw;
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      int kpos = kh * p.K + kw;
      int a_idx = attn_base + kpos;
      float g = attn[a_idx] * (grad_attn[a_idx] - inner_val);
      acc += g * key[nb_base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)];
    }
  }
  grad_q[idx] = acc * p.SCALE;
}

[[kernel]] void na2d_fused_bwd_q_softmax_u1d1_nc_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA2DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.IH * p.IW * p.H * dim4;
  if (idx >= total || dim4 <= 0 || p.SH != 1 || p.SW != 1 || p.DH != 1 || p.DW != 1 || p.CH != 0 ||
      p.CW != 0) {
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

  int out_base = nb_base_2d(b, i, w, h, d0, p.IH, p.IW, p.H, p.D);
  int nh = p.K / 2;
  int start_h = natten_get_window_start(i, p.IH, p.K, nh, 1);
  int start_w = natten_get_window_start(w, p.IW, p.K, nh, 1);
  int k2 = p.K * p.K;
  int token_idx = (((b * p.IH + i) * p.IW + w) * p.H + h);
  int attn_base = token_idx * k2;
  float inner_val = inner[token_idx];
  float4 acc = float4(0.0f);
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = start_h + kh;
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = start_w + kw;
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      int kpos = kh * p.K + kw;
      int a_idx = attn_base + kpos;
      float g = attn[a_idx] * (grad_attn[a_idx] - inner_val);
      int k_base = nb_base_2d(b, ih, iw, h, d0, p.IH, p.IW, p.H, p.D);
      const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
      acc += g * k4[0];
    }
  }
  acc *= p.SCALE;
  grad_q[out_base] = acc.x;
  grad_q[out_base + 1] = acc.y;
  grad_q[out_base + 2] = acc.z;
  grad_q[out_base + 3] = acc.w;
}

[[kernel]] void na2d_fused_bwd_k_direct_softmax_u1d1_nc_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device float* grad_k [[buffer(4)]],
    constant NA2DParams& p [[buffer(5)]],
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
  int out_h = p.IH;
  int out_w = p.IW;
  int base_h = p.IH - p.K;
  int base_w = p.IW - p.K;
  int top_h_end = min(nh, p.IH);
  int top_w_end = min(nh, p.IW);
  int bot_h_begin = max(0, p.IH - nh);
  int bot_w_begin = max(0, p.IW - nh);
  int k2 = p.K * p.K;

  float acc = 0.0f;
  for (int kh = 0; kh < p.K; ++kh) {
    int oh_list[32];
    int n_oh = 0;
    if (ih == kh) {
      for (int oh = 0; oh < top_h_end && n_oh < 32; ++oh) {
        oh_list[n_oh++] = oh;
      }
    }
    int oh_mid = ih - kh + nh;
    if (oh_mid >= nh && oh_mid < p.IH - nh && n_oh < 32) {
      oh_list[n_oh++] = oh_mid;
    }
    if (ih == base_h + kh) {
      for (int oh = bot_h_begin; oh < p.IH && n_oh < 32; ++oh) {
        oh_list[n_oh++] = oh;
      }
    }
    if (n_oh == 0) {
      continue;
    }

    for (int kw = 0; kw < p.K; ++kw) {
      int ow_list[32];
      int n_ow = 0;
      if (iw == kw) {
        for (int ow = 0; ow < top_w_end && n_ow < 32; ++ow) {
          ow_list[n_ow++] = ow;
        }
      }
      int ow_mid = iw - kw + nh;
      if (ow_mid >= nh && ow_mid < p.IW - nh && n_ow < 32) {
        ow_list[n_ow++] = ow_mid;
      }
      if (iw == base_w + kw) {
        for (int ow = bot_w_begin; ow < p.IW && n_ow < 32; ++ow) {
          ow_list[n_ow++] = ow;
        }
      }
      if (n_ow == 0) {
        continue;
      }

      int kpos = kh * p.K + kw;
      for (int i_oh = 0; i_oh < n_oh; ++i_oh) {
        int oh = oh_list[i_oh];
        int qh = oh;
        if (qh < 0 || qh >= p.IH) {
          continue;
        }
        for (int i_ow = 0; i_ow < n_ow; ++i_ow) {
          int ow = ow_list[i_ow];
          int qw = ow;
          if (qw < 0 || qw >= p.IW) {
            continue;
          }
          int token_idx = (((b * out_h + oh) * out_w + ow) * p.H + h);
          int a_idx = token_idx * k2 + kpos;
          float g = attn[a_idx] * (grad_attn[a_idx] - inner[token_idx]);
          acc += g * query[nb_base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D)] * p.SCALE;
        }
      }
    }
  }
  grad_k[idx] = acc;
}

[[kernel]] void na2d_fused_bwd_k_direct_softmax_u1d1_nc_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device float* grad_k [[buffer(4)]],
    constant NA2DParams& p [[buffer(5)]],
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
  int out_h = p.IH;
  int out_w = p.IW;
  int base_h = p.IH - p.K;
  int base_w = p.IW - p.K;
  int top_h_end = min(nh, p.IH);
  int top_w_end = min(nh, p.IW);
  int bot_h_begin = max(0, p.IH - nh);
  int bot_w_begin = max(0, p.IW - nh);
  int k2 = p.K * p.K;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int kh = 0; kh < p.K; ++kh) {
    int oh_list[32];
    int n_oh = 0;
    if (ih == kh) {
      for (int oh = 0; oh < top_h_end && n_oh < 32; ++oh) {
        oh_list[n_oh++] = oh;
      }
    }
    int oh_mid = ih - kh + nh;
    if (oh_mid >= nh && oh_mid < p.IH - nh && n_oh < 32) {
      oh_list[n_oh++] = oh_mid;
    }
    if (ih == base_h + kh) {
      for (int oh = bot_h_begin; oh < p.IH && n_oh < 32; ++oh) {
        oh_list[n_oh++] = oh;
      }
    }
    if (n_oh == 0) {
      continue;
    }

    for (int kw = 0; kw < p.K; ++kw) {
      int ow_list[32];
      int n_ow = 0;
      if (iw == kw) {
        for (int ow = 0; ow < top_w_end && n_ow < 32; ++ow) {
          ow_list[n_ow++] = ow;
        }
      }
      int ow_mid = iw - kw + nh;
      if (ow_mid >= nh && ow_mid < p.IW - nh && n_ow < 32) {
        ow_list[n_ow++] = ow_mid;
      }
      if (iw == base_w + kw) {
        for (int ow = bot_w_begin; ow < p.IW && n_ow < 32; ++ow) {
          ow_list[n_ow++] = ow;
        }
      }
      if (n_ow == 0) {
        continue;
      }

      int kpos = kh * p.K + kw;
      for (int i_oh = 0; i_oh < n_oh; ++i_oh) {
        int oh = oh_list[i_oh];
        int qh = oh;
        if (qh < 0 || qh >= p.IH) {
          continue;
        }
        for (int i_ow = 0; i_ow < n_ow; ++i_ow) {
          int ow = ow_list[i_ow];
          int qw = ow;
          if (qw < 0 || qw >= p.IW) {
            continue;
          }
          int token_idx = (((b * out_h + oh) * out_w + ow) * p.H + h);
          int a_idx = token_idx * k2 + kpos;
          float g = attn[a_idx] * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;
          int q_base = nb_base_2d(b, qh, qw, h, d0, p.IH, p.IW, p.H, p.D);
          acc0 += g * query[q_base];
          acc1 += g * query[q_base + 1];
          acc2 += g * query[q_base + 2];
          acc3 += g * query[q_base + 3];
        }
      }
    }
  }
  int out_base = nb_base_2d(b, ih, iw, h, d0, p.IH, p.IW, p.H, p.D);
  grad_k[out_base] = acc0;
  grad_k[out_base + 1] = acc1;
  grad_k[out_base + 2] = acc2;
  grad_k[out_base + 3] = acc3;
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
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
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
  int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int gl_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3 + kpos);
  float g = grad_logits[gl_idx] * p.SCALE;
  atomic_fetch_add_explicit(&grad_q[nb_base_3d(b, qd, qh, qw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], g * key[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], memory_order_relaxed);
  atomic_fetch_add_explicit(&grad_k[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], g * query[nb_base_3d(b, qd, qh, qw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], memory_order_relaxed);
}

[[kernel]] void na3d_fused_bwd_qk_softmax_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* key [[buffer(4)]],
    device atomic<float>* grad_q [[buffer(5)]],
    device atomic<float>* grad_k [[buffer(6)]],
    constant NA3DParams& p [[buffer(7)]],
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
  int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int a_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3 + kpos);
  int inner_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
  float g = attn[a_idx] * (grad_attn[a_idx] - inner[inner_idx]) * p.SCALE;
  atomic_fetch_add_explicit(
      &grad_q[nb_base_3d(b, qd, qh, qw, h, d, p.ID, p.IH, p.IW, p.H, p.D)],
      g * key[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)],
      memory_order_relaxed);
  atomic_fetch_add_explicit(
      &grad_k[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)],
      g * query[nb_base_3d(b, qd, qh, qw, h, d, p.ID, p.IH, p.IW, p.H, p.D)],
      memory_order_relaxed);
}

[[kernel]] void na3d_fused_bwd_qkv_softmax_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* key [[buffer(4)]],
    device const float* grad_out [[buffer(5)]],
    device atomic<float>* grad_q [[buffer(6)]],
    device atomic<float>* grad_k [[buffer(7)]],
    device atomic<float>* grad_v [[buffer(8)]],
    constant NA3DParams& p [[buffer(9)]],
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
  int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int a_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3 + kpos);
  int inner_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
  float g = attn[a_idx] * (grad_attn[a_idx] - inner[inner_idx]) * p.SCALE;
  atomic_fetch_add_explicit(
      &grad_q[nb_base_3d(b, qd, qh, qw, h, d, p.ID, p.IH, p.IW, p.H, p.D)],
      g * key[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)],
      memory_order_relaxed);
  atomic_fetch_add_explicit(
      &grad_k[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)],
      g * query[nb_base_3d(b, qd, qh, qw, h, d, p.ID, p.IH, p.IW, p.H, p.D)],
      memory_order_relaxed);

  int go_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D + d);
  float gv = attn[a_idx] * grad_out[go_idx];
  atomic_fetch_add_explicit(
      &grad_v[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], gv, memory_order_relaxed);
}

template <int LIST_CAP>
inline int nb3d_collect_out_candidates(
    int coord,
    int k_idx,
    int nh,
    int limit,
    int base,
    int top_end,
    int bot_begin,
    thread int* out_list) {
  int n = 0;
  if (coord == k_idx) {
    for (int o = 0; o < top_end && n < LIST_CAP; ++o) {
      out_list[n++] = o;
    }
  }
  int mid = coord - k_idx + nh;
  if (mid >= nh && mid < limit - nh && n < LIST_CAP) {
    out_list[n++] = mid;
  }
  if (coord == base + k_idx) {
    for (int o = bot_begin; o < limit && n < LIST_CAP; ++o) {
      out_list[n++] = o;
    }
  }
  return n;
}

template <int LIST_CAP>
inline void na3d_fused_bwd_qkv_softmax_tiled_scalar_impl(
    device const float* attn,
    device const float* grad_attn,
    device const float* inner,
    device const float* query,
    device const float* key,
    device const float* grad_out,
    device atomic<float>* grad_q,
    device float* grad_k,
    device float* grad_v,
    constant NA3DParams& p,
    uint tid) {
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
  int out_d = p.ID;
  int out_h = p.IH;
  int out_w = p.IW;
  int base_d = p.ID - p.K;
  int base_h = p.IH - p.K;
  int base_w = p.IW - p.K;
  int top_d_end = min(nh, p.ID);
  int top_h_end = min(nh, p.IH);
  int top_w_end = min(nh, p.IW);
  int bot_d_begin = max(0, p.ID - nh);
  int bot_h_begin = max(0, p.IH - nh);
  int bot_w_begin = max(0, p.IW - nh);
  int k3 = p.K * p.K * p.K;

  int out_base = nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D);
  float key_val = key[out_base];
  float acc_k = 0.0f;
  float acc_v = 0.0f;
  for (int kd = 0; kd < p.K; ++kd) {
    int od_list[LIST_CAP];
    int n_od = nb3d_collect_out_candidates<LIST_CAP>(
        id, kd, nh, p.ID, base_d, top_d_end, bot_d_begin, od_list);
    if (n_od == 0) {
      continue;
    }

    for (int kh = 0; kh < p.K; ++kh) {
      int oh_list[LIST_CAP];
      int n_oh = nb3d_collect_out_candidates<LIST_CAP>(
          ih, kh, nh, p.IH, base_h, top_h_end, bot_h_begin, oh_list);
      if (n_oh == 0) {
        continue;
      }

      for (int kw = 0; kw < p.K; ++kw) {
        int ow_list[LIST_CAP];
        int n_ow = nb3d_collect_out_candidates<LIST_CAP>(
            iw, kw, nh, p.IW, base_w, top_w_end, bot_w_begin, ow_list);
        if (n_ow == 0) {
          continue;
        }

        int kpos = (kd * p.K + kh) * p.K + kw;
        for (int i_od = 0; i_od < n_od; ++i_od) {
          int od = od_list[i_od];
          for (int i_oh = 0; i_oh < n_oh; ++i_oh) {
            int oh = oh_list[i_oh];
            for (int i_ow = 0; i_ow < n_ow; ++i_ow) {
              int ow = ow_list[i_ow];
              int token_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
              int a_idx = token_idx * k3 + kpos;
              float a = attn[a_idx];
              float g = a * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;
              int q_base = nb_base_3d(b, od, oh, ow, h, d, p.ID, p.IH, p.IW, p.H, p.D);
              acc_k += g * query[q_base];
              acc_v += a * grad_out[token_idx * p.D + d];
              atomic_fetch_add_explicit(&grad_q[q_base], g * key_val, memory_order_relaxed);
            }
          }
        }
      }
    }
  }
  grad_k[out_base] = acc_k;
  grad_v[out_base] = acc_v;
}

template <int LIST_CAP>
inline void na3d_fused_bwd_qkv_softmax_tiled_vec4_impl(
    device const float* attn,
    device const float* grad_attn,
    device const float* inner,
    device const float* query,
    device const float* key,
    device const float* grad_out,
    device atomic<float>* grad_q,
    device float* grad_k,
    device float* grad_v,
    constant NA3DParams& p,
    uint tid) {
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
  int out_d = p.ID;
  int out_h = p.IH;
  int out_w = p.IW;
  int base_d = p.ID - p.K;
  int base_h = p.IH - p.K;
  int base_w = p.IW - p.K;
  int top_d_end = min(nh, p.ID);
  int top_h_end = min(nh, p.IH);
  int top_w_end = min(nh, p.IW);
  int bot_d_begin = max(0, p.ID - nh);
  int bot_h_begin = max(0, p.IH - nh);
  int bot_w_begin = max(0, p.IW - nh);
  int k3 = p.K * p.K * p.K;

  int out_base = nb_base_3d(b, id, ih, iw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
  float key0 = key[out_base];
  float key1 = key[out_base + 1];
  float key2 = key[out_base + 2];
  float key3 = key[out_base + 3];
  float acc_k0 = 0.0f;
  float acc_k1 = 0.0f;
  float acc_k2 = 0.0f;
  float acc_k3 = 0.0f;
  float acc_v0 = 0.0f;
  float acc_v1 = 0.0f;
  float acc_v2 = 0.0f;
  float acc_v3 = 0.0f;
  for (int kd = 0; kd < p.K; ++kd) {
    int od_list[LIST_CAP];
    int n_od = nb3d_collect_out_candidates<LIST_CAP>(
        id, kd, nh, p.ID, base_d, top_d_end, bot_d_begin, od_list);
    if (n_od == 0) {
      continue;
    }

    for (int kh = 0; kh < p.K; ++kh) {
      int oh_list[LIST_CAP];
      int n_oh = nb3d_collect_out_candidates<LIST_CAP>(
          ih, kh, nh, p.IH, base_h, top_h_end, bot_h_begin, oh_list);
      if (n_oh == 0) {
        continue;
      }

      for (int kw = 0; kw < p.K; ++kw) {
        int ow_list[LIST_CAP];
        int n_ow = nb3d_collect_out_candidates<LIST_CAP>(
            iw, kw, nh, p.IW, base_w, top_w_end, bot_w_begin, ow_list);
        if (n_ow == 0) {
          continue;
        }

        int kpos = (kd * p.K + kh) * p.K + kw;
        for (int i_od = 0; i_od < n_od; ++i_od) {
          int od = od_list[i_od];
          for (int i_oh = 0; i_oh < n_oh; ++i_oh) {
            int oh = oh_list[i_oh];
            for (int i_ow = 0; i_ow < n_ow; ++i_ow) {
              int ow = ow_list[i_ow];
              int token_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
              int a_idx = token_idx * k3 + kpos;
              float a = attn[a_idx];
              float g = a * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;
              int q_base = nb_base_3d(b, od, oh, ow, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
              float q0 = query[q_base];
              float q1 = query[q_base + 1];
              float q2 = query[q_base + 2];
              float q3 = query[q_base + 3];
              int go_base = token_idx * p.D + d0;
              float go0 = grad_out[go_base];
              float go1 = grad_out[go_base + 1];
              float go2 = grad_out[go_base + 2];
              float go3 = grad_out[go_base + 3];

              acc_k0 += g * q0;
              acc_k1 += g * q1;
              acc_k2 += g * q2;
              acc_k3 += g * q3;
              acc_v0 += a * go0;
              acc_v1 += a * go1;
              acc_v2 += a * go2;
              acc_v3 += a * go3;

              atomic_fetch_add_explicit(&grad_q[q_base], g * key0, memory_order_relaxed);
              atomic_fetch_add_explicit(&grad_q[q_base + 1], g * key1, memory_order_relaxed);
              atomic_fetch_add_explicit(&grad_q[q_base + 2], g * key2, memory_order_relaxed);
              atomic_fetch_add_explicit(&grad_q[q_base + 3], g * key3, memory_order_relaxed);
            }
          }
        }
      }
    }
  }

  grad_k[out_base] = acc_k0;
  grad_k[out_base + 1] = acc_k1;
  grad_k[out_base + 2] = acc_k2;
  grad_k[out_base + 3] = acc_k3;
  grad_v[out_base] = acc_v0;
  grad_v[out_base + 1] = acc_v1;
  grad_v[out_base + 2] = acc_v2;
  grad_v[out_base + 3] = acc_v3;
}

template <int LIST_CAP>
inline void na3d_fused_bwd_kv_softmax_tiled_scalar_impl(
    device const float* attn,
    device const float* grad_attn,
    device const float* inner,
    device const float* query,
    device const float* grad_out,
    device float* grad_k,
    device float* grad_v,
    constant NA3DParams& p,
    uint tid) {
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
  int out_d = p.ID;
  int out_h = p.IH;
  int out_w = p.IW;
  int base_d = p.ID - p.K;
  int base_h = p.IH - p.K;
  int base_w = p.IW - p.K;
  int top_d_end = min(nh, p.ID);
  int top_h_end = min(nh, p.IH);
  int top_w_end = min(nh, p.IW);
  int bot_d_begin = max(0, p.ID - nh);
  int bot_h_begin = max(0, p.IH - nh);
  int bot_w_begin = max(0, p.IW - nh);
  int k3 = p.K * p.K * p.K;

  int out_base = nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D);
  float acc_k = 0.0f;
  float acc_v = 0.0f;
  for (int kd = 0; kd < p.K; ++kd) {
    int od_list[LIST_CAP];
    int n_od = nb3d_collect_out_candidates<LIST_CAP>(
        id, kd, nh, p.ID, base_d, top_d_end, bot_d_begin, od_list);
    if (n_od == 0) {
      continue;
    }

    for (int kh = 0; kh < p.K; ++kh) {
      int oh_list[LIST_CAP];
      int n_oh = nb3d_collect_out_candidates<LIST_CAP>(
          ih, kh, nh, p.IH, base_h, top_h_end, bot_h_begin, oh_list);
      if (n_oh == 0) {
        continue;
      }

      for (int kw = 0; kw < p.K; ++kw) {
        int ow_list[LIST_CAP];
        int n_ow = nb3d_collect_out_candidates<LIST_CAP>(
            iw, kw, nh, p.IW, base_w, top_w_end, bot_w_begin, ow_list);
        if (n_ow == 0) {
          continue;
        }

        int kpos = (kd * p.K + kh) * p.K + kw;
        for (int i_od = 0; i_od < n_od; ++i_od) {
          int od = od_list[i_od];
          for (int i_oh = 0; i_oh < n_oh; ++i_oh) {
            int oh = oh_list[i_oh];
            for (int i_ow = 0; i_ow < n_ow; ++i_ow) {
              int ow = ow_list[i_ow];
              int token_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
              int a_idx = token_idx * k3 + kpos;
              float a = attn[a_idx];
              float g = a * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;
              int q_base = nb_base_3d(b, od, oh, ow, h, d, p.ID, p.IH, p.IW, p.H, p.D);
              acc_k += g * query[q_base];
              acc_v += a * grad_out[token_idx * p.D + d];
            }
          }
        }
      }
    }
  }
  grad_k[out_base] = acc_k;
  grad_v[out_base] = acc_v;
}

template <int LIST_CAP>
inline void na3d_fused_bwd_kv_softmax_tiled_vec4_impl(
    device const float* attn,
    device const float* grad_attn,
    device const float* inner,
    device const float* query,
    device const float* grad_out,
    device float* grad_k,
    device float* grad_v,
    constant NA3DParams& p,
    uint tid) {
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
  int out_d = p.ID;
  int out_h = p.IH;
  int out_w = p.IW;
  int base_d = p.ID - p.K;
  int base_h = p.IH - p.K;
  int base_w = p.IW - p.K;
  int top_d_end = min(nh, p.ID);
  int top_h_end = min(nh, p.IH);
  int top_w_end = min(nh, p.IW);
  int bot_d_begin = max(0, p.ID - nh);
  int bot_h_begin = max(0, p.IH - nh);
  int bot_w_begin = max(0, p.IW - nh);
  int k3 = p.K * p.K * p.K;

  int out_base = nb_base_3d(b, id, ih, iw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
  float4 acc_k = float4(0.0f);
  float4 acc_v = float4(0.0f);
  for (int kd = 0; kd < p.K; ++kd) {
    int od_list[LIST_CAP];
    int n_od = nb3d_collect_out_candidates<LIST_CAP>(
        id, kd, nh, p.ID, base_d, top_d_end, bot_d_begin, od_list);
    if (n_od == 0) {
      continue;
    }

    for (int kh = 0; kh < p.K; ++kh) {
      int oh_list[LIST_CAP];
      int n_oh = nb3d_collect_out_candidates<LIST_CAP>(
          ih, kh, nh, p.IH, base_h, top_h_end, bot_h_begin, oh_list);
      if (n_oh == 0) {
        continue;
      }

      for (int kw = 0; kw < p.K; ++kw) {
        int ow_list[LIST_CAP];
        int n_ow = nb3d_collect_out_candidates<LIST_CAP>(
            iw, kw, nh, p.IW, base_w, top_w_end, bot_w_begin, ow_list);
        if (n_ow == 0) {
          continue;
        }

        int kpos = (kd * p.K + kh) * p.K + kw;
        for (int i_od = 0; i_od < n_od; ++i_od) {
          int od = od_list[i_od];
          for (int i_oh = 0; i_oh < n_oh; ++i_oh) {
            int oh = oh_list[i_oh];
            for (int i_ow = 0; i_ow < n_ow; ++i_ow) {
              int ow = ow_list[i_ow];
              int token_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
              int a_idx = token_idx * k3 + kpos;
              float a = attn[a_idx];
              float g = a * (grad_attn[a_idx] - inner[token_idx]) * p.SCALE;
              int q_base = nb_base_3d(b, od, oh, ow, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
              int go_base = token_idx * p.D + d0;
              const device float4* q4 = reinterpret_cast<const device float4*>(query + q_base);
              const device float4* go4 = reinterpret_cast<const device float4*>(grad_out + go_base);

              acc_k += g * q4[0];
              acc_v += a * go4[0];
            }
          }
        }
      }
    }
  }

  grad_k[out_base] = acc_k.x;
  grad_k[out_base + 1] = acc_k.y;
  grad_k[out_base + 2] = acc_k.z;
  grad_k[out_base + 3] = acc_k.w;
  grad_v[out_base] = acc_v.x;
  grad_v[out_base + 1] = acc_v.y;
  grad_v[out_base + 2] = acc_v.z;
  grad_v[out_base + 3] = acc_v.w;
}

[[kernel]] void na3d_fused_bwd_qkv_softmax_tiled_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* key [[buffer(4)]],
    device const float* grad_out [[buffer(5)]],
    device atomic<float>* grad_q [[buffer(6)]],
    device float* grad_k [[buffer(7)]],
    device float* grad_v [[buffer(8)]],
    constant NA3DParams& p [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  na3d_fused_bwd_qkv_softmax_tiled_scalar_impl<32>(
      attn, grad_attn, inner, query, key, grad_out, grad_q, grad_k, grad_v, p, tid);
}

[[kernel]] void na3d_fused_bwd_qkv_softmax_tiled_k3_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* key [[buffer(4)]],
    device const float* grad_out [[buffer(5)]],
    device atomic<float>* grad_q [[buffer(6)]],
    device float* grad_k [[buffer(7)]],
    device float* grad_v [[buffer(8)]],
    constant NA3DParams& p [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 3) {
    return;
  }
  na3d_fused_bwd_qkv_softmax_tiled_scalar_impl<4>(
      attn, grad_attn, inner, query, key, grad_out, grad_q, grad_k, grad_v, p, tid);
}

[[kernel]] void na3d_fused_bwd_qkv_softmax_tiled_k5_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* key [[buffer(4)]],
    device const float* grad_out [[buffer(5)]],
    device atomic<float>* grad_q [[buffer(6)]],
    device float* grad_k [[buffer(7)]],
    device float* grad_v [[buffer(8)]],
    constant NA3DParams& p [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 5) {
    return;
  }
  na3d_fused_bwd_qkv_softmax_tiled_scalar_impl<8>(
      attn, grad_attn, inner, query, key, grad_out, grad_q, grad_k, grad_v, p, tid);
}

[[kernel]] void na3d_fused_bwd_qkv_softmax_tiled_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* key [[buffer(4)]],
    device const float* grad_out [[buffer(5)]],
    device atomic<float>* grad_q [[buffer(6)]],
    device float* grad_k [[buffer(7)]],
    device float* grad_v [[buffer(8)]],
    constant NA3DParams& p [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  na3d_fused_bwd_qkv_softmax_tiled_vec4_impl<32>(
      attn, grad_attn, inner, query, key, grad_out, grad_q, grad_k, grad_v, p, tid);
}

[[kernel]] void na3d_fused_bwd_qkv_softmax_tiled_k3_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* key [[buffer(4)]],
    device const float* grad_out [[buffer(5)]],
    device atomic<float>* grad_q [[buffer(6)]],
    device float* grad_k [[buffer(7)]],
    device float* grad_v [[buffer(8)]],
    constant NA3DParams& p [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 3) {
    return;
  }
  na3d_fused_bwd_qkv_softmax_tiled_vec4_impl<4>(
      attn, grad_attn, inner, query, key, grad_out, grad_q, grad_k, grad_v, p, tid);
}

[[kernel]] void na3d_fused_bwd_qkv_softmax_tiled_k5_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* key [[buffer(4)]],
    device const float* grad_out [[buffer(5)]],
    device atomic<float>* grad_q [[buffer(6)]],
    device float* grad_k [[buffer(7)]],
    device float* grad_v [[buffer(8)]],
    constant NA3DParams& p [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 5) {
    return;
  }
  na3d_fused_bwd_qkv_softmax_tiled_vec4_impl<8>(
      attn, grad_attn, inner, query, key, grad_out, grad_q, grad_k, grad_v, p, tid);
}

[[kernel]] void na3d_fused_bwd_kv_softmax_tiled_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA3DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  na3d_fused_bwd_kv_softmax_tiled_scalar_impl<32>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na3d_fused_bwd_kv_softmax_tiled_k3_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA3DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 3) {
    return;
  }
  na3d_fused_bwd_kv_softmax_tiled_scalar_impl<4>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na3d_fused_bwd_kv_softmax_tiled_k5_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA3DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 5) {
    return;
  }
  na3d_fused_bwd_kv_softmax_tiled_scalar_impl<8>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na3d_fused_bwd_kv_softmax_tiled_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA3DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  na3d_fused_bwd_kv_softmax_tiled_vec4_impl<32>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na3d_fused_bwd_kv_softmax_tiled_k3_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA3DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 3) {
    return;
  }
  na3d_fused_bwd_kv_softmax_tiled_vec4_impl<4>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na3d_fused_bwd_kv_softmax_tiled_k5_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device const float* grad_out [[buffer(4)]],
    device float* grad_k [[buffer(5)]],
    device float* grad_v [[buffer(6)]],
    constant NA3DParams& p [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 5) {
    return;
  }
  na3d_fused_bwd_kv_softmax_tiled_vec4_impl<8>(
      attn, grad_attn, inner, query, grad_out, grad_k, grad_v, p, tid);
}

[[kernel]] void na3d_fused_bwd_q_softmax_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA3DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = nb_ceil_div(p.ID, p.SD);
  int out_h = nb_ceil_div(p.IH, p.SH);
  int out_w = nb_ceil_div(p.IW, p.SW);
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
  int token_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
  int attn_base = token_idx * k3;
  float inner_val = inner[token_idx];
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
        int a_idx = attn_base + kpos;
        float g = attn[a_idx] * (grad_attn[a_idx] - inner_val);
        acc += g * key[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)];
      }
    }
  }
  grad_q[idx] = acc * p.SCALE;
}

template <int K_FIXED>
inline void na3d_fused_bwd_q_softmax_k_impl(
    device const float* attn,
    device const float* grad_attn,
    device const float* inner,
    device const float* key,
    device float* grad_q,
    constant NA3DParams& p,
    uint tid) {
  int out_d = nb_ceil_div(p.ID, p.SD);
  int out_h = nb_ceil_div(p.IH, p.SH);
  int out_w = nb_ceil_div(p.IW, p.SW);
  int idx = static_cast<int>(tid);
  int total = p.B * p.ID * p.IH * p.IW * p.H * p.D;
  if (idx >= total || p.K != K_FIXED) {
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

  constexpr int NH = K_FIXED / 2;
  constexpr int K3 = K_FIXED * K_FIXED * K_FIXED;
  int token_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
  int attn_base = token_idx * K3;
  float inner_val = inner[token_idx];
  float acc = 0.0f;
  bool noncausal = (p.CD == 0 && p.CH == 0 && p.CW == 0);
  if (noncausal) {
    int d_start_nc = natten_get_window_start(z, p.ID, K_FIXED, NH, p.DD);
    int h_start_nc = natten_get_window_start(i, p.IH, K_FIXED, NH, p.DH);
    int w_start_nc = natten_get_window_start(w, p.IW, K_FIXED, NH, p.DW);
    for (int kd = 0; kd < K_FIXED; ++kd) {
      int id = d_start_nc + kd * p.DD;
      for (int kh = 0; kh < K_FIXED; ++kh) {
        int ih = h_start_nc + kh * p.DH;
        for (int kw = 0; kw < K_FIXED; ++kw) {
          int iw = w_start_nc + kw * p.DW;
          int kpos = (kd * K_FIXED + kh) * K_FIXED + kw;
          int a_idx = attn_base + kpos;
          float g = attn[a_idx] * (grad_attn[a_idx] - inner_val);
          acc += g * key[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)];
        }
      }
    }
  } else {
    for (int kd = 0; kd < K_FIXED; ++kd) {
      int id = p.CD ? (z - (K_FIXED - 1 - kd) * p.DD)
                    : (natten_get_window_start(z, p.ID, K_FIXED, NH, p.DD) + kd * p.DD);
      if (id < 0 || id >= p.ID) {
        continue;
      }
      for (int kh = 0; kh < K_FIXED; ++kh) {
        int ih = p.CH ? (i - (K_FIXED - 1 - kh) * p.DH)
                      : (natten_get_window_start(i, p.IH, K_FIXED, NH, p.DH) + kh * p.DH);
        if (ih < 0 || ih >= p.IH) {
          continue;
        }
        for (int kw = 0; kw < K_FIXED; ++kw) {
          int iw = p.CW ? (w - (K_FIXED - 1 - kw) * p.DW)
                        : (natten_get_window_start(w, p.IW, K_FIXED, NH, p.DW) + kw * p.DW);
          if (iw < 0 || iw >= p.IW) {
            continue;
          }
          int kpos = (kd * K_FIXED + kh) * K_FIXED + kw;
          int a_idx = attn_base + kpos;
          float g = attn[a_idx] * (grad_attn[a_idx] - inner_val);
          acc += g * key[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)];
        }
      }
    }
  }
  grad_q[idx] = acc * p.SCALE;
}

[[kernel]] void na3d_fused_bwd_q_softmax_k3_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA3DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  na3d_fused_bwd_q_softmax_k_impl<3>(attn, grad_attn, inner, key, grad_q, p, tid);
}

[[kernel]] void na3d_fused_bwd_q_softmax_k5_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA3DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  na3d_fused_bwd_q_softmax_k_impl<5>(attn, grad_attn, inner, key, grad_q, p, tid);
}

[[kernel]] void na3d_fused_bwd_q_softmax_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA3DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int out_d = nb_ceil_div(p.ID, p.SD);
  int out_h = nb_ceil_div(p.IH, p.SH);
  int out_w = nb_ceil_div(p.IW, p.SW);
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

  int out_base = nb_base_3d(b, z, i, w, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
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
  int d_start_nc = natten_get_window_start(z, p.ID, p.K, nh, p.DD);
  int h_start_nc = natten_get_window_start(i, p.IH, p.K, nh, p.DH);
  int w_start_nc = natten_get_window_start(w, p.IW, p.K, nh, p.DW);
  int k3 = p.K * p.K * p.K;
  int token_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
  int attn_base = token_idx * k3;
  float inner_val = inner[token_idx];
  float4 acc = float4(0.0f);
  for (int kd = 0; kd < p.K; ++kd) {
    int id = p.CD ? (z - (p.K - 1 - kd) * p.DD)
                  : (d_start_nc + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH ? (i - (p.K - 1 - kh) * p.DH)
                    : (h_start_nc + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW ? (w - (p.K - 1 - kw) * p.DW)
                      : (w_start_nc + kw * p.DW);
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int kpos = (kd * p.K + kh) * p.K + kw;
        int a_idx = attn_base + kpos;
        float g = attn[a_idx] * (grad_attn[a_idx] - inner_val);
        int k_base = nb_base_3d(b, id, ih, iw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
        const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
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

template <int K_FIXED>
inline void na3d_fused_bwd_q_softmax_vec4_k_impl(
    device const float* attn,
    device const float* grad_attn,
    device const float* inner,
    device const float* key,
    device float* grad_q,
    constant NA3DParams& p,
    uint tid) {
  int dim4 = p.D / 4;
  int out_d = nb_ceil_div(p.ID, p.SD);
  int out_h = nb_ceil_div(p.IH, p.SH);
  int out_w = nb_ceil_div(p.IW, p.SW);
  int idx = static_cast<int>(tid);
  int total = p.B * p.ID * p.IH * p.IW * p.H * dim4;
  if (idx >= total || dim4 <= 0 || p.K != K_FIXED) {
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

  int out_base = nb_base_3d(b, z, i, w, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
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

  constexpr int NH = K_FIXED / 2;
  constexpr int K3 = K_FIXED * K_FIXED * K_FIXED;
  int token_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
  int attn_base = token_idx * K3;
  float inner_val = inner[token_idx];
  float4 acc = float4(0.0f);
  bool noncausal = (p.CD == 0 && p.CH == 0 && p.CW == 0);
  if (noncausal) {
    int d_start_nc = natten_get_window_start(z, p.ID, K_FIXED, NH, p.DD);
    int h_start_nc = natten_get_window_start(i, p.IH, K_FIXED, NH, p.DH);
    int w_start_nc = natten_get_window_start(w, p.IW, K_FIXED, NH, p.DW);
    for (int kd = 0; kd < K_FIXED; ++kd) {
      int id = d_start_nc + kd * p.DD;
      for (int kh = 0; kh < K_FIXED; ++kh) {
        int ih = h_start_nc + kh * p.DH;
        for (int kw = 0; kw < K_FIXED; ++kw) {
          int iw = w_start_nc + kw * p.DW;
          int kpos = (kd * K_FIXED + kh) * K_FIXED + kw;
          int a_idx = attn_base + kpos;
          float g = attn[a_idx] * (grad_attn[a_idx] - inner_val);
          int k_base = nb_base_3d(b, id, ih, iw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
          const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
          acc += g * k4[0];
        }
      }
    }
  } else {
    int d_start_nc = natten_get_window_start(z, p.ID, K_FIXED, NH, p.DD);
    int h_start_nc = natten_get_window_start(i, p.IH, K_FIXED, NH, p.DH);
    int w_start_nc = natten_get_window_start(w, p.IW, K_FIXED, NH, p.DW);
    for (int kd = 0; kd < K_FIXED; ++kd) {
      int id = p.CD ? (z - (K_FIXED - 1 - kd) * p.DD)
                    : (d_start_nc + kd * p.DD);
      if (id < 0 || id >= p.ID) {
        continue;
      }
      for (int kh = 0; kh < K_FIXED; ++kh) {
        int ih = p.CH ? (i - (K_FIXED - 1 - kh) * p.DH)
                      : (h_start_nc + kh * p.DH);
        if (ih < 0 || ih >= p.IH) {
          continue;
        }
        for (int kw = 0; kw < K_FIXED; ++kw) {
          int iw = p.CW ? (w - (K_FIXED - 1 - kw) * p.DW)
                        : (w_start_nc + kw * p.DW);
          if (iw < 0 || iw >= p.IW) {
            continue;
          }
          int kpos = (kd * K_FIXED + kh) * K_FIXED + kw;
          int a_idx = attn_base + kpos;
          float g = attn[a_idx] * (grad_attn[a_idx] - inner_val);
          int k_base = nb_base_3d(b, id, ih, iw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
          const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
          acc += g * k4[0];
        }
      }
    }
  }
  acc *= p.SCALE;
  grad_q[out_base] = acc.x;
  grad_q[out_base + 1] = acc.y;
  grad_q[out_base + 2] = acc.z;
  grad_q[out_base + 3] = acc.w;
}

[[kernel]] void na3d_fused_bwd_q_softmax_k3_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA3DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  na3d_fused_bwd_q_softmax_vec4_k_impl<3>(attn, grad_attn, inner, key, grad_q, p, tid);
}

[[kernel]] void na3d_fused_bwd_q_softmax_k5_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA3DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  na3d_fused_bwd_q_softmax_vec4_k_impl<5>(attn, grad_attn, inner, key, grad_q, p, tid);
}

[[kernel]] void na3d_fused_bwd_q_softmax_token_k3_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* key [[buffer(3)]],
    device float* grad_q [[buffer(4)]],
    constant NA3DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  int dim4 = p.D / 4;
  int idx = static_cast<int>(tid);
  int total = p.B * p.ID * p.IH * p.IW * p.H;
  if (idx >= total || dim4 <= 0 || p.K != 3 || p.SD != 1 || p.SH != 1 || p.SW != 1 ||
      p.DD != 1 || p.DH != 1 || p.DW != 1 || p.CD != 0 || p.CH != 0 || p.CW != 0 ||
      p.ID < 3 || p.IH < 3 || p.IW < 3) {
    return;
  }

  int h = idx % p.H;
  int t = idx / p.H;
  int w = t % p.IW;
  t /= p.IW;
  int i = t % p.IH;
  t /= p.IH;
  int z = t % p.ID;
  int b = t / p.ID;

  int token_idx = ((((b * p.ID + z) * p.IH + i) * p.IW + w) * p.H + h);
  int attn_base = token_idx * 27;
  float inner_val = inner[token_idx];

  int d_start = natten_get_window_start(z, p.ID, 3, 1, 1);
  int h_start = natten_get_window_start(i, p.IH, 3, 1, 1);
  int w_start = natten_get_window_start(w, p.IW, 3, 1, 1);

  float coeff[27];
  for (int kd = 0; kd < 3; ++kd) {
    for (int kh = 0; kh < 3; ++kh) {
      for (int kw = 0; kw < 3; ++kw) {
        int kpos = (kd * 3 + kh) * 3 + kw;
        int a_idx = attn_base + kpos;
        coeff[kpos] = attn[a_idx] * (grad_attn[a_idx] - inner_val) * p.SCALE;
      }
    }
  }

  for (int d4 = 0; d4 < dim4; ++d4) {
    int d0 = d4 * 4;
    float4 acc = float4(0.0f);
    for (int kd = 0; kd < 3; ++kd) {
      int id = d_start + kd;
      for (int kh = 0; kh < 3; ++kh) {
        int ih = h_start + kh;
        for (int kw = 0; kw < 3; ++kw) {
          int iw = w_start + kw;
          int kpos = (kd * 3 + kh) * 3 + kw;
          int k_base = nb_base_3d(b, id, ih, iw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
          const device float4* k4 = reinterpret_cast<const device float4*>(key + k_base);
          acc += coeff[kpos] * k4[0];
        }
      }
    }
    int out_base = nb_base_3d(b, z, i, w, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
    grad_q[out_base] = acc.x;
    grad_q[out_base + 1] = acc.y;
    grad_q[out_base + 2] = acc.z;
    grad_q[out_base + 3] = acc.w;
  }
}

template <int LIST_CAP>
inline void na3d_fused_bwd_k_direct_softmax_scalar_impl(
    device const float* attn,
    device const float* grad_attn,
    device const float* inner,
    device const float* query,
    device float* grad_k,
    constant NA3DParams& p,
    uint tid) {
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
  int out_d = p.ID;
  int out_h = p.IH;
  int out_w = p.IW;
  int base_d = p.ID - p.K;
  int base_h = p.IH - p.K;
  int base_w = p.IW - p.K;
  int top_d_end = min(nh, p.ID);
  int top_h_end = min(nh, p.IH);
  int top_w_end = min(nh, p.IW);
  int bot_d_begin = max(0, p.ID - nh);
  int bot_h_begin = max(0, p.IH - nh);
  int bot_w_begin = max(0, p.IW - nh);
  int k3 = p.K * p.K * p.K;

  float acc = 0.0f;
  for (int kd = 0; kd < p.K; ++kd) {
    int od_list[LIST_CAP];
    int n_od = nb3d_collect_out_candidates<LIST_CAP>(
        id, kd, nh, p.ID, base_d, top_d_end, bot_d_begin, od_list);
    if (n_od == 0) {
      continue;
    }

    for (int kh = 0; kh < p.K; ++kh) {
      int oh_list[LIST_CAP];
      int n_oh = nb3d_collect_out_candidates<LIST_CAP>(
          ih, kh, nh, p.IH, base_h, top_h_end, bot_h_begin, oh_list);
      if (n_oh == 0) {
        continue;
      }

      for (int kw = 0; kw < p.K; ++kw) {
        int ow_list[LIST_CAP];
        int n_ow = nb3d_collect_out_candidates<LIST_CAP>(
            iw, kw, nh, p.IW, base_w, top_w_end, bot_w_begin, ow_list);
        if (n_ow == 0) {
          continue;
        }

        int kpos = (kd * p.K + kh) * p.K + kw;
        for (int i_od = 0; i_od < n_od; ++i_od) {
          int od = od_list[i_od];
          for (int i_oh = 0; i_oh < n_oh; ++i_oh) {
            int oh = oh_list[i_oh];
            for (int i_ow = 0; i_ow < n_ow; ++i_ow) {
              int ow = ow_list[i_ow];
              int token_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
              int a_idx = token_idx * k3 + kpos;
              float g = attn[a_idx] * (grad_attn[a_idx] - inner[token_idx]);
              acc += g * query[nb_base_3d(b, od, oh, ow, h, d, p.ID, p.IH, p.IW, p.H, p.D)];
            }
          }
        }
      }
    }
  }
  grad_k[idx] = acc * p.SCALE;
}

template <int LIST_CAP>
inline void na3d_fused_bwd_k_direct_softmax_vec4_impl(
    device const float* attn,
    device const float* grad_attn,
    device const float* inner,
    device const float* query,
    device float* grad_k,
    constant NA3DParams& p,
    uint tid) {
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
  int out_d = p.ID;
  int out_h = p.IH;
  int out_w = p.IW;
  int base_d = p.ID - p.K;
  int base_h = p.IH - p.K;
  int base_w = p.IW - p.K;
  int top_d_end = min(nh, p.ID);
  int top_h_end = min(nh, p.IH);
  int top_w_end = min(nh, p.IW);
  int bot_d_begin = max(0, p.ID - nh);
  int bot_h_begin = max(0, p.IH - nh);
  int bot_w_begin = max(0, p.IW - nh);
  int k3 = p.K * p.K * p.K;

  float4 acc = float4(0.0f);
  for (int kd = 0; kd < p.K; ++kd) {
    int od_list[LIST_CAP];
    int n_od = nb3d_collect_out_candidates<LIST_CAP>(
        id, kd, nh, p.ID, base_d, top_d_end, bot_d_begin, od_list);
    if (n_od == 0) {
      continue;
    }

    for (int kh = 0; kh < p.K; ++kh) {
      int oh_list[LIST_CAP];
      int n_oh = nb3d_collect_out_candidates<LIST_CAP>(
          ih, kh, nh, p.IH, base_h, top_h_end, bot_h_begin, oh_list);
      if (n_oh == 0) {
        continue;
      }

      for (int kw = 0; kw < p.K; ++kw) {
        int ow_list[LIST_CAP];
        int n_ow = nb3d_collect_out_candidates<LIST_CAP>(
            iw, kw, nh, p.IW, base_w, top_w_end, bot_w_begin, ow_list);
        if (n_ow == 0) {
          continue;
        }

        int kpos = (kd * p.K + kh) * p.K + kw;
        for (int i_od = 0; i_od < n_od; ++i_od) {
          int od = od_list[i_od];
          for (int i_oh = 0; i_oh < n_oh; ++i_oh) {
            int oh = oh_list[i_oh];
            for (int i_ow = 0; i_ow < n_ow; ++i_ow) {
              int ow = ow_list[i_ow];
              int token_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
              int a_idx = token_idx * k3 + kpos;
              float g = attn[a_idx] * (grad_attn[a_idx] - inner[token_idx]);
              int q_base = nb_base_3d(b, od, oh, ow, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
              const device float4* q4 = reinterpret_cast<const device float4*>(query + q_base);
              acc += g * q4[0];
            }
          }
        }
      }
    }
  }
  int out_base = nb_base_3d(b, id, ih, iw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
  acc *= p.SCALE;
  grad_k[out_base] = acc.x;
  grad_k[out_base + 1] = acc.y;
  grad_k[out_base + 2] = acc.z;
  grad_k[out_base + 3] = acc.w;
}

inline void na3d_fused_bwd_k_direct_softmax_k3_scalar_impl(
    device const float* attn,
    device const float* grad_attn,
    device const float* inner,
    device const float* query,
    device float* grad_k,
    constant NA3DParams& p,
    uint tid) {
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

  constexpr int K_FIXED = 3;
  constexpr int NH = 1;
  constexpr int K3 = 27;
  int out_d = p.ID;
  int out_h = p.IH;
  int out_w = p.IW;
  int base_d = p.ID - K_FIXED;
  int base_h = p.IH - K_FIXED;
  int base_w = p.IW - K_FIXED;
  int top_d_end = min(NH, p.ID);
  int top_h_end = min(NH, p.IH);
  int top_w_end = min(NH, p.IW);
  int bot_d_begin = max(0, p.ID - NH);
  int bot_h_begin = max(0, p.IH - NH);
  int bot_w_begin = max(0, p.IW - NH);

  float acc = 0.0f;
  for (int kd = 0; kd < K_FIXED; ++kd) {
    int od_list[4];
    int n_od = nb3d_collect_out_candidates<4>(
        id, kd, NH, p.ID, base_d, top_d_end, bot_d_begin, od_list);
    if (n_od == 0) {
      continue;
    }
    for (int kh = 0; kh < K_FIXED; ++kh) {
      int oh_list[4];
      int n_oh = nb3d_collect_out_candidates<4>(
          ih, kh, NH, p.IH, base_h, top_h_end, bot_h_begin, oh_list);
      if (n_oh == 0) {
        continue;
      }
      for (int kw = 0; kw < K_FIXED; ++kw) {
        int ow_list[4];
        int n_ow = nb3d_collect_out_candidates<4>(
            iw, kw, NH, p.IW, base_w, top_w_end, bot_w_begin, ow_list);
        if (n_ow == 0) {
          continue;
        }
        int kpos = (kd * K_FIXED + kh) * K_FIXED + kw;
        for (int i_od = 0; i_od < n_od; ++i_od) {
          int od = od_list[i_od];
          for (int i_oh = 0; i_oh < n_oh; ++i_oh) {
            int oh = oh_list[i_oh];
            for (int i_ow = 0; i_ow < n_ow; ++i_ow) {
              int ow = ow_list[i_ow];
              int token_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
              int a_idx = token_idx * K3 + kpos;
              float g = attn[a_idx] * (grad_attn[a_idx] - inner[token_idx]);
              acc += g * query[nb_base_3d(b, od, oh, ow, h, d, p.ID, p.IH, p.IW, p.H, p.D)];
            }
          }
        }
      }
    }
  }
  grad_k[idx] = acc * p.SCALE;
}

inline void na3d_fused_bwd_k_direct_softmax_k3_vec4_impl(
    device const float* attn,
    device const float* grad_attn,
    device const float* inner,
    device const float* query,
    device float* grad_k,
    constant NA3DParams& p,
    uint tid) {
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

  constexpr int K_FIXED = 3;
  constexpr int NH = 1;
  constexpr int K3 = 27;
  int out_d = p.ID;
  int out_h = p.IH;
  int out_w = p.IW;
  int base_d = p.ID - K_FIXED;
  int base_h = p.IH - K_FIXED;
  int base_w = p.IW - K_FIXED;
  int top_d_end = min(NH, p.ID);
  int top_h_end = min(NH, p.IH);
  int top_w_end = min(NH, p.IW);
  int bot_d_begin = max(0, p.ID - NH);
  int bot_h_begin = max(0, p.IH - NH);
  int bot_w_begin = max(0, p.IW - NH);

  float4 acc = float4(0.0f);
  for (int kd = 0; kd < K_FIXED; ++kd) {
    int od_list[4];
    int n_od = nb3d_collect_out_candidates<4>(
        id, kd, NH, p.ID, base_d, top_d_end, bot_d_begin, od_list);
    if (n_od == 0) {
      continue;
    }
    for (int kh = 0; kh < K_FIXED; ++kh) {
      int oh_list[4];
      int n_oh = nb3d_collect_out_candidates<4>(
          ih, kh, NH, p.IH, base_h, top_h_end, bot_h_begin, oh_list);
      if (n_oh == 0) {
        continue;
      }
      for (int kw = 0; kw < K_FIXED; ++kw) {
        int ow_list[4];
        int n_ow = nb3d_collect_out_candidates<4>(
            iw, kw, NH, p.IW, base_w, top_w_end, bot_w_begin, ow_list);
        if (n_ow == 0) {
          continue;
        }
        int kpos = (kd * K_FIXED + kh) * K_FIXED + kw;
        for (int i_od = 0; i_od < n_od; ++i_od) {
          int od = od_list[i_od];
          for (int i_oh = 0; i_oh < n_oh; ++i_oh) {
            int oh = oh_list[i_oh];
            for (int i_ow = 0; i_ow < n_ow; ++i_ow) {
              int ow = ow_list[i_ow];
              int token_idx = ((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h);
              int a_idx = token_idx * K3 + kpos;
              float g = attn[a_idx] * (grad_attn[a_idx] - inner[token_idx]);
              int q_base = nb_base_3d(b, od, oh, ow, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
              const device float4* q4 = reinterpret_cast<const device float4*>(query + q_base);
              acc += g * q4[0];
            }
          }
        }
      }
    }
  }
  int out_base = nb_base_3d(b, id, ih, iw, h, d0, p.ID, p.IH, p.IW, p.H, p.D);
  acc *= p.SCALE;
  grad_k[out_base] = acc.x;
  grad_k[out_base + 1] = acc.y;
  grad_k[out_base + 2] = acc.z;
  grad_k[out_base + 3] = acc.w;
}

[[kernel]] void na3d_fused_bwd_k_direct_softmax_u1d1_nc_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device float* grad_k [[buffer(4)]],
    constant NA3DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  na3d_fused_bwd_k_direct_softmax_scalar_impl<32>(attn, grad_attn, inner, query, grad_k, p, tid);
}

[[kernel]] void na3d_fused_bwd_k_direct_softmax_u1d1_nc_k3_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device float* grad_k [[buffer(4)]],
    constant NA3DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 3) {
    return;
  }
  na3d_fused_bwd_k_direct_softmax_k3_scalar_impl(attn, grad_attn, inner, query, grad_k, p, tid);
}

[[kernel]] void na3d_fused_bwd_k_direct_softmax_u1d1_nc_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device float* grad_k [[buffer(4)]],
    constant NA3DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  na3d_fused_bwd_k_direct_softmax_vec4_impl<32>(attn, grad_attn, inner, query, grad_k, p, tid);
}

[[kernel]] void na3d_fused_bwd_k_direct_softmax_u1d1_nc_k3_vec4_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_attn [[buffer(1)]],
    device const float* inner [[buffer(2)]],
    device const float* query [[buffer(3)]],
    device float* grad_k [[buffer(4)]],
    constant NA3DParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  if (p.K != 3) {
    return;
  }
  na3d_fused_bwd_k_direct_softmax_k3_vec4_impl(attn, grad_attn, inner, query, grad_k, p, tid);
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
  int id = p.CD ? (qd - (p.K - 1 - kd) * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - (p.K - 1 - kh) * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - (p.K - 1 - kw) * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int a_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3 + kpos);
  int go_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D + d);
  float g = attn[a_idx] * grad_out[go_idx];
  atomic_fetch_add_explicit(&grad_v[nb_base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], g, memory_order_relaxed);
}

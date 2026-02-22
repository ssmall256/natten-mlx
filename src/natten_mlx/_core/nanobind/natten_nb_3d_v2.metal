// High-performance 3D neighborhood attention Metal kernels.
// Key optimizations over v1:
//   - Spatial-first layout [B, ID, IH, IW, H, D] (matches MLX default — zero transpose)
//   - Stored logits + key indices to avoid 3x dot product recomputation
//   - SIMD vec4 dot products via float4/half4/bfloat4
//   - 3D grid (out_w, out_h, B*H*out_d)

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

struct NA3DParamsV2 {
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

// Window helpers
inline int natten_window_start_3d(
    int index, int length, int kernel_size, int neighborhood_size, int dilation) {
  if (dilation <= 1) {
    return max(index - neighborhood_size, 0) +
        ((index + neighborhood_size >= length)
             ? (length - index - neighborhood_size - 1) : 0);
  }
  int ni = index - neighborhood_size * dilation;
  if (ni < 0) return index % dilation;
  if (index + neighborhood_size * dilation >= length) {
    int imod = index % dilation;
    int a = (length / dilation) * dilation;
    int b = length - a;
    if (imod < b) return length - b + imod - 2 * neighborhood_size * dilation;
    return a + imod - kernel_size * dilation;
  }
  return ni;
}

// Spatial-first: [B, ID, IH, IW, H, D] -> linear = (((((b*ID+id)*IH+ih)*IW+iw)*H+h)*D+d)
inline int sf_base_3d(int b, int h, int id, int ih, int iw,
                      int ID, int IH, int IW, int H, int D) {
  return (((((b * ID + id) * IH + ih) * IW + iw) * H + h) * D);
}

// ======================================================================
// 3D Fused Forward — stored logits, scalar
// ======================================================================

template <typename T>
[[kernel]] void na3d_fused_v2_stored_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA3DParamsV2& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {

  int out_d = (p.ID + p.SD - 1) / p.SD;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;

  int ow = (int)gid.x;
  int oh = (int)gid.y;
  int z = (int)gid.z;
  if (ow >= out_w || oh >= out_h || z >= p.B * p.H * out_d) return;

  int od = z % out_d;
  int bh = z / out_d;
  int b = bh / p.H;
  int h = bh - b * p.H;

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) return;

  int K = p.K;
  int nh = K / 2;
  int K3 = K * K * K;

  int q_base = sf_base_3d(b, h, qd, qh, qw, p.ID, p.IH, p.IW, p.H, p.D);

  // Pass 1: compute scores, store logits
  float logits[343];  // max K=7 -> 343 neighbors for 3D
  int key_d[343];
  int key_h[343];
  int key_w[343];
  float max_logit = -INFINITY;
  int n_idx = 0;

  int nd = p.CD ? 0 : natten_window_start_3d(qd, p.ID, K, nh, p.DD);
  int nh_start = p.CH ? 0 : natten_window_start_3d(qh, p.IH, K, nh, p.DH);
  int nw = p.CW ? 0 : natten_window_start_3d(qw, p.IW, K, nh, p.DW);

  for (int kd = 0; kd < K; ++kd) {
    int id = p.CD ? (qd - (K - 1 - kd) * p.DD) : (nd + kd * p.DD);
    bool valid_d = p.CD ? (id >= 0 && id <= qd && id < p.ID) : (id >= 0 && id < p.ID);
    for (int kh = 0; kh < K; ++kh) {
      int ih = p.CH ? (qh - (K - 1 - kh) * p.DH) : (nh_start + kh * p.DH);
      bool valid_h = p.CH ? (ih >= 0 && ih <= qh && ih < p.IH) : (ih >= 0 && ih < p.IH);
      for (int kw = 0; kw < K; ++kw) {
        int iw = p.CW ? (qw - (K - 1 - kw) * p.DW) : (nw + kw * p.DW);
        bool valid_w = p.CW ? (iw >= 0 && iw <= qw && iw < p.IW) : (iw >= 0 && iw < p.IW);

        float score = -INFINITY;
        int kd_out = -1, kh_out = -1, kw_out = -1;
        if (valid_d && valid_h && valid_w) {
          kd_out = id; kh_out = ih; kw_out = iw;
          int k_base = sf_base_3d(b, h, id, ih, iw, p.ID, p.IH, p.IW, p.H, p.D);
          float acc = 0.0f;
          for (int d = 0; d < p.D; ++d) {
            acc += (float)query[q_base + d] * (float)key[k_base + d];
          }
          score = acc * p.SCALE;
        }
        logits[n_idx] = score;
        key_d[n_idx] = kd_out;
        key_h[n_idx] = kh_out;
        key_w[n_idx] = kw_out;
        max_logit = max(max_logit, score);
        n_idx++;
      }
    }
  }

  // Pass 2: softmax
  float denom = 0.0f;
  for (int n = 0; n < K3; ++n) {
    logits[n] = exp(logits[n] - max_logit);
    denom += logits[n];
  }
  float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

  // Pass 3: weighted value aggregation
  int out_base = sf_base_3d(b, h, od, oh, ow, out_d, out_h, out_w, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    float acc = 0.0f;
    for (int n = 0; n < K3; ++n) {
      if (key_d[n] >= 0) {
        float w = logits[n] * inv_denom;
        int v_idx = sf_base_3d(b, h, key_d[n], key_h[n], key_w[n],
                               p.ID, p.IH, p.IW, p.H, p.D) + d;
        acc += w * (float)value[v_idx];
      }
    }
    out[out_base + d] = (T)acc;
  }
}

// ======================================================================
// 3D Fused Forward — stored logits, vec4
// ======================================================================

template <typename T, typename T4>
[[kernel]] void na3d_fused_v2_stored_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA3DParamsV2& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {

  int out_d = (p.ID + p.SD - 1) / p.SD;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;

  int ow = (int)gid.x;
  int oh = (int)gid.y;
  int z = (int)gid.z;
  if (ow >= out_w || oh >= out_h || z >= p.B * p.H * out_d) return;

  int od = z % out_d;
  int bh = z / out_d;
  int b = bh / p.H;
  int h = bh - b * p.H;

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) return;

  int K = p.K;
  int nh = K / 2;
  int K3 = K * K * K;
  int dim4 = p.D / 4;

  int q_base = sf_base_3d(b, h, qd, qh, qw, p.ID, p.IH, p.IW, p.H, p.D);

  float logits[343];
  int key_d[343];
  int key_h[343];
  int key_w[343];
  float max_logit = -INFINITY;
  int n_idx = 0;

  int nd = p.CD ? 0 : natten_window_start_3d(qd, p.ID, K, nh, p.DD);
  int nh_start = p.CH ? 0 : natten_window_start_3d(qh, p.IH, K, nh, p.DH);
  int nw = p.CW ? 0 : natten_window_start_3d(qw, p.IW, K, nh, p.DW);

  for (int kd = 0; kd < K; ++kd) {
    int id = p.CD ? (qd - (K - 1 - kd) * p.DD) : (nd + kd * p.DD);
    bool valid_d = p.CD ? (id >= 0 && id <= qd && id < p.ID) : (id >= 0 && id < p.ID);
    for (int kh = 0; kh < K; ++kh) {
      int ih = p.CH ? (qh - (K - 1 - kh) * p.DH) : (nh_start + kh * p.DH);
      bool valid_h = p.CH ? (ih >= 0 && ih <= qh && ih < p.IH) : (ih >= 0 && ih < p.IH);
      for (int kw = 0; kw < K; ++kw) {
        int iw = p.CW ? (qw - (K - 1 - kw) * p.DW) : (nw + kw * p.DW);
        bool valid_w = p.CW ? (iw >= 0 && iw <= qw && iw < p.IW) : (iw >= 0 && iw < p.IW);

        float score = -INFINITY;
        int kd_out = -1, kh_out = -1, kw_out = -1;
        if (valid_d && valid_h && valid_w) {
          kd_out = id; kh_out = ih; kw_out = iw;
          int k_base = sf_base_3d(b, h, id, ih, iw, p.ID, p.IH, p.IW, p.H, p.D);
          float acc = 0.0f;
          for (int d4 = 0; d4 < dim4; ++d4) {
            const device T4* q4 = reinterpret_cast<const device T4*>(query + q_base + d4 * 4);
            const device T4* k4 = reinterpret_cast<const device T4*>(key + k_base + d4 * 4);
            acc += dot(float4(*q4), float4(*k4));
          }
          score = acc * p.SCALE;
        }
        logits[n_idx] = score;
        key_d[n_idx] = kd_out;
        key_h[n_idx] = kh_out;
        key_w[n_idx] = kw_out;
        max_logit = max(max_logit, score);
        n_idx++;
      }
    }
  }

  // Softmax
  float denom = 0.0f;
  for (int n = 0; n < K3; ++n) {
    logits[n] = exp(logits[n] - max_logit);
    denom += logits[n];
  }
  float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

  // Weighted value aggregation (vec4)
  int out_base = sf_base_3d(b, h, od, oh, ow, out_d, out_h, out_w, p.H, p.D);
  for (int d4 = 0; d4 < dim4; ++d4) {
    float4 acc = float4(0.0f);
    for (int n = 0; n < K3; ++n) {
      if (key_d[n] >= 0) {
        float w = logits[n] * inv_denom;
        int v_base = sf_base_3d(b, h, key_d[n], key_h[n], key_w[n],
                                p.ID, p.IH, p.IW, p.H, p.D) + d4 * 4;
        const device T4* v4 = reinterpret_cast<const device T4*>(value + v_base);
        acc += w * float4(*v4);
      }
    }
    device T4* o4 = reinterpret_cast<device T4*>(out + out_base + d4 * 4);
    *o4 = T4(acc);
  }
}

// ======================================================================
// 3D Backward — Fused attn recompute + grad_attn
// Thread grid: (out_w, out_h, B*H*out_d)
// ======================================================================

template <typename T>
[[kernel]] void na3d_bwd_attn_v2_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device const T* grad_out [[buffer(3)]],
    device float* attn_out [[buffer(4)]],
    device float* grad_attn_out [[buffer(5)]],
    constant NA3DParamsV2& p [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {

  int out_d = (p.ID + p.SD - 1) / p.SD;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;

  int ow = (int)gid.x;
  int oh = (int)gid.y;
  int z = (int)gid.z;
  if (ow >= out_w || oh >= out_h || z >= p.B * p.H * out_d) return;

  int od = z % out_d;
  int bh = z / out_d;
  int b = bh / p.H;
  int h = bh - b * p.H;

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;

  int K = p.K;
  int K3 = K * K * K;
  int nh = K / 2;
  int out_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * K3);

  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) {
    for (int n = 0; n < K3; ++n) {
      attn_out[out_base + n] = 0.0f;
      grad_attn_out[out_base + n] = 0.0f;
    }
    return;
  }

  int q_base = sf_base_3d(b, h, qd, qh, qw, p.ID, p.IH, p.IW, p.H, p.D);
  int go_base = sf_base_3d(b, h, od, oh, ow, out_d, out_h, out_w, p.H, p.D);

  int nd = p.CD ? 0 : natten_window_start_3d(qd, p.ID, K, nh, p.DD);
  int nh_start = p.CH ? 0 : natten_window_start_3d(qh, p.IH, K, nh, p.DH);
  int nw = p.CW ? 0 : natten_window_start_3d(qw, p.IW, K, nh, p.DW);

  float scores[343];
  float ga_vals[343];
  int key_valid[343];
  float max_score = -INFINITY;
  int n_idx = 0;

  for (int kd = 0; kd < K; ++kd) {
    int id = p.CD ? (qd - (K - 1 - kd) * p.DD) : (nd + kd * p.DD);
    bool valid_d = p.CD ? (id >= 0 && id <= qd && id < p.ID) : (id >= 0 && id < p.ID);
    for (int kh = 0; kh < K; ++kh) {
      int ih = p.CH ? (qh - (K - 1 - kh) * p.DH) : (nh_start + kh * p.DH);
      bool valid_h = p.CH ? (ih >= 0 && ih <= qh && ih < p.IH) : (ih >= 0 && ih < p.IH);
      for (int kw = 0; kw < K; ++kw) {
        int iw = p.CW ? (qw - (K - 1 - kw) * p.DW) : (nw + kw * p.DW);
        bool valid_w = p.CW ? (iw >= 0 && iw <= qw && iw < p.IW) : (iw >= 0 && iw < p.IW);

        float score = -INFINITY;
        float ga = 0.0f;
        int v_flag = 0;
        if (valid_d && valid_h && valid_w) {
          int k_base = sf_base_3d(b, h, id, ih, iw, p.ID, p.IH, p.IW, p.H, p.D);
          float s = 0.0f;
          for (int d = 0; d < p.D; ++d) {
            s += (float)query[q_base + d] * (float)key[k_base + d];
            ga += (float)grad_out[go_base + d] * (float)value[k_base + d];
          }
          score = s * p.SCALE;
          v_flag = 1;
        }
        scores[n_idx] = score;
        ga_vals[n_idx] = ga;
        key_valid[n_idx] = v_flag;
        max_score = max(max_score, score);
        n_idx++;
      }
    }
  }

  float denom = 0.0f;
  for (int n = 0; n < K3; ++n) {
    if (key_valid[n]) {
      scores[n] = exp(scores[n] - max_score);
      denom += scores[n];
    } else {
      scores[n] = 0.0f;
    }
  }
  float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

  for (int n = 0; n < K3; ++n) {
    attn_out[out_base + n] = scores[n] * inv_denom;
    grad_attn_out[out_base + n] = ga_vals[n];
  }
}

// Vec4 variant
template <typename T, typename T4>
[[kernel]] void na3d_bwd_attn_v2_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device const T* grad_out [[buffer(3)]],
    device float* attn_out [[buffer(4)]],
    device float* grad_attn_out [[buffer(5)]],
    constant NA3DParamsV2& p [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {

  int out_d = (p.ID + p.SD - 1) / p.SD;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;

  int ow = (int)gid.x;
  int oh = (int)gid.y;
  int z = (int)gid.z;
  if (ow >= out_w || oh >= out_h || z >= p.B * p.H * out_d) return;

  int od = z % out_d;
  int bh = z / out_d;
  int b = bh / p.H;
  int h = bh - b * p.H;

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;

  int K = p.K;
  int K3 = K * K * K;
  int nh = K / 2;
  int dim4 = p.D / 4;
  int out_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * K3);

  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) {
    for (int n = 0; n < K3; ++n) {
      attn_out[out_base + n] = 0.0f;
      grad_attn_out[out_base + n] = 0.0f;
    }
    return;
  }

  int q_base = sf_base_3d(b, h, qd, qh, qw, p.ID, p.IH, p.IW, p.H, p.D);
  int go_base = sf_base_3d(b, h, od, oh, ow, out_d, out_h, out_w, p.H, p.D);

  int nd = p.CD ? 0 : natten_window_start_3d(qd, p.ID, K, nh, p.DD);
  int nh_start = p.CH ? 0 : natten_window_start_3d(qh, p.IH, K, nh, p.DH);
  int nw = p.CW ? 0 : natten_window_start_3d(qw, p.IW, K, nh, p.DW);

  float scores[343];
  float ga_vals[343];
  int key_valid[343];
  float max_score = -INFINITY;
  int n_idx = 0;

  for (int kd = 0; kd < K; ++kd) {
    int id = p.CD ? (qd - (K - 1 - kd) * p.DD) : (nd + kd * p.DD);
    bool valid_d = p.CD ? (id >= 0 && id <= qd && id < p.ID) : (id >= 0 && id < p.ID);
    for (int kh = 0; kh < K; ++kh) {
      int ih = p.CH ? (qh - (K - 1 - kh) * p.DH) : (nh_start + kh * p.DH);
      bool valid_h = p.CH ? (ih >= 0 && ih <= qh && ih < p.IH) : (ih >= 0 && ih < p.IH);
      for (int kw = 0; kw < K; ++kw) {
        int iw = p.CW ? (qw - (K - 1 - kw) * p.DW) : (nw + kw * p.DW);
        bool valid_w = p.CW ? (iw >= 0 && iw <= qw && iw < p.IW) : (iw >= 0 && iw < p.IW);

        float score = -INFINITY;
        float ga = 0.0f;
        int v_flag = 0;
        if (valid_d && valid_h && valid_w) {
          int k_base = sf_base_3d(b, h, id, ih, iw, p.ID, p.IH, p.IW, p.H, p.D);
          float s = 0.0f;
          for (int d4 = 0; d4 < dim4; ++d4) {
            const device T4* q4 = reinterpret_cast<const device T4*>(query + q_base + d4 * 4);
            const device T4* k4 = reinterpret_cast<const device T4*>(key + k_base + d4 * 4);
            s += dot(float4(*q4), float4(*k4));
            const device T4* go4 = reinterpret_cast<const device T4*>(grad_out + go_base + d4 * 4);
            const device T4* v4 = reinterpret_cast<const device T4*>(value + k_base + d4 * 4);
            ga += dot(float4(*go4), float4(*v4));
          }
          score = s * p.SCALE;
          v_flag = 1;
        }
        scores[n_idx] = score;
        ga_vals[n_idx] = ga;
        key_valid[n_idx] = v_flag;
        max_score = max(max_score, score);
        n_idx++;
      }
    }
  }

  float denom = 0.0f;
  for (int n = 0; n < K3; ++n) {
    if (key_valid[n]) {
      scores[n] = exp(scores[n] - max_score);
      denom += scores[n];
    } else {
      scores[n] = 0.0f;
    }
  }
  float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

  for (int n = 0; n < K3; ++n) {
    attn_out[out_base + n] = scores[n] * inv_denom;
    grad_attn_out[out_base + n] = ga_vals[n];
  }
}

// ======================================================================
// 3D Backward — grad_q
// Thread grid: (IW, IH, B*H*ID)
// ======================================================================

template <typename T>
[[kernel]] void na3d_bwd_grad_q_v2_kernel(
    device const float* grad_logits [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device T* grad_q [[buffer(2)]],
    constant NA3DParamsV2& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {

  int iw = (int)gid.x;
  int ih = (int)gid.y;
  int z = (int)gid.z;
  if (iw >= p.IW || ih >= p.IH || z >= p.B * p.H * p.ID) return;

  int id = z % p.ID;
  int bh = z / p.ID;
  int b = bh / p.H;
  int h = bh - b * p.H;

  int out_d = (p.ID + p.SD - 1) / p.SD;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;
  int K = p.K;
  int K3 = K * K * K;
  int nh = K / 2;

  int base = sf_base_3d(b, h, id, ih, iw, p.ID, p.IH, p.IW, p.H, p.D);

  if ((id % p.SD) != 0 || (ih % p.SH) != 0 || (iw % p.SW) != 0) {
    for (int d = 0; d < p.D; ++d) grad_q[base + d] = (T)0.0f;
    return;
  }
  int od = id / p.SD;
  int oh = ih / p.SH;
  int ow = iw / p.SW;
  if (od >= out_d || oh >= out_h || ow >= out_w) {
    for (int d = 0; d < p.D; ++d) grad_q[base + d] = (T)0.0f;
    return;
  }

  int nd = p.CD ? 0 : natten_window_start_3d(id, p.ID, K, nh, p.DD);
  int nh_start = p.CH ? 0 : natten_window_start_3d(ih, p.IH, K, nh, p.DH);
  int nw = p.CW ? 0 : natten_window_start_3d(iw, p.IW, K, nh, p.DW);

  int gl_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * K3);

  for (int d = 0; d < p.D; ++d) {
    float acc = 0.0f;
    int n_idx = 0;
    for (int kd = 0; kd < K; ++kd) {
      int kid = p.CD ? (id - (K - 1 - kd) * p.DD) : (nd + kd * p.DD);
      bool vd = p.CD ? (kid >= 0 && kid <= id && kid < p.ID) : (kid >= 0 && kid < p.ID);
      for (int kh = 0; kh < K; ++kh) {
        int kih = p.CH ? (ih - (K - 1 - kh) * p.DH) : (nh_start + kh * p.DH);
        bool vh = p.CH ? (kih >= 0 && kih <= ih && kih < p.IH) : (kih >= 0 && kih < p.IH);
        for (int kw = 0; kw < K; ++kw) {
          int kiw = p.CW ? (iw - (K - 1 - kw) * p.DW) : (nw + kw * p.DW);
          bool vw = p.CW ? (kiw >= 0 && kiw <= iw && kiw < p.IW) : (kiw >= 0 && kiw < p.IW);
          if (vd && vh && vw) {
            float gl = grad_logits[gl_base + n_idx];
            int k_idx = sf_base_3d(b, h, kid, kih, kiw, p.ID, p.IH, p.IW, p.H, p.D) + d;
            acc += gl * key[k_idx];
          }
          n_idx++;
        }
      }
    }
    grad_q[base + d] = (T)(acc * p.SCALE);
  }
}

// ======================================================================
// 3D Backward — grad_k (direct nonatomic)
// Thread grid: (IW, IH, B*H*ID)
// ======================================================================

template <typename T>
[[kernel]] void na3d_bwd_grad_k_v2_kernel(
    device const float* grad_logits [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device T* grad_k [[buffer(2)]],
    constant NA3DParamsV2& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {

  int kw = (int)gid.x;
  int kh = (int)gid.y;
  int z = (int)gid.z;
  if (kw >= p.IW || kh >= p.IH || z >= p.B * p.H * p.ID) return;

  int kd = z % p.ID;
  int bh = z / p.ID;
  int b = bh / p.H;
  int h = bh - b * p.H;

  int out_d = (p.ID + p.SD - 1) / p.SD;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;
  int K = p.K;
  int K3 = K * K * K;
  int nh = K / 2;
  int k_base = sf_base_3d(b, h, kd, kh, kw, p.ID, p.IH, p.IW, p.H, p.D);

  for (int d = 0; d < p.D; ++d) {
    float acc = 0.0f;

    for (int od = 0; od < out_d; ++od) {
      int qd = od * p.SD;
      if (qd >= p.ID) continue;

      int nd = p.CD ? 0 : natten_window_start_3d(qd, p.ID, K, nh, p.DD);
      int kd_offset = -1;
      if (p.CD) {
        int diff = qd - kd;
        if (diff >= 0 && diff % p.DD == 0) {
          int ki = (K - 1) - diff / p.DD;
          if (ki >= 0 && ki < K) kd_offset = ki;
        }
      } else {
        int diff = kd - nd;
        if (diff >= 0 && diff % p.DD == 0) {
          int ki = diff / p.DD;
          if (ki >= 0 && ki < K) kd_offset = ki;
        }
      }
      if (kd_offset < 0) continue;

      for (int oh = 0; oh < out_h; ++oh) {
        int qh = oh * p.SH;
        if (qh >= p.IH) continue;

        int nh_start = p.CH ? 0 : natten_window_start_3d(qh, p.IH, K, nh, p.DH);
        int kh_offset = -1;
        if (p.CH) {
          int diff = qh - kh;
          if (diff >= 0 && diff % p.DH == 0) {
            int ki = (K - 1) - diff / p.DH;
            if (ki >= 0 && ki < K) kh_offset = ki;
          }
        } else {
          int diff = kh - nh_start;
          if (diff >= 0 && diff % p.DH == 0) {
            int ki = diff / p.DH;
            if (ki >= 0 && ki < K) kh_offset = ki;
          }
        }
        if (kh_offset < 0) continue;

        for (int ow = 0; ow < out_w; ++ow) {
          int qw = ow * p.SW;
          if (qw >= p.IW) continue;

          int nw = p.CW ? 0 : natten_window_start_3d(qw, p.IW, K, nh, p.DW);
          int kw_offset = -1;
          if (p.CW) {
            int diff = qw - kw;
            if (diff >= 0 && diff % p.DW == 0) {
              int ki = (K - 1) - diff / p.DW;
              if (ki >= 0 && ki < K) kw_offset = ki;
            }
          } else {
            int diff = kw - nw;
            if (diff >= 0 && diff % p.DW == 0) {
              int ki = diff / p.DW;
              if (ki >= 0 && ki < K) kw_offset = ki;
            }
          }
          if (kw_offset < 0) continue;

          int kpos = (kd_offset * K + kh_offset) * K + kw_offset;
          int gl_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * K3) + kpos;
          int q_idx = sf_base_3d(b, h, qd, qh, qw, p.ID, p.IH, p.IW, p.H, p.D) + d;
          acc += grad_logits[gl_idx] * query[q_idx];
        }
      }
    }
    grad_k[k_base + d] = (T)(acc * p.SCALE);
  }
}

// ======================================================================
// 3D Backward — grad_v (direct nonatomic)
// Thread grid: (IW, IH, B*H*ID)
// ======================================================================

template <typename T>
[[kernel]] void na3d_bwd_grad_v_v2_kernel(
    device const float* attn [[buffer(0)]],
    device const T* grad_out [[buffer(1)]],
    device T* grad_v [[buffer(2)]],
    constant NA3DParamsV2& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {

  int vw = (int)gid.x;
  int vh = (int)gid.y;
  int z = (int)gid.z;
  if (vw >= p.IW || vh >= p.IH || z >= p.B * p.H * p.ID) return;

  int vd = z % p.ID;
  int bh = z / p.ID;
  int b = bh / p.H;
  int h = bh - b * p.H;

  int out_d = (p.ID + p.SD - 1) / p.SD;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;
  int K = p.K;
  int K3 = K * K * K;
  int nh = K / 2;
  int v_base = sf_base_3d(b, h, vd, vh, vw, p.ID, p.IH, p.IW, p.H, p.D);

  for (int d = 0; d < p.D; ++d) {
    float acc = 0.0f;

    for (int od = 0; od < out_d; ++od) {
      int qd = od * p.SD;
      if (qd >= p.ID) continue;

      int nd = p.CD ? 0 : natten_window_start_3d(qd, p.ID, K, nh, p.DD);
      int kd_offset = -1;
      if (p.CD) {
        int diff = qd - vd;
        if (diff >= 0 && diff % p.DD == 0) {
          int ki = (K - 1) - diff / p.DD;
          if (ki >= 0 && ki < K) kd_offset = ki;
        }
      } else {
        int diff = vd - nd;
        if (diff >= 0 && diff % p.DD == 0) {
          int ki = diff / p.DD;
          if (ki >= 0 && ki < K) kd_offset = ki;
        }
      }
      if (kd_offset < 0) continue;

      for (int oh = 0; oh < out_h; ++oh) {
        int qh = oh * p.SH;
        if (qh >= p.IH) continue;

        int nh_start = p.CH ? 0 : natten_window_start_3d(qh, p.IH, K, nh, p.DH);
        int kh_offset = -1;
        if (p.CH) {
          int diff = qh - vh;
          if (diff >= 0 && diff % p.DH == 0) {
            int ki = (K - 1) - diff / p.DH;
            if (ki >= 0 && ki < K) kh_offset = ki;
          }
        } else {
          int diff = vh - nh_start;
          if (diff >= 0 && diff % p.DH == 0) {
            int ki = diff / p.DH;
            if (ki >= 0 && ki < K) kh_offset = ki;
          }
        }
        if (kh_offset < 0) continue;

        for (int ow = 0; ow < out_w; ++ow) {
          int qw = ow * p.SW;
          if (qw >= p.IW) continue;

          int nw = p.CW ? 0 : natten_window_start_3d(qw, p.IW, K, nh, p.DW);
          int kw_offset = -1;
          if (p.CW) {
            int diff = qw - vw;
            if (diff >= 0 && diff % p.DW == 0) {
              int ki = (K - 1) - diff / p.DW;
              if (ki >= 0 && ki < K) kw_offset = ki;
            }
          } else {
            int diff = vw - nw;
            if (diff >= 0 && diff % p.DW == 0) {
              int ki = diff / p.DW;
              if (ki >= 0 && ki < K) kw_offset = ki;
            }
          }
          if (kw_offset < 0) continue;

          int kpos = (kd_offset * K + kh_offset) * K + kw_offset;
          int a_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * K3) + kpos;
          int go_idx = sf_base_3d(b, h, od, oh, ow, out_d, out_h, out_w, p.H, p.D) + d;
          acc += attn[a_idx] * (float)grad_out[go_idx];
        }
      }
    }
    grad_v[v_base + d] = (T)acc;
  }
}

// ======================================================================
// Template instantiations
// ======================================================================

// 3D Fused stored scalar
template [[host_name("na3d_fused_v2_stored_fp32")]]
[[kernel]] void na3d_fused_v2_stored_kernel<float>(
    device const float*, device const float*, device const float*, device float*,
    constant NA3DParamsV2&, uint3);
template [[host_name("na3d_fused_v2_stored_fp16")]]
[[kernel]] void na3d_fused_v2_stored_kernel<half>(
    device const half*, device const half*, device const half*, device half*,
    constant NA3DParamsV2&, uint3);
template [[host_name("na3d_fused_v2_stored_bf16")]]
[[kernel]] void na3d_fused_v2_stored_kernel<bfloat16_t>(
    device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*, device bfloat16_t*,
    constant NA3DParamsV2&, uint3);

// 3D Fused stored vec4
template [[host_name("na3d_fused_v2_stored_vec4_fp32")]]
[[kernel]] void na3d_fused_v2_stored_vec4_kernel<float, float4>(
    device const float*, device const float*, device const float*, device float*,
    constant NA3DParamsV2&, uint3);
template [[host_name("na3d_fused_v2_stored_vec4_fp16")]]
[[kernel]] void na3d_fused_v2_stored_vec4_kernel<half, half4>(
    device const half*, device const half*, device const half*, device half*,
    constant NA3DParamsV2&, uint3);
template [[host_name("na3d_fused_v2_stored_vec4_bf16")]]
[[kernel]] void na3d_fused_v2_stored_vec4_kernel<bfloat16_t, bfloat4>(
    device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*, device bfloat16_t*,
    constant NA3DParamsV2&, uint3);

// Backward attn scalar
template [[host_name("na3d_bwd_attn_v2_fp32")]]
[[kernel]] void na3d_bwd_attn_v2_kernel<float>(
    device const float*, device const float*, device const float*, device const float*,
    device float*, device float*, constant NA3DParamsV2&, uint3);
template [[host_name("na3d_bwd_attn_v2_fp16")]]
[[kernel]] void na3d_bwd_attn_v2_kernel<half>(
    device const half*, device const half*, device const half*, device const half*,
    device float*, device float*, constant NA3DParamsV2&, uint3);
template [[host_name("na3d_bwd_attn_v2_bf16")]]
[[kernel]] void na3d_bwd_attn_v2_kernel<bfloat16_t>(
    device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*,
    device float*, device float*, constant NA3DParamsV2&, uint3);

// Backward attn vec4
template [[host_name("na3d_bwd_attn_v2_vec4_fp32")]]
[[kernel]] void na3d_bwd_attn_v2_vec4_kernel<float, float4>(
    device const float*, device const float*, device const float*, device const float*,
    device float*, device float*, constant NA3DParamsV2&, uint3);
template [[host_name("na3d_bwd_attn_v2_vec4_fp16")]]
[[kernel]] void na3d_bwd_attn_v2_vec4_kernel<half, half4>(
    device const half*, device const half*, device const half*, device const half*,
    device float*, device float*, constant NA3DParamsV2&, uint3);
template [[host_name("na3d_bwd_attn_v2_vec4_bf16")]]
[[kernel]] void na3d_bwd_attn_v2_vec4_kernel<bfloat16_t, bfloat4>(
    device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*,
    device float*, device float*, constant NA3DParamsV2&, uint3);

// Backward grad_q
template [[host_name("na3d_bwd_grad_q_v2_fp32")]]
[[kernel]] void na3d_bwd_grad_q_v2_kernel<float>(
    device const float*, device const float*, device float*,
    constant NA3DParamsV2&, uint3);
template [[host_name("na3d_bwd_grad_q_v2_fp16")]]
[[kernel]] void na3d_bwd_grad_q_v2_kernel<half>(
    device const float*, device const float*, device half*,
    constant NA3DParamsV2&, uint3);
template [[host_name("na3d_bwd_grad_q_v2_bf16")]]
[[kernel]] void na3d_bwd_grad_q_v2_kernel<bfloat16_t>(
    device const float*, device const float*, device bfloat16_t*,
    constant NA3DParamsV2&, uint3);

// Backward grad_k
template [[host_name("na3d_bwd_grad_k_v2_fp32")]]
[[kernel]] void na3d_bwd_grad_k_v2_kernel<float>(
    device const float*, device const float*, device float*,
    constant NA3DParamsV2&, uint3);
template [[host_name("na3d_bwd_grad_k_v2_fp16")]]
[[kernel]] void na3d_bwd_grad_k_v2_kernel<half>(
    device const float*, device const float*, device half*,
    constant NA3DParamsV2&, uint3);
template [[host_name("na3d_bwd_grad_k_v2_bf16")]]
[[kernel]] void na3d_bwd_grad_k_v2_kernel<bfloat16_t>(
    device const float*, device const float*, device bfloat16_t*,
    constant NA3DParamsV2&, uint3);

// Backward grad_v
template [[host_name("na3d_bwd_grad_v_v2_fp32")]]
[[kernel]] void na3d_bwd_grad_v_v2_kernel<float>(
    device const float*, device const float*, device float*,
    constant NA3DParamsV2&, uint3);
template [[host_name("na3d_bwd_grad_v_v2_fp16")]]
[[kernel]] void na3d_bwd_grad_v_v2_kernel<half>(
    device const float*, device const half*, device half*,
    constant NA3DParamsV2&, uint3);
template [[host_name("na3d_bwd_grad_v_v2_bf16")]]
[[kernel]] void na3d_bwd_grad_v_v2_kernel<bfloat16_t>(
    device const float*, device const bfloat16_t*, device bfloat16_t*,
    constant NA3DParamsV2&, uint3);

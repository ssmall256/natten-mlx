// High-performance 2D neighborhood attention Metal kernels.
// Key optimizations over v1:
//   - Spatial-first layout [B, IH, IW, H, D] (matches MLX default — zero transpose)
//   - 3D thread grid (out_w, out_h, B*H) instead of flat 1D decomposition
//   - Stored logits + key_lin arrays to avoid redundant neighbor recomputation
//   - SIMD vec4 dot products via float4/half4/bfloat4

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Function constant for compile-time K specialization (loop unrolling)
constant int FC_K [[function_constant(0)]];
constant bool has_fc_k = is_function_constant_defined(FC_K);

// ---- Parameter struct ----
// Inputs in [B, IH, IW, H, D] layout (spatial-first, MLX default).
struct NA2DParams {
  int B;
  int H;
  int IH;
  int IW;
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

// ---- Window helpers ----

inline int natten_window_start(
    int index,
    int length,
    int kernel_size,
    int neighborhood_size,
    int dilation) {
  if (dilation <= 1) {
    return max(index - neighborhood_size, 0) +
        ((index + neighborhood_size >= length)
             ? (length - index - neighborhood_size - 1)
             : 0);
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

inline int natten_window_end(int start, int length, int kernel_size, int dilation) {
  int end = start + kernel_size * dilation;
  return min(end, length);
}

// ---- Spatial-first index helpers ----
// Layout: [B, IH, IW, H, D]  ->  linear = ((((b*IH+ih)*IW+iw)*H+h)*D+d)

inline int sf_base_2d(int b, int h, int ih, int iw, int IH, int IW, int H, int D) {
  return ((((b * IH + ih) * IW + iw) * H + h) * D);
}

// ======================================================================
// 2D Fused Forward — stored logits strategy, fp32 accumulation
// ======================================================================

template <typename T>
[[kernel]] void na2d_fused_v2_stored_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA2DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {

  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;

  int j_out = (int)gid.x;
  int i_out = (int)gid.y;
  int bh = (int)gid.z;
  if (j_out >= out_w || i_out >= out_h || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int i = i_out * p.SH;
  int j = j_out * p.SW;
  if (i >= p.IH || j >= p.IW) return;

  int K = has_fc_k ? FC_K : p.K;
  int nh = K / 2;
  int K2 = K * K;

  // Compute window bounds
  int ni = 0, ei = p.IH;
  int nj = 0, ej = p.IW;
  if (!p.CH) {
    ni = natten_window_start(i, p.IH, K, nh, p.DH);
    ei = natten_window_end(ni, p.IH, K, p.DH);
  }
  if (!p.CW) {
    nj = natten_window_start(j, p.IW, K, nh, p.DW);
    ej = natten_window_end(nj, p.IW, K, p.DW);
  }

  // Pass 1: compute scores, find max, store base addresses
  float logits[169];  // max K=13 -> 169 neighbors
  int key_base[169];  // precomputed linear base addresses
  float max_logit = -INFINITY;
  int n_idx = 0;

  int q_base = sf_base_2d(b, h, i, j, p.IH, p.IW, p.H, p.D);

  for (int ki = 0; ki < K; ++ki) {
    int key_i = p.CH ? (i - (K - 1 - ki) * p.DH) : (ni + ki * p.DH);
    bool valid_i = p.CH ? (key_i >= 0 && key_i <= i && key_i < p.IH)
                        : (key_i >= 0 && key_i < ei);
    for (int kj = 0; kj < K; ++kj) {
      int key_j = p.CW ? (j - (K - 1 - kj) * p.DW) : (nj + kj * p.DW);
      bool valid_j = p.CW ? (key_j >= 0 && key_j <= j && key_j < p.IW)
                          : (key_j >= 0 && key_j < ej);

      float score = -INFINITY;
      int kb = -1;
      if (valid_i && valid_j) {
        kb = sf_base_2d(b, h, key_i, key_j, p.IH, p.IW, p.H, p.D);
        float acc = 0.0f;
        for (int d = 0; d < p.D; ++d) {
          acc += (float)query[q_base + d] * (float)key[kb + d];
        }
        score = acc * p.SCALE;
      }

      logits[n_idx] = score;
      key_base[n_idx] = kb;
      max_logit = max(max_logit, score);
      n_idx++;
    }
  }

  // Pass 2: softmax
  float denom = 0.0f;
  for (int n = 0; n < K2; ++n) {
    logits[n] = exp(logits[n] - max_logit);
    denom += logits[n];
  }
  float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

  // Pass 3: weighted value aggregation (precomputed base addresses)
  int out_base = sf_base_2d(b, h, i_out, j_out, out_h, out_w, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    float acc = 0.0f;
    for (int n = 0; n < K2; ++n) {
      if (key_base[n] >= 0) {
        float w = logits[n] * inv_denom;
        acc += w * (float)value[key_base[n] + d];
      }
    }
    out[out_base + d] = (T)acc;
  }
}

// ======================================================================
// 2D Fused Forward — stored logits, vec4 SIMD path
// ======================================================================

template <typename T, typename T4>
[[kernel]] void na2d_fused_v2_stored_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA2DParams& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {

  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;

  int j_out = (int)gid.x;
  int i_out = (int)gid.y;
  int bh = (int)gid.z;
  if (j_out >= out_w || i_out >= out_h || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int i = i_out * p.SH;
  int j = j_out * p.SW;
  if (i >= p.IH || j >= p.IW) return;

  int K = has_fc_k ? FC_K : p.K;
  int nh = K / 2;
  int K2 = K * K;
  int dim4 = p.D / 4;

  // Window bounds
  int ni = 0, ei = p.IH;
  int nj = 0, ej = p.IW;
  if (!p.CH) {
    ni = natten_window_start(i, p.IH, K, nh, p.DH);
    ei = natten_window_end(ni, p.IH, K, p.DH);
  }
  if (!p.CW) {
    nj = natten_window_start(j, p.IW, K, nh, p.DW);
    ej = natten_window_end(nj, p.IW, K, p.DW);
  }

  float logits[169];
  int key_base[169];  // precomputed linear base addresses
  float max_logit = -INFINITY;
  int n_idx = 0;

  int q_base = sf_base_2d(b, h, i, j, p.IH, p.IW, p.H, p.D);

  for (int ki = 0; ki < K; ++ki) {
    int key_i = p.CH ? (i - (K - 1 - ki) * p.DH) : (ni + ki * p.DH);
    bool valid_i = p.CH ? (key_i >= 0 && key_i <= i && key_i < p.IH)
                        : (key_i >= 0 && key_i < ei);
    for (int kj = 0; kj < K; ++kj) {
      int key_j = p.CW ? (j - (K - 1 - kj) * p.DW) : (nj + kj * p.DW);
      bool valid_j = p.CW ? (key_j >= 0 && key_j <= j && key_j < p.IW)
                          : (key_j >= 0 && key_j < ej);

      float score = -INFINITY;
      int kb = -1;
      if (valid_i && valid_j) {
        kb = sf_base_2d(b, h, key_i, key_j, p.IH, p.IW, p.H, p.D);
        float acc = 0.0f;
        for (int d4 = 0; d4 < dim4; ++d4) {
          const device T4* q4 = reinterpret_cast<const device T4*>(query + q_base + d4 * 4);
          const device T4* k4 = reinterpret_cast<const device T4*>(key + kb + d4 * 4);
          acc += dot(float4(*q4), float4(*k4));
        }
        score = acc * p.SCALE;
      }

      logits[n_idx] = score;
      key_base[n_idx] = kb;
      max_logit = max(max_logit, score);
      n_idx++;
    }
  }

  // Softmax
  float denom = 0.0f;
  for (int n = 0; n < K2; ++n) {
    logits[n] = exp(logits[n] - max_logit);
    denom += logits[n];
  }
  float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

  // Weighted value aggregation (vec4, precomputed base addresses)
  int out_base = sf_base_2d(b, h, i_out, j_out, out_h, out_w, p.H, p.D);
  for (int d4 = 0; d4 < dim4; ++d4) {
    float4 acc = float4(0.0f);
    for (int n = 0; n < K2; ++n) {
      if (key_base[n] >= 0) {
        float w = logits[n] * inv_denom;
        const device T4* v4 = reinterpret_cast<const device T4*>(value + key_base[n] + d4 * 4);
        acc += w * float4(*v4);
      }
    }
    device T4* o4 = reinterpret_cast<device T4*>(out + out_base + d4 * 4);
    *o4 = T4(acc);
  }
}

// ======================================================================
// 2D Split QK Forward — spatial-first, 3D grid (K², out_w, out_h * B * H)
// ======================================================================

template <typename T>
[[kernel]] void na2d_qk_v2_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {

  int K = has_fc_k ? FC_K : p.K;
  int K2 = K * K;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;

  int kpos = (int)gid.x;
  int ow = (int)gid.y;
  int z = (int)gid.z;
  if (kpos >= K2 || ow >= out_w) return;

  int oh = z % out_h;
  int bh = z / out_h;
  if (bh >= p.B * p.H) return;
  int b = bh / p.H;
  int h = bh - b * p.H;

  int kh = kpos / K;
  int kw = kpos % K;

  int qi = oh * p.SH;
  int qj = ow * p.SW;

  int out_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * K2 + kpos);
  if (qi >= p.IH || qj >= p.IW) {
    out[out_idx] = (T)(-INFINITY);
    return;
  }

  int nh = K / 2;
  int h_start = p.CH ? 0 : natten_window_start(qi, p.IH, K, nh, p.DH);
  int w_start = p.CW ? 0 : natten_window_start(qj, p.IW, K, nh, p.DW);
  int ih = p.CH ? (qi - (K - 1 - kh) * p.DH) : (h_start + kh * p.DH);
  int iw = p.CW ? (qj - (K - 1 - kw) * p.DW) : (w_start + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    out[out_idx] = (T)(-INFINITY);
    return;
  }

  int q_base = sf_base_2d(b, h, qi, qj, p.IH, p.IW, p.H, p.D);
  int k_base = sf_base_2d(b, h, ih, iw, p.IH, p.IW, p.H, p.D);
  float acc = 0.0f;
  for (int d = 0; d < p.D; ++d) {
    acc += (float)query[q_base + d] * (float)key[k_base + d];
  }
  out[out_idx] = (T)(acc * p.SCALE);
}

// ======================================================================
// 2D Split QK Forward — vec4 variant, 3D grid (K², out_w, out_h * B * H)
// ======================================================================

template <typename T, typename T4>
[[kernel]] void na2d_qk_v2_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {

  int K = has_fc_k ? FC_K : p.K;
  int K2 = K * K;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;

  int kpos = (int)gid.x;
  int ow = (int)gid.y;
  int z = (int)gid.z;
  if (kpos >= K2 || ow >= out_w) return;

  int oh = z % out_h;
  int bh = z / out_h;
  if (bh >= p.B * p.H) return;
  int b = bh / p.H;
  int h = bh - b * p.H;

  int kh = kpos / K;
  int kw = kpos % K;

  int qi = oh * p.SH;
  int qj = ow * p.SW;

  int out_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * K2 + kpos);
  if (qi >= p.IH || qj >= p.IW) {
    out[out_idx] = (T)(-INFINITY);
    return;
  }

  int nh = K / 2;
  int h_start = p.CH ? 0 : natten_window_start(qi, p.IH, K, nh, p.DH);
  int w_start = p.CW ? 0 : natten_window_start(qj, p.IW, K, nh, p.DW);
  int ih = p.CH ? (qi - (K - 1 - kh) * p.DH) : (h_start + kh * p.DH);
  int iw = p.CW ? (qj - (K - 1 - kw) * p.DW) : (w_start + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    out[out_idx] = (T)(-INFINITY);
    return;
  }

  int q_base = sf_base_2d(b, h, qi, qj, p.IH, p.IW, p.H, p.D);
  int k_base = sf_base_2d(b, h, ih, iw, p.IH, p.IW, p.H, p.D);
  int dim4 = p.D / 4;
  float acc = 0.0f;
  for (int d4 = 0; d4 < dim4; ++d4) {
    const device T4* q4 = reinterpret_cast<const device T4*>(query + q_base + d4 * 4);
    const device T4* k4 = reinterpret_cast<const device T4*>(key + k_base + d4 * 4);
    acc += dot(float4(*q4), float4(*k4));
  }
  out[out_idx] = (T)(acc * p.SCALE);
}

// ======================================================================
// 2D Split AV Forward — spatial-first, 3D grid (D, out_w, out_h * B * H)
// ======================================================================

template <typename T>
[[kernel]] void na2d_av_v2_kernel(
    device const T* attn [[buffer(0)]],
    device const T* value [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {

  int K = has_fc_k ? FC_K : p.K;
  int K2 = K * K;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;

  int d = (int)gid.x;
  int ow = (int)gid.y;
  int z = (int)gid.z;
  if (d >= p.D || ow >= out_w) return;

  int oh = z % out_h;
  int bh = z / out_h;
  if (bh >= p.B * p.H) return;
  int b = bh / p.H;
  int h = bh - b * p.H;

  int qi = oh * p.SH;
  int qj = ow * p.SW;

  int out_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D + d);
  if (qi >= p.IH || qj >= p.IW) {
    out[out_idx] = (T)0.0f;
    return;
  }

  int nh = K / 2;
  int h_start = p.CH ? 0 : natten_window_start(qi, p.IH, K, nh, p.DH);
  int w_start = p.CW ? 0 : natten_window_start(qj, p.IW, K, nh, p.DW);
  int attn_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * K2);
  float acc = 0.0f;
  for (int kh = 0; kh < K; ++kh) {
    int ih = p.CH ? (qi - (K - 1 - kh) * p.DH) : (h_start + kh * p.DH);
    if (ih < 0 || ih >= p.IH) continue;
    for (int kw = 0; kw < K; ++kw) {
      int iw = p.CW ? (qj - (K - 1 - kw) * p.DW) : (w_start + kw * p.DW);
      if (iw < 0 || iw >= p.IW) continue;
      float w = (float)attn[attn_base + kh * K + kw];
      int v_idx = sf_base_2d(b, h, ih, iw, p.IH, p.IW, p.H, p.D) + d;
      acc += w * (float)value[v_idx];
    }
  }
  out[out_idx] = (T)acc;
}

// ======================================================================
// 2D Split AV Forward — vec4, 3D grid (D/4, out_w, out_h * B * H)
// ======================================================================

template <typename T, typename T4>
[[kernel]] void na2d_av_v2_vec4_kernel(
    device const T* attn [[buffer(0)]],
    device const T* value [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {

  int K = has_fc_k ? FC_K : p.K;
  int K2 = K * K;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;
  int d4_count = p.D / 4;

  int d4 = (int)gid.x;
  int ow = (int)gid.y;
  int z = (int)gid.z;
  if (d4 >= d4_count || ow >= out_w) return;

  int oh = z % out_h;
  int bh = z / out_h;
  if (bh >= p.B * p.H) return;
  int b = bh / p.H;
  int h = bh - b * p.H;

  int qi = oh * p.SH;
  int qj = ow * p.SW;
  int out_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D + d4 * 4);
  if (qi >= p.IH || qj >= p.IW) {
    device T4* o4 = reinterpret_cast<device T4*>(out + out_base);
    *o4 = T4(float4(0.0f));
    return;
  }

  int nh = K / 2;
  int h_start = p.CH ? 0 : natten_window_start(qi, p.IH, K, nh, p.DH);
  int w_start = p.CW ? 0 : natten_window_start(qj, p.IW, K, nh, p.DW);
  int attn_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * K2);
  float4 acc = float4(0.0f);
  for (int kh = 0; kh < K; ++kh) {
    int ih = p.CH ? (qi - (K - 1 - kh) * p.DH) : (h_start + kh * p.DH);
    if (ih < 0 || ih >= p.IH) continue;
    for (int kw = 0; kw < K; ++kw) {
      int iw = p.CW ? (qj - (K - 1 - kw) * p.DW) : (w_start + kw * p.DW);
      if (iw < 0 || iw >= p.IW) continue;
      float w = (float)attn[attn_base + kh * K + kw];
      int v_base = sf_base_2d(b, h, ih, iw, p.IH, p.IW, p.H, p.D) + d4 * 4;
      const device T4* v4 = reinterpret_cast<const device T4*>(value + v_base);
      acc += w * float4(*v4);
    }
  }
  device T4* o4 = reinterpret_cast<device T4*>(out + out_base);
  *o4 = T4(acc);
}

// ======================================================================
// 2D Backward — Fused attn recompute + grad_logits
// Thread grid: (out_w, out_h, B*H)
// Per thread: recompute QK softmax, compute grad_logits via fused softmax backward
// Outputs: attn [B, OH, OW, H, K²], grad_logits [same shape]
// ======================================================================

template <typename T>
[[kernel]] void na2d_bwd_attn_v2_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device const T* grad_out [[buffer(3)]],
    device float* attn_out [[buffer(4)]],
    device float* grad_logits_out [[buffer(5)]],
    constant NA2DParams& p [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {

  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;

  int j_out = (int)gid.x;
  int i_out = (int)gid.y;
  int bh = (int)gid.z;
  if (j_out >= out_w || i_out >= out_h || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int qi = i_out * p.SH;
  int qj = j_out * p.SW;

  int K = has_fc_k ? FC_K : p.K;
  int K2 = K * K;
  int nh = K / 2;
  int out_base = ((((b * out_h + i_out) * out_w + j_out) * p.H + h) * K2);

  if (qi >= p.IH || qj >= p.IW) {
    for (int n = 0; n < K2; ++n) {
      attn_out[out_base + n] = 0.0f;
      grad_logits_out[out_base + n] = 0.0f;
    }
    return;
  }

  // Window bounds
  int ni = 0, ei = p.IH;
  int nj = 0, ej = p.IW;
  if (!p.CH) {
    ni = natten_window_start(qi, p.IH, K, nh, p.DH);
    ei = natten_window_end(ni, p.IH, K, p.DH);
  }
  if (!p.CW) {
    nj = natten_window_start(qj, p.IW, K, nh, p.DW);
    ej = natten_window_end(nj, p.IW, K, p.DW);
  }

  int q_base = sf_base_2d(b, h, qi, qj, p.IH, p.IW, p.H, p.D);
  int go_base = sf_base_2d(b, h, i_out, j_out, out_h, out_w, p.H, p.D);

  // Pass 1: Compute QK scores and ga_vals (grad_out · V^T)
  float scores[169];
  float ga_vals[169];
  int key_valid[169];
  float max_score = -INFINITY;

  int n_idx = 0;
  for (int ki = 0; ki < K; ++ki) {
    int key_i = p.CH ? (qi - (K - 1 - ki) * p.DH) : (ni + ki * p.DH);
    bool valid_i = p.CH ? (key_i >= 0 && key_i <= qi && key_i < p.IH)
                        : (key_i >= 0 && key_i < ei);
    for (int kj = 0; kj < K; ++kj) {
      int key_j = p.CW ? (qj - (K - 1 - kj) * p.DW) : (nj + kj * p.DW);
      bool valid_j = p.CW ? (key_j >= 0 && key_j <= qj && key_j < p.IW)
                          : (key_j >= 0 && key_j < ej);

      float score = -INFINITY;
      float ga = 0.0f;
      int valid = 0;
      if (valid_i && valid_j) {
        int k_base = sf_base_2d(b, h, key_i, key_j, p.IH, p.IW, p.H, p.D);
        float s = 0.0f;
        for (int d = 0; d < p.D; ++d) {
          s += (float)query[q_base + d] * (float)key[k_base + d];
          ga += (float)grad_out[go_base + d] * (float)value[k_base + d];
        }
        score = s * p.SCALE;
        valid = 1;
      }
      scores[n_idx] = score;
      ga_vals[n_idx] = ga;
      key_valid[n_idx] = valid;
      max_score = max(max_score, score);
      n_idx++;
    }
  }

  // Pass 2: Softmax
  float denom = 0.0f;
  for (int n = 0; n < K2; ++n) {
    if (key_valid[n]) {
      scores[n] = exp(scores[n] - max_score);
      denom += scores[n];
    } else {
      scores[n] = 0.0f;
    }
  }
  float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

  // Fused softmax backward: grad_logits = attn * (grad_attn - sum(attn * grad_attn))
  float inner = 0.0f;
  for (int n = 0; n < K2; ++n) {
    float a = scores[n] * inv_denom;
    attn_out[out_base + n] = a;
    inner += a * ga_vals[n];
  }
  for (int n = 0; n < K2; ++n) {
    grad_logits_out[out_base + n] = attn_out[out_base + n] * (ga_vals[n] - inner);
  }
}

// Vec4 variant of bwd_attn
template <typename T, typename T4>
[[kernel]] void na2d_bwd_attn_v2_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device const T* grad_out [[buffer(3)]],
    device float* attn_out [[buffer(4)]],
    device float* grad_logits_out [[buffer(5)]],
    constant NA2DParams& p [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {

  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;

  int j_out = (int)gid.x;
  int i_out = (int)gid.y;
  int bh = (int)gid.z;
  if (j_out >= out_w || i_out >= out_h || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int qi = i_out * p.SH;
  int qj = j_out * p.SW;

  int K = has_fc_k ? FC_K : p.K;
  int K2 = K * K;
  int nh = K / 2;
  int dim4 = p.D / 4;
  int out_base = ((((b * out_h + i_out) * out_w + j_out) * p.H + h) * K2);

  if (qi >= p.IH || qj >= p.IW) {
    for (int n = 0; n < K2; ++n) {
      attn_out[out_base + n] = 0.0f;
      grad_logits_out[out_base + n] = 0.0f;
    }
    return;
  }

  int ni = 0, ei = p.IH;
  int nj = 0, ej = p.IW;
  if (!p.CH) {
    ni = natten_window_start(qi, p.IH, K, nh, p.DH);
    ei = natten_window_end(ni, p.IH, K, p.DH);
  }
  if (!p.CW) {
    nj = natten_window_start(qj, p.IW, K, nh, p.DW);
    ej = natten_window_end(nj, p.IW, K, p.DW);
  }

  int q_base = sf_base_2d(b, h, qi, qj, p.IH, p.IW, p.H, p.D);
  int go_base = sf_base_2d(b, h, i_out, j_out, out_h, out_w, p.H, p.D);

  float scores[169];
  float ga_vals[169];
  int key_valid[169];
  float max_score = -INFINITY;

  int n_idx = 0;
  for (int ki = 0; ki < K; ++ki) {
    int key_i = p.CH ? (qi - (K - 1 - ki) * p.DH) : (ni + ki * p.DH);
    bool valid_i = p.CH ? (key_i >= 0 && key_i <= qi && key_i < p.IH)
                        : (key_i >= 0 && key_i < ei);
    for (int kj = 0; kj < K; ++kj) {
      int key_j = p.CW ? (qj - (K - 1 - kj) * p.DW) : (nj + kj * p.DW);
      bool valid_j = p.CW ? (key_j >= 0 && key_j <= qj && key_j < p.IW)
                          : (key_j >= 0 && key_j < ej);

      float score = -INFINITY;
      float ga = 0.0f;
      int valid = 0;
      if (valid_i && valid_j) {
        int k_base = sf_base_2d(b, h, key_i, key_j, p.IH, p.IW, p.H, p.D);
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
        valid = 1;
      }
      scores[n_idx] = score;
      ga_vals[n_idx] = ga;
      key_valid[n_idx] = valid;
      max_score = max(max_score, score);
      n_idx++;
    }
  }

  float denom = 0.0f;
  for (int n = 0; n < K2; ++n) {
    if (key_valid[n]) {
      scores[n] = exp(scores[n] - max_score);
      denom += scores[n];
    } else {
      scores[n] = 0.0f;
    }
  }
  float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

  // Fused softmax backward: grad_logits = attn * (grad_attn - sum(attn * grad_attn))
  float inner = 0.0f;
  for (int n = 0; n < K2; ++n) {
    float a = scores[n] * inv_denom;
    attn_out[out_base + n] = a;
    inner += a * ga_vals[n];
  }
  for (int n = 0; n < K2; ++n) {
    grad_logits_out[out_base + n] = attn_out[out_base + n] * (ga_vals[n] - inner);
  }
}

// ======================================================================
// 2D Backward — grad_q (vec4, K-outer D-inner)
// Thread grid: (IW, IH, B*H) — one thread per spatial position
// ======================================================================

template <typename T>
[[kernel]] void na2d_bwd_grad_q_v2_kernel(
    device const float* grad_logits [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device T* grad_q [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {

  int iw = (int)gid.x;
  int ih = (int)gid.y;
  int bh = (int)gid.z;
  if (iw >= p.IW || ih >= p.IH || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;
  int K = has_fc_k ? FC_K : p.K;
  int K2 = K * K;
  int nh = K / 2;
  int dim4 = p.D / 4;
  int rem = p.D - dim4 * 4;

  int base = sf_base_2d(b, h, ih, iw, p.IH, p.IW, p.H, p.D);

  // Only on-stride positions have nonzero grad_q
  if ((ih % p.SH) != 0 || (iw % p.SW) != 0) {
    for (int d = 0; d < p.D; ++d) grad_q[base + d] = (T)0.0f;
    return;
  }
  int oh = ih / p.SH;
  int ow = iw / p.SW;
  if (oh >= out_h || ow >= out_w) {
    for (int d = 0; d < p.D; ++d) grad_q[base + d] = (T)0.0f;
    return;
  }

  // Window for this query position
  int ni = p.CH ? 0 : natten_window_start(ih, p.IH, K, nh, p.DH);
  int nj = p.CW ? 0 : natten_window_start(iw, p.IW, K, nh, p.DW);

  int gl_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * K2);

  // K-outer, D-inner with vec4 + scalar remainder
  float4 acc4[16] = {};
  float acc_rem[4] = {};
  for (int ki = 0; ki < K; ++ki) {
    int key_i = p.CH ? (ih - (K - 1 - ki) * p.DH) : (ni + ki * p.DH);
    if (key_i < 0 || key_i >= p.IH) continue;
    if (p.CH && key_i > ih) continue;
    for (int kj = 0; kj < K; ++kj) {
      int key_j = p.CW ? (iw - (K - 1 - kj) * p.DW) : (nj + kj * p.DW);
      if (key_j < 0 || key_j >= p.IW) continue;
      if (p.CW && key_j > iw) continue;
      float gl = grad_logits[gl_base + ki * K + kj];
      int k_start = sf_base_2d(b, h, key_i, key_j, p.IH, p.IW, p.H, p.D);
      for (int d4 = 0; d4 < dim4; ++d4) {
        acc4[d4] += gl * *(device const float4*)(key + k_start + d4 * 4);
      }
      for (int r = 0; r < rem; ++r) {
        acc_rem[r] += gl * key[k_start + dim4 * 4 + r];
      }
    }
  }
  float scale = p.SCALE;
  for (int d4 = 0; d4 < dim4; ++d4) {
    float4 v = acc4[d4] * scale;
    grad_q[base + d4 * 4 + 0] = (T)v.x;
    grad_q[base + d4 * 4 + 1] = (T)v.y;
    grad_q[base + d4 * 4 + 2] = (T)v.z;
    grad_q[base + d4 * 4 + 3] = (T)v.w;
  }
  for (int r = 0; r < rem; ++r) {
    grad_q[base + dim4 * 4 + r] = (T)(acc_rem[r] * scale);
  }
}

// ======================================================================
// 2D Backward — grad_k (vec4, contributor-outer D-inner)
// Thread grid: (IW, IH, B*H) — one thread per key position
// ======================================================================

template <typename T>
[[kernel]] void na2d_bwd_grad_k_v2_kernel(
    device const float* grad_logits [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device T* grad_k [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {

  int kw = (int)gid.x;
  int kh = (int)gid.y;
  int bh = (int)gid.z;
  if (kw >= p.IW || kh >= p.IH || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;
  int K = has_fc_k ? FC_K : p.K;
  int K2 = K * K;
  int nh = K / 2;
  int k_base = sf_base_2d(b, h, kh, kw, p.IH, p.IW, p.H, p.D);
  int dim4 = p.D / 4;
  int rem = p.D - dim4 * 4;

  // Precompute contributing queries (D-independent inverse mapping)
  int gl_indices[256];
  int q_bases[256];
  int contrib_count = 0;

  for (int oh = 0; oh < out_h; ++oh) {
    int qi = oh * p.SH;
    if (qi >= p.IH) continue;

    int ni_h = p.CH ? 0 : natten_window_start(qi, p.IH, K, nh, p.DH);
    int kh_offset = -1;
    if (p.CH) {
      int diff = qi - kh;
      if (diff >= 0 && diff % p.DH == 0) {
        int ki = (K - 1) - diff / p.DH;
        if (ki >= 0 && ki < K) kh_offset = ki;
      }
    } else {
      int diff = kh - ni_h;
      if (diff >= 0 && diff % p.DH == 0) {
        int ki = diff / p.DH;
        if (ki >= 0 && ki < K) kh_offset = ki;
      }
    }
    if (kh_offset < 0) continue;

    for (int ow = 0; ow < out_w && contrib_count < 256; ++ow) {
      int qj = ow * p.SW;
      if (qj >= p.IW) continue;

      int nj_w = p.CW ? 0 : natten_window_start(qj, p.IW, K, nh, p.DW);
      int kw_offset = -1;
      if (p.CW) {
        int diff = qj - kw;
        if (diff >= 0 && diff % p.DW == 0) {
          int kj = (K - 1) - diff / p.DW;
          if (kj >= 0 && kj < K) kw_offset = kj;
        }
      } else {
        int diff = kw - nj_w;
        if (diff >= 0 && diff % p.DW == 0) {
          int kj = diff / p.DW;
          if (kj >= 0 && kj < K) kw_offset = kj;
        }
      }
      if (kw_offset < 0) continue;

      int kpos = kh_offset * K + kw_offset;
      gl_indices[contrib_count] = ((((b * out_h + oh) * out_w + ow) * p.H + h) * K2) + kpos;
      q_bases[contrib_count] = sf_base_2d(b, h, qi, qj, p.IH, p.IW, p.H, p.D);
      contrib_count++;
    }
  }

  // Contributor-outer, D-inner with vec4 + scalar remainder
  float4 acc4[16] = {};
  float acc_rem[4] = {};
  for (int c = 0; c < contrib_count; ++c) {
    float gl = grad_logits[gl_indices[c]];
    int qb = q_bases[c];
    for (int d4 = 0; d4 < dim4; ++d4) {
      acc4[d4] += gl * *(device const float4*)(query + qb + d4 * 4);
    }
    for (int r = 0; r < rem; ++r) {
      acc_rem[r] += gl * query[qb + dim4 * 4 + r];
    }
  }
  float scale = p.SCALE;
  for (int d4 = 0; d4 < dim4; ++d4) {
    float4 v = acc4[d4] * scale;
    grad_k[k_base + d4 * 4 + 0] = (T)v.x;
    grad_k[k_base + d4 * 4 + 1] = (T)v.y;
    grad_k[k_base + d4 * 4 + 2] = (T)v.z;
    grad_k[k_base + d4 * 4 + 3] = (T)v.w;
  }
  for (int r = 0; r < rem; ++r) {
    grad_k[k_base + dim4 * 4 + r] = (T)(acc_rem[r] * scale);
  }
}

// ======================================================================
// 2D Backward — grad_v (vec4, contributor-outer D-inner)
// Thread grid: (IW, IH, B*H) — one thread per value position
// ======================================================================

template <typename T>
[[kernel]] void na2d_bwd_grad_v_v2_kernel(
    device const float* attn [[buffer(0)]],
    device const T* grad_out [[buffer(1)]],
    device T* grad_v [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {

  int vw = (int)gid.x;
  int vh = (int)gid.y;
  int bh = (int)gid.z;
  if (vw >= p.IW || vh >= p.IH || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int out_h = (p.IH + p.SH - 1) / p.SH;
  int out_w = (p.IW + p.SW - 1) / p.SW;
  int K = has_fc_k ? FC_K : p.K;
  int K2 = K * K;
  int nh = K / 2;
  int v_base = sf_base_2d(b, h, vh, vw, p.IH, p.IW, p.H, p.D);
  int dim4 = p.D / 4;
  int rem = p.D - dim4 * 4;

  // Precompute contributing queries (D-independent inverse mapping)
  int a_indices[256];
  int go_bases[256];
  int contrib_count = 0;

  for (int oh = 0; oh < out_h; ++oh) {
    int qi = oh * p.SH;
    if (qi >= p.IH) continue;

    int ni_h = p.CH ? 0 : natten_window_start(qi, p.IH, K, nh, p.DH);
    int kh_offset = -1;
    if (p.CH) {
      int diff = qi - vh;
      if (diff >= 0 && diff % p.DH == 0) {
        int ki = (K - 1) - diff / p.DH;
        if (ki >= 0 && ki < K) kh_offset = ki;
      }
    } else {
      int diff = vh - ni_h;
      if (diff >= 0 && diff % p.DH == 0) {
        int ki = diff / p.DH;
        if (ki >= 0 && ki < K) kh_offset = ki;
      }
    }
    if (kh_offset < 0) continue;

    for (int ow = 0; ow < out_w && contrib_count < 256; ++ow) {
      int qj = ow * p.SW;
      if (qj >= p.IW) continue;

      int nj_w = p.CW ? 0 : natten_window_start(qj, p.IW, K, nh, p.DW);
      int kw_offset = -1;
      if (p.CW) {
        int diff = qj - vw;
        if (diff >= 0 && diff % p.DW == 0) {
          int kj = (K - 1) - diff / p.DW;
          if (kj >= 0 && kj < K) kw_offset = kj;
        }
      } else {
        int diff = vw - nj_w;
        if (diff >= 0 && diff % p.DW == 0) {
          int kj = diff / p.DW;
          if (kj >= 0 && kj < K) kw_offset = kj;
        }
      }
      if (kw_offset < 0) continue;

      int kpos = kh_offset * K + kw_offset;
      a_indices[contrib_count] = ((((b * out_h + oh) * out_w + ow) * p.H + h) * K2) + kpos;
      go_bases[contrib_count] = sf_base_2d(b, h, oh, ow, out_h, out_w, p.H, p.D);
      contrib_count++;
    }
  }

  // Contributor-outer, D-inner with vec4 + scalar remainder
  float4 acc4[16] = {};
  float acc_rem[4] = {};
  for (int c = 0; c < contrib_count; ++c) {
    float a = attn[a_indices[c]];
    int gob = go_bases[c];
    for (int d4 = 0; d4 < dim4; ++d4) {
      acc4[d4] += a * float4(*(device const float4*)(grad_out + gob + d4 * 4));
    }
    for (int r = 0; r < rem; ++r) {
      acc_rem[r] += a * float(grad_out[gob + dim4 * 4 + r]);
    }
  }
  for (int d4 = 0; d4 < dim4; ++d4) {
    float4 v = acc4[d4];
    grad_v[v_base + d4 * 4 + 0] = (T)v.x;
    grad_v[v_base + d4 * 4 + 1] = (T)v.y;
    grad_v[v_base + d4 * 4 + 2] = (T)v.z;
    grad_v[v_base + d4 * 4 + 3] = (T)v.w;
  }
  for (int r = 0; r < rem; ++r) {
    grad_v[v_base + dim4 * 4 + r] = (T)acc_rem[r];
  }
}

// ======================================================================
// Template instantiations
// ======================================================================

// Fused stored scalar
template [[host_name("na2d_fused_v2_stored_fp32")]]
[[kernel]] void na2d_fused_v2_stored_kernel<float>(
    device const float*, device const float*, device const float*, device float*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_fused_v2_stored_fp16")]]
[[kernel]] void na2d_fused_v2_stored_kernel<half>(
    device const half*, device const half*, device const half*, device half*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_fused_v2_stored_bf16")]]
[[kernel]] void na2d_fused_v2_stored_kernel<bfloat16_t>(
    device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*, device bfloat16_t*,
    constant NA2DParams&, uint3);

// Fused stored vec4
template [[host_name("na2d_fused_v2_stored_vec4_fp32")]]
[[kernel]] void na2d_fused_v2_stored_vec4_kernel<float, float4>(
    device const float*, device const float*, device const float*, device float*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_fused_v2_stored_vec4_fp16")]]
[[kernel]] void na2d_fused_v2_stored_vec4_kernel<half, half4>(
    device const half*, device const half*, device const half*, device half*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_fused_v2_stored_vec4_bf16")]]
[[kernel]] void na2d_fused_v2_stored_vec4_kernel<bfloat16_t, bfloat4>(
    device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*, device bfloat16_t*,
    constant NA2DParams&, uint3);

// Split QK scalar
template [[host_name("na2d_qk_v2_fp32")]]
[[kernel]] void na2d_qk_v2_kernel<float>(
    device const float*, device const float*, device float*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_qk_v2_fp16")]]
[[kernel]] void na2d_qk_v2_kernel<half>(
    device const half*, device const half*, device half*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_qk_v2_bf16")]]
[[kernel]] void na2d_qk_v2_kernel<bfloat16_t>(
    device const bfloat16_t*, device const bfloat16_t*, device bfloat16_t*,
    constant NA2DParams&, uint3);

// Split QK vec4
template [[host_name("na2d_qk_v2_vec4_fp32")]]
[[kernel]] void na2d_qk_v2_vec4_kernel<float, float4>(
    device const float*, device const float*, device float*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_qk_v2_vec4_fp16")]]
[[kernel]] void na2d_qk_v2_vec4_kernel<half, half4>(
    device const half*, device const half*, device half*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_qk_v2_vec4_bf16")]]
[[kernel]] void na2d_qk_v2_vec4_kernel<bfloat16_t, bfloat4>(
    device const bfloat16_t*, device const bfloat16_t*, device bfloat16_t*,
    constant NA2DParams&, uint3);

// Split AV scalar
template [[host_name("na2d_av_v2_fp32")]]
[[kernel]] void na2d_av_v2_kernel<float>(
    device const float*, device const float*, device float*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_av_v2_fp16")]]
[[kernel]] void na2d_av_v2_kernel<half>(
    device const half*, device const half*, device half*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_av_v2_bf16")]]
[[kernel]] void na2d_av_v2_kernel<bfloat16_t>(
    device const bfloat16_t*, device const bfloat16_t*, device bfloat16_t*,
    constant NA2DParams&, uint3);

// Split AV vec4
template [[host_name("na2d_av_v2_vec4_fp32")]]
[[kernel]] void na2d_av_v2_vec4_kernel<float, float4>(
    device const float*, device const float*, device float*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_av_v2_vec4_fp16")]]
[[kernel]] void na2d_av_v2_vec4_kernel<half, half4>(
    device const half*, device const half*, device half*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_av_v2_vec4_bf16")]]
[[kernel]] void na2d_av_v2_vec4_kernel<bfloat16_t, bfloat4>(
    device const bfloat16_t*, device const bfloat16_t*, device bfloat16_t*,
    constant NA2DParams&, uint3);

// Backward attn scalar
template [[host_name("na2d_bwd_attn_v2_fp32")]]
[[kernel]] void na2d_bwd_attn_v2_kernel<float>(
    device const float*, device const float*, device const float*, device const float*,
    device float*, device float*, constant NA2DParams&, uint3);
template [[host_name("na2d_bwd_attn_v2_fp16")]]
[[kernel]] void na2d_bwd_attn_v2_kernel<half>(
    device const half*, device const half*, device const half*, device const half*,
    device float*, device float*, constant NA2DParams&, uint3);
template [[host_name("na2d_bwd_attn_v2_bf16")]]
[[kernel]] void na2d_bwd_attn_v2_kernel<bfloat16_t>(
    device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*,
    device float*, device float*, constant NA2DParams&, uint3);

// Backward attn vec4
template [[host_name("na2d_bwd_attn_v2_vec4_fp32")]]
[[kernel]] void na2d_bwd_attn_v2_vec4_kernel<float, float4>(
    device const float*, device const float*, device const float*, device const float*,
    device float*, device float*, constant NA2DParams&, uint3);
template [[host_name("na2d_bwd_attn_v2_vec4_fp16")]]
[[kernel]] void na2d_bwd_attn_v2_vec4_kernel<half, half4>(
    device const half*, device const half*, device const half*, device const half*,
    device float*, device float*, constant NA2DParams&, uint3);
template [[host_name("na2d_bwd_attn_v2_vec4_bf16")]]
[[kernel]] void na2d_bwd_attn_v2_vec4_kernel<bfloat16_t, bfloat4>(
    device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*,
    device float*, device float*, constant NA2DParams&, uint3);

// Backward grad_q
template [[host_name("na2d_bwd_grad_q_v2_fp32")]]
[[kernel]] void na2d_bwd_grad_q_v2_kernel<float>(
    device const float*, device const float*, device float*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_bwd_grad_q_v2_fp16")]]
[[kernel]] void na2d_bwd_grad_q_v2_kernel<half>(
    device const float*, device const float*, device half*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_bwd_grad_q_v2_bf16")]]
[[kernel]] void na2d_bwd_grad_q_v2_kernel<bfloat16_t>(
    device const float*, device const float*, device bfloat16_t*,
    constant NA2DParams&, uint3);

// Backward grad_k
template [[host_name("na2d_bwd_grad_k_v2_fp32")]]
[[kernel]] void na2d_bwd_grad_k_v2_kernel<float>(
    device const float*, device const float*, device float*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_bwd_grad_k_v2_fp16")]]
[[kernel]] void na2d_bwd_grad_k_v2_kernel<half>(
    device const float*, device const float*, device half*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_bwd_grad_k_v2_bf16")]]
[[kernel]] void na2d_bwd_grad_k_v2_kernel<bfloat16_t>(
    device const float*, device const float*, device bfloat16_t*,
    constant NA2DParams&, uint3);

// Backward grad_v
template [[host_name("na2d_bwd_grad_v_v2_fp32")]]
[[kernel]] void na2d_bwd_grad_v_v2_kernel<float>(
    device const float*, device const float*, device float*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_bwd_grad_v_v2_fp16")]]
[[kernel]] void na2d_bwd_grad_v_v2_kernel<half>(
    device const float*, device const half*, device half*,
    constant NA2DParams&, uint3);
template [[host_name("na2d_bwd_grad_v_v2_bf16")]]
[[kernel]] void na2d_bwd_grad_v_v2_kernel<bfloat16_t>(
    device const float*, device const bfloat16_t*, device bfloat16_t*,
    constant NA2DParams&, uint3);

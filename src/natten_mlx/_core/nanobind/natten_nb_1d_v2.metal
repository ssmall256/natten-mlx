// High-performance 1D neighborhood attention Metal kernels.
// Key optimizations over v1:
//   - Spatial-first layout [B, L, H, D] (matches MLX default — zero transpose)
//   - Stored logits + key indices to avoid 3x dot product recomputation
//   - SIMD vec4 dot products via float4/half4/bfloat4
//   - 2D grid (out_len, B*H)

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

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

// Window helpers
inline int natten_window_start_1d(
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

// Spatial-first: [B, L, H, D] -> linear = ((b*L+l)*H+h)*D+d
inline int sf_base_1d(int b, int h, int l, int L, int H, int D) {
  return (((b * L + l) * H + h) * D);
}

// ======================================================================
// 1D Fused Forward — stored logits, scalar
// ======================================================================

template <typename T>
[[kernel]] void na1d_fused_v2_stored_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA1DParamsV2& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {

  int out_len = (p.L + p.S - 1) / p.S;
  int oq = (int)gid.x;
  int bh = (int)gid.y;
  if (oq >= out_len || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int qidx = oq * p.S;
  if (qidx >= p.L) return;

  int K = p.K;
  int nh = K / 2;
  int q_base = sf_base_1d(b, h, qidx, p.L, p.H, p.D);

  // Pass 1: compute scores, store logits and key positions
  float logits[63];  // max K=63 for 1D
  int key_pos[63];
  float max_logit = -INFINITY;
  int n_idx = 0;

  int ni = p.CAUSAL ? 0 : natten_window_start_1d(qidx, p.L, K, nh, p.DIL);

  for (int kk = 0; kk < K; ++kk) {
    int kidx = p.CAUSAL
        ? (qidx - (K - 1 - kk) * p.DIL)
        : (ni + kk * p.DIL);
    bool valid = p.CAUSAL
        ? (kidx >= 0 && kidx <= qidx && kidx < p.L)
        : (kidx >= 0 && kidx < p.L);

    float score = -INFINITY;
    int kpos = -1;
    if (valid) {
      kpos = kidx;
      int k_base = sf_base_1d(b, h, kidx, p.L, p.H, p.D);
      float acc = 0.0f;
      for (int d = 0; d < p.D; ++d) {
        acc += (float)query[q_base + d] * (float)key[k_base + d];
      }
      score = acc * p.SCALE;
    }
    logits[n_idx] = score;
    key_pos[n_idx] = kpos;
    max_logit = max(max_logit, score);
    n_idx++;
  }

  // Pass 2: softmax
  float denom = 0.0f;
  for (int n = 0; n < K; ++n) {
    logits[n] = exp(logits[n] - max_logit);
    denom += logits[n];
  }
  float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

  // Pass 3: weighted value aggregation
  int out_base = sf_base_1d(b, h, oq, out_len, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    float acc = 0.0f;
    for (int n = 0; n < K; ++n) {
      if (key_pos[n] >= 0) {
        float w = logits[n] * inv_denom;
        int v_idx = sf_base_1d(b, h, key_pos[n], p.L, p.H, p.D) + d;
        acc += w * (float)value[v_idx];
      }
    }
    out[out_base + d] = (T)acc;
  }
}

// ======================================================================
// 1D Fused Forward — stored logits, vec4
// ======================================================================

template <typename T, typename T4>
[[kernel]] void na1d_fused_v2_stored_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant NA1DParamsV2& p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {

  int out_len = (p.L + p.S - 1) / p.S;
  int oq = (int)gid.x;
  int bh = (int)gid.y;
  if (oq >= out_len || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int qidx = oq * p.S;
  if (qidx >= p.L) return;

  int K = p.K;
  int nh = K / 2;
  int dim4 = p.D / 4;
  int q_base = sf_base_1d(b, h, qidx, p.L, p.H, p.D);

  float logits[63];
  int key_pos[63];
  float max_logit = -INFINITY;
  int n_idx = 0;

  int ni = p.CAUSAL ? 0 : natten_window_start_1d(qidx, p.L, K, nh, p.DIL);

  for (int kk = 0; kk < K; ++kk) {
    int kidx = p.CAUSAL
        ? (qidx - (K - 1 - kk) * p.DIL)
        : (ni + kk * p.DIL);
    bool valid = p.CAUSAL
        ? (kidx >= 0 && kidx <= qidx && kidx < p.L)
        : (kidx >= 0 && kidx < p.L);

    float score = -INFINITY;
    int kpos = -1;
    if (valid) {
      kpos = kidx;
      int k_base = sf_base_1d(b, h, kidx, p.L, p.H, p.D);
      float acc = 0.0f;
      for (int d4 = 0; d4 < dim4; ++d4) {
        const device T4* q4 = reinterpret_cast<const device T4*>(query + q_base + d4 * 4);
        const device T4* k4 = reinterpret_cast<const device T4*>(key + k_base + d4 * 4);
        acc += dot(float4(*q4), float4(*k4));
      }
      score = acc * p.SCALE;
    }
    logits[n_idx] = score;
    key_pos[n_idx] = kpos;
    max_logit = max(max_logit, score);
    n_idx++;
  }

  // Softmax
  float denom = 0.0f;
  for (int n = 0; n < K; ++n) {
    logits[n] = exp(logits[n] - max_logit);
    denom += logits[n];
  }
  float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

  // Weighted value aggregation (vec4)
  int out_base = sf_base_1d(b, h, oq, out_len, p.H, p.D);
  for (int d4 = 0; d4 < dim4; ++d4) {
    float4 acc = float4(0.0f);
    for (int n = 0; n < K; ++n) {
      if (key_pos[n] >= 0) {
        float w = logits[n] * inv_denom;
        int v_base = sf_base_1d(b, h, key_pos[n], p.L, p.H, p.D) + d4 * 4;
        const device T4* v4 = reinterpret_cast<const device T4*>(value + v_base);
        acc += w * float4(*v4);
      }
    }
    device T4* o4 = reinterpret_cast<device T4*>(out + out_base + d4 * 4);
    *o4 = T4(acc);
  }
}

// ======================================================================
// 1D Backward — Fused attn recompute + grad_attn
// Thread grid: (out_len, B*H)
// ======================================================================

template <typename T>
[[kernel]] void na1d_bwd_attn_v2_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device const T* grad_out [[buffer(3)]],
    device float* attn_out [[buffer(4)]],
    device float* grad_attn_out [[buffer(5)]],
    constant NA1DParamsV2& p [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {

  int out_len = (p.L + p.S - 1) / p.S;
  int oq = (int)gid.x;
  int bh = (int)gid.y;
  if (oq >= out_len || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int qidx = oq * p.S;

  int K = p.K;
  int nh = K / 2;
  int out_base = (((b * out_len + oq) * p.H + h) * K);

  if (qidx >= p.L) {
    for (int n = 0; n < K; ++n) {
      attn_out[out_base + n] = 0.0f;
      grad_attn_out[out_base + n] = 0.0f;
    }
    return;
  }

  int q_base = sf_base_1d(b, h, qidx, p.L, p.H, p.D);
  int go_base = sf_base_1d(b, h, oq, out_len, p.H, p.D);

  int ni = p.CAUSAL ? 0 : natten_window_start_1d(qidx, p.L, K, nh, p.DIL);

  float scores[63];
  float ga_vals[63];
  int key_valid[63];
  float max_score = -INFINITY;

  for (int kk = 0; kk < K; ++kk) {
    int kidx = p.CAUSAL
        ? (qidx - (K - 1 - kk) * p.DIL)
        : (ni + kk * p.DIL);
    bool valid = p.CAUSAL
        ? (kidx >= 0 && kidx <= qidx && kidx < p.L)
        : (kidx >= 0 && kidx < p.L);

    float score = -INFINITY;
    float ga = 0.0f;
    int v_flag = 0;
    if (valid) {
      int k_base = sf_base_1d(b, h, kidx, p.L, p.H, p.D);
      float s = 0.0f;
      for (int d = 0; d < p.D; ++d) {
        s += (float)query[q_base + d] * (float)key[k_base + d];
        ga += (float)grad_out[go_base + d] * (float)value[k_base + d];
      }
      score = s * p.SCALE;
      v_flag = 1;
    }
    scores[kk] = score;
    ga_vals[kk] = ga;
    key_valid[kk] = v_flag;
    max_score = max(max_score, score);
  }

  // Softmax
  float denom = 0.0f;
  for (int n = 0; n < K; ++n) {
    if (key_valid[n]) {
      scores[n] = exp(scores[n] - max_score);
      denom += scores[n];
    } else {
      scores[n] = 0.0f;
    }
  }
  float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

  for (int n = 0; n < K; ++n) {
    attn_out[out_base + n] = scores[n] * inv_denom;
    grad_attn_out[out_base + n] = ga_vals[n];
  }
}

// Vec4 variant
template <typename T, typename T4>
[[kernel]] void na1d_bwd_attn_v2_vec4_kernel(
    device const T* query [[buffer(0)]],
    device const T* key [[buffer(1)]],
    device const T* value [[buffer(2)]],
    device const T* grad_out [[buffer(3)]],
    device float* attn_out [[buffer(4)]],
    device float* grad_attn_out [[buffer(5)]],
    constant NA1DParamsV2& p [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {

  int out_len = (p.L + p.S - 1) / p.S;
  int oq = (int)gid.x;
  int bh = (int)gid.y;
  if (oq >= out_len || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int qidx = oq * p.S;

  int K = p.K;
  int nh = K / 2;
  int dim4 = p.D / 4;
  int out_base = (((b * out_len + oq) * p.H + h) * K);

  if (qidx >= p.L) {
    for (int n = 0; n < K; ++n) {
      attn_out[out_base + n] = 0.0f;
      grad_attn_out[out_base + n] = 0.0f;
    }
    return;
  }

  int q_base = sf_base_1d(b, h, qidx, p.L, p.H, p.D);
  int go_base = sf_base_1d(b, h, oq, out_len, p.H, p.D);

  int ni = p.CAUSAL ? 0 : natten_window_start_1d(qidx, p.L, K, nh, p.DIL);

  float scores[63];
  float ga_vals[63];
  int key_valid[63];
  float max_score = -INFINITY;

  for (int kk = 0; kk < K; ++kk) {
    int kidx = p.CAUSAL
        ? (qidx - (K - 1 - kk) * p.DIL)
        : (ni + kk * p.DIL);
    bool valid = p.CAUSAL
        ? (kidx >= 0 && kidx <= qidx && kidx < p.L)
        : (kidx >= 0 && kidx < p.L);

    float score = -INFINITY;
    float ga = 0.0f;
    int v_flag = 0;
    if (valid) {
      int k_base = sf_base_1d(b, h, kidx, p.L, p.H, p.D);
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
    scores[kk] = score;
    ga_vals[kk] = ga;
    key_valid[kk] = v_flag;
    max_score = max(max_score, score);
  }

  float denom = 0.0f;
  for (int n = 0; n < K; ++n) {
    if (key_valid[n]) {
      scores[n] = exp(scores[n] - max_score);
      denom += scores[n];
    } else {
      scores[n] = 0.0f;
    }
  }
  float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;

  for (int n = 0; n < K; ++n) {
    attn_out[out_base + n] = scores[n] * inv_denom;
    grad_attn_out[out_base + n] = ga_vals[n];
  }
}

// ======================================================================
// 1D Backward — grad_q
// Thread grid: (L, B*H) — one thread per spatial position
// ======================================================================

template <typename T>
[[kernel]] void na1d_bwd_grad_q_v2_kernel(
    device const float* grad_logits [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device T* grad_q [[buffer(2)]],
    constant NA1DParamsV2& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {

  int il = (int)gid.x;
  int bh = (int)gid.y;
  if (il >= p.L || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int out_len = (p.L + p.S - 1) / p.S;
  int K = p.K;
  int nh = K / 2;

  int base = sf_base_1d(b, h, il, p.L, p.H, p.D);

  // Only on-stride positions have nonzero grad_q
  if ((il % p.S) != 0) {
    for (int d = 0; d < p.D; ++d) grad_q[base + d] = (T)0.0f;
    return;
  }
  int ol = il / p.S;
  if (ol >= out_len) {
    for (int d = 0; d < p.D; ++d) grad_q[base + d] = (T)0.0f;
    return;
  }

  int ni = p.CAUSAL ? 0 : natten_window_start_1d(il, p.L, K, nh, p.DIL);
  int gl_base = ((b * out_len + ol) * p.H + h) * K;

  for (int d = 0; d < p.D; ++d) {
    float acc = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
      int kidx = p.CAUSAL ? (il - (K - 1 - kk) * p.DIL) : (ni + kk * p.DIL);
      if (kidx < 0 || kidx >= p.L) continue;
      if (p.CAUSAL && kidx > il) continue;
      float gl = grad_logits[gl_base + kk];
      int k_idx = sf_base_1d(b, h, kidx, p.L, p.H, p.D) + d;
      acc += gl * key[k_idx];
    }
    grad_q[base + d] = (T)(acc * p.SCALE);
  }
}

// ======================================================================
// 1D Backward — grad_k (direct nonatomic)
// Thread grid: (L, B*H) — one thread per key position
// ======================================================================

template <typename T>
[[kernel]] void na1d_bwd_grad_k_v2_kernel(
    device const float* grad_logits [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device T* grad_k [[buffer(2)]],
    constant NA1DParamsV2& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {

  int kl = (int)gid.x;
  int bh = (int)gid.y;
  if (kl >= p.L || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int out_len = (p.L + p.S - 1) / p.S;
  int K = p.K;
  int nh = K / 2;
  int k_base = sf_base_1d(b, h, kl, p.L, p.H, p.D);

  for (int d = 0; d < p.D; ++d) {
    float acc = 0.0f;

    for (int ol = 0; ol < out_len; ++ol) {
      int qi = ol * p.S;
      if (qi >= p.L) continue;

      int ni = p.CAUSAL ? 0 : natten_window_start_1d(qi, p.L, K, nh, p.DIL);
      int k_offset = -1;
      if (p.CAUSAL) {
        int diff = qi - kl;
        if (diff >= 0 && diff % p.DIL == 0) {
          int ki = (K - 1) - diff / p.DIL;
          if (ki >= 0 && ki < K) k_offset = ki;
        }
      } else {
        int diff = kl - ni;
        if (diff >= 0 && diff % p.DIL == 0) {
          int ki = diff / p.DIL;
          if (ki >= 0 && ki < K) k_offset = ki;
        }
      }
      if (k_offset < 0) continue;

      int gl_idx = ((b * out_len + ol) * p.H + h) * K + k_offset;
      int q_idx = sf_base_1d(b, h, qi, p.L, p.H, p.D) + d;
      acc += grad_logits[gl_idx] * query[q_idx];
    }
    grad_k[k_base + d] = (T)(acc * p.SCALE);
  }
}

// ======================================================================
// 1D Backward — grad_v (direct nonatomic)
// Thread grid: (L, B*H) — one thread per value position
// ======================================================================

template <typename T>
[[kernel]] void na1d_bwd_grad_v_v2_kernel(
    device const float* attn [[buffer(0)]],
    device const T* grad_out [[buffer(1)]],
    device T* grad_v [[buffer(2)]],
    constant NA1DParamsV2& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {

  int vl = (int)gid.x;
  int bh = (int)gid.y;
  if (vl >= p.L || bh >= p.B * p.H) return;

  int b = bh / p.H;
  int h = bh - b * p.H;
  int out_len = (p.L + p.S - 1) / p.S;
  int K = p.K;
  int nh = K / 2;
  int v_base = sf_base_1d(b, h, vl, p.L, p.H, p.D);

  for (int d = 0; d < p.D; ++d) {
    float acc = 0.0f;

    for (int ol = 0; ol < out_len; ++ol) {
      int qi = ol * p.S;
      if (qi >= p.L) continue;

      int ni = p.CAUSAL ? 0 : natten_window_start_1d(qi, p.L, K, nh, p.DIL);
      int k_offset = -1;
      if (p.CAUSAL) {
        int diff = qi - vl;
        if (diff >= 0 && diff % p.DIL == 0) {
          int ki = (K - 1) - diff / p.DIL;
          if (ki >= 0 && ki < K) k_offset = ki;
        }
      } else {
        int diff = vl - ni;
        if (diff >= 0 && diff % p.DIL == 0) {
          int ki = diff / p.DIL;
          if (ki >= 0 && ki < K) k_offset = ki;
        }
      }
      if (k_offset < 0) continue;

      int a_idx = ((b * out_len + ol) * p.H + h) * K + k_offset;
      int go_idx = sf_base_1d(b, h, ol, out_len, p.H, p.D) + d;
      acc += attn[a_idx] * (float)grad_out[go_idx];
    }
    grad_v[v_base + d] = (T)acc;
  }
}

// ======================================================================
// Template instantiations
// ======================================================================

// Fused stored scalar
template [[host_name("na1d_fused_v2_stored_fp32")]]
[[kernel]] void na1d_fused_v2_stored_kernel<float>(
    device const float*, device const float*, device const float*, device float*,
    constant NA1DParamsV2&, uint3);
template [[host_name("na1d_fused_v2_stored_fp16")]]
[[kernel]] void na1d_fused_v2_stored_kernel<half>(
    device const half*, device const half*, device const half*, device half*,
    constant NA1DParamsV2&, uint3);
template [[host_name("na1d_fused_v2_stored_bf16")]]
[[kernel]] void na1d_fused_v2_stored_kernel<bfloat16_t>(
    device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*, device bfloat16_t*,
    constant NA1DParamsV2&, uint3);

// Fused stored vec4
template [[host_name("na1d_fused_v2_stored_vec4_fp32")]]
[[kernel]] void na1d_fused_v2_stored_vec4_kernel<float, float4>(
    device const float*, device const float*, device const float*, device float*,
    constant NA1DParamsV2&, uint3);
template [[host_name("na1d_fused_v2_stored_vec4_fp16")]]
[[kernel]] void na1d_fused_v2_stored_vec4_kernel<half, half4>(
    device const half*, device const half*, device const half*, device half*,
    constant NA1DParamsV2&, uint3);
template [[host_name("na1d_fused_v2_stored_vec4_bf16")]]
[[kernel]] void na1d_fused_v2_stored_vec4_kernel<bfloat16_t, bfloat4>(
    device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*, device bfloat16_t*,
    constant NA1DParamsV2&, uint3);

// Backward attn scalar
template [[host_name("na1d_bwd_attn_v2_fp32")]]
[[kernel]] void na1d_bwd_attn_v2_kernel<float>(
    device const float*, device const float*, device const float*, device const float*,
    device float*, device float*, constant NA1DParamsV2&, uint3);
template [[host_name("na1d_bwd_attn_v2_fp16")]]
[[kernel]] void na1d_bwd_attn_v2_kernel<half>(
    device const half*, device const half*, device const half*, device const half*,
    device float*, device float*, constant NA1DParamsV2&, uint3);
template [[host_name("na1d_bwd_attn_v2_bf16")]]
[[kernel]] void na1d_bwd_attn_v2_kernel<bfloat16_t>(
    device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*,
    device float*, device float*, constant NA1DParamsV2&, uint3);

// Backward attn vec4
template [[host_name("na1d_bwd_attn_v2_vec4_fp32")]]
[[kernel]] void na1d_bwd_attn_v2_vec4_kernel<float, float4>(
    device const float*, device const float*, device const float*, device const float*,
    device float*, device float*, constant NA1DParamsV2&, uint3);
template [[host_name("na1d_bwd_attn_v2_vec4_fp16")]]
[[kernel]] void na1d_bwd_attn_v2_vec4_kernel<half, half4>(
    device const half*, device const half*, device const half*, device const half*,
    device float*, device float*, constant NA1DParamsV2&, uint3);
template [[host_name("na1d_bwd_attn_v2_vec4_bf16")]]
[[kernel]] void na1d_bwd_attn_v2_vec4_kernel<bfloat16_t, bfloat4>(
    device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*, device const bfloat16_t*,
    device float*, device float*, constant NA1DParamsV2&, uint3);

// Backward grad_q
template [[host_name("na1d_bwd_grad_q_v2_fp32")]]
[[kernel]] void na1d_bwd_grad_q_v2_kernel<float>(
    device const float*, device const float*, device float*,
    constant NA1DParamsV2&, uint3);
template [[host_name("na1d_bwd_grad_q_v2_fp16")]]
[[kernel]] void na1d_bwd_grad_q_v2_kernel<half>(
    device const float*, device const float*, device half*,
    constant NA1DParamsV2&, uint3);
template [[host_name("na1d_bwd_grad_q_v2_bf16")]]
[[kernel]] void na1d_bwd_grad_q_v2_kernel<bfloat16_t>(
    device const float*, device const float*, device bfloat16_t*,
    constant NA1DParamsV2&, uint3);

// Backward grad_k
template [[host_name("na1d_bwd_grad_k_v2_fp32")]]
[[kernel]] void na1d_bwd_grad_k_v2_kernel<float>(
    device const float*, device const float*, device float*,
    constant NA1DParamsV2&, uint3);
template [[host_name("na1d_bwd_grad_k_v2_fp16")]]
[[kernel]] void na1d_bwd_grad_k_v2_kernel<half>(
    device const float*, device const float*, device half*,
    constant NA1DParamsV2&, uint3);
template [[host_name("na1d_bwd_grad_k_v2_bf16")]]
[[kernel]] void na1d_bwd_grad_k_v2_kernel<bfloat16_t>(
    device const float*, device const float*, device bfloat16_t*,
    constant NA1DParamsV2&, uint3);

// Backward grad_v
template [[host_name("na1d_bwd_grad_v_v2_fp32")]]
[[kernel]] void na1d_bwd_grad_v_v2_kernel<float>(
    device const float*, device const float*, device float*,
    constant NA1DParamsV2&, uint3);
template [[host_name("na1d_bwd_grad_v_v2_fp16")]]
[[kernel]] void na1d_bwd_grad_v_v2_kernel<half>(
    device const float*, device const half*, device half*,
    constant NA1DParamsV2&, uint3);
template [[host_name("na1d_bwd_grad_v_v2_bf16")]]
[[kernel]] void na1d_bwd_grad_v_v2_kernel<bfloat16_t>(
    device const float*, device const bfloat16_t*, device bfloat16_t*,
    constant NA1DParamsV2&, uint3);

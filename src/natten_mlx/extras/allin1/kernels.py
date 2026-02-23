"""
Split NATTEN Kernels for Compatibility Layer

Partially-fused Metal kernels that expose intermediate stages:
- QK+RPB kernels: Fused QK computation, returns scores BEFORE softmax
- AV kernels: Fused attention-to-values, takes softmaxed scores

This allows compatibility with NATTEN functional API while maintaining
most of the performance benefits of fusion.
"""

import re

# Helper macros (same as kernels_shift.py)
NATTEN_HELPERS_SOURCE = '''
// NATTEN Helper Macros - Author-Compatible Shift Semantics

#define NATTEN_GET_WINDOW_START(OUT, IDX, LEN, K, NH, DIL) do {                 \\
    int _idx = (IDX);                                                          \\
    int _len = (LEN);                                                          \\
    int _K   = (K);                                                            \\
    int _nh  = (NH);                                                           \\
    int _d   = (DIL);                                                          \\
    int _dilation_idx = _idx % _d;                                             \\
    int _index_pdp = _idx / _d;                                                \\
    int _length_pdp = (_len + _d - 1) / _d;                                    \\
    int _num_padded = (_length_pdp * _d) - _len;                               \\
    if (_dilation_idx >= (_d - _num_padded)) {                                 \\
        _length_pdp -= 1;                                                      \\
    }                                                                          \\
    int _start_idx = _index_pdp - _nh;                                         \\
    if (_start_idx < 0) _start_idx = 0;                                        \\
    if (_index_pdp + _nh >= _length_pdp) {                                     \\
        _start_idx += (_length_pdp - _index_pdp - _nh - 1);                    \\
    }                                                                          \\
    (OUT) = _start_idx * _d + _dilation_idx;                                   \\
} while(0)

#define NATTEN_GET_WINDOW_END(OUT, START, LEN, K, DIL) do {                    \\
    int _end = (START) + (K) * (DIL);                                          \\
    if (_end > (LEN)) _end = (LEN);                                            \\
    (OUT) = _end;                                                              \\
} while(0)

#define NATTEN_GET_PB_START(OUT, IDX, LEN, K, NH, DIL) do {                    \\
    int _idx = (IDX);                                                          \\
    int _len = (LEN);                                                          \\
    int _K   = (K);                                                            \\
    int _nh  = (NH);                                                           \\
    int _d   = (DIL);                                                          \\
    int _pb;                                                                   \\
    if (_d <= 1) {                                                             \\
        _pb = _nh;                                                             \\
        if (_idx < _nh) _pb += (_nh - _idx);                                   \\
        if (_idx + _nh >= _len) _pb += (_len - _idx - 1 - _nh);                \\
    } else {                                                                   \\
        if (_idx - _nh * _d < 0) {                                             \\
            _pb = (_K - 1) - (_idx / _d);                                      \\
        } else if (_idx + _nh * _d >= _len) {                                  \\
            _pb = (_len - _idx - 1) / _d;                                      \\
        } else {                                                               \\
            _pb = _nh;                                                         \\
        }                                                                      \\
    }                                                                          \\
    (OUT) = _pb;                                                               \\
} while(0)
'''

# 1D K=3 QK+RPB kernel
NATTEN_1D_K3_QKRPB_SOURCE = NATTEN_HELPERS_SOURCE + '''
// 1D K=3 QK+RPB Kernel
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

int ni, ei, pi;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
NATTEN_GET_PB_START(pi, i, length, K, NH, dilation);

for (int ki = 0; ki < K; ki++) {
    int key_i = ni + ki * dilation;
    float score;
    if (key_i >= 0 && key_i < ei) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            int q_idx = (((b * heads + h) * length + i) * dim + d);
            int k_idx = (((b * heads + h) * length + key_i) * dim + d);
            sum += query[q_idx] * key[k_idx];
        }
        int rpb_idx = h * (2 * K - 1) + (pi + ki);
        score = sum + rpb[rpb_idx];
    } else {
        score = -INFINITY;
    }
    int out_idx = (((b * heads + h) * length + i) * K + ki);
    out[out_idx] = score;
}
'''

# 1D K=3 AV kernel
NATTEN_1D_K3_AV_SOURCE = NATTEN_HELPERS_SOURCE + '''
// 1D K=3 AV Kernel
uint3 gid = thread_position_in_grid;
const int batch_size = attention_probs_shape[0];
const int heads = attention_probs_shape[1];
const int length = attention_probs_shape[2];
const int dim = value_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

int ni, ei;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);

for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int ki = 0; ki < K; ki++) {
        int val_i = ni + ki * dilation;
        if (val_i >= 0 && val_i < ei) {
            int attn_idx = (((b * heads + h) * length + i) * K + ki);
            int val_idx = (((b * heads + h) * length + val_i) * dim + d);
            sum += attention_probs[attn_idx] * value[val_idx];
        }
    }
    int out_idx = (((b * heads + h) * length + i) * dim + d);
    out[out_idx] = sum;
}
'''

# 1D K=5 QK+RPB kernel
NATTEN_1D_K5_QKRPB_SOURCE = NATTEN_HELPERS_SOURCE + '''
// 1D K=5 QK+RPB Kernel
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

int ni, ei, pi;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
NATTEN_GET_PB_START(pi, i, length, K, NH, dilation);

for (int ki = 0; ki < K; ki++) {
    int key_i = ni + ki * dilation;
    float score;
    if (key_i >= 0 && key_i < ei) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            int q_idx = (((b * heads + h) * length + i) * dim + d);
            int k_idx = (((b * heads + h) * length + key_i) * dim + d);
            sum += query[q_idx] * key[k_idx];
        }
        int rpb_idx = h * (2 * K - 1) + (pi + ki);
        score = sum + rpb[rpb_idx];
    } else {
        score = -INFINITY;
    }
    int out_idx = (((b * heads + h) * length + i) * K + ki);
    out[out_idx] = score;
}
'''

# 1D K=5 QK+RPB kernel (fast D=12, precomputed window arrays)
NATTEN_1D_K5_QKRPB_FAST_D12_SOURCE = NATTEN_HELPERS_SOURCE + '''
// 1D K=5 QK+RPB Kernel (fast D=12, precomputed ni/ei/pi)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;

if (dim != 12) return;

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

int ni = ni_arr[i];
int ei = ei_arr[i];
int pi = pi_arr[i];

int q_base = (((b * heads + h) * length + i) * dim);
float4 q0 = *((device const float4*)(query + q_base + 0));
float4 q1 = *((device const float4*)(query + q_base + 4));
float4 q2 = *((device const float4*)(query + q_base + 8));

for (int ki = 0; ki < K; ki++) {
    int key_i = ni + ki * dilation;
    float score;
    if (key_i >= 0 && key_i < ei) {
        int k_base = (((b * heads + h) * length + key_i) * dim);
        float4 k0 = *((device const float4*)(key + k_base + 0));
        float4 k1 = *((device const float4*)(key + k_base + 4));
        float4 k2 = *((device const float4*)(key + k_base + 8));
        float sum = dot(q0, k0) + dot(q1, k1) + dot(q2, k2);
        int rpb_idx = h * (2 * K - 1) + (pi + ki);
        score = sum + rpb[rpb_idx];
    } else {
        score = -INFINITY;
    }
    int out_idx = (((b * heads + h) * length + i) * K + ki);
    out[out_idx] = score;
}
'''

# 1D K=5 AV kernel
NATTEN_1D_K5_AV_SOURCE = NATTEN_HELPERS_SOURCE + '''
// 1D K=5 AV Kernel
uint3 gid = thread_position_in_grid;
const int batch_size = attention_probs_shape[0];
const int heads = attention_probs_shape[1];
const int length = attention_probs_shape[2];
const int dim = value_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

int ni, ei;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);

for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int ki = 0; ki < K; ki++) {
        int val_i = ni + ki * dilation;
        if (val_i >= 0 && val_i < ei) {
            int attn_idx = (((b * heads + h) * length + i) * K + ki);
            int val_idx = (((b * heads + h) * length + val_i) * dim + d);
            sum += attention_probs[attn_idx] * value[val_idx];
        }
    }
    int out_idx = (((b * heads + h) * length + i) * dim + d);
    out[out_idx] = sum;
}
'''

# 1D K=5 AV kernel (fast D=12, precomputed window arrays)
NATTEN_1D_K5_AV_FAST_D12_SOURCE = NATTEN_HELPERS_SOURCE + '''
// 1D K=5 AV Kernel (fast D=12, precomputed ni/ei)
uint3 gid = thread_position_in_grid;
const int batch_size = attention_probs_shape[0];
const int heads = attention_probs_shape[1];
const int length = attention_probs_shape[2];
const int dim = value_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;

if (dim != 12) return;

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

int ni = ni_arr[i];
int ei = ei_arr[i];

float4 acc0 = float4(0.0f);
float4 acc1 = float4(0.0f);
float4 acc2 = float4(0.0f);

for (int ki = 0; ki < K; ki++) {
    int val_i = ni + ki * dilation;
    if (val_i >= 0 && val_i < ei) {
        int attn_idx = (((b * heads + h) * length + i) * K + ki);
        float w = attention_probs[attn_idx];
        int v_base = (((b * heads + h) * length + val_i) * dim);
        float4 v0 = *((device const float4*)(value + v_base + 0));
        float4 v1 = *((device const float4*)(value + v_base + 4));
        float4 v2 = *((device const float4*)(value + v_base + 8));
        acc0 += v0 * w;
        acc1 += v1 * w;
        acc2 += v2 * w;
    }
}

int out_base = (((b * heads + h) * length + i) * dim);
*((device float4*)(out + out_base + 0)) = acc0;
*((device float4*)(out + out_base + 4)) = acc1;
*((device float4*)(out + out_base + 8)) = acc2;
'''

# 1D K=7 QK+RPB kernel
NATTEN_1D_K7_QKRPB_SOURCE = NATTEN_HELPERS_SOURCE + '''
// 1D K=7 QK+RPB Kernel
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

int ni, ei, pi;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
NATTEN_GET_PB_START(pi, i, length, K, NH, dilation);

for (int ki = 0; ki < K; ki++) {
    int key_i = ni + ki * dilation;
    float score;
    if (key_i >= 0 && key_i < ei) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            int q_idx = (((b * heads + h) * length + i) * dim + d);
            int k_idx = (((b * heads + h) * length + key_i) * dim + d);
            sum += query[q_idx] * key[k_idx];
        }
        int rpb_idx = h * (2 * K - 1) + (pi + ki);
        score = sum + rpb[rpb_idx];
    } else {
        score = -INFINITY;
    }
    int out_idx = (((b * heads + h) * length + i) * K + ki);
    out[out_idx] = score;
}
'''

# 1D K=7 AV kernel
NATTEN_1D_K7_AV_SOURCE = NATTEN_HELPERS_SOURCE + '''
// 1D K=7 AV Kernel
uint3 gid = thread_position_in_grid;
const int batch_size = attention_probs_shape[0];
const int heads = attention_probs_shape[1];
const int length = attention_probs_shape[2];
const int dim = value_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

int ni, ei;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);

for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int ki = 0; ki < K; ki++) {
        int val_i = ni + ki * dilation;
        if (val_i >= 0 && val_i < ei) {
            int attn_idx = (((b * heads + h) * length + i) * K + ki);
            int val_idx = (((b * heads + h) * length + val_i) * dim + d);
            sum += attention_probs[attn_idx] * value[val_idx];
        }
    }
    int out_idx = (((b * heads + h) * length + i) * dim + d);
    out[out_idx] = sum;
}
'''

# K=3 QK+RPB kernel (returns scores BEFORE softmax)
NATTEN_K3_QKRPB_SOURCE = NATTEN_HELPERS_SOURCE + '''
// K=3 QK+RPB Kernel - Returns attention scores BEFORE softmax
// mx.fast.metal_kernel format (body only, MLX generates signature)

uint3 gid = thread_position_in_grid;

// Extract dimensions from input shapes
// query: [B, H, height, width, dim]
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dim = query_shape[4];

// Dilation from parameter array
const int dilation = (int)dilation_param[0];

// Kernel constants
const int K = 3;
const int NH = 1;  // K // 2
const int L = 9;   // K * K

// Thread mapping
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;

if (b >= batch_size || h >= heads || i >= height || j >= width) return;

// Compute shifted window boundaries
int ni, nj, ei, ej, pi, pj;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
NATTEN_GET_PB_START(pi, i, height, K, NH, dilation);
NATTEN_GET_PB_START(pj, j, width, K, NH, dilation);

// Compute QK+RPB for all neighbors
int neighbor_idx = 0;
for (int ki = 0; ki < K; ki++) {
    for (int kj = 0; kj < K; kj++) {
        int key_i = ni + ki * dilation;
        int key_j = nj + kj * dilation;

        float score;
        if (key_i >= 0 && key_i < ei && key_j >= 0 && key_j < ej) {
            // Compute QK in registers
            float sum = 0.0f;
            for (int d = 0; d < dim; d++) {
                int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
                int k_idx = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
                sum += query[q_idx] * key[k_idx];
            }

            // Add RPB
            int rpb_row = pi + ki;
            int rpb_col = pj + kj;
            int rpb_idx = h * (2 * K - 1) * (2 * K - 1) + rpb_row * (2 * K - 1) + rpb_col;
            score = sum + rpb[rpb_idx];
        } else {
            score = -INFINITY;
        }

        // Write to output
        int out_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
        out[out_idx] = score;
        neighbor_idx++;
    }
}
'''

# K=3 AV kernel (takes softmaxed attention, applies to values)
NATTEN_K3_AV_SOURCE = NATTEN_HELPERS_SOURCE + '''
// K=3 AV Kernel - Applies softmaxed attention to values
// mx.fast.metal_kernel format (body only, MLX generates signature)

uint3 gid = thread_position_in_grid;

// Extract dimensions from input shapes
// attention_probs: [B, H, height, width, L]
// value: [B, H, height, width, dim]
const int batch_size = attention_probs_shape[0];
const int heads = attention_probs_shape[1];
const int height = attention_probs_shape[2];
const int width = attention_probs_shape[3];
const int dim = value_shape[4];

// Dilation from parameter array
const int dilation = (int)dilation_param[0];

// Kernel constants
const int K = 3;
const int NH = 1;
const int L = 9;

// Thread mapping
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;

if (b >= batch_size || h >= heads || i >= height || j >= width) return;

// Compute shifted window boundaries
int ni, nj, ei, ej;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);

// Apply attention to values
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    int neighbor_idx = 0;

    for (int ki = 0; ki < K; ki++) {
        for (int kj = 0; kj < K; kj++) {
            int val_i = ni + ki * dilation;
            int val_j = nj + kj * dilation;

            if (val_i >= 0 && val_i < ei && val_j >= 0 && val_j < ej) {
                // Get attention weight
                int attn_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
                float attn_weight = attention_probs[attn_idx];

                // Get value
                int val_idx = (((b * heads + h) * height + val_i) * width + val_j) * dim + d;
                sum += attn_weight * value[val_idx];
            }
            neighbor_idx++;
        }
    }

    // Write output
    int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
    out[out_idx] = sum;
}
'''

# K=5 QK+RPB kernel
NATTEN_K5_QKRPB_SOURCE = NATTEN_HELPERS_SOURCE + '''
// K=5 QK+RPB Kernel
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dim = query_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
const int L = 25;

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;

int ni, nj, ei, ej, pi, pj;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
NATTEN_GET_PB_START(pi, i, height, K, NH, dilation);
NATTEN_GET_PB_START(pj, j, width, K, NH, dilation);

int neighbor_idx = 0;
for (int ki = 0; ki < K; ki++) {
    for (int kj = 0; kj < K; kj++) {
        int key_i = ni + ki * dilation;
        int key_j = nj + kj * dilation;
        float score;
        if (key_i >= 0 && key_i < ei && key_j >= 0 && key_j < ej) {
            float sum = 0.0f;
            for (int d = 0; d < dim; d++) {
                int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
                int k_idx = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
                sum += query[q_idx] * key[k_idx];
            }
            int rpb_idx = h * 81 + (pi + ki) * 9 + (pj + kj);
            score = sum + rpb[rpb_idx];
        } else {
            score = -INFINITY;
        }
        int out_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
        out[out_idx] = score;
        neighbor_idx++;
    }
}
'''

# K=5 AV kernel
NATTEN_K5_AV_SOURCE = NATTEN_HELPERS_SOURCE + '''
// K=5 AV Kernel
uint3 gid = thread_position_in_grid;
const int batch_size = attention_probs_shape[0];
const int heads = attention_probs_shape[1];
const int height = attention_probs_shape[2];
const int width = attention_probs_shape[3];
const int dim = value_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
const int L = 25;

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;

int ni, nj, ei, ej;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);

for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    int neighbor_idx = 0;
    for (int ki = 0; ki < K; ki++) {
        for (int kj = 0; kj < K; kj++) {
            int val_i = ni + ki * dilation;
            int val_j = nj + kj * dilation;
            if (val_i >= 0 && val_i < ei && val_j >= 0 && val_j < ej) {
                int attn_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
                int val_idx = (((b * heads + h) * height + val_i) * width + val_j) * dim + d;
                sum += attention_probs[attn_idx] * value[val_idx];
            }
            neighbor_idx++;
        }
    }
    int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
    out[out_idx] = sum;
}
'''

# K=7 QK+RPB kernel
NATTEN_K7_QKRPB_SOURCE = NATTEN_HELPERS_SOURCE + '''
// K=7 QK+RPB Kernel
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dim = query_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
const int L = 49;

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;

int ni, nj, ei, ej, pi, pj;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
NATTEN_GET_PB_START(pi, i, height, K, NH, dilation);
NATTEN_GET_PB_START(pj, j, width, K, NH, dilation);

int neighbor_idx = 0;
for (int ki = 0; ki < K; ki++) {
    for (int kj = 0; kj < K; kj++) {
        int key_i = ni + ki * dilation;
        int key_j = nj + kj * dilation;
        float score;
        if (key_i >= 0 && key_i < ei && key_j >= 0 && key_j < ej) {
            float sum = 0.0f;
            for (int d = 0; d < dim; d++) {
                int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
                int k_idx = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
                sum += query[q_idx] * key[k_idx];
            }
            int rpb_idx = h * 169 + (pi + ki) * 13 + (pj + kj);
            score = sum + rpb[rpb_idx];
        } else {
            score = -INFINITY;
        }
        int out_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
        out[out_idx] = score;
        neighbor_idx++;
    }
}
'''

# K=7 AV kernel
NATTEN_K7_AV_SOURCE = NATTEN_HELPERS_SOURCE + '''
// K=7 AV Kernel
uint3 gid = thread_position_in_grid;
const int batch_size = attention_probs_shape[0];
const int heads = attention_probs_shape[1];
const int height = attention_probs_shape[2];
const int width = attention_probs_shape[3];
const int dim = value_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
const int L = 49;

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;

int ni, nj, ei, ej;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);

for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    int neighbor_idx = 0;
    for (int ki = 0; ki < K; ki++) {
        for (int kj = 0; kj < K; kj++) {
            int val_i = ni + ki * dilation;
            int val_j = nj + kj * dilation;
            if (val_i >= 0 && val_i < ei && val_j >= 0 && val_j < ej) {
                int attn_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
                int val_idx = (((b * heads + h) * height + val_i) * width + val_j) * dim + d;
                sum += attention_probs[attn_idx] * value[val_idx];
            }
            neighbor_idx++;
        }
    }
    int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
    out[out_idx] = sum;
}
'''


# Backward kernels (slow, reduction-based)
NATTEN_K3_QK_BWD_DQ_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_query (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dim = query_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int L = 9;
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;
int ni, nj, ei, ej;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    int neighbor_idx = 0;
    for (int ki = 0; ki < K; ki++) {
        for (int kj = 0; kj < K; kj++) {
            int key_i = ni + ki * dilation;
            int key_j = nj + kj * dilation;
            if (key_i >= 0 && key_i < ei && key_j >= 0 && key_j < ej) {
                int k_idx = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
                int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
                sum += d_attn[da_idx] * key[k_idx];
            }
            neighbor_idx++;
        }
    }
    int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_K3_QK_BWD_DK_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_key (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int height = key_shape[2];
const int width = key_shape[3];
const int dim = key_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int L = 9;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.y;
int vj = gid.x;
if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = 0; i < height; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        for (int j = 0; j < width; j++) {
            int nj, ej;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            if (vj < nj || vj >= ej) continue;
            int dj = vj - nj;
            if (dj % dilation != 0) continue;
            int kj = dj / dilation;
            if (kj < 0 || kj >= K) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
            sum += d_attn[da_idx] * query[q_idx];
        }
    }
    int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_K3_QK_BWD_DRPB_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_rpb (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int rpb_h = 2 * K - 1;
const int rpb_w = 2 * K - 1;
int h = gid.z;
int ri = gid.y;
int rj = gid.x;
if (h >= heads || ri >= rpb_h || rj >= rpb_w) return;
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < height; i++) {
        int ni, ei, pi;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        NATTEN_GET_PB_START(pi, i, height, K, NH, dilation);
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dilation;
        if (key_i < 0 || key_i >= ei) continue;
        for (int j = 0; j < width; j++) {
            int nj, ej, pj;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            NATTEN_GET_PB_START(pj, j, width, K, NH, dilation);
            int kj = rj - pj;
            if (kj < 0 || kj >= K) continue;
            int key_j = nj + kj * dilation;
            if (key_j < 0 || key_j >= ej) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * (K*K) + neighbor_idx;
            sum += d_attn[da_idx];
        }
    }
}
int out_idx = (h * rpb_h + ri) * rpb_w + rj;
out[out_idx] = sum;

'''

NATTEN_K3_AV_BWD_DATTN_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_attn (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int height = d_out_shape[2];
const int width = d_out_shape[3];
const int dim = d_out_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int L = 9;
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;
int ni, nj, ei, ej;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
int neighbor_idx = 0;
for (int ki = 0; ki < K; ki++) {
    for (int kj = 0; kj < K; kj++) {
        int val_i = ni + ki * dilation;
        int val_j = nj + kj * dilation;
        float sum = 0.0f;
        if (val_i >= 0 && val_i < ei && val_j >= 0 && val_j < ej) {
            for (int d = 0; d < dim; d++) {
                int do_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
                int v_idx = (((b * heads + h) * height + val_i) * width + val_j) * dim + d;
                sum += d_out[do_idx] * value[v_idx];
            }
        }
        int out_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
        out[out_idx] = sum;
        neighbor_idx++;
    }
}

'''

NATTEN_K3_AV_BWD_DVAL_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_value (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int height = value_shape[2];
const int width = value_shape[3];
const int dim = value_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int L = 9;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.y;
int vj = gid.x;
if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = 0; i < height; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        for (int j = 0; j < width; j++) {
            int nj, ej;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            if (vj < nj || vj >= ej) continue;
            int dj = vj - nj;
            if (dj % dilation != 0) continue;
            int kj = dj / dilation;
            if (kj < 0 || kj >= K) continue;
            int neighbor_idx = ki * K + kj;
            int do_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
            int attn_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            sum += d_out[do_idx] * attn[attn_idx];
        }
    }
    int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_1D_K3_QK_BWD_DQ_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_query (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;
int ni, ei;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    int neighbor_idx = 0;
    for (int ki = 0; ki < K; ki++) {
        int key_i = ni + ki * dilation;
        if (key_i >= 0 && key_i < ei) {
            int k_idx = (((b * heads + h) * length + key_i) * dim + d);
            int da_idx = (((b * heads + h) * length + i) * K + neighbor_idx);
            sum += d_attn[da_idx] * key[k_idx];
        }
        neighbor_idx++;
    }
    int out_idx = (((b * heads + h) * length + i) * dim + d);
    out[out_idx] = sum;
}

'''

NATTEN_1D_K3_QK_BWD_DK_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_key (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int length = key_shape[2];
const int dim = key_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int da_idx = (((b * heads + h) * length + i) * K + ki);
        int q_idx = (((b * heads + h) * length + i) * dim + d);
        sum += d_attn[da_idx] * query[q_idx];
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d);
    out[out_idx] = sum;
}

'''

NATTEN_1D_K3_QK_BWD_DRPB_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_rpb (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int rpb_len = 2 * K - 1;
int h = gid.z;
int ri = gid.x;
if (h >= heads || ri >= rpb_len) return;
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < length; i++) {
        int ni, ei, pi;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        NATTEN_GET_PB_START(pi, i, length, K, NH, dilation);
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dilation;
        if (key_i < 0 || key_i >= ei) continue;
        int da_idx = (((b * heads + h) * length + i) * K + ki);
        sum += d_attn[da_idx];
    }
}
int out_idx = h * rpb_len + ri;
out[out_idx] = sum;

'''

NATTEN_1D_K3_AV_BWD_DATTN_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D AV backward d_attn (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int length = d_out_shape[2];
const int dim = d_out_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;
int ni, ei;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
for (int ki = 0; ki < K; ki++) {
    int val_i = ni + ki * dilation;
    float sum = 0.0f;
    if (val_i >= 0 && val_i < ei) {
        for (int d = 0; d < dim; d++) {
            int do_idx = (((b * heads + h) * length + i) * dim + d);
            int v_idx = (((b * heads + h) * length + val_i) * dim + d);
            sum += d_out[do_idx] * value[v_idx];
        }
    }
    int out_idx = (((b * heads + h) * length + i) * K + ki);
    out[out_idx] = sum;
}

'''

NATTEN_1D_K3_AV_BWD_DVAL_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D AV backward d_value (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int length = value_shape[2];
const int dim = value_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int do_idx = (((b * heads + h) * length + i) * dim + d);
        int attn_idx = (((b * heads + h) * length + i) * K + ki);
        sum += d_out[do_idx] * attn[attn_idx];
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d);
    out[out_idx] = sum;
}

'''

NATTEN_K5_QK_BWD_DQ_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_query (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dim = query_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
const int L = 25;
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;
int ni, nj, ei, ej;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    int neighbor_idx = 0;
    for (int ki = 0; ki < K; ki++) {
        for (int kj = 0; kj < K; kj++) {
            int key_i = ni + ki * dilation;
            int key_j = nj + kj * dilation;
            if (key_i >= 0 && key_i < ei && key_j >= 0 && key_j < ej) {
                int k_idx = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
                int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
                sum += d_attn[da_idx] * key[k_idx];
            }
            neighbor_idx++;
        }
    }
    int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_K5_QK_BWD_DK_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_key (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int height = key_shape[2];
const int width = key_shape[3];
const int dim = key_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
const int L = 25;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.y;
int vj = gid.x;
if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = 0; i < height; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        for (int j = 0; j < width; j++) {
            int nj, ej;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            if (vj < nj || vj >= ej) continue;
            int dj = vj - nj;
            if (dj % dilation != 0) continue;
            int kj = dj / dilation;
            if (kj < 0 || kj >= K) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
            sum += d_attn[da_idx] * query[q_idx];
        }
    }
    int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_K5_QK_BWD_DRPB_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_rpb (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
const int rpb_h = 2 * K - 1;
const int rpb_w = 2 * K - 1;
int h = gid.z;
int ri = gid.y;
int rj = gid.x;
if (h >= heads || ri >= rpb_h || rj >= rpb_w) return;
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < height; i++) {
        int ni, ei, pi;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        NATTEN_GET_PB_START(pi, i, height, K, NH, dilation);
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dilation;
        if (key_i < 0 || key_i >= ei) continue;
        for (int j = 0; j < width; j++) {
            int nj, ej, pj;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            NATTEN_GET_PB_START(pj, j, width, K, NH, dilation);
            int kj = rj - pj;
            if (kj < 0 || kj >= K) continue;
            int key_j = nj + kj * dilation;
            if (key_j < 0 || key_j >= ej) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * (K*K) + neighbor_idx;
            sum += d_attn[da_idx];
        }
    }
}
int out_idx = (h * rpb_h + ri) * rpb_w + rj;
out[out_idx] = sum;

'''

NATTEN_K5_AV_BWD_DATTN_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_attn (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int height = d_out_shape[2];
const int width = d_out_shape[3];
const int dim = d_out_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
const int L = 25;
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;
int ni, nj, ei, ej;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
int neighbor_idx = 0;
for (int ki = 0; ki < K; ki++) {
    for (int kj = 0; kj < K; kj++) {
        int val_i = ni + ki * dilation;
        int val_j = nj + kj * dilation;
        float sum = 0.0f;
        if (val_i >= 0 && val_i < ei && val_j >= 0 && val_j < ej) {
            for (int d = 0; d < dim; d++) {
                int do_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
                int v_idx = (((b * heads + h) * height + val_i) * width + val_j) * dim + d;
                sum += d_out[do_idx] * value[v_idx];
            }
        }
        int out_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
        out[out_idx] = sum;
        neighbor_idx++;
    }
}

'''

NATTEN_K5_AV_BWD_DVAL_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_value (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int height = value_shape[2];
const int width = value_shape[3];
const int dim = value_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
const int L = 25;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.y;
int vj = gid.x;
if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = 0; i < height; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        for (int j = 0; j < width; j++) {
            int nj, ej;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            if (vj < nj || vj >= ej) continue;
            int dj = vj - nj;
            if (dj % dilation != 0) continue;
            int kj = dj / dilation;
            if (kj < 0 || kj >= K) continue;
            int neighbor_idx = ki * K + kj;
            int do_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
            int attn_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            sum += d_out[do_idx] * attn[attn_idx];
        }
    }
    int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_1D_K5_QK_BWD_DQ_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_query (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;
int ni, ei;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    int neighbor_idx = 0;
    for (int ki = 0; ki < K; ki++) {
        int key_i = ni + ki * dilation;
        if (key_i >= 0 && key_i < ei) {
            int k_idx = (((b * heads + h) * length + key_i) * dim + d);
            int da_idx = (((b * heads + h) * length + i) * K + neighbor_idx);
            sum += d_attn[da_idx] * key[k_idx];
        }
        neighbor_idx++;
    }
    int out_idx = (((b * heads + h) * length + i) * dim + d);
    out[out_idx] = sum;
}

'''

NATTEN_1D_K5_QK_BWD_DK_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_key (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int length = key_shape[2];
const int dim = key_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int da_idx = (((b * heads + h) * length + i) * K + ki);
        int q_idx = (((b * heads + h) * length + i) * dim + d);
        sum += d_attn[da_idx] * query[q_idx];
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d);
    out[out_idx] = sum;
}

'''

NATTEN_1D_K5_QK_BWD_DRPB_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_rpb (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
const int rpb_len = 2 * K - 1;
int h = gid.z;
int ri = gid.x;
if (h >= heads || ri >= rpb_len) return;
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < length; i++) {
        int ni, ei, pi;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        NATTEN_GET_PB_START(pi, i, length, K, NH, dilation);
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dilation;
        if (key_i < 0 || key_i >= ei) continue;
        int da_idx = (((b * heads + h) * length + i) * K + ki);
        sum += d_attn[da_idx];
    }
}
int out_idx = h * rpb_len + ri;
out[out_idx] = sum;

'''

NATTEN_1D_K5_AV_BWD_DATTN_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D AV backward d_attn (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int length = d_out_shape[2];
const int dim = d_out_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;
int ni, ei;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
for (int ki = 0; ki < K; ki++) {
    int val_i = ni + ki * dilation;
    float sum = 0.0f;
    if (val_i >= 0 && val_i < ei) {
        for (int d = 0; d < dim; d++) {
            int do_idx = (((b * heads + h) * length + i) * dim + d);
            int v_idx = (((b * heads + h) * length + val_i) * dim + d);
            sum += d_out[do_idx] * value[v_idx];
        }
    }
    int out_idx = (((b * heads + h) * length + i) * K + ki);
    out[out_idx] = sum;
}

'''

NATTEN_1D_K5_AV_BWD_DVAL_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D AV backward d_value (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int length = value_shape[2];
const int dim = value_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int do_idx = (((b * heads + h) * length + i) * dim + d);
        int attn_idx = (((b * heads + h) * length + i) * K + ki);
        sum += d_out[do_idx] * attn[attn_idx];
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d);
    out[out_idx] = sum;
}

'''

NATTEN_K7_QK_BWD_DQ_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_query (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dim = query_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
const int L = 49;
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;
int ni, nj, ei, ej;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    int neighbor_idx = 0;
    for (int ki = 0; ki < K; ki++) {
        for (int kj = 0; kj < K; kj++) {
            int key_i = ni + ki * dilation;
            int key_j = nj + kj * dilation;
            if (key_i >= 0 && key_i < ei && key_j >= 0 && key_j < ej) {
                int k_idx = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
                int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
                sum += d_attn[da_idx] * key[k_idx];
            }
            neighbor_idx++;
        }
    }
    int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_K7_QK_BWD_DK_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_key (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int height = key_shape[2];
const int width = key_shape[3];
const int dim = key_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
const int L = 49;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.y;
int vj = gid.x;
if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = 0; i < height; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        for (int j = 0; j < width; j++) {
            int nj, ej;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            if (vj < nj || vj >= ej) continue;
            int dj = vj - nj;
            if (dj % dilation != 0) continue;
            int kj = dj / dilation;
            if (kj < 0 || kj >= K) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
            sum += d_attn[da_idx] * query[q_idx];
        }
    }
    int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_K7_QK_BWD_DRPB_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_rpb (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
const int rpb_h = 2 * K - 1;
const int rpb_w = 2 * K - 1;
int h = gid.z;
int ri = gid.y;
int rj = gid.x;
if (h >= heads || ri >= rpb_h || rj >= rpb_w) return;
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < height; i++) {
        int ni, ei, pi;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        NATTEN_GET_PB_START(pi, i, height, K, NH, dilation);
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dilation;
        if (key_i < 0 || key_i >= ei) continue;
        for (int j = 0; j < width; j++) {
            int nj, ej, pj;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            NATTEN_GET_PB_START(pj, j, width, K, NH, dilation);
            int kj = rj - pj;
            if (kj < 0 || kj >= K) continue;
            int key_j = nj + kj * dilation;
            if (key_j < 0 || key_j >= ej) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * (K*K) + neighbor_idx;
            sum += d_attn[da_idx];
        }
    }
}
int out_idx = (h * rpb_h + ri) * rpb_w + rj;
out[out_idx] = sum;

'''

NATTEN_K7_AV_BWD_DATTN_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_attn (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int height = d_out_shape[2];
const int width = d_out_shape[3];
const int dim = d_out_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
const int L = 49;
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;
int ni, nj, ei, ej;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
int neighbor_idx = 0;
for (int ki = 0; ki < K; ki++) {
    for (int kj = 0; kj < K; kj++) {
        int val_i = ni + ki * dilation;
        int val_j = nj + kj * dilation;
        float sum = 0.0f;
        if (val_i >= 0 && val_i < ei && val_j >= 0 && val_j < ej) {
            for (int d = 0; d < dim; d++) {
                int do_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
                int v_idx = (((b * heads + h) * height + val_i) * width + val_j) * dim + d;
                sum += d_out[do_idx] * value[v_idx];
            }
        }
        int out_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
        out[out_idx] = sum;
        neighbor_idx++;
    }
}

'''

NATTEN_K7_AV_BWD_DVAL_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_value (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int height = value_shape[2];
const int width = value_shape[3];
const int dim = value_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
const int L = 49;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.y;
int vj = gid.x;
if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = 0; i < height; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        for (int j = 0; j < width; j++) {
            int nj, ej;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            if (vj < nj || vj >= ej) continue;
            int dj = vj - nj;
            if (dj % dilation != 0) continue;
            int kj = dj / dilation;
            if (kj < 0 || kj >= K) continue;
            int neighbor_idx = ki * K + kj;
            int do_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
            int attn_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            sum += d_out[do_idx] * attn[attn_idx];
        }
    }
    int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_1D_K7_QK_BWD_DQ_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_query (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;
int ni, ei;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    int neighbor_idx = 0;
    for (int ki = 0; ki < K; ki++) {
        int key_i = ni + ki * dilation;
        if (key_i >= 0 && key_i < ei) {
            int k_idx = (((b * heads + h) * length + key_i) * dim + d);
            int da_idx = (((b * heads + h) * length + i) * K + neighbor_idx);
            sum += d_attn[da_idx] * key[k_idx];
        }
        neighbor_idx++;
    }
    int out_idx = (((b * heads + h) * length + i) * dim + d);
    out[out_idx] = sum;
}

'''

NATTEN_1D_K7_QK_BWD_DK_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_key (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int length = key_shape[2];
const int dim = key_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int da_idx = (((b * heads + h) * length + i) * K + ki);
        int q_idx = (((b * heads + h) * length + i) * dim + d);
        sum += d_attn[da_idx] * query[q_idx];
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d);
    out[out_idx] = sum;
}

'''

NATTEN_1D_K7_QK_BWD_DRPB_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_rpb (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
const int rpb_len = 2 * K - 1;
int h = gid.z;
int ri = gid.x;
if (h >= heads || ri >= rpb_len) return;
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < length; i++) {
        int ni, ei, pi;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        NATTEN_GET_PB_START(pi, i, length, K, NH, dilation);
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dilation;
        if (key_i < 0 || key_i >= ei) continue;
        int da_idx = (((b * heads + h) * length + i) * K + ki);
        sum += d_attn[da_idx];
    }
}
int out_idx = h * rpb_len + ri;
out[out_idx] = sum;

'''

NATTEN_1D_K7_AV_BWD_DATTN_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D AV backward d_attn (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int length = d_out_shape[2];
const int dim = d_out_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;
int ni, ei;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
for (int ki = 0; ki < K; ki++) {
    int val_i = ni + ki * dilation;
    float sum = 0.0f;
    if (val_i >= 0 && val_i < ei) {
        for (int d = 0; d < dim; d++) {
            int do_idx = (((b * heads + h) * length + i) * dim + d);
            int v_idx = (((b * heads + h) * length + val_i) * dim + d);
            sum += d_out[do_idx] * value[v_idx];
        }
    }
    int out_idx = (((b * heads + h) * length + i) * K + ki);
    out[out_idx] = sum;
}

'''

NATTEN_1D_K7_AV_BWD_DVAL_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D AV backward d_value (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int length = value_shape[2];
const int dim = value_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int do_idx = (((b * heads + h) * length + i) * dim + d);
        int attn_idx = (((b * heads + h) * length + i) * K + ki);
        sum += d_out[do_idx] * attn[attn_idx];
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d);
    out[out_idx] = sum;
}

'''


# Backward kernels (FAST bounded-range variants)
NATTEN_K3_QK_BWD_DK_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_key FAST (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int height = key_shape[2];
const int width = key_shape[3];
const int dim = key_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int L = 9;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.y;
int vj = gid.x;
if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(height - 1, vi + (K - 1) * dilation);
int j_start = max(0, vj - (K - 1) * dilation);
int j_end = min(width - 1, vj + (K - 1) * dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        for (int j = j_start; j <= j_end; j++) {
            int nj, ej;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            if (vj < nj || vj >= ej) continue;
            int dj = vj - nj;
            if (dj % dilation != 0) continue;
            int kj = dj / dilation;
            if (kj < 0 || kj >= K) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
            sum += d_attn[da_idx] * query[q_idx];
        }
    }
    int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_K3_AV_BWD_DVAL_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_value FAST (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int height = value_shape[2];
const int width = value_shape[3];
const int dim = value_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int L = 9;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.y;
int vj = gid.x;
if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(height - 1, vi + (K - 1) * dilation);
int j_start = max(0, vj - (K - 1) * dilation);
int j_end = min(width - 1, vj + (K - 1) * dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        for (int j = j_start; j <= j_end; j++) {
            int nj, ej;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            if (vj < nj || vj >= ej) continue;
            int dj = vj - nj;
            if (dj % dilation != 0) continue;
            int kj = dj / dilation;
            if (kj < 0 || kj >= K) continue;
            int neighbor_idx = ki * K + kj;
            int do_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
            int attn_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            sum += d_out[do_idx] * attn[attn_idx];
        }
    }
    int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_1D_K3_QK_BWD_DK_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_key FAST (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int length = key_shape[2];
const int dim = key_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(length - 1, vi + (K - 1) * dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int da_idx = (((b * heads + h) * length + i) * K + ki);
        int q_idx = (((b * heads + h) * length + i) * dim + d);
        sum += d_attn[da_idx] * query[q_idx];
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d);
    out[out_idx] = sum;
}

'''

NATTEN_1D_K3_AV_BWD_DVAL_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D AV backward d_value FAST (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int length = value_shape[2];
const int dim = value_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(length - 1, vi + (K - 1) * dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int do_idx = (((b * heads + h) * length + i) * dim + d);
        int attn_idx = (((b * heads + h) * length + i) * K + ki);
        sum += d_out[do_idx] * attn[attn_idx];
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d);
out[out_idx] = sum;
}

'''

NATTEN_K3_QK_BWD_DK_FAST_V4_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_key FAST (K=3) vectorized
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int height = key_shape[2];
const int width = key_shape[3];
const int dim = key_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int L = 9;
const int DTILE = 4;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.y;
int vj = gid.x;
if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(height - 1, vi + (K - 1) * dilation);
int j_start = max(0, vj - (K - 1) * dilation);
int j_end = min(width - 1, vj + (K - 1) * dilation);
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 sum4 = float4(0.0f);
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        for (int j = j_start; j <= j_end; j++) {
            int nj, ej;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            if (vj < nj || vj >= ej) continue;
            int dj = vj - nj;
            if (dj % dilation != 0) continue;
            int kj = dj / dilation;
            if (kj < 0 || kj >= K) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            float w = d_attn[da_idx];
            int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d0;
            float4 qv = float4(0.0f);
            if (d0 + 3 < dim) {
                qv = *((device const float4*)(query + q_idx));
            } else {
                if (d0 + 0 < dim) qv.x = query[q_idx + 0];
                if (d0 + 1 < dim) qv.y = query[q_idx + 1];
                if (d0 + 2 < dim) qv.z = query[q_idx + 2];
                if (d0 + 3 < dim) qv.w = query[q_idx + 3];
            }
            sum4 = fma(w, qv, sum4);
        }
    }
    int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d0;
    if (d0 + 0 < dim) out[out_idx + 0] = sum4.x;
    if (d0 + 1 < dim) out[out_idx + 1] = sum4.y;
    if (d0 + 2 < dim) out[out_idx + 2] = sum4.z;
    if (d0 + 3 < dim) out[out_idx + 3] = sum4.w;
}

'''

NATTEN_1D_K3_QK_BWD_DK_FAST_V4_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_key FAST (K=3) vectorized
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int length = key_shape[2];
const int dim = key_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int DTILE = 4;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(length - 1, vi + (K - 1) * dilation);
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 sum4 = float4(0.0f);
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int da_idx = (((b * heads + h) * length + i) * K + ki);
        float w = d_attn[da_idx];
        int q_idx = (((b * heads + h) * length + i) * dim + d0);
        float4 qv = float4(0.0f);
        if (d0 + 3 < dim) {
            qv = *((device const float4*)(query + q_idx));
        } else {
            if (d0 + 0 < dim) qv.x = query[q_idx + 0];
            if (d0 + 1 < dim) qv.y = query[q_idx + 1];
            if (d0 + 2 < dim) qv.z = query[q_idx + 2];
            if (d0 + 3 < dim) qv.w = query[q_idx + 3];
        }
        sum4 = fma(w, qv, sum4);
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d0);
    if (d0 + 0 < dim) out[out_idx + 0] = sum4.x;
    if (d0 + 1 < dim) out[out_idx + 1] = sum4.y;
    if (d0 + 2 < dim) out[out_idx + 2] = sum4.z;
    if (d0 + 3 < dim) out[out_idx + 3] = sum4.w;
}

'''

NATTEN_1D_K3_AV_BWD_DVAL_FAST_V4_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D AV backward d_value FAST (K=3) vectorized
uint3 gid = thread_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int length = value_shape[2];
const int dim = value_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int DTILE = 4;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(length - 1, vi + (K - 1) * dilation);
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 sum4 = float4(0.0f);
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int do_idx = (((b * heads + h) * length + i) * dim + d0);
        float4 do_v = float4(0.0f);
        if (d0 + 3 < dim) {
            do_v = *((device const float4*)(d_out + do_idx));
        } else {
            if (d0 + 0 < dim) do_v.x = d_out[do_idx + 0];
            if (d0 + 1 < dim) do_v.y = d_out[do_idx + 1];
            if (d0 + 2 < dim) do_v.z = d_out[do_idx + 2];
            if (d0 + 3 < dim) do_v.w = d_out[do_idx + 3];
        }
        int attn_idx = (((b * heads + h) * length + i) * K + ki);
        float attn_val = attn[attn_idx];
        sum4 = fma(attn_val, do_v, sum4);
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d0);
    if (d0 + 0 < dim) out[out_idx + 0] = sum4.x;
    if (d0 + 1 < dim) out[out_idx + 1] = sum4.y;
    if (d0 + 2 < dim) out[out_idx + 2] = sum4.z;
    if (d0 + 3 < dim) out[out_idx + 3] = sum4.w;
}

'''

NATTEN_K5_QK_BWD_DK_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_key FAST (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int height = key_shape[2];
const int width = key_shape[3];
const int dim = key_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
const int L = 25;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.y;
int vj = gid.x;
if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(height - 1, vi + (K - 1) * dilation);
int j_start = max(0, vj - (K - 1) * dilation);
int j_end = min(width - 1, vj + (K - 1) * dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        for (int j = j_start; j <= j_end; j++) {
            int nj, ej;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            if (vj < nj || vj >= ej) continue;
            int dj = vj - nj;
            if (dj % dilation != 0) continue;
            int kj = dj / dilation;
            if (kj < 0 || kj >= K) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
            sum += d_attn[da_idx] * query[q_idx];
        }
    }
    int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_K5_AV_BWD_DVAL_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_value FAST (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int height = value_shape[2];
const int width = value_shape[3];
const int dim = value_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
const int L = 25;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.y;
int vj = gid.x;
if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(height - 1, vi + (K - 1) * dilation);
int j_start = max(0, vj - (K - 1) * dilation);
int j_end = min(width - 1, vj + (K - 1) * dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        for (int j = j_start; j <= j_end; j++) {
            int nj, ej;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            if (vj < nj || vj >= ej) continue;
            int dj = vj - nj;
            if (dj % dilation != 0) continue;
            int kj = dj / dilation;
            if (kj < 0 || kj >= K) continue;
            int neighbor_idx = ki * K + kj;
            int do_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
            int attn_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            sum += d_out[do_idx] * attn[attn_idx];
        }
    }
    int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_1D_K5_QK_BWD_DK_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_key FAST (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int length = key_shape[2];
const int dim = key_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(length - 1, vi + (K - 1) * dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int da_idx = (((b * heads + h) * length + i) * K + ki);
        int q_idx = (((b * heads + h) * length + i) * dim + d);
        sum += d_attn[da_idx] * query[q_idx];
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d);
    out[out_idx] = sum;
}

'''

NATTEN_1D_K5_AV_BWD_DVAL_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D AV backward d_value FAST (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int length = value_shape[2];
const int dim = value_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(length - 1, vi + (K - 1) * dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int do_idx = (((b * heads + h) * length + i) * dim + d);
        int attn_idx = (((b * heads + h) * length + i) * K + ki);
        sum += d_out[do_idx] * attn[attn_idx];
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d);
    out[out_idx] = sum;
}

'''

NATTEN_K7_QK_BWD_DK_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_key FAST (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int height = key_shape[2];
const int width = key_shape[3];
const int dim = key_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
const int L = 49;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.y;
int vj = gid.x;
if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(height - 1, vi + (K - 1) * dilation);
int j_start = max(0, vj - (K - 1) * dilation);
int j_end = min(width - 1, vj + (K - 1) * dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        for (int j = j_start; j <= j_end; j++) {
            int nj, ej;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            if (vj < nj || vj >= ej) continue;
            int dj = vj - nj;
            if (dj % dilation != 0) continue;
            int kj = dj / dilation;
            if (kj < 0 || kj >= K) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
            sum += d_attn[da_idx] * query[q_idx];
        }
    }
    int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_K7_AV_BWD_DVAL_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_value FAST (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int height = value_shape[2];
const int width = value_shape[3];
const int dim = value_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
const int L = 49;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.y;
int vj = gid.x;
if (b >= batch_size || h >= heads || vi >= height || vj >= width) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(height - 1, vi + (K - 1) * dilation);
int j_start = max(0, vj - (K - 1) * dilation);
int j_end = min(width - 1, vj + (K - 1) * dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        for (int j = j_start; j <= j_end; j++) {
            int nj, ej;
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            if (vj < nj || vj >= ej) continue;
            int dj = vj - nj;
            if (dj % dilation != 0) continue;
            int kj = dj / dilation;
            if (kj < 0 || kj >= K) continue;
            int neighbor_idx = ki * K + kj;
            int do_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
            int attn_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            sum += d_out[do_idx] * attn[attn_idx];
        }
    }
    int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d;
    out[out_idx] = sum;
}

'''

NATTEN_1D_K7_QK_BWD_DK_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_key FAST (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int length = key_shape[2];
const int dim = key_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(length - 1, vi + (K - 1) * dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int da_idx = (((b * heads + h) * length + i) * K + ki);
        int q_idx = (((b * heads + h) * length + i) * dim + d);
        sum += d_attn[da_idx] * query[q_idx];
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d);
    out[out_idx] = sum;
}

'''

NATTEN_1D_K7_AV_BWD_DVAL_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D AV backward d_value FAST (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int length = value_shape[2];
const int dim = value_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
int b = gid.z / heads;
int h = gid.z % heads;
int vi = gid.x;
if (b >= batch_size || h >= heads || vi >= length) return;
int i_start = max(0, vi - (K - 1) * dilation);
int i_end = min(length - 1, vi + (K - 1) * dilation);
for (int d = 0; d < dim; d++) {
    float sum = 0.0f;
    for (int i = i_start; i <= i_end; i++) {
        int ni, ei;
        NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
        NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        if (vi < ni || vi >= ei) continue;
        int di = vi - ni;
        if (di % dilation != 0) continue;
        int ki = di / dilation;
        if (ki < 0 || ki >= K) continue;
        int do_idx = (((b * heads + h) * length + i) * dim + d);
        int attn_idx = (((b * heads + h) * length + i) * K + ki);
        sum += d_out[do_idx] * attn[attn_idx];
    }
    int out_idx = (((b * heads + h) * length + vi) * dim + d);
out[out_idx] = sum;
}

'''

NATTEN_K5_QK_BWD_DK_FAST_V4_SOURCE = NATTEN_K3_QK_BWD_DK_FAST_V4_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2").replace("L = 9", "L = 25")
NATTEN_K7_QK_BWD_DK_FAST_V4_SOURCE = NATTEN_K3_QK_BWD_DK_FAST_V4_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3").replace("L = 9", "L = 49")

NATTEN_1D_K5_QK_BWD_DK_FAST_V4_SOURCE = NATTEN_1D_K3_QK_BWD_DK_FAST_V4_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2")
NATTEN_1D_K7_QK_BWD_DK_FAST_V4_SOURCE = NATTEN_1D_K3_QK_BWD_DK_FAST_V4_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3")

NATTEN_1D_K5_AV_BWD_DVAL_FAST_V4_SOURCE = NATTEN_1D_K3_AV_BWD_DVAL_FAST_V4_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2")
NATTEN_1D_K7_AV_BWD_DVAL_FAST_V4_SOURCE = NATTEN_1D_K3_AV_BWD_DVAL_FAST_V4_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3")


# Backward kernels (FAST drpb precomputed indices)
NATTEN_K3_QK_BWD_DRPB_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_rpb FAST (precomputed pi/pj/ni/nj/ei/ej) (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int K = 3;
const int NH = 1;
const int rpb_h = 2 * K - 1;
const int rpb_w = 2 * K - 1;
int h = gid.z;
int ri = gid.y;
int rj = gid.x;
if (h >= heads || ri >= rpb_h || rj >= rpb_w) return;
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < height; i++) {
        int pi = pi_arr[i];
        int ni = ni_arr[i];
        int ei = ei_arr[i];
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dilation_param[0];
        if (key_i < 0 || key_i >= ei) continue;
        for (int j = 0; j < width; j++) {
            int pj = pj_arr[j];
            int nj = nj_arr[j];
            int ej = ej_arr[j];
            int kj = rj - pj;
            if (kj < 0 || kj >= K) continue;
            int key_j = nj + kj * dilation_param[0];
            if (key_j < 0 || key_j >= ej) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * (K*K) + neighbor_idx;
            sum += d_attn[da_idx];
        }
    }
}
int out_idx = (h * rpb_h + ri) * rpb_w + rj;
out[out_idx] = sum;

'''

NATTEN_K3_QK_BWD_DRPB_FAST_U2_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_rpb FAST (precomputed pi/pj/ni/nj/ei/ej) (K=3) unroll j by 2
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int K = 3;
const int NH = 1;
const int rpb_h = 2 * K - 1;
const int rpb_w = 2 * K - 1;
const int L = K * K;
int h = gid.z;
int ri = gid.y;
int rj = gid.x;
if (h >= heads || ri >= rpb_h || rj >= rpb_w) return;
int dil = (int)dilation_param[0];
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < height; i++) {
        int pi = pi_arr[i];
        int ni = ni_arr[i];
        int ei = ei_arr[i];
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dil;
        if (key_i < 0 || key_i >= ei) continue;
        int j = 0;
        for (; j + 1 < width; j += 2) {
            int pj0 = pj_arr[j];
            int nj0 = nj_arr[j];
            int ej0 = ej_arr[j];
            int kj0 = rj - pj0;
            if (kj0 >= 0 && kj0 < K) {
                int key_j0 = nj0 + kj0 * dil;
                if (key_j0 >= 0 && key_j0 < ej0) {
                    int neighbor_idx0 = ki * K + kj0;
                    int da_idx0 = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx0;
                    sum += d_attn[da_idx0];
                }
            }
            int j1 = j + 1;
            int pj1 = pj_arr[j1];
            int nj1 = nj_arr[j1];
            int ej1 = ej_arr[j1];
            int kj1 = rj - pj1;
            if (kj1 >= 0 && kj1 < K) {
                int key_j1 = nj1 + kj1 * dil;
                if (key_j1 >= 0 && key_j1 < ej1) {
                    int neighbor_idx1 = ki * K + kj1;
                    int da_idx1 = (((b * heads + h) * height + i) * width + j1) * L + neighbor_idx1;
                    sum += d_attn[da_idx1];
                }
            }
        }
        for (; j < width; j++) {
            int pj = pj_arr[j];
            int nj = nj_arr[j];
            int ej = ej_arr[j];
            int kj = rj - pj;
            if (kj < 0 || kj >= K) continue;
            int key_j = nj + kj * dil;
            if (key_j < 0 || key_j >= ej) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            sum += d_attn[da_idx];
        }
    }
}
int out_idx = (h * rpb_h + ri) * rpb_w + rj;
out[out_idx] = sum;

'''

NATTEN_K3_QK_BWD_DRPB_FAST_V4_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_rpb FAST (precomputed pi/pj/ni/nj/ei/ej) (K=3) vectorized
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int K = 3;
const int NH = 1;
const int rpb_h = 2 * K - 1;
const int rpb_w = 2 * K - 1;
const int L = K * K;
int h = gid.z;
int ri = gid.y;
int rj = gid.x;
if (h >= heads || ri >= rpb_h || rj >= rpb_w) return;
int dil = (int)dilation_param[0];
float4 acc = float4(0.0f);
float sum_tail = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < height; i++) {
        int pi = pi_arr[i];
        int ni = ni_arr[i];
        int ei = ei_arr[i];
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dil;
        if (key_i < 0 || key_i >= ei) continue;
        int j = 0;
        for (; j + 3 < width; j += 4) {
            int pj0 = pj_arr[j];
            int nj0 = nj_arr[j];
            int ej0 = ej_arr[j];
            int kj0 = rj - pj0;
            if (kj0 >= 0 && kj0 < K) {
                int key_j0 = nj0 + kj0 * dil;
                if (key_j0 >= 0 && key_j0 < ej0) {
                    int neighbor_idx0 = ki * K + kj0;
                    int da_idx0 = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx0;
                    acc.x += d_attn[da_idx0];
                }
            }
            int j1 = j + 1;
            int pj1 = pj_arr[j1];
            int nj1 = nj_arr[j1];
            int ej1 = ej_arr[j1];
            int kj1 = rj - pj1;
            if (kj1 >= 0 && kj1 < K) {
                int key_j1 = nj1 + kj1 * dil;
                if (key_j1 >= 0 && key_j1 < ej1) {
                    int neighbor_idx1 = ki * K + kj1;
                    int da_idx1 = (((b * heads + h) * height + i) * width + j1) * L + neighbor_idx1;
                    acc.y += d_attn[da_idx1];
                }
            }
            int j2 = j + 2;
            int pj2 = pj_arr[j2];
            int nj2 = nj_arr[j2];
            int ej2 = ej_arr[j2];
            int kj2 = rj - pj2;
            if (kj2 >= 0 && kj2 < K) {
                int key_j2 = nj2 + kj2 * dil;
                if (key_j2 >= 0 && key_j2 < ej2) {
                    int neighbor_idx2 = ki * K + kj2;
                    int da_idx2 = (((b * heads + h) * height + i) * width + j2) * L + neighbor_idx2;
                    acc.z += d_attn[da_idx2];
                }
            }
            int j3 = j + 3;
            int pj3 = pj_arr[j3];
            int nj3 = nj_arr[j3];
            int ej3 = ej_arr[j3];
            int kj3 = rj - pj3;
            if (kj3 >= 0 && kj3 < K) {
                int key_j3 = nj3 + kj3 * dil;
                if (key_j3 >= 0 && key_j3 < ej3) {
                    int neighbor_idx3 = ki * K + kj3;
                    int da_idx3 = (((b * heads + h) * height + i) * width + j3) * L + neighbor_idx3;
                    acc.w += d_attn[da_idx3];
                }
            }
        }
        for (; j < width; j++) {
            int pj = pj_arr[j];
            int nj = nj_arr[j];
            int ej = ej_arr[j];
            int kj = rj - pj;
            if (kj < 0 || kj >= K) continue;
            int key_j = nj + kj * dil;
            if (key_j < 0 || key_j >= ej) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            sum_tail += d_attn[da_idx];
        }
    }
}
int out_idx = (h * rpb_h + ri) * rpb_w + rj;
out[out_idx] = (acc.x + acc.y + acc.z + acc.w) + sum_tail;

'''

NATTEN_K5_QK_BWD_DRPB_FAST_V4_SOURCE = NATTEN_K3_QK_BWD_DRPB_FAST_V4_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2").replace("L = K * K", "L = 25")
NATTEN_K7_QK_BWD_DRPB_FAST_V4_SOURCE = NATTEN_K3_QK_BWD_DRPB_FAST_V4_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3").replace("L = K * K", "L = 49")

NATTEN_K3_QK_BWD_DRPB_FAST_SPLIT_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_rpb FAST (precomputed pi/pj/ni/nj/ei/ej) (K=3) split over height
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int K = 3;
const int NH = 1;
const int rpb_h = 2 * K - 1;
const int rpb_w = 2 * K - 1;
const int L = K * K;
int h = gid.z;
int ri = gid.y;
int rj = gid.x;
if (h >= heads || ri >= rpb_h || rj >= rpb_w) return;
int split_start = split_param[0];
int split_end = split_param[1];
if (split_start < 0) split_start = 0;
if (split_end > height) split_end = height;
int dil = (int)dilation_param[0];
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = split_start; i < split_end; i++) {
        int pi = pi_arr[i];
        int ni = ni_arr[i];
        int ei = ei_arr[i];
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dil;
        if (key_i < 0 || key_i >= ei) continue;
        for (int j = 0; j < width; j++) {
            int pj = pj_arr[j];
            int nj = nj_arr[j];
            int ej = ej_arr[j];
            int kj = rj - pj;
            if (kj < 0 || kj >= K) continue;
            int key_j = nj + kj * dil;
            if (key_j < 0 || key_j >= ej) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
            sum += d_attn[da_idx];
        }
    }
}
int out_idx = (h * rpb_h + ri) * rpb_w + rj;
out[out_idx] = sum;

'''

NATTEN_K5_QK_BWD_DRPB_FAST_SPLIT_SOURCE = NATTEN_K3_QK_BWD_DRPB_FAST_SPLIT_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2").replace("L = K * K", "L = 25")
NATTEN_K7_QK_BWD_DRPB_FAST_SPLIT_SOURCE = NATTEN_K3_QK_BWD_DRPB_FAST_SPLIT_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3").replace("L = K * K", "L = 49")

NATTEN_1D_K3_QK_BWD_DRPB_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_rpb FAST (precomputed pi/ni/ei) (K=3)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int K = 3;
const int NH = 1;
const int rpb_len = 2 * K - 1;
int h = gid.z;
int ri = gid.x;
if (h >= heads || ri >= rpb_len) return;
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < length; i++) {
        int pi = pi_arr[i];
        int ni = ni_arr[i];
        int ei = ei_arr[i];
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dilation_param[0];
        if (key_i < 0 || key_i >= ei) continue;
        int da_idx = (((b * heads + h) * length + i) * K + ki);
        sum += d_attn[da_idx];
    }
}
int out_idx = h * rpb_len + ri;
out[out_idx] = sum;

'''

NATTEN_K5_QK_BWD_DRPB_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_rpb FAST (precomputed pi/pj/ni/nj/ei/ej) (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int K = 5;
const int NH = 2;
const int rpb_h = 2 * K - 1;
const int rpb_w = 2 * K - 1;
int h = gid.z;
int ri = gid.y;
int rj = gid.x;
if (h >= heads || ri >= rpb_h || rj >= rpb_w) return;
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < height; i++) {
        int pi = pi_arr[i];
        int ni = ni_arr[i];
        int ei = ei_arr[i];
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dilation_param[0];
        if (key_i < 0 || key_i >= ei) continue;
        for (int j = 0; j < width; j++) {
            int pj = pj_arr[j];
            int nj = nj_arr[j];
            int ej = ej_arr[j];
            int kj = rj - pj;
            if (kj < 0 || kj >= K) continue;
            int key_j = nj + kj * dilation_param[0];
            if (key_j < 0 || key_j >= ej) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * (K*K) + neighbor_idx;
            sum += d_attn[da_idx];
        }
    }
}
int out_idx = (h * rpb_h + ri) * rpb_w + rj;
out[out_idx] = sum;

'''

NATTEN_1D_K5_QK_BWD_DRPB_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_rpb FAST (precomputed pi/ni/ei) (K=5)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int K = 5;
const int NH = 2;
const int rpb_len = 2 * K - 1;
int h = gid.z;
int ri = gid.x;
if (h >= heads || ri >= rpb_len) return;
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < length; i++) {
        int pi = pi_arr[i];
        int ni = ni_arr[i];
        int ei = ei_arr[i];
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dilation_param[0];
        if (key_i < 0 || key_i >= ei) continue;
        int da_idx = (((b * heads + h) * length + i) * K + ki);
        sum += d_attn[da_idx];
    }
}
int out_idx = h * rpb_len + ri;
out[out_idx] = sum;

'''

NATTEN_K7_QK_BWD_DRPB_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_rpb FAST (precomputed pi/pj/ni/nj/ei/ej) (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int K = 7;
const int NH = 3;
const int rpb_h = 2 * K - 1;
const int rpb_w = 2 * K - 1;
int h = gid.z;
int ri = gid.y;
int rj = gid.x;
if (h >= heads || ri >= rpb_h || rj >= rpb_w) return;
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < height; i++) {
        int pi = pi_arr[i];
        int ni = ni_arr[i];
        int ei = ei_arr[i];
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dilation_param[0];
        if (key_i < 0 || key_i >= ei) continue;
        for (int j = 0; j < width; j++) {
            int pj = pj_arr[j];
            int nj = nj_arr[j];
            int ej = ej_arr[j];
            int kj = rj - pj;
            if (kj < 0 || kj >= K) continue;
            int key_j = nj + kj * dilation_param[0];
            if (key_j < 0 || key_j >= ej) continue;
            int neighbor_idx = ki * K + kj;
            int da_idx = (((b * heads + h) * height + i) * width + j) * (K*K) + neighbor_idx;
            sum += d_attn[da_idx];
        }
    }
}
int out_idx = (h * rpb_h + ri) * rpb_w + rj;
out[out_idx] = sum;

'''

NATTEN_K5_QK_BWD_DRPB_FAST_U2_SOURCE = NATTEN_K3_QK_BWD_DRPB_FAST_U2_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2")
NATTEN_K7_QK_BWD_DRPB_FAST_U2_SOURCE = NATTEN_K3_QK_BWD_DRPB_FAST_U2_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3")

NATTEN_1D_K7_QK_BWD_DRPB_FAST_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_rpb FAST (precomputed pi/ni/ei) (K=7)
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int K = 7;
const int NH = 3;
const int rpb_len = 2 * K - 1;
int h = gid.z;
int ri = gid.x;
if (h >= heads || ri >= rpb_len) return;
float sum = 0.0f;
for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < length; i++) {
        int pi = pi_arr[i];
        int ni = ni_arr[i];
        int ei = ei_arr[i];
        int ki = ri - pi;
        if (ki < 0 || ki >= K) continue;
        int key_i = ni + ki * dilation_param[0];
        if (key_i < 0 || key_i >= ei) continue;
        int da_idx = (((b * heads + h) * length + i) * K + ki);
        sum += d_attn[da_idx];
    }
}
int out_idx = h * rpb_len + ri;
out[out_idx] = sum;

'''


# Backward kernels (TG tiled dQ/dAttn, dilation=1)
NATTEN_K3_QK_BWD_DQ_TG_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_query TG tiled (K=3) - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dim = query_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int L = 9;
const int TILE = 8;
const int DTILE = 4;
threadgroup float4 key_tile[TILE + 2*NH][TILE + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.y * TILE;
int base_j = tg_gid.x * TILE;
int li = (int)tg_pos.y;
int lj = (int)tg_pos.x;
int gi = base_i + li - NH;
int gj = base_j + lj - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 kval = float4(0.0f);
    if (gi >= 0 && gi < height && gj >= 0 && gj < width) {
        int k_base = (((b * heads + h) * height + gi) * width + gj) * dim + d0;
        float4 tmp = float4(0.0f);
        if (d0 + 0 < dim) tmp.x = key[k_base + 0];
        if (d0 + 1 < dim) tmp.y = key[k_base + 1];
        if (d0 + 2 < dim) tmp.z = key[k_base + 2];
        if (d0 + 3 < dim) tmp.w = key[k_base + 3];
        kval = tmp;
    }
    key_tile[li][lj] = kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
        int i = base_i + (li - NH);
        int j = base_j + (lj - NH);
        if (i < height && j < width) {
            int ni, nj, ei, ej;
            NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);

            float4 sum4 = float4(0.0f);
            int neighbor_idx = 0;
            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {
                    int key_i = ni + ki;
                    int key_j = nj + kj;
                    if (key_i < 0 || key_i >= ei || key_j < 0 || key_j >= ej) {
                        neighbor_idx++;
                        continue;
                    }
                    int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
                    float w = d_attn[da_idx];
                    int ti = key_i - base_i + NH;
                    int tj = key_j - base_j + NH;
                    float4 kval2 = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2 * NH) && tj >= 0 && tj < (TILE + 2 * NH)) {
                        kval2 = key_tile[ti][tj];
                    } else {
                        int k_base2 = (((b * heads + h) * height + key_i) * width + key_j) * dim + d0;
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 0 < dim) tmp2.x = key[k_base2 + 0];
                        if (d0 + 1 < dim) tmp2.y = key[k_base2 + 1];
                        if (d0 + 2 < dim) tmp2.z = key[k_base2 + 2];
                        if (d0 + 3 < dim) tmp2.w = key[k_base2 + 3];
                        kval2 = tmp2;
                    }
                    sum4 += w * kval2;
                    neighbor_idx++;
                }
            }
            int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d0;
            if (d0 + 0 < dim) out[out_idx + 0] = sum4.x;
            if (d0 + 1 < dim) out[out_idx + 1] = sum4.y;
            if (d0 + 2 < dim) out[out_idx + 2] = sum4.z;
            if (d0 + 3 < dim) out[out_idx + 3] = sum4.w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

'''

NATTEN_K3_AV_BWD_DATTN_TG_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_attn TG tiled (K=3) - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int height = d_out_shape[2];
const int width = d_out_shape[3];
const int dim = d_out_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int L = 9;
const int TILE = 16;
const int DTILE = 4;
threadgroup float4 val_tile[TILE + 2*NH][TILE + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.y * TILE;
int base_j = tg_gid.x * TILE;
int li = (int)tg_pos.y;
int lj = (int)tg_pos.x;
int gi = base_i + li - NH;
int gj = base_j + lj - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
// Initialize output for this (i,j) if inside tile
if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
    int i = base_i + (li - NH);
    int j = base_j + (lj - NH);
    if (i < height && j < width) {
        int out_base = (((b * heads + h) * height + i) * width + j) * L;
        for (int n = 0; n < L; n++) out[out_base + n] = 0.0f;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 vval = float4(0.0f);
    if (gi >= 0 && gi < height && gj >= 0 && gj < width) {
        int v_base = (((b * heads + h) * height + gi) * width + gj) * dim + d0;
        float4 tmp = float4(0.0f);
        if (d0 + 0 < dim) tmp.x = value[v_base + 0];
        if (d0 + 1 < dim) tmp.y = value[v_base + 1];
        if (d0 + 2 < dim) tmp.z = value[v_base + 2];
        if (d0 + 3 < dim) tmp.w = value[v_base + 3];
        vval = tmp;
    }
    val_tile[li][lj] = vval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
        int i = base_i + (li - NH);
        int j = base_j + (lj - NH);
        if (i < height && j < width) {
            int ni, nj, ei, ej;
            NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            int do_base = (((b * heads + h) * height + i) * width + j) * dim + d0;
            float4 dout = float4(0.0f);
            if (d0 + 0 < dim) dout.x = d_out[do_base + 0];
            if (d0 + 1 < dim) dout.y = d_out[do_base + 1];
            if (d0 + 2 < dim) dout.z = d_out[do_base + 2];
            if (d0 + 3 < dim) dout.w = d_out[do_base + 3];
            int neighbor_idx = 0;
            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {
                    int val_i = ni + ki;
                    int val_j = nj + kj;
                    if (val_i < 0 || val_i >= ei || val_j < 0 || val_j >= ej) {
                        neighbor_idx++;
                        continue;
                    }
                    if (neighbor_idx < split_start || neighbor_idx >= split_end) {
                        neighbor_idx++;
                        continue;
                    }
                    int ti = val_i - base_i + NH;
                    int tj = val_j - base_j + NH;
                    float4 v = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2 * NH) && tj >= 0 && tj < (TILE + 2 * NH)) {
                        v = val_tile[ti][tj];
                    } else {
                        int v_base2 = (((b * heads + h) * height + val_i) * width + val_j) * dim + d0;
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 0 < dim) tmp2.x = value[v_base2 + 0];
                        if (d0 + 1 < dim) tmp2.y = value[v_base2 + 1];
                        if (d0 + 2 < dim) tmp2.z = value[v_base2 + 2];
                        if (d0 + 3 < dim) tmp2.w = value[v_base2 + 3];
                        v = tmp2;
                    }
                    float sum = dot(v, dout);
                    int out_idx = (((b * heads + h) * height + i) * width + j) * split_len + (neighbor_idx - split_start);
                    out[out_idx] += sum;
                    neighbor_idx++;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

'''

NATTEN_K5_QK_BWD_DQ_TG_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_query TG tiled (K=5) - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dim = query_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
const int L = 25;
const int TILE = 8;
const int DTILE = 4;
threadgroup float4 key_tile[TILE + 2*NH][TILE + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.y * TILE;
int base_j = tg_gid.x * TILE;
int li = (int)tg_pos.y;
int lj = (int)tg_pos.x;
int gi = base_i + li - NH;
int gj = base_j + lj - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 kval = float4(0.0f);
    if (gi >= 0 && gi < height && gj >= 0 && gj < width) {
        int k_base = (((b * heads + h) * height + gi) * width + gj) * dim + d0;
        float4 tmp = float4(0.0f);
        if (d0 + 0 < dim) tmp.x = key[k_base + 0];
        if (d0 + 1 < dim) tmp.y = key[k_base + 1];
        if (d0 + 2 < dim) tmp.z = key[k_base + 2];
        if (d0 + 3 < dim) tmp.w = key[k_base + 3];
        kval = tmp;
    }
    key_tile[li][lj] = kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
        int i = base_i + (li - NH);
        int j = base_j + (lj - NH);
        if (i < height && j < width) {
            int ni, nj, ei, ej;
            NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);

            float4 sum4 = float4(0.0f);
            int neighbor_idx = 0;
            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {
                    int key_i = ni + ki;
                    int key_j = nj + kj;
                    if (key_i < 0 || key_i >= ei || key_j < 0 || key_j >= ej) {
                        neighbor_idx++;
                        continue;
                    }
                    int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
                    float w = d_attn[da_idx];
                    int ti = key_i - base_i + NH;
                    int tj = key_j - base_j + NH;
                    float4 kval2 = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2 * NH) && tj >= 0 && tj < (TILE + 2 * NH)) {
                        kval2 = key_tile[ti][tj];
                    } else {
                        int k_base2 = (((b * heads + h) * height + key_i) * width + key_j) * dim + d0;
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 0 < dim) tmp2.x = key[k_base2 + 0];
                        if (d0 + 1 < dim) tmp2.y = key[k_base2 + 1];
                        if (d0 + 2 < dim) tmp2.z = key[k_base2 + 2];
                        if (d0 + 3 < dim) tmp2.w = key[k_base2 + 3];
                        kval2 = tmp2;
                    }
                    sum4 += w * kval2;
                    neighbor_idx++;
                }
            }
            int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d0;
            if (d0 + 0 < dim) out[out_idx + 0] = sum4.x;
            if (d0 + 1 < dim) out[out_idx + 1] = sum4.y;
            if (d0 + 2 < dim) out[out_idx + 2] = sum4.z;
            if (d0 + 3 < dim) out[out_idx + 3] = sum4.w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

'''

NATTEN_K5_AV_BWD_DATTN_TG_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_attn TG tiled (K=5) - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int height = d_out_shape[2];
const int width = d_out_shape[3];
const int dim = d_out_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 5;
const int NH = 2;
const int L = 25;
const int TILE = 16;
const int DTILE = 4;
threadgroup float4 val_tile[TILE + 2*NH][TILE + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.y * TILE;
int base_j = tg_gid.x * TILE;
int li = (int)tg_pos.y;
int lj = (int)tg_pos.x;
int gi = base_i + li - NH;
int gj = base_j + lj - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
// Initialize output for this (i,j) if inside tile
if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
    int i = base_i + (li - NH);
    int j = base_j + (lj - NH);
    if (i < height && j < width) {
        int out_base = (((b * heads + h) * height + i) * width + j) * L;
        for (int n = 0; n < L; n++) out[out_base + n] = 0.0f;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 vval = float4(0.0f);
    if (gi >= 0 && gi < height && gj >= 0 && gj < width) {
        int v_base = (((b * heads + h) * height + gi) * width + gj) * dim + d0;
        float4 tmp = float4(0.0f);
        if (d0 + 0 < dim) tmp.x = value[v_base + 0];
        if (d0 + 1 < dim) tmp.y = value[v_base + 1];
        if (d0 + 2 < dim) tmp.z = value[v_base + 2];
        if (d0 + 3 < dim) tmp.w = value[v_base + 3];
        vval = tmp;
    }
    val_tile[li][lj] = vval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
        int i = base_i + (li - NH);
        int j = base_j + (lj - NH);
        if (i < height && j < width) {
            int ni, nj, ei, ej;
            NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            int do_base = (((b * heads + h) * height + i) * width + j) * dim + d0;
            float4 dout = float4(0.0f);
            if (d0 + 0 < dim) dout.x = d_out[do_base + 0];
            if (d0 + 1 < dim) dout.y = d_out[do_base + 1];
            if (d0 + 2 < dim) dout.z = d_out[do_base + 2];
            if (d0 + 3 < dim) dout.w = d_out[do_base + 3];
            int neighbor_idx = 0;
            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {
                    int val_i = ni + ki;
                    int val_j = nj + kj;
                    if (val_i < 0 || val_i >= ei || val_j < 0 || val_j >= ej) {
                        neighbor_idx++;
                        continue;
                    }
                    int ti = val_i - base_i + NH;
                    int tj = val_j - base_j + NH;
                    float4 v = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2 * NH) && tj >= 0 && tj < (TILE + 2 * NH)) {
                        v = val_tile[ti][tj];
                    } else {
                        int v_base2 = (((b * heads + h) * height + val_i) * width + val_j) * dim + d0;
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 0 < dim) tmp2.x = value[v_base2 + 0];
                        if (d0 + 1 < dim) tmp2.y = value[v_base2 + 1];
                        if (d0 + 2 < dim) tmp2.z = value[v_base2 + 2];
                        if (d0 + 3 < dim) tmp2.w = value[v_base2 + 3];
                        v = tmp2;
                    }
                    float sum = dot(v, dout);
                    int out_idx = (((b * heads + h) * height + i) * width + j) * split_len + (neighbor_idx - split_start);
                    out[out_idx] += sum;
                    neighbor_idx++;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

'''

NATTEN_K7_QK_BWD_DQ_TG_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D QK backward d_query TG tiled (K=7) - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dim = query_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
const int L = 49;
const int TILE = 8;
const int DTILE = 4;
threadgroup float4 key_tile[TILE + 2*NH][TILE + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.y * TILE;
int base_j = tg_gid.x * TILE;
int li = (int)tg_pos.y;
int lj = (int)tg_pos.x;
int gi = base_i + li - NH;
int gj = base_j + lj - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 kval = float4(0.0f);
    if (gi >= 0 && gi < height && gj >= 0 && gj < width) {
        int k_base = (((b * heads + h) * height + gi) * width + gj) * dim + d0;
        float4 tmp = float4(0.0f);
        if (d0 + 0 < dim) tmp.x = key[k_base + 0];
        if (d0 + 1 < dim) tmp.y = key[k_base + 1];
        if (d0 + 2 < dim) tmp.z = key[k_base + 2];
        if (d0 + 3 < dim) tmp.w = key[k_base + 3];
        kval = tmp;
    }
    key_tile[li][lj] = kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
        int i = base_i + (li - NH);
        int j = base_j + (lj - NH);
        if (i < height && j < width) {
            int ni, nj, ei, ej;
            NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);

            float4 sum4 = float4(0.0f);
            int neighbor_idx = 0;
            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {
                    int key_i = ni + ki;
                    int key_j = nj + kj;
                    if (key_i < 0 || key_i >= ei || key_j < 0 || key_j >= ej) {
                        neighbor_idx++;
                        continue;
                    }
                    int da_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
                    float w = d_attn[da_idx];
                    int ti = key_i - base_i + NH;
                    int tj = key_j - base_j + NH;
                    float4 kval2 = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2 * NH) && tj >= 0 && tj < (TILE + 2 * NH)) {
                        kval2 = key_tile[ti][tj];
                    } else {
                        int k_base2 = (((b * heads + h) * height + key_i) * width + key_j) * dim + d0;
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 0 < dim) tmp2.x = key[k_base2 + 0];
                        if (d0 + 1 < dim) tmp2.y = key[k_base2 + 1];
                        if (d0 + 2 < dim) tmp2.z = key[k_base2 + 2];
                        if (d0 + 3 < dim) tmp2.w = key[k_base2 + 3];
                        kval2 = tmp2;
                    }
                    sum4 += w * kval2;
                    neighbor_idx++;
                }
            }
            int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d0;
            if (d0 + 0 < dim) out[out_idx + 0] = sum4.x;
            if (d0 + 1 < dim) out[out_idx + 1] = sum4.y;
            if (d0 + 2 < dim) out[out_idx + 2] = sum4.z;
            if (d0 + 3 < dim) out[out_idx + 3] = sum4.w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

'''

NATTEN_K7_AV_BWD_DATTN_TG_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_attn TG tiled (K=7) - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int height = d_out_shape[2];
const int width = d_out_shape[3];
const int dim = d_out_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 7;
const int NH = 3;
const int L = 49;
const int TILE = 16;
const int DTILE = 4;
threadgroup float4 val_tile[TILE + 2*NH][TILE + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.y * TILE;
int base_j = tg_gid.x * TILE;
int li = (int)tg_pos.y;
int lj = (int)tg_pos.x;
int gi = base_i + li - NH;
int gj = base_j + lj - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
// Initialize output for this (i,j) if inside tile
if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
    int i = base_i + (li - NH);
    int j = base_j + (lj - NH);
    if (i < height && j < width) {
        int out_base = (((b * heads + h) * height + i) * width + j) * L;
        for (int n = 0; n < L; n++) out[out_base + n] = 0.0f;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 vval = float4(0.0f);
    if (gi >= 0 && gi < height && gj >= 0 && gj < width) {
        int v_base = (((b * heads + h) * height + gi) * width + gj) * dim + d0;
        float4 tmp = float4(0.0f);
        if (d0 + 0 < dim) tmp.x = value[v_base + 0];
        if (d0 + 1 < dim) tmp.y = value[v_base + 1];
        if (d0 + 2 < dim) tmp.z = value[v_base + 2];
        if (d0 + 3 < dim) tmp.w = value[v_base + 3];
        vval = tmp;
    }
    val_tile[li][lj] = vval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
        int i = base_i + (li - NH);
        int j = base_j + (lj - NH);
        if (i < height && j < width) {
            int ni, nj, ei, ej;
            NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            int do_base = (((b * heads + h) * height + i) * width + j) * dim + d0;
            float4 dout = float4(0.0f);
            if (d0 + 0 < dim) dout.x = d_out[do_base + 0];
            if (d0 + 1 < dim) dout.y = d_out[do_base + 1];
            if (d0 + 2 < dim) dout.z = d_out[do_base + 2];
            if (d0 + 3 < dim) dout.w = d_out[do_base + 3];
            int neighbor_idx = 0;
            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {
                    int val_i = ni + ki;
                    int val_j = nj + kj;
                    if (val_i < 0 || val_i >= ei || val_j < 0 || val_j >= ej) {
                        neighbor_idx++;
                        continue;
                    }
                    int ti = val_i - base_i + NH;
                    int tj = val_j - base_j + NH;
                    float4 v = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2 * NH) && tj >= 0 && tj < (TILE + 2 * NH)) {
                        v = val_tile[ti][tj];
                    } else {
                        int v_base2 = (((b * heads + h) * height + val_i) * width + val_j) * dim + d0;
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 0 < dim) tmp2.x = value[v_base2 + 0];
                        if (d0 + 1 < dim) tmp2.y = value[v_base2 + 1];
                        if (d0 + 2 < dim) tmp2.z = value[v_base2 + 2];
                        if (d0 + 3 < dim) tmp2.w = value[v_base2 + 3];
                        v = tmp2;
                    }
                    float sum = v.x * dout.x + v.y * dout.y + v.z * dout.z + v.w * dout.w;
                    int out_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
                    out[out_idx] += sum;
                    neighbor_idx++;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

'''

NATTEN_K3_AV_BWD_DATTN_SPLIT_TG_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_attn TG tiled (K=3) split keys - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int height = d_out_shape[2];
const int width = d_out_shape[3];
const int dim = d_out_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int L = 9;
const int split_start = (int)split_param[0];
const int split_len = (int)split_param[1];
const int TILE = 16;
const int DTILE = 4;
threadgroup float4 val_tile[TILE + 2*NH][TILE + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.y * TILE;
int base_j = tg_gid.x * TILE;
int li = (int)tg_pos.y;
int lj = (int)tg_pos.x;
int gi = base_i + li - NH;
int gj = base_j + lj - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
// Initialize output for this (i,j) if inside tile
if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
    int i = base_i + (li - NH);
    int j = base_j + (lj - NH);
    if (i < height && j < width) {
        int out_base = (((b * heads + h) * height + i) * width + j) * split_len;
        for (int n = 0; n < split_len; n++) out[out_base + n] = 0.0f;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 vval = float4(0.0f);
    if (gi >= 0 && gi < height && gj >= 0 && gj < width) {
        int v_base = (((b * heads + h) * height + gi) * width + gj) * dim + d0;
        float4 tmp = float4(0.0f);
        if (d0 + 0 < dim) tmp.x = value[v_base + 0];
        if (d0 + 1 < dim) tmp.y = value[v_base + 1];
        if (d0 + 2 < dim) tmp.z = value[v_base + 2];
        if (d0 + 3 < dim) tmp.w = value[v_base + 3];
        vval = tmp;
    }
    val_tile[li][lj] = vval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
        int i = base_i + (li - NH);
        int j = base_j + (lj - NH);
        if (i < height && j < width) {
            int ni, nj, ei, ej;
            NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            int do_base = (((b * heads + h) * height + i) * width + j) * dim + d0;
            float4 dout = float4(0.0f);
            if (d0 + 0 < dim) dout.x = d_out[do_base + 0];
            if (d0 + 1 < dim) dout.y = d_out[do_base + 1];
            if (d0 + 2 < dim) dout.z = d_out[do_base + 2];
            if (d0 + 3 < dim) dout.w = d_out[do_base + 3];
            int neighbor_idx = 0;
            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {
                    int out_local = neighbor_idx - split_start;
                    if (out_local < 0 || out_local >= split_len) {
                        neighbor_idx++;
                        continue;
                    }
                    int val_i = ni + ki;
                    int val_j = nj + kj;
                    if (val_i < 0 || val_i >= ei || val_j < 0 || val_j >= ej) {
                        neighbor_idx++;
                        continue;
                    }
                    int ti = val_i - base_i + NH;
                    int tj = val_j - base_j + NH;
                    float4 v = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2 * NH) && tj >= 0 && tj < (TILE + 2 * NH)) {
                        v = val_tile[ti][tj];
                    } else {
                        int v_base2 = (((b * heads + h) * height + val_i) * width + val_j) * dim + d0;
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 0 < dim) tmp2.x = value[v_base2 + 0];
                        if (d0 + 1 < dim) tmp2.y = value[v_base2 + 1];
                        if (d0 + 2 < dim) tmp2.z = value[v_base2 + 2];
                        if (d0 + 3 < dim) tmp2.w = value[v_base2 + 3];
                        v = tmp2;
                    }
                    float sum = v.x * dout.x + v.y * dout.y + v.z * dout.z + v.w * dout.w;
                    int out_idx = (((b * heads + h) * height + i) * width + j) * split_len + out_local;
                    out[out_idx] += sum;
                    neighbor_idx++;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

'''

NATTEN_K5_AV_BWD_DATTN_SPLIT_TG_SOURCE = NATTEN_K3_AV_BWD_DATTN_SPLIT_TG_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2").replace("L = 9", "L = 25")
NATTEN_K7_AV_BWD_DATTN_SPLIT_TG_SOURCE = NATTEN_K3_AV_BWD_DATTN_SPLIT_TG_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3").replace("L = 9", "L = 49")

NATTEN_K3_AV_BWD_DATTN_TG_T8_SOURCE = NATTEN_K3_AV_BWD_DATTN_TG_SOURCE.replace("TILE = 16", "TILE = 8")
NATTEN_K5_AV_BWD_DATTN_TG_T8_SOURCE = NATTEN_K5_AV_BWD_DATTN_TG_SOURCE.replace("TILE = 16", "TILE = 8")
NATTEN_K7_AV_BWD_DATTN_TG_T8_SOURCE = NATTEN_K7_AV_BWD_DATTN_TG_SOURCE.replace("TILE = 16", "TILE = 8")
NATTEN_K3_AV_BWD_DATTN_SPLIT_TG_T8_SOURCE = NATTEN_K3_AV_BWD_DATTN_SPLIT_TG_SOURCE.replace("TILE = 16", "TILE = 8")
NATTEN_K5_AV_BWD_DATTN_SPLIT_TG_T8_SOURCE = NATTEN_K5_AV_BWD_DATTN_SPLIT_TG_SOURCE.replace("TILE = 16", "TILE = 8")
NATTEN_K7_AV_BWD_DATTN_SPLIT_TG_T8_SOURCE = NATTEN_K7_AV_BWD_DATTN_SPLIT_TG_SOURCE.replace("TILE = 16", "TILE = 8")

# 2D AV backward d_value TG tiled (K=3/5/7) - dilation=1 only
NATTEN_K3_AV_BWD_DVAL_TG_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward d_value TG tiled (K=3) - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = value_shape[0];
const int heads = value_shape[1];
const int height = value_shape[2];
const int width = value_shape[3];
const int dim = value_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int TILE = 16;
const int DTILE = 4;
threadgroup float4 dout_tile[TILE + 2*NH][TILE + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.y * TILE;
int base_j = tg_gid.x * TILE;
int li = (int)tg_pos.y;
int lj = (int)tg_pos.x;
int gi = base_i + li - NH;
int gj = base_j + lj - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 doval = float4(0.0f);
    if (gi >= 0 && gi < height && gj >= 0 && gj < width) {
        int do_base = (((b * heads + h) * height + gi) * width + gj) * dim + d0;
        float4 tmp = float4(0.0f);
        if (d0 + 0 < dim) tmp.x = d_out[do_base + 0];
        if (d0 + 1 < dim) tmp.y = d_out[do_base + 1];
        if (d0 + 2 < dim) tmp.z = d_out[do_base + 2];
        if (d0 + 3 < dim) tmp.w = d_out[do_base + 3];
        doval = tmp;
    }
    dout_tile[li][lj] = doval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
        int vi = base_i + (li - NH);
        int vj = base_j + (lj - NH);
        if (vi < height && vj < width) {
            float4 sum4 = float4(0.0f);
            // Interior: no i/j bounds checks, but still compute shifted window start per (i,j)
            int i_start = max(0, vi - (K - 1) * dilation);
            int i_end = min(height - 1, vi + (K - 1) * dilation);
            int j_start = max(0, vj - (K - 1) * dilation);
            int j_end = min(width - 1, vj + (K - 1) * dilation);
            for (int i = i_start; i <= i_end; i++) {
                int ni, ei;
                NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
                NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
                if (vi < ni || vi >= ei) continue;
                int di = vi - ni;
                if (di % dilation != 0) continue;
                int ki = di / dilation;
                for (int j = j_start; j <= j_end; j++) {
                    int nj, ej;
                    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
                    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
                    if (vj < nj || vj >= ej) continue;
                    int dj = vj - nj;
                    if (dj % dilation != 0) continue;
                    int kj = dj / dilation;
                    int neighbor_idx = ki * K + kj;
                    int attn_idx = (((b * heads + h) * height + i) * width + j) * (K*K) + neighbor_idx;
                    float attn_val = attn[attn_idx];
                    int ti = i - base_i + NH;
                    int tj = j - base_j + NH;
                    float4 do_v = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2*NH) && tj >= 0 && tj < (TILE + 2*NH)) {
                        do_v = dout_tile[ti][tj];
                    } else {
                        int do_base2 = (((b * heads + h) * height + i) * width + j) * dim + d0;
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 0 < dim) tmp2.x = d_out[do_base2 + 0];
                        if (d0 + 1 < dim) tmp2.y = d_out[do_base2 + 1];
                        if (d0 + 2 < dim) tmp2.z = d_out[do_base2 + 2];
                        if (d0 + 3 < dim) tmp2.w = d_out[do_base2 + 3];
                        do_v = tmp2;
                    }
                    sum4 = fma(attn_val, do_v, sum4);
                }
            }
            int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d0;
            if (d0 + 0 < dim) out[out_idx + 0] = sum4.x;
            if (d0 + 1 < dim) out[out_idx + 1] = sum4.y;
            if (d0 + 2 < dim) out[out_idx + 2] = sum4.z;
            if (d0 + 3 < dim) out[out_idx + 3] = sum4.w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

'''

NATTEN_K5_AV_BWD_DVAL_TG_SOURCE = NATTEN_K3_AV_BWD_DVAL_TG_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2")
NATTEN_K7_AV_BWD_DVAL_TG_SOURCE = NATTEN_K3_AV_BWD_DVAL_TG_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3")

def _av_bwd_fused_unroll2_source(source: str) -> str:
    marker = "for (int d0 = 0; d0 < dim; d0 += DTILE) {"
    if marker not in source:
        return source
    prefix, rest = source.split(marker, 1)
    m = re.search(r"\n\\s*// d_value for value positions", rest)
    if m is None:
        return source
    loop_body = rest[: m.start()]
    dvalue_and_after = rest[m.start() :]
    end_loop = dvalue_and_after.find("\n}\n\n")
    if end_loop == -1:
        return source
    dvalue_block = dvalue_and_after[:end_loop]
    suffix = dvalue_and_after[end_loop + 3 :]
    # Replace d0 with d1 safely (avoid d_out, d0_base, etc.)
    body_d1 = re.sub(r"\bd0\b", "d1", loop_body)
    body_d1 = "\n".join(("    " + line) if line else "" for line in body_d1.splitlines())
    # Unroll d_attn portion by 2, then keep d_value in its own loop.
    unrolled = (
        "for (int d0_base = 0; d0_base < dim; d0_base += DTILE * 2) {\n"
        "    int d0 = d0_base;\n"
        f"{loop_body}\n"
        "    int d1 = d0_base + DTILE;\n"
        "    if (d1 < dim) {\n"
        f"{body_d1}\n"
        "    }\n"
        "}\n\n"
        "for (int d0 = 0; d0 < dim; d0 += DTILE) {"
        f"{dvalue_block}\n"
        "}\n"
    )
    return prefix + unrolled + suffix

NATTEN_K3_AV_BWD_FUSED_TG_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward fused d_attn + d_value TG tiled (K=3) - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int height = d_out_shape[2];
const int width = d_out_shape[3];
const int dim = d_out_shape[4];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int L = 9;
const int TILE_H = 16;
const int TILE_W = 16;
const int DTILE = 4;
threadgroup float4 val_tile[2][TILE_H + 2*NH][TILE_W + 2*NH];
threadgroup float4 dout_tile[2][TILE_H + 2*NH][TILE_W + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.y * TILE_H;
int base_j = tg_gid.x * TILE_W;
int li = (int)tg_pos.y;
int lj = (int)tg_pos.x;
int gi = base_i + li - NH;
int gj = base_j + lj - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
// Initialize d_attn output for this (i,j) if inside tile
if (li >= NH && li < NH + TILE_H && lj >= NH && lj < NH + TILE_W) {
    int i = base_i + (li - NH);
    int j = base_j + (lj - NH);
    if (i < height && j < width) {
        int out_base = (((b * heads + h) * height + i) * width + j) * L;
        for (int n = 0; n < L; n++) out_attn[out_base + n] = 0.0f;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Preload first tile into buffer 0.
float4 vval0 = float4(0.0f);
float4 doval0 = float4(0.0f);
if (gi >= 0 && gi < height && gj >= 0 && gj < width) {
    int v_base0 = (((b * heads + h) * height + gi) * width + gj) * dim + 0;
    float4 tmp0 = float4(0.0f);
    if (3 < dim) {
        tmp0 = *((device const float4*)(value + v_base0));
    } else {
        if (0 < dim) tmp0.x = value[v_base0 + 0];
        if (1 < dim) tmp0.y = value[v_base0 + 1];
        if (2 < dim) tmp0.z = value[v_base0 + 2];
        if (3 < dim) tmp0.w = value[v_base0 + 3];
    }
    vval0 = tmp0;
    int do_base0 = (((b * heads + h) * height + gi) * width + gj) * dim + 0;
    float4 tmpd0 = float4(0.0f);
    if (3 < dim) {
        tmpd0 = *((device const float4*)(d_out + do_base0));
    } else {
        if (0 < dim) tmpd0.x = d_out[do_base0 + 0];
        if (1 < dim) tmpd0.y = d_out[do_base0 + 1];
        if (2 < dim) tmpd0.z = d_out[do_base0 + 2];
        if (3 < dim) tmpd0.w = d_out[do_base0 + 3];
    }
    doval0 = tmpd0;
}
val_tile[0][li][lj] = vval0;
dout_tile[0][li][lj] = doval0;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int d0 = 0; d0 < dim; d0 += DTILE) {
    int buf = (d0 / DTILE) & 1;
    int next_d0 = d0 + DTILE;
    if (next_d0 < dim) {
        float4 vval = float4(0.0f);
        float4 doval = float4(0.0f);
        if (gi >= 0 && gi < height && gj >= 0 && gj < width) {
            int v_base = (((b * heads + h) * height + gi) * width + gj) * dim + next_d0;
            float4 tmp = float4(0.0f);
            if (next_d0 + 3 < dim) {
                tmp = *((device const float4*)(value + v_base));
            } else {
                if (next_d0 + 0 < dim) tmp.x = value[v_base + 0];
                if (next_d0 + 1 < dim) tmp.y = value[v_base + 1];
                if (next_d0 + 2 < dim) tmp.z = value[v_base + 2];
                if (next_d0 + 3 < dim) tmp.w = value[v_base + 3];
            }
            vval = tmp;
            int do_base = (((b * heads + h) * height + gi) * width + gj) * dim + next_d0;
            float4 tmpd = float4(0.0f);
            if (next_d0 + 3 < dim) {
                tmpd = *((device const float4*)(d_out + do_base));
            } else {
                if (next_d0 + 0 < dim) tmpd.x = d_out[do_base + 0];
                if (next_d0 + 1 < dim) tmpd.y = d_out[do_base + 1];
                if (next_d0 + 2 < dim) tmpd.z = d_out[do_base + 2];
                if (next_d0 + 3 < dim) tmpd.w = d_out[do_base + 3];
            }
            doval = tmpd;
        }
        val_tile[buf ^ 1][li][lj] = vval;
        dout_tile[buf ^ 1][li][lj] = doval;
    }

    // d_attn for query positions
    if (li >= NH && li < NH + TILE_H && lj >= NH && lj < NH + TILE_W) {
        int i = base_i + (li - NH);
        int j = base_j + (lj - NH);
        if (i < height && j < width) {
            int ni, nj, ei, ej;
            NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            int do_base = (((b * heads + h) * height + i) * width + j) * dim + d0;
            float4 dout = float4(0.0f);
            if (d0 + 3 < dim) {
                dout = *((device const float4*)(d_out + do_base));
            } else {
                if (d0 + 0 < dim) dout.x = d_out[do_base + 0];
                if (d0 + 1 < dim) dout.y = d_out[do_base + 1];
                if (d0 + 2 < dim) dout.z = d_out[do_base + 2];
                if (d0 + 3 < dim) dout.w = d_out[do_base + 3];
            }
            int neighbor_idx = 0;
            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {
                    int val_i = ni + ki;
                    int val_j = nj + kj;
                    if (val_i < 0 || val_i >= ei || val_j < 0 || val_j >= ej) {
                        neighbor_idx++;
                        continue;
                    }
                    int ti = val_i - base_i + NH;
                    int tj = val_j - base_j + NH;
                    float4 v = float4(0.0f);
                    if (ti >= 0 && ti < (TILE_H + 2*NH) && tj >= 0 && tj < (TILE_W + 2*NH)) {
                        v = val_tile[buf][ti][tj];
                    } else {
                        int v_base2 = (((b * heads + h) * height + val_i) * width + val_j) * dim + d0;
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 3 < dim) {
                            tmp2 = *((device const float4*)(value + v_base2));
                        } else {
                            if (d0 + 0 < dim) tmp2.x = value[v_base2 + 0];
                            if (d0 + 1 < dim) tmp2.y = value[v_base2 + 1];
                            if (d0 + 2 < dim) tmp2.z = value[v_base2 + 2];
                            if (d0 + 3 < dim) tmp2.w = value[v_base2 + 3];
                        }
                        v = tmp2;
                    }
                    float sum = dot(v, dout);
                    int out_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
                    out_attn[out_idx] += sum;
                    neighbor_idx++;
                }
            }
        }
    }

    // d_value for value positions
    if (li >= NH && li < NH + TILE_H && lj >= NH && lj < NH + TILE_W) {
        int vi = base_i + (li - NH);
        int vj = base_j + (lj - NH);
        if (vi < height && vj < width) {
            float4 sum4 = float4(0.0f);
            int i_start = max(0, vi - (K - 1) * dilation);
            int i_end = min(height - 1, vi + (K - 1) * dilation);
            int j_start = max(0, vj - (K - 1) * dilation);
            int j_end = min(width - 1, vj + (K - 1) * dilation);
            for (int i = i_start; i <= i_end; i++) {
                int ni, ei;
                NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
                NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
                if (vi < ni || vi >= ei) continue;
                int di = vi - ni;
                if (di % dilation != 0) continue;
                int ki = di / dilation;
                for (int j = j_start; j <= j_end; j++) {
                    int nj, ej;
                    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
                    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
                    if (vj < nj || vj >= ej) continue;
                    int dj = vj - nj;
                    if (dj % dilation != 0) continue;
                    int kj = dj / dilation;
                    int neighbor_idx = ki * K + kj;
                    int attn_idx = (((b * heads + h) * height + i) * width + j) * (K*K) + neighbor_idx;
                    float attn_val = attn[attn_idx];
                    int ti = i - base_i + NH;
                    int tj = j - base_j + NH;
                    float4 do_v = float4(0.0f);
                    if (ti >= 0 && ti < (TILE_H + 2*NH) && tj >= 0 && tj < (TILE_W + 2*NH)) {
                        do_v = dout_tile[buf][ti][tj];
                    } else {
                        int do_base2 = (((b * heads + h) * height + i) * width + j) * dim + d0;
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 3 < dim) {
                            tmp2 = *((device const float4*)(d_out + do_base2));
                        } else {
                            if (d0 + 0 < dim) tmp2.x = d_out[do_base2 + 0];
                            if (d0 + 1 < dim) tmp2.y = d_out[do_base2 + 1];
                            if (d0 + 2 < dim) tmp2.z = d_out[do_base2 + 2];
                            if (d0 + 3 < dim) tmp2.w = d_out[do_base2 + 3];
                        }
                        do_v = tmp2;
                    }
                    sum4 = fma(attn_val, do_v, sum4);
                }
            }
            int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d0;
            if (d0 + 0 < dim) out_val[out_idx + 0] = sum4.x;
            if (d0 + 1 < dim) out_val[out_idx + 1] = sum4.y;
            if (d0 + 2 < dim) out_val[out_idx + 2] = sum4.z;
            if (d0 + 3 < dim) out_val[out_idx + 3] = sum4.w;
        }
    }
    if (next_d0 < dim) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

'''

NATTEN_1D_K3_AV_BWD_DATTN_TG_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D AV backward d_attn TG tiled (K=3) - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int length = d_out_shape[2];
const int dim = d_out_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int TILE = 64;
const int DTILE = 4;
threadgroup float4 val_tile[TILE + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.x * TILE;
int li = (int)tg_pos.x;
int gi = base_i + li - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
// Initialize output for this i
if (li >= NH && li < NH + TILE) {
    int i = base_i + (li - NH);
    if (i < length) {
        int out_base = (((b * heads + h) * length + i) * K);
        for (int n = 0; n < K; n++) out[out_base + n] = 0.0f;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 vval = float4(0.0f);
    if (gi >= 0 && gi < length) {
        int v_base = (((b * heads + h) * length + gi) * dim + d0);
        float4 tmp = float4(0.0f);
        if (d0 + 0 < dim) tmp.x = value[v_base + 0];
        if (d0 + 1 < dim) tmp.y = value[v_base + 1];
        if (d0 + 2 < dim) tmp.z = value[v_base + 2];
        if (d0 + 3 < dim) tmp.w = value[v_base + 3];
        vval = tmp;
    }
    val_tile[li] = vval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (li >= NH && li < NH + TILE) {
        int i = base_i + (li - NH);
        if (i < length) {
            int do_base = (((b * heads + h) * length + i) * dim + d0);
            float4 dout = float4(0.0f);
            if (d0 + 0 < dim) dout.x = d_out[do_base + 0];
            if (d0 + 1 < dim) dout.y = d_out[do_base + 1];
            if (d0 + 2 < dim) dout.z = d_out[do_base + 2];
            if (d0 + 3 < dim) dout.w = d_out[do_base + 3];
            int neighbor_idx = 0;
            if (i >= NH && i < (length - NH)) {
                // Interior: full window, no bounds checks
                for (int ki = 0; ki < K; ki++) {
                    int val_i = i + (ki - NH);
                    int ti = val_i - base_i + NH;
                    float4 v = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2*NH)) {
                        v = val_tile[ti];
                    } else {
                        int v_base2 = (((b * heads + h) * length + val_i) * dim + d0);
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 0 < dim) tmp2.x = value[v_base2 + 0];
                        if (d0 + 1 < dim) tmp2.y = value[v_base2 + 1];
                        if (d0 + 2 < dim) tmp2.z = value[v_base2 + 2];
                        if (d0 + 3 < dim) tmp2.w = value[v_base2 + 3];
                        v = tmp2;
                    }
                    float sum = v.x * dout.x + v.y * dout.y + v.z * dout.z + v.w * dout.w;
                    int out_idx = (((b * heads + h) * length + i) * K + ki);
                    out[out_idx] += sum;
                }
            } else {
                // Edge: use bounds checks
                int ni, ei;
                NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
                NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
                for (int ki = 0; ki < K; ki++) {
                    int val_i = ni + ki;
                    if (val_i < 0 || val_i >= ei) {
                        neighbor_idx++;
                        continue;
                    }
                    int ti = val_i - base_i + NH;
                    float4 v = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2*NH)) {
                        v = val_tile[ti];
                    } else {
                        int v_base2 = (((b * heads + h) * length + val_i) * dim + d0);
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 0 < dim) tmp2.x = value[v_base2 + 0];
                        if (d0 + 1 < dim) tmp2.y = value[v_base2 + 1];
                        if (d0 + 2 < dim) tmp2.z = value[v_base2 + 2];
                        if (d0 + 3 < dim) tmp2.w = value[v_base2 + 3];
                        v = tmp2;
                    }
                    float sum = v.x * dout.x + v.y * dout.y + v.z * dout.z + v.w * dout.w;
                    int out_idx = (((b * heads + h) * length + i) * K + neighbor_idx);
                    out[out_idx] += sum;
                    neighbor_idx++;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

'''

NATTEN_1D_K3_AV_BWD_DATTN_TG_V4_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D AV backward d_attn TG tiled (K=3) vectorized - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int length = d_out_shape[2];
const int dim = d_out_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int TILE = 64;
const int DTILE = 4;
threadgroup float4 val_tile[TILE + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.x * TILE;
int li = (int)tg_pos.x;
int gi = base_i + li - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
// Initialize output for this i
if (li >= NH && li < NH + TILE) {
    int i = base_i + (li - NH);
    if (i < length) {
        int out_base = (((b * heads + h) * length + i) * K);
        for (int n = 0; n < K; n++) out[out_base + n] = 0.0f;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 vval = float4(0.0f);
    if (gi >= 0 && gi < length) {
        int v_base = (((b * heads + h) * length + gi) * dim + d0);
        float4 tmp = float4(0.0f);
        if (d0 + 3 < dim) {
            tmp = *((device const float4*)(value + v_base));
        } else {
            if (d0 + 0 < dim) tmp.x = value[v_base + 0];
            if (d0 + 1 < dim) tmp.y = value[v_base + 1];
            if (d0 + 2 < dim) tmp.z = value[v_base + 2];
            if (d0 + 3 < dim) tmp.w = value[v_base + 3];
        }
        vval = tmp;
    }
    val_tile[li] = vval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (li >= NH && li < NH + TILE) {
        int i = base_i + (li - NH);
        if (i < length) {
            int do_base = (((b * heads + h) * length + i) * dim + d0);
            float4 dout = float4(0.0f);
            if (d0 + 3 < dim) {
                dout = *((device const float4*)(d_out + do_base));
            } else {
                if (d0 + 0 < dim) dout.x = d_out[do_base + 0];
                if (d0 + 1 < dim) dout.y = d_out[do_base + 1];
                if (d0 + 2 < dim) dout.z = d_out[do_base + 2];
                if (d0 + 3 < dim) dout.w = d_out[do_base + 3];
            }
            int neighbor_idx = 0;
            if (i >= NH && i < (length - NH)) {
                // Interior: full window, no bounds checks
                for (int ki = 0; ki < K; ki++) {
                    int val_i = i + (ki - NH);
                    int ti = val_i - base_i + NH;
                    float4 v = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2*NH)) {
                        v = val_tile[ti];
                    } else {
                        int v_base2 = (((b * heads + h) * length + val_i) * dim + d0);
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 3 < dim) {
                            tmp2 = *((device const float4*)(value + v_base2));
                        } else {
                            if (d0 + 0 < dim) tmp2.x = value[v_base2 + 0];
                            if (d0 + 1 < dim) tmp2.y = value[v_base2 + 1];
                            if (d0 + 2 < dim) tmp2.z = value[v_base2 + 2];
                            if (d0 + 3 < dim) tmp2.w = value[v_base2 + 3];
                        }
                        v = tmp2;
                    }
                    float sum = dot(v, dout);
                    int out_idx = (((b * heads + h) * length + i) * K + ki);
                    out[out_idx] += sum;
                }
            } else {
                // Edge: use bounds checks
                int ni, ei;
                NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
                NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
                for (int ki = 0; ki < K; ki++) {
                    int val_i = ni + ki;
                    if (val_i < 0 || val_i >= ei) {
                        neighbor_idx++;
                        continue;
                    }
                    int ti = val_i - base_i + NH;
                    float4 v = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2*NH)) {
                        v = val_tile[ti];
                    } else {
                        int v_base2 = (((b * heads + h) * length + val_i) * dim + d0);
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 3 < dim) {
                            tmp2 = *((device const float4*)(value + v_base2));
                        } else {
                            if (d0 + 0 < dim) tmp2.x = value[v_base2 + 0];
                            if (d0 + 1 < dim) tmp2.y = value[v_base2 + 1];
                            if (d0 + 2 < dim) tmp2.z = value[v_base2 + 2];
                            if (d0 + 3 < dim) tmp2.w = value[v_base2 + 3];
                        }
                        v = tmp2;
                    }
                    float sum = dot(v, dout);
                    int out_idx = (((b * heads + h) * length + i) * K + neighbor_idx);
                    out[out_idx] += sum;
                    neighbor_idx++;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

'''

NATTEN_K5_AV_BWD_FUSED_TG_SOURCE = NATTEN_K3_AV_BWD_FUSED_TG_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2").replace("L = 9", "L = 25")
NATTEN_K7_AV_BWD_FUSED_TG_SOURCE = NATTEN_K3_AV_BWD_FUSED_TG_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3").replace("L = 9", "L = 49")

# Fused TG variants with TILE=8 (better for large K/large spatial sizes)
NATTEN_K3_AV_BWD_FUSED_TG_T8_SOURCE = NATTEN_K3_AV_BWD_FUSED_TG_SOURCE.replace("TILE_H = 16", "TILE_H = 8").replace("TILE_W = 16", "TILE_W = 8")
NATTEN_K5_AV_BWD_FUSED_TG_T8_SOURCE = NATTEN_K5_AV_BWD_FUSED_TG_SOURCE.replace("TILE_H = 16", "TILE_H = 8").replace("TILE_W = 16", "TILE_W = 8")
NATTEN_K7_AV_BWD_FUSED_TG_T8_SOURCE = NATTEN_K7_AV_BWD_FUSED_TG_SOURCE.replace("TILE_H = 16", "TILE_H = 8").replace("TILE_W = 16", "TILE_W = 8")

# Fused TG variants with unroll-by-2 over D (opt-in)
NATTEN_K3_AV_BWD_FUSED_TG_UNROLL2_SOURCE = _av_bwd_fused_unroll2_source(NATTEN_K3_AV_BWD_FUSED_TG_SOURCE)
NATTEN_K5_AV_BWD_FUSED_TG_UNROLL2_SOURCE = _av_bwd_fused_unroll2_source(NATTEN_K5_AV_BWD_FUSED_TG_SOURCE)
NATTEN_K7_AV_BWD_FUSED_TG_UNROLL2_SOURCE = _av_bwd_fused_unroll2_source(NATTEN_K7_AV_BWD_FUSED_TG_SOURCE)
NATTEN_K3_AV_BWD_FUSED_TG_T8_UNROLL2_SOURCE = _av_bwd_fused_unroll2_source(NATTEN_K3_AV_BWD_FUSED_TG_T8_SOURCE)
NATTEN_K5_AV_BWD_FUSED_TG_T8_UNROLL2_SOURCE = _av_bwd_fused_unroll2_source(NATTEN_K5_AV_BWD_FUSED_TG_T8_SOURCE)
NATTEN_K7_AV_BWD_FUSED_TG_T8_UNROLL2_SOURCE = _av_bwd_fused_unroll2_source(NATTEN_K7_AV_BWD_FUSED_TG_T8_SOURCE)

NATTEN_K3_AV_BWD_FUSED_SPLIT_TG_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 2D AV backward fused d_attn + d_value TG tiled (K=3) - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = d_out_shape[0];
const int heads = d_out_shape[1];
const int height = d_out_shape[2];
const int width = d_out_shape[3];
const int dim = d_out_shape[4];
const int dilation = (int)dilation_param[0];
const int split_start = (int)split_param[0];
const int split_len = (int)split_param[1];
const int split_end = split_start + split_len;
const int K = 3;
const int NH = 1;
const int L = 9;
const int TILE = 16;
const int DTILE = 4;
threadgroup float4 val_tile[TILE + 2*NH][TILE + 2*NH];
threadgroup float4 dout_tile[TILE + 2*NH][TILE + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.y * TILE;
int base_j = tg_gid.x * TILE;
int li = (int)tg_pos.y;
int lj = (int)tg_pos.x;
int gi = base_i + li - NH;
int gj = base_j + lj - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
// Initialize d_attn output for this (i,j) if inside tile
if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
    int i = base_i + (li - NH);
    int j = base_j + (lj - NH);
    if (i < height && j < width) {
        int out_base = (((b * heads + h) * height + i) * width + j) * split_len;
        for (int n = 0; n < split_len; n++) out_attn[out_base + n] = 0.0f;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 vval = float4(0.0f);
    float4 doval = float4(0.0f);
    if (gi >= 0 && gi < height && gj >= 0 && gj < width) {
        int v_base = (((b * heads + h) * height + gi) * width + gj) * dim + d0;
        float4 tmp = float4(0.0f);
        if (d0 + 3 < dim) {
            tmp = *((device const float4*)(value + v_base));
        } else {
            if (d0 + 0 < dim) tmp.x = value[v_base + 0];
            if (d0 + 1 < dim) tmp.y = value[v_base + 1];
            if (d0 + 2 < dim) tmp.z = value[v_base + 2];
            if (d0 + 3 < dim) tmp.w = value[v_base + 3];
        }
        vval = tmp;
        int do_base = (((b * heads + h) * height + gi) * width + gj) * dim + d0;
        float4 tmpd = float4(0.0f);
        if (d0 + 3 < dim) {
            tmpd = *((device const float4*)(d_out + do_base));
        } else {
            if (d0 + 0 < dim) tmpd.x = d_out[do_base + 0];
            if (d0 + 1 < dim) tmpd.y = d_out[do_base + 1];
            if (d0 + 2 < dim) tmpd.z = d_out[do_base + 2];
            if (d0 + 3 < dim) tmpd.w = d_out[do_base + 3];
        }
        doval = tmpd;
    }
    val_tile[li][lj] = vval;
    dout_tile[li][lj] = doval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // d_attn for query positions
    if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
        int i = base_i + (li - NH);
        int j = base_j + (lj - NH);
        if (i < height && j < width) {
            int ni, nj, ei, ej;
            NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
            int do_base = (((b * heads + h) * height + i) * width + j) * dim + d0;
            float4 dout = float4(0.0f);
            if (d0 + 3 < dim) {
                dout = *((device const float4*)(d_out + do_base));
            } else {
                if (d0 + 0 < dim) dout.x = d_out[do_base + 0];
                if (d0 + 1 < dim) dout.y = d_out[do_base + 1];
                if (d0 + 2 < dim) dout.z = d_out[do_base + 2];
                if (d0 + 3 < dim) dout.w = d_out[do_base + 3];
            }
            int neighbor_idx = 0;
            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {
                    int val_i = ni + ki;
                    int val_j = nj + kj;
                    if (val_i < 0 || val_i >= ei || val_j < 0 || val_j >= ej) {
                        neighbor_idx++;
                        continue;
                    }
                    if (neighbor_idx < split_start || neighbor_idx >= split_end) {
                        neighbor_idx++;
                        continue;
                    }
                    int ti = val_i - base_i + NH;
                    int tj = val_j - base_j + NH;
                    float4 v = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2*NH) && tj >= 0 && tj < (TILE + 2*NH)) {
                        v = val_tile[ti][tj];
                    } else {
                        int v_base2 = (((b * heads + h) * height + val_i) * width + val_j) * dim + d0;
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 3 < dim) {
                            tmp2 = *((device const float4*)(value + v_base2));
                        } else {
                            if (d0 + 0 < dim) tmp2.x = value[v_base2 + 0];
                            if (d0 + 1 < dim) tmp2.y = value[v_base2 + 1];
                            if (d0 + 2 < dim) tmp2.z = value[v_base2 + 2];
                            if (d0 + 3 < dim) tmp2.w = value[v_base2 + 3];
                        }
                        v = tmp2;
                    }
                    float sum = dot(v, dout);
                    int out_idx = (((b * heads + h) * height + i) * width + j) * split_len + (neighbor_idx - split_start);
                    out_attn[out_idx] += sum;
                    neighbor_idx++;
                }
            }
        }
    }

    // d_value for value positions
    if (li >= NH && li < NH + TILE && lj >= NH && lj < NH + TILE) {
        int vi = base_i + (li - NH);
        int vj = base_j + (lj - NH);
        if (vi < height && vj < width) {
            float4 sum4 = float4(0.0f);
            int i_start = max(0, vi - (K - 1) * dilation);
            int i_end = min(height - 1, vi + (K - 1) * dilation);
            int j_start = max(0, vj - (K - 1) * dilation);
            int j_end = min(width - 1, vj + (K - 1) * dilation);
            for (int i = i_start; i <= i_end; i++) {
                int ni, ei;
                NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
                NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
                if (vi < ni || vi >= ei) continue;
                int di = vi - ni;
                if (di % dilation != 0) continue;
                int ki = di / dilation;
                for (int j = j_start; j <= j_end; j++) {
                    int nj, ej;
                    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
                    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
                    if (vj < nj || vj >= ej) continue;
                    int dj = vj - nj;
                    if (dj % dilation != 0) continue;
                    int kj = dj / dilation;
                    int neighbor_idx = ki * K + kj;
                    if (neighbor_idx < split_start || neighbor_idx >= split_end) continue;
                    int attn_idx = (((b * heads + h) * height + i) * width + j) * (K*K) + neighbor_idx;
                    float attn_val = attn[attn_idx];
                    int ti = i - base_i + NH;
                    int tj = j - base_j + NH;
                    float4 do_v = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2*NH) && tj >= 0 && tj < (TILE + 2*NH)) {
                        do_v = dout_tile[ti][tj];
                    } else {
                        int do_base2 = (((b * heads + h) * height + i) * width + j) * dim + d0;
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 3 < dim) {
                            tmp2 = *((device const float4*)(d_out + do_base2));
                        } else {
                            if (d0 + 0 < dim) tmp2.x = d_out[do_base2 + 0];
                            if (d0 + 1 < dim) tmp2.y = d_out[do_base2 + 1];
                            if (d0 + 2 < dim) tmp2.z = d_out[do_base2 + 2];
                            if (d0 + 3 < dim) tmp2.w = d_out[do_base2 + 3];
                        }
                        do_v = tmp2;
                    }
                    sum4 = fma(attn_val, do_v, sum4);
                }
            }
            int out_idx = (((b * heads + h) * height + vi) * width + vj) * dim + d0;
            if (d0 + 0 < dim) out_val[out_idx + 0] = sum4.x;
            if (d0 + 1 < dim) out_val[out_idx + 1] = sum4.y;
            if (d0 + 2 < dim) out_val[out_idx + 2] = sum4.z;
            if (d0 + 3 < dim) out_val[out_idx + 3] = sum4.w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

'''

NATTEN_K5_AV_BWD_FUSED_SPLIT_TG_SOURCE = NATTEN_K3_AV_BWD_FUSED_SPLIT_TG_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2").replace("L = 9", "L = 25")
NATTEN_K7_AV_BWD_FUSED_SPLIT_TG_SOURCE = NATTEN_K3_AV_BWD_FUSED_SPLIT_TG_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3").replace("L = 9", "L = 49")

NATTEN_K3_AV_BWD_FUSED_SPLIT_TG_T8_SOURCE = NATTEN_K3_AV_BWD_FUSED_SPLIT_TG_SOURCE.replace("TILE = 16", "TILE = 8")
NATTEN_K5_AV_BWD_FUSED_SPLIT_TG_T8_SOURCE = NATTEN_K5_AV_BWD_FUSED_SPLIT_TG_SOURCE.replace("TILE = 16", "TILE = 8")
NATTEN_K7_AV_BWD_FUSED_SPLIT_TG_T8_SOURCE = NATTEN_K7_AV_BWD_FUSED_SPLIT_TG_SOURCE.replace("TILE = 16", "TILE = 8")

NATTEN_1D_K3_QK_BWD_DQ_TG_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_query TG tiled (K=3) - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int TILE = 64;
const int DTILE = 4;
threadgroup float4 key_tile[TILE + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.x * TILE;
int li = (int)tg_pos.x;
int gi = base_i + li - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 kval = float4(0.0f);
    if (gi >= 0 && gi < length) {
        int k_base = (((b * heads + h) * length + gi) * dim + d0);
        float4 tmp = float4(0.0f);
        if (d0 + 0 < dim) tmp.x = key[k_base + 0];
        if (d0 + 1 < dim) tmp.y = key[k_base + 1];
        if (d0 + 2 < dim) tmp.z = key[k_base + 2];
        if (d0 + 3 < dim) tmp.w = key[k_base + 3];
        kval = tmp;
    }
    key_tile[li] = kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (li >= NH && li < NH + TILE) {
        int i = base_i + (li - NH);
        if (i < length) {
            float4 sum4 = float4(0.0f);
            int ni, ei;
            NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
            int neighbor_idx = 0;
            for (int ki = 0; ki < K; ki++) {
                int key_i = ni + ki * dilation;
                if (key_i >= 0 && key_i < ei) {
                    int da_idx = (((b * heads + h) * length + i) * K + neighbor_idx);
                    float w = d_attn[da_idx];
                    int ti = key_i - base_i + NH;
                    float4 kval2 = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2*NH)) {
                        kval2 = key_tile[ti];
                    } else {
                        int k_base2 = (((b * heads + h) * length + key_i) * dim + d0);
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 0 < dim) tmp2.x = key[k_base2 + 0];
                        if (d0 + 1 < dim) tmp2.y = key[k_base2 + 1];
                        if (d0 + 2 < dim) tmp2.z = key[k_base2 + 2];
                        if (d0 + 3 < dim) tmp2.w = key[k_base2 + 3];
                        kval2 = tmp2;
                    }
                    sum4 += w * kval2;
                }
                neighbor_idx++;
            }
            int out_idx = (((b * heads + h) * length + i) * dim + d0);
            if (d0 + 0 < dim) out[out_idx + 0] = sum4.x;
            if (d0 + 1 < dim) out[out_idx + 1] = sum4.y;
            if (d0 + 2 < dim) out[out_idx + 2] = sum4.z;
            if (d0 + 3 < dim) out[out_idx + 3] = sum4.w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

'''

NATTEN_1D_K3_QK_BWD_DQ_TG_V4_SOURCE = NATTEN_HELPERS_SOURCE + '''

// 1D QK backward d_query TG tiled (K=3) vectorized - dilation=1 only
uint3 tg_pos = thread_position_in_threadgroup;
uint3 tg_gid = threadgroup_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int dilation = (int)dilation_param[0];
const int K = 3;
const int NH = 1;
const int TILE = 64;
const int DTILE = 4;
threadgroup float4 key_tile[TILE + 2*NH];
int b = tg_gid.z / heads;
int h = tg_gid.z % heads;
int base_i = tg_gid.x * TILE;
int li = (int)tg_pos.x;
int gi = base_i + li - NH;
if (b >= batch_size || h >= heads) return;
if (dilation != 1) return;
for (int d0 = 0; d0 < dim; d0 += DTILE) {
    float4 kval = float4(0.0f);
    if (gi >= 0 && gi < length) {
        int k_base = (((b * heads + h) * length + gi) * dim + d0);
        float4 tmp = float4(0.0f);
        if (d0 + 3 < dim) {
            tmp = *((device const float4*)(key + k_base));
        } else {
            if (d0 + 0 < dim) tmp.x = key[k_base + 0];
            if (d0 + 1 < dim) tmp.y = key[k_base + 1];
            if (d0 + 2 < dim) tmp.z = key[k_base + 2];
            if (d0 + 3 < dim) tmp.w = key[k_base + 3];
        }
        kval = tmp;
    }
    key_tile[li] = kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (li >= NH && li < NH + TILE) {
        int i = base_i + (li - NH);
        if (i < length) {
            float4 sum4 = float4(0.0f);
            int ni, ei;
            NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
            int neighbor_idx = 0;
            for (int ki = 0; ki < K; ki++) {
                int key_i = ni + ki * dilation;
                if (key_i >= 0 && key_i < ei) {
                    int da_idx = (((b * heads + h) * length + i) * K + neighbor_idx);
                    float w = d_attn[da_idx];
                    int ti = key_i - base_i + NH;
                    float4 kval2 = float4(0.0f);
                    if (ti >= 0 && ti < (TILE + 2*NH)) {
                        kval2 = key_tile[ti];
                    } else {
                        int k_base2 = (((b * heads + h) * length + key_i) * dim + d0);
                        float4 tmp2 = float4(0.0f);
                        if (d0 + 3 < dim) {
                            tmp2 = *((device const float4*)(key + k_base2));
                        } else {
                            if (d0 + 0 < dim) tmp2.x = key[k_base2 + 0];
                            if (d0 + 1 < dim) tmp2.y = key[k_base2 + 1];
                            if (d0 + 2 < dim) tmp2.z = key[k_base2 + 2];
                            if (d0 + 3 < dim) tmp2.w = key[k_base2 + 3];
                        }
                        kval2 = tmp2;
                    }
                    sum4 = fma(w, kval2, sum4);
                }
                neighbor_idx++;
            }
            int out_idx = (((b * heads + h) * length + i) * dim + d0);
            if (d0 + 0 < dim) out[out_idx + 0] = sum4.x;
            if (d0 + 1 < dim) out[out_idx + 1] = sum4.y;
            if (d0 + 2 < dim) out[out_idx + 2] = sum4.z;
            if (d0 + 3 < dim) out[out_idx + 3] = sum4.w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

'''

NATTEN_1D_K5_QK_BWD_DQ_TG_SOURCE = NATTEN_1D_K3_QK_BWD_DQ_TG_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2")
NATTEN_1D_K5_QK_BWD_DQ_TG_V4_SOURCE = NATTEN_1D_K3_QK_BWD_DQ_TG_V4_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2")
NATTEN_1D_K5_AV_BWD_DATTN_TG_SOURCE = NATTEN_1D_K3_AV_BWD_DATTN_TG_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2")
NATTEN_1D_K5_AV_BWD_DATTN_TG_V4_SOURCE = NATTEN_1D_K3_AV_BWD_DATTN_TG_V4_SOURCE.replace("K = 3", "K = 5").replace("NH = 1", "NH = 2")
NATTEN_1D_K7_QK_BWD_DQ_TG_SOURCE = NATTEN_1D_K3_QK_BWD_DQ_TG_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3")
NATTEN_1D_K7_QK_BWD_DQ_TG_V4_SOURCE = NATTEN_1D_K3_QK_BWD_DQ_TG_V4_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3")
NATTEN_1D_K7_AV_BWD_DATTN_TG_SOURCE = NATTEN_1D_K3_AV_BWD_DATTN_TG_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3")
NATTEN_1D_K7_AV_BWD_DATTN_TG_V4_SOURCE = NATTEN_1D_K3_AV_BWD_DATTN_TG_V4_SOURCE.replace("K = 3", "K = 7").replace("NH = 1", "NH = 3")

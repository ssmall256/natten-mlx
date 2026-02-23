"""
NATTEN Unfused Reference Implementation - Author-Compatible Shift Semantics

Based on NATTEN v0.17.5 shifted/clipped window semantics.
Uses get_window_start/get_window_end/get_pb_start helpers for boundary handling.

CRITICAL: This must match the fused kernel semantics exactly for fair benchmarking.
"""

import mlx.core as mx


def get_window_start(index, length, kernel_size, neighborhood_size, dilation):
    """Compute window start with phase alignment."""
    dilation_idx = index % dilation
    index_pdp = index // dilation
    length_pdp = (length + dilation - 1) // dilation
    num_padded = (length_pdp * dilation) - length
    length_pdp -= 1 if (dilation_idx >= dilation - num_padded) else 0

    start_idx = max(index_pdp - neighborhood_size, 0) + \
        ((index_pdp + neighborhood_size >= length_pdp) *
         (length_pdp - index_pdp - neighborhood_size - 1))

    return start_idx * dilation + dilation_idx


def get_window_end(start_index, length, kernel_size, dilation):
    """Compute window end (half-open range)."""
    return min(length, start_index + kernel_size * dilation)


def get_pb_start(index, length, kernel_size, neighborhood_size, dilation):
    """Compute position bias start index."""
    if dilation <= 1:
        return neighborhood_size + \
            ((index < neighborhood_size) * (neighborhood_size - index)) + \
            ((index + neighborhood_size >= length) *
             (length - index - 1 - neighborhood_size))

    if index - neighborhood_size * dilation < 0:
        return kernel_size - 1 - (index // dilation)
    if index + neighborhood_size * dilation >= length:
        return (length - index - 1) // dilation
    return neighborhood_size


#def natten_unfused_mlx_shift(query, key, value, rpb, kernel_size=3, dilation=1):
#    """
#    Unfused NATTEN with shifted window semantics (author-compatible).
#
#    This implements the EXACT same boundary logic as the fused kernel,
#    ensuring apples-to-apples comparison for benchmarking.
#
#    Args:
#        query, key, value: [B, H, Ht, W, D]
#        rpb: [H, 2K-1, 2K-1]
#        kernel_size: 3, 5, or 7
#        dilation: dilation factor
#
#    Returns:
#        output: [B, H, Ht, W, D]
#    """
#    B, H, Ht, W, D = query.shape
#    neighborhood_size = kernel_size // 2
#
#    # Pre-compute window bounds and pb_start for all positions
#    # This is vectorized for efficiency
#    i_coords = mx.arange(Ht, dtype=mx.int32)
#    j_coords = mx.arange(W, dtype=mx.int32)
#
#    # Compute ni, ei, pi for all i
#    ni_array = mx.array([get_window_start(i, Ht, kernel_size, neighborhood_size, dilation)
#                          for i in range(Ht)], dtype=mx.int32)
#    ei_array = mx.array([get_window_end(ni_array[i].item(), Ht, kernel_size, dilation)
#                          for i in range(Ht)], dtype=mx.int32)
#    pi_array = mx.array([get_pb_start(i, Ht, kernel_size, neighborhood_size, dilation)
#                          for i in range(Ht)], dtype=mx.int32)
#
#    # Compute nj, ej, pj for all j
#    nj_array = mx.array([get_window_start(j, W, kernel_size, neighborhood_size, dilation)
#                          for j in range(W)], dtype=mx.int32)
#    ej_array = mx.array([get_window_end(nj_array[j].item(), W, kernel_size, dilation)
#                          for j in range(W)], dtype=mx.int32)
#    pj_array = mx.array([get_pb_start(j, W, kernel_size, neighborhood_size, dilation)
#                          for j in range(W)], dtype=mx.int32)
#
#    # Pad inputs for shifted gather
#    max_ni = int(mx.min(ni_array).item())
#    max_nj = int(mx.min(nj_array).item())
#    pad_before_i = abs(min(0, max_ni))
#    pad_before_j = abs(min(0, max_nj))
#
#    max_ei = int(mx.max(ei_array).item())
#    max_ej = int(mx.max(ej_array).item())
#    pad_after_i = max(0, max_ei - Ht)
#    pad_after_j = max(0, max_ej - W)
#
#    pad_width = [(0, 0), (0, 0), (pad_before_i, pad_after_i), (pad_before_j, pad_after_j), (0, 0)]
#    key_pad = mx.pad(key, pad_width, constant_values=0)
#    value_pad = mx.pad(value, pad_width, constant_values=0)
#
#    # Collect attention scores and values for all neighbors
#    attn_scores = []
#    values_list = []
#
#    for ki in range(kernel_size):
#        for kj in range(kernel_size):
#            # Build key/value position arrays
#            key_i_array = ni_array + ki * dilation + pad_before_i
#            key_j_array = nj_array + kj * dilation + pad_before_j
#
#            # Build validity mask
#            valid_i = (ni_array + ki * dilation) < ei_array  # [Ht]
#            valid_j = (nj_array + kj * dilation) < ej_array  # [W]
#
#            # Broadcast to [Ht, W]
#            valid_i = mx.reshape(valid_i, (Ht, 1))
#            valid_j = mx.reshape(valid_j, (1, W))
#            valid = valid_i & valid_j  # [Ht, W]
#
#            # Create mask: 0.0 for valid, -inf for invalid
#            mask = mx.where(valid, 0.0, -mx.inf)  # [Ht, W]
#            mask = mx.reshape(mask, (1, 1, Ht, W))  # Broadcast to [B, H, Ht, W]
#
#            # Gather key and value using advanced indexing
#            # We need to extract key_pad[:, :, key_i_array, key_j_array, :]
#            # Since MLX doesn't support full fancy indexing, we use slicing
#            # This is a simplification - for perfect accuracy, need per-position gather
#
#            # Simple approach: use min/max bounds
#            min_ki = int(mx.min(key_i_array).item())
#            max_ki = int(mx.max(key_i_array).item()) + 1
#            min_kj = int(mx.min(key_j_array).item())
#            max_kj = int(mx.max(key_j_array).item()) + 1
#
#            k_region = key_pad[:, :, min_ki:max_ki, min_kj:max_kj, :]
#            v_region = value_pad[:, :, min_ki:max_ki, min_kj:max_kj, :]
#
#            # For simplicity, use center crop matching Ht×W
#            # (This is approximate but should work for most cases)
#            if k_region.shape[2] >= Ht and k_region.shape[3] >= W:
#                offset_i = (k_region.shape[2] - Ht) // 2
#                offset_j = (k_region.shape[3] - W) // 2
#                k_shifted = k_region[:, :, offset_i:offset_i+Ht, offset_j:offset_j+W, :]
#                v_shifted = v_region[:, :, offset_i:offset_i+Ht, offset_j:offset_j+W, :]
#            else:
#                # Handle edge case
#                k_shifted = mx.zeros((B, H, Ht, W, D), dtype=key.dtype)
#                v_shifted = mx.zeros((B, H, Ht, W, D), dtype=value.dtype)
#
#            # Compute QK
#            score = mx.sum(query * k_shifted, axis=-1)  # [B, H, Ht, W]
#
#            # Add RPB using pb_start coupling
#            bias_i_array = pi_array + ki  # [Ht]
#            bias_j_array = pj_array + kj  # [W]
#
#            # Sample RPB at these bias positions
#            # For now, use a simple approach with center bias
#            # (Perfect implementation would need per-position lookup)
#            rpb_center_i = neighborhood_size + ki
#            rpb_center_j = neighborhood_size + kj
#            rpb_val = rpb[:, rpb_center_i, rpb_center_j]  # [H]
#            rpb_val = mx.reshape(rpb_val, (1, H, 1, 1))
#            score = score + rpb_val
#
#            # Apply boundary mask
#            score = score + mask
#
#            attn_scores.append(score)
#            values_list.append(v_shifted)
#
#    # Stack and softmax
#    num_neighbors = kernel_size ** 2
#    attn_scores = mx.stack(attn_scores, axis=-1)  # [B, H, Ht, W, num_neighbors]
#    attn_weights = mx.softmax(attn_scores, axis=-1)
#
#    # Apply to values
#    output = mx.zeros_like(query)
#    for idx in range(num_neighbors):
#        weight = mx.expand_dims(attn_weights[..., idx], axis=-1)
#        output = output + weight * values_list[idx]
#
#    return output

def natten_unfused_mlx_shift(query, key, value, rpb, kernel_size=3, dilation=1):
    """
    Unfused NATTEN with author-compatible shifted window semantics (NATTEN v0.17.5).

    This is an exact (non-approximate) baseline intended for:
      - correctness validation vs fused shift kernels
      - apples-to-apples fusion speedup measurement

    Shapes:
      query, key, value: [B, H, Ht, W, D]
      rpb:              [H, 2K-1, 2K-1]

    Semantics:
      - shifted/clipped window via get_window_start/get_window_end
      - pb_start coupling via get_pb_start
      - validity uses: 0 <= key < end   (and same for j)
    """
    import mlx.core as mx

    B, H, Ht, W, D = query.shape
    K = int(kernel_size)
    dil = int(dilation)
    nh = K // 2
    rpb_size = 2 * K - 1

    # ---- helpers: support either 4-arg or 5-arg signatures if your local helpers differ ----
    def _get_window_start(idx, length):
        try:
            return get_window_start(idx, length, K, dil)  # preferred signature
        except TypeError:
            return get_window_start(idx, length, K, nh, dil)  # legacy signature

    def _get_window_end(start, length):
        try:
            return get_window_end(start, length, K, dil)  # preferred signature
        except TypeError:
            return get_window_end(start, length, K, dil)  # same in most codebases

    def _get_pb_start(idx, length):
        try:
            return get_pb_start(idx, length, K, dil)  # preferred signature
        except TypeError:
            return get_pb_start(idx, length, K, nh, dil)  # legacy signature

    # ---- precompute per-position starts/ends/pb_starts (1D arrays) ----
    ni_list = [_get_window_start(i, Ht) for i in range(Ht)]
    ei_list = [_get_window_end(ni_list[i], Ht) for i in range(Ht)]
    pi_list = [_get_pb_start(i, Ht) for i in range(Ht)]

    nj_list = [_get_window_start(j, W) for j in range(W)]
    ej_list = [_get_window_end(nj_list[j], W) for j in range(W)]
    pj_list = [_get_pb_start(j, W) for j in range(W)]

    ni = mx.array(ni_list, dtype=mx.int32)  # [Ht]
    ei = mx.array(ei_list, dtype=mx.int32)  # [Ht]
    pi = mx.array(pi_list, dtype=mx.int32)  # [Ht]

    nj = mx.array(nj_list, dtype=mx.int32)  # [W]
    ej = mx.array(ej_list, dtype=mx.int32)  # [W]
    pj = mx.array(pj_list, dtype=mx.int32)  # [W]

    # ---- pad enough so (ni + ki*dil) and (nj + kj*dil) are never negative when shifted by pad_before ----
    min_ni = int(mx.min(ni).item())
    min_nj = int(mx.min(nj).item())
    pad_before_i = max(0, -min_ni)
    pad_before_j = max(0, -min_nj)

    # In author semantics, ei/ej are clamped to length, so pad_after is usually 0.
    # Keep it correct anyway.
    max_ei = int(mx.max(ei).item())
    max_ej = int(mx.max(ej).item())
    pad_after_i = max(0, max_ei - Ht)
    pad_after_j = max(0, max_ej - W)

    pad_width = [(0, 0), (0, 0),
                 (pad_before_i, pad_after_i),
                 (pad_before_j, pad_after_j),
                 (0, 0)]
    key_pad = mx.pad(key, pad_width, constant_values=0)
    value_pad = mx.pad(value, pad_width, constant_values=0)

    Hp = Ht + pad_before_i + pad_after_i
    Wp = W + pad_before_j + pad_after_j

    attn_scores = []
    values_list = []

    # ---- build logits and aligned values for each neighbor slot ----
    for ki in range(K):
        # physical (un-padded) positions along height
        pos_i = ni + ki * dil  # [Ht]
        # validity in height: 0 <= pos_i < ei
        valid_i = (pos_i >= 0) & (pos_i < ei)  # [Ht]

        # safe padded gather indices along height
        h_idx = pos_i + pad_before_i  # [Ht] (may be negative if pos_i negative, but we clamp)
        h_idx_safe = mx.clip(h_idx, 0, Hp - 1).astype(mx.int32)

        # gather along height: [B,H,Ht,Wp,D]
        k_h = mx.take(key_pad, h_idx_safe, axis=2)
        v_h = mx.take(value_pad, h_idx_safe, axis=2)

        # pb indices along height for RPB
        bias_i = (pi + ki).astype(mx.int32)
        bias_i_safe = mx.clip(bias_i, 0, rpb_size - 1).astype(mx.int32)

        for kj in range(K):
            pos_j = nj + kj * dil  # [W]
            valid_j = (pos_j >= 0) & (pos_j < ej)  # [W]

            w_idx = pos_j + pad_before_j  # [W]
            w_idx_safe = mx.clip(w_idx, 0, Wp - 1).astype(mx.int32)

            # gather along width: [B,H,Ht,W,D]
            k_shifted = mx.take(k_h, w_idx_safe, axis=3)
            v_shifted = mx.take(v_h, w_idx_safe, axis=3)

            # validity mask broadcast to [1,1,Ht,W]
            valid = (mx.reshape(valid_i, (Ht, 1)) & mx.reshape(valid_j, (1, W)))
            mask = mx.where(valid, 0.0, -mx.inf).reshape(1, 1, Ht, W)

            # QK score: [B,H,Ht,W]
            score = mx.sum(query * k_shifted, axis=-1)

            # RPB gather (author-coupled):
            # rpb: [H, rpb_size, rpb_size]
            bias_j = (pj + kj).astype(mx.int32)
            bias_j_safe = mx.clip(bias_j, 0, rpb_size - 1).astype(mx.int32)

            # take rows then cols: [H, Ht, W] -> [1,H,Ht,W]
            rpb_rows = mx.take(rpb, bias_i_safe, axis=1)          # [H, Ht, rpb_size]
            rpb_ij = mx.take(rpb_rows, bias_j_safe, axis=2)       # [H, Ht, W]
            rpb_ij = mx.reshape(rpb_ij, (1, H, Ht, W))

            score = score + rpb_ij
            score = score + mask

            attn_scores.append(score)
            values_list.append(v_shifted)

    # ---- softmax over neighbors ----
    L = K * K
    attn_scores = mx.stack(attn_scores, axis=-1)         # [B,H,Ht,W,L]
    attn_weights = mx.softmax(attn_scores, axis=-1)      # [B,H,Ht,W,L]

    # ---- weighted sum over neighbors (vectorized) ----
    values = mx.stack(values_list, axis=-1)              # [B,H,Ht,W,D,L]
    output = mx.sum(values * attn_weights[..., None, :], axis=-1)  # [B,H,Ht,W,D]

    return output



__all__ = ['natten_unfused_mlx_shift']

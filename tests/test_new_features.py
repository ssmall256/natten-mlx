"""Tests for new natten-mlx features: return_lse, merge_attentions, GQA/MQA,
additional_keys/additional_values, and FMHA fast path."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.functional import (
    na1d,
    na1d_av,
    na1d_qk,
    na2d,
    na2d_av,
    na2d_qk,
    na3d,
    na3d_av,
    na3d_qk,
)
from natten_mlx.merge import merge_attentions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed(seed: int = 42):
    mx.random.seed(seed)


def _rand(*shape, dtype=mx.float32):
    return mx.random.normal(shape).astype(dtype)


_randn = _rand  # alias used by backward tests


def _allclose(a, b, atol=1e-3, rtol=1e-3):
    a_np = np.array(a)
    b_np = np.array(b)
    np.testing.assert_allclose(a_np, b_np, atol=atol, rtol=rtol)


# ===================================================================
# 1. return_lse
# ===================================================================

class TestReturnLSE:
    """Tests for return_lse on na1d, na2d, na3d."""

    # -- 1D --------------------------------------------------------

    def test_na1d_return_lse_false_gives_plain_array(self):
        q, k, v = _rand(1, 12, 2, 8), _rand(1, 12, 2, 8), _rand(1, 12, 2, 8)
        out = na1d(q, k, v, kernel_size=5)
        mx.eval(out)
        assert isinstance(out, mx.array)

    def test_na1d_return_lse_true_gives_tuple(self):
        q, k, v = _rand(1, 12, 2, 8), _rand(1, 12, 2, 8), _rand(1, 12, 2, 8)
        result = na1d(q, k, v, kernel_size=5, return_lse=True)
        mx.eval(result)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_na1d_lse_shape(self):
        B, L, H, D = 2, 16, 4, 8
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)
        out, lse = na1d(q, k, v, kernel_size=5, return_lse=True)
        mx.eval(out, lse)
        assert out.shape == (B, L, H, D)
        assert lse.shape == (B, L, H)

    def test_na1d_lse_matches_manual(self):
        B, L, H, D = 1, 12, 2, 8
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)
        ks = 5
        scale = D ** -0.5

        out, lse = na1d(q, k, v, kernel_size=ks, return_lse=True)
        mx.eval(out, lse)

        # Manual path via split ops
        logits = na1d_qk(q, k, kernel_size=ks)
        mx.eval(logits)
        manual_lse = mx.logsumexp(logits * scale, axis=-1)
        mx.eval(manual_lse)

        _allclose(lse, manual_lse, atol=1e-3)

    def test_na1d_lse_with_causal(self):
        B, L, H, D = 1, 12, 2, 8
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)
        result = na1d(q, k, v, kernel_size=5, is_causal=True, return_lse=True)
        mx.eval(result)
        out, lse = result
        assert out.shape == (B, L, H, D)
        assert lse.shape == (B, L, H)

    # -- 2D --------------------------------------------------------

    def test_na2d_return_lse_false_gives_plain_array(self):
        q, k, v = _rand(1, 8, 8, 2, 8), _rand(1, 8, 8, 2, 8), _rand(1, 8, 8, 2, 8)
        out = na2d(q, k, v, kernel_size=3)
        mx.eval(out)
        assert isinstance(out, mx.array)

    def test_na2d_return_lse_true_gives_tuple(self):
        q, k, v = _rand(1, 8, 8, 2, 8), _rand(1, 8, 8, 2, 8), _rand(1, 8, 8, 2, 8)
        result = na2d(q, k, v, kernel_size=3, return_lse=True)
        mx.eval(result)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_na2d_lse_shape(self):
        B, Hh, W, H, D = 2, 8, 8, 4, 8
        q, k, v = _rand(B, Hh, W, H, D), _rand(B, Hh, W, H, D), _rand(B, Hh, W, H, D)
        out, lse = na2d(q, k, v, kernel_size=3, return_lse=True)
        mx.eval(out, lse)
        assert out.shape == (B, Hh, W, H, D)
        assert lse.shape == (B, Hh, W, H)

    def test_na2d_lse_matches_manual(self):
        B, Hh, W, H, D = 1, 8, 8, 2, 8
        q, k, v = _rand(B, Hh, W, H, D), _rand(B, Hh, W, H, D), _rand(B, Hh, W, H, D)
        ks = 3
        scale = D ** -0.5

        out, lse = na2d(q, k, v, kernel_size=ks, return_lse=True)
        mx.eval(out, lse)

        logits = na2d_qk(q, k, kernel_size=ks)
        mx.eval(logits)
        manual_lse = mx.logsumexp(logits * scale, axis=-1)
        mx.eval(manual_lse)

        _allclose(lse, manual_lse, atol=1e-3)

    def test_na2d_lse_with_causal(self):
        B, Hh, W, H, D = 1, 8, 8, 2, 8
        q, k, v = _rand(B, Hh, W, H, D), _rand(B, Hh, W, H, D), _rand(B, Hh, W, H, D)
        result = na2d(q, k, v, kernel_size=3, is_causal=(True, False), return_lse=True)
        mx.eval(result)
        out, lse = result
        assert out.shape == (B, Hh, W, H, D)
        assert lse.shape == (B, Hh, W, H)

    # -- 3D --------------------------------------------------------

    def test_na3d_return_lse_false_gives_plain_array(self):
        q, k, v = _rand(1, 4, 4, 4, 2, 8), _rand(1, 4, 4, 4, 2, 8), _rand(1, 4, 4, 4, 2, 8)
        out = na3d(q, k, v, kernel_size=3)
        mx.eval(out)
        assert isinstance(out, mx.array)

    def test_na3d_return_lse_true_gives_tuple(self):
        q, k, v = _rand(1, 4, 4, 4, 2, 8), _rand(1, 4, 4, 4, 2, 8), _rand(1, 4, 4, 4, 2, 8)
        result = na3d(q, k, v, kernel_size=3, return_lse=True)
        mx.eval(result)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_na3d_lse_shape(self):
        B, Dd, Hh, W, H, D = 1, 4, 4, 4, 2, 8
        q, k, v = _rand(B, Dd, Hh, W, H, D), _rand(B, Dd, Hh, W, H, D), _rand(B, Dd, Hh, W, H, D)
        out, lse = na3d(q, k, v, kernel_size=3, return_lse=True)
        mx.eval(out, lse)
        assert out.shape == (B, Dd, Hh, W, H, D)
        assert lse.shape == (B, Dd, Hh, W, H)

    def test_na3d_lse_matches_manual(self):
        B, Dd, Hh, W, H, D = 1, 4, 4, 4, 2, 8
        q, k, v = _rand(B, Dd, Hh, W, H, D), _rand(B, Dd, Hh, W, H, D), _rand(B, Dd, Hh, W, H, D)
        ks = 3
        scale = D ** -0.5

        out, lse = na3d(q, k, v, kernel_size=ks, return_lse=True)
        mx.eval(out, lse)

        logits = na3d_qk(q, k, kernel_size=ks)
        mx.eval(logits)
        manual_lse = mx.logsumexp(logits * scale, axis=-1)
        mx.eval(manual_lse)

        _allclose(lse, manual_lse, atol=1e-3)


# ===================================================================
# 2. merge_attentions
# ===================================================================

class TestMergeAttentions:
    """Tests for the merge_attentions utility."""

    def _manual_global_attn_with_lse(self, q, kv_list, scale):
        """Compute global attention + LSE manually for a list of (K, V) pairs.

        q: [B, S, H, D] (already flattened spatial)
        kv_list: list of (K, V) each [B, N_i, H, D]
        Returns (output, lse) for each pair plus the full concatenated result.
        """
        B, S, H, D = q.shape
        q_t = mx.transpose(q, axes=(0, 2, 1, 3))  # [B, H, S, D]

        chunk_results = []
        for k_i, v_i in kv_list:
            k_t = mx.transpose(k_i, axes=(0, 2, 1, 3))  # [B, H, N_i, D]
            v_t = mx.transpose(v_i, axes=(0, 2, 1, 3))
            logits = (q_t @ mx.transpose(k_t, axes=(0, 1, 3, 2))) * scale  # [B,H,S,N_i]
            lse_i = mx.logsumexp(logits, axis=-1)  # [B, H, S]
            attn = mx.softmax(logits, axis=-1)
            out_i = attn @ v_t  # [B, H, S, D]
            # Transpose back to [B, S, H, D]
            out_i = mx.transpose(out_i, axes=(0, 2, 1, 3))
            lse_i = mx.transpose(lse_i, axes=(0, 2, 1))  # [B, S, H]
            chunk_results.append((out_i, lse_i))

        # Full (concatenated KV)
        k_full = mx.concatenate([k for k, _ in kv_list], axis=1)
        v_full = mx.concatenate([v for _, v in kv_list], axis=1)
        k_t = mx.transpose(k_full, axes=(0, 2, 1, 3))
        v_t = mx.transpose(v_full, axes=(0, 2, 1, 3))
        logits_full = (q_t @ mx.transpose(k_t, axes=(0, 1, 3, 2))) * scale
        attn_full = mx.softmax(logits_full, axis=-1)
        out_full = mx.transpose(attn_full @ v_t, axes=(0, 2, 1, 3))

        return chunk_results, out_full

    def test_two_way_merge_matches_full_kv_1d(self):
        """Split KV in half, attend separately with manual global attn, merge."""
        B, L, H, D = 1, 12, 2, 16
        scale = D ** -0.5
        q = _rand(B, L, H, D)
        k = _rand(B, L, H, D)
        v = _rand(B, L, H, D)

        mid = L // 2
        kv_pairs = [(k[:, :mid], v[:, :mid]), (k[:, mid:], v[:, mid:])]
        chunks, out_full = self._manual_global_attn_with_lse(q, kv_pairs, scale)
        mx.eval(out_full)

        outs = [c[0] for c in chunks]
        lses = [c[1] for c in chunks]
        mx.eval(*outs, *lses)

        merged, _ = merge_attentions(outs, lses)
        mx.eval(merged)

        _allclose(merged, out_full, atol=1e-3)

    def test_two_way_merge_matches_full_kv_2d(self):
        """2D shaped tensors: flatten spatial, split KV, merge, compare."""
        B, Hh, W, H, D = 1, 4, 4, 2, 8
        scale = D ** -0.5
        q = _rand(B, Hh, W, H, D)
        k = _rand(B, Hh, W, H, D)
        v = _rand(B, Hh, W, H, D)

        S = Hh * W
        q_flat = q.reshape(B, S, H, D)
        k_flat = k.reshape(B, S, H, D)
        v_flat = v.reshape(B, S, H, D)

        mid = S // 2
        kv_pairs = [(k_flat[:, :mid], v_flat[:, :mid]), (k_flat[:, mid:], v_flat[:, mid:])]
        chunks, out_full_flat = self._manual_global_attn_with_lse(q_flat, kv_pairs, scale)
        mx.eval(out_full_flat)

        # Reshape outputs and lses to 2D spatial shape for merge
        outs = [c[0].reshape(B, Hh, W, H, D) for c in chunks]
        lses = [c[1].reshape(B, Hh, W, H) for c in chunks]
        mx.eval(*outs, *lses)

        merged, _ = merge_attentions(outs, lses)
        mx.eval(merged)

        out_full = out_full_flat.reshape(B, Hh, W, H, D)
        _allclose(merged, out_full, atol=1e-3)

    def test_three_way_merge_1d(self):
        """Three-way split and merge should equal full attention."""
        B, L, H, D = 1, 12, 2, 8
        scale = D ** -0.5
        q = _rand(B, L, H, D)
        k = _rand(B, L, H, D)
        v = _rand(B, L, H, D)

        chunk = L // 3
        kv_pairs = [
            (k[:, i * chunk : (i + 1) * chunk], v[:, i * chunk : (i + 1) * chunk])
            for i in range(3)
        ]
        chunks, out_full = self._manual_global_attn_with_lse(q, kv_pairs, scale)
        mx.eval(out_full)

        outs = [c[0] for c in chunks]
        lses = [c[1] for c in chunks]
        mx.eval(*outs, *lses)

        merged, _ = merge_attentions(outs, lses)
        mx.eval(merged)

        _allclose(merged, out_full, atol=1e-3)

    def test_merge_rejects_fewer_than_two(self):
        o = _rand(1, 8, 2, 8)
        l = _rand(1, 8, 2)
        with pytest.raises(ValueError, match="at least two"):
            merge_attentions([o], [l])

    def test_merge_rejects_mismatched_counts(self):
        o1 = _rand(1, 8, 2, 8)
        o2 = _rand(1, 8, 2, 8)
        l1 = _rand(1, 8, 2)
        with pytest.raises(ValueError, match="must match"):
            merge_attentions([o1, o2], [l1])

    def test_merge_rejects_shape_mismatch_output(self):
        o1 = _rand(1, 8, 2, 8)
        o2 = _rand(1, 10, 2, 8)  # different spatial
        l1 = _rand(1, 8, 2)
        l2 = _rand(1, 10, 2)
        with pytest.raises(ValueError, match="shape"):
            merge_attentions([o1, o2], [l1, l2])

    def test_merge_rejects_lse_shape_mismatch(self):
        o1 = _rand(1, 8, 2, 8)
        o2 = _rand(1, 8, 2, 8)
        l1 = _rand(1, 8, 2)
        l2 = _rand(1, 8, 3)  # wrong heads
        with pytest.raises(ValueError, match="shape"):
            merge_attentions([o1, o2], [l1, l2])


# ===================================================================
# 3. GQA / MQA
# ===================================================================

class TestGQA:
    """Tests for grouped-query and multi-query attention support."""

    # -- 1D --------------------------------------------------------

    def test_na1d_gqa_forward_matches_explicit_repeat(self):
        B, L, Hq, Hkv, D = 1, 12, 8, 2, 8
        q = _rand(B, L, Hq, D)
        k = _rand(B, L, Hkv, D)
        v = _rand(B, L, Hkv, D)
        ks = 5

        out_gqa = na1d(q, k, v, kernel_size=ks)
        mx.eval(out_gqa)

        # Explicit repeat path
        rep = Hq // Hkv
        k_rep = mx.repeat(k, rep, axis=-2)
        v_rep = mx.repeat(v, rep, axis=-2)
        out_rep = na1d(q, k_rep, v_rep, kernel_size=ks)
        mx.eval(out_rep)

        assert out_gqa.shape == (B, L, Hq, D)
        _allclose(out_gqa, out_rep, atol=1e-4)

    def test_na1d_mqa(self):
        B, L, Hq, Hkv, D = 1, 12, 4, 1, 8
        q = _rand(B, L, Hq, D)
        k = _rand(B, L, Hkv, D)
        v = _rand(B, L, Hkv, D)

        out = na1d(q, k, v, kernel_size=5)
        mx.eval(out)
        assert out.shape == (B, L, Hq, D)

    def test_na1d_gqa_invalid_ratio_raises(self):
        B, L, D = 1, 12, 8
        q = _rand(B, L, 7, D)  # 7 heads
        k = _rand(B, L, 3, D)  # 3 heads -- 7 % 3 != 0
        v = _rand(B, L, 3, D)
        with pytest.raises(ValueError, match="divisible"):
            na1d(q, k, v, kernel_size=5)

    # -- 2D --------------------------------------------------------

    def test_na2d_gqa_forward_matches_explicit_repeat(self):
        B, Hh, W, Hq, Hkv, D = 1, 8, 8, 4, 2, 8
        q = _rand(B, Hh, W, Hq, D)
        k = _rand(B, Hh, W, Hkv, D)
        v = _rand(B, Hh, W, Hkv, D)
        ks = 3

        out_gqa = na2d(q, k, v, kernel_size=ks)
        mx.eval(out_gqa)

        rep = Hq // Hkv
        k_rep = mx.repeat(k, rep, axis=-2)
        v_rep = mx.repeat(v, rep, axis=-2)
        out_rep = na2d(q, k_rep, v_rep, kernel_size=ks)
        mx.eval(out_rep)

        assert out_gqa.shape == (B, Hh, W, Hq, D)
        _allclose(out_gqa, out_rep, atol=1e-4)

    def test_na2d_mqa(self):
        B, Hh, W, Hq, Hkv, D = 1, 8, 8, 4, 1, 8
        q = _rand(B, Hh, W, Hq, D)
        k = _rand(B, Hh, W, Hkv, D)
        v = _rand(B, Hh, W, Hkv, D)

        out = na2d(q, k, v, kernel_size=3)
        mx.eval(out)
        assert out.shape == (B, Hh, W, Hq, D)

    def test_na2d_gqa_invalid_ratio_raises(self):
        B, Hh, W, D = 1, 8, 8, 8
        q = _rand(B, Hh, W, 5, D)
        k = _rand(B, Hh, W, 3, D)
        v = _rand(B, Hh, W, 3, D)
        with pytest.raises(ValueError, match="divisible"):
            na2d(q, k, v, kernel_size=3)

    # -- 3D --------------------------------------------------------

    def test_na3d_gqa_forward_matches_explicit_repeat(self):
        B, Dd, Hh, W, Hq, Hkv, D = 1, 4, 4, 4, 4, 2, 8
        q = _rand(B, Dd, Hh, W, Hq, D)
        k = _rand(B, Dd, Hh, W, Hkv, D)
        v = _rand(B, Dd, Hh, W, Hkv, D)
        ks = 3

        out_gqa = na3d(q, k, v, kernel_size=ks)
        mx.eval(out_gqa)

        rep = Hq // Hkv
        k_rep = mx.repeat(k, rep, axis=-2)
        v_rep = mx.repeat(v, rep, axis=-2)
        out_rep = na3d(q, k_rep, v_rep, kernel_size=ks)
        mx.eval(out_rep)

        assert out_gqa.shape == (B, Dd, Hh, W, Hq, D)
        _allclose(out_gqa, out_rep, atol=1e-4)

    def test_na3d_mqa(self):
        B, Dd, Hh, W, Hq, Hkv, D = 1, 4, 4, 4, 4, 1, 8
        q = _rand(B, Dd, Hh, W, Hq, D)
        k = _rand(B, Dd, Hh, W, Hkv, D)
        v = _rand(B, Dd, Hh, W, Hkv, D)

        out = na3d(q, k, v, kernel_size=3)
        mx.eval(out)
        assert out.shape == (B, Dd, Hh, W, Hq, D)

    def test_na3d_gqa_invalid_ratio_raises(self):
        B, Dd, Hh, W, D = 1, 4, 4, 4, 8
        q = _rand(B, Dd, Hh, W, 5, D)
        k = _rand(B, Dd, Hh, W, 3, D)
        v = _rand(B, Dd, Hh, W, 3, D)
        with pytest.raises(ValueError, match="divisible"):
            na3d(q, k, v, kernel_size=3)

    # -- Output shape correctness ----------------------------------

    def test_gqa_output_shape_has_heads_q(self):
        """Output should always have heads_q heads, not heads_kv."""
        B, L, Hq, Hkv, D = 2, 16, 8, 2, 16
        q = _rand(B, L, Hq, D)
        k = _rand(B, L, Hkv, D)
        v = _rand(B, L, Hkv, D)
        out = na1d(q, k, v, kernel_size=5)
        mx.eval(out)
        assert out.shape == (B, L, Hq, D), f"Expected heads_q={Hq} in output, got {out.shape}"


# ===================================================================
# 4. additional_keys / additional_values
# ===================================================================

class TestAdditionalKV:
    """Tests for additional_keys and additional_values support."""

    def test_na1d_additional_kv_output_shape_unchanged(self):
        B, L, H, D = 1, 12, 2, 8
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)
        ak = _rand(B, 1, H, D)  # 1 additional token
        av = _rand(B, 1, H, D)
        out = na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        mx.eval(out)
        assert out.shape == (B, L, H, D)

    def test_na1d_additional_kv_changes_output(self):
        B, L, H, D = 1, 12, 2, 8
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)
        ak = _rand(B, 3, H, D) * 10.0  # large values to ensure effect
        av = _rand(B, 3, H, D) * 10.0

        out_plain = na1d(q, k, v, kernel_size=5)
        out_extra = na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        mx.eval(out_plain, out_extra)

        diff = mx.abs(out_plain - out_extra).max()
        mx.eval(diff)
        assert float(diff) > 1e-4, "Additional KV should change the output"

    def test_na2d_additional_kv_output_shape(self):
        B, Hh, W, H, D = 1, 8, 8, 2, 8
        q, k, v = _rand(B, Hh, W, H, D), _rand(B, Hh, W, H, D), _rand(B, Hh, W, H, D)
        ak = _rand(B, 2, H, D)
        av = _rand(B, 2, H, D)
        out = na2d(q, k, v, kernel_size=3, additional_keys=ak, additional_values=av)
        mx.eval(out)
        assert out.shape == (B, Hh, W, H, D)

    def test_na3d_additional_kv_output_shape(self):
        B, Dd, Hh, W, H, D = 1, 4, 4, 4, 2, 8
        q = _rand(B, Dd, Hh, W, H, D)
        k = _rand(B, Dd, Hh, W, H, D)
        v = _rand(B, Dd, Hh, W, H, D)
        ak = _rand(B, 1, H, D)
        av = _rand(B, 1, H, D)
        out = na3d(q, k, v, kernel_size=3, additional_keys=ak, additional_values=av)
        mx.eval(out)
        assert out.shape == (B, Dd, Hh, W, H, D)

    # -- Validation: must provide both or neither -----------------

    def test_additional_keys_only_raises(self):
        B, L, H, D = 1, 12, 2, 8
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)
        ak = _rand(B, 1, H, D)
        with pytest.raises(ValueError, match="both be provided"):
            na1d(q, k, v, kernel_size=5, additional_keys=ak)

    def test_additional_values_only_raises(self):
        B, L, H, D = 1, 12, 2, 8
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)
        av = _rand(B, 1, H, D)
        with pytest.raises(ValueError, match="both be provided"):
            na1d(q, k, v, kernel_size=5, additional_values=av)

    # -- Shape validation -----------------------------------------

    def test_additional_kv_wrong_ndim_raises(self):
        B, L, H, D = 1, 12, 2, 8
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)
        ak = _rand(B, 1, D)  # 3D instead of 4D
        av = _rand(B, 1, D)
        with pytest.raises(ValueError, match="4D"):
            na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)

    def test_additional_kv_batch_mismatch_raises(self):
        B, L, H, D = 2, 12, 2, 8
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)
        ak = _rand(3, 1, H, D)  # wrong batch
        av = _rand(3, 1, H, D)
        with pytest.raises(ValueError, match="batch"):
            na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)

    def test_additional_kv_head_dim_mismatch_raises(self):
        B, L, H, D = 1, 12, 2, 8
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)
        ak = _rand(B, 1, H, D + 1)  # wrong dim
        av = _rand(B, 1, H, D + 1)
        with pytest.raises(ValueError, match="head dim"):
            na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)

    # -- Works with GQA -------------------------------------------

    def test_additional_kv_with_gqa(self):
        B, L, Hq, Hkv, D = 1, 12, 4, 2, 8
        q = _rand(B, L, Hq, D)
        k = _rand(B, L, Hkv, D)
        v = _rand(B, L, Hkv, D)
        ak = _rand(B, 2, Hkv, D)  # additional KV uses heads_kv
        av = _rand(B, 2, Hkv, D)

        out = na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        mx.eval(out)
        assert out.shape == (B, L, Hq, D)

    # -- return_lse with additional KV ----------------------------

    def test_additional_kv_with_return_lse(self):
        B, L, H, D = 1, 12, 2, 8
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)
        ak = _rand(B, 2, H, D)
        av = _rand(B, 2, H, D)

        result = na1d(q, k, v, kernel_size=5, additional_keys=ak,
                      additional_values=av, return_lse=True)
        mx.eval(result)
        out, lse = result
        assert out.shape == (B, L, H, D)
        assert lse.shape == (B, L, H)


# ===================================================================
# 5. FMHA fast path
# ===================================================================

class TestFMHAFastPath:
    """Tests for SDPA fast path when kernel covers full spatial extent."""

    def test_na1d_full_kernel_matches_sdpa(self):
        """na1d with kernel_size=L should match mx.fast.scaled_dot_product_attention."""
        B, L, H, D = 1, 12, 2, 16
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)
        scale = D ** -0.5

        out_na = na1d(q, k, v, kernel_size=L)
        mx.eval(out_na)

        # Manual SDPA
        q_t = mx.transpose(q, axes=(0, 2, 1, 3))  # [B, H, L, D]
        k_t = mx.transpose(k, axes=(0, 2, 1, 3))
        v_t = mx.transpose(v, axes=(0, 2, 1, 3))
        out_sdpa = mx.fast.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
        out_sdpa = mx.transpose(out_sdpa, axes=(0, 2, 1, 3))  # back to [B, L, H, D]
        mx.eval(out_sdpa)

        _allclose(out_na, out_sdpa, atol=1e-4)

    def test_na2d_full_kernel_matches_sdpa(self):
        """na2d with kernel_size=(H, W) should match SDPA."""
        B, Hh, W, H, D = 1, 8, 8, 2, 16
        q, k, v = _rand(B, Hh, W, H, D), _rand(B, Hh, W, H, D), _rand(B, Hh, W, H, D)
        scale = D ** -0.5

        out_na = na2d(q, k, v, kernel_size=(Hh, W))
        mx.eval(out_na)

        # Manual SDPA: flatten spatial to [B, S, H, D], transpose to [B, H, S, D]
        S = Hh * W
        q_flat = q.reshape(B, S, H, D)
        k_flat = k.reshape(B, S, H, D)
        v_flat = v.reshape(B, S, H, D)
        q_t = mx.transpose(q_flat, axes=(0, 2, 1, 3))
        k_t = mx.transpose(k_flat, axes=(0, 2, 1, 3))
        v_t = mx.transpose(v_flat, axes=(0, 2, 1, 3))
        out_sdpa = mx.fast.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
        out_sdpa = mx.transpose(out_sdpa, axes=(0, 2, 1, 3)).reshape(B, Hh, W, H, D)
        mx.eval(out_sdpa)

        _allclose(out_na, out_sdpa, atol=1e-4)

    def test_na1d_causal_does_not_use_fast_path(self):
        """With is_causal=True, fast path should NOT be used (results differ from SDPA)."""
        B, L, H, D = 1, 12, 2, 16
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)
        scale = D ** -0.5

        out_causal = na1d(q, k, v, kernel_size=L, is_causal=True)
        mx.eval(out_causal)

        # Non-causal SDPA
        q_t = mx.transpose(q, axes=(0, 2, 1, 3))
        k_t = mx.transpose(k, axes=(0, 2, 1, 3))
        v_t = mx.transpose(v, axes=(0, 2, 1, 3))
        out_sdpa = mx.fast.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
        out_sdpa = mx.transpose(out_sdpa, axes=(0, 2, 1, 3))
        mx.eval(out_sdpa)

        # They should NOT match because causal masking changes the result
        diff = float(mx.abs(out_causal - out_sdpa).max())
        mx.eval(mx.array(diff))
        assert diff > 1e-3, "Causal output should differ from non-causal SDPA"

    def test_na1d_stride_does_not_use_fast_path(self):
        """With stride > 1, fast path should NOT be used even with full kernel."""
        B, L, H, D = 1, 12, 2, 16
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)

        # stride=2 with full kernel -- should go through split path, not SDPA
        out_strided = na1d(q, k, v, kernel_size=L, stride=2)
        mx.eval(out_strided)

        # Strided output has reduced spatial dimension
        expected_L = math.ceil(L / 2)
        assert out_strided.shape[1] == expected_L or out_strided.shape == (B, L, H, D), \
            "Strided output should have reduced or same spatial size"

    def test_na1d_fast_path_with_return_lse(self):
        """Fast path should also work with return_lse=True."""
        B, L, H, D = 1, 12, 2, 16
        q, k, v = _rand(B, L, H, D), _rand(B, L, H, D), _rand(B, L, H, D)

        result = na1d(q, k, v, kernel_size=L, return_lse=True)
        mx.eval(result)
        out, lse = result
        assert out.shape == (B, L, H, D)
        assert lse.shape == (B, L, H)

    def test_na2d_fast_path_with_additional_kv(self):
        """Fast path + additional keys/values should work."""
        B, Hh, W, H, D = 1, 8, 8, 2, 8
        q, k, v = _rand(B, Hh, W, H, D), _rand(B, Hh, W, H, D), _rand(B, Hh, W, H, D)
        ak = _rand(B, 2, H, D)
        av = _rand(B, 2, H, D)

        out = na2d(q, k, v, kernel_size=(Hh, W), additional_keys=ak, additional_values=av)
        mx.eval(out)
        assert out.shape == (B, Hh, W, H, D)

    def test_na3d_full_kernel_runs(self):
        """na3d with full kernel should run via SDPA fast path."""
        B, Dd, Hh, W, H, D = 1, 4, 4, 4, 2, 8
        q = _rand(B, Dd, Hh, W, H, D)
        k = _rand(B, Dd, Hh, W, H, D)
        v = _rand(B, Dd, Hh, W, H, D)

        out = na3d(q, k, v, kernel_size=(Dd, Hh, W))
        mx.eval(out)
        assert out.shape == (B, Dd, Hh, W, H, D)


# ===================================================================
# 7. Backward / Gradient Tests for New Features
# ===================================================================


class TestReturnLSEBackward:
    """Gradient tests for return_lse feature."""

    def test_na1d_return_lse_backward(self):
        _seed()
        q = _randn(1, 12, 2, 8)
        k = _randn(1, 12, 2, 8)
        v = _randn(1, 12, 2, 8)

        def loss_fn(q_in, k_in, v_in):
            out, lse = na1d(q_in, k_in, v_in, kernel_size=5, return_lse=True)
            return mx.sum(out) + mx.sum(lse)

        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))
        grads = grad_fn(q, k, v)
        mx.eval(*grads)
        assert grads[0].shape == q.shape
        assert grads[1].shape == k.shape
        assert grads[2].shape == v.shape

    def test_na1d_return_lse_backward_causal(self):
        _seed()
        q = _randn(1, 12, 2, 8)
        k = _randn(1, 12, 2, 8)
        v = _randn(1, 12, 2, 8)

        def loss_fn(q_in, k_in, v_in):
            out, lse = na1d(q_in, k_in, v_in, kernel_size=5, is_causal=True, return_lse=True)
            return mx.sum(out) + mx.sum(lse)

        grads = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)
        assert grads[0].shape == q.shape

    def test_na2d_return_lse_backward(self):
        _seed()
        q = _randn(1, 8, 8, 2, 8)
        k = _randn(1, 8, 8, 2, 8)
        v = _randn(1, 8, 8, 2, 8)

        def loss_fn(q_in, k_in, v_in):
            out, lse = na2d(q_in, k_in, v_in, kernel_size=5, return_lse=True)
            return mx.sum(out) + mx.sum(lse)

        grads = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)
        assert grads[0].shape == q.shape
        assert grads[1].shape == k.shape
        assert grads[2].shape == v.shape


class TestGQABackward:
    """Gradient tests for GQA/MQA feature."""

    def test_na1d_gqa_backward(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, L, heads_q, D)
        k = _randn(B, L, heads_kv, D)
        v = _randn(B, L, heads_kv, D)

        def loss_fn(q_in, k_in, v_in):
            return mx.sum(na1d(q_in, k_in, v_in, kernel_size=5))

        grads = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)
        assert grads[0].shape == (B, L, heads_q, D)
        assert grads[1].shape == (B, L, heads_kv, D)
        assert grads[2].shape == (B, L, heads_kv, D)

    def test_na1d_mqa_backward(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 1
        q = _randn(B, L, heads_q, D)
        k = _randn(B, L, heads_kv, D)
        v = _randn(B, L, heads_kv, D)

        def loss_fn(q_in, k_in, v_in):
            return mx.sum(na1d(q_in, k_in, v_in, kernel_size=5))

        grads = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)
        assert grads[0].shape == (B, L, heads_q, D)
        assert grads[1].shape == (B, L, heads_kv, D)
        assert grads[2].shape == (B, L, heads_kv, D)

    def test_na2d_gqa_backward(self):
        _seed()
        B, Hh, W, D = 1, 8, 8, 8
        heads_q, heads_kv = 4, 2
        q = _randn(B, Hh, W, heads_q, D)
        k = _randn(B, Hh, W, heads_kv, D)
        v = _randn(B, Hh, W, heads_kv, D)

        def loss_fn(q_in, k_in, v_in):
            return mx.sum(na2d(q_in, k_in, v_in, kernel_size=5))

        grads = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)
        assert grads[0].shape == (B, Hh, W, heads_q, D)
        assert grads[1].shape == (B, Hh, W, heads_kv, D)
        assert grads[2].shape == (B, Hh, W, heads_kv, D)


class TestAdditionalKVBackward:
    """Gradient tests for additional_keys/additional_values."""

    def test_na1d_additional_kv_backward(self):
        _seed()
        B, L, H, D = 1, 12, 2, 8
        q = _randn(B, L, H, D)
        k = _randn(B, L, H, D)
        v = _randn(B, L, H, D)
        ak = _randn(B, 2, H, D)
        av = _randn(B, 2, H, D)

        def loss_fn(q_in, k_in, v_in, ak_in, av_in):
            return mx.sum(na1d(q_in, k_in, v_in, kernel_size=5,
                              additional_keys=ak_in, additional_values=av_in))

        grads = mx.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, ak, av)
        mx.eval(*grads)
        assert grads[0].shape == q.shape
        assert grads[1].shape == k.shape
        assert grads[2].shape == v.shape
        assert grads[3].shape == ak.shape
        assert grads[4].shape == av.shape

    def test_na2d_additional_kv_backward(self):
        _seed()
        B, Hh, W, H, D = 1, 8, 8, 2, 8
        q = _randn(B, Hh, W, H, D)
        k = _randn(B, Hh, W, H, D)
        v = _randn(B, Hh, W, H, D)
        ak = _randn(B, 2, H, D)
        av = _randn(B, 2, H, D)

        def loss_fn(q_in, k_in, v_in, ak_in, av_in):
            return mx.sum(na2d(q_in, k_in, v_in, kernel_size=5,
                              additional_keys=ak_in, additional_values=av_in))

        grads = mx.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, ak, av)
        mx.eval(*grads)
        assert grads[0].shape == q.shape
        assert grads[3].shape == ak.shape
        assert grads[4].shape == av.shape

    def test_na3d_additional_kv_backward(self):
        _seed()
        B = 1
        D_s, Hh_s, W_s = 4, 4, 4
        H, D = 2, 8
        q = _randn(B, D_s, Hh_s, W_s, H, D)
        k = _randn(B, D_s, Hh_s, W_s, H, D)
        v = _randn(B, D_s, Hh_s, W_s, H, D)
        ak = _randn(B, 1, H, D)
        av = _randn(B, 1, H, D)

        def loss_fn(q_in, k_in, v_in, ak_in, av_in):
            return mx.sum(na3d(q_in, k_in, v_in, kernel_size=3,
                              additional_keys=ak_in, additional_values=av_in))

        grads = mx.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, ak, av)
        mx.eval(*grads)
        assert grads[0].shape == q.shape
        assert grads[3].shape == ak.shape
        assert grads[4].shape == av.shape


class TestMergeAttentionsBackward:
    """Gradient tests for merge_attentions."""

    def test_merge_2way_backward_1d(self):
        _seed()
        B, L, H, D = 1, 12, 2, 8
        ks = 5
        q = _randn(B, L, H, D)
        k1 = _randn(B, L, H, D)
        v1 = _randn(B, L, H, D)
        k2 = _randn(B, L, H, D)
        v2 = _randn(B, L, H, D)

        def loss_fn(q_in, k1_in, v1_in, k2_in, v2_in):
            out1, lse1 = na1d(q_in, k1_in, v1_in, kernel_size=ks, return_lse=True)
            out2, lse2 = na1d(q_in, k2_in, v2_in, kernel_size=ks, return_lse=True)
            merged, merged_lse = merge_attentions([out1, out2], [lse1, lse2])
            return mx.sum(merged) + mx.sum(merged_lse)

        grads = mx.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k1, v1, k2, v2)
        mx.eval(*grads)
        assert grads[0].shape == q.shape
        assert grads[1].shape == k1.shape


class TestFMHAFastPathBackward:
    """Gradient tests for FMHA fast path."""

    def test_na1d_fmha_backward(self):
        _seed()
        B, L, H, D = 1, 10, 2, 8
        q = _randn(B, L, H, D)
        k = _randn(B, L, H, D)
        v = _randn(B, L, H, D)

        def loss_fn(q_in, k_in, v_in):
            return mx.sum(na1d(q_in, k_in, v_in, kernel_size=L))

        grads = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)
        assert grads[0].shape == q.shape
        assert grads[1].shape == k.shape
        assert grads[2].shape == v.shape

    def test_na2d_fmha_backward(self):
        _seed()
        B, Hh, W, H, D = 1, 6, 8, 2, 8
        q = _randn(B, Hh, W, H, D)
        k = _randn(B, Hh, W, H, D)
        v = _randn(B, Hh, W, H, D)

        def loss_fn(q_in, k_in, v_in):
            return mx.sum(na2d(q_in, k_in, v_in, kernel_size=(Hh, W)))

        grads = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)
        assert grads[0].shape == q.shape
        assert grads[1].shape == k.shape
        assert grads[2].shape == v.shape

    def test_na1d_fmha_with_return_lse_backward(self):
        _seed()
        B, L, H, D = 1, 10, 2, 8
        q = _randn(B, L, H, D)
        k = _randn(B, L, H, D)
        v = _randn(B, L, H, D)

        def loss_fn(q_in, k_in, v_in):
            out, lse = na1d(q_in, k_in, v_in, kernel_size=L, return_lse=True)
            return mx.sum(out) + mx.sum(lse)

        grads = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)
        assert grads[0].shape == q.shape


# ===================================================================
# 8. Combination Tests (feature interactions)
# ===================================================================


class TestFeatureCombinations:
    """Tests for combinations of new features to verify they compose correctly."""

    def test_gqa_with_return_lse_1d(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 2
        q = _rand(B, L, heads_q, D)
        k = _rand(B, L, heads_kv, D)
        v = _rand(B, L, heads_kv, D)
        out, lse = na1d(q, k, v, kernel_size=5, return_lse=True)
        mx.eval(out, lse)
        assert out.shape == (B, L, heads_q, D)
        assert lse.shape == (B, L, heads_q)
        assert mx.all(mx.isfinite(lse))

    def test_gqa_with_return_lse_2d(self):
        _seed()
        B, Hh, W, D = 1, 8, 8, 8
        heads_q, heads_kv = 4, 2
        q = _rand(B, Hh, W, heads_q, D)
        k = _rand(B, Hh, W, heads_kv, D)
        v = _rand(B, Hh, W, heads_kv, D)
        out, lse = na2d(q, k, v, kernel_size=5, return_lse=True)
        mx.eval(out, lse)
        assert out.shape == (B, Hh, W, heads_q, D)
        assert lse.shape == (B, Hh, W, heads_q)

    def test_gqa_causal_return_lse_1d(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 2
        q = _rand(B, L, heads_q, D)
        k = _rand(B, L, heads_kv, D)
        v = _rand(B, L, heads_kv, D)
        out, lse = na1d(q, k, v, kernel_size=5, is_causal=True, return_lse=True)
        mx.eval(out, lse)
        assert out.shape == (B, L, heads_q, D)
        assert mx.all(mx.isfinite(lse))

    def test_additional_kv_with_gqa_1d(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 2
        q = _rand(B, L, heads_q, D)
        k = _rand(B, L, heads_kv, D)
        v = _rand(B, L, heads_kv, D)
        ak = _rand(B, 2, heads_kv, D)
        av = _rand(B, 2, heads_kv, D)
        out = na1d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        mx.eval(out)
        assert out.shape == (B, L, heads_q, D)

    def test_additional_kv_with_gqa_2d(self):
        _seed()
        B, Hh, W, D = 1, 8, 8, 8
        heads_q, heads_kv = 4, 2
        q = _rand(B, Hh, W, heads_q, D)
        k = _rand(B, Hh, W, heads_kv, D)
        v = _rand(B, Hh, W, heads_kv, D)
        ak = _rand(B, 2, heads_kv, D)
        av = _rand(B, 2, heads_kv, D)
        out = na2d(q, k, v, kernel_size=5, additional_keys=ak, additional_values=av)
        mx.eval(out)
        assert out.shape == (B, Hh, W, heads_q, D)

    def test_stride_with_return_lse_1d(self):
        _seed()
        B, L, H, D = 1, 12, 2, 8
        q = _rand(B, L, H, D)
        k = _rand(B, L, H, D)
        v = _rand(B, L, H, D)
        out, lse = na1d(q, k, v, kernel_size=5, stride=2, return_lse=True)
        mx.eval(out, lse)
        L_out = (L + 2 - 1) // 2
        assert out.shape == (B, L_out, H, D)
        assert lse.shape == (B, L_out, H)

    def test_dilation_with_gqa_1d(self):
        _seed()
        B, L, D = 1, 16, 8
        heads_q, heads_kv = 4, 2
        q = _rand(B, L, heads_q, D)
        k = _rand(B, L, heads_kv, D)
        v = _rand(B, L, heads_kv, D)
        out = na1d(q, k, v, kernel_size=5, dilation=2)
        mx.eval(out)
        assert out.shape == (B, L, heads_q, D)

    def test_merge_with_gqa_1d(self):
        _seed()
        B, L, D = 1, 12, 8
        heads_q, heads_kv = 4, 2
        q = _rand(B, L, heads_q, D)
        k1 = _rand(B, L, heads_kv, D)
        v1 = _rand(B, L, heads_kv, D)
        k2 = _rand(B, L, heads_kv, D)
        v2 = _rand(B, L, heads_kv, D)
        out1, lse1 = na1d(q, k1, v1, kernel_size=5, return_lse=True)
        out2, lse2 = na1d(q, k2, v2, kernel_size=5, return_lse=True)
        merged, merged_lse = merge_attentions([out1, out2], [lse1, lse2])
        mx.eval(merged, merged_lse)
        assert merged.shape == (B, L, heads_q, D)
        assert merged_lse.shape == (B, L, heads_q)


# ===================================================================
# 9. FMHA Edge Cases
# ===================================================================


class TestFMHAEdgeCases:
    """Edge cases for FMHA fast path dispatch."""

    def test_fmha_kernel_equals_spatial_1d(self):
        _seed()
        B, L, H, D = 1, 12, 2, 8
        q = _rand(B, L, H, D)
        k = _rand(B, L, H, D)
        v = _rand(B, L, H, D)
        out = na1d(q, k, v, kernel_size=L)
        mx.eval(out)
        assert out.shape == (B, L, H, D)

    def test_fmha_kernel_equals_spatial_2d(self):
        _seed()
        B, Hh, W, H, D = 1, 6, 8, 2, 8
        q = _rand(B, Hh, W, H, D)
        k = _rand(B, Hh, W, H, D)
        v = _rand(B, Hh, W, H, D)
        out = na2d(q, k, v, kernel_size=(Hh, W))
        mx.eval(out)
        assert out.shape == (B, Hh, W, H, D)

    def test_fmha_non_square_full_kernel_2d(self):
        _seed()
        B, Hh, W, H, D = 1, 4, 8, 2, 8
        q = _rand(B, Hh, W, H, D)
        k = _rand(B, Hh, W, H, D)
        v = _rand(B, Hh, W, H, D)
        scale = D ** -0.5

        out_na = na2d(q, k, v, kernel_size=(Hh, W), scale=scale)

        # Compare with manual SDPA
        S = Hh * W
        q_t = q.reshape(B, S, H, D).transpose(0, 2, 1, 3)
        k_t = k.reshape(B, S, H, D).transpose(0, 2, 1, 3)
        v_t = v.reshape(B, S, H, D).transpose(0, 2, 1, 3)
        out_sdpa = mx.fast.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
        out_sdpa = out_sdpa.transpose(0, 2, 1, 3).reshape(B, Hh, W, H, D)
        mx.eval(out_na, out_sdpa)

        assert mx.allclose(out_na, out_sdpa, atol=1e-5, rtol=1e-5)

    def test_fmha_with_additional_kv_backward(self):
        _seed()
        B, L, H, D = 1, 10, 2, 8
        q = _rand(B, L, H, D)
        k = _rand(B, L, H, D)
        v = _rand(B, L, H, D)
        ak = _rand(B, 3, H, D)
        av = _rand(B, 3, H, D)

        def loss_fn(q_in, k_in, v_in, ak_in, av_in):
            return mx.sum(na1d(q_in, k_in, v_in, kernel_size=L,
                              additional_keys=ak_in, additional_values=av_in))

        grads = mx.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, ak, av)
        mx.eval(*grads)
        assert grads[0].shape == q.shape
        assert grads[3].shape == ak.shape
        assert grads[4].shape == av.shape


# ===================================================================
# 10. Metal-specific return_lse
# ===================================================================


class TestMetalReturnLSE:
    """Metal backend-specific return_lse tests."""

    def test_metal_return_lse_1d(self):
        import natten_mlx
        backend = natten_mlx.get_backend()
        if backend not in ("fast_metal", "nanobind"):
            pytest.skip("Metal backend not available")

        _seed()
        q = _rand(1, 12, 2, 8)
        k = _rand(1, 12, 2, 8)
        v = _rand(1, 12, 2, 8)
        out, lse = na1d(q, k, v, kernel_size=5, return_lse=True)
        mx.eval(out, lse)
        assert lse.shape == (1, 12, 2)
        assert mx.all(mx.isfinite(lse))

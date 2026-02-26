"""Finite-difference gradient checks for natten-mlx functional API.

MLX does not have an equivalent of torch.autograd.gradcheck, so we implement
central-difference validation manually.  Because MLX only supports float32,
we use looser tolerances (eps=1e-3, atol=1e-2) compared to the float64
gradchecks in natten-mps.

For each test we:
  1. Compute analytical gradients via mx.grad.
  2. Perturb a random subset of elements with central differences.
  3. Assert the two agree within tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.functional import na1d, na1d_av, na1d_qk, na2d, na3d

_EPS = 1e-3
_ATOL = 1e-2
_RTOL = 1e-2
_N_PROBES = 10  # elements to check per tensor
_SEED = 42


def _finite_diff_check(loss_fn, arrays, *, argnums=None, n_probes=_N_PROBES,
                        eps=_EPS, atol=_ATOL, rtol=_RTOL):
    """Compare mx.grad analytical gradients against central differences.

    Parameters
    ----------
    loss_fn : callable
        Scalar-valued function of ``arrays``.
    arrays : list[mx.array]
        Input arrays (will not be mutated).
    argnums : tuple[int, ...] | None
        Which arguments to differentiate w.r.t. (default: all).
    """
    if argnums is None:
        argnums = tuple(range(len(arrays)))

    # --- analytical gradients ---
    grad_fn = mx.grad(loss_fn, argnums=argnums)
    analytical = grad_fn(*arrays)
    if not isinstance(analytical, (list, tuple)):
        analytical = (analytical,)
    mx.eval(*analytical)

    rng = np.random.RandomState(_SEED)

    for idx, argnum in enumerate(argnums):
        arr_np = np.array(arrays[argnum], copy=True)
        ana_np = np.array(analytical[idx])
        flat_size = arr_np.size
        probe_indices = rng.choice(flat_size, size=min(n_probes, flat_size),
                                   replace=False)

        for pi in probe_indices:
            multi_idx = np.unravel_index(pi, arr_np.shape)

            # f(x + eps)
            arr_plus = arr_np.copy()
            arr_plus[multi_idx] += eps
            inputs_plus = list(arrays)
            inputs_plus[argnum] = mx.array(arr_plus)
            f_plus = loss_fn(*inputs_plus)

            # f(x - eps)
            arr_minus = arr_np.copy()
            arr_minus[multi_idx] -= eps
            inputs_minus = list(arrays)
            inputs_minus[argnum] = mx.array(arr_minus)
            f_minus = loss_fn(*inputs_minus)

            mx.eval(f_plus, f_minus)
            numerical = (f_plus.item() - f_minus.item()) / (2 * eps)
            ana_val = ana_np[multi_idx]

            diff = abs(numerical - ana_val)
            tol = atol + rtol * abs(numerical)
            assert diff < tol, (
                f"arg {argnum}, index {multi_idx}: "
                f"analytical={ana_val:.6f}, numerical={numerical:.6f}, "
                f"diff={diff:.6f}, tol={tol:.6f}"
            )


def _rand(*shape):
    mx.random.seed(_SEED)
    return mx.random.normal(shape) * 0.5


# -------------------------------------------------------------------
# Fused forward
# -------------------------------------------------------------------


class TestFusedGradcheck:

    def test_na1d_gradcheck(self):
        q, k, v = _rand(1, 8, 2, 4), _rand(1, 8, 2, 4), _rand(1, 8, 2, 4)

        def loss_fn(q_in, k_in, v_in):
            return mx.sum(na1d(q_in, k_in, v_in, kernel_size=3))

        _finite_diff_check(loss_fn, [q, k, v])

    def test_na2d_gradcheck(self):
        q = _rand(1, 4, 4, 2, 4)
        k = _rand(1, 4, 4, 2, 4)
        v = _rand(1, 4, 4, 2, 4)

        def loss_fn(q_in, k_in, v_in):
            return mx.sum(na2d(q_in, k_in, v_in, kernel_size=3))

        _finite_diff_check(loss_fn, [q, k, v])

    def test_na3d_gradcheck(self):
        q = _rand(1, 4, 4, 4, 2, 4)
        k = _rand(1, 4, 4, 4, 2, 4)
        v = _rand(1, 4, 4, 4, 2, 4)

        def loss_fn(q_in, k_in, v_in):
            return mx.sum(na3d(q_in, k_in, v_in, kernel_size=3))

        _finite_diff_check(loss_fn, [q, k, v])


# -------------------------------------------------------------------
# Split QK / AV
# -------------------------------------------------------------------


class TestSplitGradcheck:

    def test_na1d_qk_gradcheck(self):
        q, k = _rand(1, 8, 2, 4), _rand(1, 8, 2, 4)

        def loss_fn(q_in, k_in):
            return mx.sum(na1d_qk(q_in, k_in, kernel_size=3))

        _finite_diff_check(loss_fn, [q, k])

    def test_na1d_av_gradcheck(self):
        mx.random.seed(_SEED)
        attn_raw = mx.random.normal((1, 8, 2, 3)) * 0.5
        attn = mx.softmax(attn_raw, axis=-1)
        v = _rand(1, 8, 2, 4)

        def loss_fn(a_in, v_in):
            return mx.sum(na1d_av(a_in, v_in, kernel_size=3))

        _finite_diff_check(loss_fn, [attn, v])


# -------------------------------------------------------------------
# Fused with features (causal, dilation)
# -------------------------------------------------------------------


class TestFeatureGradcheck:

    def test_na1d_causal_gradcheck(self):
        q, k, v = _rand(1, 8, 2, 4), _rand(1, 8, 2, 4), _rand(1, 8, 2, 4)

        def loss_fn(q_in, k_in, v_in):
            return mx.sum(na1d(q_in, k_in, v_in, kernel_size=3, is_causal=True))

        _finite_diff_check(loss_fn, [q, k, v])

    def test_na1d_dilation_gradcheck(self):
        q, k, v = _rand(1, 12, 2, 4), _rand(1, 12, 2, 4), _rand(1, 12, 2, 4)

        def loss_fn(q_in, k_in, v_in):
            return mx.sum(na1d(q_in, k_in, v_in, kernel_size=3, dilation=2))

        _finite_diff_check(loss_fn, [q, k, v])

    def test_na2d_causal_gradcheck(self):
        q = _rand(1, 4, 4, 2, 4)
        k = _rand(1, 4, 4, 2, 4)
        v = _rand(1, 4, 4, 2, 4)

        def loss_fn(q_in, k_in, v_in):
            return mx.sum(na2d(q_in, k_in, v_in, kernel_size=3, is_causal=True))

        _finite_diff_check(loss_fn, [q, k, v])

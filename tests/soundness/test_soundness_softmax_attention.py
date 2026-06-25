"""Soundness of the correlated row-softmax bound (attention weights).

softmax over the key axis must be soundly over-approximated: every concrete
``softmax(S)`` for logits ``S`` in the box must lie in the output set. We sample
logit boxes across sign regimes, with a per-head [heads, L_q, L_k] layout and
softmax over the last axis (and a non-default axis), and check containment.
"""

import numpy as np
import pytest

from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops.softmax_attention_reach import (
    correlated_row_softmax_bounds,
    softmax_attn_box,
    softmax_attn_zono,
    softmax_attn_star,
)

SEED = 20260625
N_SAMPLES = 300
TOL = 1e-7


def _softmax_np(x, axis):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _star_contains(star, pt, tol=1e-7):
    """Authoritative LP membership, with a point-star (nVar==0) guard.

    n2v's Star.contains LP path does not handle zero-variable point stars; for
    those the set is exactly {center}, so test containment directly.
    """
    if star.nVar == 0:
        return bool(np.all(np.abs(pt - np.asarray(star.V[:, 0])) <= tol))
    return star.contains(pt, method="lp")


def _logit_box(rng, shape, regime):
    if regime == "pos":
        lo = rng.uniform(0.0, 3.0, size=shape)
        hi = lo + rng.uniform(0.0, 3.0, size=shape)
    elif regime == "neg":
        hi = rng.uniform(-3.0, 0.0, size=shape)
        lo = hi - rng.uniform(0.0, 3.0, size=shape)
    elif regime == "wide":
        lo = rng.uniform(-6.0, 0.0, size=shape)
        hi = lo + rng.uniform(0.0, 12.0, size=shape)
    else:  # mixed
        lo = rng.uniform(-3.0, -0.1, size=shape)
        hi = rng.uniform(0.1, 3.0, size=shape)
    return lo.astype(np.float64), hi.astype(np.float64)


CASES = [
    ((3, 5, 5), -1),
    ((3, 17, 17), -1),
    ((5, 5), -1),
    ((4, 3), 0),     # softmax over a non-default axis
    ((3, 5, 1), -1),  # degenerate single key -> softmax == 1 exactly
]
REGIMES = ["pos", "neg", "wide", "mixed"]


@pytest.mark.parametrize("shape,axis", CASES)
@pytest.mark.parametrize("regime", REGIMES)
def test_softmax_box_sound(shape, axis, regime):
    rng = np.random.default_rng(SEED)
    s_lo, s_hi = _logit_box(rng, shape, regime)
    out = softmax_attn_box([Box(s_lo.reshape(-1), s_hi.reshape(-1))], shape, axis)[0]
    lo, hi = np.asarray(out.lb).reshape(-1), np.asarray(out.ub).reshape(-1)
    assert np.all(lo >= -TOL) and np.all(hi <= 1.0 + TOL)
    for _ in range(N_SAMPLES):
        s = rng.uniform(s_lo, s_hi)
        a = _softmax_np(s, axis).reshape(-1)
        assert np.all(a >= lo - TOL) and np.all(a <= hi + TOL)


@pytest.mark.parametrize("shape,axis", CASES)
@pytest.mark.parametrize("regime", REGIMES)
def test_softmax_star_sound(shape, axis, regime):
    rng = np.random.default_rng(SEED + 1)
    s_lo, s_hi = _logit_box(rng, shape, regime)
    out = softmax_attn_star(
        [Star.from_bounds(s_lo.reshape(-1), s_hi.reshape(-1))], shape, axis)[0]
    for _ in range(N_SAMPLES):
        s = rng.uniform(s_lo, s_hi)
        a = _softmax_np(s, axis).reshape(-1)
        assert _star_contains(out, a)


@pytest.mark.parametrize("shape,axis", CASES)
@pytest.mark.parametrize("regime", REGIMES)
def test_softmax_zono_sound(shape, axis, regime):
    rng = np.random.default_rng(SEED + 2)
    s_lo, s_hi = _logit_box(rng, shape, regime)
    out = softmax_attn_zono(
        [Zono.from_bounds(s_lo.reshape(-1), s_hi.reshape(-1))], shape, axis)[0]
    lo, hi = (np.asarray(b).reshape(-1) for b in out.get_bounds())
    for _ in range(N_SAMPLES):
        s = rng.uniform(s_lo, s_hi)
        a = _softmax_np(s, axis).reshape(-1)
        assert np.all(a >= lo - TOL) and np.all(a <= hi + TOL)


def test_single_key_is_exact_one():
    # L_k = 1 -> softmax is identically 1, bound must be [1,1].
    s_lo = np.array([-2.0, 0.5, 3.0])
    s_hi = np.array([1.0, 2.0, 4.0])
    a_lb, a_ub = correlated_row_softmax_bounds(
        s_lo.reshape(3, 1), s_hi.reshape(3, 1), axis=-1)
    assert np.allclose(a_lb, 1.0) and np.allclose(a_ub, 1.0)


def test_row_bounds_bracket_uniform_attention():
    # Equal logits -> uniform softmax 1/n; must lie within the bound.
    n = 5
    s_lo = np.full((1, n), -0.5)
    s_hi = np.full((1, n), 0.5)
    a_lb, a_ub = correlated_row_softmax_bounds(s_lo, s_hi, axis=-1)
    assert np.all(a_lb <= 1.0 / n + 1e-9) and np.all(a_ub >= 1.0 / n - 1e-9)

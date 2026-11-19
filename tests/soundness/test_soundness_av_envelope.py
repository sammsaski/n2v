"""Soundness of the symbolic A @ V envelope (``av_envelope_star``).

``O = A @ V`` with ``A`` an interval ``[a_lb, a_ub]`` of attention weights
(``a_lb >= 0``) of shape (heads, rows, keys) and ``V`` a value Star of shape
(heads, keys, columns). The returned Star must hold ``A @ V`` for every ``A``
in the box and every member ``V``. No model entry calls this function today
(a model answers an A @ V product by the set @ set MatMul handler), so the
cells call it directly.

Reference: for a member point of V, each output column ``d`` is least when
``A[h, m, k]`` is ``a_lb`` where ``V[h, k, d] >= 0`` and ``a_ub`` where it is
negative, and greatest the other way round; that choice is made for all rows
and heads at once, giving two output vectors per column. Products in
fractions of dyadic data, so the float64 outputs are exact; membership by LP
on the returned Star's rows (``assert_covered``, the oracle's own LP, not
n2v's ``contains``). The envelope's slopes are divisions, so the slack is the
LP tolerance.
"""

from fractions import Fraction

import numpy as np
import pytest

from n2v.nn.layer_ops.bilinear_matmul_reach import av_envelope_star
from n2v.sets import Star
from tests.cell_support import assert_covered, constrained_star

H, M, K, D = 2, 3, 4, 5


def _vbox(regime, seed):
    """Dyadic bounds of V (multiples of 1/8) in one sign regime: every entry
    at or above zero, at or below zero, or crossing zero."""
    rng = np.random.default_rng(seed)
    n = H * K * D
    if regime == "pos":
        lo = rng.integers(0, 9, n) / 8.0
        return lo, lo + rng.integers(1, 9, n) / 8.0
    if regime == "neg":
        hi = -rng.integers(0, 9, n) / 8.0
        return hi - rng.integers(1, 9, n) / 8.0, hi
    return -rng.integers(1, 9, n) / 8.0, rng.integers(1, 9, n) / 8.0


def _abox(seed):
    rng = np.random.default_rng(seed)
    a_lb = rng.integers(0, 5, (H, M, K)) / 8.0
    return a_lb, a_lb + rng.integers(0, 4, (H, M, K)) / 8.0


def _extreme_outputs(a_lb, a_ub, v_points):
    """Exact ``A @ V`` (fractions, then float64, exact for dyadic data) at
    each member point of V, with A at the corner that makes each output
    column least or greatest."""
    outputs = []
    for v in v_points:
        V = np.array([Fraction(float(x)) for x in np.ravel(v)],
                     dtype=object).reshape(H, K, D)
        signs = np.asarray(v, dtype=np.float64).reshape(H, K, D) >= 0
        for d in range(D):
            for low in (True, False):
                # (H, 1, K) sign pattern of column d broadcast over the rows
                take_lb = (signs[:, None, :, d] == low)
                A = np.where(take_lb, a_lb, a_ub)
                Aq = np.array([Fraction(float(x)) for x in A.ravel()],
                              dtype=object).reshape(H, M, K)
                outputs.append([float(x) for x in
                                np.matmul(Aq, V).reshape(-1)])
    return outputs


def _value_points(lo, hi, seed, n=12):
    rng = np.random.default_rng(seed)
    points = [lo, hi]
    for _ in range(n - 2):
        points.append(np.where(rng.integers(0, 2, lo.size) > 0, lo, hi))
    return points


@pytest.mark.parametrize("regime", ["pos", "neg", "mixed", "mixed-cut"])
def test_av_envelope_sound(regime):
    """Every ``A @ V`` with A in its box and V a member of the value Star is
    in the returned Star, for values at or above zero, at or below zero,
    crossing zero, and crossing zero with a binding row that is not a box
    row (``mixed-cut``: the value Star's own rows are kept in the output and
    must be kept right). Heads 2, rows 3, keys 4, columns 5, so the
    (heads, rows, keys, columns) layout is checked on unequal axes.

    Catches: a slope or bias taken from the wrong attention bound in any
    regime, the bias summed over the wrong axis, the new variables' box
    taken from the wrong value bound, the lower facet written with the upper
    facet's coefficients, and the value Star's rows written with the wrong
    sign (``mixed-cut``)."""
    lo, hi = _vbox("mixed" if regime == "mixed-cut" else regime, seed=3)
    if regime == "mixed-cut":
        v_star, v_points = constrained_star(lo, hi, n=12, seed=4)
    else:
        v_star = Star.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))
        v_points = _value_points(lo, hi, seed=4)
    a_lb, a_ub = _abox(seed=5)
    out = av_envelope_star(a_lb, a_ub, v_star, H, M, K, D)
    if out.dim != H * M * D:
        raise AssertionError(f"{regime}: the envelope has {out.dim} entries, "
                             f"A @ V has {H * M * D}")
    assert_covered([out], _extreme_outputs(a_lb, a_ub, v_points),
                   f"A @ V envelope, values {regime}")


def test_av_envelope_rejects_negative_a_lb():
    """Attention weights below zero break the envelope's slopes (it assumes
    the softmax invariant ``a_lb >= 0``) and must be refused with
    ``ValueError``, never answered."""
    v = Star.from_bounds(np.zeros((H * K * D, 1)), np.ones((H * K * D, 1)))
    a_lb = -0.125 * np.ones((H, M, K))
    a_ub = np.ones((H, M, K))
    with pytest.raises(ValueError):
        av_envelope_star(a_lb, a_ub, v, H, M, K, D)

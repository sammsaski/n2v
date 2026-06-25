"""Soundness of the symbolic A.V envelope (value-path-preserving attention output).

O = A @ V with A a concretized interval [a_lb,a_ub] (a_lb>=0) and V a symbolic
Star. The output Star must enclose every A@V for A in the box and V in the star.
We sample both independently across V sign regimes and assert containment.
"""

import numpy as np
import pytest

from n2v.sets import Star
from n2v.nn.layer_ops.bilinear_matmul_reach import av_envelope_star

SEED = 20260625
N_SAMPLES = 250
TOL = 1e-7
H, M, K, D = 2, 3, 4, 5


def _vbox(rng, regime):
    n = H * K * D
    if regime == "pos":
        lo = rng.uniform(0.0, 1.0, n); hi = lo + rng.uniform(0.0, 1.0, n)
    elif regime == "neg":
        hi = rng.uniform(-1.0, 0.0, n); lo = hi - rng.uniform(0.0, 1.0, n)
    else:  # mixed
        lo = rng.uniform(-1.0, -0.05, n); hi = rng.uniform(0.05, 1.0, n)
    return lo, hi


def _star_contains(star, pt, tol=1e-7):
    if star.nVar == 0:
        return bool(np.all(np.abs(pt - np.asarray(star.V[:, 0])) <= tol))
    return star.contains(pt, method="lp")


@pytest.mark.parametrize("regime", ["pos", "neg", "mixed"])
def test_av_envelope_sound(regime):
    rng = np.random.default_rng(SEED)
    vlo, vhi = _vbox(rng, regime)
    v_star = Star.from_bounds(vlo, vhi)
    # attention weights interval, a_lb >= 0
    a_lb = rng.uniform(0.0, 0.6, (H, M, K))
    a_ub = a_lb + rng.uniform(0.0, 0.4, (H, M, K))

    O = av_envelope_star(a_lb, a_ub, v_star, H, M, K, D)
    assert O.dim == H * M * D

    nviol = 0
    for _ in range(N_SAMPLES):
        V = rng.uniform(vlo, vhi).reshape(H, K, D)
        A = rng.uniform(a_lb, a_ub)
        true_O = np.einsum("hmk,hkd->hmd", A, V).reshape(-1)
        if not _star_contains(O, true_O):
            nviol += 1
    assert nviol == 0, f"{regime}: {nviol}/{N_SAMPLES} true outputs excluded"


def test_av_envelope_rejects_negative_a_lb():
    v = Star.from_bounds(np.zeros(H * K * D), np.ones(H * K * D))
    a_lb = -0.1 * np.ones((H, M, K))
    a_ub = np.ones((H, M, K))
    with pytest.raises(ValueError):
        av_envelope_star(a_lb, a_ub, v, H, M, K, D)

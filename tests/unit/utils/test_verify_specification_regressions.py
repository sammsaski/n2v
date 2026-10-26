"""Regressions for verify_specification's sound dispatch.

  * The Box disjointness fast check must not round a touching box into
    "disjoint": its per-row minimum is a round-to-nearest sum, so it only
    certifies a row once the margin clears a rigorous rounding bound.
  * ImageZono reach sets are checked in CHW order (spec columns index the
    torch/ONNX flattened output), like ImageStar.
  * An LP failure in one AND-group leaves only that group undecided; a later
    group that is certified disjoint still proves UNSAT.
"""

import logging
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

import n2v.utils.verify_specification as vs
from n2v.nn import NeuralNetwork
from n2v.sets import Box, ImageZono, Star
from n2v.sets.halfspace import HalfSpace
from n2v.utils.lpsolver import LPFailure
from n2v.utils.verify_specification import verify_specification


def _exact_box_min(row, lb, ub):
    """Exact rational min of row . x over the box [lb, ub]."""
    total = Fraction(0)
    for r, lo, hi in zip(row, lb, ub):
        r = Fraction(float(r))
        total += min(r * Fraction(float(lo)), r * Fraction(float(hi)))
    return total


# ---------------------------------------------------------------------------
# Box fast check: outward rounding slack
# ---------------------------------------------------------------------------

def test_box_fast_check_touching_corner_is_not_unsat():
    # The float minimum rounds to 2.22e-16 > g, the exact one is 1.665e-16 <= g,
    # so the corner x = lb lies in the unsafe region.
    lb = np.array([1.0, 1.5 * 2.0 ** -53, -1.0])
    ub = lb + 1.0
    row = np.array([1.0, 1.0, 1.0])
    g = 2e-16
    assert _exact_box_min(row, lb, ub) <= Fraction(g)
    hs = HalfSpace(row.reshape(1, -1), np.array([[g]]))
    assert verify_specification([Box(lb, ub)], hs).verdict == "UNKNOWN"


def test_box_fast_check_random_touching_boxes_never_unsat():
    # g is the smallest double >= the exact minimum, so the minimizing corner
    # is always unsafe; before the fix ~10% of these were declared UNSAT.
    rng = np.random.default_rng(0)
    false_unsat = 0
    for _ in range(2000):
        n = int(rng.integers(2, 12))
        lb = rng.uniform(-1, 1, n)
        ub = lb + rng.uniform(0, 1, n)
        row = rng.normal(size=n)
        exact = _exact_box_min(row, lb, ub)
        g = float(exact)
        if Fraction(g) < exact:
            g = float(np.nextafter(g, np.inf))
        hs = HalfSpace(row.reshape(1, -1), np.array([[g]]))
        if verify_specification([Box(lb, ub)], hs).verdict == "UNSAT":
            false_unsat += 1
    assert false_unsat == 0


def test_box_fast_check_still_certifies_real_margins():
    box = Box(np.zeros(3), np.ones(3))
    # Clear margin: sum(x) >= 5 is out of reach.
    far = HalfSpace(-np.ones((1, 3)), np.array([[-5.0]]))
    assert verify_specification([box], far).verdict == "UNSAT"
    # A 1e-12 margin at unit scale is far above the rounding slack (~1e-15).
    near = HalfSpace(-np.ones((1, 3)), np.array([[-3.0 - 1e-12]]))
    assert verify_specification([box], near).verdict == "UNSAT"
    # Half-unbounded box: the minimizing corner is finite, so the slack is too.
    half = Box(np.array([0.0, 1.0]), np.array([np.inf, 2.0]))
    assert verify_specification(
        [half], HalfSpace(np.ones((1, 2)), np.array([[0.5]]))).verdict == "UNSAT"
    # A zero coefficient on an unbounded coordinate contributes 0, not NaN.
    zero_col = Box(np.array([0.0, 0.0]), np.array([1.0, np.inf]))
    assert verify_specification(
        [zero_col], HalfSpace(np.array([[-1.0, 0.0]]),
                              np.array([[-2.0]]))).verdict == "UNSAT"


# ---------------------------------------------------------------------------
# ImageZono specs use CHW order
# ---------------------------------------------------------------------------

def test_imagezono_spec_uses_chw_order():
    # H=1, W=2, C=2 stored HWC: pixel p, channel c has centre 0.5 + 5 c and
    # radius 0.5, so CHW ranges are [0,1], [0,1], [5,6], [5,6] while the HWC
    # ones are [0,1], [5,6], [0,1], [5,6].
    c = np.array([0.5, 5.5, 0.5, 5.5]).reshape(-1, 1)
    Z = ImageZono(c, 0.5 * np.eye(4), height=1, width=2, num_channels=2)
    # CHW y2 >= 5.5 is reachable (HWC row 2 is in [0, 1]).
    reachable = HalfSpace(np.array([[0.0, 0.0, -1.0, 0.0]]), np.array([[-5.5]]))
    assert verify_specification([Z], reachable).verdict == "UNKNOWN"
    # CHW y1 >= 5.5 is unreachable (HWC row 1 is in [5, 6]).
    unreachable = HalfSpace(np.array([[0.0, -1.0, 0.0, 0.0]]),
                            np.array([[-5.5]]))
    assert verify_specification([Z], unreachable).verdict == "UNSAT"


def test_imagezono_conv_output_matches_torch_order():
    # Conv2d(1 -> 2, k=1): ch0 = x, ch1 = 10 - x on a 1x2 image, x in [0, 1].
    net = nn.Sequential(nn.Conv2d(1, 2, 1))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([1.0, -1.0]).reshape(2, 1, 1, 1))
        net[0].bias.copy_(torch.tensor([0.0, 10.0]))
    y = net(torch.zeros(1, 1, 1, 2)).detach().numpy().ravel()   # CHW
    assert np.allclose(y, [0.0, 0.0, 10.0, 10.0])
    S = ImageZono.from_bounds(np.zeros((1, 2, 1)), np.ones((1, 2, 1)), 1, 2, 1)
    R = NeuralNetwork(net).reach(S, method="approx")
    assert isinstance(R[0], ImageZono)
    G = np.array([[0.0, 0.0, -1.0, 0.0]])
    # y2 >= 9.5 is reached at x = 0; y2 >= 10.5 is not reachable.
    assert verify_specification(R, HalfSpace(G, np.array([[-9.5]]))).verdict \
        == "UNKNOWN"
    assert verify_specification(R, HalfSpace(G, np.array([[-10.5]]))).verdict \
        == "UNSAT"


# ---------------------------------------------------------------------------
# LP failures are per AND-group
# ---------------------------------------------------------------------------

_Y0_LE_HALF = HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.5]]))   # meets
_Y0_GE_5 = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-5.0]]))    # disjoint
_Y1_LE_HALF = HalfSpace(np.array([[0.0, 1.0]]), np.array([[0.5]]))   # meets


def test_lp_failure_in_one_group_keeps_other_groups(monkeypatch, caplog):
    real = vs._is_disjoint
    failing = {id(_Y0_LE_HALF)}

    def flaky(S, hs):
        if id(hs) in failing:
            raise LPFailure("simulated solver failure")
        return real(S, hs)

    monkeypatch.setattr(vs, "_is_disjoint", flaky)
    S = Star.from_bounds(np.zeros((2, 1)), np.ones((2, 1)))
    spec = [{'Hg': _Y0_LE_HALF}, {'Hg': _Y0_GE_5}]
    with caplog.at_level(logging.WARNING, logger=vs.__name__):
        assert verify_specification([S], spec).verdict == "UNSAT"
    assert "property group 1 of 2" in caplog.text
    assert verify_specification([S], spec[::-1]).verdict == "UNSAT"
    # With no disjoint group left, the failed group must not count as one.
    spec_open = [{'Hg': _Y0_LE_HALF}, {'Hg': _Y1_LE_HALF}]
    assert verify_specification([S], spec_open).verdict == "UNKNOWN"
    failing.add(id(_Y0_GE_5))
    assert verify_specification([S], spec).verdict == "UNKNOWN"


def test_box_lp_failure_in_one_group_keeps_other_groups(monkeypatch):
    # The two-row group reaches the box feasibility LP, stubbed to fail with
    # scipy status 4; the single-row group is disjoint by the fast check.
    class _Res:
        status = 4
        success = False
        message = "numerical difficulties encountered"

    monkeypatch.setattr(vs, "linprog", lambda *a, **k: _Res())
    box = Box(np.zeros(2), np.ones(2))
    two_rows = HalfSpace(np.eye(2), np.array([[0.5], [0.5]]))
    with pytest.raises(LPFailure):
        vs._verify_specification_box(box, two_rows)
    spec = [{'Hg': two_rows}, {'Hg': _Y0_GE_5}]
    assert verify_specification([box], spec).verdict == "UNSAT"
    assert verify_specification([box], [{'Hg': two_rows}]).verdict == "UNKNOWN"


def test_box_reach_through_linear_never_gives_a_false_unsat():
    """End to end: reach(Box) through a Linear layer, then a spec whose
    threshold is the smallest double at or above the exact minimum of the
    true image. Box.affine_map used a round-to-nearest center/radius and
    could shrink the Box by a few ulps (835 false UNSATs in 3000 trials)."""
    rng = np.random.default_rng(3)
    for t in range(300):
        n = int(rng.integers(1, 4))
        lb = rng.uniform(-1, 1, n) * 10.0 ** rng.integers(-2, 3)
        ub = lb + rng.uniform(0, 1, n) * 10.0 ** rng.integers(-3, 1)
        W = rng.normal(size=(1, n)) * 10.0 ** rng.integers(0, 6)
        b = -(W @ ((lb + ub) / 2)) + rng.normal(size=1) * 1e-3
        lin = nn.Linear(n, 1).double()
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(W))
            lin.bias.copy_(torch.tensor(b))
        a = 1.0 if t % 2 == 0 else -1.0
        exact = Fraction(float(b[0])) * Fraction(a) + sum(
            min(Fraction(float(W[0, j])) * Fraction(a) * Fraction(float(lb[j])),
                Fraction(float(W[0, j])) * Fraction(a) * Fraction(float(ub[j])))
            for j in range(n))
        g = float(exact)
        if Fraction(g) < exact:
            g = float(np.nextafter(g, np.inf))
        R = NeuralNetwork(nn.Sequential(lin)).reach(Box(lb, ub), method='approx')
        spec = HalfSpace(np.array([[a]]), np.array([[g]]))
        assert verify_specification(R, spec).verdict != 'UNSAT', t


def test_star_interval_margin_is_decided_without_an_lp(monkeypatch):
    """A star whose rigorous interval bound clears the spec threshold is
    disjoint without the LP emptiness check (which a solver can leave at
    'optimal' on a thin slackened system); a touching threshold is not."""
    S = Star.from_bounds(np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]]))
    S = S.affine_map(np.array([[1.0, 1.0]]), np.array([[-2.0]]))  # y in [-2, 0]
    spec_clear = HalfSpace(np.array([[-1.0]]), np.array([[-1e-6]]))  # y >= 1e-6
    spec_touch = HalfSpace(np.array([[-1.0]]), np.array([[0.0]]))    # y >= 0

    def no_lp(*args, **kwargs):
        raise AssertionError("LP path reached")
    monkeypatch.setattr(Star, 'intersect_half_space', no_lp)
    assert vs._verify_specification_star(S, spec_clear) is True
    monkeypatch.undo()
    assert vs._verify_specification_star(S, spec_touch) is False

"""Tests for APGD (Auto-PGD) falsifier."""
from __future__ import annotations
import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets.halfspace import HalfSpace
from n2v.utils.falsify import falsify


class _LinearNet(nn.Module):
    """y = x @ W.T + b with deterministic init."""
    def __init__(self, dim_in=2, dim_out=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, x):
        return self.lin(x)


def test_apgd_finds_known_sat():
    """APGD should find a real counterexample on a simple 2D linear net."""
    net = _LinearNet(dim_in=2, dim_out=2, seed=0)
    lb = np.array([-1.0, -1.0], dtype=np.float32)
    ub = np.array([ 1.0,  1.0], dtype=np.float32)
    # Unsafe region that every x in the box maps into: y_0 <= 1e6.
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1e6]]))
    result, cex = falsify(net, lb, ub, hs, method='apgd', seed=0)
    assert result == 0
    assert cex is not None
    x_cex, y_cex = cex
    assert x_cex.shape == (2,)
    assert y_cex.shape == (2,)


def test_apgd_returns_unknown_if_no_sat():
    """APGD on an unreachable unsafe region returns 2 (unknown)."""
    net = _LinearNet(dim_in=2, dim_out=2, seed=0)
    lb = np.array([-1.0, -1.0], dtype=np.float32)
    ub = np.array([ 1.0,  1.0], dtype=np.float32)
    # Unsafe region unreachable by any input in [-1, 1]^2.
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-1e6]]))
    result, cex = falsify(net, lb, ub, hs, method='apgd',
                          seed=0, n_restarts=3, n_steps=20)
    assert result == 2
    assert cex is None


def test_apgd_respects_input_bounds():
    """Counterexample must be inside [lb, ub] element-wise, exactly: the
    witness is graded at zero input tolerance (repaired 2026-11-18: the
    check allowed 1e-5 outside the box)."""
    net = _LinearNet(dim_in=2, dim_out=2, seed=0)
    lb = np.array([-0.5, -0.5], dtype=np.float32)
    ub = np.array([ 0.5,  0.5], dtype=np.float32)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1e6]]))
    result, cex = falsify(net, lb, ub, hs, method='apgd', seed=0)
    assert result == 0
    x_cex, _ = cex
    assert (x_cex >= lb).all()
    assert (x_cex <= ub).all()


def test_random_pgd_apgd_ensemble_short_circuits():
    """The ensemble method returns as soon as any stage finds SAT."""
    net = _LinearNet(dim_in=2, dim_out=2, seed=0)
    lb = np.array([-1.0, -1.0], dtype=np.float32)
    ub = np.array([ 1.0,  1.0], dtype=np.float32)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1e6]]))
    result, cex = falsify(net, lb, ub, hs, method='random+pgd+apgd', seed=0)
    assert result == 0
    assert cex is not None


def test_random_apgd_in_methods():
    from n2v.utils.falsify import METHODS
    assert 'random+apgd' in METHODS


def test_random_apgd_falls_through_to_apgd():
    """'random+apgd' (random -> APGD, skipping the slow fixed-step PGD leg):
    with the random leg disabled (n_samples=0), the cascade must reach APGD
    and still find the counterexample."""
    net = _LinearNet(dim_in=2, dim_out=2, seed=0)
    lb = np.array([-1.0, -1.0], dtype=np.float32)
    ub = np.array([ 1.0,  1.0], dtype=np.float32)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1e6]]))
    result, cex = falsify(net, lb, ub, hs, method='random+apgd',
                          seed=0, n_samples=0, n_restarts=2, n_steps=20)
    assert result == 0
    assert cex is not None


def test_autoattack_scaffold_raises_when_not_installed():
    """If the 'autoattack' package is not installed, method='autoattack'
    raises ImportError with install instructions."""
    from n2v.utils.falsify import _HAS_AUTOATTACK
    from n2v.utils.falsify import falsify
    net = _LinearNet(dim_in=2, dim_out=2, seed=0)
    lb = np.array([-1.0, -1.0], dtype=np.float32)
    ub = np.array([ 1.0,  1.0], dtype=np.float32)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1e6]]))
    if _HAS_AUTOATTACK:
        # Package is installed: scaffold raises NotImplementedError
        with pytest.raises(NotImplementedError):
            falsify(net, lb, ub, hs, method='autoattack')
    else:
        # Package not installed: import-time guard raises ImportError
        with pytest.raises(ImportError):
            falsify(net, lb, ub, hs, method='autoattack')


# --------------------------------------------------------------------------- #
# The strength of the search (added 2026-11-18). Each model is linear or ReLU  #
# with dyadic weights and boxes, so the unsafe set is known by hand and every  #
# witness is replayed in rationals against the property as written here. Each #
# case states why the unmodified APGD reaches the counterexample from any     #
# start; the seed is fixed (and the autouse fixture seeds the global          #
# generators), and everything runs on the CPU, so no case can flake.          #
# --------------------------------------------------------------------------- #
from fractions import Fraction  # noqa: E402


def _dyadic_linear(W, b):
    """A float32 ``nn.Linear`` holding exactly the dyadic ``W`` and ``b``."""
    W = np.asarray(W, dtype=np.float64)
    layer = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(W))
        layer.bias.copy_(torch.tensor(np.asarray(b, dtype=np.float64)))
    return layer


def _exact_affine(W, b, x):
    """``W x + b`` in rationals, for a point ``x`` read exactly."""
    xs = [Fraction(float(v)) for v in np.asarray(x, dtype=np.float64).ravel()]
    return [sum(Fraction(float(w)) * v for w, v in zip(row, xs))
            + Fraction(float(bb)) for row, bb in zip(W, b)]


def _in_box(x, lb, ub):
    x = np.asarray(x, dtype=np.float64).ravel()
    return x.shape == np.shape(lb) and bool(np.all(lb <= x) and np.all(x <= ub))


# The mixed box: intervals of different widths, both signs, magnitudes above
# and below one, one degenerate coordinate (x3). y0 = W0 . x + 1/2 reaches its
# maximum 11.125 at exactly one point, the corner that takes ub where the
# weight is positive and lb where it is negative.
_W_MIX = [[1.0, -2.0, 0.5, 4.0, -1.0, 0.25],
          [0.5, 0.25, -1.0, 0.0, 2.0, -0.5]]
_B_MIX = [0.5, -1.0]
_LB_MIX = np.array([-2.0, -0.5, 1.0, 0.75, -3.0, 0.0])
_UB_MIX = np.array([2.0, 0.25, 3.0, 0.75, -1.0, 0.5])
_CORNER_MIX = np.array([2.0, -0.5, 3.0, 0.75, -3.0, 0.5])
_UNSAFE_CORNER = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-11.125]]))


def _assert_is_the_top_corner(result, cex):
    assert result == 0, "the unsafe corner of the box was not found"
    x = np.asarray(cex[0], dtype=np.float64).ravel()
    assert _in_box(x, _LB_MIX, _UB_MIX), f"witness {x} outside the box"
    assert _exact_affine(_W_MIX, _B_MIX, x)[0] >= Fraction(11.125), (
        f"witness {x} is not unsafe")
    np.testing.assert_array_equal(x, _CORNER_MIX)


def test_apgd_walks_to_the_one_unsafe_corner_of_a_mixed_box():
    """The loss -y0 + 11.125 is linear, so every step moves each coordinate
    one auto step (5% of the widest range, 0.2) toward the bound its weight
    points to and the projection holds it there; the loss falls at every
    step, so the step is never halved, and from any start the corner is
    reached within 20 steps and is a hit at margin exactly 0 (the unsafe
    region is closed). n_restarts=1, n_steps=40, seed 3. Catches an ascent
    step, a hit test that needs a negative margin (APGD has no check after
    its last step), an improvement test read backwards (every window sends
    the iterate back to its start), a step that is not projected (the
    degenerate x3 moves, the hit lies outside the box and the re-check in the
    box fails), and an auto step taken from the smallest range (0 here)."""
    result, cex = falsify(_dyadic_linear(_W_MIX, _B_MIX), _LB_MIX, _UB_MIX,
                          _UNSAFE_CORNER, method='apgd', seed=3,
                          n_restarts=1, n_steps=40)
    _assert_is_the_top_corner(result, cex)


def test_apgd_takes_the_step_size_it_is_given():
    """One step of the given size 4 (the widest range) moves every coordinate
    onto the bound its weight points to; the second of n_steps=2 evaluations
    sees the unsafe corner. The auto step (0.2) cannot get there in one step.
    Catches an explicit ``step_size`` that is replaced by the auto step."""
    result, cex = falsify(_dyadic_linear(_W_MIX, _B_MIX), _LB_MIX, _UB_MIX,
                          _UNSAFE_CORNER, method='apgd', seed=3,
                          n_restarts=1, n_steps=2, step_size=4.0)
    _assert_is_the_top_corner(result, cex)


def test_apgd_halves_its_step_to_land_in_a_narrow_band():
    """y = x on [-1, 3]; unsafe is the band 1 <= y <= 9/8 (two rows), whose
    width 1/8 is below the initial step 0.2 and above its half 0.1. Descent
    walks to the band; if it steps over it, it oscillates around the band
    without improving, the plateau window (6 steps) halves the step to 0.1 and
    sends the iterate back to the best point, within 0.0375 of the band, and
    one step of 0.1 lands in it. So the unmodified schedule finds the band
    from any start within 20 of n_steps=30 steps. With seed 3 the first
    approach steps over the band, so a schedule that doubles the step on a
    plateau (it then never lands in the band) or an ascent step misses it."""
    hs = HalfSpace(np.array([[1.0], [-1.0]]), np.array([[1.125], [-1.0]]))
    result, cex = falsify(_dyadic_linear([[1.0]], [0.0]), np.array([-1.0]),
                          np.array([3.0]), hs, method='apgd', seed=3,
                          n_restarts=1, n_steps=30)
    assert result == 0, "the band was not found"
    x = float(np.asarray(cex[0], dtype=np.float64).ravel()[0])
    assert 1.0 <= x <= 1.125, f"witness {x} is not in the band"


def test_apgd_reads_every_group_row_and_disjunct():
    """y = x on [-2, 2] x [-3, 1] x [0, 4]. Two groups (AND): group A is the
    OR of A1: y0 >= 8 (no point of the box; its margin, at least 6, is always
    above A2's) and A2: y0 >= 7/4 AND y1 <= -11/4 (two rows); group B is
    y2 >= 15/4. The loss is the max over groups of the min over half-spaces of
    the max over rows, so descent lowers the worst term until all of A2's rows
    and B hold (0.2 per step, at most 3 * 19 steps of n_steps=60). Each wrong
    reading leaves a coordinate without gradient: the min over rows stops once
    one row of A2 holds, the min over groups once one group holds, the max over
    disjuncts follows the infeasible A1. The region is 1/4096 of the box."""
    a1 = HalfSpace(np.array([[-1.0, 0.0, 0.0]]), np.array([[-8.0]]))
    a2 = HalfSpace(np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                   np.array([[-1.75], [-2.75]]))
    b = HalfSpace(np.array([[0.0, 0.0, -1.0]]), np.array([[-3.75]]))
    lb, ub = np.array([-2.0, -3.0, 0.0]), np.array([2.0, 1.0, 4.0])
    result, cex = falsify(_dyadic_linear(np.eye(3), np.zeros(3)), lb, ub,
                          [{'Hg': [a1, a2]}, {'Hg': b}], method='apgd',
                          seed=3, n_restarts=1, n_steps=60)
    assert result == 0, "the corner region of the property was not found"
    x = np.asarray(cex[0], dtype=np.float64).ravel()
    assert _in_box(x, lb, ub)
    assert x[0] >= 1.75 and x[1] <= -2.75 and x[2] >= 3.75, (
        f"witness {x} is not in the unsafe region")


def test_apgd_restarts_are_independent_and_start_with_the_initial_step():
    """f(x) = 1 - 4x + 8 relu(x - 1/4) - 8 relu(x - 1/2) on [0, 1] falls to a
    local minimum f(1/4) = 0, rises to f(1/2) = 1 and falls to f(1) = -1;
    unsafe f <= -3/4 holds for x >= 15/16 only. Descent from x < 1/2 is
    trapped, descent from x > 1/2 needs at most 9 steps of the initial 0.05.
    Seed 3 draws the restart points 0.09, 0.24, 0.80, ...: the first two are
    trapped (and halve their steps on the plateau), the third escapes. Catches
    restarts that all begin at the same point and a step size that carries
    the halvings of earlier restarts into the next one (from 0.80 a step of
    0.05 / 16 cannot reach 15/16 in n_steps=40)."""
    net = nn.Sequential(nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 1))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0], [1.0], [1.0]]))
        net[0].bias.copy_(torch.tensor([0.0, -0.25, -0.5]))
        net[2].weight.copy_(torch.tensor([[-4.0, 8.0, -8.0]]))
        net[2].bias.copy_(torch.tensor([1.0]))
    hs = HalfSpace(np.array([[1.0]]), np.array([[-0.75]]))
    result, cex = falsify(net, np.array([0.0]), np.array([1.0]), hs,
                          method='apgd', seed=3, n_restarts=8, n_steps=40)
    assert result == 0, "no restart escaped the local minimum"
    x = Fraction(float(np.asarray(cex[0], dtype=np.float64).ravel()[0]))
    relu = lambda v: max(v, Fraction(0))                       # noqa: E731
    f = 1 - 4 * x + 8 * relu(x - Fraction(1, 4)) - 8 * relu(x - Fraction(1, 2))
    assert 0 <= x <= 1 and f <= Fraction(-3, 4), f"witness {x}: f = {f}"

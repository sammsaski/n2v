"""Regressions for verify_specification's sound dispatch.

  * The Box disjointness fast check must not round a touching box into
    "disjoint": its per-row minimum is a round-to-nearest sum, so it only
    certifies a row once the margin clears a rigorous rounding bound.
  * ImageZono reach sets are checked in CHW order (spec columns index the
    torch/ONNX flattened output), like ImageStar.
  * An LP failure in one AND-group leaves only that group undecided; a later
    group that is certified disjoint still proves UNSAT.
  * ImageStar reach sets are checked in CHW order too, directly and through
    a model; a Box LP's "infeasible" is a proof only once confirmed.
  * ``spec_summary`` (and the ``verbose`` route) and
    ``distribute_and_of_or_of_and``, against the meaning of each spec shape.
"""

import logging
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

import n2v.utils.verify_specification as vs
from n2v.nn import NeuralNetwork
from n2v.sets import Box, ImageStar, ImageZono, Star
from n2v.sets.halfspace import HalfSpace
from n2v.utils.lpsolver import LPFailure
from n2v.utils.verify_specification import verify_specification
from tests.known_bugs import known_bug


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


# ---------------------------------------------------------------------------
# ImageStar specs use CHW order (the counterpart of the ImageZono cases)
# ---------------------------------------------------------------------------

def _chw_to_stored(channels, height, width, k):
    """The stored (height, width, channel) index of CHW entry k."""
    c, rest = divmod(k, height * width)
    h, w = divmod(rest, width)
    return (h * width + w) * channels + c


def test_imagestar_spec_uses_chw_order():
    # (C, H, W) = (2, 2, 3) stored HWC: channel c of every pixel is in
    # [5c, 5c + 1]. CHW entry 6 is channel 1 at pixel (0, 0), in [5, 6], while
    # stored entry 6 is channel 0 at pixel (1, 0), in [0, 1]; CHW entry 1 is
    # channel 0 at pixel (0, 1), in [0, 1], while stored entry 1 is channel 1
    # at pixel (0, 0), in [5, 6]. Catches an ImageStar checked in its stored
    # order (a false UNSAT on the first spec, a missed one on the second).
    if (_chw_to_stored(2, 2, 3, 6), _chw_to_stored(2, 2, 3, 1)) != (1, 2):
        pytest.fail("the CHW to HWC index map of the case is wrong")
    lo = np.zeros((2, 3, 2))
    lo[:, :, 1] = 5.0
    S = ImageStar.from_bounds(lo, lo + 1.0, 2, 3, 2)
    G = np.zeros((1, 12))
    G[0, 6] = -1.0                       # CHW y6 >= 5.5: reachable
    assert verify_specification([S], HalfSpace(G, np.array([[-5.5]]))).verdict \
        == "UNKNOWN"
    G = np.zeros((1, 12))
    G[0, 1] = -1.0                       # CHW y1 >= 5.5: unreachable
    assert verify_specification([S], HalfSpace(G, np.array([[-5.5]]))).verdict \
        == "UNSAT"


def test_imagestar_conv_output_matches_torch_order():
    # Conv2d(1 -> 2, k=1) on a 2x3 image, x in [0, 1]: channel 0 = x,
    # channel 1 = 10 - x. CHW entry 6 is channel 1 at pixel (0, 0); the stored
    # entry 6 is channel 0 at pixel (1, 0). Through the model entry point.
    net = nn.Sequential(nn.Conv2d(1, 2, 1))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([1.0, -1.0]).reshape(2, 1, 1, 1))
        net[0].bias.copy_(torch.tensor([0.0, 10.0]))
    y = net(torch.zeros(1, 1, 2, 3)).detach().numpy().ravel()   # CHW
    if not np.array_equal(y, [0.0] * 6 + [10.0] * 6):
        pytest.fail(f"the network computes {y}")
    S = ImageStar.from_bounds(np.zeros((2, 3, 1)), np.ones((2, 3, 1)), 2, 3, 1)
    R = NeuralNetwork(net).reach(S, method="approx")
    assert all(isinstance(r, ImageStar) for r in R)
    G = np.zeros((1, 12))
    G[0, 6] = -1.0
    # y6 >= 9.5 is reached at x = 0; y6 >= 10.5 is not reachable.
    assert verify_specification(R, HalfSpace(G, np.array([[-9.5]]))).verdict \
        == "UNKNOWN"
    assert verify_specification(R, HalfSpace(G, np.array([[-10.5]]))).verdict \
        == "UNSAT"


# ---------------------------------------------------------------------------
# A Box LP that says "infeasible" is a proof only once confirmed
# ---------------------------------------------------------------------------

def test_box_lp_infeasible_status_is_confirmed_before_unsat(monkeypatch):
    """The two-row region y0 <= 1/2, y1 <= 1/2 meets the box [0, 1]^2 at
    (0, 0). The box feasibility LP is stubbed to report scipy status 2
    (infeasible), as scipy's presolve does on some feasible tiny-scale
    systems; the confirmation re-solves and finds the point, so the verdict
    stays UNKNOWN. Catches an unconfirmed 'infeasible' read as disjoint (a
    false UNSAT)."""
    class _Infeasible:
        status = 2
        success = False
        message = "The problem is infeasible. (stubbed)"

    monkeypatch.setattr(vs, "linprog", lambda *a, **k: _Infeasible())
    box = Box(np.zeros(2), np.ones(2))
    two_rows = HalfSpace(np.eye(2), np.array([[0.5], [0.5]]))
    assert vs._verify_specification_box(box, two_rows) is False
    assert verify_specification([box], two_rows).verdict == "UNKNOWN"


# ---------------------------------------------------------------------------
# spec_summary and the verbose route
# ---------------------------------------------------------------------------

def _half_space(rows, cols):
    return HalfSpace(np.ones((rows, cols)), np.ones((rows, 1)))


@pytest.mark.parametrize("spec,parts,absent", [
    pytest.param(lambda: _half_space(2, 5), ("HalfSpace", "dim=5",
                                              "2 constraints", "AND"),
                 ("OR",), id="halfspace-2-rows-over-5"),
    pytest.param(lambda: _half_space(1, 3), ("dim=3", "1 constraint"),
                 ("AND", "constraints"), id="halfspace-1-row-over-3"),
    pytest.param(lambda: {"Hg": _half_space(2, 5)},
                 ("dim=5", "2 constraints"), ("OR",), id="dict-halfspace"),
    pytest.param(lambda: {"Hg": [_half_space(1, 2)] * 3},
                 ("OR of 3 HalfSpaces",), (), id="dict-list-of-3"),
    pytest.param(lambda: [{"Hg": _half_space(1, 2)}] * 2,
                 ("AND of 2",), ("OR",), id="list-of-2-dicts"),
    pytest.param(lambda: [_half_space(1, 2)] * 3, ("OR of 3",), ("AND",),
                 id="list-of-3-halfspaces"),
    pytest.param(lambda: [], ("empty",), (), id="empty-list"),
])
def test_spec_summary_states_the_structure_and_the_counts(spec, parts,
                                                          absent):
    """The summary names the dimension (columns of G), the number of rows of a
    half-space, the number of disjuncts or groups, and which of AND and OR
    joins them. Catches the row count read from the columns, the dimension
    read from the rows, a list of groups described as an OR, and a count of
    the dict's keys instead of its disjuncts."""
    text = vs.spec_summary(spec())
    for part in parts:
        assert part in text, f"{part!r} missing from {text!r}"
    for part in absent:
        assert part not in text, f"{part!r} in {text!r}"


@pytest.mark.parametrize("spec", [
    pytest.param("y0 <= 1", id="a-string"),
    pytest.param({"G": np.ones((1, 2))}, id="a-dict-without-Hg"),
    pytest.param({"Hg": 3}, id="a-dict-whose-Hg-is-a-number"),
])
def test_spec_summary_refuses_what_it_cannot_describe(spec):
    with pytest.raises(TypeError):
        vs.spec_summary(spec)


def test_verbose_prints_the_summary_and_the_verdict_and_changes_nothing(
        capsys):
    """``verbose=True`` prints the spec summary and the verdict; the result
    is the one the quiet call returns. Catches a verbose route that prints
    another structure than the property has or changes the verdict."""
    S = Star.from_bounds(np.zeros((2, 1)), np.ones((2, 1)))
    spec = [{'Hg': _Y0_LE_HALF}, {'Hg': _Y0_GE_5}]
    quiet = verify_specification([S], spec)
    capsys.readouterr()
    loud = verify_specification([S], spec, verbose=True)
    out = capsys.readouterr().out
    assert loud == quiet and loud.verdict == "UNSAT"
    assert "AND of 2" in out and "verdict: UNSAT" in out, out


# ---------------------------------------------------------------------------
# distribute_and_of_or_of_and: the same unsafe region in one OR
# ---------------------------------------------------------------------------

def _in_region(hs, y):
    """Every row of ``hs`` holds at ``y``, in fractions."""
    return all(sum(Fraction(float(a)) * Fraction(float(v))
                   for a, v in zip(row, y)) <= Fraction(float(b))
               for row, b in zip(np.atleast_2d(hs.G), np.ravel(hs.g)))


def _unsafe_by_the_docstring(groups, y):
    """AND over groups, OR within a group, AND over the rows of a HalfSpace."""
    return all(any(_in_region(hs, y) for hs in group) for group in groups)


_GRID = [Fraction(k, 2) for k in range(-5, 6)]          # -5/2 .. 5/2, step 1/2


@pytest.mark.parametrize("groups", [
    pytest.param(lambda: [[HalfSpace([[-1.0, 0.0]], [[-1.0]]),     # y0 >= 1
                           HalfSpace([[1.0, 0.0]], [[-1.0]])],     # y0 <= -1
                          [HalfSpace([[0.0, -1.0]], [[-1.0]]),     # y1 >= 1
                           HalfSpace([[0.0, 1.0]], [[-1.0]])]],    # y1 <= -1
                 id="two-groups-of-two"),
    pytest.param(lambda: [[HalfSpace([[1.0, 1.0], [-1.0, 1.0]],    # two rows
                                     [[1.0], [0.5]])],
                          [HalfSpace([[0.0, -1.0]], [[-0.5]]),
                           HalfSpace([[1.0, 0.0]], [[0.0]]),
                           HalfSpace([[-1.0, -1.0]], [[2.0]])]],
                 id="one-then-three-with-two-rows"),
    pytest.param(lambda: [[HalfSpace([[-1.0, 0.0]], [[-1.0]]),
                           HalfSpace([[0.0, 1.0]], [[0.0]])]],
                 id="one-group-of-two"),
    pytest.param(lambda: [[HalfSpace([[-1.0, 0.0]], [[-1.0]])], []],
                 id="a-group-with-no-disjunct"),
])
def test_distribute_keeps_the_unsafe_region_point_by_point(groups):
    """The distributed form is one group (outer length 1) whose disjuncts are
    the stacked choices, and it classifies every point of a half-integer grid
    over [-5/2, 5/2]^2, boundaries included, as the AND-of-OR input does
    (docstring meaning, evaluated in fractions). Catches the choices paired
    by position instead of all combinations, a one-group input split into an
    AND of its disjuncts, rows or thresholds dropped when stacking, and an
    empty group not read as an empty region."""
    from n2v.utils.verify_specification import distribute_and_of_or_of_and
    groups = groups()
    out = distribute_and_of_or_of_and(groups)
    assert len(out) == 1, f"{len(out)} groups returned"
    wrong = [(float(a), float(b)) for a in _GRID for b in _GRID
             if _unsafe_by_the_docstring(out, (a, b))
             != _unsafe_by_the_docstring(groups, (a, b))]
    assert not wrong, f"classified otherwise than the input at {wrong[:5]}"


@known_bug(6, "distribute_and_of_or_of_and returns [[]], the empty unsafe "
              "region, for an input with no group, whose meaning (an AND of "
              "nothing) is the whole output space; it neither refuses the "
              "input nor returns that region (no library code calls it today)")
def test_distribute_of_no_group_is_refused_or_the_whole_space():
    from n2v.utils.verify_specification import distribute_and_of_or_of_and
    try:
        out = distribute_and_of_or_of_and([])
    except ValueError:
        return
    unsafe = [(float(a), float(b)) for a in _GRID for b in _GRID
              if _unsafe_by_the_docstring(out, (a, b))]
    assert len(unsafe) == len(_GRID) ** 2, (
        f"{out} holds {len(unsafe)} of {len(_GRID) ** 2} points")


# ---------------------------------------------------------------------------
# The dispatch: a probabilistic set goes to its own lane
# ---------------------------------------------------------------------------

def test_a_probabilistic_set_goes_to_its_lane_with_the_default_config(
        monkeypatch):
    """A ``ProbabilisticSet`` is handed to the probabilistic lane with
    ``ProbVerifyConfig()`` when no config is given, and with the caller's
    config otherwise; the lane's result is returned as it is. The lane itself
    is stubbed: this checks the routing line only.
    Catches the default config dropped (the lane then reads ``None``) and a
    probabilistic set sent down the sound path."""
    from n2v.probabilistic.flow.sets import ProbabilisticSet
    seen = []
    sentinel = vs.VerificationResult(verdict="UNKNOWN", epsilon_total=0.5)

    def lane(prob_set, spec, config, verbose=False):
        seen.append((prob_set, spec, config))
        return sentinel

    monkeypatch.setattr(vs, "_verify_specification_probabilistic", lane)
    prob = ProbabilisticSet.__new__(ProbabilisticSet)
    assert verify_specification(prob, _Y0_GE_5) is sentinel
    assert seen[-1][0] is prob and seen[-1][1] is _Y0_GE_5
    assert seen[-1][2] == vs.ProbVerifyConfig()
    mine = vs.ProbVerifyConfig(method="amls", n_samples=7)
    verify_specification(prob, _Y0_GE_5, config=mine)
    assert seen[-1][2] is mine

"""Soundness of the refinement verifier's witness module (``n2v/refine/witness.py``).

The module holds the two decisions of ``verify_refine`` that turn a node of the
search into a verdict, and this file checks both sides of that promise:

- Pruning: ``certified_prune_wave`` drops a star only when it is certified
  disjoint from the unsafe region ``{y : G y <= g}``, and ``violation_lp``
  answers ``None`` (read by the search as "vacuously safe") only when the
  predicate polytope is certified empty. A wrong prune or a wrong ``None`` is a
  false UNSAT. A star that is disjoint by a clear margin must be pruned (unit
  harm: the search would not end otherwise).
- Witnesses: ``is_true_counterexample`` answers True only for a concrete input
  that is a member of the input star (its box and its own rows) and whose
  float64 output lies in the closed unsafe region at zero tolerance;
  ``clip_to_raw_box`` snaps a box corner into the user's float64 box before the
  output is computed. A wrong True is a SAT with an invalid witness.

The oracle is exact: stars and networks with dyadic data, whose minimum
violation over the star (the optimum of the epigraph linear program), whose
meeting with the unsafe region and whose image under the network are computed
in ``fractions.Fraction`` by ``tests/_oracles/rational.py``. Instances cover a
star disjoint by a clear margin, one meeting the region in a full-dimensional
part, one touching it in exactly one point, one disjoint and one meeting by
2**-32 (below every solver tolerance), an empty star, a star with an
equality-like pair of rows, a point star, a star without a predicate box, a
touching star that float64 rounding of ``G c`` turns into a disjoint one, and
specifications with several rows and several disjuncts. Every case runs on the
CPU (the certified GPU engine runs on CPU torch).
"""

from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.refine import verify_refine
from n2v.refine.types import LinearSpec, Status
from n2v.refine.witness import (
    certified_prune_wave,
    clip_to_raw_box,
    is_true_counterexample,
    violation_lp,
)
from n2v.sets.star import Star
from tests._oracles import rational as R
from tests.known_bugs import known_bug


# --------------------------------------------------------------------------- #
# Exact oracles (no n2v arithmetic)                                             #
# --------------------------------------------------------------------------- #
def _star(c, Vp, C, d, lb, ub):
    """A Star ``c + Vp alpha`` with rows ``C alpha <= d`` and box [lb, ub]."""
    n = len(lb)
    V = np.hstack([np.asarray(c, float).reshape(-1, 1),
                   np.asarray(Vp, float).reshape(len(c), n)])
    C = np.asarray(C, float).reshape(-1, n)
    d = np.asarray(d, float).reshape(-1, 1)
    return Star(V, C, d, np.asarray(lb, float).reshape(-1, 1),
                np.asarray(ub, float).reshape(-1, 1))


def _polytope_rows(S, include_Cd=True):
    """``(c, Vp, A, b)`` in Fractions: ``x = c + Vp a`` with ``A a <= b`` the
    star's own rows (when ``include_Cd``) and its finite box rows."""
    c, Vp, C, d, lb, ub = R.star(S)
    n = len(lb)
    A = [list(r) for r in C] if include_Cd else []
    b = list(d) if include_Cd else []
    for j in range(n):
        e = [Fraction(int(k == j)) for k in range(n)]
        if ub[j] is not None:
            A.append(e)
            b.append(ub[j])
        if lb[j] is not None:
            A.append([-v for v in e])
            b.append(-lb[j])
    return c, Vp, A, b


def _margin_rows(S, G, g):
    """Each spec row as ``(coef, const)``: its margin is ``coef . a + const``."""
    c, Vp, _, _ = _polytope_rows(S)
    n = len(Vp[0]) if Vp and Vp[0] else 0
    out = []
    for grow, gr in zip(np.atleast_2d(G), np.ravel(g)):
        gw = [Fraction(float(v)) for v in grow]
        coef = [sum((gi * Vp[i][j] for i, gi in enumerate(gw)), Fraction(0))
                for j in range(n)]
        const = R.dot(gw, c) - Fraction(float(gr))
        out.append((coef, const))
    return out


def _meets(S, G, g):
    """Exactly: does some point of ``S`` satisfy every row ``G y <= g``?"""
    _, _, A, b = _polytope_rows(S)
    A, b = list(A), list(b)
    for coef, const in _margin_rows(S, G, g):
        A.append(coef)
        b.append(-const)
    return R.feasible(A_ub=A, b_ub=b) is not None


def _min_row(S, grow):
    """Exactly: the minimum of ``grow . y`` over ``S`` (vertex enumeration)."""
    c, Vp, A, b = _polytope_rows(S)
    gw = [Fraction(float(v)) for v in grow]
    vals = [R.dot(gw, [ci + R.dot(Vi, v) for ci, Vi in zip(c, Vp)])
            for v in R.vertices(A, b)]
    return min(vals)


def _t_star(S, G, g, include_Cd=True):
    """Exactly: ``min over the polytope (or its box) of max_r margin_r``, the
    optimum of the epigraph program, by vertex enumeration in ``(a, t)``;
    None when the polytope is empty."""
    _, _, A, b = _polytope_rows(S, include_Cd)
    n = len(A[0])
    EA = [list(r) + [Fraction(0)] for r in A]
    Eb = list(b)
    for coef, const in _margin_rows(S, G, g):
        EA.append(list(coef) + [Fraction(-1)])
        Eb.append(-const)
    EA.append([Fraction(0)] * n + [Fraction(1)])
    Eb.append(Fraction(10 ** 6))
    vs = R.vertices(EA, Eb)
    return min(v[-1] for v in vs) if vs else None


def _exact_max_margin(S, G, g, alpha):
    """Exactly: the worst row margin at the predicate point ``alpha``."""
    a = [Fraction(float(v)) for v in alpha]
    return max(R.dot(coef, a) + const for coef, const in _margin_rows(S, G, g))


def _shifted(S, grow, target):
    """``S`` moved along the first output coordinate so that the exact
    minimum of ``grow . y - 0`` over it is ``target`` (``grow[0] == 1``)."""
    delta = Fraction(target) - _min_row(S, grow)
    V = S.V.copy()
    V[0, 0] = float(Fraction(float(V[0, 0])) + delta)
    T = Star(V, S.C.copy(), S.d.copy(), S.predicate_lb.copy(),
             S.predicate_ub.copy())
    if _min_row(T, grow) != Fraction(target):
        pytest.fail("instance builder: the shift is not exact")
    return T


# Two predicate stars over a 2-D output, with asymmetric boxes, bounds beyond
# [-1, 1], magnitudes above and below one and binding rows that are not box rows.
def _base2():
    return _star([0.25, -0.5], [[1, 0.5], [-0.25, 1]], [[1, 1], [-1, 0.5]],
                 [2, 3], [-3, 0.25], [0.5, 4])


def _base3():
    return _star([0.5, 0.25], [[1, -0.5, 0.25], [0.5, 1, -1]], [[1, 1, 1]],
                 [1], [-1, -2, 0], [1, 0.5, 3])


def _equality_pair():
    """``_base2`` cut to the segment alpha_1 = alpha_0 + 1 by two opposite rows."""
    return _star([0.25, -0.5], [[1, 0.5], [-0.25, 1]],
                 [[1, 1], [-1, 0.5], [1, -1], [-1, 1]], [2, 3, -1, 1],
                 [-3, 0.25], [0.5, 4])


def _empty():
    """``_base2`` with alpha_0 <= -2 and alpha_0 >= -1: empty by a margin of 1."""
    return _star([0.25, -0.5], [[1, 0.5], [-0.25, 1]],
                 [[1, 1], [-1, 0.5], [1, 0], [-1, 0]], [2, 3, -2, 1],
                 [-3, 0.25], [0.5, 4])


# The single-row property of the 2-D instances: unsafe iff y0 - y1 <= 0.
_ROW = [1.0, -1.0]
_SPEC1 = LinearSpec(np.array([_ROW]), np.array([0.0]))
_TINY = Fraction(1, 2 ** 32)

# (name, builder, exact minimum of the margin y0 - y1 over the star)
_REGIMES = [
    ("disjoint by 1/2", _base2, Fraction(1, 2)),
    ("meets in a region", _base2, Fraction(-1)),
    ("touches in one point", _base2, Fraction(0)),
    ("disjoint by 2**-32", _base2, _TINY),
    ("meets by 2**-32", _base2, -_TINY),
    ("3 variables, touches in one point", _base3, Fraction(0)),
    ("equality pair, disjoint by 1/2", _equality_pair, Fraction(1, 2)),
    ("equality pair, touches", _equality_pair, Fraction(0)),
    ("equality pair, meets", _equality_pair, Fraction(-1, 4)),
]


def _instance(builder, target):
    return _shifted(builder(), _ROW, target)


# The prune promises nothing for a star disjoint by less than a clear margin
# (keeping it is sound, dropping it is right): those regimes are checked only
# by the violation program below.
_PRUNE_REGIMES = [r for r in _REGIMES
                  if r[2] <= 0 or r[2] >= Fraction(1, 2)]


# --------------------------------------------------------------------------- #
# certified_prune_wave: never drops a star that meets the unsafe region          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,builder,target", _PRUNE_REGIMES,
                         ids=[r[0] for r in _PRUNE_REGIMES])
def test_prune_wave_keeps_a_star_that_meets_the_region_and_drops_a_clear_one(
        name, builder, target):
    """A star meeting the unsafe region (in a region, in one point, or by
    2**-32, also as a segment cut by an equality pair) is never pruned: a
    prune there is a false UNSAT. Catches a population solve that bounds the
    maximum instead of the minimum, a wrong-direction slack on the comparison
    (``lows > rhs - 1e-3 (1 + |rhs|)``) and any loss of the outward rhs pad. A
    star disjoint by 1/2 must be pruned (unit harm: the search never ends)."""
    S = _instance(builder, target)
    meets = _meets(S, _SPEC1.G, _SPEC1.g)
    if meets != (target <= 0):
        pytest.fail(f"{name}: the instance is not in its regime")
    pruned = certified_prune_wave([S], _SPEC1, device="cpu")
    assert len(pruned) == 1
    if meets:
        assert pruned[0] is False, \
            f"{name}: a star that meets the unsafe region was pruned"
    else:
        assert pruned[0] is True, \
            f"{name}: a star disjoint by {target} was not pruned"


def test_prune_wave_keeps_a_touching_star_that_rounding_makes_look_disjoint():
    """Exactly, ``G c - g = 2**53 + 3 - 2**53 - 3 = 0`` and the generator only
    adds a non-negative term, so the star touches the unsafe region at
    alpha = 0; ``G @ c`` rounds to 4, which reads as disjoint by 1. The prune
    must charge that rounding to the rhs (``outward_spec_rows``); without it the
    star is certified disjoint (a false UNSAT)."""
    S = _star([2.0 ** 53, 3.0, 2.0 ** 53], [[0.0], [1.0], [0.0]],
              np.zeros((0, 1)), np.zeros(0), [0.0], [1.0])
    spec = LinearSpec(np.array([[1.0, 1.0, -1.0]]), np.array([3.0]))
    if not _meets(S, spec.G, spec.g):
        pytest.fail("instance builder: the star must touch the region")
    assert certified_prune_wave([S], spec, device="cpu") == [False]


def test_prune_wave_flags_belong_to_their_own_star_in_a_mixed_wave():
    """One wave of stars of every shape the prune handles: point stars (no
    predicate variable), a star without a predicate box, 2- and 3-variable
    stars with 2 and 1 rows (zero-padded to one population), an equality pair
    and an empty star. Each flag must be the verdict of its own star: a flag
    written to the position among the solved stars instead of the position in
    the wave hands a disjoint star's certificate to a point inside the region
    (false UNSAT). Points are decided on the outward rhs: inside or touching
    must stay, violating a row by a clear margin must go."""
    inside = Star(np.array([[-1.0], [0.5]]), np.zeros((0, 0)), np.zeros((0, 1)))
    outside = Star(np.array([[2.0], [0.5]]), np.zeros((0, 0)), np.zeros((0, 1)))
    on_edge = Star(np.array([[0.75], [0.75]]), np.zeros((0, 0)),
                   np.zeros((0, 1)))
    boxless = Star(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                   np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]),
                   np.ones((4, 1)))
    wave = [
        (inside, False),
        (_instance(_base2, Fraction(1, 2)), True),
        (_instance(_base3, Fraction(-1)), False),
        (boxless, False),
        (outside, True),
        (_instance(_equality_pair, Fraction(0)), False),
        (on_edge, False),
        (_instance(_base2, -_TINY), False),
        (_instance(_base3, Fraction(1, 2)), True),
        (_empty(), None),                       # empty: either flag is sound
        (_instance(_base2, Fraction(-1)), False),
    ]
    stars = [s for s, _ in wave]
    for k, (s, want) in enumerate(wave):
        if want is False and s.predicate_lb is not None and s.nVar and \
                not _meets(s, _SPEC1.G, _SPEC1.g):
            pytest.fail(f"wave entry {k}: the instance must meet the region")
    flags = certified_prune_wave(stars, _SPEC1, device="cpu")
    assert len(flags) == len(stars)
    for k, ((s, want), got) in enumerate(zip(wave, flags)):
        if want is not None:
            assert got is want, f"wave entry {k}: flag {got}, expected {want}"
    # a budget of two gradient steps is still sound: nothing that meets goes
    flags2 = certified_prune_wave(stars, _SPEC1, max_iters=2, device="cpu")
    for k, ((s, want), got) in enumerate(zip(wave, flags2)):
        if want is False:
            assert got is False, f"wave entry {k} pruned at max_iters=2"


def _two_row_cases():
    """``_base2`` moved so that row 0 alone reaches -1; row 1 is -y0 <= g1.
    g1 = -2: both rows meet jointly; g1 = -3: each row meets alone, the
    conjunction misses by 61/288; g1 = -4: row 1 is violated on the whole star
    by 17/32."""
    S = _instance(_base2, Fraction(-1))
    G = np.array([[1.0, -1.0], [-1.0, 0.0]])
    return S, G


@pytest.mark.parametrize("g1,meets,clear", [(-2.0, True, False),
                                            (-4.0, False, True)],
                         ids=["jointly met", "second row violated everywhere"])
def test_prune_wave_on_a_two_row_region(g1, meets, clear):
    """A conjunction of two rows: the prune may drop the star only when one
    row is certified violated on the whole star (a sufficient test), so a
    jointly met region stays, and a star that violates the second row
    everywhere by 17/32 goes (unit harm otherwise). A prune that reads only the
    first row would keep the last case; one that requires every row violated
    would keep it too. (Where each row is met alone but not jointly, either
    flag is sound and nothing is promised; the violation program below covers
    that regime.)"""
    S, G = _two_row_cases()
    spec = LinearSpec(G, np.array([0.0, g1]))
    if _meets(S, G, spec.g) != meets:
        pytest.fail("instance builder: wrong regime")
    flag = certified_prune_wave([S], spec, device="cpu")[0]
    if meets:
        assert flag is False, "a star that meets a two-row region was pruned"
    if clear:
        assert flag is True, "a star violating one row everywhere was kept"


# --------------------------------------------------------------------------- #
# violation_lp: None only for an empty polytope; the optimum is the exact one   #
# --------------------------------------------------------------------------- #
# Every regime over the polytope; over the box only where the rows bind, so
# that the box optimum differs from the polytope one (-1.53125 against 1/2,
# -4.6875 against 0).
_LP_CASES = ([(r, True) for r in _REGIMES]
             + [(r, False) for r in _REGIMES
                if r[0] in ("disjoint by 1/2", "equality pair, touches")])


@pytest.mark.parametrize(
    "name,builder,target,include_Cd",
    [(*r, cd) for r, cd in _LP_CASES],
    ids=[f"{r[0]}-{'polytope' if cd else 'box'}" for r, cd in _LP_CASES])
def test_violation_lp_is_the_exact_epigraph_optimum(name, builder, target,
                                                    include_Cd):
    """``violation_lp`` returns ``(alpha, t)`` with ``t`` the exact minimum of
    the worst row margin over the polytope (``include_Cd=True``) or over its
    box (``False``), and ``alpha`` a point of that set where the worst margin
    is ``t``; it never answers ``None`` (vacuously safe) for a non-empty star,
    touching and 2**-32 cases included. Catches an objective on the wrong
    variable, a dropped or ignored ``include_Cd``, and a big-M that cuts off
    the optimum. Slack 1e-9: the optimum is a solver's, and ``t`` is used to
    order disjuncts and guide splits, not as a bound."""
    S = _instance(builder, target)
    want = _t_star(S, _SPEC1.G, _SPEC1.g, include_Cd)
    res = violation_lp(S, _SPEC1, include_Cd=include_Cd)
    assert res is not None, f"{name}: a non-empty star was reported empty"
    alpha, t = res
    assert alpha.shape == (S.nVar,)
    assert abs(t - float(want)) <= 1e-9, f"{name}: t = {t}, exact {want}"
    lb, ub = S.predicate_lb.ravel(), S.predicate_ub.ravel()
    assert np.all(alpha >= lb - 1e-9) and np.all(alpha <= ub + 1e-9)
    if include_Cd:
        assert np.all(S.C @ alpha <= S.d.ravel() + 1e-9), \
            f"{name}: the faithful witness is outside the polytope"
    at_alpha = _exact_max_margin(S, _SPEC1.G, _SPEC1.g, alpha)
    assert abs(float(at_alpha) - t) <= 1e-9


@pytest.mark.parametrize("g1", [-2.0, -3.0], ids=["jointly met",
                                                   "each row alone, not jointly"])
def test_violation_lp_states_every_row_of_the_region(g1):
    """The epigraph program must state the unsafe region of the whole
    property, not of its first row: with row 0 alone reaching -1 and the
    conjunction missing by 61/288, ``t`` is 61/288 > 0 (the node does not
    reach the region), not -1. Exact optimum by vertex enumeration."""
    S, G = _two_row_cases()
    spec = LinearSpec(G, np.array([0.0, g1]))
    want = _t_star(S, G, spec.g)
    alpha, t = violation_lp(S, spec, include_Cd=True)
    assert abs(t - float(want)) <= 1e-9, f"t = {t}, exact {want}"
    assert abs(float(_exact_max_margin(S, G, spec.g, alpha)) - t) <= 1e-9


def test_violation_lp_bounds_t_by_the_magnitude_of_negative_generators():
    """y = -10 a0 - 10 a1 + 5 over [-1, 1]**2, unsafe y <= 0: the optimum is
    t* = -15 at (1, 1). The bound on t must hold every margin over the box,
    with absolute values of the generator coefficients; a signed sum makes
    big-M negative, the program infeasible and the node 'empty' (a false
    UNSAT)."""
    S = _star([5.0], [[-10.0, -10.0]], np.zeros((0, 2)), np.zeros(0),
              [-1.0, -1.0], [1.0, 1.0])
    spec = LinearSpec(np.array([[1.0]]), np.array([0.0]))
    if _t_star(S, spec.G, spec.g) != -15:
        pytest.fail("instance builder: the exact optimum must be -15")
    res = violation_lp(S, spec, include_Cd=True)
    assert res is not None, "a reachable node was reported empty"
    assert abs(res[1] + 15.0) <= 1e-9


# --------------------------------------------------------------------------- #
# is_true_counterexample: membership of the input and of the output              #
# --------------------------------------------------------------------------- #
def _first_coordinate_net(n_in):
    """y = relu(x0) - relu(-x0) = x0, exactly in float64 for every x0 (one
    term is 0, the other |x0|), so a one-ulp change of x0 shows in y."""
    net = nn.Sequential(nn.Linear(n_in, 2), nn.ReLU(), nn.Linear(2, 1)).double()
    with torch.no_grad():
        w = torch.zeros((2, n_in), dtype=torch.float64)
        w[0, 0], w[1, 0] = 1.0, -1.0
        net[0].weight.copy_(w)
        net[0].bias.zero_()
        net[2].weight.copy_(torch.tensor([[1.0, -1.0]], dtype=torch.float64))
        net[2].bias.zero_()
    return net


def _replay(net, x):
    with torch.no_grad():
        return net(torch.as_tensor(np.asarray(x, np.float64)).reshape(1, -1)) \
            .numpy().ravel()


def test_ce_rejects_a_candidate_inside_the_box_and_outside_a_row():
    """The input star is the box [-1, 1]**2 cut by x0 + x1 <= -3/2 and, in a
    second star, also by x0 - x1 <= 1/2. A candidate inside the box with an
    unsafe output (x0 >= 0) that violates one cut row is not a member and must
    not be a counterexample, whether it satisfies the other rows or not;
    catches a membership test that drops the rows or accepts any one row. A
    member of the cut set with an unsafe output is a counterexample."""
    box = Star.from_bounds(np.array([[-1.0], [-1.0]]), np.array([[1.0], [1.0]]))
    one_cut = box.intersect_half_space(np.array([[1.0, 1.0]]),
                                       np.array([[-1.5]]))
    two_cuts = one_cut.intersect_half_space(np.array([[1.0, -1.0]]),
                                            np.array([[0.5]]))
    net = _first_coordinate_net(2)
    unsafe = LinearSpec(np.array([[-1.0]]), np.array([0.5]))   # x0 >= -1/2
    for S in (one_cut, two_cuts):
        for x, member in (([0.5, 0.5], False), ([0.25, -1.0], False),
                          ([-0.5, -1.0], True)):
            exact_member = (x[0] + x[1] <= -1.5 and
                            (S is one_cut or x[0] - x[1] <= 0.5))
            if exact_member != member:
                pytest.fail("instance builder: membership by hand is wrong")
            alpha = np.linalg.solve(S.V[:, 1:], np.array(x) - S.V[:, 0])
            is_ce, x_in, y = is_true_counterexample(alpha, S, net, unsafe)
            if member:
                assert is_ce, f"the member {x} with an unsafe output was rejected"
            else:
                assert not is_ce, f"the non-member {x} was accepted"


def test_ce_rejects_an_output_one_ulp_outside_the_unsafe_region():
    """The region is closed and the tolerance is zero: an output on the
    boundary is a counterexample, an output one unit in the last place
    outside it is not (from above and from below), with the default ``tol``.
    Catches any positive default tolerance (a near miss accepted as SAT). An
    explicit ``tol`` widens the region by that much."""
    net = _first_coordinate_net(1)
    S = Star.from_bounds(np.array([[-1.0]]), np.array([[1.0]]))
    alpha = np.array([0.5])                                      # x = y = 0.5
    if _replay(net, [0.5])[0] != 0.5:
        pytest.fail("the network must compute x exactly")
    below = np.nextafter(0.5, 0.0)
    above = np.nextafter(0.5, 1.0)
    for G, g, want in (([[1.0]], [0.5], True), ([[1.0]], [below], False),
                       ([[-1.0]], [-0.5], True), ([[-1.0]], [-above], False)):
        is_ce, x_in, y = is_true_counterexample(alpha, S, net,
                                                LinearSpec(np.array(G),
                                                           np.array(g)))
        assert y[0] == 0.5
        assert is_ce is want, f"G={G} g={g!r}: answered {is_ce}"
    spec = LinearSpec(np.array([[1.0]]), np.array([below]))
    assert is_true_counterexample(alpha, S, net, spec, tol=1e-15)[0] is True


def test_ce_evaluates_the_point_it_returns():
    """A box corner rebuilt from alpha = +1 lands one ulp above the box in x0
    (1.6120000000000003 for the bound 1.612); it must be snapped into the box
    BEFORE the forward pass, so that the returned output is the output of the
    returned point. With the unsafe region y >= 1.6120000000000003, no point
    of the box is a counterexample: evaluating the unsnapped corner and
    returning the snapped one is a SAT whose replay is safe."""
    lb, ub = np.array([0.615, 0.384]), np.array([1.612, 1.365])
    S = Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1))
    if not S.V[0, 0] + S.V[0, 1] > ub[0]:
        pytest.fail("instance builder: the corner must round above the box")
    net = _first_coordinate_net(2)
    thr = np.nextafter(1.612, 2.0)
    spec = LinearSpec(np.array([[-1.0]]), np.array([-thr]))      # y >= thr
    for signs in ([1.0, 1.0], [1.0, -1.0]):
        is_ce, x_in, y = is_true_counterexample(np.array(signs), S, net, spec)
        assert np.all(x_in >= lb) and np.all(x_in <= ub), "witness outside box"
        assert np.array_equal(y, _replay(net, x_in)), \
            "the returned output is not the output of the returned point"
        assert is_ce is False, "no point of the box reaches the region"


def test_ce_accepts_both_closed_faces_of_the_predicate_box():
    """The predicate box is closed on both sides: alpha exactly on the lower
    face or on the upper face, with an unsafe output, is a counterexample
    (a strict comparison on either face drops the vertex witnesses the exact
    program returns); one ulp outside either face is not a member. The star
    has a box that is not [-1, 1] and no raw float64 box, so nothing is
    snapped."""
    S = _star([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], np.zeros((0, 2)),
              np.zeros(0), [-0.75, -3.0], [0.5, 2.0])
    net = _first_coordinate_net(2)
    spec = LinearSpec(np.array([[1.0]]), np.array([1e9]))       # always unsafe
    for alpha, want in (([-0.75, 1.0], True), ([0.5, -3.0], True),
                        ([-0.75, -3.0], True), ([0.5, 2.0], True),
                        ([np.nextafter(-0.75, -1.0), 0.0], False),
                        ([np.nextafter(0.5, 1.0), 0.0], False),
                        ([0.0, np.nextafter(-3.0, -4.0)], False),
                        ([0.0, np.nextafter(2.0, 3.0)], False)):
        is_ce, x_in, _ = is_true_counterexample(np.array(alpha), S, net, spec)
        assert is_ce is want, f"alpha={alpha!r}: answered {is_ce}"


def test_ce_reads_only_the_input_prefix_of_a_longer_witness():
    """A witness of an output star carries the input's predicate variables
    first and the fresh variables of relaxed neurons after them: only the
    prefix of length ``input_star.nVar`` is the input. Trailing entries far
    outside any box must neither reject a member nor move the input."""
    S = _star([0.0, 0.0], [[2.0, 0.0], [0.0, 0.5]], np.zeros((0, 2)),
              np.zeros(0), [-1.0, -1.0], [1.0, 1.0])
    net = _first_coordinate_net(2)
    spec = LinearSpec(np.array([[-1.0]]), np.array([-0.5]))     # y >= 1/2
    alpha = np.array([0.5, -1.0, 1e9, -7.0])
    is_ce, x_in, y = is_true_counterexample(alpha, S, net, spec)
    assert np.array_equal(x_in, np.array([1.0, -0.5]))
    assert is_ce is True


def test_ce_runs_a_model_that_flattens_a_batched_input():
    """The check feeds the model one batched input of shape (1, n): a model
    that starts with ``nn.Flatten()`` (start_dim 1, the usual first layer of
    an image classifier) must run and give the same answer as the model
    without it; an unbatched input of shape (n,) makes Flatten fail."""
    net = _first_coordinate_net(2)
    flat = nn.Sequential(nn.Flatten(), *list(net))
    S = Star.from_bounds(np.array([[-1.0], [-1.0]]), np.array([[1.0], [1.0]]))
    spec = LinearSpec(np.array([[-1.0]]), np.array([-0.5]))     # x0 >= 1/2
    for alpha, want in (([0.75, 0.0], True), ([0.25, 0.0], False)):
        is_ce, x_in, y = is_true_counterexample(np.array(alpha), S, flat, spec)
        assert is_ce is want
        assert np.array_equal(y, _replay(net, x_in))


def test_ce_checks_the_rows_of_an_input_star_without_a_box():
    """An input star given by its rows alone (no predicate box): membership is
    decided by every row; a candidate that violates one row is rejected and a
    member is accepted."""
    S = Star(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
             np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.], [1., 1.]]),
             np.array([[1.], [1.], [1.], [1.], [0.25]]))
    net = _first_coordinate_net(2)
    spec = LinearSpec(np.array([[1.0]]), np.array([1e9]))
    assert is_true_counterexample(np.array([0.5, -0.5]), S, net, spec)[0] is True
    assert is_true_counterexample(np.array([0.5, 0.5]), S, net, spec)[0] is False


# --------------------------------------------------------------------------- #
# clip_to_raw_box                                                               #
# --------------------------------------------------------------------------- #
def test_clip_snaps_corners_of_a_box_with_a_zero_width_coordinate():
    """A box with a degenerate coordinate has fewer predicate variables than
    entries (2 * nVar box rows, fewer than 2 * dim): it is still a plain box,
    so every rebuilt corner, in a 1-D or a (1, n) array, is snapped into the
    raw float64 box, and the degenerate entry is its bound exactly."""
    lb = np.array([0.615, 0.3, 0.384])
    ub = np.array([1.612, 0.3, 1.365])
    S = Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1))
    if S.nVar != 2:
        pytest.fail("instance builder: expected two predicate variables")
    for signs in ([1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]):
        x = S.V[:, 0] + S.V[:, 1:] @ np.array(signs)
        for shaped in (x, x.reshape(1, -1)):
            y = clip_to_raw_box(shaped, S).ravel()
            assert np.all(y >= lb) and np.all(y <= ub)
            assert y[1] == 0.3


# --------------------------------------------------------------------------- #
# The public route: verify_refine                                              #
# --------------------------------------------------------------------------- #
_W1 = [[1.0, 1.0], [1.0, -1.0], [0.5, -1.0]]
_B1 = [0.0, 0.0, 0.25]
_W2 = [[1.0, 1.0, 0.0], [-0.25, 0.5, 1.0]]
_B2 = [0.0, -0.25]
_LB = [-1.0, -0.5]
_UB = [0.5, 0.25]


def _net():
    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor(_W1, dtype=torch.float64))
        net[0].bias.copy_(torch.tensor(_B1, dtype=torch.float64))
        net[2].weight.copy_(torch.tensor(_W2, dtype=torch.float64))
        net[2].bias.copy_(torch.tensor(_B2, dtype=torch.float64))
    return net


def _pieces():
    return R.affine(R.relu(R.affine(R.box(_LB, _UB), _W1, _B1)), _W2, _B2)


def _exact_reaches(pieces, G, g):
    """Exactly: does the network map some input of the box into G y <= g?"""
    G = [[Fraction(float(v)) for v in row] for row in np.atleast_2d(G)]
    g = [Fraction(float(v)) for v in np.ravel(g)]
    for p in pieces:
        A, b = list(p.A), list(p.b)
        for grow, gr in zip(G, g):
            A.append([R.dot(grow, [Wi[j] for Wi in p.W])
                      for j in range(len(p.A[0]))])
            b.append(gr - R.dot(grow, p.c))
        if R.feasible(A_ub=A, b_ub=b) is not None:
            return True
    return False


def _extremes():
    imgs = [v for p in _pieces() for v in p.image()]
    return max(v[0] for v in imgs), min(v[1] for v in imgs)


def _properties():
    """Properties of ``_net`` on the box, each with its exact verdict and
    whether it is decided by a margin of at least 1/8. The maximum of y0
    (exactly 1) is attained on the whole edge x0 = 1/2."""
    max0, min1 = _extremes()
    one = lambda G, g: [LinearSpec(np.array(G), np.array(g))]   # noqa: E731
    return [
        ("y0 >= max (touches)", one([[-1.0, 0.0]], [-float(max0)]), True, False),
        ("y0 >= max + 2**-31 (near miss)",
         one([[-1.0, 0.0]], [-float(max0 + Fraction(1, 2 ** 31))]), False, False),
        ("y0 >= max - 2**-31 (meets by a sliver)",
         one([[-1.0, 0.0]], [-float(max0 - Fraction(1, 2 ** 31))]), True, False),
        ("y0 >= max - 1/8", one([[-1.0, 0.0]], [-float(max0 - Fraction(1, 8))]),
         True, True),
        ("two disjuncts, the second met",
         one([[-1.0, 0.0]], [-float(max0 + Fraction(1, 4))])
         + one([[0.0, 1.0]], [float(min1 + Fraction(1, 8))]), True, True),
        ("two disjuncts, neither met",
         one([[-1.0, 0.0]], [-float(max0 + Fraction(1, 4))])
         + one([[0.0, 1.0]], [float(min1 - Fraction(1, 4))]), False, True),
        ("two rows, each met alone, not jointly",
         one([[-1.0, 0.0], [0.0, 1.0]], [-0.75, -0.25]), False, False),
        ("two rows, met jointly",
         one([[-1.0, 0.0], [0.0, 1.0]], [-0.75, 0.25]), True, True),
    ]


_PROPS = _properties()
# The exact checks; the wave witness behind the exact prune; the certified
# prune with the wave witness (the GPU defaults). The certified prune with the
# exact witness runs the same prune code and the same exact-leaf backstop.
_MODES = [("exact", "exact"), ("exact", "dual_gpu"), ("dual_gpu", "dual_gpu")]


def _assert_honest(res, net, specs, lb, ub, what):
    """A SAT's counterexample lies in the raw float64 box and its float64
    forward pass lies in the closed unsafe region of some disjunct, at zero
    tolerance (the text of the property, not the library's checker)."""
    if res.status != Status.SAT:
        return
    x = np.asarray(res.counterexample_x, np.float64).ravel()
    assert np.all(x >= np.asarray(lb)) and np.all(x <= np.asarray(ub)), \
        f"{what}: counterexample {x!r} outside the input box"
    y = _replay(net, x)
    assert any(np.all(np.asarray(s.G) @ y <= np.asarray(s.g)) for s in specs), \
        f"{what}: counterexample output {y!r} is in no unsafe region"


@pytest.mark.parametrize("prune_mode,witness_mode", _MODES,
                         ids=[f"prune={p},witness={w}" for p, w in _MODES])
@pytest.mark.parametrize("name,specs,truth,clear", _PROPS,
                         ids=[p[0] for p in _PROPS])
def test_verify_refine_verdict_matches_the_exact_image(
        name, specs, truth, clear, prune_mode, witness_mode, monkeypatch):
    """``verify_refine`` against the exact image of a 2-3-2 ReLU network with
    dyadic weights on an asymmetric box, for single rows, a conjunction of two
    rows and a disjunction of two properties, under every prune and witness
    mode (the certified GPU engine on CPU torch). UNSAT only when no input
    reaches the region; SAT only with a counterexample in the box whose
    float64 output is in a region; a property decided by a margin of 1/8 or
    more gets its verdict. The near miss (2**-31 outside) catches a positive
    default tolerance of the counterexample check; the sliver (2**-31 inside)
    catches a prune with a wrong-direction slack or on upper bounds; the
    disjunctions catch a search that reads only the first disjunct."""
    monkeypatch.setattr("n2v.utils.lpsolver_gpu.gpu_available", lambda: False)
    if any(_exact_reaches(_pieces(), s.G, s.g) for s in specs) is not truth:
        pytest.fail(f"{name}: the exact verdict is not the stated one")
    net = _net()
    S = Star.from_bounds(np.array(_LB).reshape(-1, 1),
                         np.array(_UB).reshape(-1, 1))
    res = verify_refine(S, net, specs if len(specs) > 1 else specs[0],
                        prune_mode=prune_mode, witness_mode=witness_mode,
                        frontier_width=4 if prune_mode == "dual_gpu" else 1,
                        node_budget=400)
    _assert_honest(res, net, specs, _LB, _UB, name)
    if truth:
        assert res.status != Status.UNSAT, f"{name}: false UNSAT"
        if clear:
            assert res.status == Status.SAT, f"{name}: {res.status}"
    else:
        assert res.status != Status.SAT, f"{name}: false SAT"
        if clear:
            assert res.status == Status.UNSAT, f"{name}: {res.status}"


def test_verify_refine_finds_a_counterexample_behind_negative_generators():
    """y = -10 relu(x0 + 2) - 10 relu(x1 + 2) + 40 over [-1, 1]**2, unsafe
    y <= -5: the corner (1, 1) gives y = -20. The exact witness program of the
    root node has negative spec-row generators; a big-M from their signed sum
    makes the node 'empty' and the answer a false UNSAT."""
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.eye(2, dtype=torch.float64))
        net[0].bias.fill_(2.0)
        net[2].weight.copy_(torch.tensor([[-10.0, -10.0]], dtype=torch.float64))
        net[2].bias.fill_(40.0)
    lb, ub = [-1.0, -1.0], [1.0, 1.0]
    S = Star.from_bounds(np.array(lb).reshape(-1, 1), np.array(ub).reshape(-1, 1))
    spec = LinearSpec(np.array([[1.0]]), np.array([-5.0]))
    if _replay(net, [1.0, 1.0])[0] != -20.0:
        pytest.fail("instance builder: (1, 1) must give -20")
    res = verify_refine(S, net, spec, prune_mode="exact", witness_mode="exact")
    _assert_honest(res, net, [spec], lb, ub, "negative generators")
    assert res.status == Status.SAT


@pytest.mark.parametrize("offset,truth", [(0, True), (1, False)],
                         ids=["region starts at the bound", "one ulp above"])
def test_verify_refine_corner_witness_is_snapped_into_the_raw_box(offset, truth):
    """y = x0 on the box [0.615, 1.612] x [0.384, 1.365], whose corner rebuilt
    from alpha = +1 lands one ulp above 1.612. Unsafe y >= 1.612: SAT, and the
    counterexample must be the snapped corner, inside the box (a zero-tolerance
    replay rejects the unsnapped one). Unsafe y >= 1.6120000000000003: no
    point of the box reaches it, so SAT is wrong; a check that evaluates the
    unsnapped corner answers it."""
    lb, ub = np.array([0.615, 0.384]), np.array([1.612, 1.365])
    S = Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1))
    net = _first_coordinate_net(2)
    thr = 1.612 if offset == 0 else np.nextafter(1.612, 2.0)
    spec = LinearSpec(np.array([[-1.0]]), np.array([-thr]))
    res = verify_refine(S, net, spec, prune_mode="exact", witness_mode="exact")
    _assert_honest(res, net, [spec], lb, ub, f"threshold {thr!r}")
    if truth:
        assert res.status == Status.SAT
    else:
        assert res.status != Status.SAT


def _cut_box_instance():
    """A box cut by one half-space, found by search: the vertex witness of
    the exact program is accepted by ``is_true_counterexample`` but the input
    rebuilt from it in float64 is not a member of the input star."""
    lb = np.array([-1.556, 0.459, 0.05])
    ub = np.array([0.038999999999999924, 0.923, 1.6740000000000002])
    H = np.array([[-0.5, 0.88, -1.07]])
    h = np.array([[0.1649899999999999]])
    d = np.array([0.91, -0.02, -1.25])
    S = Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1)) \
        .intersect_half_space(H, h)
    return S, d


def test_verify_refine_witness_of_a_plain_box_is_a_member_of_the_input_star():
    """Neighbour of the marked case below, with the same reference: on the
    plain box (raw-box snap applies) the counterexample of an always-unsafe
    property is an exact member of the input star (``tests/_oracles``)."""
    S0, d = _cut_box_instance()
    lb = np.array([-1.556, 0.459, 0.05])
    ub = np.array([0.038999999999999924, 0.923, 1.6740000000000002])
    S = Star.from_bounds(lb.reshape(-1, 1), ub.reshape(-1, 1))
    net = nn.Sequential(nn.Linear(3, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor(d.reshape(1, -1)))
        net[0].bias.zero_()
    spec = LinearSpec(np.array([[1.0]]), np.array([1e9]))
    res = verify_refine(S, net, spec, prune_mode="exact", witness_mode="exact")
    assert res.status == Status.SAT
    x = np.asarray(res.counterexample_x, np.float64).ravel()
    assert R.member(R.star(S), [Fraction(v) for v in x])


@known_bug(19, "verify_refine accepts a counterexample by checking its predicate "
           "point against the input star's rows in float64, then returns the "
           "input x = c + V alpha rounded to nearest; for an input star that is "
           "not a plain box (no raw-box snap) x can lie outside the input star, "
           "so SAT comes with a witness that a zero-tolerance replay rejects")
def test_verify_refine_witness_of_a_cut_box_is_a_member_of_the_input_star():
    """On a box cut by a half-space, the counterexample of an always-unsafe
    property must be a member of the input star at zero tolerance, decided
    exactly in rationals (``tests/_oracles/rational.py``) on the returned
    point. Today the float64 rebuild of the vertex witness lands outside it."""
    S, d = _cut_box_instance()
    net = nn.Sequential(nn.Linear(3, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor(d.reshape(1, -1)))
        net[0].bias.zero_()
    spec = LinearSpec(np.array([[1.0]]), np.array([1e9]))
    res = verify_refine(S, net, spec, prune_mode="exact", witness_mode="exact")
    if res.status != Status.SAT:
        pytest.fail(f"an always-unsafe property must be SAT, got {res.status}")
    x = np.asarray(res.counterexample_x, np.float64).ravel()
    assert R.member(R.star(S), [Fraction(v) for v in x]), \
        f"counterexample {x!r} is not a member of the input star"

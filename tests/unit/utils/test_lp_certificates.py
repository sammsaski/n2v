"""Regressions for the LP emptiness certificates and LP-derived bounds in
``n2v.utils.lpsolver`` and ``n2v.sets.predicate_box``.

  * The predicate-box finitizer must not report a feasible polytope empty,
    nor return a box that cuts off feasible points, on tiny-coefficient rows
    (the default presolve-on status and optima are wrong there).
  * Without a finite predicate box no Farkas/NS certificate can be formed;
    emptiness must still be decidable (membership, ``is_empty_set``, UNSAT).
  * ``check_feasibility`` decides a certifiable system with one careful solve.
  * The certificate's row slack is a rounding bound, not a flat 1e-7 that
    compounds through the rows and the output weights.
  * The default solve row-scales, so tiny-coefficient rows are not dropped,
    without scaling a wide-range row so far that its small entries are.
  * With a finite box, LP optima never lie inside the true optimum: they are
    pushed outward to the NS bound built from the solver's duals.
  * On ill-conditioned polytopes the finitizer keeps the more outward of two
    solves and never returns an inverted (empty) box.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

import n2v.utils.lpsolver as lpmod
from n2v.nn.reach import reach_pytorch_model
from n2v.sets.halfspace import HalfSpace
from n2v.sets.predicate_box import ensure_finite_predicate_box
from n2v.sets.star import Star
from n2v.utils.lpsolver import (_HAS_HIGHSPY, _row_scaled, _scipy_status,
                                check_feasibility, confirm_infeasible,
                                solve_lp, solve_lp_batch)
from n2v.utils.verify_specification import verify_specification

needs_highspy = pytest.mark.skipif(
    not _HAS_HIGHSPY,
    reason="the Farkas/NS certificate needs highspy's dual ray")


def _tiny_polytope(rng, scale):
    """Rows with coefficients ~``scale`` that contain a known point ``x0``
    (slack ~1e-3 * scale), bounded by +-e_i rows at the same scale."""
    n = int(rng.integers(2, 5))
    m = int(rng.integers(n + 1, 3 * n + 3))
    x0 = rng.uniform(-1, 1, n)
    A = rng.standard_normal((m, n)) * scale
    b = A @ x0 + np.abs(rng.standard_normal(m)) * scale * 1e-3
    A = np.vstack([A, np.eye(n) * scale, -np.eye(n) * scale])
    b = np.concatenate([b, (x0 + 2) * scale, -(x0 - 2) * scale])
    assert np.all(A @ x0 <= b)
    return A, b, x0


def _identity_star(A, b, lb=None, ub=None):
    n = A.shape[1]
    return Star(np.hstack([np.zeros((n, 1)), np.eye(n)]), A, b.reshape(-1, 1),
                lb, ub)


# ---------------------------------------------------------------------------
# Predicate-box finitizer on tiny-coefficient polytopes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("box", ["none", "partly_infinite"])
@pytest.mark.parametrize("scale", [1e-8, 1e-9])
def test_finitizer_keeps_known_feasible_tiny_coefficient_polytopes(scale, box):
    """Each polytope contains ``x0``. The default presolve-on bound LPs
    reported ~1 in 4 of them empty at 1e-9 (reach -> [] -> vacuous UNSAT)
    and cut ``x0`` out of the box on half of the rest at 1e-8."""
    rng = np.random.default_rng(7)
    for _ in range(40):
        A, b, x0 = _tiny_polytope(rng, scale)
        n = x0.size
        if box == "none":
            S = _identity_star(A, b)
        else:
            lb = np.full((n, 1), -np.inf)
            lb[0, 0] = -3.0
            S = _identity_star(A, b, lb, np.full((n, 1), np.inf))
        S2, empty = ensure_finite_predicate_box(S)
        assert not empty
        assert np.all(S2.predicate_lb.ravel() <= x0)
        assert np.all(x0 <= S2.predicate_ub.ravel())


@pytest.mark.parametrize("scale", [1e-8, 1e-9])
def test_tiny_coefficient_input_star_is_not_a_false_unsat(scale):
    """End to end on a box-less input polytope containing x0: the unsafe
    region contains f(x0), so UNSAT is wrong. It was UNSAT at both scales:
    at 1e-9 the finitizer called the polytope empty (reach -> []), at 1e-8
    the finitized box cut x0 off."""
    A, b, x0 = _tiny_polytope(np.random.default_rng(0), scale)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(x0.size, 4), nn.ReLU(),
                        nn.Linear(4, 1)).double()
    y0 = net(torch.tensor(x0)).item()
    unsafe = HalfSpace(np.array([[-1.0]]), np.array([[-(y0 - 0.5)]]))
    for method in ("approx", "exact"):
        R = reach_pytorch_model(net, _identity_star(A, b), method=method)
        assert len(R) > 0, method
        assert verify_specification(R, unsafe).verdict != "UNSAT", method


def test_unconfirmed_infeasible_bound_lp_is_not_emptiness(monkeypatch):
    """A bound LP that says 'infeasible' on a feasible polytope must not make
    the finitizer report the star empty; with no trustworthy box it raises."""
    fake = lambda *a, **k: (None, None, "infeasible", {})     # noqa: E731
    monkeypatch.setattr(lpmod, "solve_lp", fake)
    monkeypatch.setattr(lpmod, "_solve_lp_careful", fake, raising=False)
    C = np.vstack([np.eye(2), -np.eye(2)])
    S = _identity_star(C, np.ones(4))                         # [-1, 1]^2
    with pytest.raises(ValueError, match="could not be confirmed"):
        ensure_finite_predicate_box(S)


# ---------------------------------------------------------------------------
# Emptiness without a finite predicate box
# ---------------------------------------------------------------------------

def test_infeasible_system_without_finite_box_is_reported_empty():
    A = np.array([[1.0], [-1.0]])
    b = np.array([-1.0, -1.0])                                 # x <= -1, x >= 1
    assert check_feasibility(A=A, b=b) is False
    assert check_feasibility(A=A, b=b, lb=np.array([-np.inf]),
                             ub=np.array([np.inf])) is False
    # lower bound only: x0 <= -1 with x >= 0
    assert check_feasibility(A=np.array([[1.0, 0.0]]), b=np.array([-1.0]),
                             lb=np.zeros(2)) is False
    assert check_feasibility(A=np.array([[1.0, 0.0]]), b=np.array([1.0]),
                             lb=np.zeros(2)) is True


def test_boxless_star_membership_checks_cd_rows():
    """[0, 1]^2 given only through C/d: membership used to be True for every
    point, since emptiness could never be shown without a finite box."""
    C = np.vstack([np.eye(2), -np.eye(2)])
    S = _identity_star(C, np.array([1.0, 1.0, 0.0, 0.0]))
    assert S.contains(np.array([0.5, 0.5]))
    for p in ([5.0, 5.0], [100.0, 100.0], [100.0, -100.0]):
        assert not S.contains(np.array(p)), p
    # triangle a, b >= 0, a + b <= 1
    T = _identity_star(np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]),
                       np.array([0.0, 0.0, 1.0]))
    assert not T.contains(np.array([0.9, 0.9]))


def test_boxless_star_disjoint_from_the_unsafe_region_is_unsat():
    C = np.vstack([np.eye(2), -np.eye(2)])
    S = _identity_star(C, np.array([1.0, 1.0, 0.0, 0.0]))
    G, g = np.array([[-1.0, 0.0]]), np.array([[-10.0]])       # y0 >= 10
    assert S.intersect_half_space(G, g).is_empty_set()
    hull = S.convex_hull(S)
    assert hull.intersect_half_space(G, g).is_empty_set()
    assert verify_specification([S], HalfSpace(G, g)).verdict == "UNSAT"


# ---------------------------------------------------------------------------
# One careful solve per certifiable emptiness check
# ---------------------------------------------------------------------------

@needs_highspy
@pytest.mark.parametrize("rhs, expected", [(-2.0, False), (0.5, True)])
def test_certifiable_check_runs_one_careful_solve(rhs, expected, monkeypatch):
    """The default solve's 'infeasible' has to be confirmed by the careful
    solve anyway, so on a certifiable system it only duplicated that solve."""
    calls = {"default": 0, "careful": 0}
    solve, robust = lpmod.solve_lp, lpmod._feasibility_robust

    def spy_solve(*a, **k):
        calls["default"] += 1
        return solve(*a, **k)

    def spy_robust(*a, **k):
        calls["careful"] += 1
        return robust(*a, **k)

    monkeypatch.setattr(lpmod, "solve_lp", spy_solve)
    monkeypatch.setattr(lpmod, "_feasibility_robust", spy_robust)
    A = np.array([[1.0, 1.0], [1.0, -1.0]])
    feasible = check_feasibility(A=A, b=np.array([rhs, 1.0]),
                                 lb=np.zeros(2), ub=np.ones(2))
    assert feasible is expected
    assert calls == {"default": 0, "careful": 1}


# ---------------------------------------------------------------------------
# The certificate's row slack
# ---------------------------------------------------------------------------

@needs_highspy
def test_small_margin_over_many_rows_is_still_certified():
    """x_i <= 1 for 20 variables but sum x >= 20 + 1e-6: infeasible by 1e-6.
    A flat 1e-7 * (1 + |b_i|) slack on every row adds up to 4e-6 and made
    the system feasible, so the certificate was lost."""
    n = 20
    A = np.vstack([np.eye(n), -np.ones((1, n))])
    b = np.concatenate([np.ones(n), [-(n + 1e-6)]])
    lb, ub = np.zeros(n), 2.0 * np.ones(n)
    assert confirm_infeasible(A, b, lb, ub)
    assert check_feasibility(A=A, b=b, lb=lb, ub=ub) is False
    # the same system with a margin of -1e-6 is feasible: never certified
    b_ok = np.concatenate([np.ones(n), [-(n - 1e-6)]])
    assert not confirm_infeasible(A, b_ok, lb, ub)
    assert check_feasibility(A=A, b=b_ok, lb=lb, ub=ub) is True


@needs_highspy
def test_certified_disjoint_output_star_is_unsat():
    """Output gain 1e3 amplified the flat row slack into ~2.5e-4 on y0, so a
    spec 1.9e-4 above the NS-certified maximum came back UNKNOWN."""
    torch.manual_seed(4)
    net = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 12),
                        nn.ReLU(), nn.Linear(12, 3))
    with torch.no_grad():
        net[4].weight.mul_(1e3)
        net[4].bias.mul_(1e3)
    S = reach_pytorch_model(
        net, Star.from_bounds(-0.3 * np.ones((4, 1)), 0.3 * np.ones((4, 1))),
        method="approx")[0]
    y0_max = float(S.get_max(0))
    thr = y0_max + 1e-6 * (1.0 + abs(y0_max))
    unsafe = HalfSpace(np.array([[-1.0, 0.0, 0.0]]), np.array([[-thr]]))
    assert verify_specification([S], unsafe).verdict == "UNSAT"


# ---------------------------------------------------------------------------
# Default-solve optima on tiny-coefficient rows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1e-8, 1e-9])
def test_default_lp_bounds_bracket_a_known_feasible_point(scale):
    """HiGHS drops matrix entries below 1e-9 and applies its tolerances to
    unscaled rows: the presolve-on optimum cut off the known point on ~1 in 3
    bound LPs at these scales, or the LP came back 'infeasible'."""
    rng = np.random.default_rng(5)
    for _ in range(25):
        A, b, x0 = _tiny_polytope(rng, scale)
        n = x0.size
        lb, ub = -5.0 * np.ones(n), 5.0 * np.ones(n)
        for i in range(n):
            f = np.eye(n)[i]
            _, lo, st_lo, _ = solve_lp(f, A, b, lb=lb, ub=ub, minimize=True)
            _, hi, st_hi, _ = solve_lp(f, A, b, lb=lb, ub=ub, minimize=False)
            assert st_lo == "optimal" and st_hi == "optimal"
            assert lo <= x0[i] + 1e-7 and x0[i] - 1e-7 <= hi
        vals = solve_lp_batch([np.eye(n)[i] for i in range(n) for _ in (0, 1)],
                              A, b, lb, ub, [True, False] * n)
        for i in range(n):
            assert vals[2 * i] <= x0[i] + 1e-7 <= vals[2 * i + 1] + 2e-7


def test_exact_reach_on_tiny_coefficient_star_keeps_the_true_output():
    """A FINITE-box star whose rows have ~1e-8 coefficients; x_0 ranges over
    about [0.8849, 0.8861]. With y = relu(x_0 - 1) the neuron is always
    inactive, so y = 0 and the unsafe region y >= 0 is reached. The
    presolve-on LP put min x_0 at 1.08, so the neuron was classified always
    active and exact reach returned y = x_0 - 1 < 0 only. The unscaled
    emptiness solve dropped the same rows and hid this as UNKNOWN; with the
    certified emptiness check it is a false UNSAT unless the bound LPs are
    right. The polytope is the 632nd draw of the generator that found it."""
    rng = np.random.default_rng(123)
    for _ in range(632):
        x0 = rng.uniform(-1, 1, 4)
        A = rng.normal(size=(12, 4)) * 1e-8
        b = A @ x0 + rng.uniform(0, 1e-3, 12) * 1e-8
    A = np.vstack([A, np.eye(4) * 1e-8, -np.eye(4) * 1e-8])
    b = np.concatenate([b, (x0 + 1.0) * 1e-8, -(x0 - 1.0) * 1e-8])
    assert np.all(A @ x0 <= b)
    S = _identity_star(A, b, -5.0 * np.ones((4, 1)), 5.0 * np.ones((4, 1)))
    lin = nn.Linear(4, 1)
    with torch.no_grad():
        lin.weight.zero_()
        lin.weight[0, 0] = 1.0
        lin.bias.fill_(-1.0)
    net = nn.Sequential(lin, nn.ReLU()).double()
    unsafe = HalfSpace(np.array([[-1.0]]), np.array([[0.0]]))
    for method in ("exact", "approx"):
        R = reach_pytorch_model(net, S, method=method)
        assert verify_specification(R, unsafe).verdict != "UNSAT", method
    assert S.get_min(0) <= x0[0] <= S.get_max(0)


# ---------------------------------------------------------------------------
# Ill-conditioned polytopes (coefficients over many orders of magnitude in
# the same rows).
# Each contains the known point x0 exactly (checked in rational arithmetic).
# ---------------------------------------------------------------------------

# Columns 0-1 carry ~1e-5 coefficients and column 2 up to 3e5 in the same
# rows; the true set is a thin sliver around x0.
_SLIVER_A = np.array([
    [-0.0009444068006011457, -0.00028036707751405725, -95537.5432287469],
    [0.000794512760904477, -0.0003291355611878068, -93039.85021883316],
    [-0.0005849717059829613, -1.3571722473123786e-05, -107439.82292262034],
    [0.0007454528993576339, 0.00019342785340530717, -316087.74201939837],
    [-2.8134729550293924e-05, -0.00035114419358625294, 299680.77607638325],
    [0.002594520773374021, -2.4554183122037945e-05, 49189.80013208265],
    [-0.0008351420471036032, -0.0003394050216239027, -272048.3054075197],
    [9.600620774917835e-05, 0.0001135760180194239, -26175.116298003806],
    [0.0008533524651040744, 0.0005902749168788954, 40849.79200359032],
    [0.001606759460140607, -0.0003777195047838554, -234310.51596463422],
    [0.0004219292650719562, 0.00035526788507658373, 44697.313900914996],
    [1.0, 0.0, 0.0], [-1.0, -0.0, -0.0], [0.0, 1.0, 0.0],
    [-0.0, -1.0, -0.0], [0.0, 0.0, 1.0], [-0.0, -0.0, -1.0]])
_SLIVER_B = np.array([
    -55376.47261719984, -53928.735187113336, -62275.39718569669,
    -183213.43945469032, 173704.11650608698, 28511.905800596913,
    -157687.4931442766, -15171.89568671531, 23677.784433982564,
    -135813.52182658567, 25907.92588844256, 1.0387266452327888,
    2.8784059243760565, 1.109827505759288, 0.8818923999029602,
    4.85372914623387, 3.694468156627032])
_SLIVER_X0 = np.array([-0.9198396395716337, 0.11396755292816385,
                       0.5796304948034188])

# Rows 0-3 span ~1e-14 .. 7e6; x_2 ranges over about [-0.13952496, x0_2].
_WIDE_A = np.array([
    [-6612990.421458409, -0.0004587988223973387, 2036.018382907806],
    [1.8324714678092613, 7.142725962911682e-11, -9.804892864743469e-05],
    [0.013636917363310469, -4.419421293956286e-14, -9.463643275315362e-08],
    [-0.2529112061610295, -1.9625569020208688e-12, -7.5727483665358515e-06],
    [-0.24262591817219106, 0.4115682827403698, -0.2891489725289978],
    [0.24262591817219106, -0.4115682827403698, 0.2891489725289978],
    [-0.21944468588194543, -2.1311771602253686, 1.0801557419503236],
    [0.21944468588194543, 2.1311771602253686, -1.0801557419503236]])
_WIDE_B = np.array([
    -730724655255.8989, 202488.0033413029, 1506.9027486843158,
    -27946.675324781787, -26819.152582133363, 26819.152582133367,
    -24201.95922843778, 24201.959228437783])
_WIDE_X0 = np.array([110499.94877666538, -21.968154587861655,
                     -0.1395247698857329])


def test_finitized_box_of_a_column_scaled_sliver_contains_the_known_point():
    """The finitizer returned lb > ub on dims 0 and 1 (an empty box on a
    feasible polytope): reach and verify_specification then said UNSAT for
    every spec, while x0 is a counterexample."""
    n = _SLIVER_X0.size
    S = _identity_star(_SLIVER_A, _SLIVER_B, np.full((n, 1), -np.inf),
                       np.full((n, 1), np.inf))
    S2, empty = ensure_finite_predicate_box(S)
    assert not empty
    assert np.all(S2.predicate_lb.ravel() <= _SLIVER_X0)
    assert np.all(_SLIVER_X0 <= S2.predicate_ub.ravel())
    lin = nn.Linear(n, n).double()
    with torch.no_grad():
        lin.weight.copy_(torch.eye(n, dtype=torch.float64))
        lin.bias.fill_(100.0)
    net = nn.Sequential(lin, nn.ReLU()).double()
    y0 = _SLIVER_X0[0] + 100.0            # f(x0)_0: the ReLU is the identity
    unsafe = HalfSpace(np.eye(n)[:1], np.array([[y0 + 1e-4 * (1 + y0)]]))
    for method in ("approx", "exact"):
        R = reach_pytorch_model(net, S, method=method)
        assert verify_specification(R, unsafe).verdict != "UNSAT", method


def _fake_solves(monkeypatch, careful, default):
    """Replace the finitizer's two bound solves by ``(minimize, dim) ->
    (x, val)`` lookups."""
    def wrap(table):
        def solve(f, A, b, lb=None, ub=None, lp_solver=None, minimize=True,
                  **kw):
            x, val = table(minimize, int(np.argmax(f)))
            return np.asarray(x, dtype=float), val, "optimal", {}
        return solve
    monkeypatch.setattr(lpmod, "_solve_lp_careful",
                        lambda f, A, b, lb, ub, minimize=True:
                        wrap(careful)(f, A, b, minimize=minimize))
    monkeypatch.setattr(lpmod, "solve_lp", wrap(default))


def test_finitizer_keeps_the_more_outward_of_its_two_solves(monkeypatch):
    """[0, 1]^2: the careful solve stops inside on dim 0 (min 0.4, max 0.6);
    the box must come from the other solve."""
    def exact(mn, d):
        return ([0.0, 0.0], 0.0) if mn else ([1.0, 1.0], 1.0)

    def inside(mn, d):
        if d == 0:
            return ([0.4, 0.0], 0.4) if mn else ([0.6, 1.0], 0.6)
        return exact(mn, d)

    _fake_solves(monkeypatch, careful=inside, default=exact)
    C = np.vstack([np.eye(2), -np.eye(2)])
    S2, empty = ensure_finite_predicate_box(
        _identity_star(C, np.array([1.0, 1.0, 0.0, 0.0])))
    assert not empty
    assert np.all(S2.predicate_lb.ravel() <= 0.0)
    assert np.all(S2.predicate_ub.ravel() >= 1.0)


def test_finitizer_box_contains_every_lp_solution_point(monkeypatch):
    """Both solves stop inside on dim 0 ([0.4, 0.6]), but the dim-1 LPs
    return feasible points with x_0 = 0 and 1: the box must contain them."""
    def solve(mn, d):
        if d == 0:
            return ([0.4, 0.5], 0.4) if mn else ([0.6, 0.5], 0.6)
        return ([0.0, 0.0], 0.0) if mn else ([1.0, 1.0], 1.0)

    _fake_solves(monkeypatch, careful=solve, default=solve)
    C = np.vstack([np.eye(2), -np.eye(2)])
    S2, _ = ensure_finite_predicate_box(
        _identity_star(C, np.array([1.0, 1.0, 0.0, 0.0])))
    assert S2.predicate_lb[0, 0] <= 0.0 and S2.predicate_ub[0, 0] >= 1.0


def test_finitizer_raises_when_its_bound_lps_contradict(monkeypatch):
    """A minimum above the maximum is an empty box: a vacuous UNSAT."""
    def solve(mn, d):
        return ([0.6, 0.5], 0.6) if mn else ([0.4, 0.5], 0.4)

    _fake_solves(monkeypatch, careful=solve, default=solve)
    C = np.vstack([np.eye(2), -np.eye(2)])
    with pytest.raises(ValueError, match="contradict"):
        ensure_finite_predicate_box(
            _identity_star(C, np.array([1.0, 1.0, 0.0, 0.0])))


def test_row_scaling_keeps_small_entries_of_wide_range_rows():
    """Scaling a row's largest entry to [0.5, 1) pushed its small entries
    under HiGHS's drop threshold; the scaling must stay exact and keep every
    entry that was above the threshold at least twice above it."""
    A = np.array([[3.0, 1e-3, 0.0], [1e-8, 2e-8, 0.0], [1e4, 1e-7, 1.0],
                  [1e6, 3e-6, 0.0], [0.0, 0.0, 0.0], [0.7, 0.1, 0.0]])
    b = np.arange(6.0)
    for keep in (1e-12, 1e-9):
        A2, b2, s = _row_scaled(A, b, keep)
        assert np.array_equal(A2 * s[:, None], A)
        assert np.array_equal(b2 * s, b)
        kept = np.abs(A2[np.abs(A) > keep])
        assert np.all(kept >= 2 * keep)
        # rows the guard does not bind reach [0.5, 1)
        for i in (0, 1, 5):
            assert 0.5 <= np.abs(A2[i]).max() < 1.0
    # a range beyond ~1e24 is still scaled under HiGHS's 1e15 entry limit
    A2, _, _ = _row_scaled(np.array([[1e30, 1e-10]]), np.array([1e30]))
    assert np.abs(A2).max() < 1e15


def test_scipy_failures_are_not_read_as_infeasible():
    """scipy reports HiGHS kModelError under its infeasible code 2, and a
    code-4 failure message names HiGHS's primal status ('Infeasible'); both
    were read as the verdict 'infeasible'."""
    class Res:
        def __init__(self, status, message):
            self.status, self.message = status, message
    assert _scipy_status(Res(2, "The problem is infeasible. (HiGHS Status "
                                "8: model_status is Infeasible)")) == \
        "infeasible"
    assert _scipy_status(Res(2, "(HiGHS Status 2: Model error)")) != \
        "infeasible"
    assert _scipy_status(Res(4, "The HiGHS status code was not recognized. "
                                "(HiGHS Status 15: model_status is Unknown; "
                                "primal_status is Infeasible)")) != \
        "infeasible"
    assert _scipy_status(Res(4, "The problem is unbounded or infeasible. "
                                "")).startswith("failed")


@pytest.mark.parametrize("highspy_available", [False, True])
def test_row_with_an_entry_above_the_highs_limit_is_still_solved(
        highspy_available, monkeypatch):
    """x = (0.5, 0.5) is feasible; row 1 spans 1e30 .. 1e-10. Unscaled,
    HiGHS rejects the model (scipy files that under code 2, 'infeasible')."""
    if highspy_available and not _HAS_HIGHSPY:
        pytest.skip("needs highspy")
    monkeypatch.setattr(lpmod, "_HAS_HIGHSPY", highspy_available)
    A = np.array([[1.0, 1.0], [1e30, 1e-10], [-1.0, 0.0]])
    b = np.array([1.5, 1e30, 0.0])
    lb, ub = np.zeros(2), np.ones(2)
    assert not confirm_infeasible(A, b, lb, ub)
    assert check_feasibility(A=A, b=b, lb=lb, ub=ub) is True
    _, hi, status, _ = solve_lp(np.array([1.0, 0.0]), A, b, lb=lb, ub=ub,
                                minimize=False)
    assert status == "optimal" and hi >= 1.0


@pytest.mark.parametrize("path", ["solve_lp", "solve_lp_batch"])
@pytest.mark.parametrize("solver", ["default", "highs-ds"])
def test_lp_optima_never_lie_inside_the_true_range(path, solver):
    """On wide-range rows HiGHS put min x_2 1.5e-6 ABOVE the known point and
    max x_2 below it (min > max). With a finite box every optimum must bound
    the known point: 'default' is highspy's path, 'highs-ds' scipy's."""
    n = _WIDE_X0.size
    lb, ub = _WIDE_X0 - 5.0, _WIDE_X0 + 5.0
    for i in range(n):
        f = np.eye(n)[i]
        if path == "solve_lp":
            _, lo, st_lo, _ = solve_lp(f, _WIDE_A, _WIDE_B, lb=lb, ub=ub,
                                       lp_solver=solver, minimize=True)
            _, hi, st_hi, _ = solve_lp(f, _WIDE_A, _WIDE_B, lb=lb, ub=ub,
                                       lp_solver=solver, minimize=False)
            assert st_lo == "optimal" and st_hi == "optimal"
        else:
            lo, hi = solve_lp_batch([f, f], _WIDE_A, _WIDE_B, lb, ub,
                                    [True, False], lp_solver=solver)
        assert lo <= _WIDE_X0[i] <= hi, (i, lo, hi)


def test_exact_reach_on_a_wide_range_star_is_not_a_false_unsat():
    """y = relu(x_2 - t) with t just above x0_2, so f(x0) = 0 and the
    unsafe region y >= 0 is reached. The inward LP minimum put t below
    every x_2, the neuron was classified always active, and exact reach
    returned only y = x_2 - t < 0: UNSAT."""
    n = _WIDE_X0.size
    S = _identity_star(_WIDE_A, _WIDE_B, (_WIDE_X0 - 5.0).reshape(-1, 1),
                       (_WIDE_X0 + 5.0).reshape(-1, 1))
    t = _WIDE_X0[2] + 7.5e-7
    lin = nn.Linear(n, 1).double()
    with torch.no_grad():
        lin.weight.zero_()
        lin.weight[0, 2] = 1.0
        lin.bias.fill_(-t)
    net = nn.Sequential(lin, nn.ReLU()).double()
    unsafe = HalfSpace(np.array([[-1.0]]), np.array([[0.0]]))
    for method in ("exact", "approx"):
        R = reach_pytorch_model(net, S, method=method)
        assert verify_specification(R, unsafe).verdict != "UNSAT", method


@pytest.mark.parametrize("solver", ["default", "highs-ds"])
def test_outward_optima_stay_tight_on_a_well_conditioned_lp(solver):
    """min x0 + x1 s.t. x0 + 2 x1 >= 1 over [0, 1]^2 is 0.5; max x0 + x1
    s.t. x0 + 2 x1 <= 2 is 1.5. The rigorous value may move outward only by
    rounding."""
    f, lb, ub = np.ones(2), np.zeros(2), np.ones(2)
    _, lo, _, _ = solve_lp(f, np.array([[-1.0, -2.0]]), np.array([-1.0]),
                           lb=lb, ub=ub, lp_solver=solver, minimize=True)
    _, hi, _, _ = solve_lp(f, np.array([[1.0, 2.0]]), np.array([2.0]),
                           lb=lb, ub=ub, lp_solver=solver, minimize=False)
    assert 0.5 - 1e-12 <= lo <= 0.5
    assert 1.5 <= hi <= 1.5 + 1e-12

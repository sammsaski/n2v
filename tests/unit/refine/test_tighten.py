"""Unit tests for LP-over-P bound tightening."""

import numpy as np
import pytest
import torch

from n2v.sets.star import Star
from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
from n2v.refine.tighten import contract_input_boxes, neuron_bounds
from tests.unit.refine.conftest import seq_net as _seq


def _star_with_constraints(seed):
    """A star carrying non-trivial C/d (triangle rows from relaxed ReLUs), with
    the net + input box so we can sample exact outputs that lie in [[S]]."""
    rng = np.random.default_rng(seed)
    net = _seq(3, 6, 6, 4, seed=seed)
    layers = extract_layers(net)
    lb = rng.uniform(-1, -0.2, size=3)
    ub = rng.uniform(0.2, 1, size=3)
    S_out, _ = relaxed_reach(Star.from_bounds(lb, ub), layers)
    assert S_out.C.size > 0  # has triangle constraints
    return S_out, net, lb, ub


def _exact_outputs(net, lb, ub, n, seed=0):
    """Exact model outputs over the input box -- provably a subset of [[S_out]]."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(lb, ub, size=(n, len(lb)))
    with torch.no_grad():
        return net(torch.as_tensor(X, dtype=torch.float64)).numpy()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lp_cpu_matches_get_range(seed):
    """lp_cpu LPs every dim and matches Star.get_range."""
    S, _, _, _ = _star_with_constraints(seed)
    lb, ub = neuron_bounds(S, "lp_cpu")
    for i in range(S.dim):
        gmin, gmax = S.get_range(i)
        assert lb[i, 0] == pytest.approx(gmin, abs=1e-6)
        assert ub[i, 0] == pytest.approx(gmax, abs=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_lp_bounds_tighter_than_box_and_sound(seed):
    S, net, lb, ub = _star_with_constraints(seed)
    lb_box, ub_box = S.estimate_ranges()
    lb_lp, ub_lp = neuron_bounds(S, "lp_cpu")
    # tighter-or-equal to the box interval
    assert np.all(lb_lp >= lb_box - 1e-7)
    assert np.all(ub_lp <= ub_box + 1e-7)
    # and strictly tighter somewhere (C/d actually constrains the range)
    assert np.any(lb_lp > lb_box + 1e-6) or np.any(ub_lp < ub_box - 1e-6)
    # sound: every exact output (a subset of [[S]]) is within the LP bounds
    Y = _exact_outputs(net, lb, ub, 400, seed)
    assert np.all(Y >= lb_lp.flatten() - 1e-6)
    assert np.all(Y <= ub_lp.flatten() + 1e-6)


def test_nvar_zero_returns_center():
    V = np.array([[1.5], [-2.0]])  # dim 2, nVar 0
    S = Star(V, np.zeros((0, 0)), np.zeros((0, 1)))
    lb, ub = neuron_bounds(S, "lp_cpu")
    assert lb.shape == (2, 1) and ub.shape == (2, 1)
    assert np.allclose(lb, V) and np.allclose(ub, V)


def test_lp_cpu_ranges_are_padded_outward():
    from n2v.refine.tighten import _lp_ranges
    rng = np.random.default_rng(2)
    V = np.hstack([rng.standard_normal((4, 1)), rng.standard_normal((4, 3))])
    S = Star(V, np.zeros((0, 3)), np.zeros((0, 1)),
             -np.ones((3, 1)), np.ones((3, 1)))
    # exact optimum over the [-1, 1]^3 box (no polytope rows): c -/+ sum|V|
    lb_x = V[:, 0] - np.abs(V[:, 1:]).sum(axis=1)
    ub_x = V[:, 0] + np.abs(V[:, 1:]).sum(axis=1)
    pad_lo = 1e-7 * (1.0 + np.abs(lb_x))
    pad_hi = 1e-7 * (1.0 + np.abs(ub_x))
    lo, hi = _lp_ranges(S, "lp_cpu", np.arange(4))
    assert np.all(lo <= lb_x - 0.5 * pad_lo)
    assert np.all(hi >= ub_x + 0.5 * pad_hi)


def test_contract_input_boxes_rejects_inf_boxes():
    """Certified contraction is undefined on infinite predicate intervals
    (0 * inf = NaN); it must fail loudly instead of returning poisoned boxes."""
    # alpha_0 in [-1, 1] carried only by C/d rows (box +/-inf on that dim)
    V = np.hstack([np.zeros((2, 1)), np.eye(2)])
    C = np.array([[1.0, 0.0], [-1.0, 0.0]])
    d = np.array([[1.0], [1.0]])
    S = Star(V, C, d,
             np.array([[-np.inf], [-1.0]]), np.array([[np.inf], [1.0]]))
    with pytest.raises(ValueError, match="finite predicate boxes"):
        contract_input_boxes(
            [(S.C, S.d)], [(S.predicate_lb, S.predicate_ub)], n_dims=2,
            device="cpu")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dual_gpu_bounds_enclose_lp_cpu(seed, engine_device):
    """NS-certified GPU bounds must be OUTWARD of the exact CPU LP bounds.

    The lp_cpu bound is the certified HiGHS optimum padded outward by
    ``_lp_pad`` (1e-7 (1 + |v|)), so the exact optimum lies within one pad
    of it and a correct dual_gpu bound within two pads; the slack is two
    pads (it was a flat 1e-5, which let a dual_gpu bound moved 5e-6 inward
    pass). The independent reference is
    ``tests/soundness/test_tighten_sound.py::test_neuron_bounds_hold_the_exact_range``."""
    from n2v.utils.lpsolver import _lp_pad

    S, _, _, _ = _star_with_constraints(seed)
    lb_cpu, ub_cpu = neuron_bounds(S, "lp_cpu")
    lb_gpu, ub_gpu = neuron_bounds(S, "dual_gpu")
    # outward enclosure: lb_gpu <= lb_cpu, ub_gpu >= ub_cpu (within the pads)
    assert np.all(lb_gpu <= lb_cpu + 2.0 * _lp_pad(lb_cpu))
    assert np.all(ub_gpu >= ub_cpu - 2.0 * _lp_pad(ub_cpu))


# --------------------------------------------------------------------------- #
# The dual cache, refusal, and the stars                                       #
# that need no solve                                                           #
# --------------------------------------------------------------------------- #
def _stable_first_star():
    """Three predicate variables with binding rows; dimension 0 is
    box-stable (positive) and dimension 2 box-stable (negative), so
    dual_gpu solves dimensions 1, 3 and 4, not 0..2."""
    plb = np.array([[-3.0], [-0.5], [0.0]])
    pub = np.array([[2.0], [4.0], [1.5]])
    C = np.array([[1.0, 1.0, 0.0], [-1.0, 0.0, 2.0], [0.0, -1.0, 1.0],
                  [1.0, 0.0, -1.0]])
    d = np.array([[1.5], [2.25], [1.0], [0.5]])
    V = np.array([[5.0, 0.5, 0.25, 0.0],
                  [-0.75, 1.0, 1.0, 0.0],
                  [-10.0, 0.0, 0.0, 0.25],
                  [1.25, -2.0, 0.5, 3.0],
                  [0.5, 0.125, -1.0, 2.0]])
    return Star(V, C, d, plb, pub)


def test_safe_f16_keeps_every_cached_dual_finite_and_non_negative():
    """``_safe_f16`` stores a dual vector in float16 with every value finite
    and non-negative: values above the float16 maximum and +inf become
    65504, negative values, -inf and NaN become 0, values in range are kept.
    An infinite cached multiplier makes the NS certificate NaN, which the
    ReLU classifier would read as stably active. Catches a dropped upper or
    lower clamp."""
    from n2v.refine.tighten import _safe_f16

    y = np.array([1e6, np.inf, 70000.0, 65504.0, 2.5, 0.0, -3.0, -np.inf,
                  np.nan])
    out = _safe_f16(y)
    assert out.dtype == np.float16
    assert out.tolist() == [65504.0, 65504.0, 65504.0, 65504.0, 2.5, 0.0,
                            0.0, 0.0, 0.0]


@pytest.mark.parametrize("warm", ["cold", "huge"])
def test_collected_duals_are_keyed_by_the_solved_dimensions(warm,
                                                            engine_device):
    """``neuron_bounds(S, "dual_gpu", collect_duals=True)`` returns a dual
    cache keyed by the dimensions it solved (the box-unstable ones, here 1, 3
    and 4: dimension 0 is box-stable), each a pair of float16 vectors of the
    star's row count with finite non-negative values, also when warm-started
    with duals of 1e6 (above the float16 range). ``_lp_ranges`` on the
    dimensions [4, 1], in that order, returns bounds aligned to that order
    that hold the exact range (4: [-3.53125, 3.125], 1: [-3.5, 0.75], from
    the vertices of the predicate polytope) and caches under 4 and 1.
    Catches a cache keyed by the position inside the solved dimensions and
    a dual cached as inf."""
    from n2v.refine.tighten import _lp_ranges

    S = _stable_first_star()
    m = S.C.shape[0]
    wd = None
    if warm == "huge":
        wd = {i: (np.full(m, 1e6), np.full(m, 1e6)) for i in range(S.dim)}
    lb, ub, duals = neuron_bounds(S, "dual_gpu", warm_duals=wd,
                                  collect_duals=True)
    assert sorted(duals) == [1, 3, 4]
    for i, pair in duals.items():
        assert len(pair) == 2
        for y in pair:
            assert y.dtype == np.float16 and y.shape == (m,)
            y = y.astype(np.float64)
            assert np.all(np.isfinite(y)) and np.all(y >= 0), (i, y)
    lo, hi, duals = _lp_ranges(S, "dual_gpu", np.array([4, 1]),
                               warm_duals=wd, collect_duals=True)
    assert sorted(duals) == [1, 4]
    assert lo[0] <= -3.53125 and hi[0] >= 3.125
    assert lo[1] <= -3.5 and hi[1] >= 0.75


def test_an_unknown_backend_is_refused():
    """A backend other than "lp_cpu" and "dual_gpu" raises ``ValueError``,
    directly and through ``neuron_bounds``, instead of answering with some
    other backend's bounds. Catches an else branch that answers."""
    from n2v.refine.tighten import _lp_ranges

    S = _stable_first_star()
    for backend in ("lp", "LP_CPU", "dual-gpu", ""):
        with pytest.raises(ValueError, match="unknown backend"):
            _lp_ranges(S, backend, np.arange(S.dim))
        with pytest.raises(ValueError, match="unknown backend"):
            neuron_bounds(S, backend)


@pytest.mark.parametrize("backend", ["lp_cpu", "dual_gpu"])
def test_stars_that_need_no_solve_return_their_exact_bounds(backend,
                                                            engine_device):
    """A star on no predicate variable has its centre as both bounds, with
    and without the dual cache (which is then empty), on both backends. A
    star whose every dimension is box-stable gets bounds that hold its exact
    range ([3.75, 5.75] and [-10, -9.625], from the vertices of its
    predicate polytope) and an empty dual cache: dual_gpu keeps the box
    interval with no solve, lp_cpu solves and caches nothing. Catches a
    missing or misplaced element of the returned tuple on these paths."""
    V = np.array([[1.5], [-2.0]])
    P = Star(V, np.zeros((0, 0)), np.zeros((0, 1)))
    lb, ub = neuron_bounds(P, backend)
    assert lb.tolist() == [[1.5], [-2.0]] and ub.tolist() == [[1.5], [-2.0]]
    lb, ub, duals = neuron_bounds(P, backend, collect_duals=True)
    assert lb.tolist() == [[1.5], [-2.0]] and ub.tolist() == [[1.5], [-2.0]]
    assert duals == {}
    S = _stable_first_star()
    T = Star(S.V[[0, 2]], S.C, S.d, S.predicate_lb, S.predicate_ub)
    lb, ub, duals = neuron_bounds(T, backend, collect_duals=True)
    assert duals == {}
    assert lb[0, 0] <= 3.75 and ub[0, 0] >= 5.75
    assert lb[1, 0] <= -10.0 and ub[1, 0] >= -9.625


def test_bound_functions_hand_on_their_options(monkeypatch):
    """``neuron_bounds(..., "dual_gpu", gpu_iters=...)`` runs the bound solve
    with that iteration budget; ``neuron_bounds_population`` and
    ``contract_input_boxes`` hand ``gpu_iters`` and ``device`` to the
    population solve. Catches an option that is dropped (the solve then
    runs at its default budget, or on CUDA when the caller asked for the
    CPU)."""
    import n2v.refine.tighten as tighten
    from n2v.refine.tighten import contract_input_boxes, neuron_bounds_population

    calls = {"batch": [], "population": []}
    real_batch = tighten.solve_lp_batch_gpu
    real_pop = tighten.solve_lp_population_gpu

    def _batch(*args, **kwargs):
        calls["batch"].append(kwargs)
        return real_batch(*args, **{**kwargs, "device": "cpu"})

    def _pop(*args, **kwargs):
        calls["population"].append(kwargs)
        return real_pop(*args, **{**kwargs, "device": "cpu"})

    monkeypatch.setattr(tighten, "solve_lp_batch_gpu", _batch)
    monkeypatch.setattr(tighten, "solve_lp_population_gpu", _pop)
    S = _stable_first_star()
    neuron_bounds(S, "dual_gpu", gpu_iters=7)
    assert [k.get("max_iters") for k in calls["batch"]] == [7]
    neuron_bounds_population([S], gpu_iters=9, device="cpu")
    contract_input_boxes([(S.C, S.d)], [(S.predicate_lb, S.predicate_ub)], 2,
                         gpu_iters=11, device="cpu")
    assert [(k.get("max_iters"), k.get("device"))
            for k in calls["population"]] == [(9, "cpu"), (11, "cpu")]


def test_a_bound_lp_without_an_optimum_falls_back_to_the_box(monkeypatch):
    """When the bound LP returns no optimum for a dimension (an infeasible
    or failed solve), ``_lp_ranges`` falls back to the box interval of that
    dimension, which still holds every member: the bounds of the star below
    must hold its exact range (dimension 0: [3.75, 5.75], 1: [-3.5, 0.75],
    3: [0, 7.625], 4: [-3.53125, 3.125], from the vertices of its predicate
    polytope). Catches a fallback that swaps or drops an end."""
    S = _stable_first_star()
    monkeypatch.setattr(Star, "_solve_lp_batch",
                        lambda self, objectives, *a, **k: [None] * len(objectives))
    lb, ub = neuron_bounds(S, "lp_cpu")
    exact = {0: (3.75, 5.75), 1: (-3.5, 0.75), 2: (-10.0, -9.625),
             3: (0.0, 7.625), 4: (-3.53125, 3.125)}
    for i, (lo, hi) in exact.items():
        assert lb[i, 0] <= lo and ub[i, 0] >= hi, (i, lb[i, 0], ub[i, 0])


def test_a_population_with_nothing_to_solve_returns_every_star(engine_device):
    """``neuron_bounds_population`` on stars none of which needs a solve (a
    star on no predicate variable and a star whose every dimension is
    box-stable) returns one result per star, in order, without a population
    solve: the centre for the first, and for the second bounds that hold its
    exact range ([3.75, 5.75] and [-10, -9.625], from the vertices of its
    predicate polytope) with an empty dual cache. The lockstep child reach
    calls it so when every neuron of a deeper layer is box-stable for both
    children. Catches an early return that drops the per-star results."""
    from n2v.refine.tighten import neuron_bounds_population

    P = Star(np.array([[1.5], [-2.0]]), np.zeros((0, 0)), np.zeros((0, 1)))
    S = _stable_first_star()
    T = Star(S.V[[0, 2]], S.C, S.d, S.predicate_lb, S.predicate_ub)
    res = neuron_bounds_population([P, T])
    assert len(res) == 2
    (lb0, ub0, d0), (lb1, ub1, d1) = res
    assert lb0.tolist() == [[1.5], [-2.0]] and ub0.tolist() == [[1.5], [-2.0]]
    assert d0 == {} and d1 == {}
    assert lb1[0, 0] <= 3.75 and ub1[0, 0] >= 5.75
    assert lb1[1, 0] <= -10.0 and ub1[1, 0] >= -9.625

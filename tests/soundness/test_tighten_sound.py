"""
Soundness of LP-over-P bound tightening and of the split operations of the
refinement verifier (``n2v/refine/tighten.py``, ``n2v/refine/operations.py``).

Tightening changes which neurons are relaxed/split -- it must NOT change the
verdict (only node count) and must keep the reach a sound over-approximation.
The promise checked: ``neuron_bounds`` (both backends, any warm start),
``neuron_bounds_population`` and ``contract_input_boxes`` return bounds that
hold the exact range of the star or system (in rationals, zero slack, also
where the centre sum rounds), ``split`` and ``split_wave`` return children
that hold every true output of their phase region, and ``verify_refine``
through these operations answers UNSAT only when no counterexample exists.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.reach import reach_pytorch_model
from n2v.sets.star import Star
from n2v.utils.lpsolver import check_feasibility
from n2v.refine import FaithfulSelector, LinearSpec, Status, verify_refine
from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
from tests.known_bugs import known_bug
from tests.unit.refine.conftest import _assert_honest_ce


# The seed and arm grids below are reduced: the cases kept are those whose
# nets caught every injected bug that the whole grid caught, with every arm
# of an arm grid kept.

# The exact CPU configuration, pinned so a CUDA host does not switch these
# tests to the GPU defaults.
CPU_CONFIG = dict(bound_mode="box", prune_mode="exact", witness_mode="exact",
                  frontier_width=1)

# Spec ``y0 <= tau`` placed relative to the exact range [mn, mx] of output 0:
# inside the range (SAT: the image of the box is connected), below it, and
# just below it (both UNSAT).
_TAU_ARMS = {"inside": (0.5, Status.SAT), "below": (-0.1, Status.UNSAT),
             "near": (-0.01, Status.UNSAT)}


def _rand_net(seed, dims=(2, 5, 5, 2)):
    torch.manual_seed(seed)
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers).double()


def _intersects_unsafe(S, spec):
    A = [spec.G @ S.V[:, 1:]]
    b = [spec.g - spec.G @ S.V[:, 0]]
    if S.C.size > 0:
        A.append(S.C)
        b.append(S.d.flatten())
    return check_feasibility(A=np.vstack(A), b=np.concatenate(b),
                             lb=S.predicate_lb.flatten(), ub=S.predicate_ub.flatten())


def _oracle(model, S_in, spec):
    for S in reach_pytorch_model(model, S_in, method="exact"):
        if _intersects_unsafe(S, spec):
            return Status.SAT
    return Status.UNSAT


def _tau_arm_spec(model, S_in, arm):
    """The tau-arm spec on output 0 and its ground truth."""
    out0 = reach_pytorch_model(model, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    frac, truth = _TAU_ARMS[arm]
    spec = LinearSpec(G=np.array([[1.0, 0.0]]),
                      g=np.array([mn + frac * (mx - mn)]))
    return spec, truth


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_lp_reach_is_sound_and_tighter_than_box(seed):
    rng = np.random.default_rng(seed)
    net = _rand_net(seed, dims=(3, 6, 6, 2))
    layers = extract_layers(net)
    lb = rng.uniform(-1, -0.2, size=3)
    ub = rng.uniform(0.2, 1, size=3)
    S_in = Star.from_bounds(lb, ub)

    S_box, _ = relaxed_reach(S_in, layers, bound_mode="box")
    S_lp, _ = relaxed_reach(S_in, layers, bound_mode="lp_cpu")

    # sound: exact output samples inside the LP-tightened reach
    X = rng.uniform(lb, ub, size=(400, 3))
    with torch.no_grad():
        Y = net(torch.as_tensor(X, dtype=torch.float64)).numpy()
    assert np.all(S_lp.contains(Y, method="lp")), "LP-tightened reach is unsound"

    # tighter: the LP reach's output bounding box is within the box reach's
    lo_box, hi_box = S_box.get_ranges()
    lo_lp, hi_lp = S_lp.get_ranges()
    assert np.all(lo_lp >= lo_box - 1e-6)
    assert np.all(hi_lp <= hi_box + 1e-6)


@pytest.mark.parametrize("seed", [3, 4, 7])
def test_verdict_invariant_across_bound_modes(seed):
    rng = np.random.default_rng(100 + seed)
    net = _rand_net(seed)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle(net, S_in, spec)
    # dual_gpu is never dropped: without CUDA the certified engine runs on CPU
    # torch, which is exactly the path a CPU-only CI must exercise.
    for mode in ("box", "lp_cpu", "dual_gpu"):
        res = verify_refine(S_in, net, spec, FaithfulSelector(),
                            bound_mode=mode, prune_mode="exact",
                            witness_mode="exact", frontier_width=1,
                            node_budget=20000)
        assert res.status != Status.UNKNOWN
        assert res.status == oracle, f"bound_mode={mode}: {res.status} != oracle {oracle}"


# The certified GPU engine guards below take the ``engine_device`` fixture
# (tests/conftest.py): CPU torch on every host, CUDA where it exists. They
# used to sit behind a CUDA-only skip, so a CPU-only CI never ran the
# production-default engine's soundness checks.


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_certified_prune_implies_exact_unsat(seed, engine_device):
    """A certified GPU prune must imply the exact disjointness LP also proves
    emptiness (the NS bound is outward, so a GPU prune can never be a false
    prune). The converse may fail (fewer prunes) -- that is allowed."""
    from n2v.refine.witness import certified_prune_wave
    from n2v.sets.halfspace import HalfSpace
    from n2v.utils.verify_specification import verify_specification

    net = _rand_net(seed, dims=(2, 6, 6, 2))
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    root, _ = relaxed_reach(S_in, extract_layers(net), {}, bound_mode="box")
    lo, hi = root.get_ranges()
    # sweep thresholds across (and beyond) the reachable range of output 0 to
    # exercise both prunable and non-prunable specs
    for frac in (-0.2, 0.1, 0.5, 0.9, 1.2):
        tau = (lo[0] + (hi[0] - lo[0]) * frac).item()
        spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))
        if certified_prune_wave([root], spec)[0]:
            exact = verify_specification([root], HalfSpace(spec.G, spec.g)).verdict
            assert exact == "UNSAT", f"false GPU prune at tau={tau}"


@pytest.mark.parametrize("seed", [0, 1, 4])
def test_verdict_invariant_with_gpu_prune(seed, engine_device):
    """(bound_mode=dual_gpu, prune_mode=dual_gpu) must reproduce the oracle
    verdict -- the GPU prune changes node count, never the answer."""
    rng = np.random.default_rng(100 + seed)
    net = _rand_net(seed)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle(net, S_in, spec)
    res = verify_refine(S_in, net, spec, FaithfulSelector(),
                        bound_mode="dual_gpu", prune_mode="dual_gpu",
                        witness_mode="exact", frontier_width=1,
                        node_budget=20000)
    assert res.status != Status.UNKNOWN
    assert res.status == oracle, f"gpu-prune verdict {res.status} != {oracle}"


@pytest.mark.parametrize("seed", [0, 1, 4])
def test_verdict_invariant_with_gpu_witness_and_prune(seed, engine_device):
    """The full GPU node path (dual_gpu bounds + GPU prune + approximate GPU
    witness) must reproduce the oracle verdict. The witness is approximate --
    SAT soundness rests on the exact forward-pass CE check, UNSAT on the
    certified prune / exact pruning of children -- so only node count may
    change, never the answer."""
    rng = np.random.default_rng(100 + seed)
    net = _rand_net(seed)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle(net, S_in, spec)
    res = verify_refine(S_in, net, spec, FaithfulSelector(),
                        bound_mode="dual_gpu", prune_mode="dual_gpu",
                        witness_mode="dual_gpu", frontier_width=1,
                        node_budget=20000)
    assert res.status != Status.UNKNOWN
    assert res.status == oracle, f"gpu-witness verdict {res.status} != {oracle}"


@pytest.mark.parametrize("seed", range(4))
def test_population_bounds_enclose_exact(seed, engine_device):
    """neuron_bounds_population (the frontier/sibling batching primitive) must
    be outward of the exact per-star LP range for every star in the batch,
    including heterogeneous shapes and dual warm-starts.

    The reference is the lp_cpu bound, the certified HiGHS optimum padded
    outward by ``_lp_pad``, so a correct population bound lies within two
    pads of it (a flat slack of 1e-5 would let a bound moved 5e-6 inward
    pass). The exact reference on small stars is
    ``test_population_bounds_hold_the_exact_range``."""
    from n2v.refine.tighten import neuron_bounds, neuron_bounds_population
    from n2v.utils.lpsolver import _lp_pad

    rng = np.random.default_rng(400 + seed)
    stars = []
    for j, dims in enumerate([(2, 6, 6, 2), (2, 8, 8, 2)]):
        net = _rand_net(seed * 7 + j, dims=dims)
        S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        root, _ = relaxed_reach(S_in, extract_layers(net), {}, bound_mode="box")
        stars.append(root)

    # cold and (arbitrary-valued, hence still-sound) warm-started runs
    warm = [{d: (np.abs(rng.normal(size=S.C.shape[0])).astype(np.float32),) * 2
             for d in range(S.dim)} for S in stars]
    for w in (None, warm):
        res = neuron_bounds_population(stars, warm=w)
        for S, (lb_p, ub_p, duals) in zip(stars, res):
            lb_e, ub_e = neuron_bounds(S, "lp_cpu")
            assert np.all(lb_p <= lb_e + 2.0 * _lp_pad(lb_e)), \
                "population lb not outward"
            assert np.all(ub_p >= ub_e - 2.0 * _lp_pad(ub_e)), \
                "population ub not outward"


@pytest.mark.parametrize("seed, arm", [(0, "below"), (0, "near"),
                                       (1, "inside"), (4, "inside")],
                         ids=["0-below", "0-near", "1-inside", "4-inside"])
def test_verdict_invariant_full_gpu_wave(seed, arm, engine_device):
    """The complete scaled node path -- dual_gpu bounds + sibling-pair lockstep
    reach + wave prune (frontier_width > 1) + GPU witness + dual inheritance --
    must reproduce the oracle verdict, on SAT and UNSAT ground truths, with
    honest counterexamples."""
    net = _rand_net(seed)
    lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    S_in = Star.from_bounds(lb, ub)
    spec, truth = _tau_arm_spec(net, S_in, arm)
    assert _oracle(net, S_in, spec) == truth, "test setup: oracle disagrees"
    res = verify_refine(S_in, net, spec, FaithfulSelector(),
                        bound_mode="dual_gpu", prune_mode="dual_gpu",
                        witness_mode="dual_gpu", frontier_width=4,
                        node_budget=20000)
    assert res.status == truth, f"wave verdict {arm}: {res.status} != {truth}"
    _assert_honest_ce(res, net, spec, lb, ub)


@pytest.mark.parametrize("seed, arm", [(0, "near"), (4, "inside"),
                                       (5, "below"), (5, "inside")],
                         ids=["0-near", "4-inside", "5-below", "5-inside"])
def test_default_configuration_matches_oracle(seed, arm, monkeypatch, engine_device):
    """With CUDA, the defaults are the certified GPU configuration (dual_gpu
    bounds, prune and witnesses over waves of 32 nodes): every arm must be
    decided (never UNKNOWN) and match the ground truth, with honest
    counterexamples. The certified GPU prune must actually run (the CPU
    configuration gives the same verdicts)."""
    import n2v.refine.bab as bab

    prunes = []
    real_prune = bab.certified_prune_wave

    def counting_prune(*args, **kwargs):
        prunes.append(1)
        return real_prune(*args, **kwargs)

    monkeypatch.setattr(bab, "certified_prune_wave", counting_prune)
    if engine_device == "cpu":
        # The defaults are chosen from ``bab.gpu_available`` (its own import
        # of the name); the fixture only redirects the device resolution to
        # CPU torch. Present a CUDA host here so the GPU configuration is
        # selected and then runs on CPU torch.
        monkeypatch.setattr(bab, "gpu_available", lambda: True)
    net = _rand_net(seed)
    lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    S_in = Star.from_bounds(lb, ub)
    spec, truth = _tau_arm_spec(net, S_in, arm)
    res = verify_refine(S_in, net, spec, node_budget=20000)
    assert prunes, "the defaults did not run the certified GPU prune"
    assert res.status == truth, \
        f"default configuration {arm}: {res.status} != {truth}"
    _assert_honest_ce(res, net, spec, lb, ub)


@pytest.mark.parametrize("seed", [2, 3])
def test_layer_order_selector_verdict(seed):
    """LayerOrderSelector (the enumeration-regime branch order) is just another
    selector: exhaustive splitting stays exhaustive, so verdicts match the
    oracle -- only the tree shape changes."""
    from n2v.refine import LayerOrderSelector

    rng = np.random.default_rng(500 + seed)
    net = _rand_net(seed)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle(net, S_in, spec)
    res = verify_refine(S_in, net, spec, LayerOrderSelector(),
                        node_budget=20000, **CPU_CONFIG)
    assert res.status != Status.UNKNOWN
    assert res.status == oracle, f"layer-order verdict {res.status} != {oracle}"


@pytest.mark.parametrize("seed", range(5))
def test_contract_input_boxes_outward_of_exact(seed, engine_device):
    """Certified domain contraction must stay OUTWARD of the exact input-dim
    range over {C alpha <= d, box} (never cut a feasible point), and must
    declare ``empty`` only for a truly infeasible system."""
    from n2v.utils.lp_solver_enum import LPSolver
    from n2v.refine.tighten import contract_input_boxes

    net = _rand_net(600 + seed, dims=(3, 6, 6, 2))
    S_in = Star.from_bounds(np.array([-1.0, -0.5, -1.0]),
                            np.array([0.5, 1.0, 1.0]))
    root, _ = relaxed_reach(S_in, extract_layers(net), {}, bound_mode="box")
    assert root.C.size > 0
    k0 = S_in.nVar

    (clb, cub, empty), = contract_input_boxes(
        [(root.C, root.d)], [(root.predicate_lb, root.predicate_ub)], k0)

    # exact range of each input dim over P (HiGHS on the alpha-space objectives)
    objectives, flags = [], []
    for i in range(k0):
        e = np.zeros(root.nVar)
        e[i] = 1.0
        objectives.extend([e, e])
        flags.extend([True, False])
    exact = root._solve_lp_batch(objectives, flags, LPSolver.DEFAULT)
    feasible = all(v is not None for v in exact)
    if feasible:
        assert not empty, "certified-empty on a feasible system"
        for i in range(k0):
            assert clb[i, 0] <= exact[2 * i] + 1e-7, "contraction cut the min"
            assert cub[i, 0] >= exact[2 * i + 1] - 1e-7, "contraction cut the max"
    # contraction never widens the box
    assert np.all(clb >= root.predicate_lb[:k0] - 1e-12)
    assert np.all(cub <= root.predicate_ub[:k0] + 1e-12)


@pytest.mark.parametrize("seed", [2, 3])
def test_witness_wave_matches_scalar_contract(seed, engine_device):
    """The batched wave witness must honour the scalar witness's contract per
    star: alpha box-feasible by construction, and t equal to the EXACT max
    spec-row margin evaluated at alpha."""
    from n2v.refine.witness import violation_witness_wave

    stars = []
    for j in range(3):
        net = _rand_net(1000 + seed * 3 + j, dims=(2, 6, 6, 2))
        S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        root, _ = relaxed_reach(S_in, extract_layers(net), {}, bound_mode="box")
        stars.append(root)
    spec = LinearSpec(G=np.array([[1.0, -1.0]]), g=np.array([0.0]))

    for S, res in zip(stars, violation_witness_wave(stars, spec)):
        assert res is not None
        alpha, t = res
        assert np.all(alpha >= S.predicate_lb.flatten() - 1e-6)
        assert np.all(alpha <= S.predicate_ub.flatten() + 1e-6)
        y = S.V[:, 0] + S.V[:, 1:] @ alpha
        assert abs(t - float(np.max(spec.margins(y)))) < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 3])
def test_verdict_invariant_contract_split_config(seed, engine_device):
    """The contract-split configuration (layer-ordered branching + certified
    input contraction at every split + group reach with certified dual bounds
    at every layer + certified GPU prune) must reproduce the oracle verdict,
    with either witness mode: contraction is NS-outward (never cuts a feasible
    input), dropped children are certified-empty, and SAT still rests on the
    exact forward-pass CE check. The bound_mode="box" arm covers the
    per-child fallback reach path in split_wave (non-dual bound modes)."""
    from n2v.refine import LayerOrderSelector

    rng = np.random.default_rng(100 + seed)
    net = _rand_net(seed)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mn = min(float(S.get_range(0)[0]) for S in out0)
    mx = max(float(S.get_range(0)[1]) for S in out0)
    tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle(net, S_in, spec)
    for wmode, bmode in (("dual_gpu", "dual_gpu"), ("exact", "dual_gpu"),
                         ("dual_gpu", "box")):
        res = verify_refine(S_in, net, spec, LayerOrderSelector(),
                            bound_mode=bmode, contract_inputs=True,
                            prune_mode="dual_gpu", witness_mode=wmode,
                            frontier_width=4, node_budget=20000)
        assert res.status != Status.UNKNOWN
        assert res.status == oracle, \
            f"contract-split verdict ({wmode}, {bmode}) {res.status} != {oracle}"


# --------------------------------------------------------------------------- #
# Neuron bounds, population bounds and input-box contraction against the exact #
# range in rationals                                                            #
# --------------------------------------------------------------------------- #
# The oracle: a linear function over a polytope takes its extreme values at a
# vertex, so the exact range of every dimension of a star with a handful of
# predicate variables is the minimum and maximum over the images of the
# vertices of its predicate polytope, enumerated in fractions
# (``tests/_oracles/rational.py``). Bounds are compared one-sided with zero
# slack: an LP bound and an NS bound are outward by construction.
_Q_RANGES = {}


def _q_range(S):
    """The exact range of every dimension of ``S`` as two lists of
    Fractions, from the vertices of its predicate polytope (no n2v call).
    Cached on the bytes of the star's arrays, so a star built twice by the
    same builder is enumerated once."""
    from tests._oracles import rational as Q

    arrays = [S.V, S.C, S.d, S.predicate_lb, S.predicate_ub]
    key = tuple(None if a is None else np.asarray(a, dtype=np.float64).tobytes()
                + str(np.shape(a)).encode() for a in arrays)
    if key not in _Q_RANGES:
        verts = Q.set_vertices(Q.star(S))
        if not verts:
            pytest.fail("test setup: the star is empty")
        dim = len(verts[0])
        _Q_RANGES[key] = ([min(v[i] for v in verts) for i in range(dim)],
                          [max(v[i] for v in verts) for i in range(dim)])
    return _Q_RANGES[key]


def _assert_bounds_hold(lb, ub, S, what):
    """``lb <= exact minimum`` and ``ub >= exact maximum`` on every
    dimension of ``S``, in rationals with zero slack."""
    from fractions import Fraction

    lo, hi = _q_range(S)
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    assert lb.size == len(lo) and ub.size == len(hi), \
        f"{what}: {lb.size} bounds for a star of dimension {len(lo)}"
    bad = [(i, float(lb[i]), float(lo[i]), float(ub[i]), float(hi[i]))
           for i in range(len(lo))
           if not (np.isfinite(lb[i]) and np.isfinite(ub[i]))
           or Fraction(float(lb[i])) > lo[i] or Fraction(float(ub[i])) < hi[i]]
    assert not bad, (f"{what}: bounds inside the exact range (dim, lb, exact "
                     f"min, ub, exact max): {bad}")


def _rows_star():
    """Three predicate variables over a box wider than 2 with bounds of both
    signs, four binding rows that are not box rows (one with a variable
    beyond the leading two, so dropping it changes the range of a leading
    variable), a box-stable dimension first and a centre per dimension that
    differs from its neighbour's: dual_gpu solves dimensions 1, 3 and 4."""
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


def _no_rows_star():
    """A box-only star (no constraint row): the NS bound has no dual term."""
    S = _rows_star()
    return Star(S.V, np.zeros((0, 3)), np.zeros((0, 1)),
                S.predicate_lb, S.predicate_ub)


def _boxless_star():
    """The rows star with its box written as rows and no predicate box: the
    lp_cpu path without a box and the dual_gpu fallback to it (bounded)."""
    S = _rows_star()
    eye = np.eye(3)
    C = np.vstack([S.C, eye, -eye])
    d = np.vstack([S.d, S.predicate_ub, -S.predicate_lb])
    return Star(S.V, C, d)


def _stable_star():
    """Every dimension box-stable (two positive, one negative): dual_gpu
    solves nothing and returns the box interval."""
    S = _rows_star()
    V = np.array([[6.0, 0.5, 0.25, 0.0], [-10.0, 0.0, 0.0, 0.25],
                  [3.5, 0.25, -0.125, 0.5]])
    return Star(V, S.C, S.d, S.predicate_lb, S.predicate_ub)


def _point_star():
    """A star on no predicate variable: its bounds are its centre."""
    return Star(np.array([[1.5], [-2.0]]), np.zeros((0, 0)), np.zeros((0, 1)))


def _relaxed_star():
    """The output star of a relaxed reach of a dyadic 2-3-2 net (two relaxed
    neurons, so four predicate variables and triangle rows), as the
    refinement search builds them."""
    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0, 0.5], [-0.75, 1.0],
                                          [0.25, 0.25]]))
        net[0].bias.copy_(torch.tensor([-0.25, 0.5, 3.0]))
        net[2].weight.copy_(torch.tensor([[1.0, -2.0, 0.5], [0.5, 1.0, -1.0]]))
        net[2].bias.copy_(torch.tensor([0.125, -1.5]))
    # the input box [-2, 1.5] x [-0.75, 3] over a predicate box [-1, 1]^2,
    # without the duplicate box rows ``from_bounds`` adds (fewer vertices)
    S_in = Star(np.array([[-0.25, 1.75, 0.0], [1.125, 0.0, 1.875]]),
                np.zeros((0, 2)), np.zeros((0, 1)),
                -np.ones((2, 1)), np.ones((2, 1)))
    S, meta = relaxed_reach(S_in, extract_layers(net), bound_mode="box")
    if len(meta) != 2:
        pytest.fail(f"test setup: {len(meta)} relaxed neurons, expected 2")
    return S


def _arbitrary_warm(S):
    """Warm-start duals that no solver would produce: huge, NaN, infinite,
    negative and longer than the star has rows. A warm start may change the
    time of the solve, never the validity of the bound."""
    m = S.C.shape[0]
    junk = np.array([1e6, np.nan, np.inf, -5.0, 3.0, 0.0] * (m // 6 + 2))
    return {i: (junk[:m + 2].copy(), junk[1:m + 3].copy())
            for i in range(S.dim)}


_BOUND_STARS = {"rows": _rows_star, "no-rows": _no_rows_star,
                "no-box": _boxless_star, "relaxed": _relaxed_star}


@pytest.mark.parametrize("kind", list(_BOUND_STARS))
@pytest.mark.parametrize("backend, warm", [("lp_cpu", "cold"),
                                           ("dual_gpu", "cold"),
                                           ("dual_gpu", "warm")])
def test_neuron_bounds_hold_the_exact_range(backend, warm, kind, engine_device):
    """``neuron_bounds`` returns, on every dimension, a lower bound at or
    below the exact minimum and an upper bound at or above the exact maximum
    over the star, with zero slack, on both backends, cold and with arbitrary
    warm-start duals, with and without the dual cache. Catches a bound moved
    inward (an lp_cpu pad of the wrong sign, a dual_gpu bound shrunk after
    the certificate), the centre read at the position inside the solved
    dimensions instead of the dimension (dimension 0 of the rows star is
    box-stable, so dual_gpu solves 1, 3 and 4), a bound intersected with the
    wrong end, a warm start that breaks the certificate, and a dual_gpu
    fallback for a star with no predicate box (bounded by its rows) that
    solves other dimensions than those asked for."""
    from n2v.refine.tighten import neuron_bounds

    S = _BOUND_STARS[kind]()
    wd = _arbitrary_warm(S) if warm == "warm" else None
    lb, ub = neuron_bounds(S, backend, warm_duals=wd)
    _assert_bounds_hold(lb, ub, S, f"neuron_bounds({backend}, {kind}, {warm})")
    lb2, ub2, duals = neuron_bounds(S, backend, warm_duals=wd,
                                    collect_duals=True)
    _assert_bounds_hold(lb2, ub2, S,
                        f"neuron_bounds({backend}, {kind}, {warm}, duals)")


@pytest.mark.parametrize("warm", ["cold", "warm"])
def test_population_bounds_hold_the_exact_range(warm, engine_device):
    """``neuron_bounds_population`` returns, for every star of a mixed
    population, bounds that hold the exact range with zero slack: a star on
    no predicate variable, a star whose dimensions are all box-stable and a
    star with no predicate box come before and between the stars that the
    population call solves (with different row, variable and unstable
    dimension counts and box-stable dimensions before unstable ones), so a
    result stored at the position in the solved subset, a flag layout that
    does not match the objective layout, a centre read at the wrong index
    and a bound intersected with the wrong end all show; warm starts are
    arbitrary values, longer than the rows and with NaN and inf. The dual
    cache of a solved star is keyed by its box-unstable dimensions and holds
    finite non-negative values."""
    from n2v.refine.tighten import neuron_bounds_population

    stars = [_point_star(), _stable_star(), _rows_star(), _boxless_star(),
             _relaxed_star(), _no_rows_star()]
    warm_arg = None
    if warm == "warm":
        warm_arg = [None if S.nVar == 0 else _arbitrary_warm(S)
                    for S in stars]
        warm_arg[4] = None              # a solved star with no warm entry
    res = neuron_bounds_population(stars, warm=warm_arg)
    assert len(res) == len(stars)
    for j, (S, r) in enumerate(zip(stars, res)):
        assert r is not None, f"star {j}: no result"
        lb, ub, duals = r
        if S.nVar == 0:
            assert np.array_equal(np.asarray(lb).reshape(-1), S.V[:, 0])
            assert np.array_equal(np.asarray(ub).reshape(-1), S.V[:, 0])
            continue
        _assert_bounds_hold(lb, ub, S, f"population star {j} ({warm})")
        if S.predicate_lb is not None:
            blb, bub = S.estimate_ranges()
            unstable = set(np.flatnonzero((blb.reshape(-1) < 0)
                                          & (bub.reshape(-1) > 0)).tolist())
            assert set(duals) == unstable, \
                f"star {j}: dual cache keyed by {sorted(duals)}, the " \
                f"box-unstable dimensions are {sorted(unstable)}"
            for i, (y0, y1) in duals.items():
                for y in (y0, y1):
                    y = np.asarray(y, dtype=np.float64)
                    assert np.all(np.isfinite(y)) and np.all(y >= 0), \
                        f"star {j} dimension {i}: cached dual {y}"


def _q_leading_range(C, d, lb, ub, n_dims):
    """The exact range of the leading ``n_dims`` predicate variables over
    ``{C a <= d, lb <= a <= ub}`` (Fractions), or None when it is empty."""
    from tests._oracles import rational as Q

    n = np.asarray(lb).size
    V = np.hstack([np.zeros((n_dims, 1)), np.eye(n_dims, n)])
    verts = Q.set_vertices(Q.star(Star(V, C, d, np.reshape(lb, (-1, 1)),
                                       np.reshape(ub, (-1, 1)))))
    if not verts:
        return None
    return ([min(v[i] for v in verts) for i in range(n_dims)],
            [max(v[i] for v in verts) for i in range(n_dims)])


def test_contract_input_boxes_hold_every_feasible_point(engine_device):
    """``contract_input_boxes`` returns, for each system of one call, a box of
    the leading dimensions that holds the exact range of those dimensions
    over the system (zero slack) and lies inside the old box, and calls a
    system empty only when it is empty (checked exactly). The systems differ
    in row and variable count (zero padding), one has a row in which a
    non-leading variable loosens a leading one, one has a zero-width leading
    coordinate (a feasible system whose contracted interval has width 0 must
    not be called empty), one has no row and one is infeasible. Catches an
    upper bound read from the minimisation half or a lower bound from the
    maximisation half, a strict emptiness test made non-strict, and rows
    padded into the wrong columns."""
    from fractions import Fraction

    from n2v.refine.tighten import contract_input_boxes

    R = _rows_star()
    systems = [
        # the rows star (three variables, two contracted)
        (R.C, R.d, R.predicate_lb, R.predicate_ub),
        # a zero-width leading coordinate: a0 = 0.5, a0 + a1 <= 0.75
        (np.array([[1.0, 1.0]]), np.array([[0.75]]),
         np.array([[0.5], [-1.0]]), np.array([[0.5], [1.0]])),
        # no constraint row
        (np.zeros((0, 3)), np.zeros((0, 1)), R.predicate_lb, R.predicate_ub),
        # infeasible: a0 + a1 >= 5 over [-1, 1]^2
        (np.array([[-1.0, -1.0]]), np.array([[-5.0]]),
         -np.ones((2, 1)), np.ones((2, 1))),
    ]
    out = contract_input_boxes([(C, d) for C, d, _, _ in systems],
                               [(lb, ub) for _, _, lb, ub in systems], 2)
    assert len(out) == len(systems)
    for j, ((C, d, lb, ub), (clb, cub, empty)) in enumerate(zip(systems, out)):
        exact = _q_leading_range(C, d, lb, ub, 2)
        clb = np.asarray(clb, dtype=np.float64).reshape(-1)
        cub = np.asarray(cub, dtype=np.float64).reshape(-1)
        assert clb.size == 2 and cub.size == 2
        if exact is None:
            continue                    # empty: any box holds its no point
        assert not empty, f"system {j}: called empty, but it is feasible"
        lo, hi = exact
        for i in range(2):
            assert Fraction(float(clb[i])) <= lo[i], \
                f"system {j} dim {i}: contracted lb {clb[i]} > exact {lo[i]}"
            assert Fraction(float(cub[i])) >= hi[i], \
                f"system {j} dim {i}: contracted ub {cub[i]} < exact {hi[i]}"
        old_lb = np.asarray(lb, dtype=np.float64).reshape(-1)[:2]
        old_ub = np.asarray(ub, dtype=np.float64).reshape(-1)[:2]
        assert np.all(clb >= old_lb) and np.all(cub <= old_ub), \
            f"system {j}: the contracted box leaves the old box"
    assert _q_leading_range(*systems[3], 2) is None, "test setup"


_BOXLESS_BOUND_BUG = (
    "on a Star with no predicate box the lp_cpu bound LP (also the dual_gpu "
    "fallback) takes HiGHS's optimum without a certificate (Star._solve_lp_batch, "
    "as Star.get_range does), and HiGHS reports "
    "a finite optimum for an unbounded direction whose slope is below its "
    "tolerance: the upper bound 1.0000002 leaves out members up to 1001, and "
    "a relaxed reach built on it leaves out true outputs")


@pytest.mark.parametrize("backend, box", [
    pytest.param("lp_cpu", False, marks=known_bug(17, _BOXLESS_BOUND_BUG)),
    pytest.param("dual_gpu", False, marks=known_bug(17, _BOXLESS_BOUND_BUG)),
    ("lp_cpu", True),
    ("dual_gpu", True),
])
def test_neuron_bounds_hold_a_direction_with_a_small_slope(backend, box,
                                                          engine_device):
    """x = a1 + 1e-9 a2 with a1 in [-1, 1] and a2 >= 0 (rows only, or a
    predicate box a2 in [0, 1e12]): the member a = (1, 1e12) has x =
    1 + 1e-9 * 1e12, about 1001, and a = (-1, 0) has x = -1; ``neuron_bounds``
    must hold both, compared in rationals. Without a predicate box the
    direction a2 is unbounded and the true upper bound is infinite; the
    bound LP's raw optimum is not a bound there (marked). With a finite
    predicate box the optimum is certified (the passing neighbour, same
    reference). Catches a bound LP whose solver answer is used without a
    certificate."""
    from fractions import Fraction

    from n2v.refine.tighten import neuron_bounds

    V = np.array([[0.0, 1.0, 1e-9]])
    C = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]])
    d = np.array([[1.0], [1.0], [0.0]])
    if box:
        S = Star(V, C, d, np.array([[-1.0], [0.0]]), np.array([[1.0], [1e12]]))
    else:
        S = Star(V, C, d)
    members = [np.array([1.0, 1e12]), np.array([-1.0, 0.0])]
    for a in members:
        if not np.all(C @ a <= d.reshape(-1)):
            pytest.fail("test setup: the point is not a member")
    lb, ub = neuron_bounds(S, backend)
    for a in members:
        x = sum((Fraction(float(g)) * Fraction(float(v))
                 for g, v in zip(V[0, 1:], a)), Fraction(float(V[0, 0])))
        assert Fraction(float(lb[0, 0])) <= x <= Fraction(float(ub[0, 0])), \
            f"{backend}: member value {float(x)} outside [{lb[0, 0]}, {ub[0, 0]}]"


# --------------------------------------------------------------------------- #
# Rounding of the centre sum, and a vacuous row                                #
# --------------------------------------------------------------------------- #
# A centre of 3 * 2^-56 makes the round-to-nearest sums fl(3/4 + c) = 3/4 and
# fl(-3/4 - c) = -3/4 lie inside the exact range, so a bound path that adds the
# centre to an optimum with no outward slack shows a rounding-scale escape.
_C3 = 3 * 2.0 ** -56


def _rounding_star(kind):
    """x0 = c + a0 and x1 = -c - a0 with c = 3 * 2^-56 over a0 in [-1, 1],
    a1 in [-1, 1/4] and the binding row a0 - a1 <= 1/2 (so the maximum of a0
    is 3/4): the exact ranges are [-1 + c, 3/4 + c] and [-3/4 - c, 1 - c].
    ``kind`` "box" has the predicate box (the bound LP is certified);
    "rows-only" writes the box as rows and has no predicate box (the raw
    HiGHS optimum is used, so the outward pad is the only slack)."""
    V = np.array([[_C3, 1.0, 0.0], [-_C3, -1.0, 0.0]])
    C = np.array([[1.0, -1.0]])
    d = np.array([[0.5]])
    lb = np.array([[-1.0], [-1.0]])
    ub = np.array([[1.0], [0.25]])
    if kind == "box":
        return Star(V, C, d, lb, ub)
    eye = np.eye(2)
    return Star(V, np.vstack([C, eye, -eye]), np.vstack([d, ub, -lb]))


@pytest.mark.parametrize("kind, entry", [
    ("box", "lp_cpu"), ("box", "dual_gpu"), ("box", "population"),
    ("rows-only", "lp_cpu"), ("rows-only", "population")])
def test_bounds_hold_the_exact_range_when_the_centre_sum_rounds(kind, entry,
                                                                engine_device):
    """The exact-fraction rounding case of the bound paths: on the star of
    ``_rounding_star`` the sum of the optimum and the centre, rounded to
    nearest, lies inside the exact range (fl(3/4 + 3 * 2^-56) = 3/4), and
    ``neuron_bounds`` (both backends) and ``neuron_bounds_population`` must
    still return bounds that hold the exact range, in rationals with zero
    slack: through the certified optimum with a predicate box, and through
    the raw optimum plus the outward pad without one (the population entry
    sends such a star to the scalar lp_cpu path). Catches an lp_cpu pad that
    is dropped and a centre sum taken as a bound with no outward slack."""
    from n2v.refine.tighten import neuron_bounds, neuron_bounds_population

    if 0.75 + _C3 != 0.75 or -0.75 - _C3 != -0.75:
        pytest.fail("test setup: the centre sums do not round to nearest")
    S = _rounding_star(kind)
    if entry == "population":
        (lb, ub, _), = neuron_bounds_population([S])
    else:
        lb, ub = neuron_bounds(S, entry)
    _assert_bounds_hold(lb, ub, S, f"{entry} on the {kind} star")


_VACUOUS_ROW_BUG = (
    "a constraint row with right-hand side +inf (a vacuous row, which "
    "Star.convex_hull documents as possible) makes the certificate of the "
    "population solve NaN (0 * inf in b'y), so neuron_bounds_population "
    "returns NaN bounds, which tighten.py's own comment says the ReLU "
    "classifier reads as stably active, and contract_input_boxes returns a "
    "NaN box that it does not call empty (registry LP-1)")


def _vacuous_row_star(rhs):
    """The box a in [-1, 1]^2 cut by a0 + a1 <= 1/2 and by
    0.5 a0 - 0.25 a1 <= ``rhs``: with ``rhs`` = +inf the second row is
    vacuous, with ``rhs`` = 4 it holds on the whole box (its left side is
    at most 3/4), so both are the same set. x0 = a0 - a1 and x1 = 1/4 + a0
    are box-unstable, so the population call solves both."""
    V = np.array([[0.0, 1.0, -1.0], [0.25, 1.0, 0.0]])
    C = np.array([[1.0, 1.0], [0.5, -0.25]])
    d = np.array([[0.5], [rhs]])
    return Star(V, C, d, -np.ones((2, 1)), np.ones((2, 1)))


@pytest.mark.parametrize("entry", ["population", "contract"])
@pytest.mark.parametrize("row", [
    pytest.param("vacuous", marks=known_bug(17, _VACUOUS_ROW_BUG)),
    "finite"])
def test_population_solves_hold_a_star_with_a_vacuous_row(row, entry,
                                                          engine_device):
    """``neuron_bounds_population`` and ``contract_input_boxes`` on a star
    one of whose rows has right-hand side +inf (vacuous: it cuts nothing)
    must return finite bounds that hold the exact range of the set, and a
    contracted box that holds the exact range of its input variables and is
    not called empty, compared in rationals with zero slack against the same
    set written with a finite right-hand side (the passing neighbour, the
    same reference). Catches a certificate that turns 0 * inf into NaN on
    the population path, which a ReLU classifier reads as stably active."""
    from fractions import Fraction

    from n2v.refine.tighten import contract_input_boxes, neuron_bounds_population

    S = _vacuous_row_star(np.inf if row == "vacuous" else 4.0)
    ref = _vacuous_row_star(4.0)
    if entry == "population":
        (lb, ub, _), = neuron_bounds_population([S])
        _assert_bounds_hold(lb, ub, ref, f"population ({row} row)")
        return
    (clb, cub, empty), = contract_input_boxes(
        [(S.C, S.d)], [(S.predicate_lb, S.predicate_ub)], 2)
    clb = np.asarray(clb, dtype=np.float64).reshape(-1)
    cub = np.asarray(cub, dtype=np.float64).reshape(-1)
    assert np.all(np.isfinite(clb)) and np.all(np.isfinite(cub)), \
        f"contracted box ({row} row): [{clb}, {cub}]"
    assert not empty, f"contracted box ({row} row) called empty"
    lo, hi = _q_leading_range(ref.C, ref.d, ref.predicate_lb,
                              ref.predicate_ub, 2)
    for i in range(2):
        assert Fraction(float(clb[i])) <= lo[i] and \
            Fraction(float(cub[i])) >= hi[i], \
            f"contracted box ({row} row) dim {i}: [{clb[i]}, {cub[i]}], " \
            f"exact [{float(lo[i])}, {float(hi[i])}]"

# --------------------------------------------------------------------------- #
# split and split_wave: every child holds the true outputs of its phase region  #
# --------------------------------------------------------------------------- #
# A dyadic 2-3-3-2 net over the box x0 in [-2, 1.5], x1 in [-0.75, 3]. Layer 0:
# n0 = x0 - 0.25 and n1 = x0 + x1 + 0.5 cross zero, n2 = 0.5 x1 + 4 is stably
# active. Layer 1: m0 = 2 relu(n0) - 0.5 is the constant -0.5 where n0 <= 0 and
# crosses zero where n0 >= 0, so the two children of a split on n0 have
# different ranges for m0 (a child classified with its sibling's bounds loses
# outputs); m0 has the index of n0, so a fixed phase applied to the same index
# of every layer cuts inputs with n0 > 0 and m0 < 0 (0.25 < x0 < 0.5).
# The outputs y0 = relu(m0) - relu(m1) and y1 = relu(m1) + 0.5 relu(m2) show m0.
_SPLIT_LO = np.array([-2.0, -0.75])
_SPLIT_HI = np.array([1.5, 3.0])
_SPLIT_SPEC = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([0.0]))


def _split_net():
    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 3), nn.ReLU(),
                        nn.Linear(3, 2)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.5]]))
        net[0].bias.copy_(torch.tensor([-0.25, 0.5, 4.0]))
        net[2].weight.copy_(torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, -0.5],
                                          [-1.0, 0.0, 0.25]]))
        net[2].bias.copy_(torch.tensor([-0.5, 1.0, -0.5]))
        net[4].weight.copy_(torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, 0.5]]))
        net[4].bias.zero_()
    return net


def _split_points():
    """160 inputs of the box: the 96 corner and quarter-grid points of
    ``samples`` and 64 seeded uniform points (which fall between the grid
    lines, for example in 0.25 < x0 < 0.5)."""
    from tests.cell_support import samples

    rng = np.random.default_rng(7)
    pts = samples(_SPLIT_LO, _SPLIT_HI, 96)
    pts += list(rng.uniform(_SPLIT_LO, _SPLIT_HI, size=(64, 2)))
    return np.array(pts)


def _true_preact(net, key, X):
    """The true pre-activation of ReLU neuron ``key`` at the inputs ``X``
    (float64 forward pass of the layers before it)."""
    with torch.no_grad():
        z = net[:2 * key.layer + 1](torch.as_tensor(X, dtype=torch.float64))
    return z[:, key.neuron].numpy()


def _in_phases(net, X, fixed):
    """The inputs whose true pre-activations have every fixed phase."""
    from n2v.refine.types import Phase

    keep = np.ones(len(X), dtype=bool)
    for key, phase in fixed.items():
        z = _true_preact(net, key, X)
        keep &= (z >= 0.0) if phase == Phase.ACTIVE else (z <= 0.0)
    return keep


def _witness_at(S, S_in, x):
    """A witness whose input coordinates are those of the input point ``x``
    (the remaining predicate variables 0)."""
    from n2v.refine.types import Witness

    a_in = np.linalg.solve(S_in.V[:, 1:], np.asarray(x) - S_in.V[:, 0])
    alpha = np.zeros(S.nVar)
    alpha[:S_in.nVar] = a_in
    return Witness(alpha, 1.0, 0, "faithful")


def _assert_children_hold_their_phase_regions(net, X, parent_fixed, children,
                                              what):
    """The children of a split of a parent with fixed phases
    ``parent_fixed``: each child fixes the parent's phases and one more
    neuron, the same neuron in every child with a different phase; the true
    output of every input of the parent's region whose split neuron is
    active (inactive) is a member of the ACTIVE (INACTIVE) child by its own
    rows. A child that is missing must have no input in its phase region."""
    from n2v.refine.types import Phase
    from tests.cell_support import assert_covered, replay_torch

    region = _in_phases(net, X, parent_fixed)
    if children is None:
        pytest.fail(f"{what}: no children (None)")
    if not children:
        assert not region.any(), \
            f"{what}: no child, but {int(region.sum())} inputs are in the region"
        return
    keys = set()
    for c in children:
        extra = {k: p for k, p in c.fixed.items() if k not in parent_fixed}
        assert {k: p for k, p in c.fixed.items() if k in parent_fixed} \
            == dict(parent_fixed), f"{what}: a child lost a parent phase"
        assert len(extra) == 1, f"{what}: a child fixes {len(extra)} new neurons"
        keys |= set(extra)
    assert len(keys) == 1, f"{what}: the children split different neurons"
    key = keys.pop()
    phases = [c.fixed[key] for c in children]
    assert len(set(phases)) == len(phases), \
        f"{what}: two children fix {key} to the same phase"
    z = _true_preact(net, key, X)
    for phase in (Phase.ACTIVE, Phase.INACTIVE):
        sel = region & ((z >= 0.0) if phase == Phase.ACTIVE else (z <= 0.0))
        if not sel.any():
            continue
        kids = [c for c in children if c.fixed[key] == phase]
        assert kids, (f"{what}: the {phase.name} child of {key} is missing, "
                      f"but {int(sel.sum())} inputs are in its region")
        assert_covered(kids, replay_torch(net, X[sel], (1, 2)),
                       f"{what}: {phase.name} child of {key}")


@pytest.mark.parametrize("bound_mode, incremental", [
    ("box", True), ("box", False), ("lp_cpu", True), ("dual_gpu", True),
    ("dual_gpu", False)])
def test_split_children_hold_their_phase_regions(bound_mode, incremental,
                                                  engine_device):
    """``split`` of a root star (a split at ReLU layer 0) and of a star that
    already fixes both crossing neurons of layer 0 (a split at layer 1,
    resumed from the checkpoint there): every true output of an input of the
    parent's region lies in the child of its own phase, by membership on the
    child's rows, on each bound mode, on the incremental path (one resume per
    child, or the lockstep pair under dual_gpu) and on the path that
    re-reaches from the input. Catches two children fixed to the same phase,
    a child re-reached with the parent's phases instead of its own, and a
    child that drops a phase its parent fixed."""
    from n2v.refine import LayerOrderSelector, split
    from n2v.refine.types import NeuronKey, Phase

    net = _split_net()
    layers = extract_layers(net)
    S_in = Star.from_bounds(_SPLIT_LO, _SPLIT_HI)
    X = _split_points()
    parents = [({}, np.array([-1.0, 1.0])),
               ({NeuronKey(0, 0): Phase.ACTIVE, NeuronKey(0, 1): Phase.ACTIVE},
                np.array([1.0, 1.0]))]
    for fixed, x_w in parents:
        parent, _ = relaxed_reach(S_in, layers, fixed, bound_mode=bound_mode)
        children = split(parent, S_in, layers, _SPLIT_SPEC,
                         LayerOrderSelector(), _witness_at(parent, S_in, x_w),
                         incremental=incremental)
        _assert_children_hold_their_phase_regions(
            net, X, fixed, children,
            f"split({bound_mode}, incremental={incremental}, fixed={fixed})")


@pytest.mark.parametrize("modes", [("box", "box"), ("lp_cpu", "lp_cpu"),
                                   ("dual_gpu", "dual_gpu"),
                                   ("dual_gpu", "box")],
                         ids=["box", "lp_cpu", "dual_gpu", "mixed"])
def test_split_wave_children_hold_their_phase_regions(modes, engine_device):
    """``split_wave`` on a wave of three jobs: a star with no relaxed neuron
    first (its result stays None), then the root and a star that fixes n0
    INACTIVE (region x0 <= 0.25), each reached with its own bound mode
    (all ``dual_gpu``: the group reach; otherwise one reach per child at
    the parent's mode). Every true output of an input of a job's region lies
    in that job's child of its own phase, by membership on the child's rows,
    and a child that the contraction drops has no input in its phase region.
    Catches a child reached from another child's contracted box, results
    stored at the wrong job, a child classified with its sibling's bounds, a
    fixed phase applied to the same index of every layer, pin rows with the
    wrong sign or on the wrong variable, and a contraction that cuts
    inputs."""
    from n2v.refine import LayerOrderSelector, split_wave
    from n2v.refine.types import NeuronKey, Phase

    net = _split_net()
    layers = extract_layers(net)
    S_in = Star.from_bounds(_SPLIT_LO, _SPLIT_HI)
    X = _split_points()
    plain = Star.from_bounds(_SPLIT_LO, _SPLIT_HI)
    jobs_fixed = [{}, {NeuronKey(0, 0): Phase.INACTIVE}]
    stars = [relaxed_reach(S_in, layers, f, bound_mode=m)[0]
             for f, m in zip(jobs_fixed, modes)]
    witness_x = [np.array([-1.0, 1.0]), np.array([-1.0, -0.5])]
    jobs = [(plain, _witness_at(plain, S_in, witness_x[0]))]
    jobs += [(S, _witness_at(S, S_in, x)) for S, x in zip(stars, witness_x)]
    res = split_wave(jobs, S_in, layers, _SPLIT_SPEC, LayerOrderSelector())
    assert len(res) == 3
    assert res[0] is None, "a job with no relaxed neuron must stay None"
    for j, (fixed, mode) in enumerate(zip(jobs_fixed, modes)):
        _assert_children_hold_their_phase_regions(
            net, X, fixed, res[j + 1], f"split_wave({modes}) job {j + 1}")


# --------------------------------------------------------------------------- #
# The public route: verify_refine through split, split_wave and the contraction #
# --------------------------------------------------------------------------- #
class _AbsentKeySelector(FaithfulSelector):
    """A selector whose key is absent from every star's metadata."""

    def choose(self, *args, **kwargs):
        from n2v.refine.types import NeuronKey

        return NeuronKey(7, 99)


@pytest.mark.parametrize("contract_inputs", [False, True],
                         ids=["split", "split_wave"])
def test_a_branch_that_cannot_be_split_is_not_unsat(contract_inputs,
                                                     engine_device):
    """On the split net over its box, y0 has the true maximum 1.5625 at the
    corner (1.5, -0.75) (y0 = relu(2 relu(x0 - 0.25) - 0.5) - relu(m1), m1 =
    0.4375 there), so the unsafe region y0 >= 1.5546875 holds a
    counterexample: the true answer is SAT, and the root relaxation needs a
    split to find it. With a selector whose key is absent from the
    metadata no split can be made, so ``verify_refine`` must not answer
    UNSAT (it answers UNKNOWN, or SAT with a real counterexample), through
    ``split`` and through ``split_wave``. Catches a split operation that
    returns an empty child list (read as a certified-empty region) where it
    must return None."""
    net = _split_net()
    with torch.no_grad():
        y = net(torch.tensor([[1.5, -0.75]], dtype=torch.float64)).numpy()
    if y[0, 0] != 1.5625:
        pytest.fail(f"test setup: y0 at the corner is {y[0, 0]}")
    spec = LinearSpec(G=np.array([[-1.0, 0.0]]), g=np.array([-1.5546875]))
    S_in = Star.from_bounds(_SPLIT_LO, _SPLIT_HI)
    res = verify_refine(S_in, net, spec, _AbsentKeySelector(),
                        contract_inputs=contract_inputs, node_budget=2000,
                        **CPU_CONFIG)
    assert res.status != Status.UNSAT, \
        "UNSAT although a counterexample exists and no split could be made"
    _assert_honest_ce(res, net, spec, _SPLIT_LO, _SPLIT_HI)


@pytest.mark.parametrize("bound_mode", ["box", "dual_gpu"])
def test_contraction_of_an_input_with_a_zero_width_coordinate(bound_mode,
                                                              engine_device):
    """``verify_refine(contract_inputs=True)`` on an input whose first
    coordinate has zero width (x0 = 0.5, x1 in [-0.75, 3]): there y0 =
    -relu(0.75 x1), whose minimum -2.25 is at x1 = 3, so the unsafe region
    y0 <= -2.2421875 holds a counterexample and the answer is SAT with a
    counterexample in the input set; the search needs splits, so the
    contraction runs on the zero-width coordinate (the per-child reach
    under box bounds, the group reach under dual_gpu). Catches a
    contraction that calls a zero-width interval empty and drops every
    child (a false UNSAT)."""
    from n2v.refine import LayerOrderSelector

    net = _split_net()
    with torch.no_grad():
        y = net(torch.tensor([[0.5, 3.0]], dtype=torch.float64)).numpy()
    if y[0, 0] != -2.25:
        pytest.fail(f"test setup: y0 at (0.5, 3) is {y[0, 0]}")
    V = np.array([[0.0, 1.0, 0.0], [1.125, 0.0, 1.875]])
    S_in = Star(V, np.zeros((0, 2)), np.zeros((0, 1)),
                np.array([[0.5], [-1.0]]), np.array([[0.5], [1.0]]))
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([-2.2421875]))
    res = verify_refine(S_in, net, spec, LayerOrderSelector(),
                        contract_inputs=True, node_budget=2000,
                        **{**CPU_CONFIG, "bound_mode": bound_mode})
    assert res.status == Status.SAT, f"{res.status} (nodes {res.nodes})"
    _assert_honest_ce(res, net, spec, np.array([0.5, -0.75]),
                      np.array([0.5, 3.0]))

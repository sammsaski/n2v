"""
Soundness of the refinement branch-and-bound (verify_refine).

The critical claims under test:

  * No false UNSAT, and completeness on small nets: the BaB verdict equals a
    trusted oracle (n2v's exact star reach) on random instances, under EVERY
    selector. Selection cannot change the verdict -- only the node count.
  * Honest counterexamples (Theorem 1): every SAT carries a concrete input the
    exact model maps into the unsafe region.
  * Localization (Theorem 1, contrapositive): a fully-fixed (exact) leaf whose
    witness has eps == 0 everywhere is a real counterexample.
  * On a dyadic model whose answer is known exactly (the exact image in
    rationals, ``tests/_oracles/rational.py``): every combination of
    ``bound_mode``, ``prune_mode`` and ``witness_mode`` gives UNSAT on an
    instance that is safe by 1/64, SAT with an honest witness on one unsafe in
    a small region, and never UNSAT on one unsafe in a single point; the same
    for a conjunction of rows that are reachable one by one but not together,
    for an input star with a binding row that is not a box row, for a
    degenerate input coordinate, for ``contract_inputs`` and for a
    disjunction whose frontier waves mix nodes with different live disjuncts.
  * A budget that runs out, a selector that cannot choose, an input star that
    has no finite box and a model left partly in training mode never give
    UNSAT or a false SAT, and ``nodes`` never exceeds ``node_budget``.

Every case runs the certified GPU engine on CPU torch (``gpu_available`` is
patched to False where a GPU mode is used).
"""

import functools
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.reach import reach_pytorch_model
from n2v.sets.star import Star
from n2v.utils.lpsolver import check_feasibility
from n2v.refine import (
    FaithfulSelector,
    LayerOrderSelector,
    LinearSpec,
    NeuronKey,
    Selector,
    Status,
    verify_refine,
    verify_refine_input_crown,
)
from tests._oracles import rational as R
from tests.known_bugs import known_bug


# --------------------------------------------------------------------------- #
# Trusted oracle: exact star reach + per-star unsafe-intersection feasibility. #
# --------------------------------------------------------------------------- #
def _star_intersects_unsafe(S: Star, spec: LinearSpec) -> bool:
    c_out = S.V[:, 0]
    V_out = S.V[:, 1:]
    A_rows = [spec.G @ V_out]
    b_rows = [spec.g - spec.G @ c_out]
    if S.C.size > 0:
        A_rows.append(S.C)
        b_rows.append(S.d.flatten())
    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)
    return check_feasibility(
        A=A, b=b, lb=S.predicate_lb.flatten(), ub=S.predicate_ub.flatten()
    )


def _oracle_verdict(model, input_star: Star, spec: LinearSpec) -> Status:
    """Sound+complete verdict via exact reach."""
    out_stars = reach_pytorch_model(model, input_star, method="exact")
    for S in out_stars:
        if _star_intersects_unsafe(S, spec):
            return Status.SAT
    return Status.UNSAT


def _rand_net(seed, dims=(2, 5, 5, 2)):
    torch.manual_seed(seed)
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers).double()


ALL_SELECTORS = [FaithfulSelector, LayerOrderSelector]

# The exact CPU configuration, pinned so a CUDA host does not switch these
# tests to the GPU defaults.
CPU_CONFIG = dict(bound_mode="box", prune_mode="exact", witness_mode="exact",
                  frontier_width=1)


@pytest.mark.parametrize("seed", range(8))
def test_verdict_matches_exact_oracle_all_selectors(seed):
    """BaB verdict == exact-reach oracle, for every selector (sound + complete)."""
    rng = np.random.default_rng(seed)
    net = _rand_net(seed)
    lb = rng.uniform(-1.0, -0.2, size=2)
    ub = rng.uniform(0.2, 1.0, size=2)
    S_in = Star.from_bounds(lb, ub)

    # Pick a spec that lands near the output range so we get a mix of SAT/UNSAT:
    # unsafe = { y_0 <= tau }. Use the exact-reach midpoint as a tunable threshold.
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mins = min(float(S.get_range(0)[0]) for S in out0)
    maxs = max(float(S.get_range(0)[1]) for S in out0)
    tau = mins + (maxs - mins) * (0.25 + 0.5 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    oracle = _oracle_verdict(net, S_in, spec)

    for make in ALL_SELECTORS:
        res = verify_refine(S_in, net, spec, make(), node_budget=20000,
                            **CPU_CONFIG)
        assert res.status != Status.UNKNOWN, "ran out of budget on a tiny net"
        assert res.status == oracle, (
            f"selector {make().name}: BaB={res.status} oracle={oracle} (seed={seed})"
        )
        if res.status == Status.SAT:
            # Theorem 1 honesty: returned CE lies in the box and truly violates
            # the property, checked on the rows directly (not by
            # LinearSpec.is_unsafe, which is code under test).
            assert res.counterexample_x is not None
            x = np.asarray(res.counterexample_x, dtype=np.float64).flatten()
            assert np.all(x >= lb) and np.all(x <= ub), "CE outside the box"
            with torch.no_grad():
                y = net(torch.as_tensor(x, dtype=torch.float64).unsqueeze(0))
            y = y.numpy().flatten()
            assert np.all(spec.G @ y - spec.g <= 0.0), "returned a fake CE"


@pytest.mark.parametrize("seed", range(8))
def test_no_false_unsat_against_sampling(seed):
    """If dense sampling finds a counterexample, the BaB must NOT say UNSAT."""
    rng = np.random.default_rng(1000 + seed)
    net = _rand_net(seed + 100)
    lb = rng.uniform(-1.0, -0.2, size=2)
    ub = rng.uniform(0.2, 1.0, size=2)
    S_in = Star.from_bounds(lb, ub)

    X = rng.uniform(lb, ub, size=(40000, 2))
    with torch.no_grad():
        Y = net(torch.as_tensor(X, dtype=torch.float64)).numpy()
    # unsafe = { y_0 <= tau } with tau chosen so a CE provably exists.
    tau = float(np.quantile(Y[:, 0], 0.10))
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))
    assert np.any(Y[:, 0] <= tau), "test setup: sampling found no CE"

    for make in ALL_SELECTORS:
        res = verify_refine(S_in, net, spec, make(), node_budget=20000,
                            **CPU_CONFIG)
        assert res.status == Status.SAT, (
            f"false non-SAT ({res.status}) by {make().name} despite sampled CE (seed={seed})"
        )


@pytest.mark.parametrize("seed", [1, 3, 6])
@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu"])
def test_incremental_reach_matches_from_scratch(seed, bound_mode):
    """
    Incremental (shared-prefix) reach is a pure speedup: same verdict AND same
    node count as from-scratch reach, for every selector and bound mode.

    Catches a resumed child reach that differs from the from-scratch one (a
    wrong resume layer, checkpoint, ReLU index or inherited bound): the trees
    then differ in size or verdict. Seeds 1, 3 and 6 of the original eight:
    the other five seeds caught no injected bug that these three do not
    catch under the same bound mode.
    """
    rng = np.random.default_rng(seed)
    net = _rand_net(seed)
    lb = rng.uniform(-1.0, -0.2, size=2)
    ub = rng.uniform(0.2, 1.0, size=2)
    S_in = Star.from_bounds(lb, ub)
    out0 = reach_pytorch_model(net, S_in, method="exact")
    mins = min(float(S.get_range(0)[0]) for S in out0)
    maxs = max(float(S.get_range(0)[1]) for S in out0)
    tau = mins + (maxs - mins) * (0.25 + 0.5 * rng.random())
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))

    for make in ALL_SELECTORS:
        full = verify_refine(S_in, net, spec, make(), node_budget=20000,
                             incremental=False,
                             **{**CPU_CONFIG, "bound_mode": bound_mode})
        incr = verify_refine(S_in, net, spec, make(), node_budget=20000,
                             incremental=True,
                             **{**CPU_CONFIG, "bound_mode": bound_mode})
        assert incr.status == full.status, f"{make().name}: verdict differs (seed={seed})"
        assert incr.nodes == full.nodes, (
            f"{make().name}: node count {incr.nodes} != {full.nodes} (seed={seed})"
        )


def test_returned_counterexample_lies_in_input_box():
    """The SAT witness is a point of the raw input box at zero tolerance (the
    competition replays membership without slack), given in input
    coordinates. The box is [-3, 1] x [0.5, 2.5], so a witness returned in the
    normalised predicate coordinates of the box star (each in [-1, 1]), or
    one rounded a little outside the box, fails."""
    net = _rand_net(7)
    lb, ub = np.array([-3.0, 0.5]), np.array([1.0, 2.5])
    S_in = Star.from_bounds(lb, ub)
    # A loose spec that every output satisfies: y_0 <= 1e3.
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([1e3]))
    res = verify_refine(S_in, net, spec, FaithfulSelector(), **CPU_CONFIG)
    assert res.status == Status.SAT
    x = np.asarray(res.counterexample_x, dtype=np.float64).flatten()
    assert x.shape == (2,)
    assert np.all(x >= lb) and np.all(x <= ub), x
    with torch.no_grad():
        y = net(torch.as_tensor(x).unsqueeze(0)).numpy().flatten()
    assert y[0] <= 1e3


def test_theorem1_zero_epsilon_leaf_is_real_ce():
    """
    A fully-fixed (exact) leaf that is feasible-in-unsafe must be a real CE
    (Theorem 1): with no relaxed neurons, eps == 0 everywhere by definition, and
    the witness reproduces the exact output.
    """
    from n2v.refine.reach_relaxed import extract_layers, relaxed_reach
    from n2v.refine.types import NeuronKey, Phase
    from n2v.refine.witness import epsilon_vector, is_true_counterexample, violation_lp

    net = _rand_net(3, dims=(2, 4, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))

    # Fix every hidden neuron ACTIVE -> exact (no relaxation).
    fixed = {NeuronKey(0, j): Phase.ACTIVE for j in range(4)}
    S_out, meta = relaxed_reach(S_in, layers, fixed)
    assert meta == []  # exact star: nothing relaxed

    # A spec guaranteed to intersect this exact star: y_0 <= its own max.
    ymax = float(S_out.get_range(0)[1])
    spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([ymax]))
    res = violation_lp(S_out, spec, include_Cd=True)
    assert res is not None
    alpha, t = res
    assert t <= 1e-7  # feasible in unsafe
    assert epsilon_vector(alpha, meta).size == 0  # eps vacuously zero
    is_ce, x_in, y = is_true_counterexample(alpha, S_in, net, spec)
    assert is_ce, "Theorem 1 violated: exact feasible leaf is not a real CE"


# --------------------------------------------------------------------------- #
# Input star whose predicate box is +/-inf (bounded only by its C/d rows).     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def diamond():
    """y = |x1| + |x2| over the diamond |a1| + |a2| <= 1, carried ONLY by
    C/d rows (predicate box +/-inf everywhere). True output range [0, 1].
    The instance that made verify_refine collapse to y = 0 (all four ReLUs
    silently identity-passed) and return a 1-node false UNSAT."""
    net = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=torch.float64))
        net[0].bias.zero_()
        net[2].weight.copy_(torch.ones((1, 4), dtype=torch.float64))
        net[2].bias.zero_()
    V = np.hstack([np.zeros((2, 1)), np.eye(2)])
    C = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    d = np.ones((4, 1))
    S = Star(V, C, d, np.full((2, 1), -np.inf), np.full((2, 1), np.inf))
    spec_sat = LinearSpec(G=np.array([[-1.0]]), g=np.array([-0.5]))
    spec_unsat = LinearSpec(G=np.array([[-1.0]]), g=np.array([-3.0]))
    return net, S, spec_sat, spec_unsat


def test_verify_refine_no_false_unsat_on_diamond(diamond):
    """Regression for the demonstrated verify_refine 1-node false UNSAT."""
    from n2v.refine.bab import verify_refine

    net, S, spec_sat, spec_unsat = diamond
    for bound_mode in ("box", "lp_cpu"):
        config = {**CPU_CONFIG, "bound_mode": bound_mode}
        res = verify_refine(S, net, spec_sat, **config)
        assert res.status.value == "SAT", bound_mode
        res = verify_refine(S, net, spec_unsat, **config)
        assert res.status.value == "UNSAT", bound_mode


# --------------------------------------------------------------------------- #
# LP failure is not a certificate of emptiness                                #
# --------------------------------------------------------------------------- #
_FAILURE_STATUSES = ["unbounded", "failed: iteration limit", "error: numerical"]


def _input_star():
    return Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))


# A spec every output satisfies: the whole reach is "unsafe", so any honest
# run must find a counterexample (SAT).
_EVERYTHING = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([1e9]))


@pytest.mark.parametrize("status", _FAILURE_STATUSES)
def test_verify_refine_degrades_to_unknown_when_the_witness_lp_fails(
        status, monkeypatch):
    """The headline regression: on a SAT instance, a failing witness LP must
    never yield UNSAT. UNKNOWN (or SAT from another path) is the sound outcome.
    """
    import n2v.refine.witness as wmod
    from n2v.refine.reach_relaxed import extract_layers

    net, S_in = _rand_net(5, dims=(2, 6, 6, 2)), _input_star()
    layers = extract_layers(net)
    monkeypatch.setattr(wmod, "solve_lp",
                        lambda **k: (None, None, status, None))
    res = verify_refine(S_in, net, _EVERYTHING, FaithfulSelector(),
                        layers=layers, bound_mode="box", witness_mode="exact",
                        prune_mode="exact", frontier_width=1, node_budget=200)
    assert res.status != Status.UNSAT


# --------------------------------------------------------------------------- #
# A dyadic model whose answer is known exactly                                 #
# --------------------------------------------------------------------------- #
# Hidden layer 1: h0..h3 = relu(+-(x0 - 1/4)), relu(+-(x1 - 3/4)), and with
# s = x0 + x1, h4 = relu(s - 3/2), h5 = relu(2 s - 3) = 2 h4. Hidden layer 2:
# v0 = relu(h0 + h1 + h2 + h3) = |x0 - 1/4| + |x1 - 3/4|,
# v1 = relu(2 h4 - h5 + 1/8) = 1/8 everywhere, v2 = relu(h4 - 1/4).
# Outputs: y0 = -v0 + 8 v1 - 1 = -(|x0 - 1/4| + |x1 - 3/4|), whose only
# maximum is 0 at the interior point (1/4, 3/4); y1 = v2; y2 = -v0/2 + v1 + v2.
# The triangle relaxations of h4, h5 and v1 are loose, so the relaxed y0 is
# far above 0 near the line s = 3/2: no instance below is decided at the
# root, and the root's witness is not a counterexample. Every weight is a
# multiple of 1/8, so float64 evaluation of the model is exact on dyadic
# inputs and the rational image is the model's own.
_DY_LAYERS = (
    ([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [2, 2]],
     [-0.25, 0.25, -0.75, 0.75, -1.5, -3.0]),
    ([[1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 2, -1], [0, 0, 0, 0, 1, 0]],
     [0.0, 0.125, -0.25]),
    ([[-1, 8, 0], [0, 0, 1], [-0.5, 1, 1]], [-1.0, 0.0, 0.0]),
)
# The input box: bounds beyond +-1, widths 4 and 3, centre off the origin.
_DY_LB, _DY_UB = [-3.0, -0.5], [1.0, 2.5]

# The instances: unsafe region {y : G y <= g}, in rationals.
_DY_SPECS = {
    # safe by 1/64: y0 >= 1/64 is never reached (max y0 = 0)
    "safe": ([[-1, 0, 0]], [Fraction(-1, 64)]),
    # unsafe in a small region around (1/4, 3/4): y0 >= -1/64
    "region": ([[-1, 0, 0]], [Fraction(1, 64)]),
    # unsafe in the single point (1/4, 3/4): y0 >= 0
    "point": ([[-1, 0, 0]], [Fraction(0)]),
    # y0 >= -1/4 and y1 >= 1/4: each row alone is reached, both together not
    # (y1 >= 1/4 needs s >= 2, where y0 <= -1)
    "conjunction": ([[-1, 0, 0], [0, -1, 0]],
                    [Fraction(1, 4), Fraction(-1, 4)]),
}


def _dyadic_net(layers=_DY_LAYERS):
    dims = [len(layers[0][0][0])] + [len(W) for W, _ in layers]
    mods = []
    for i in range(len(layers)):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(layers) - 1:
            mods.append(nn.ReLU())
    net = nn.Sequential(*mods).double()
    with torch.no_grad():
        for lin, (W, b) in zip(net[::2], layers):
            lin.weight.copy_(torch.tensor(W, dtype=torch.float64))
            lin.bias.copy_(torch.tensor(b, dtype=torch.float64))
    return net


def _exact_pieces(region, layers=_DY_LAYERS):
    """The exact image of ``region`` (a list of ``Piece``) under the model, in
    rationals: one affine piece per activation pattern."""
    pieces = region
    for i, (W, b) in enumerate(layers):
        pieces = R.affine(pieces, W, b)
        if i < len(layers) - 1:
            pieces = R.relu(pieces)
    return pieces


@functools.lru_cache(maxsize=None)
def _box_pieces():
    return _exact_pieces(R.box(_DY_LB, _DY_UB))


def _box_rows(lb, ub):
    n = len(lb)
    eye = [[int(i == j) for j in range(n)] for i in range(n)]
    return (eye + [[-v for v in row] for row in eye],
            [Fraction(v) for v in ub] + [-Fraction(v) for v in lb])


def _spec(name):
    G, g = _DY_SPECS[name]
    return LinearSpec(np.array(G, dtype=np.float64),
                      np.array([float(v) for v in g]))


def _exactly_reached(pieces, specs):
    """True when some input of some piece maps into the unsafe region of some
    spec, decided by an exact feasibility problem in rationals per piece."""
    for sp in specs:
        G = R.matrix(sp.G.tolist())
        g = R.vector(sp.g.tolist())
        for p in pieces:
            rows = [[R.dot(Gr, col) for col in zip(*p.W)] for Gr in G]
            rhs = [gi - R.dot(Gr, p.c) for Gr, gi in zip(G, g)]
            if R.feasible(A_ub=p.A + rows, b_ub=p.b + rhs) is not None:
                return True
    return False


def _assert_honest_witness(res, net, specs, rows, pieces, keep=None):
    """A SAT result's witness lies in the input region ``rows`` (``A x <= b``
    in rationals, the region the input star stands for) at zero tolerance,
    and both the float64 forward pass and the exact value of the model there
    are in the closed unsafe region of at least one spec. ``keep`` lists the
    coordinates of the witness that ``pieces`` read (all by default)."""
    if res.status != Status.SAT:
        return
    if isinstance(specs, LinearSpec):
        specs = [specs]
    assert res.counterexample_x is not None, "SAT without a counterexample"
    x = np.asarray(res.counterexample_x, dtype=np.float64).flatten()
    A, b = rows
    assert x.size == len(A[0]), f"the witness {x.tolist()} has the wrong size"
    xq = R.vector(x.tolist())
    assert all(R.dot(R.vector(a), xq) <= bi for a, bi in zip(A, b)), \
        f"the witness {x.tolist()} is not in the input region"
    with torch.no_grad():
        y = net(torch.as_tensor(x).unsqueeze(0)).numpy().flatten()
    assert any(np.all(sp.G @ y - sp.g <= 0.0) for sp in specs), \
        f"float64 forward pass at the witness {x.tolist()} is safe: {y}"
    yq = R.evaluate(pieces, xq if keep is None else [xq[i] for i in keep])
    assert any(all(R.dot(R.vector(Gr), yq) <= R.frac(gi)
                   for Gr, gi in zip(sp.G.tolist(), sp.g.tolist()))
               for sp in specs), \
        f"the exact output at the witness {x.tolist()} is safe"


@pytest.fixture
def cpu_engine(monkeypatch):
    """The certified GPU engine runs on CPU torch (device None -> 'cpu')."""
    monkeypatch.setattr("n2v.utils.lpsolver_gpu.gpu_available", lambda: False)


def _grid_params():
    params = []
    for bound in ("box", "lp_cpu", "dual_gpu"):
        for prune in ("exact", "dual_gpu"):
            for witness in ("exact", "dual_gpu"):
                names = ["safe", "region", "point"]
                if witness == "exact":
                    # the approximate witness cannot see an empty polytope, so
                    # the conjunction takes some 30 to 90 nodes under it: the
                    # exact-witness arms carry the conjunction
                    names.append("conjunction")
                for name in names:
                    params.append(pytest.param(
                        bound, prune, witness, name,
                        id=f"{name}-{bound}-{prune}-{witness}"))
    return params


@pytest.mark.parametrize("bound_mode,prune_mode,witness_mode,instance",
                         _grid_params())
def test_option_grid_matches_the_exact_oracle(bound_mode, prune_mode,
                                              witness_mode, instance,
                                              cpu_engine):
    """Every (bound_mode, prune_mode, witness_mode) the driver accepts, on the
    dyadic model whose answer is known in rationals: UNSAT on the instance
    that is safe by 1/64 and on the conjunction whose rows are reached one by
    one but not together, SAT with an honest witness on the instance unsafe
    in a small region, never UNSAT on the instance unsafe in one point.

    Catches a prune that retires a reachable node (false UNSAT: a dropped
    child, a wrong live mask, a GPU flag on the wrong node, an exact-leaf
    backstop that prunes a feasible leaf, an inward-rounded margin on the
    touching instance), a witness that is not a member of the input box or
    whose output is safe (a row-wise ``any`` in ``LinearSpec.is_unsafe`` on
    the conjunction, a normalised-coordinate witness), and a mode whose
    verdict differs from the others (the documented verdict invariance:
    ``prune_mode`` and ``witness_mode`` change node counts only)."""
    pieces = _box_pieces()
    spec = _spec(instance)
    truth = _exactly_reached(pieces, [spec])
    assert truth == (instance in ("region", "point")), "test setup"
    net = _dyadic_net()
    S = Star.from_bounds(np.array(_DY_LB), np.array(_DY_UB))
    width = 1 if (prune_mode, witness_mode) == ("exact", "exact") else 3
    res = verify_refine(S, net, spec, FaithfulSelector(),
                        bound_mode=bound_mode, prune_mode=prune_mode,
                        witness_mode=witness_mode, frontier_width=width,
                        gpu_iters=50)
    if instance in ("safe", "conjunction"):
        assert res.status is Status.UNSAT, res
    elif instance == "region":
        assert res.status is Status.SAT, res
    else:
        assert res.status is not Status.UNSAT, res
    _assert_honest_witness(res, net, spec, _box_rows(_DY_LB, _DY_UB), pieces)


# The two engine configurations of the driver: exact CPU checks on single
# nodes, and the certified GPU bounds, prune and witnesses on waves of 3.
_CONFIGS = {
    "cpu": dict(bound_mode="box", prune_mode="exact", witness_mode="exact",
                frontier_width=1),
    "gpu": dict(bound_mode="dual_gpu", prune_mode="dual_gpu",
                witness_mode="dual_gpu", frontier_width=3, gpu_iters=50),
}


@pytest.mark.parametrize("cut,config", [(1, "cpu"), (1, "gpu"),
                                        (Fraction(3, 4), "cpu")],
                         ids=["s<=1-cpu", "s<=1-gpu", "s<=3/4-cpu"])
def test_witness_respects_a_binding_input_row(cut, config, cpu_engine):
    """The input star is the box cut by ``x0 + x1 <= cut``, a row that is not
    a box row, written over a predicate box that is the raw input box (not
    the normalised [-1, 1] box of ``from_bounds``); the instance is the
    region one (y0 >= -1/64). With cut = 1 the row passes through the only
    maximum (1/4, 3/4) of y0 and half of the small unsafe region is cut
    away, so the answer is SAT and the witness must satisfy the row at zero
    tolerance. With cut = 3/4 the row removes every counterexample
    (max y0 = -1/4 on the cut box), so the answer is UNSAT.

    Catches a witness taken from the box alone (an approximate witness that
    breaks the row, a membership check that reads only the predicate box)
    and an input row lost on the way to the reach (false SAT at cut = 3/4)."""
    A, b = _box_rows(_DY_LB, _DY_UB)
    rows = (A + [[1, 1]], b + [Fraction(cut)])
    pieces = _exact_pieces(R.polytope(*rows))
    spec = _spec("region")
    truth = _exactly_reached(pieces, [spec])
    assert truth == (cut == 1), "test setup"
    V = np.hstack([np.zeros((2, 1)), np.eye(2)])
    S = Star(V, np.array([[1.0, 1.0]]), np.array([[float(cut)]]),
             np.array(_DY_LB).reshape(-1, 1), np.array(_DY_UB).reshape(-1, 1))
    net = _dyadic_net()
    res = verify_refine(S, net, spec, FaithfulSelector(), **_CONFIGS[config])
    assert res.status is (Status.SAT if truth else Status.UNSAT), res
    _assert_honest_witness(res, net, spec, rows, pieces)


@known_bug(19, "verify_refine rebuilds the SAT witness as x = c + V alpha "
                 "rounded to nearest and snaps it only into a plain box, so "
                 "on an input star with a binding row in normalised "
                 "coordinates the returned x lies outside the star by about "
                 "one ulp (here 5 x0 + x1 = 1 + 2^-52 against the row "
                 "5 x0 + x1 <= 1): an invalid witness at zero tolerance")
def test_sat_witness_is_a_member_of_a_box_star_cut_by_a_row():
    """The box star of ``from_bounds`` (normalised predicate coordinates,
    c = (-1, 1), V = diag(2, 3/2)) cut by the row 10 a0 + 3/2 a1 <= 5, which
    is exactly 5 x0 + x1 <= 1 in input coordinates (all stored values are
    dyadic, so the two rows are the same set). The instance y0 >= -1/4 is
    reached inside the cut box, so the answer is SAT; the witness must be a
    member of the input set at zero tolerance, decided in rationals.

    Catches a witness reconstructed from the predicate point without a
    membership check of the returned point itself (the passing neighbour,
    ``test_witness_respects_a_binding_input_row``, uses a star whose basis
    is the identity, where no rounding occurs)."""
    A, b = _box_rows(_DY_LB, _DY_UB)
    rows = (A + [[5, 1]], b + [Fraction(1)])
    pieces = _exact_pieces(R.polytope(*rows))
    spec = LinearSpec(np.array([[-1.0, 0.0, 0.0]]), np.array([0.25]))
    if not _exactly_reached(pieces, [spec]):
        pytest.fail("test setup: the instance is not reached on the cut box")
    base = Star.from_bounds(np.array(_DY_LB), np.array(_DY_UB))
    if not (np.array_equal(base.V, [[-1.0, 2.0, 0.0], [1.0, 0.0, 1.5]])
            and np.all(base.predicate_lb == -1.0)
            and np.all(base.predicate_ub == 1.0)):
        pytest.fail("test setup: unexpected box star parametrisation")
    S = Star(base.V, np.vstack([base.C, [[10.0, 1.5]]]),
             np.vstack([base.d, [[5.0]]]), base.predicate_lb,
             base.predicate_ub)
    net = _dyadic_net()
    res = verify_refine(S, net, spec, FaithfulSelector(), **_CONFIGS["cpu"])
    if res.status is Status.UNSAT:
        # a false UNSAT is another defect: the marker must not absorb it
        pytest.fail(f"false UNSAT on a reached instance: {res}")
    _assert_honest_witness(res, net, spec, rows, pieces)


@pytest.mark.parametrize("config", list(_CONFIGS))
@pytest.mark.parametrize("instance", ["safe", "region"])
def test_degenerate_input_coordinate(instance, config, cpu_engine):
    """A third input fixed at x2 = 1/2 (a zero-width box coordinate, so the
    box star has 2 predicate variables for 3 inputs); the model reads x2 in
    h4 and h5 and computes the same function as the two-input model there.
    UNSAT on the safe instance, SAT with a witness whose x2 is exactly 1/2 on
    the region instance.

    Catches a witness assembled with the predicate count in place of the
    input dimension (a wrong length, a dropped or shifted coordinate) and a
    degenerate coordinate read as free."""
    W1, b1 = _DY_LAYERS[0]
    layers = (([row + [c] for row, c in zip(W1, [0, 0, 0, 0, 1, 2])],
               [v - c for v, c in zip(b1, [0, 0, 0, 0, 0.5, 1.0])]),
              ) + _DY_LAYERS[1:]
    lb, ub = _DY_LB + [0.5], _DY_UB + [0.5]
    # At x2 = 1/2 the three-input model is the two-input one, so the exact
    # image of the two-input model is the oracle (the witness's x2 is held to
    # 1/2 exactly by the box rows of the check).
    pieces = _box_pieces()
    spec = _spec(instance)
    assert _exactly_reached(pieces, [spec]) == (instance == "region"), \
        "test setup"
    S = Star.from_bounds(np.array(lb), np.array(ub))
    assert S.nVar == 2, "test setup: the fixed coordinate has no variable"
    net = _dyadic_net(layers)
    with torch.no_grad():
        probe = torch.tensor([[-3.0, -0.5, 0.5], [1.0, 2.5, 0.5],
                              [0.25, 0.75, 0.5], [0.5, 1.5, 0.5]],
                             dtype=torch.float64)
        assert torch.equal(net(probe), _dyadic_net()(probe[:, :2])), \
            "test setup: the models differ at x2 = 1/2"
    res = verify_refine(S, net, spec, FaithfulSelector(), **_CONFIGS[config])
    assert res.status is (Status.SAT if instance == "region"
                          else Status.UNSAT), res
    _assert_honest_witness(res, net, spec, _box_rows(lb, ub), pieces,
                           keep=[0, 1])


@pytest.mark.parametrize("frontier_width", [1, 3])
@pytest.mark.parametrize("node_budget", [1, 2, 3])
def test_an_exhausted_node_budget_is_unknown_never_unsat(node_budget,
                                                         frontier_width):
    """On the region instance (unsafe, and the CPU configuration needs 4 nodes
    to find the counterexample) a node budget of 1 to 3 runs out: the answer
    is UNKNOWN or an honest SAT, never UNSAT, and no more than
    ``node_budget`` nodes are expanded.

    Catches a budget exit that answers UNSAT (at the loop head, after the
    root, or before a split whose children can no longer be expanded) and a
    wave that expands nodes past the budget (frontier_width 3)."""
    pieces = _box_pieces()
    spec = _spec("region")
    net = _dyadic_net()
    S = Star.from_bounds(np.array(_DY_LB), np.array(_DY_UB))
    res = verify_refine(S, net, spec, FaithfulSelector(),
                        **{**_CONFIGS["cpu"], "frontier_width": frontier_width},
                        node_budget=node_budget)
    assert res.status in (Status.SAT, Status.UNKNOWN), res
    assert res.nodes <= node_budget, res
    _assert_honest_witness(res, net, spec, _box_rows(_DY_LB, _DY_UB), pieces)


@pytest.mark.parametrize("config", list(_CONFIGS))
@pytest.mark.parametrize("ticks", [1, 2, 3, 4, 5, 6])
def test_a_deadline_that_expires_mid_search_is_unknown(ticks, config,
                                                       monkeypatch,
                                                       cpu_engine):
    """A fake clock that advances by one second per reading makes the
    deadline expire at the ``ticks``-th reading after the start. On this
    instance, in both configurations, ticks 1 to 5 fire the deadline checks
    at the loop head, after the wave prune, after the wave witnesses, per
    node and before a split, in that order, and tick 6 the loop head again
    with the root's children waiting. On the region instance (unsafe) the
    answer is UNKNOWN or an honest SAT, never UNSAT.

    Catches a deadline exit that answers anything but UNKNOWN, or that drops
    the node it interrupts and lets the search finish with UNSAT."""
    import time as time_module

    clock = iter(range(10 ** 6))
    monkeypatch.setattr(time_module, "perf_counter",
                        lambda: float(next(clock)))
    pieces = _box_pieces()
    spec = _spec("region")
    net = _dyadic_net()
    S = Star.from_bounds(np.array(_DY_LB), np.array(_DY_UB))
    res = verify_refine(S, net, spec, FaithfulSelector(), **_CONFIGS[config],
                        time_budget=ticks - 0.5)
    assert res.status in (Status.SAT, Status.UNKNOWN), res
    _assert_honest_witness(res, net, spec, _box_rows(_DY_LB, _DY_UB), pieces)


class _Abstain(Selector):
    """A selector that declines to choose."""

    name = "abstain"

    def choose(self, S_out, spec, meta, faithful_witness, lp_solver=None):
        return None


class _AbsentKey(Selector):
    """A selector that names a neuron the star does not relax."""

    name = "absent"

    def choose(self, S_out, spec, meta, faithful_witness, lp_solver=None):
        return NeuronKey(99, 0)


@pytest.mark.parametrize("contract_inputs", [False, True],
                         ids=["split", "contract-split"])
@pytest.mark.parametrize("selector", [_Abstain, _AbsentKey],
                         ids=["none", "absent-key"])
def test_a_selector_that_cannot_choose_is_unknown(selector, contract_inputs,
                                                  cpu_engine):
    """The root of the region instance is neither pruned nor a
    counterexample, so it must be split; a selector that returns None or a
    key the star does not relax leaves it unrefined, and the run must answer
    UNKNOWN (the counterexamples are in that root), on the plain split and on
    the deferred contract split.

    Catches an unrefinable branch dropped without marking the run
    inconclusive (false UNSAT)."""
    net = _dyadic_net()
    S = Star.from_bounds(np.array(_DY_LB), np.array(_DY_UB))
    res = verify_refine(S, net, _spec("region"), selector(),
                        **_CONFIGS["cpu"], contract_inputs=contract_inputs)
    assert res.status is Status.UNKNOWN, res


@pytest.mark.parametrize("config", list(_CONFIGS))
@pytest.mark.parametrize("instance", ["safe", "region", "point"])
def test_contract_inputs_matches_the_exact_oracle(instance, config,
                                                  cpu_engine):
    """``contract_inputs=True`` (input-box contraction at every split, wave
    batched split, children that replace their parent) with the default
    selector on the dyadic instances: UNSAT, SAT, never UNSAT.

    Catches a child of a contracted split that is not pushed (the witness
    phase child holds the counterexamples here: false UNSAT on region and
    point) and a contracted box that cuts a counterexample."""
    pieces = _box_pieces()
    spec = _spec(instance)
    net = _dyadic_net()
    S = Star.from_bounds(np.array(_DY_LB), np.array(_DY_UB))
    res = verify_refine(S, net, spec, FaithfulSelector(),
                        **_CONFIGS[config], contract_inputs=True)
    if instance == "safe":
        assert res.status is Status.UNSAT, res
    elif instance == "region":
        assert res.status is Status.SAT, res
    else:
        assert res.status is not Status.UNSAT, res
    _assert_honest_witness(res, net, spec, _box_rows(_DY_LB, _DY_UB), pieces)


# A dyadic 2-4-4-3 ReLU model and a three-way disjunction on it: y2 >= 313/256
# (max y2 = 305/256: never reached), y0 >= 5/4 (max y0 = 41/32: reached in a
# small region) and y0 <= -129/128 (min y0 = -125/128: never reached). The
# certified prune retires different disjuncts on different nodes, so the
# frontier waves of width 3 mix nodes whose live disjuncts differ.
_WAVE_LAYERS = (
    ([[-0.75, 0.0], [0.75, 0.25], [0.75, 0.0], [-0.75, -0.75]],
     [0.75, 0.25, -0.25, 1.0]),
    ([[0.0, 0.0, -0.5, -0.75], [0.0, -0.5, -0.5, 0.5],
      [-0.75, 0.0, -1.0, 0.75], [1.0, 0.75, 0.5, -0.5]],
     [0.0, 1.0, 0.75, -0.75]),
    ([[0.75, -0.75, 0.25, 1.0], [-0.75, 0.5, -0.75, -0.5],
      [-0.25, -0.75, 1.0, 0.25]],
     [0.5, 0.5, 0.75]),
)
_WAVE_SPECS = [LinearSpec(np.array([[0.0, 0.0, -1.0]]), np.array([-313 / 256])),
               LinearSpec(np.array([[-1.0, 0.0, 0.0]]), np.array([-1.25])),
               LinearSpec(np.array([[1.0, 0.0, 0.0]]), np.array([-129 / 128]))]


@pytest.mark.parametrize("prune_mode", ["exact", "dual_gpu"])
def test_multi_disjunct_waves_keep_each_node_own_live_mask(prune_mode,
                                                           cpu_engine):
    """One search tree for the disjunction, popped in waves of 3: the answer
    is SAT (only the second disjunct is reached), with an honest witness.

    Catches children that inherit the live disjuncts of another node of the
    wave instead of their parent's, and a certified wave prune whose flags
    are assigned to the wave's nodes by position instead of to the nodes that
    still had the disjunct live (both drop the reachable disjunct from the
    nodes that hold its counterexamples: false UNSAT)."""
    pieces = _exact_pieces(R.box(_DY_LB, _DY_UB), _WAVE_LAYERS)
    reached = [_exactly_reached(pieces, [sp]) for sp in _WAVE_SPECS]
    assert reached == [False, True, False], "test setup"
    net = _dyadic_net(_WAVE_LAYERS)
    S = Star.from_bounds(np.array(_DY_LB), np.array(_DY_UB))
    res = verify_refine(S, net, _WAVE_SPECS, FaithfulSelector(),
                        bound_mode="box", prune_mode=prune_mode,
                        witness_mode="exact", frontier_width=3, gpu_iters=50)
    assert res.status is Status.SAT, res
    _assert_honest_witness(res, net, _WAVE_SPECS, _box_rows(_DY_LB, _DY_UB),
                           pieces)


@pytest.mark.parametrize("driver", ["verify_refine",
                                    "verify_refine_input_crown"])
def test_an_input_star_without_a_finite_box_is_unknown(driver):
    """The input star bounds x0 to [-1, 1] by its rows and leaves x1 free:
    no finite predicate box exists, and the documented answer is UNKNOWN. A
    counterexample exists (y1 = relu(relu(x0 + x1 - 3/2) - 1/4) >= 1 at
    x = (0, 3), checked below on the model), so UNSAT is false.

    Catches a finitizer failure read as a proof (false UNSAT) and an
    unbounded variable silently clipped to a finite box."""
    net = _dyadic_net()
    spec = LinearSpec(np.array([[0.0, -1.0, 0.0]]), np.array([-1.0]))
    with torch.no_grad():
        y = net(torch.tensor([[0.0, 3.0]], dtype=torch.float64)).numpy()[0]
    assert np.all(spec.G @ y - spec.g <= 0.0), "test setup: no CE at (0, 3)"
    V = np.hstack([np.zeros((2, 1)), np.eye(2)])
    S = Star(V, np.array([[1.0, 0.0], [-1.0, 0.0]]), np.array([[1.0], [1.0]]),
             np.full((2, 1), -np.inf), np.full((2, 1), np.inf))
    if driver == "verify_refine":
        res = verify_refine(S, net, spec, **_CONFIGS["cpu"])
    else:
        res = verify_refine_input_crown(S, net, spec, node_budget=500)
    assert res.status is Status.UNKNOWN, res


@pytest.mark.parametrize("config", ["gpu", "gpu-prune-exact-witness"])
def test_an_empty_input_star_is_never_sat(config, cpu_engine):
    """The raw box cut by x0 <= -1 and x0 >= 1/2 is empty, with a finite
    predicate box (so the finitizer passes it on to the search). Every output
    is unsafe for the spec, so a SAT can only come from a witness outside the
    input set. Under the GPU configuration, whose approximate witnesses are
    only box-feasible, the answer must be UNSAT (vacuous) or UNKNOWN. Under
    the certified prune (which does not retire the empty root) with the exact
    witness LP, which finds the predicate polytope empty, the answer is the
    vacuous UNSAT.

    Catches a counterexample check that reads the predicate box and not the
    input rows (a false SAT with a witness outside the input set), and a node
    whose exact witness LP finds it empty kept as undecided (UNKNOWN where
    the vacuous UNSAT is due)."""
    if config == "gpu":
        cfg = _CONFIGS["gpu"]
    else:
        cfg = dict(bound_mode="box", prune_mode="dual_gpu",
                   witness_mode="exact", frontier_width=1, gpu_iters=50)
    V = np.hstack([np.zeros((2, 1)), np.eye(2)])
    S = Star(V, np.array([[1.0, 0.0], [-1.0, 0.0]]),
             np.array([[-1.0], [-0.5]]),
             np.array(_DY_LB).reshape(-1, 1), np.array(_DY_UB).reshape(-1, 1))
    spec = LinearSpec(np.array([[1.0, 0.0, 0.0]]), np.array([1e9]))
    res = verify_refine(S, _dyadic_net(), spec, FaithfulSelector(), **cfg,
                        node_budget=200)
    if config == "gpu":
        assert res.status is not Status.SAT, res
    else:
        assert res.status is Status.UNSAT, res


def test_a_model_left_partly_in_training_mode_is_checked_in_eval_mode():
    """y = relu(x0 + x1) + relu(x0 - x1) with a Dropout after the ReLU that
    is left in training mode while every other module is in eval mode. The
    verified function is the eval-mode one, whose maximum over [-1, 1]^2 is 2,
    so the unsafe region y >= 5/2 is never reached: UNSAT. A training-mode
    Dropout scales kept units by 2, so a counterexample check that runs it
    can see y = 4 and answer a false SAT. The caller's modules keep their
    modes.

    Catches an eval-mode switch that fires only when every module is in
    training mode, and one that switches the caller's model."""
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(2, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0, 1.0], [1.0, -1.0]],
                                         dtype=torch.float64))
        net[0].bias.zero_()
        net[3].weight.copy_(torch.tensor([[1.0, 1.0]], dtype=torch.float64))
        net[3].bias.zero_()
    net.eval()
    net[2].train()
    S = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    spec = LinearSpec(np.array([[-1.0]]), np.array([-2.5]))
    res = verify_refine(S, net, spec, FaithfulSelector(), **_CONFIGS["cpu"])
    assert res.status is Status.UNSAT, res
    assert [m.training for m in net] == [False, False, True, False]

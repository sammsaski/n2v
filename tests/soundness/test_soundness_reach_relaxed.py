"""Soundness cells of the refinement verifier's own reach,
``n2v/refine/reach_relaxed.py``.

The refinement verifier (``n2v.refine.verify_refine``) does not use
``n2v.nn.reach``. It extracts a layer list from the model and pushes a star
through it with a triangle relaxation of its own, in which chosen ReLU neurons
are fixed to one phase. Every verdict it gives is about that reach, so the
promise has three links, and this file checks each against a reference that
does not come from n2v:

1. ``extract_layers(model, in_shape)`` returns layers whose composition is the
   model. Checked by reaching single points (stars with no predicate
   variable) through the extracted layers and comparing, exactly, with the
   float64 torch forward pass of the model on dyadic weights and inputs, for
   every module kind and convolution parameter the extraction accepts; an
   affine conv-type layer is checked as the exact affine image of a star with
   generators.
2. ``relaxed_reach(input_star, layers, fixed, bound_mode)``: for every member
   of the input set whose neurons have the phases in ``fixed``, the true output
   is a member of the returned star. Checked in exact rational arithmetic
   (``tests/_oracles/rational.py``: every vertex of the exact image of the
   phase region is a member of the star by its own rows, with zero slack, and
   for a star with no relaxed neuron left every vertex of the star has a
   preimage) on a one-layer network whose triangle slopes are dyadic, and by
   point replay (float64 torch forward, membership by a linear program on the
   star's rows, ``tests/_oracles/membership.py``) on deeper networks, for the
   three bound modes, input sets with a binding row that is not a box row,
   with no box rows, with no predicate box and with a zero-width coordinate.
3. ``resume_reach`` and ``resume_reach_pair`` continue from a stored prefix:
   each child, and a grandchild resumed from a child, holds every true output
   of its own phase region (point replay as in 2), under every bound mode and
   both pair paths (two delegated resumes, and the lockstep walk of
   ``dual_gpu``). The public route, ``verify_refine``, never answers UNSAT
   for a property whose violation the exact image shows.

Every case runs on the CPU: the ``dual_gpu`` bound engine runs on CPU torch
(an autouse fixture patches ``gpu_available``, as ``engine_device`` does for
its CPU parameter).

Defects of today's library are marked with strict expected failures (see the
markers): a Sequential that holds one module instance at two positions is
extracted with that module once (a false UNSAT end to end); the triangle's
upper row is stored rounded to nearest, so it can cut the vertex (u, u) of a
crossing neuron by one unit in the last place; an nn.Unflatten size with -1
crashes the extraction; an nn.Linear on an image activation is accepted and
fails later inside the reach.

Convention: SAT = unsafe (the unsafe region is ``G y <= g``), UNSAT = safe.
"""

import copy
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.refine import LinearSpec, Status, verify_refine
from n2v.refine.reach_relaxed import (ConvLayer, extract_layers,
                                      relaxed_reach, resume_reach,
                                      resume_reach_pair)
from n2v.refine.types import NeuronKey, Phase
from n2v.sets.star import Star
from tests._oracles import rational as R
from tests._oracles.membership import uncovered
from tests.cell_support import constrained_star, samples
from tests.known_bugs import known_bug

A, I = Phase.ACTIVE, Phase.INACTIVE

CPU_CONFIG = dict(bound_mode="box", prune_mode="exact", witness_mode="exact",
                  frontier_width=1)


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Every case runs on the CPU: the certified dual_gpu bound engine
    resolves its device through ``gpu_available`` (CPU torch when False, the
    path a CPU-only host takes) and ``ConvLayer.apply`` through
    ``torch.cuda.is_available``."""
    monkeypatch.setattr("n2v.utils.lpsolver_gpu.gpu_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _set(lin, W, b=None):
    with torch.no_grad():
        lin.weight.copy_(torch.as_tensor(np.asarray(W, dtype=np.float64)))
        if b is not None:
            lin.bias.copy_(torch.as_tensor(np.asarray(b, dtype=np.float64)))
    return lin


def _dyadic(model, seed, bits=3, span=12):
    """Every weight and bias of ``model`` a multiple of 2**-bits in
    [-span, span] * 2**-bits, seeded: float32, float64 and rational
    evaluations then agree exactly."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.as_tensor(
                rng.integers(-span, span + 1, size=tuple(p.shape)) / 2.0 ** bits,
                dtype=p.dtype))
    return model


def _forward(model, X):
    with torch.no_grad():
        return model(torch.as_tensor(np.asarray(X, dtype=np.float64))).numpy()


def _preacts(model, X):
    """The input of every nn.ReLU of a flat Sequential, in order, by the
    float64 torch forward pass: a list of (n_points, width) arrays."""
    h = torch.as_tensor(np.asarray(X, dtype=np.float64))
    out = []
    with torch.no_grad():
        for mod in model:
            if isinstance(mod, nn.ReLU):
                out.append(h.numpy().reshape(h.shape[0], -1).copy())
            h = mod(h)
    return out


def _in_phase(model, X, fixed):
    """Mask of the points whose fixed neurons have their fixed phase (the
    closed regions: ACTIVE is pre >= 0, INACTIVE is pre <= 0)."""
    pre = _preacts(model, X)
    keep = np.ones(len(X), dtype=bool)
    for k, ph in fixed.items():
        v = pre[k.layer][:, k.neuron]
        keep &= (v >= 0.0) if ph == Phase.ACTIVE else (v <= 0.0)
    return keep


def _assert_holds(star, model, X, fixed, what, min_points=16):
    """Every true output of a point of ``X`` in the phase region of
    ``fixed`` is a member of ``star`` by an LP on its own rows."""
    X = np.asarray(X, dtype=np.float64)
    keep = _in_phase(model, X, fixed)
    if keep.sum() < min_points:
        pytest.fail(f"{what}: only {keep.sum()} sample points in the phase "
                    f"region (setup)")
    Y = _forward(model, X[keep])
    missed = uncovered([star], Y)
    assert not missed, (f"{what}: {len(missed)} of {len(Y)} true outputs "
                        f"outside the star (largest distance "
                        f"{max(d for _, d in missed):.3g})")


def _box_star(lo, hi):
    return Star.from_bounds(np.asarray(lo, dtype=np.float64).reshape(-1, 1),
                            np.asarray(hi, dtype=np.float64).reshape(-1, 1))


# --------------------------------------------------------------------------- #
# 2. relaxed_reach with fixed phases, exact: one ReLU layer, dyadic slopes      #
# --------------------------------------------------------------------------- #
# x0 in [-1, 3], x1 in [-2, 2]. Pre-activations over that box:
#   n0 = x0/2 + 1/2           in [0, 2]         lower bound exactly 0, active
#   n1 = x0/2 + x1/2 + 1/2    in [-1, 3]        crossing, centre +1, slope 3/4
#   n2 = -x0/2 + x1/4 - 1/4   in [-9/4, 3/4]    crossing, centre -3/4, slope 1/4
#   n3 = x1/4 - 1/2           in [-1, 0]        upper bound exactly 0, inactive
#   n4 = x0/4 - 1             in [-5/4, -1/4]   inactive
# A stable neuron precedes each crossing one, and every output mixes all
# five, so a neuron read with the wrong phase, column or bound moves it.
_LO1, _HI1 = [-1.0, -2.0], [3.0, 2.0]
_W1 = [[0.5, 0.0], [0.5, 0.5], [-0.5, 0.25], [0.0, 0.25], [0.25, 0.0]]
_B1 = [0.5, 0.5, -0.25, -0.5, -1.0]
_W2 = [[1.0, -0.5, 0.25, 1.0, 0.5], [-0.5, 1.0, 1.0, -2.0, 1.0]]
_B2 = [0.25, -0.5]


def _one_layer_net():
    net = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 2)).double()
    _set(net[0], _W1, _B1)
    _set(net[2], _W2, _B2)
    return net


def _exact_pieces(fixed):
    """The exact image of the phase region of ``fixed`` (keys of ReLU layer
    0) under the one-layer network, as rational pieces."""
    pieces = R.affine(R.box(_LO1, _HI1), _W1, _B1)
    for k, ph in fixed.items():
        out = []
        for p in pieces:
            row, c = p.W[k.neuron], p.c[k.neuron]
            # ACTIVE: row x + c >= 0, i.e. -row x <= c; INACTIVE: row x <= -c
            a, r = (([-v for v in row], c) if ph == Phase.ACTIVE
                    else (list(row), -c))
            child = R.Piece(p.A + [a], p.b + [r], p.W, p.c, p.strict)
            if child.is_inhabited():
                out.append(child)
        pieces = out
    return R.affine(R.relu(pieces), _W2, _B2)


def _exact_range(j):
    vals = [sum(Fraction(w) * Fraction(x) for w, x in zip(_W1[j], v))
            + Fraction(_B1[j]) for v in ((a, b) for a in (_LO1[0], _HI1[0])
                                         for b in (_LO1[1], _HI1[1]))]
    return min(vals), max(vals)


_FIXES = {
    "none": {},
    "n1-active": {NeuronKey(0, 1): A},
    "n2-inactive": {NeuronKey(0, 2): I},
    "n1-inactive-n2-active": {NeuronKey(0, 1): I, NeuronKey(0, 2): A},
    "n1-active-n2-inactive": {NeuronKey(0, 1): A, NeuronKey(0, 2): I},
    "n0-inactive-on-its-zero-bound": {NeuronKey(0, 0): I},
    "n3-active-on-its-zero-bound": {NeuronKey(0, 3): A},
    "n4-active-excluded": {NeuronKey(0, 4): A},
}

_EXACT_CASES = ([("box", name) for name in _FIXES]
                + [(mode, name) for mode in ("lp_cpu", "dual_gpu")
                   for name in ("none", "n1-active-n2-inactive")])


@pytest.mark.parametrize("bound_mode,fix", _EXACT_CASES,
                         ids=[f"{m}-{f}" for m, f in _EXACT_CASES])
def test_fixed_phase_reach_holds_the_exact_image_of_the_phase_region(
        bound_mode, fix):
    """Every vertex of the exact image of the phase region is a member of the
    relaxed star by its own rows, in rationals with zero slack; with no
    crossing neuron left unfixed the star is that image exactly (every vertex
    of the star has a preimage), and an excluded phase gives an empty star.
    The relaxed neurons are exactly the unfixed crossing ones, each with
    outward bounds. Catches a phase row with the wrong sign for either phase
    (centres +1 and -3/4), a neuron with an upper bound of exactly 0 passed
    through, a lower bound of exactly 0 relaxed, a fresh variable read from
    the wrong column or bounded by -l, a triangle with the wrong slope or a
    lower row with the wrong sign, a fixed neuron that is also relaxed, phase
    rows that are built and not stored, and a dual_gpu or lp_cpu bound read
    in the wrong order."""
    fixed = _FIXES[fix]
    net = _one_layer_net()
    star, meta = relaxed_reach(_box_star(_LO1, _HI1), extract_layers(net),
                               fixed, bound_mode=bound_mode)
    computed = R.star(star)

    crossing = {NeuronKey(0, j) for j in range(5)
                if _exact_range(j)[0] < 0 < _exact_range(j)[1]}
    assert {m.key for m in meta} == crossing - set(fixed), \
        f"relaxed neurons {[m.key for m in meta]}"
    for m in meta:
        lo, hi = _exact_range(m.key.neuron)
        assert m.l <= lo and m.u >= hi, f"{m.key}: [{m.l}, {m.u}] misses [{lo}, {hi}]"

    pieces = _exact_pieces(fixed)
    if not pieces:
        assert not R.meets_box(computed, [-2.0 ** 20] * 2, [2.0 ** 20] * 2), \
            "an empty phase region gave a non-empty star"
    escapes = R.escapes(pieces, computed)
    assert not escapes, f"{len(escapes)} exact vertices outside the star: {escapes[:3]}"
    if not meta:
        vertices = R.set_vertices(computed)
        unreached = R.unreached(pieces, vertices)
        assert not unreached, f"star vertices without a preimage: {unreached[:3]}"


# --------------------------------------------------------------------------- #
# 2. relaxed_reach with fixed phases, point replay: three ReLU layers           #
# --------------------------------------------------------------------------- #
# Intervals of their own per coordinate, beyond +-1, on both sides of 0.
_LO, _HI = [-3.0, 0.25, -1.5], [0.5, 2.5, -0.25]


def _deep_net():
    """3 -> 6 -> 5 -> 4 -> 2, dyadic weights; seed 5 relaxes 5, 4 and 3
    neurons at the three ReLU layers of the box [_LO, _HI] in every bound
    mode, and the first relaxed neuron of each layer splits the box's sample
    points into two phase regions of about equal size."""
    net = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 5), nn.ReLU(),
                        nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    return _dyadic(net, 5, bits=3, span=10)


def _regime(name):
    """``(input_star, member points)`` of an input regime."""
    lo, hi = np.array(_LO), np.array(_HI)
    if name == "box":
        return _box_star(lo, hi), samples(lo, hi, n=128, seed=1)
    if name == "binding-row":
        star, points = constrained_star(lo, hi, n=128, seed=2)
        return star, points
    if name == "no-box-rows":
        # more predicate variables than entries and no constraint row: the
        # set is c + G a with a in [-1, 1]^4
        c, h = (lo + hi) / 2, (hi - lo) / 2
        G = np.hstack([np.diag(h), (h * np.array([0.25, -0.5, 0.25])).reshape(-1, 1)])
        star = Star(np.hstack([c.reshape(-1, 1), G]), np.zeros((0, 4)),
                    np.zeros((0, 1)), -np.ones((4, 1)), np.ones((4, 1)))
        rng = np.random.default_rng(3)
        alphas = rng.integers(-2, 3, size=(128, 4)) / 2.0
        alphas[:16] = np.array([[(k >> b & 1) * 2 - 1 for b in range(4)]
                                for k in range(16)])
        return star, [c + G @ a for a in alphas]
    if name == "no-predicate-box":
        # the box as constraint rows only: no predicate bounds at all
        c, h = (lo + hi) / 2, (hi - lo) / 2
        star = Star(np.hstack([c.reshape(-1, 1), np.diag(h)]),
                    np.vstack([np.eye(3), -np.eye(3)]), np.ones((6, 1)))
        return star, samples(lo, hi, n=128, seed=1)
    if name == "zero-width":
        lo[1] = hi[1] = 1.25
        return _box_star(lo, hi), samples(lo, hi, n=128, seed=1)
    raise ValueError(name)


def _first_relaxed(meta):
    first = {}
    for m in meta:
        first.setdefault(m.key.layer, m.key)
    return first


def _deep_fix(name, net, first, X):
    """The fixes of a case: ``l0-l2`` fixes the first relaxed neuron of
    layer 0 ACTIVE and that of layer 2 to the phase more of the remaining
    points have; ``l1-inactive`` fixes the first relaxed neuron of layer 1
    INACTIVE."""
    if name == "none":
        return {}
    if name == "l1-inactive":
        return {first[1]: I}
    fixed = {first[0]: A}
    return {**fixed, first[2]: _richer_phase(net, X[_in_phase(net, X, fixed)],
                                             first[2])}


def _richer_phase(net, X, key):
    """The phase of ``key`` that more of the points ``X`` have."""
    return A if _in_phase(net, X, {key: A}).sum() >= _in_phase(
        net, X, {key: I}).sum() else I


_DEEP_CASES = (
    [("box", mode, fix) for mode in ("box", "lp_cpu", "dual_gpu")
     for fix in ("none", "l0-l2", "l1-inactive")]
    + [("binding-row", mode, fix) for mode in ("box", "lp_cpu", "dual_gpu")
       for fix in ("none", "l0-l2")]
    + [("no-box-rows", "box", "none"), ("no-predicate-box", "box", "none"),
       ("no-predicate-box", "box", "l0-l2"), ("zero-width", "box", "none")])


@pytest.mark.parametrize("regime,bound_mode,fix", _DEEP_CASES,
                         ids=[f"{r}-{m}-{f}" for r, m, f in _DEEP_CASES])
def test_fixed_phase_reach_holds_every_true_output_of_the_phase_region(
        regime, bound_mode, fix):
    """Three ReLU layers: every true output (float64 torch) of a member
    point of the input set whose fixed neurons have their fixed phases is a
    member of the relaxed star by an LP on its rows. Fixes at the first and
    the last layer together catch a fix applied to the same-index neuron of
    a deeper layer; the input regimes reach the LP bound modes over a binding
    row, a star with no constraint row and more predicate variables than
    entries, a star with no predicate box (its fresh variables then have no
    box either) and a zero-width coordinate."""
    net = _deep_net()
    layers = extract_layers(net)
    star, points = _regime(regime)
    out, meta = relaxed_reach(star, layers, {}, bound_mode=bound_mode)
    fixed = {}
    if fix != "none":
        first = _first_relaxed(meta)
        if not {0, 1, 2} <= set(first):
            pytest.fail(f"setup: no relaxed neuron at some layer: {sorted(first)}")
        fixed = _deep_fix(fix, net, first, np.array(points))
        out, _ = relaxed_reach(star, layers, fixed, bound_mode=bound_mode)
    _assert_holds(out, net, points, fixed, f"{regime}/{bound_mode}/{fix}")


# --------------------------------------------------------------------------- #
# 3. resume_reach: children and a grandchild chain                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu", "dual_gpu"])
def test_resumed_children_and_grandchildren_hold_their_phase_regions(
        bound_mode):
    """A child resumed at the first and at the last ReLU layer, in both
    phases, and a chain resumed layer after layer (child at 0, grandchild at
    1 from the child, great-grandchild at 2 from the grandchild): each holds
    every true output of its own phase region. The chain reads the
    checkpoints, recorded bounds and meta that a resumed star carries, so it
    catches a child whose stored prefix is shifted by one; the children catch
    a restart one layer off, bounds read from another layer, and a wrong
    ReLU numbering of the restarted walk."""
    net = _deep_net()
    layers = extract_layers(net)
    X = np.array(samples(_LO, _HI, n=128, seed=1))
    root, meta = relaxed_reach(_box_star(_LO, _HI), layers, {},
                               bound_mode=bound_mode)
    first = _first_relaxed(meta)
    for L in (0, 2):
        for ph in (A, I):
            fixed = {first[L]: ph}
            child = resume_reach(root, layers, fixed, bound_mode, L)
            _assert_holds(child, net, X, fixed,
                          f"child at layer {L}, {ph.name}", min_points=12)
    star, fixed = root, {}
    for L in (0, 1, 2):
        key = _first_relaxed(star.relax_meta).get(L)
        if key is None:
            pytest.fail(f"setup: no relaxed neuron at layer {L} to split")
        fixed = {**fixed, key: _richer_phase(net, X[_in_phase(net, X, fixed)],
                                             key)}
        star = resume_reach(star, layers, fixed, bound_mode, L)
        _assert_holds(star, net, X, fixed, f"chain at layer {L}",
                      min_points=12)


_PAIR_CASES = [("box", "root-split-at-0", "active-first"),
               ("box", "fixed-at-0-split-at-1", "active-first"),
               ("dual_gpu", "root-split-at-0", "active-first"),
               ("dual_gpu", "root-split-at-0", "inactive-first"),
               ("dual_gpu", "fixed-at-0-split-at-1", "active-first")]


@pytest.mark.parametrize("bound_mode,parent_kind,order", _PAIR_CASES,
                         ids=["-".join(c) for c in _PAIR_CASES])
def test_both_children_of_a_pair_hold_their_phase_regions(
        bound_mode, parent_kind, order):
    """``resume_reach_pair`` (two delegated resumes under box; the lockstep
    walk with one population bound call per layer under dual_gpu): each
    child holds every true output of its own phase region, and so does a
    grandchild resumed from each child at a deeper layer. Catches a second
    child built with the first child's fixes or bounds, a restart numbered
    as ReLU layer 0 (the parent's layer-0 fix then lands on the split
    layer), the split layer's bounds reused deeper, and a child whose
    checkpoints are shifted by one."""
    net = _deep_net()
    layers = extract_layers(net)
    X = np.array(samples(_LO, _HI, n=128, seed=1))
    S = _box_star(_LO, _HI)
    _, meta = relaxed_reach(S, layers, {}, bound_mode=bound_mode)
    first = _first_relaxed(meta)
    if parent_kind == "root-split-at-0":
        base, L = {}, 0
    else:
        base, L = {first[0]: _richer_phase(net, X, first[0])}, 1
    parent, _ = relaxed_reach(S, layers, base, bound_mode=bound_mode)
    key = _first_relaxed(parent.relax_meta).get(L)
    if key is None:
        pytest.fail(f"setup: the parent has no relaxed neuron at layer {L}")
    phases = (A, I) if order == "active-first" else (I, A)
    fixed_pair = [{**base, key: ph} for ph in phases]
    children = resume_reach_pair(parent, layers, fixed_pair, bound_mode, L)
    assert len(children) == 2
    for child, fixed in zip(children, fixed_pair):
        assert child.fixed == fixed
        _assert_holds(child, net, X, fixed, f"pair child {fixed[key].name}",
                      min_points=12)
        deeper = _first_relaxed(child.relax_meta).get(2)
        if deeper is None:
            continue
        sub = X[_in_phase(net, X, fixed)]
        gfixed = {**fixed, deeper: _richer_phase(net, sub, deeper)}
        grandchild = resume_reach(child, layers, gfixed, bound_mode, 2)
        _assert_holds(grandchild, net, X, gfixed,
                      f"grandchild of the pair child {fixed[key].name}",
                      min_points=12)


# --------------------------------------------------------------------------- #
# The public route: verify_refine never proves a property the model violates   #
# --------------------------------------------------------------------------- #
_LO2, _HI2 = [-1.0, -2.0], [3.0, 2.0]


def _two_layer_net():
    net = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 4), nn.ReLU(),
                        nn.Linear(4, 2)).double()
    return _dyadic(net, 11, bits=3, span=10)


def _exact_max_y0(net):
    """The exact maximum of output 0 over the box, from the rational pieces
    of the network (every linear piece's image vertices)."""
    W = [m.weight.detach().numpy() for m in net if isinstance(m, nn.Linear)]
    b = [m.bias.detach().numpy() for m in net if isinstance(m, nn.Linear)]
    pieces = R.affine(R.box(_LO2, _HI2), W[0], b[0])
    pieces = R.affine(R.relu(pieces), W[1], b[1])
    pieces = R.affine(R.relu(pieces), W[2], b[2])
    return max(y[0] for p in pieces for y in p.image())


@pytest.mark.parametrize("bound_mode,incremental", [
    ("box", True), ("box", False), ("lp_cpu", True), ("dual_gpu", True)])
def test_verify_refine_never_proves_a_property_the_model_violates(
        bound_mode, incremental):
    """End to end through ``verify_refine``: the unsafe region ``y0 >=
    max - 1/64``, with ``max`` the exact maximum of y0 over the box (rational
    pieces), holds a true output, so UNSAT is wrong and a SAT must carry a
    counterexample in the box that the float64 model maps into the region.
    The root star is not decided, so the search splits through
    ``resume_reach`` (box, lp_cpu), ``resume_reach_pair`` (dual_gpu) or a
    reach from scratch (incremental off); a child that misses true outputs
    can prune the branch that holds the counterexample."""
    net = _two_layer_net()
    top = _exact_max_y0(net)
    spec = LinearSpec([[-1.0, 0.0]], [-float(top - Fraction(1, 64))])
    S = _box_star(_LO2, _HI2)
    res = verify_refine(S, net, spec, bound_mode=bound_mode,
                        prune_mode="exact", witness_mode="exact",
                        frontier_width=1, incremental=incremental,
                        node_budget=400)
    assert res.status != Status.UNSAT, "UNSAT for a property the model violates"
    if res.status == Status.SAT:
        x = np.asarray(res.counterexample_x, dtype=np.float64).ravel()
        assert np.all(x >= _LO2) and np.all(x <= _HI2), "counterexample outside the box"
        y = _forward(net, x.reshape(1, -1)).ravel()
        assert Fraction(float(y[0])) >= top - Fraction(1, 64), \
            f"counterexample output {y[0]} is not in the unsafe region"


# --------------------------------------------------------------------------- #
# 1. Conv-type layers: the star is the exact affine image                       #
# --------------------------------------------------------------------------- #
_IMG = (2, 3, 5)
# flat (channel, height, width) index -> interval; two input sets, so that one
# layer object is used twice
_VARY = ({7: (-2.5, 1.5), 19: (0.5, 3.0), 25: (-1.0, -0.25)},
         {0: (-1.0, 2.0), 14: (-3.0, -1.0), 21: (0.25, 0.75)})


def _image_bounds(varying):
    """Every pixel of a (2, 3, 5) image fixed at a dyadic value in
    [-1.5, 1.5], except the listed ones, each with its own interval."""
    lo = ((np.arange(30) * 5) % 13 - 6) / 4.0
    hi = lo.copy()
    for idx, (a, b) in varying.items():
        lo[idx], hi[idx] = a, b
    return lo, hi


def _corners(lo, hi):
    idx = np.flatnonzero(hi > lo)
    out = []
    for k in range(2 ** idx.size):
        x = lo.copy()
        for j, i in enumerate(idx):
            if k >> j & 1:
                x[i] = hi[i]
        out.append(x)
    return out


def _dyadic_bn(bn):
    """Running statistics and affine terms of a BatchNorm2d(2) chosen so
    that 1/sqrt(var + eps) is exact (var + eps in {1/4, 4})."""
    with torch.no_grad():
        bn.eps = 2.0 ** -8
        bn.running_mean.copy_(torch.tensor([0.5, -1.25]))
        bn.running_var.copy_(torch.tensor([0.25, 4.0]) - 2.0 ** -8)
        if bn.affine:
            bn.weight.copy_(torch.tensor([1.5, -0.75]))
            bn.bias.copy_(torch.tensor([0.375, -2.0]))
    return bn


_CONV_KINDS = {
    "conv2d-stride-padding-dilation-groups": lambda: _dyadic(nn.Conv2d(
        2, 4, (2, 3), stride=(1, 2), padding=(1, 2), dilation=(1, 2),
        groups=2), 7),
    "conv2d-without-bias": lambda: _dyadic(nn.Conv2d(2, 3, (3, 2),
                                                     bias=False), 8),
    "conv2d-reflect-padding": lambda: _dyadic(nn.Conv2d(
        2, 2, 3, padding=1, padding_mode="reflect"), 9),
    "conv-transpose2d": lambda: _dyadic(nn.ConvTranspose2d(
        2, 4, (2, 3), stride=2, padding=(1, 0), output_padding=(1, 1),
        groups=2), 10),
    "batchnorm2d": lambda: _dyadic_bn(nn.BatchNorm2d(2)),
    "batchnorm2d-without-affine": lambda: _dyadic_bn(nn.BatchNorm2d(
        2, affine=False)),
    "avgpool2d-padding": lambda: nn.AvgPool2d(2, stride=(1, 2), padding=1),
    "avgpool2d-divisor-override": lambda: nn.AvgPool2d(
        (3, 2), stride=2, divisor_override=2),
}


@pytest.mark.parametrize("kind", list(_CONV_KINDS))
def test_conv_type_layer_star_is_the_exact_affine_image(kind):
    """A conv-type layer is affine, so the relaxed reach of a box through it
    is exactly the image of the box: the vertices of the returned star (the
    images of its predicate polytope's vertices, in rationals) are exactly
    the float64 torch images of the box's corners, dyadic data throughout.
    Two input sets go through one layer object. Catches a bias (of a Conv2d,
    a ConvTranspose2d or a BatchNorm2d) or a BatchNorm2d running mean left in
    the map of the generator columns, a bias dropped from the anchor, and a
    layer object whose second use maps with the wrong module."""
    module = _CONV_KINDS[kind]()
    model = nn.Sequential(nn.Unflatten(1, _IMG), module,
                          nn.Flatten()).double().eval()
    layers = extract_layers(model)
    assert [type(layer) for layer in layers] == [ConvLayer]
    for varying in _VARY:
        lo, hi = _image_bounds(varying)
        out, _ = relaxed_reach(_box_star(lo, hi), layers)
        image = {tuple(Fraction(float(v)) for v in y)
                 for y in _forward(model, np.array(_corners(lo, hi)))}
        vertices = set(R.set_vertices(R.star(out)))
        assert vertices == image, (
            f"{len(vertices - image)} star vertices are not images of a "
            f"corner, {len(image - vertices)} corner images are not vertices")


def test_conv_layer_maps_more_than_1024_generator_columns():
    """``ConvLayer.apply`` maps the generator columns in chunks of 1024: a
    (1, 33, 34) image with every pixel free has 1122. For dyadic predicate
    points the star's point equals the float64 torch image of the input it
    stands for, exactly. Catches a chunk dropped, reordered or mapped
    twice."""
    conv = _dyadic(nn.Conv2d(1, 2, (2, 3), stride=(2, 1)), 12)
    model = nn.Sequential(nn.Unflatten(1, (1, 33, 34)), conv,
                          nn.Flatten()).double().eval()
    k = np.arange(33 * 34)
    lo = ((k * 7) % 11 - 5) / 4.0
    hi = lo + (k % 3 + 1) / 4.0
    star = _box_star(lo, hi)
    c, G = star.V[:, 0], star.V[:, 1:]
    if G.shape != (k.size, k.size) or np.count_nonzero(G - np.diag(np.diag(G))):
        pytest.fail("setup: the box star's generators are not one per pixel")
    out, _ = relaxed_reach(star, extract_layers(model))
    assert out.nVar == k.size
    rng = np.random.default_rng(0)
    for alpha in (np.ones(k.size), -np.ones(k.size),
                  np.where(k % 2, 1.0, -1.0),
                  rng.integers(-4, 5, size=k.size) / 4.0):
        y = _forward(model, (c + G @ alpha).reshape(1, -1)).ravel()
        assert np.array_equal(out.V[:, 0] + out.V[:, 1:] @ alpha, y)


# --------------------------------------------------------------------------- #
# 1. extract_layers: the layers compute the model                               #
# --------------------------------------------------------------------------- #
_LO3, _HI3 = np.array([-1.5, -0.5, 0.25]), np.array([1.0, 2.0, 1.75])


def _bn_chain():
    m = nn.Sequential(nn.Linear(3, 8), nn.Unflatten(1, (2, 2, 2)),
                      nn.BatchNorm2d(2), nn.Flatten(), nn.ReLU(),
                      nn.Linear(8, 2)).double()
    _dyadic(m, 21)
    _dyadic_bn(m[2])
    return m


def _conv_chain():
    m = nn.Sequential(
        nn.Linear(3, 16), nn.Unflatten(1, (1, 4, 4)),
        nn.Conv2d(1, 2, 3, padding=1, stride=(1, 2)), nn.ReLU(),
        nn.BatchNorm2d(2), nn.AvgPool2d((2, 1)), nn.Flatten(),
        nn.Unflatten(-1, (2, 2, 2)), nn.ConvTranspose2d(2, 1, 2, stride=2),
        nn.Flatten(), nn.ReLU(), nn.Linear(16, 2)).double()
    _dyadic(m, 22)
    _dyadic_bn(m[4])
    return m


def _shared_relu():
    act = nn.ReLU()
    return _dyadic(nn.Sequential(nn.Linear(3, 4), act, nn.Linear(4, 4), act,
                                 nn.Linear(4, 2)).double(), 23)


def _shared_linear():
    lin = nn.Linear(4, 4)
    return _dyadic(nn.Sequential(nn.Linear(3, 4), nn.ReLU(), lin, nn.ReLU(),
                                 lin, nn.ReLU(), nn.Linear(4, 2)).double(), 24)


_SHARED = known_bug(
    19, "extract_layers reads a Sequential through Module.children(), which "
        "yields a module instance once although the Sequential runs it at "
        "every position it holds it: a shared ReLU or Linear is left out of "
        "the layer list, the reach is about another network and "
        "verify_refine answers UNSAT for a property the model violates")

# (make the model, in_shape, the reference model, a refusal is a correct
#  answer)
_MODELS = {
    "linear-without-bias-and-dropout": (lambda: _dyadic(nn.Sequential(
        nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 4, bias=False), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(4, 2)).double(), 20), None),
    "nested-sequential": (lambda: _dyadic(nn.Sequential(
        nn.Sequential(nn.Linear(3, 4), nn.ReLU()),
        nn.Sequential(nn.Sequential(nn.Linear(4, 3)), nn.ReLU()),
        nn.Linear(3, 2)).double(), 25), None),
    "a-bare-linear-module": (lambda: _dyadic(nn.Linear(3, 2).double(), 26),
                             None),
    "conv-batchnorm-pool-flatten-unflatten-transpose": (_conv_chain, None),
    "conv-that-reshapes-then-pool": (lambda: _dyadic(nn.Sequential(
        nn.Linear(3, 16), nn.Unflatten(1, (2, 2, 4)),
        nn.Conv2d(2, 4, (1, 2), stride=(1, 2)), nn.AvgPool2d(2),
        nn.Flatten(), nn.Linear(4, 2)).double(), 27), None),
    "image-input-with-in-shape": (lambda: _dyadic(nn.Sequential(
        nn.Conv2d(2, 3, (2, 3)), nn.ReLU(), nn.Flatten(),
        nn.Linear(18, 2)).double(), 28), _IMG),
    "batchnorm-in-training-mode": (lambda: _bn_chain().train(), None),
    "float32-model": (lambda: _dyadic(nn.Sequential(
        nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)), 29), None),
}
_MARKED_MODELS = {
    "shared-relu-instance": (_shared_relu, None, _SHARED, AssertionError),
    "shared-linear-instance": (_shared_linear, None, _SHARED, AssertionError),
    "unflatten-size-minus-one": (lambda: _dyadic(nn.Sequential(
        nn.Linear(3, 8), nn.Unflatten(1, (2, -1, 2)), nn.Conv2d(2, 1, 1),
        nn.Flatten(), nn.ReLU(), nn.Linear(4, 1)).double(), 30), None,
        known_bug(23, "extract_layers tracks an nn.Unflatten size that holds "
                      "-1 as the shape itself, and the next conv-type layer "
                      "is built on a tensor of negative size: a torch "
                      "RuntimeError where the model torch runs is due an "
                      "answer or a clear refusal", raises=RuntimeError),
        RuntimeError),
    "linear-on-an-image-activation": (lambda: _dyadic(nn.Sequential(
        nn.Linear(3, 8), nn.Unflatten(1, (2, 2, 2)), nn.Linear(2, 3),
        nn.Flatten(), nn.ReLU(), nn.Linear(12, 1)).double(), 31), None,
        known_bug(23, "extract_layers accepts an nn.Linear after a (C, H, W) "
                      "activation, which torch applies along the last axis, "
                      "as a dense map of the flat vector; the reach then "
                      "fails inside Star.affine_map with a ValueError about "
                      "a column count instead of a refusal at extraction",
                  raises=ValueError), ValueError),
}


def _composition_params():
    out = [pytest.param(name, id=name) for name in _MODELS]
    out += [pytest.param(name, id=name, marks=spec[2])
            for name, spec in _MARKED_MODELS.items()]
    return out


@pytest.mark.parametrize("name", _composition_params())
def test_extracted_layers_compute_the_model(name):
    """Single points (stars with no predicate variable) reached through the
    extracted layers land exactly on the float64 torch output of the model
    (of its eval-mode function, for a model in training mode; of its float64
    copy, for a float32 model), on dyadic weights and inputs. Covers every
    module kind the extraction accepts: Linear with and without bias, ReLU,
    Dropout, nested Sequential, a bare module, Conv2d, BatchNorm2d,
    AvgPool2d, Flatten followed by Unflatten (dim -1), ConvTranspose2d, a
    conv that changes the shape before a pool, and an image-input Sequential
    with ``in_shape``. Catches a bare module read as no layer, the tracked
    shape not flattened or not updated, a BatchNorm2d copy left in training
    mode, and a module left out or read twice. For the marked models a
    refusal (TypeError or NotImplementedError) is a correct answer too."""
    if name in _MODELS:
        make, in_shape = _MODELS[name]
        refusal_ok = False
    else:
        make, in_shape = _MARKED_MODELS[name][:2]
        refusal_ok = True
    model = make()
    reference = copy.deepcopy(model).double().eval()
    try:
        layers = extract_layers(model, in_shape=in_shape)
    except (TypeError, NotImplementedError):
        if refusal_ok:
            return
        raise
    n_in = 30 if in_shape else 3
    lo, hi = ((_image_bounds(_VARY[0])[0] - 1.0, _image_bounds(_VARY[0])[0] + 1.0)
              if in_shape else (_LO3, _HI3))
    for x in samples(lo, hi, n=12, seed=4):
        out, _ = relaxed_reach(_box_star(x, x), layers)
        if out.nVar != 0:
            pytest.fail("setup: a point was reached as a set")
        xin = torch.as_tensor(x.reshape((1,) + tuple(in_shape)) if in_shape
                              else x.reshape(1, n_in))
        with torch.no_grad():
            y = reference(xin).numpy().ravel()
        assert np.array_equal(out.V[:, 0], y), (
            f"at x = {x.tolist()}: layers give {out.V[:, 0].tolist()}, the "
            f"model {y.tolist()}")


@_SHARED
def test_verify_refine_does_not_prove_a_property_a_shared_relu_model_violates():
    """x in [0, 2]; h = relu(x); (1 - h, h) through the SAME ReLU instance
    again. The true output at x = 2 is (0, 2), so the unsafe region
    {y1 + y2 >= 3/2} is reached; the layer list without the second ReLU
    computes (1 - h, h), whose sum is 1 everywhere, and the search answered
    UNSAT."""
    act = nn.ReLU()
    net = nn.Sequential(nn.Linear(1, 1), act, nn.Linear(1, 2), act,
                        nn.Linear(2, 2)).double()
    _set(net[0], [[1.0]], [0.0])
    _set(net[2], [[-1.0], [1.0]], [1.0, 0.0])
    _set(net[4], np.eye(2), [0.0, 0.0])
    y = _forward(net, [[2.0]]).ravel()
    if y.sum() < 1.5:
        pytest.fail(f"setup: the model gives {y.tolist()} at x = 2")
    spec = LinearSpec([[-1.0, -1.0]], [-1.5])
    res = verify_refine(_box_star([0.0], [2.0]), net, spec, **CPU_CONFIG)
    assert res.status != Status.UNSAT, "UNSAT for a property the model violates"


# --------------------------------------------------------------------------- #
# Rounding of the triangle's upper row                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("g,c", [
    pytest.param(1.0, 0.5, id="dyadic-slope-3/4"),
    pytest.param(1.375, 0.5, id="slope-15/22", marks=known_bug(
        19, "_relu_layer_relaxed stores the triangle's upper row (slope "
            "u/(u-l), its products with the generator row and its right-hand "
            "side) rounded to nearest with no outward slack, so with exact "
            "bounds the row can cut the vertex (u, u): the true output at "
            "x = 1 lies outside the star by 2^-52 in exact arithmetic (the "
            "core approx ReLU kernel raises the slope and pads the row)")),
])
def test_triangle_upper_row_holds_the_vertex_u_u_in_exact_arithmetic(g, c):
    """x in [-1, 1], hidden (g x + c, x + 2), identity output; the box
    bounds of the crossing neuron are exact (dyadic data), so its triangle's
    upper row passes exactly through the vertex (u, u), the true output at
    x = 1. Every true output at the corners and on the quarter grid must be
    a member of the star in rationals with zero slack. The slope 3/4 is a
    float; 15/22 is not."""
    net = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2)).double()
    _set(net[0], [[g], [1.0]], [c, 2.0])
    _set(net[2], np.eye(2), [0.0, 0.0])
    out, meta = relaxed_reach(_box_star([-1.0], [1.0]), extract_layers(net))
    if len(meta) != 1 or meta[0].u != g + c or meta[0].l != c - g:
        pytest.fail(f"setup: the crossing neuron's bounds are not exact: {meta}")
    member = R.Membership(R.star(out))
    for x in np.arange(-4, 5) / 4.0:
        y = [max(Fraction(g) * Fraction(x) + Fraction(c), Fraction(0)),
             Fraction(x) + 2]
        assert member(y), f"the true output {[float(v) for v in y]} at x = {x} is outside"

"""Soundness of branch and bound in ``n2v/nn/bab.py``: input-box bisection
(``verify_bab``, ``verify_bab_model``, with the split rule and the default
bound) and ReLU phase splitting (``verify_bab_relu`` and its node bound
``_seq_reach_relu``), and the branching score ``star_input_sensitivity``.

The promise these cells check (DESIGN section 13 and the module docstring):

- VERIFIED only when every subdomain of a cover of the input box was proved
  (input bisection), or every phase pattern of the split neurons was proved or
  shown infeasible (ReLU splitting): the two children of a split tile their
  parent, no subdomain is dropped, and a subdomain that can neither be proved
  nor split ends the search with UNKNOWN;
- FALSIFIED only with a counterexample that lies in the input box and violates
  the property when the model is replayed on onnxruntime at zero tolerance;
- UNKNOWN when the node budget or the time budget runs out, never the verdict
  of the last node; and the counters of ``BaBResult`` (nodes bounded, splits,
  depth of the deepest node) describe the search that ran.

Oracles, none of them n2v: the exact image of small ReLU models with dyadic
weights in rationals (``tests/_oracles/rational.py``), which gives the exact
extremes of the output over the box and the exact region of each phase
pattern; the bisection tree of a box, computed here in fractions; onnxruntime
replay of every witness, checked against the half-spaces of the property in
plain numpy. ``verify_bab`` takes its reach, bound, falsify and sensitivity
functions and its clock as arguments, so its search is checked exactly with
stand-ins written here.

Convention: the half-space ``G y <= g`` of a property is the UNSAFE region.
"""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper

from n2v.nn.bab import (_seq_reach_relu, star_input_sensitivity, verify_bab,
                        verify_bab_model, verify_bab_relu)
from n2v.nn.neural_network import NeuralNetwork
from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.sets.halfspace import HalfSpace
from n2v.utils.lpsolver import LPFailure
from n2v.utils.verify_specification import verify_specification
from tests._oracles import rational as R
from tests.cell_support import replay_onnx, save_onnx
from tests.known_bugs import DID_NOT_RAISE, known_bug

F = Fraction


# --------------------------------------------------------------------------- #
# Small ReLU models with dyadic weights: torch, onnx and the exact image       #
# --------------------------------------------------------------------------- #
class _Net:
    """A chain of affine layers with ReLU between them, given as dyadic
    ``(W, b)`` pairs, so float32, float64 and the rational image agree."""

    def __init__(self, *layers):
        self.layers = [(np.asarray(W, dtype=np.float64),
                        np.asarray(b, dtype=np.float64)) for W, b in layers]
        self.n_in = self.layers[0][0].shape[1]

    def torch(self, flatten_first=False):
        modules = [nn.Flatten()] if flatten_first else []
        for i, (W, b) in enumerate(self.layers):
            layer = nn.Linear(W.shape[1], W.shape[0])
            with torch.no_grad():
                layer.weight.copy_(torch.as_tensor(W, dtype=torch.float32))
                layer.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
            modules.append(layer)
            if i < len(self.layers) - 1:
                modules.append(nn.ReLU())
        return nn.Sequential(*modules).eval()

    def onnx(self, tmp_path):
        nodes, inits, h = [], {}, "X"
        for i, (W, b) in enumerate(self.layers):
            inits[f"W{i}"] = W.T.astype(np.float32)
            inits[f"b{i}"] = b.astype(np.float32)
            nodes.append(helper.make_node("MatMul", [h, f"W{i}"], [f"m{i}"]))
            last = i == len(self.layers) - 1
            out = "Y" if last else f"a{i}"
            nodes.append(helper.make_node("Add", [f"m{i}", f"b{i}"], [out]))
            if not last:
                nodes.append(helper.make_node("Relu", [out], [f"h{i}"]))
                h = f"h{i}"
        return save_onnx(tmp_path, nodes, [1, self.n_in],
                         [1, self.layers[-1][0].shape[0]], inits)

    def pieces(self, lb, ub):
        """The exact pieces of the network over the box."""
        pieces = R.box(lb, ub)
        for i, (W, b) in enumerate(self.layers):
            pieces = R.affine(pieces, W, b)
            if i < len(self.layers) - 1:
                pieces = R.relu(pieces)
        return pieces

    def probe(self):
        """The same network with the last hidden layer passed to the output
        beside y, so a wrong phase region shows in the output set."""
        W, b = self.layers[-1]
        k = W.shape[1]
        return _Net(*self.layers[:-1],
                    (np.vstack([np.eye(k), W]), np.concatenate([np.zeros(k), b])))

    def extremes(self, lb, ub):
        """Exact minimum and maximum of the single output over the box."""
        values = [y[0] for p in self.pieces(lb, ub) for y in p.image()]
        return min(values), max(values)

    def unsafe_vertices(self, lb, ub, t):
        """Vertices of {x in box : y(x) >= t}, exactly."""
        found = set()
        for p in self.pieces(lb, ub):
            A = p.A + [[-w for w in p.W[0]]]
            b = p.b + [p.c[0] - R.frac(t)]
            found.update(R.vertices(A, b))
        return sorted(found)


# y = relu(x0 + x1) + relu(x0 - x1) on [-1, 1]^2: the maximum 2 is reached on
# the whole edge x0 = 1; the triangle relaxation of the full box bounds it by 3.
_V = _Net(([[1, 1], [1, -1]], [0, 0]), ([[1, 1]], [0]))
# y = relu(x0) + relu(x1): the maximum 2 only at the corner (1, 1), and
# y >= 2 - 1/64 only where both neurons are active.
_CORNER = _Net(([[1, 0], [0, 1]], [0, 0]), ([[1, 1]], [0]))
# y = -(|x0 - 1/4| + |x1 + 1/2|): the maximum 0 only at the point (1/4, -1/2).
_TENT = _Net(([[1, 0], [-1, 0], [0, 1], [0, -1]], [-0.25, 0.25, 0.5, -0.5]),
             ([[-1, -1, -1, -1]], [0]))
# Two hidden layers: h = relu(x0 / 2 - x1 - 1/4, x0 / 2 + x1), z = relu(h1 - h0
# + 1/4, h1 - h0 / 2 + 1/4), y = z0 / 2 - z1: exact maximum 0. One relaxed
# reach does not prove y >= 1/64 unreachable; the split search needs neurons
# of both layers.
_DEEP = _Net(([[0.5, -1], [0.5, 1]], [-0.25, 0]),
             ([[-1, 1], [-0.5, 1]], [0.25, 0.25]),
             ([[0.5, -1]], [0]))
# Two hidden layers on which the split search meets a phase pattern that no
# input has: h = relu(x0 - x1/2 + 1/4, -x0 + x1/2 + 1/4), z = relu(h0 - h1 -
# 1/4, -h0 - h1/2 - 1/2), y = z0 + z1/2: exact maximum 3/2.
_PRUNE = _Net(([[1, -0.5], [-1, 0.5]], [0.25, 0.25]),
              ([[1, -1], [-1, -0.5]], [-0.25, -0.5]),
              ([[1, 0.5]], [0]))

_LB, _UB = np.array([-1.0, -1.0]), np.array([1.0, 1.0])


def _geq(t):
    """The unsafe region y >= t, as the half-space -y <= -t."""
    return HalfSpace(np.array([[-1.0]]), np.array([[-float(t)]]))


def _violates(y, groups):
    """The property read in plain numpy: ``groups`` is a list (AND) of lists
    (OR) of ``(G, g)``; the output is unsafe when every group has a member
    whose rows all hold, at zero tolerance."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return all(any(np.all(np.asarray(G, dtype=np.float64) @ y
                          <= np.asarray(g, dtype=np.float64).reshape(-1))
                   for G, g in group) for group in groups)


def _assert_witness(result, net, lb, ub, groups, tmp_path):
    """A FALSIFIED result carries an input inside [lb, ub] whose onnxruntime
    output violates the property at zero tolerance."""
    assert result.verdict == "FALSIFIED", result
    x = np.asarray(result.counterexample, dtype=np.float64).reshape(-1)
    lb, ub = np.asarray(lb, dtype=np.float64).reshape(-1), \
        np.asarray(ub, dtype=np.float64).reshape(-1)
    assert x.size == lb.size, f"the witness has {x.size} entries, the box {lb.size}"
    assert np.all(lb <= x) and np.all(x <= ub), f"witness {x} outside the box"
    y = replay_onnx(net.onnx(tmp_path), [x], [1, net.n_in])[0]
    assert _violates(y, groups), f"witness {x} gives {y}, which is safe"


def _single_shot(net, spec):
    out = NeuralNetwork(net.torch()).reach(Star.from_bounds(_LB, _UB),
                                           method="approx")
    return verify_specification(out, spec).verdict


def _needs_splitting(net, t):
    """Precondition of the safe instances: the exact maximum is below t and
    one approx reach of the whole box does not prove it."""
    _, top = net.extremes(_LB, _UB)
    if not top < t:
        pytest.fail(f"the exact maximum {top} is not below {t}")
    if _single_shot(net, _geq(t)) == "UNSAT":
        pytest.fail("one reach of the whole box proves it: no split is needed")


def _answers_or_refuses(call, allowed):
    """A combination whose support is promised either answers with one of the
    allowed verdicts or refuses with the declared NotImplementedError; any
    other exception is the defect the marker of the calling test names."""
    try:
        result = call()
    except NotImplementedError:
        return
    assert result.verdict in allowed, result


# --------------------------------------------------------------------------- #
# verify_bab with stand-ins: the bisection tree, exactly                       #
# --------------------------------------------------------------------------- #
class _Sub:
    """What the stand-in reach function returns: the subdomain it was given."""

    def __init__(self, lb, ub):
        self.lb, self.ub = lb, ub


class _Search:
    """Stand-in reach, bound and falsify functions that record every call.

    A subdomain is proved when it does not contain the point ``p`` (closed
    box) or, if ``w_min`` is given, when its widest side is at most
    ``w_min``. The falsifier, when asked for, finds ``p`` in a subdomain
    whose widest side is at most ``find_at`` and finds nothing elsewhere."""

    def __init__(self, shape, p, w_min=None, find_at=None):
        self.shape, self.p, self.w_min, self.find_at = shape, p, w_min, find_at
        self.bounded, self.proved, self.falsified, self.shapes = [], [], [], []

    @staticmethod
    def _box(lb, ub):
        return (tuple(F(float(v)) for v in np.ravel(lb)),
                tuple(F(float(v)) for v in np.ravel(ub)))

    def _holds_p(self, box):
        return all(lo <= v <= hi for lo, v, hi in zip(box[0], self.p, box[1]))

    def reach(self, lb, ub):
        self.shapes.append((np.shape(lb), np.shape(ub)))
        box = self._box(lb, ub)
        self.bounded.append(box)
        return [_Sub(*box)]

    def bound(self, sets, spec):
        assert spec == "spec", "the bound function got another spec"
        box = (sets[0].lb, sets[0].ub)
        widest = max(hi - lo for lo, hi in zip(*box))
        ok = not self._holds_p(box) or (self.w_min is not None
                                       and widest <= self.w_min)
        if ok:
            self.proved.append(box)
        return ok

    def falsify(self, lb, ub):
        self.shapes.append((np.shape(lb), np.shape(ub)))
        box = self._box(lb, ub)
        self.falsified.append(box)
        widest = max(hi - lo for lo, hi in zip(*box))
        if self.find_at is not None and self._holds_p(box) \
                and widest <= self.find_at:
            return np.array([float(v) for v in self.p]).reshape(self.shape)
        return None


def _pick(lo, hi, sens):
    """The documented split rule: the widest side for branch='widest' (sens
    None), else the largest sensitivity times width over the sides of nonzero
    width; the first on ties."""
    widths = [h - l for l, h in zip(lo, hi)]
    if sens is None or not any(w > 0 for w in widths):
        return max(range(len(widths)), key=lambda d: (widths[d], -d))
    scores = [F(s) * w if w > 0 else None for s, w in zip(sens, widths)]
    best = max(s for s in scores if s is not None)
    return next(d for d, s in enumerate(scores) if s == best)


def _tree(lo, hi, proved, sens):
    """Every subdomain of the bisection tree of [lo, hi], with its depth."""
    out = []

    def visit(lo, hi, depth):
        out.append(((lo, hi), depth))
        if proved((lo, hi)):
            return
        d = _pick(lo, hi, sens)
        mid = (lo[d] + hi[d]) / 2
        visit(lo, hi[:d] + (mid,) + hi[d + 1:], depth + 1)
        visit(lo[:d] + (mid,) + lo[d + 1:], hi, depth + 1)
    visit(tuple(lo), tuple(hi), 0)
    return out


def _volume(box, active):
    return math.prod((box[1][d] - box[0][d]) for d in active)


# (lb, ub, p, w_min, branch, sensitivity, falsifier): a different interval per
# coordinate, bounds beyond +-1, a zero-width coordinate, every rule of the
# split, with and without a falsifier that finds nothing.
_TREES = [
    pytest.param([-3], [5], [1.25], 0.5, "widest", None, False,
                 id="1d-widest"),
    pytest.param([-3, 0.5], [5, 0.75], [-1.75, 0.625], 0.25, "widest", None,
                 True, id="2d-widest-falsifier-finds-nothing"),
    pytest.param([-3, 0.5, -10], [5, 0.5, 6], [1.25, 0.5, -2.5], 1.0,
                 "sensitivity", [1, 64, 0.25], True,
                 id="3d-sensitivity-times-width-falsifier-finds-nothing"),
    pytest.param([0.5, -3, 2], [0.5, 5, 2], [0.5, 1.25, 2], 0.5,
                 "sensitivity", [5, 0, 0], False,
                 id="3d-zero-width-first-zero-scores"),
    pytest.param([-3, 0.5, -10], [5, 0.75, 6], [1.25, 0.625, -2.5], 1.0,
                 "widest", [64, 1, 0.25], False,
                 id="3d-widest-ignores-sensitivity"),
    pytest.param([-3, 0.5, -10], [5, 0.75, 6], [1.25, 0.625, -2.5], 1.0,
                 "sensitivity", "no-function", False,
                 id="3d-no-sensitivity-function"),
    pytest.param([-3, 0.5, -10], [5, 0.75, 6], [1.25, 0.625, -2.5], 1.0,
                 "sensitivity", "returns-none", False,
                 id="3d-sensitivity-returns-none"),
]


@pytest.mark.parametrize("lb, ub, p, w_min, branch, sens, with_falsifier",
                         _TREES)
def test_bisection_leaves_tile_the_box_and_the_counters_match_the_tree(
        lb, ub, p, w_min, branch, sens, with_falsifier):
    """``verify_bab`` with a bound that proves a subdomain when it misses the
    point p or is narrow: the subdomains it bounds are exactly those of the
    bisection tree computed here in fractions (halve the side the documented
    rule picks), the proved leaves are inside the box, do not overlap and
    their volumes add up to the box's, and nodes, splits and max_depth are
    those of the tree, with nodes = 2 splits + 1 for a finished search.
    Catches a child box built from the wrong corner, a dropped child, a slice
    lost at the split point, a split of a zero-width side, the branch rule
    misread (widest versus sensitivity times width, a missing sensitivity),
    a falsifier's None read as a counterexample, a left child pushed at its
    parent's depth and the nodes and splits fields swapped."""
    p = [F(v) for v in p]
    search = _Search(np.shape(lb), p, w_min=F(w_min))
    if sens == "no-function":
        sens_fn, rule = None, None
    elif sens == "returns-none":
        sens_fn, rule = (lambda sets, spec, n: None), None
    else:
        def sens_fn(sets, spec, n):
            assert n == len(lb)
            return np.array(sens, dtype=np.float64)
        rule = None if branch == "widest" else sens
    result = verify_bab(search.reach, np.array(lb, float), np.array(ub, float),
                        "spec", bound_fn=search.bound, sensitivity_fn=sens_fn,
                        falsify_fn=search.falsify if with_falsifier else None,
                        branch=branch, max_nodes=10_000)
    root = (tuple(F(v) for v in lb), tuple(F(v) for v in ub))
    tree = _tree(*root, lambda box: not search._holds_p(box) or max(
        hi - lo for lo, hi in zip(*box)) <= F(w_min), rule)
    assert result.verdict == "VERIFIED", result
    assert result.counterexample is None
    # Soundness: the proved leaves tile the box.
    active = [d for d in range(len(lb)) if ub[d] > lb[d]]
    for box in search.bounded:
        assert all(root[0][d] <= box[0][d] <= box[1][d] <= root[1][d]
                   for d in range(len(lb))), f"{box} is not inside the box"
    leaves = search.proved
    assert sum(_volume(b, active) for b in leaves) == _volume(root, active)
    for i, a in enumerate(leaves):
        for b in leaves[i + 1:]:
            assert any(min(a[1][d], b[1][d]) <= max(a[0][d], b[0][d])
                       for d in active), f"{a} and {b} overlap"
    # The search that ran is the documented one.
    assert sorted(search.bounded) == sorted(box for box, _ in tree)
    splits = sum(1 for box, _ in tree if box not in search.proved)
    assert (result.nodes, result.splits, result.max_depth) == (
        len(tree), splits, max(depth for _, depth in tree))
    assert result.nodes == 2 * result.splits + 1
    if with_falsifier:
        unproved = [box for box in search.bounded if box not in search.proved]
        assert sorted(search.falsified) == sorted(unproved)


def test_bab_hands_the_stand_ins_the_bounds_in_their_own_shape():
    """Bounds given as a (2, 3) array are split flat but handed to the reach
    and falsify functions in the caller's shape, and the counterexample comes
    back as the falsifier returned it. Catches the flat vector handed on."""
    lb = np.array([[-3.0, 0.5, -1.0], [0.0, 2.0, -4.0]])
    ub = np.array([[5.0, 0.75, 1.0], [0.25, 2.0, 4.0]])
    p = [F(v) for v in (1.25, 0.625, 0.5, 0.125, 2.0, -2.5)]
    search = _Search(lb.shape, p, find_at=F(1))
    result = verify_bab(search.reach, lb, ub, "spec", bound_fn=search.bound,
                        falsify_fn=search.falsify, sensitivity_fn=None,
                        max_nodes=10_000)
    assert set(search.shapes) == {((2, 3), (2, 3))}
    _assert_found_p(result, search, p, lb.shape)


def _assert_found_p(result, search, p, shape):
    assert result.verdict == "FALSIFIED", result
    assert np.shape(result.counterexample) == shape
    assert [F(float(v)) for v in np.ravel(result.counterexample)] == p
    last = search.falsified[-1]
    assert search._holds_p(last) and max(
        hi - lo for lo, hi in zip(*last)) <= search.find_at
    assert result.nodes == len(search.bounded)


@pytest.mark.parametrize("p", [
    pytest.param([1.25, -2.5], id="interior"),
    pytest.param([1.0, -2.0], id="on-the-first-split-planes"),
    pytest.param([5.0, 6.0], id="upper-corner"),
    pytest.param([-3.0, -10.0], id="lower-corner"),
    pytest.param([5.0, -10.0], id="mixed-corner"),
])
def test_a_point_that_is_never_proved_keeps_the_search_from_verified(p):
    """A bound that proves every subdomain except those that hold p: no
    finite search may answer VERIFIED; after max_nodes subdomains it answers
    UNKNOWN, having bounded exactly max_nodes of them. p sits inside, on the
    first two split planes and on corners, so a lost child, a child built
    from the wrong corner or a lost slice drops it somewhere. Catches the
    budget answered VERIFIED, max_nodes + 1 subdomains bounded, and (with a
    falsifier that finds p in small subdomains) a counterexample other than
    the one the falsifier returned."""
    p = [F(v) for v in p]
    lb, ub = np.array([-3.0, -10.0]), np.array([5.0, 6.0])
    for falsify in (False, True):
        search = _Search((2,), p, find_at=F(1, 4) if falsify else None)
        result = verify_bab(search.reach, lb, ub, "spec",
                            bound_fn=search.bound, sensitivity_fn=None,
                            falsify_fn=search.falsify if falsify else None,
                            max_nodes=60)
        if not falsify:
            assert result.verdict == "UNKNOWN", result
            assert result.nodes == 60 == len(search.bounded)
        else:
            _assert_found_p(result, search, p, (2,))


@pytest.mark.parametrize("lb, ub", [
    pytest.param([0.25, -0.5], [0.25, -0.5], id="a-point"),
    pytest.param([1.0, -0.5], [float(np.nextafter(1.0, 2.0)), -0.5],
                 id="one-ulp-wide"),
])
def test_a_subdomain_that_cannot_be_split_ends_unknown(lb, ub):
    """A box that holds p and cannot be bisected (zero or one-ulp width):
    the bound cannot prove it and no split can, so the answer is UNKNOWN
    after the one node. Catches such a subdomain skipped, which answers
    VERIFIED for a problem it never decided."""
    p = [F(v) for v in lb]
    search = _Search((2,), p)
    result = verify_bab(search.reach, np.array(lb), np.array(ub), "spec",
                        bound_fn=search.bound, max_nodes=1000)
    assert result.verdict == "UNKNOWN", result
    assert result.nodes == 1


def test_node_budget_zero_bounds_nothing():
    """max_nodes=0 answers UNKNOWN without calling the reach function."""
    search = _Search((2,), [F(0), F(0)])
    result = verify_bab(search.reach, np.array([-1.0, -1.0]),
                        np.array([1.0, 1.0]), "spec", bound_fn=search.bound,
                        max_nodes=0)
    assert result.verdict == "UNKNOWN" and result.nodes == 0
    assert search.bounded == []


class _Clock:
    """An injected clock far from zero that advances by one per subdomain
    bounded: it reads ``start + (reach calls so far)``."""

    def __init__(self, search, start=1.0e6):
        self.search, self.start = search, start

    def __call__(self):
        return self.start + len(self.search.bounded)


def test_time_budget_ends_unknown_after_the_nodes_it_allows():
    """With a clock that advances one unit per subdomain and timeout_s=2.5,
    the search bounds three subdomains (elapsed 0, 1 and 2 are within the
    budget, 3 is not) and answers UNKNOWN, never the VERIFIED a full search
    of this problem gives; time_s is the elapsed time. The clock starts at
    1e6, so a check of the clock's absolute reading instead of the elapsed
    time stops the search before the first node."""
    search = _Search((2,), [F(1, 4), F(-1, 2)], w_min=F(1, 8))
    result = verify_bab(search.reach, np.array([-1.0, -1.0]),
                        np.array([1.0, 1.0]), "spec", bound_fn=search.bound,
                        max_nodes=10_000, timeout_s=2.5,
                        _clock=_Clock(search))
    assert result.verdict == "UNKNOWN", result
    assert result.nodes == 3 == len(search.bounded)
    assert result.time_s == 3.0
    full = _Search((2,), [F(1, 4), F(-1, 2)], w_min=F(1, 8))
    assert verify_bab(full.reach, np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
                      "spec", bound_fn=full.bound).verdict == "VERIFIED"


def test_a_generous_time_budget_on_the_real_clock_does_not_cut_the_search():
    """timeout_s=600 with the real clock, on a search that takes far less:
    the answer is the VERIFIED of the full search, also with verbose=True.
    Catches the budget compared with the clock's reading (seconds since
    boot) instead of the elapsed time."""
    search = _Search((2,), [F(1, 4), F(-1, 2)], w_min=F(1, 8))
    result = verify_bab(search.reach, np.array([-1.0, -1.0]),
                        np.array([1.0, 1.0]), "spec", bound_fn=search.bound,
                        timeout_s=600.0, verbose=True)
    assert result.verdict == "VERIFIED", result


@known_bug(19, "verify_bab accepts any value of branch and runs the "
               "sensitivity rule for a value it does not know",
           raises=DID_NOT_RAISE)
def test_an_unknown_branch_rule_is_refused():
    """DESIGN rule 4: unknown options raise. branch='widset' (a misspelt
    'widest') must raise ValueError instead of running another rule."""
    search = _Search((2,), [F(1, 4), F(-1, 2)], w_min=F(1, 8))
    with pytest.raises(ValueError):
        verify_bab(search.reach, np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
                   "spec", bound_fn=search.bound, branch="widset")


# --------------------------------------------------------------------------- #
# verify_bab with the default bound and real reach sets                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", ["as-returned", "reversed"])
def test_default_bound_reads_every_set_of_an_exact_reach(order):
    """The default bound with the several stars of an exact reach of
    y = relu(x0) + relu(x1): y >= 2 - 1/64 is reached only where both neurons
    are active, which is one of the stars. Whatever the order of the stars,
    the search must not answer VERIFIED. Catches a bound that reads only the
    first set of the list."""
    net = _CORNER.torch()
    t = F(2) - F(1, 64)
    if not all(v[0] > 0 and v[1] > 0 for v in _CORNER.unsafe_vertices(
            _LB, _UB, t)):
        pytest.fail("the unsafe region needs both neurons active")
    nn_ = NeuralNetwork(net)
    if len(nn_.reach(Star.from_bounds(_LB, _UB), method="exact")) < 2:
        pytest.fail("the exact reach of the box did not split")

    def reach_fn(lb, ub):
        sets = nn_.reach(Star.from_bounds(lb, ub), method="exact")
        return sets if order == "as-returned" else sets[::-1]
    result = verify_bab(reach_fn, _LB, _UB, _geq(t), max_nodes=40)
    assert result.verdict == "UNKNOWN", result


# --------------------------------------------------------------------------- #
# star_input_sensitivity                                                      #
# --------------------------------------------------------------------------- #
# Three inputs on [-3, 1] x [0.5, 0.75] x [-1, 3] (radii 2, 1/8, 2). Hidden
# a = 8 + x0 - 2 x1 + x2 >= 2.5 (stable active), b = -8 + x0 + x1 + x2 <=
# -3.25 (stable inactive), c = x0 - x2 in [-6, 2] (crossing, relaxed with a
# predicate variable of its own). Outputs y = W2 relu(a, b, c).
_S_LB, _S_UB = np.array([-3.0, 0.5, -1.0]), np.array([1.0, 0.75, 3.0])
_S_NET = _Net(([[1, -2, 1], [1, 1, 1], [1, 0, -1]], [8, -8, 0]),
              ([[1, 4, -1], [-2, 1, 0.5]], [0.5, -1]))
# The offsets make both half-spaces reached everywhere (the score reads only
# the rows), so no search proves the root.
_S_H1 = HalfSpace(np.array([[1.0, 0.0], [0.0, -1.0]]),
                  np.array([[100.0], [100.0]]))
_S_H2 = HalfSpace(np.array([[-1.0, -1.0]]), np.array([[100.0]]))


def _sensitivity_by_hand(radii):
    """|h . (W2 D W1)[:, j]| * r_j, maximal over the rows h of both
    half-spaces; D keeps the stable active neuron a only (b is inactive and
    the relaxed c carries no input generator)."""
    W1, _ = _S_NET.layers[0]
    W2, _ = _S_NET.layers[1]
    M = [[F(W2[k][0]) * F(W1[0][j]) * F(radii[j]) for j in range(3)]
         for k in range(2)]
    rows = [[1, 0], [0, -1], [-1, -1]]
    return [max(abs(sum(F(h[k]) * M[k][j] for k in range(2))) for h in rows)
            for j in range(3)]


@pytest.mark.parametrize("spec", [
    pytest.param([_S_H1, _S_H2], id="list"),
    pytest.param((_S_H1, _S_H2), id="tuple"),
    pytest.param({"Hg": [_S_H1, _S_H2]}, id="dict"),
    pytest.param([{"Hg": _S_H1}, {"Hg": [_S_H2]}], id="list-of-dicts"),
])
def test_input_sensitivity_is_the_largest_spec_row_coefficient(spec):
    """The score of input j is the largest magnitude of a spec row times the
    output's generator column of input predicate j (column j + 1 of V, after
    the centre), computed by hand from the weights: (4, 1/2, 4). Every spec
    form names the same two half-spaces. Catches the centre column read as
    input 0, the magnitude dropped (row coefficient -1/2 on input 1), a spec
    form whose half-spaces are not found, and the maximum over half-spaces
    replaced by the last one."""
    out = NeuralNetwork(_S_NET.torch()).reach(Star.from_bounds(_S_LB, _S_UB),
                                              method="approx")
    if [s.nVar for s in out] != [4]:
        pytest.fail(f"expected one star with 3 input and 1 relaxation "
                    f"variable, got {[s.nVar for s in out]}")
    want = _sensitivity_by_hand((2, F(1, 8), 2))
    assert want == [4, F(1, 2), 4]
    got = star_input_sensitivity(out, spec, 3)
    assert [F(float(v)) for v in got] == want


def test_input_sensitivity_takes_the_maximum_over_reach_sets():
    """Two reach sets, the second with twice the generators: the score is
    that of the larger. Catches the maximum over sets replaced by the last
    or the first set."""
    out = NeuralNetwork(_S_NET.torch()).reach(Star.from_bounds(_S_LB, _S_UB),
                                              method="approx")[0]
    big = Star(out.V * 2.0, out.C, out.d, out.predicate_lb, out.predicate_ub)
    want = [2 * v for v in _sensitivity_by_hand((2, F(1, 8), 2))]
    for sets in ([out, big], [big, out]):
        got = star_input_sensitivity(sets, [_S_H1, _S_H2], 3)
        assert [F(float(v)) for v in got] == want


@pytest.mark.parametrize("sets", [
    pytest.param("box", id="a-box-among-the-sets"),
    pytest.param("too-few-predicates", id="fewer-predicates-than-inputs"),
])
def test_input_sensitivity_declines_sets_it_cannot_read(sets):
    """A reach set that is not a Star (first or after a Star), or a Star with
    fewer predicate variables than inputs, has no input-predicate columns:
    the answer is None (the caller then splits the widest side). Catches
    such sets skipped, which scores the inputs from the other sets only, or
    from nothing."""
    star = Star.from_bounds(np.zeros(2), np.ones(2))
    spec = HalfSpace(np.array([[1.0, -2.0]]), np.array([[0.0]]))
    if sets == "box":
        box = Box(np.zeros(2), np.ones(2))
        for given in ([star, box], [box, star]):
            assert star_input_sensitivity(given, spec, 2) is None
    else:
        assert star_input_sensitivity([star], spec, 3) is None


@known_bug(19, "star_input_sensitivity reads columns 1..n of V as the n "
               "inputs, but Star.from_bounds gives a zero-width input no "
               "predicate, so with a relaxation predicate behind them the "
               "scores belong to other inputs (unit harm: the split side)")
def test_input_sensitivity_with_a_zero_width_input_scores_the_right_inputs():
    """x1 fixed at 5/8 (zero width): its score does not matter (it is never
    split), but inputs 0 and 2 must keep their own scores, 4 and 4 by hand,
    or the function must answer None. Today the columns shift by one and
    input 2 is scored with the relaxation variable of neuron c (1)."""
    lb, ub = _S_LB.copy(), _S_UB.copy()
    lb[1] = ub[1] = 0.625
    out = NeuralNetwork(_S_NET.torch()).reach(Star.from_bounds(lb, ub),
                                              method="approx")
    got = star_input_sensitivity(out, [_S_H1, _S_H2], 3)
    want = _sensitivity_by_hand((2, 0, 2))
    assert got is None or [F(float(got[0])), F(float(got[2]))] == \
        [want[0], want[2]], (got, want)


def test_verify_bab_model_hands_the_sensitivity_the_root_reach_and_input_count():
    """Through verify_bab_model with branch='sensitivity': the sensitivity
    function is called with the reach sets of the box and the number of
    inputs, and returns the hand-computed scores of the root. Catches the
    reach of another box or a wrong input count handed on."""
    calls = []

    def recording(sets, spec, n):
        score = star_input_sensitivity(sets, spec, n)
        calls.append((n, score))
        return score
    verify_bab_model(_S_NET.torch(), _S_LB, _S_UB, [_S_H1, _S_H2],
                     falsify_method=None, sensitivity_fn=recording,
                     max_nodes=1)
    assert calls, "the sensitivity function was never called"
    n, score = calls[0]
    assert n == 3
    assert [F(float(v)) for v in score] == _sensitivity_by_hand((2, F(1, 8), 2))


# --------------------------------------------------------------------------- #
# verify_bab_model end to end                                                 #
# --------------------------------------------------------------------------- #
_SET_TYPES = [pytest.param(None, id="default-star"),
              pytest.param(Zono, id="zono"),
              pytest.param(Hexatope, id="hexatope"),
              pytest.param(Octatope, id="octatope")]


@pytest.mark.parametrize("set_type, branch", [
    pytest.param(None, "widest", id="default-star-widest"),
    pytest.param(None, "sensitivity", id="default-star-sensitivity"),
    pytest.param(Zono, "sensitivity", id="zono-sensitivity-falls-back"),
    pytest.param(Hexatope, "widest", id="hexatope"),
    pytest.param(Octatope, "widest", id="octatope"),
])
def test_model_bab_proves_a_property_safe_by_a_small_margin(set_type, branch):
    """y = relu(x0 + x1) + relu(x0 - x1) has exact maximum 2 on [-1, 1]^2;
    y >= 2 + 1/64 is unreachable but one approx Star reach of the whole box
    does not prove it. The search proves it, with nodes = 2 splits + 1, and
    the falsifier (random, seeded) finds nothing. On a Zono the sensitivity
    rule has no Star to read and falls back to the widest side. Catches a
    falsifier's None read as a counterexample, a node count swapped with
    the split count, and a search option that is not handed on."""
    t = F(2) + F(1, 64)
    _needs_splitting(_V, t)
    result = verify_bab_model(_V.torch(), _LB, _UB, _geq(t), set_type=set_type,
                              branch=branch, falsify_method="random",
                              falsify_kwargs={"seed": 0, "n_samples": 64},
                              max_nodes=200)
    assert result.verdict == "VERIFIED", result
    assert result.nodes == 2 * result.splits + 1


@pytest.mark.parametrize("set_type", _SET_TYPES)
def test_model_bab_never_proves_a_property_reached_in_a_small_region(set_type):
    """y >= 2 - 1/64 holds on a strip along x0 = 1 (exactly: the region has
    vertices). Without a falsifier the search must end UNKNOWN. Catches a
    reach of the subdomain's lower corner only, a dropped or shrunken child,
    and a bound that prunes what it did not prove."""
    t = F(2) - F(1, 64)
    if not _V.unsafe_vertices(_LB, _UB, t):
        pytest.fail("the unsafe region is empty")
    result = verify_bab_model(_V.torch(), _LB, _UB, _geq(t), set_type=set_type,
                              falsify_method=None, max_nodes=20)
    assert result.verdict == "UNKNOWN", result


def test_model_bab_with_exact_reach_never_proves_a_reached_property():
    """reach_method='exact' gives several stars per subdomain; y >= 2 - 1/64
    on y = relu(x0) + relu(x1) is reached only in the star where both neurons
    are active. Catches a bound that reads the first star only."""
    t = F(2) - F(1, 64)
    result = verify_bab_model(_CORNER.torch(), _LB, _UB, _geq(t),
                              reach_method="exact", falsify_method=None,
                              max_nodes=30)
    assert result.verdict == "UNKNOWN", result


def test_model_bab_with_exact_reach_proves_at_the_root():
    """With reach_method='exact' the first reach is the exact image, so the
    property safe by 1/64 is proved at the root: one node, no split. Catches
    reach_method not handed to the reach."""
    t = F(2) + F(1, 64)
    _needs_splitting(_V, t)
    result = verify_bab_model(_V.torch(), _LB, _UB, _geq(t),
                              reach_method="exact", falsify_method=None)
    assert (result.verdict, result.nodes, result.splits) == ("VERIFIED", 1, 0)


@pytest.mark.parametrize("net, t, method", [
    pytest.param(_V, F(2) - F(1, 64), "random+pgd", id="strip-random-pgd"),
    pytest.param(_CORNER, F(2) - F(1, 64), "random+pgd", id="corner-random-pgd"),
    pytest.param(_V, F(1), "random", id="quarter-of-the-box-random"),
])
def test_model_bab_witness_is_a_real_counterexample(net, t, method, tmp_path):
    """A FALSIFIED answer carries an input inside the box whose onnxruntime
    output is at least t. Catches the output half of the falsifier's pair
    returned as the witness, the falsifier run on a widened box, and a
    witness that is not the falsifier's."""
    result = verify_bab_model(net.torch(), _LB, _UB, _geq(t),
                              falsify_method=method,
                              falsify_kwargs={"seed": 0, "n_restarts": 2,
                                              "n_steps": 50},
                              max_nodes=60)
    _assert_witness(result, net, _LB, _UB, [[([[-1.0]], [-float(t)])]],
                    tmp_path)


def test_model_bab_at_a_single_unsafe_point_is_never_verified(tmp_path):
    """y = -(|x0 - 1/4| + |x1 + 1/2|) reaches y >= 0 at the one point
    (1/4, -1/2) (exactly: the unsafe region's only vertex). The search
    answers UNKNOWN, or FALSIFIED with that point replayed. Catches any
    dropped subdomain: the point lies on split planes of the bisection, so
    every lost child or slice near it shows."""
    vertices = _TENT.unsafe_vertices(_LB, _UB, 0)
    if vertices != [(F(1, 4), F(-1, 2))]:
        pytest.fail(f"the unsafe region is not the one point: {vertices}")
    for method in (None, "random"):
        result = verify_bab_model(_TENT.torch(), _LB, _UB, _geq(0),
                                  falsify_method=method,
                                  falsify_kwargs={"seed": 0, "n_samples": 64},
                                  max_nodes=60)
        assert result.verdict in ("UNKNOWN", "FALSIFIED"), result
        if result.verdict == "FALSIFIED":
            _assert_witness(result, _TENT, _LB, _UB, [[([[-1.0]], [0.0])]],
                            tmp_path)


def test_model_bab_hands_its_search_options_to_the_search():
    """max_nodes=3 and an injected clock given to verify_bab_model reach the
    search: the safe-by-1/64 problem, which needs more than three nodes,
    ends UNKNOWN after exactly three; with timeout_s=1.5 and a clock that
    advances one unit per call it ends UNKNOWN before the full search."""
    t = F(2) + F(1, 64)
    result = verify_bab_model(_V.torch(), _LB, _UB, _geq(t),
                              falsify_method=None, max_nodes=3)
    assert (result.verdict, result.nodes) == ("UNKNOWN", 3), result
    ticks = iter(range(10_000))
    result = verify_bab_model(_V.torch(), _LB, _UB, _geq(t),
                              falsify_method=None, timeout_s=1.5,
                              _clock=lambda: 1.0e6 + next(ticks))
    assert result.verdict == "UNKNOWN", result
    assert 1 <= result.nodes < 9


def test_model_bab_with_a_leading_flatten_and_a_shaped_box(tmp_path):
    """Bounds of shape (1, 2) and a model that starts with Flatten: the reach
    gets a flat set of each sub-box and the falsifier the sub-box in its own
    shape. The safe instance is VERIFIED and the strip instance FALSIFIED
    with a witness in the box that onnxruntime replays as unsafe. Catches
    the bounds handed to the set constructor unflattened (Star.from_bounds
    refuses a (1, 2) array) and a witness from outside the box."""
    model = _V.torch(flatten_first=True)
    lb, ub = _LB.reshape(1, 2), _UB.reshape(1, 2)
    assert verify_bab_model(model, lb, ub, _geq(F(2) + F(1, 64)),
                            falsify_method=None).verdict == "VERIFIED"
    t = F(2) - F(1, 64)
    result = verify_bab_model(model, lb, ub, _geq(t),
                              falsify_kwargs={"seed": 0, "n_restarts": 2,
                                              "n_steps": 50},
                              max_nodes=60)
    _assert_witness(result, _V, lb, ub, [[([[-1.0]], [-float(t)])]], tmp_path)


def test_model_bab_hands_its_falsifier_options_to_the_falsifier():
    """falsify_kwargs={'n_samples': 0} leaves the random falsifier nothing to
    try, so y >= 1, which holds on a quarter of the box, is not falsified and
    the budget ends the search; with the default 500 samples it is falsified
    at the root. Catches falsify_kwargs dropped."""
    for kwargs, want in (({"seed": 0, "n_samples": 0}, "UNKNOWN"),
                         ({"seed": 0}, "FALSIFIED")):
        result = verify_bab_model(_V.torch(), _LB, _UB, _geq(1),
                                  falsify_method="random",
                                  falsify_kwargs=kwargs, max_nodes=5)
        assert result.verdict == want, (kwargs, result)


def _assert_bisection_tree(boxes, root):
    """The boxes (Fractions, in call order, the root first) form a full
    bisection tree of the root: each later box is one half of an earlier box
    along one side, the halves of every split box both appear, so the
    leaves tile the root."""
    assert boxes and boxes[0] == root, f"the first box {boxes[:1]} is not {root}"
    halves = {}
    for i, box in enumerate(boxes[1:], start=1):
        parents = []
        for j in range(i):
            P = boxes[j]
            diff = [d for d in range(len(root[0]))
                    if (P[0][d], P[1][d]) != (box[0][d], box[1][d])]
            if len(diff) != 1:
                continue
            d = diff[0]
            mid = (P[0][d] + P[1][d]) / 2
            if (box[0][d], box[1][d]) in ((P[0][d], mid), (mid, P[1][d])):
                parents.append((j, d))
        assert parents, f"{box} is no half of an earlier box"
        halves.setdefault(parents[-1], []).append(box)
    for (j, d), kids in halves.items():
        assert len(kids) == 2, f"box {boxes[j]} split along {d} into {kids}"


def test_model_bab_builds_each_subdomain_with_the_given_set_type():
    """A set type that records its from_bounds calls: verify_bab_model builds
    every subdomain with it, from the subdomain's own bounds, the root first,
    and the subdomains form a bisection tree whose leaves tile the box (the
    real search, checked in fractions). Catches set_type ignored and a
    subdomain built from the wrong corner."""
    boxes = []

    class Recording(Star):
        @classmethod
        def from_bounds(cls, lb, ub):
            boxes.append((tuple(F(float(v)) for v in np.ravel(lb)),
                          tuple(F(float(v)) for v in np.ravel(ub))))
            return Star.from_bounds(lb, ub)
    t = F(2) + F(1, 64)
    result = verify_bab_model(_V.torch(), _LB, _UB, _geq(t), set_type=Recording,
                              falsify_method=None, branch="widest")
    assert result.verdict == "VERIFIED", result
    assert len(boxes) == result.nodes
    _assert_bisection_tree(boxes, (tuple(F(v) for v in _LB),
                                   tuple(F(v) for v in _UB)))


@known_bug(19, "verify_bab_model builds its input set with "
               "set_type.from_bounds, which Box does not have: set_type=Box "
               "raises AttributeError", raises=AttributeError)
def test_model_bab_with_box_sets_answers():
    """DESIGN section 13: verify_bab_model bounds with n2v.reach on any set
    type, and reach supports Box. The safe-by-1/64 problem is VERIFIED or
    UNKNOWN, the strip problem is not VERIFIED; a declared refusal
    (NotImplementedError) would also keep the promise of no wrong answer.
    The passing neighbours are the Star, Zono, Hexatope and Octatope cases
    of test_model_bab_never_proves_a_property_reached_in_a_small_region."""
    for t, allowed in ((F(2) + F(1, 64), ("VERIFIED", "UNKNOWN")),
                       (F(2) - F(1, 64), ("UNKNOWN",))):
        _answers_or_refuses(
            lambda: verify_bab_model(_V.torch(), _LB, _UB, _geq(t),
                                     set_type=Box, falsify_method=None,
                                     max_nodes=30), allowed)


# Conv2d(1, 1, 2) on a (1, 2, 3) image, ReLU, Flatten, Linear(2, 1): the
# convolution is the 2 x 6 matrix below on the row-major pixels.
_CONV_KERNEL = [[1.0, 0.5], [0.25, -1.0]]
_CONV = _Net(([[1, 0.5, 0, 0.25, -1, 0], [0, 1, 0.5, 0, 0.25, -1]], [0, 0]),
             ([[1, 1]], [0]))


def _conv_model():
    conv = nn.Conv2d(1, 1, 2)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([[_CONV_KERNEL]]))
        conv.bias.zero_()
    head = nn.Linear(2, 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 1.0]]))
        head.bias.zero_()
    return nn.Sequential(conv, nn.ReLU(), nn.Flatten(), head).eval()


_CONV_LB, _CONV_UB = np.zeros((1, 2, 3)), np.ones((1, 2, 3))


def _conv_top():
    """The exact maximum of the convolutional model on the (1, 2, 3) image box.
    y is a sum of ReLUs of affine maps with positive weights, so it is convex
    and its maximum is the largest value at the 64 corners, in fractions;
    the matrix is first checked against the torch convolution on dyadic
    points (exact in float32)."""
    W1, W2 = _CONV.layers[0][0], _CONV.layers[1][0]
    for x in np.random.default_rng(0).integers(0, 5, (8, 6)) / 4.0:
        with torch.no_grad():
            y = float(_conv_model()(torch.tensor(x.reshape(1, 1, 2, 3),
                                                 dtype=torch.float32)))
        if y != float((W2 @ np.maximum(W1 @ x, 0.0))[0]):
            pytest.fail("the matrix is not the convolution")
    return max(sum(F(W2[0][k]) * max(F(0), sum(F(w) * c for w, c in
                                               zip(W1[k], corner)))
                   for k in range(2))
               for corner in itertools.product((F(0), F(1)), repeat=6))


@known_bug(19, "verify_bab_model flattens the box and builds a flat set, so a "
               "model whose first layer needs an image set (Conv2d) is refused "
               "by that layer with ValueError instead of answered",
           raises=ValueError)
def test_model_bab_on_a_convolutional_model_answers():
    """The docstring and DESIGN section 13: verify_bab_model works for any
    model n2v's reach handles, and reach handles Conv2d on an image set.
    On a (1, 2, 3) image box the property y >= top + 1/64 (top the exact
    maximum) is VERIFIED or UNKNOWN, y >= top - 1/64 is not VERIFIED; a
    declared refusal (NotImplementedError) would also keep the promise of
    no wrong answer. The passing neighbour, with the same reference, is
    test_bab_with_an_image_reach_answers_on_a_convolutional_model."""
    top = _conv_top()
    for t, allowed in ((top + F(1, 64), ("VERIFIED", "UNKNOWN")),
                       (top - F(1, 64), ("UNKNOWN",))):
        _answers_or_refuses(
            lambda: verify_bab_model(_conv_model(), _CONV_LB, _CONV_UB,
                                     _geq(t), falsify_method=None,
                                     max_nodes=30), allowed)


def test_bab_with_an_image_reach_answers_on_a_convolutional_model():
    """verify_bab on the convolutional model with a reach function that builds
    the ImageStar of each (1, 2, 3) sub-box in height, width, channel order:
    the sub-boxes arrive in the caller's shape, the default bound reads the
    reach of each, and the answers keep the exact maximum: y >= top + 1/64
    is VERIFIED or UNKNOWN, y >= top - 1/64 is not VERIFIED. Catches a
    shaped box split or handed on wrongly and a pruned subdomain that holds
    the maximising corner."""
    from n2v.sets import ImageStar
    top, net = _conv_top(), NeuralNetwork(_conv_model())

    def reach_fn(lb, ub):
        if np.shape(lb) != (1, 2, 3):
            pytest.fail(f"the sub-box has shape {np.shape(lb)}")
        return net.reach(ImageStar.from_bounds(
            np.transpose(lb, (1, 2, 0)), np.transpose(ub, (1, 2, 0)), 2, 3, 1),
            method="approx")
    for t, allowed in ((top + F(1, 64), ("VERIFIED", "UNKNOWN")),
                       (top - F(1, 64), ("UNKNOWN",))):
        result = verify_bab(reach_fn, _CONV_LB, _CONV_UB, _geq(t), max_nodes=30)
        assert result.verdict in allowed, result


# --------------------------------------------------------------------------- #
# verify_bab_relu end to end                                                  #
# --------------------------------------------------------------------------- #
def _unstable_count(net, lb, ub):
    """Number of hidden neurons whose exact pre-activation range over the
    box meets zero: a path of the split tree forces each neuron at most
    once, and only such a neuron can be undecided."""
    count, pieces = 0, R.box(lb, ub)
    for W, b in net.layers[:-1]:
        pieces = R.affine(pieces, W, b)
        for i in range(len(W)):
            values = [p.value(v)[i] for p in pieces for v in p.corners()]
            count += min(values) <= 0 <= max(values)
        pieces = R.relu(pieces)
    return count


@pytest.mark.parametrize("net, t", [
    pytest.param(_V, F(2) + F(1, 64), id="one-hidden-layer"),
    pytest.param(_DEEP, None, id="two-hidden-layers"),
])
@pytest.mark.parametrize("falsify", [None, "random+pgd"])
def test_relu_bab_proves_a_property_safe_by_a_small_margin(net, t, falsify):
    """The safe instance (exact maximum below t; one relaxed reach does not
    prove it): VERIFIED, with nodes = 2 splits + 1 and a depth of the split
    tree between ceil(log2(splits + 1)) and the number of neurons whose
    exact range meets zero. Catches the depth counted per layer instead of
    per neuron, nodes and splits swapped, a node with nothing left to split
    answered UNKNOWN when it was exact, and a falsifier's miss read as a
    counterexample."""
    if t is None:
        _, top = net.extremes(_LB, _UB)
        t = top + F(1, 64)
    _needs_splitting(net, t)
    result = verify_bab_relu(net.torch(), _LB, _UB, _geq(t),
                             falsify_method=falsify,
                             falsify_kwargs={"seed": 0, "n_restarts": 2},
                             max_nodes=500)
    assert result.verdict == "VERIFIED", result
    assert result.nodes == 2 * result.splits + 1
    assert math.ceil(math.log2(result.splits + 1)) <= result.max_depth \
        <= _unstable_count(net, _LB, _UB)


_RELU_UNSAFE = [
    pytest.param(_CORNER, F(2) - F(1, 64), id="both-neurons-active"),
    pytest.param(_V, F(2) - F(1, 64), id="strip"),
    pytest.param(_TENT, F(0), id="one-point"),
    pytest.param(_DEEP, "deep", id="two-hidden-layers"),
]


def _deep_unsafe_t():
    _, top = _DEEP.extremes(_LB, _UB)
    return top - F(1, 64)


@pytest.mark.parametrize("net, t", _RELU_UNSAFE)
def test_relu_bab_never_proves_a_reached_property(net, t):
    """Without a falsifier, a property the model reaches (exactly: the
    unsafe region has a vertex) must end UNKNOWN. The corner instance needs
    the second neuron split in the same layer to be active, the one-point
    instance reaches the unsafe set only on the boundary of every phase
    region. Catches the children of a split sharing one decision map, the
    sign half-space or the pinned bound of a forced neuron turned round, an
    exact leaf that is not proved skipped, and decisions applied to the
    wrong layer."""
    if t == "deep":
        t = _deep_unsafe_t()
    if not net.unsafe_vertices(_LB, _UB, t):
        pytest.fail("the unsafe region is empty")
    result = verify_bab_relu(net.torch(), _LB, _UB, _geq(t),
                             falsify_method=None, max_nodes=500)
    assert result.verdict == "UNKNOWN", result


def test_relu_bab_an_infeasible_phase_pattern_does_not_end_the_search(
        monkeypatch):
    """y >= 3/2 - 1/64 on _PRUNE is reached (exactly: the unsafe region has
    vertices), and the split search meets a node whose phase pattern no
    input has (the search is watched here, and every such pattern is checked
    to be empty by the exact pieces). Pruning that node must not end the
    search: without a falsifier the answer is UNKNOWN. Catches an infeasible
    node read as the end of the search (break written for continue), which
    answers VERIFIED with subdomains still on the stack, and an emptiness
    claim for a pattern that has inputs."""
    import n2v.nn.bab as bab
    t = F(3, 2) - F(1, 64)
    if not _PRUNE.unsafe_vertices(_LB, _UB, t):
        pytest.fail("the unsafe region is empty")
    empty_patterns, real = [], bab._seq_reach_relu

    def watched(layers, star, decisions, *args):
        out = real(layers, star, decisions, *args)
        if out[2]:
            empty_patterns.append({k: dict(v) for k, v in decisions.items()})
        return out
    monkeypatch.setattr(bab, "_seq_reach_relu", watched)
    result = verify_bab_relu(_PRUNE.torch(), _LB, _UB, _geq(t),
                             falsify_method=None, max_nodes=300)
    if not empty_patterns:
        pytest.fail("the search met no infeasible phase pattern")
    for decisions in empty_patterns:
        assert _pattern_pieces(_PRUNE, _LB, _UB, decisions) == [], decisions
    assert result.verdict == "UNKNOWN", result


@pytest.mark.parametrize("net, t", [
    pytest.param(_CORNER, F(2) - F(1, 64), id="corner"),
    pytest.param(_V, F(2) - F(1, 64), id="strip"),
    pytest.param(_DEEP, "deep", id="two-hidden-layers"),
])
def test_relu_bab_witness_is_a_real_counterexample(net, t, tmp_path):
    """With the default falsifier (seeded through falsify_kwargs) the answer
    is FALSIFIED with an input in the box whose onnxruntime output reaches
    t. Catches a witness that is not the falsifier's input (zeros, the
    output half of the pair) and the falsifier skipped."""
    if t == "deep":
        t = _deep_unsafe_t()
    result = verify_bab_relu(net.torch(), _LB, _UB, _geq(t),
                             falsify_kwargs={"seed": 0}, max_nodes=500)
    _assert_witness(result, net, _LB, _UB, [[([[-1.0]], [-float(t)])]],
                    tmp_path)


def test_relu_bab_witness_on_a_box_away_from_zero(tmp_path):
    """y = relu(x0) + relu(x1) on [1/2, 3/2] x [1/4, 7/4], a box with
    positive lower bounds and two widths: y >= 3 holds only in the corner
    triangle x0 + x1 >= 3 (exactly: the unsafe region has vertices and all
    have x0 > 1). The seeded random falsifier must sample the whole box and
    return a witness in it that onnxruntime replays as unsafe. Catches a
    sampler that covers [lb, ub - lb] instead of [lb, ub], which never
    reaches that corner, and a witness taken from outside the box."""
    lb, ub = np.array([0.5, 0.25]), np.array([1.5, 1.75])
    vertices = _CORNER.unsafe_vertices(lb, ub, 3)
    if not vertices or not all(v[0] > 1 for v in vertices):
        pytest.fail(f"the unsafe corner is not where it should be: {vertices}")
    result = verify_bab_relu(_CORNER.torch(), lb, ub, _geq(3),
                             falsify_method="random",
                             falsify_kwargs={"seed": 0}, max_nodes=50)
    _assert_witness(result, _CORNER, lb, ub, [[([[-1.0]], [-3.0])]], tmp_path)


def test_relu_bab_with_a_leading_flatten_and_a_shaped_box(tmp_path):
    """Bounds of shape (1, 2) and a model that starts with Flatten: the safe
    instance is VERIFIED and the strip instance FALSIFIED with a witness in
    the box. Catches the Flatten path and the flattening of the bounds."""
    model = _V.torch(flatten_first=True)
    lb, ub = _LB.reshape(1, 2), _UB.reshape(1, 2)
    assert verify_bab_relu(model, lb, ub, _geq(F(2) + F(1, 64)),
                           falsify_method=None).verdict == "VERIFIED"
    t = F(2) - F(1, 64)
    result = verify_bab_relu(model, lb, ub, _geq(t), falsify_kwargs={"seed": 0})
    _assert_witness(result, _V, lb, ub, [[([[-1.0]], [-float(t)])]], tmp_path)


_A, _B = F(2) - F(1, 64), F(2) + F(1, 64)
_FAR = HalfSpace(np.array([[-1.0]]), np.array([[-10.0]]))          # y >= 10
_LOW = HalfSpace(np.array([[1.0]]), np.array([[-1.0]]))            # y <= -1


@pytest.mark.parametrize("spec, groups, reached", [
    pytest.param({"Hg": _geq(_A)}, [[([[-1.0]], [-float(_A)])]], True,
                 id="dict"),
    pytest.param([_FAR, _geq(_A)], [[([[-1.0]], [-10.0]),
                                     ([[-1.0]], [-float(_A)])]], True,
                 id="or-list-second-reached"),
    pytest.param([_FAR, _LOW], [[([[-1.0]], [-10.0]), ([[1.0]], [-1.0])]],
                 False, id="or-list-neither-reached"),
    pytest.param([{"Hg": _geq(_A)}, {"Hg": [_FAR, _geq(_B - 1)]}],
                 [[([[-1.0]], [-float(_A)])],
                  [([[-1.0]], [-10.0]), ([[-1.0]], [-float(_B - 1)])]], True,
                 id="and-of-groups-reached"),
    pytest.param([{"Hg": _geq(_A)}, {"Hg": [_FAR]}],
                 [[([[-1.0]], [-float(_A)])], [([[-1.0]], [-10.0])]], False,
                 id="and-with-an-unreached-group"),
])
def test_relu_bab_reads_every_spec_form(spec, groups, reached, tmp_path):
    """The corner instance under every form of the unsafe region: a dict, an
    OR list whose second member is reached, an OR list of two unreached
    half-spaces, an AND of two groups both reached at (1, 1), an AND with an
    unreached group. Reached: not VERIFIED without a falsifier, and a
    replayed witness with it. Not reached: VERIFIED."""
    model = _CORNER.torch()
    plain = verify_bab_relu(model, _LB, _UB, spec, falsify_method=None,
                            max_nodes=200)
    if reached:
        assert plain.verdict == "UNKNOWN", plain
        result = verify_bab_relu(model, _LB, _UB, spec,
                                 falsify_kwargs={"seed": 0}, max_nodes=200)
        _assert_witness(result, _CORNER, _LB, _UB, groups, tmp_path)
    else:
        assert plain.verdict == "VERIFIED", plain


def test_relu_bab_budgets_end_unknown():
    """The safe-by-1/64 instance needs five nodes: max_nodes=1 ends UNKNOWN
    after one; max_nodes=0 after none; a clock that advances one unit per
    call, started at 1e6, with timeout_s=1.5 ends UNKNOWN before the full
    search; timeout_s=600 on the real clock (and verbose=True) lets it
    finish VERIFIED."""
    model, spec = _V.torch(), _geq(F(2) + F(1, 64))
    for budget in (1, 0):
        result = verify_bab_relu(model, _LB, _UB, spec, falsify_method=None,
                                 max_nodes=budget)
        assert (result.verdict, result.nodes) == ("UNKNOWN", budget), result
    ticks = iter(range(10_000))
    result = verify_bab_relu(model, _LB, _UB, spec, falsify_method=None,
                             timeout_s=1.5, _clock=lambda: 1.0e6 + next(ticks))
    assert result.verdict == "UNKNOWN", result
    assert result.nodes >= 1
    assert verify_bab_relu(model, _LB, _UB, spec, falsify_method=None,
                           timeout_s=600.0, verbose=True).verdict == "VERIFIED"


def test_relu_bab_hands_its_falsifier_options_to_the_falsifier():
    """falsify_kwargs={'n_samples': 0} leaves the random falsifier nothing to
    try: y >= 1 on y = relu(x0 + x1) + relu(x0 - x1), reached on a quarter
    of the box, is then not falsified and the search ends UNKNOWN; with the
    default number of samples it is FALSIFIED. Catches falsify_kwargs
    dropped."""
    for kwargs, want in (({"seed": 0, "n_samples": 0}, "UNKNOWN"),
                         ({"seed": 0}, "FALSIFIED")):
        result = verify_bab_relu(_V.torch(), _LB, _UB, _geq(1),
                                 falsify_method="random",
                                 falsify_kwargs=kwargs, max_nodes=50)
        assert result.verdict == want, (kwargs, result)


def test_relu_bab_hands_its_options_to_the_node_bound(monkeypatch):
    """relax_factor and lp_solver given to verify_bab_relu reach the relaxed
    ReLU of every node and the emptiness check of the split nodes. Catches
    either option dropped on the way to the node bound."""
    import n2v.nn.layer_ops.relu_reach as relu_reach
    seen = {"relax_factor": set(), "lp_solver": set(), "emptiness": set()}
    real_relu, real_empty = relu_reach.relu_star_approx, Star.is_empty_set

    def relu(*args, **kwargs):
        seen["relax_factor"].add(kwargs.get("relax_factor"))
        seen["lp_solver"].add(kwargs.get("lp_solver"))
        return real_relu(*args, **kwargs)

    def empty(self, *args, **kwargs):
        seen["emptiness"].add(args[0] if args else kwargs.get("lp_solver"))
        return real_empty(self, *args, **kwargs)
    monkeypatch.setattr(relu_reach, "relu_star_approx", relu)
    monkeypatch.setattr(Star, "is_empty_set", empty)
    result = verify_bab_relu(_V.torch(), _LB, _UB, _geq(F(2) + F(1, 64)),
                             falsify_method=None, relax_factor=0.75,
                             lp_solver="highs-ds")
    assert result.verdict == "VERIFIED", result
    assert seen["relax_factor"] == {0.75}
    assert seen["lp_solver"] == {"highs-ds"}
    assert "highs-ds" in seen["emptiness"]     # other callers use the default


@pytest.mark.parametrize("lp_solver, relax_factor", [
    pytest.param("linprog", 0.5, id="linprog"),
    pytest.param("highs-ds", 0.5, id="highs-ds"),
    pytest.param("CLARABEL", 0.5, id="cvxpy-clarabel"),
    pytest.param("default", 1.0, id="relax-factor-1"),
])
def test_relu_bab_options_keep_the_verdicts(lp_solver, relax_factor):
    """Every lp_solver family (scipy HiGHS, a HiGHS variant, a CVXPY solver)
    and relax_factor at the top of its range: the corner instance is not
    VERIFIED, the safe instance is VERIFIED."""
    assert verify_bab_relu(_CORNER.torch(), _LB, _UB, _geq(F(2) - F(1, 64)),
                           falsify_method=None, lp_solver=lp_solver,
                           relax_factor=relax_factor).verdict == "UNKNOWN"
    assert verify_bab_relu(_V.torch(), _LB, _UB, _geq(F(2) + F(1, 64)),
                           falsify_method=None, lp_solver=lp_solver,
                           relax_factor=relax_factor).verdict == "VERIFIED"


@pytest.mark.parametrize("relax_factor", [1.5, float("nan")])
def test_relu_bab_refuses_relax_factor_outside_its_range(relax_factor):
    """relax_factor must lie in (0, 1]; a value above the range and NaN
    raise ValueError (0, the value below, is in
    tests/unit/test_bab_regressions.py). Catches the upper end unchecked and
    a range check written so that NaN passes it."""
    with pytest.raises(ValueError, match="relax_factor"):
        verify_bab_relu(_V.torch(), _LB, _UB, _geq(3), falsify_method=None,
                        relax_factor=relax_factor)


@pytest.mark.parametrize("layer", [nn.Sigmoid(), nn.Identity(),
                                   nn.BatchNorm1d(2).eval()],
                         ids=["sigmoid", "identity", "batchnorm"])
def test_relu_bab_refuses_a_layer_it_has_no_rule_for(layer):
    """Only Linear, Flatten and ReLU are supported; another layer raises
    NotImplementedError, never an answer."""
    model = nn.Sequential(nn.Linear(2, 2), layer, nn.Linear(2, 1)).eval()
    with pytest.raises(NotImplementedError):
        verify_bab_relu(model, _LB, _UB, _geq(100), falsify_method=None)


class _Wrapped(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


@known_bug(23, "verify_bab_relu calls list(model), so a model that is not an "
               "nn.Sequential raises TypeError ('object is not iterable') "
               "instead of a refusal", raises=TypeError)
def test_relu_bab_refuses_a_model_that_is_not_a_sequential():
    """A module that wraps the Sequential is outside the documented support
    (DESIGN section 13: Linear/Flatten/ReLU nn.Sequential): the call must
    raise NotImplementedError, or answer soundly (y >= 3 is unreachable, the
    exact maximum is 2), and never crash. The passing neighbour is
    test_relu_bab_refuses_a_layer_it_has_no_rule_for."""
    _answers_or_refuses(
        lambda: verify_bab_relu(_Wrapped(_V.torch()), _LB, _UB, _geq(3),
                                falsify_method=None), ("VERIFIED", "UNKNOWN"))


def test_relu_bab_does_not_read_a_failed_emptiness_check_as_empty(monkeypatch):
    """An LPFailure from the emptiness check of a split node is no
    certificate: the corner instance must still end UNKNOWN. Catches a
    solver failure read as an infeasible phase pattern."""
    def failing(self, *args, **kwargs):
        raise LPFailure("injected: the emptiness LP did not decide")
    monkeypatch.setattr(Star, "is_empty_set", failing)
    result = verify_bab_relu(_CORNER.torch(), _LB, _UB, _geq(F(2) - F(1, 64)),
                             falsify_method=None)
    assert result.verdict == "UNKNOWN", result


# --------------------------------------------------------------------------- #
# _seq_reach_relu: the bound of one phase pattern                              #
# --------------------------------------------------------------------------- #
def _pattern_pieces(net, lb, ub, decisions):
    """The exact pieces of the inputs of the box whose pre-activations have
    the forced signs: neuron n of ReLU layer k is cut to pre >= 0 (+1) or
    pre <= 0 (-1) on every piece, where pre is linear."""
    pieces = R.box(lb, ub)
    for k, (W, b) in enumerate(net.layers):
        pieces = R.affine(pieces, W, b)
        if k == len(net.layers) - 1:
            break
        for n, sign in decisions.get(k, {}).items():
            cut = []
            for p in pieces:
                row = [-sign * w for w in p.W[n]]
                child = R.Piece(p.A + [row], p.b + [sign * p.c[n]], p.W, p.c)
                if child.is_inhabited():
                    cut.append(child)
            pieces = cut
        pieces = R.relu(pieces)
    return pieces


@pytest.mark.parametrize("net, decisions", [
    pytest.param(_V, {0: {0: 1}}, id="v-active"),
    pytest.param(_V, {0: {0: 1, 1: -1}}, id="v-both-forced"),
    pytest.param(_CORNER, {0: {1: 1}}, id="corner-second-active"),
    pytest.param(_DEEP, {0: {0: 1}}, id="deep-first-layer"),
    pytest.param(_DEEP, {1: {0: -1}}, id="deep-second-layer-inactive"),
    pytest.param(_DEEP, {1: {1: 1}}, id="deep-second-layer-active"),
    pytest.param(_DEEP, {0: {1: -1}, 1: {0: 1}}, id="deep-both-layers"),
])
def test_phase_pattern_bound_holds_the_exact_image_of_its_region(net,
                                                                  decisions):
    """The star of a phase pattern holds every vertex of the exact image of
    the inputs whose pre-activations have the forced signs, in rationals
    with no slack, so the two children of a split cover their parent. The
    network is probed: its output is the last hidden layer and y, so a star
    of the wrong region cannot hide behind a one-dimensional output. It is
    called empty only when that region is empty, and its unstable list names
    every neuron of the right layer whose exact range on the region crosses
    zero. Catches the sign half-space or the pinned bound turned round,
    decisions applied to the wrong layer, and the layer counter not
    advanced."""
    net = net.probe()
    pieces = _pattern_pieces(net, _LB, _UB, decisions)
    model = net.torch()
    out, unstable, empty = _seq_reach_relu(
        list(model), Star.from_bounds(_LB, _UB), decisions, 0.5, "default")
    if not pieces:
        assert empty
        return
    assert not empty, "a phase pattern with inputs was called empty"
    assert R.escapes(pieces, R.star(out)) == []
    listed = {(k, n) for k, n, _, _ in unstable}
    for k, n in _crossing(net, decisions):
        assert (k, n) in listed, f"neuron {n} of ReLU layer {k} not listed"


def _crossing(net, decisions):
    """(layer, neuron) of the undecided neurons whose exact pre-activation
    on the region of the pattern takes both signs strictly."""
    found, pieces = [], R.box(_LB, _UB)
    for k, (W, b) in enumerate(net.layers[:-1]):
        pieces = R.affine(pieces, W, b)
        for n, sign in decisions.get(k, {}).items():
            pieces = [c for c in (R.Piece(p.A + [[-sign * w for w in p.W[n]]],
                                          p.b + [sign * p.c[n]], p.W, p.c)
                                  for p in pieces) if c.is_inhabited()]
        for n in range(len(W)):
            if n in decisions.get(k, {}):
                continue
            values = [p.value(v)[n] for p in pieces for v in p.corners()]
            if values and min(values) < 0 < max(values):
                found.append((k, n))
        pieces = R.relu(pieces)
    return found


def test_an_infeasible_phase_pattern_is_called_empty():
    """On [1/2, 1] x [-1, 1] the neuron x0 of y = relu(x0) + relu(x1) is
    always active: forcing it inactive leaves no input (the region
    x0 <= 0 meets the box nowhere, exactly), and the node is empty."""
    lb, ub = np.array([0.5, -1.0]), np.array([1.0, 1.0])
    if _pattern_pieces(_CORNER, lb, ub, {0: {0: -1}}):
        pytest.fail("the pattern has inputs")
    _, _, empty = _seq_reach_relu(list(_CORNER.torch()),
                                  Star.from_bounds(lb, ub), {0: {0: -1}},
                                  0.5, "default")
    assert empty

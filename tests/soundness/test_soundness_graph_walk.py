"""Soundness cells for the walk over a traced or ONNX graph in
``n2v/nn/reach.py``: which node's sets are handed to which consumer, what the
walk returns, how function nodes, tensor methods and dropout are read, how the
bound pre-pass reaches each node, and the graph-level handlers that select or
combine rows (Split and its outputs, Transpose and ReduceSum / ReduceMean on
flat sets, constant subgraphs, Pad's constant value, BatchNorm fusion).

What a cell here must make visible is plumbing: the right value passed to the
wrong place. Every graph is built so that the alternatives differ: a node fed
by a value that another consumer read first, a returned node that is not the
last node computed, two calls of one activation with different input ranges,
a join of two branches whose outputs differ, the second output of a Split, a
permutation that is not its own inverse, a reduction whose divisor is not the
count of the kept axes.

Oracles. Every cell replays member points of the input set on the true
network, torch in float64 for a torch model and onnxruntime for an ONNX file,
with dyadic weights and bounds so that float32 and float64 evaluation are
exact (a float64 graph where a mean divides by 3 or 5), and checks that each
true output is a member of a returned set by an LP on that set's own rows
(``tests/_oracles/membership.py``). The cells of the exact method also compare
with the exact image in rationals (``tests/_oracles/rational.py``) of a Star
with two or three predicate variables, in both directions: no vertex of the
true image outside the result, and no vertex of a returned set without a
preimage. The affine parts of those networks come from the model's own
weights or from onnxruntime on unit vectors, never from n2v. The BatchNorm
cell evaluates the network in fractions, with weights chosen so that the
square root in the normalisation is exact.

Layout: bounds, member points and true outputs are in the network's order
(channel, height, width for an image tensor); an image set stores height,
width, channel, and the coverage check converts with ``out_chw``.

Sections 13 to 15 add what the earlier cells could not see. A Gemm
to a few outputs after a step that moves entries projects a misplacement
away, so the layout cells of section 13 (Gather, Reshape into a convolution
and into a flat head, a join of an image and a flat branch of one tensor, an
image Concat, an image and a constant of the whole tensor, MatMul with the
constant on the left) return the moved tensor itself; the tensor-shape pass
is checked with ``input_shape`` and without it on an ImageStar; the
McCormick product and quotient are reached with every LP stopping inside its
optimum. Section 14 hands the flat Reduce and BatchNorm handlers the pieces
of an exact split and checks exact image-set graphs. Section 15 checks joins
after an exact split, a value read twice by one product, calls that carry
more than the set, a model reached again with another input, the shape each
set type tracks, refusals, the layout of the pre-pass bounds of an image,
and the walk's own known defects with their passing neighbours. Section 16
adds shape queries in the batch entry of a view, the warning of the
configuration, and a dropout between a view and a convolution.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from fractions import Fraction
from onnx import TensorProto, helper, numpy_helper

from n2v.nn.reach import ReachConfig
from n2v.sets import Box, Star
from n2v.sets.image_star import ImageStar
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational
from tests.cell_support import (assert_covered, constrained_star, flat_box,
                                flat_set, image_set, image_split_cover,
                                pixel_box, reach_model, replay_onnx,
                                replay_torch, samples, save_onnx, split_cover)
from tests.known_bugs import DID_NOT_RAISE, known_bug


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #
def _dyadic(rng, *shape):
    """Quarters in [-1, 1] as float32: exact in float32 and float64."""
    return (rng.integers(-4, 5, size=shape) / 4.0).astype(np.float32)


def _dyadic_module(module, seed):
    """``module`` in float64 eval mode with every parameter a quarter in
    [-1, 1] (a Linear's weight and bias, a Conv's kernel)."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.tensor(rng.integers(-4, 5, size=tuple(p.shape))
                                 / 4.0))
    return module.double().eval()


def _flat_inputs(kind, form, lo, hi):
    """``(sets, member points)`` of a flat kind: "box" (the box itself),
    "row" (a Star with a binding row that is not a box row), "several"
    (three sets whose union is the box). ``reach_model`` reaches each set of
    "several" with its own call of the model entry, so no graph handler
    receives more than one set from this form; the cells of section 14 that
    reach the pieces of an exact split are the ones that hand a handler
    several sets at once."""
    if form == "box":
        return [flat_set(kind, lo, hi)], samples(lo, hi)
    if form == "row":
        s, points = constrained_star(lo, hi)
        return [s], points
    if form == "several":
        return split_cover(kind, lo, hi), samples(lo, hi)
    raise ValueError(form)


def _low_rank_star(rng, n, k=3):
    """``(star, x0, E)``: the Star {x0 + E t : t in [-1, 1]^k} of ``n``
    entries, dyadic, with no constraint row; ``x0`` and ``E`` in the
    network's order."""
    x0 = rng.integers(-8, 9, size=n) / 8.0
    E = rng.integers(-4, 5, size=(n, k)) / 8.0
    star = Star(np.hstack([x0[:, None], E]), np.zeros((0, k)),
                np.zeros((0, 1)), -np.ones((k, 1)), np.ones((k, 1)))
    return star, x0, E


def _low_rank_points(x0, E, count=64, seed=1):
    t = np.random.default_rng(seed).integers(-4, 5, size=(count, E.shape[1]))
    t = np.vstack([t / 4.0, -np.ones(E.shape[1]), np.ones(E.shape[1])])
    return [x0 + E @ ti for ti in t]


def _without_unused_variables(computed):
    """The same set over fewer predicate variables: a variable that no basis
    vector and no row uses, and whose interval is not empty, can take any
    value of its interval, so dropping it leaves the set unchanged (the tag
    variables of a join are of this kind). Vertex enumeration is exponential
    in the number of variables, so this keeps the exact check fast."""
    c, G, C, d, lb, ub = computed
    keep = [j for j in range(len(lb))
            if any(row[j] != 0 for row in G) or any(row[j] != 0 for row in C)
            or (lb[j] is not None and ub[j] is not None and lb[j] > ub[j])]
    return (c, [[row[j] for j in keep] for row in G],
            [[row[j] for j in keep] for row in C], d,
            [lb[j] for j in keep], [ub[j] for j in keep])


def _assert_exact(out, pieces, what):
    """Both directions against the exact image ``pieces``: every vertex of
    every piece in one returned set (no escape, no piece spread over several
    sets), and every vertex of every returned set with a preimage.

    Before the vertices of a returned set are enumerated (exponential in its
    number of predicate variables), the set must use no more predicate
    variables than the input: the exact rules met in these cells (affine
    maps, ReLU splits, Split, Concat, a join of one predicate system) keep
    the input's variables and add rows only, so a set over more variables
    holds relaxation variables (a ReLU triangle, two systems joined as
    independent) and is not the exact image."""
    computed = [_without_unused_variables(rational.star(s)) for s in out]
    escaped = rational.escapes(pieces, *computed)
    assert not escaped, (f"{what}: {len(escaped)} vertices of the true image "
                         f"are in no returned set, first {escaped[0]}")
    spread = rational.straddlers(pieces, *computed)
    assert not spread, (f"{what}: pieces {spread} of the true image are not "
                        f"held by a single returned set")
    n_input = len(pieces[0].A[0])
    for k, s in enumerate(computed):
        assert len(s[4]) <= n_input, (
            f"{what}: returned set {k} uses {len(s[4])} predicate variables, "
            f"the input {n_input}: it holds relaxation variables")
    for k, s in enumerate(computed):
        extra = rational.unreached(pieces, rational.set_vertices(s))
        assert not extra, (f"{what}: returned set {k} has {len(extra)} "
                           f"vertices with no preimage, first {extra[0][1]}")


def _linear_of(layer):
    """``(W, b)`` of a torch Linear as nested lists of floats (exact)."""
    return (layer.weight.detach().double().numpy().tolist(),
            layer.bias.detach().double().numpy().tolist())


def _shifted_relu(pieces, keep, shift=64):
    """ReLU on the coordinates not in ``keep``, the identity on those in
    ``keep``: the kept coordinates are moved by ``shift`` (larger than any
    value they take, so ReLU leaves them alone) and moved back."""
    n = len(pieces[0].c)
    up = [shift if i in keep else 0 for i in range(n)]
    eye = [[int(i == j) for j in range(n)] for i in range(n)]
    pieces = rational.relu(rational.affine(pieces, eye, up))
    return rational.affine(pieces, eye, [-u for u in up])


# --------------------------------------------------------------------------- #
# 1. torch models: which sets an activation node reads, which node the walk   #
#    returns                                                                   #
# --------------------------------------------------------------------------- #
# Spellings of an activation in a torch forward, with the slope of the true
# function on the negative side (None for a smooth function).
_ACTIVATIONS = {
    "F.relu": 0,
    "torch.relu": 0,
    "nn.ReLU": 0,
    "F.leaky_relu-keyword": 0.25,
    "F.leaky_relu-default": None,       # slope 0.01, not dyadic: replay only
    "nn.LeakyReLU": 0.25,
    "torch.sigmoid": None,
    "torch.tanh": None,
}


def _activate(module, how, h):
    if how == "F.relu":
        return F.relu(h)
    if how == "torch.relu":
        return torch.relu(h)
    if how == "nn.ReLU":
        return module.relu(h)
    if how == "F.leaky_relu-keyword":
        return F.leaky_relu(h, negative_slope=0.25)
    if how == "F.leaky_relu-default":
        return F.leaky_relu(h)
    if how == "nn.LeakyReLU":
        return module.leaky(h)
    if how == "torch.sigmoid":
        return torch.sigmoid(h)
    if how == "torch.tanh":
        return torch.tanh(h)
    raise ValueError(how)


class _AuxHeads(nn.Module):
    """x -> fc1 = h; aux(h) runs before the activation reads h; y = fc2(act(h))
    is returned; late(act(h)) runs after y and is not returned. ``aux`` has
    the width of ``h`` and ``late`` the width of ``y``, so reading the wrong
    node's sets gives a set of the right size with the wrong entries."""

    def __init__(self, how, n=4):
        super().__init__()
        self.fc1 = nn.Linear(n, 4)
        self.aux = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 3)
        self.late = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU(0.25)
        self.how = how

    def forward(self, x):
        h = self.fc1(x)
        early = self.aux(h)                                   # noqa: F841
        r = _activate(self, self.how, h)
        y = self.fc2(r)
        late = self.late(r)                                   # noqa: F841
        return y


_AUX_CELLS = (
    [(how, "Star", "row") for how in _ACTIVATIONS]
    + [("F.relu", kind, form) for kind, form in (
        ("Star", "several"), ("Zono", "box"), ("Box", "several"),
        ("Hexatope", "box"))]
    + [("nn.ReLU", kind, form) for kind, form in (("Zono", "box"),
                                                  ("Octatope", "box"))])


@pytest.mark.parametrize("how,kind,form", _AUX_CELLS)
def test_activation_reads_its_own_input_and_the_returned_node_is_returned(
        how, kind, form):
    """A torch model in which a second consumer of the activation's input
    runs before the activation, and an auxiliary head runs after the returned
    node. Catches an activation node (function node or module) that is
    applied to the sets computed last instead of the sets of its own input
    node, a walk that returns the sets of the last node computed instead of
    the node the output names, a function node mapped to the wrong layer
    (``torch.sigmoid`` read as ``tanh``, ``torch.relu`` not mapped), and a
    ``F.leaky_relu`` slope read from the wrong argument or ignored (torch.fx
    records the slope as the keyword ``negative_slope`` for every spelling of
    a traced forward, so the keyword 1/4 and the default 0.01 are the two
    values; the positional branch is reached only by a hand-built graph, see
    ``test_leaky_relu_slope_given_by_position_in_a_built_graph``)."""
    model = _dyadic_module(_AuxHeads(how), seed=21)
    lo, hi = flat_box(4, seed=3)
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"{how} with a fan-out and a late head on a {form} {kind}"
    out = reach_model(model, sets, "approx", what)
    assert_covered(out, replay_torch(model, points, (1, 4)), what)


@pytest.mark.parametrize("how", [h for h, s in _ACTIVATIONS.items()
                                 if s is not None])
def test_exact_activation_after_a_fan_out_is_the_exact_image(how):
    """Exact method on the same model, a Star with three predicate variables:
    both directions against the exact image in rationals (ReLU or LeakyReLU
    with the stated slope 1/4). Catches the plumbing bugs of the approx cell
    in exact arithmetic, a slope read from the wrong argument (the image of
    slope 0.01 has vertices without a preimage and misses true ones), and an
    exact reach that does not split at the activation."""
    model = _dyadic_module(_AuxHeads(how), seed=22)
    star, x0, E = _low_rank_star(np.random.default_rng(23), 4)
    what = f"exact {how} after a fan-out"
    out = reach_model(model, star, "exact", what)
    W1, b1 = _linear_of(model.fc1)
    W2, b2 = _linear_of(model.fc2)
    pieces = rational.affine(rational.box([-1] * 3, [1] * 3), E.tolist(),
                             x0.tolist())
    pieces = rational.affine(pieces, W1, b1)
    slope = _ACTIVATIONS[how]
    pieces = (rational.relu(pieces) if slope == 0
              else rational.leaky_relu(pieces, slope))
    _assert_exact(out, rational.affine(pieces, W2, b2), what)
    assert_covered(out, replay_torch(model, _low_rank_points(x0, E, 14),
                                     (1, 4)), what)


# --------------------------------------------------------------------------- #
# 2. The bound pre-pass: each call of an activation gets its own bounds        #
# --------------------------------------------------------------------------- #
class _TwoCalls(nn.Module):
    """fc1 -> act -> fc2 -> act -> fc3, the same activation called twice (one
    module object or one function). With ``crossing_first`` False the first
    call sees [1/2, 3/2] on every neuron (stably active) and the second
    [-7/8, 9/8] (crossing); with True the other way round. Both calls have
    three neurons, so bounds of one call fit the other. The offset 1/8 keeps
    every kink off the vertices of the pieces of the exact split, so that no
    neuron's range on a piece ends exactly at 0 (the exact ReLU kernel
    relaxes such a neuron within its LP pad; that kernel is not this
    file's subject)."""

    def __init__(self, how, crossing_first=False):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(2, 3), nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 2)
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU(0.25)
        self.how = how
        W1 = [[0.25, 0.25], [0.25, -0.25], [-0.5, 0.0]]
        with torch.no_grad():
            if crossing_first:
                # [-7/8, 9/8] first; then 1/2 h + 1 is at least 7/8 after
                # either activation.
                self.fc1.weight.copy_(2 * torch.tensor(W1))
                self.fc1.bias.fill_(0.125)
                self.fc2.weight.copy_(0.5 * torch.eye(3))
                self.fc2.bias.fill_(1.0)
            else:
                self.fc1.weight.copy_(torch.tensor(W1))
                self.fc1.bias.fill_(1.0)
                self.fc2.weight.copy_(2 * torch.eye(3))
                self.fc2.bias.fill_(-1.875)
            self.fc3.weight.copy_(torch.tensor([[1.0, 1.0, 1.0],
                                                [1.0, -1.0, 0.5]]))
            self.fc3.bias.copy_(torch.tensor([0.0, 0.25]))
        self.double().eval()

    def act(self, h):
        if self.how == "nn.ReLU":
            return self.relu(h)
        if self.how == "nn.LeakyReLU":
            return self.leaky(h)
        if self.how == "F.relu":
            return F.relu(h)
        if self.how == "torch.relu":
            return torch.relu(h)
        return F.leaky_relu(h, negative_slope=0.25)

    def forward(self, x):
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))


# torch.relu takes the function-node path of F.relu (section 1 checks that it
# is mapped); the lookup of bounds by node name is checked on F.relu.
_TWO_CALL_SLOPES = {"nn.ReLU": 0, "F.relu": 0, "nn.LeakyReLU": 0.25,
                    "F.leaky_relu": 0.25}


# ``precompute_bounds=True`` means "ibp"; it is checked on one spelling.
_TWO_CALL_CELLS = (
    [(how, pre, method, False) for how in _TWO_CALL_SLOPES
     for pre in ("ibp", "zono") for method in ("approx", "exact")]
    + [("nn.ReLU", True, method, False) for method in ("approx", "exact")]
    + [("nn.ReLU", "ibp", method, True) for method in ("approx", "exact")])


@pytest.mark.parametrize("how,precompute,method,crossing_first",
                         _TWO_CALL_CELLS)
def test_each_call_of_a_reused_activation_gets_its_own_bounds(
        how, precompute, method, crossing_first):
    """One activation (a module object or a function) called at two places,
    reached with the bound pre-pass. The two calls are two graph nodes with
    different pre-activation ranges. Catches bounds looked up by the module's
    attribute name or the function's name instead of the node's name (the
    second call gets the first call's bounds and its crossing neurons are
    taken as stably active), and a zonotope pre-pass that records a layer's
    output sets as its input bounds. Exact method: both directions against
    the exact image in rationals, and a replay of 16 points; approx: a
    replay of 128 points, corners included.
    The order with the crossing call first is the passing neighbour in which
    a mixed-up lookup is harmless."""
    model = _TwoCalls(how, crossing_first)
    what = (f"{how} called twice, precompute_bounds={precompute!r}, "
            f"{method}, crossing {'first' if crossing_first else 'second'}")
    lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    out = reach_model(model, flat_set("Star", lo, hi), method, what,
                      precompute_bounds=precompute)
    slope = _TWO_CALL_SLOPES[how]

    def act(p):
        return rational.relu(p) if slope == 0 else rational.leaky_relu(p,
                                                                       slope)
    pieces = rational.box([-1, -1], [1, 1])
    for layer in (model.fc1, model.fc2):
        pieces = act(rational.affine(pieces, *_linear_of(layer)))
    pieces = rational.affine(pieces, *_linear_of(model.fc3))
    if method == "exact":
        _assert_exact(out, pieces, what)
    points = samples(lo, hi, n=16 if method == "exact" else 128)
    assert_covered(out, replay_torch(model, points, (1, 2)), what)


# --------------------------------------------------------------------------- #
# 3. Dropout and tensor-method casts: identity only when the value is kept     #
# --------------------------------------------------------------------------- #
class _Middle(nn.Module):
    """fc1 -> <middle spelling> -> ReLU -> fc2 -> <end spelling>."""

    def __init__(self, middle="h", end="y"):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(4, 4), nn.Linear(4, 3)
        self.drop = nn.Dropout(0.5)
        self.middle, self.end = middle, end

    def forward(self, x):
        h = self.fc1(x)                                       # noqa: F841
        h = eval(self.middle)
        y = self.fc2(F.relu(h))                               # noqa: F841
        return eval(self.end)


# Spellings that keep every value (eval-mode dropout, copies, casts between
# float32 and float64). A float32 cast stands at the end, where the replay in
# float64 stays exact on dyadic values.
_KEEPS_VALUES = [
    # F.dropout(h, p=0.5, training=False) and F.dropout(h, 0.5,
    # training=self.training) in eval mode trace to the same node as the
    # first spelling (every argument a keyword), so one of them stands here.
    ("F.dropout(h, 0.5, False)", "y"),
    ("torch.dropout(h, 0.5, False)", "y"),
    ("F.alpha_dropout(h, 0.5)", "y"),
    ("self.drop(h)", "y"),
    ("h.double()", "y"),
    ("h.to(torch.float64)", "y"),
    ("h.type(torch.float64)", "y"),
    ("h.to(dtype=torch.float64)", "y"),
    ("h.to(torch.float64, non_blocking=True)", "y"),
    ("h.clone()", "y"),
    ("h.detach()", "y"),
    ("h.contiguous()", "y"),
    ("h.contiguous(memory_format=torch.contiguous_format)", "y"),
    ("h", "y.float()"),
    ("h", "y.to(torch.float32)"),
]

# Spellings that change values: a random dropout (training flag true, by
# default or literally) and casts to an integer or a narrower float type
# (cast back to float64 so the network is legal). Each must be refused.
_CHANGES_VALUES = [
    "F.dropout(h, 0.5)",
    "F.dropout(h, 0.5, True)",
    "F.dropout(h, training=True)",
    "torch.dropout(h, 0.5, True)",
    "F.alpha_dropout(h, 0.5, True)",
    "h.to(torch.int32).double()",
    "h.to(torch.int64).double()",
    "h.type(torch.int64).double()",
    "h.to(dtype=torch.int32).double()",
    "h.int().double()",
    "h.long().double()",
    "h.bool().double()",
    "h.half().double()",
    "h.to(torch.float16).double()",
    "h.to(torch.bfloat16).double()",
]


_KEEPS_CELLS = (
    [(middle, end, "Star") for middle, end in _KEEPS_VALUES]
    + [("F.dropout(h, 0.5, False)", "y", "Zono"), ("h.double()", "y", "Box"),
       ("h", "y.float()", "Zono")])


@pytest.mark.parametrize("middle,end,kind", _KEEPS_CELLS)
def test_value_keeping_dropout_and_casts_pass_the_sets_on(middle, end, kind):
    """Dropout in inference (the training flag literally False: torch.fx
    records every ``F.dropout`` and ``F.alpha_dropout`` spelling, ``training=
    self.training`` in eval mode included, with the flag as a keyword, and
    ``torch.dropout`` with it in position 2), ``nn.Dropout`` in eval mode,
    copies and float32/float64 casts (``to`` and ``type``, the dtype by
    position and by keyword) between two layers: the true network is
    unchanged, so every replayed output must be in the result. Catches a pass-through that hands on the sets of another
    node and a value-keeping spelling that is refused or mis-read (a
    keyword such as ``non_blocking`` or ``memory_format`` taken for a
    dtype)."""
    model = _dyadic_module(_Middle(middle, end), seed=31)
    lo, hi = flat_box(4, seed=5)
    sets, points = _flat_inputs(kind, "several" if kind == "Box" else "box",
                                lo, hi)
    what = f"{middle} / {end} on a {kind}"
    out = reach_model(model, sets, "approx", what)
    assert_covered(out, replay_torch(model, points, (1, 4)), what)


@pytest.mark.parametrize("middle", _CHANGES_VALUES)
def test_value_changing_dropout_and_casts_are_refused(middle):
    """Dropout with its training flag true (random) and casts to an integer,
    boolean or half-precision type (which truncate or round) must be refused
    with NotImplementedError, never passed through as the identity. Catches
    a dtype test that accepts every dtype, ``half`` added to the identity
    methods, and a training flag that is not read. Which branch of
    ``_is_inference_identity_function`` each spelling reaches: torch.fx
    records every ``F.dropout`` and ``F.alpha_dropout`` call with
    ``training`` as a keyword (filled in when the forward leaves it out), so
    those take the keyword branch; ``torch.dropout`` keeps its flag in
    position 2 and takes the positional branch; no traced spelling reaches
    the default branch."""
    model = _dyadic_module(_Middle(middle, "y"), seed=32)
    lo, hi = flat_box(4, seed=5)
    with pytest.raises(NotImplementedError):
        reach_model(model, flat_set("Star", lo, hi), "approx", middle)


# --------------------------------------------------------------------------- #
# 4. A Softmax module over one axis of a flat set routed to the axis rule      #
# --------------------------------------------------------------------------- #
class _AxisSoftmax(nn.Module):
    """fc1 (6 -> 6) -> view (1, 2, 3) -> Softmax over one axis -> flatten ->
    fc2 (6 -> 3)."""

    def __init__(self, dim):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(6, 6), nn.Linear(6, 3)
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        h = self.fc1(x).view(1, 2, 3)
        return self.fc2(self.softmax(h).flatten(1))


@pytest.mark.parametrize("kind,form", [("Star", "box"), ("Star", "row"),
                                       ("Zono", "box"), ("Box", "several")])
@pytest.mark.parametrize("dim", [-1, 1])
def test_softmax_module_over_one_axis_of_a_flat_set(dim, kind, form):
    """A torch ``nn.Softmax`` over one axis of a (1, 2, 3) tensor carried as a
    flat set: every row of the axis is normalised on its own. Catches a walk
    that sends the module to the generic Softmax rule (one softmax over all
    six entries) and an axis read on the wrong tensor shape."""
    model = _dyadic_module(_AxisSoftmax(dim), seed=33)
    lo, hi = flat_box(6, seed=6)
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"Softmax(dim={dim}) of a (1, 2, 3) view on a {form} {kind}"
    out = reach_model(model, sets, "approx", what)
    assert_covered(out, replay_torch(model, points, (1, 6)), what)


# --------------------------------------------------------------------------- #
# 5. ONNX graphs: a value read by two consumers, joined again                  #
# --------------------------------------------------------------------------- #
# Add pairs the operands as Sub does (with the other sign); it is kept in the
# exact cell only.
_JOINS = ("Sub", "Concat", "Add-same-node")


def _join_model(tmp_path, join, rng, name="join"):
    """``(path, weights)``: X (1, 4) -> Gemm = H; R = Relu(H); A = Gemm(H);
    J = join(R, A); Y = Gemm(J). A last node D = Gemm(R) of Y's width runs
    after Y and is not an output."""
    w = {"W1": _dyadic(rng, 4, 4), "b1": _dyadic(rng, 4),
         "Wa": _dyadic(rng, 4, 4), "ba": _dyadic(rng, 4),
         "W2": _dyadic(rng, 3, 8 if join == "Concat" else 4),
         "b2": _dyadic(rng, 3), "W3": _dyadic(rng, 3, 4), "b3": _dyadic(rng, 3)}
    if join == "Concat":
        joined = helper.make_node("Concat", ["R", "A"], ["J"], axis=1)
    elif join == "Add-same-node":
        joined = helper.make_node("Add", ["R", "R"], ["J"])
    else:
        joined = helper.make_node(join, ["R", "A"], ["J"])
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["H"], transB=1),
             helper.make_node("Relu", ["H"], ["R"]),
             helper.make_node("Gemm", ["H", "Wa", "ba"], ["A"], transB=1),
             joined,
             helper.make_node("Gemm", ["J", "W2", "b2"], ["Y"], transB=1),
             helper.make_node("Gemm", ["R", "W3", "b3"], ["D"], transB=1)]
    return save_onnx(tmp_path, nodes, [1, 4], [1, 3], inits=w,
                     name=name), w


_JOIN_INPUTS = [("Star", "box"), ("Star", "row"), ("Star", "several"),
                ("Zono", "box"), ("Box", "several")]


@pytest.mark.parametrize("kind,form", _JOIN_INPUTS)
@pytest.mark.parametrize("join", _JOINS)
def test_onnx_fan_out_and_join_hold_every_output(tmp_path, join, kind, form):
    """An ONNX graph in which H feeds a ReLU and a Gemm, the two branches are
    joined by Sub, Add or Concat (or one branch is added to itself), and a
    node that is not an output runs after the output node. Catches a layer
    that reads the sets computed last instead of its own input's, a join
    that pairs the wrong nodes' sets, a walk that returns the last node's
    sets, and a predicate tagging that is skipped, inverted or appended in
    the wrong place (the two branches then share or lose variables)."""
    path, _ = _join_model(tmp_path, join, np.random.default_rng(41))
    lo, hi = flat_box(4, seed=7)
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"Gemm/(Relu, Gemm)/{join}/Gemm on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, (1, 4)), what)


def _exact_join_model(tmp_path, join, rng):
    """``(path, weights)``: X (1, 4) -> Gemm = H; A = Gemm(H); J = join(H, A)
    (or H + H); Y = Gemm(Relu(J)). The join comes before the ReLU, so both
    operands carry the input's predicate variables: a join of two different
    predicate systems under the exact method over-approximates (registry
    REACH-4, the pairing of operand lists; not this file's subject)."""
    w = {"W1": _dyadic(rng, 4, 4), "b1": _dyadic(rng, 4),
         "Wa": _dyadic(rng, 4, 4), "ba": _dyadic(rng, 4),
         "W2": _dyadic(rng, 3, 8 if join == "Concat" else 4),
         "b2": _dyadic(rng, 3)}
    if join == "Concat":
        joined = helper.make_node("Concat", ["H", "A"], ["J"], axis=1)
    elif join == "Add-same-node":
        joined = helper.make_node("Add", ["H", "H"], ["J"])
    else:
        joined = helper.make_node(join, ["H", "A"], ["J"])
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["H"], transB=1),
             helper.make_node("Gemm", ["H", "Wa", "ba"], ["A"], transB=1),
             joined, helper.make_node("Relu", ["J"], ["R"]),
             helper.make_node("Gemm", ["R", "W2", "b2"], ["Y"], transB=1)]
    return save_onnx(tmp_path, nodes, [1, 4], [1, 3], inits=w,
                     name="exact-join"), w


@pytest.mark.parametrize("join", _JOINS + ("Add",))
def test_exact_onnx_join_is_the_exact_image(tmp_path, join):
    """Exact method: H and Gemm(H) joined by Sub, Add or Concat (or H added to
    itself), then a ReLU and a Gemm, on a Star with three predicate
    variables: both directions against the exact image in rationals.
    Catches a join handed the wrong node's sets, operands in the wrong
    order, and a ReLU after the join that the exact method does not split."""
    # Seed 43: on 3 of 20 seeds (42, 51, 61) a neuron's range on a piece of
    # the exact split ends exactly at 0 and the exact ReLU kernel relaxes it
    # within its LP pad (vertices 1e-7 outside the image), a defect of that
    # kernel reported by this file's author, not the subject of this cell.
    rng = np.random.default_rng(43)
    path, w = _exact_join_model(tmp_path, join, rng)
    star, x0, E = _low_rank_star(rng, 4)
    what = f"exact Gemm/(id, Gemm)/{join}/Relu/Gemm"
    out = reach_model(load_onnx(path), star, "exact", what)
    f = {k: v.astype(np.float64) for k, v in w.items()}
    eye = np.eye(4)
    if join == "Sub":
        M, c = eye - f["Wa"], -f["ba"]
    elif join == "Add":
        M, c = eye + f["Wa"], f["ba"]
    elif join == "Concat":
        M, c = np.vstack([eye, f["Wa"]]), np.concatenate([np.zeros(4),
                                                          f["ba"]])
    else:
        M, c = 2 * eye, np.zeros(4)
    pieces = rational.affine(rational.box([-1] * 3, [1] * 3), E.tolist(),
                             x0.tolist())
    pieces = rational.affine(pieces, f["W1"].tolist(), f["b1"].tolist())
    pieces = rational.relu(rational.affine(pieces, M.tolist(), c.tolist()))
    pieces = rational.affine(pieces, f["W2"].tolist(), f["b2"].tolist())
    _assert_exact(out, pieces, what)
    assert_covered(out, replay_onnx(path, _low_rank_points(x0, E, 14),
                                    (1, 4)), what)


@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_onnx_join_on_a_hexatope_or_octatope_is_refused(tmp_path, kind):
    """Hexatope and Octatope have no rule for a join of two branches: the
    graph must be refused with NotImplementedError, never answered with one
    branch's sets."""
    rng = np.random.default_rng(41)
    nodes = [helper.make_node("Gemm", ["X", "Wa", "ba"], ["A"], transB=1),
             helper.make_node("Sub", ["X", "A"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 4], [1, 4],
                     inits={"Wa": _dyadic(rng, 4, 4), "ba": _dyadic(rng, 4)})
    lo, hi = flat_box(4, seed=7)
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                    f"join on a {kind}")


def _mirror_model(tmp_path, act, join, rng):
    """X (1, 3) -> Gemm(W) and Gemm(-W) (same input ranges per neuron, the
    input box being symmetric about 0) -> the activation on each -> join ->
    Gemm. The two activations add new predicate variables with equal bounds
    on each neuron, so the two branches look alike unless each node's new
    variables are told apart."""
    W = _dyadic(rng, 3, 3)
    w = {"W": W, "Wn": -W, "W2": _dyadic(rng, 2, 6 if join == "Concat" else 3),
         "b2": _dyadic(rng, 2)}
    attrs = {"alpha": 0.25} if act == "LeakyRelu" else {}       # Sign, Relu
    joined = (helper.make_node("Concat", ["P", "Q"], ["J"], axis=1)
              if join == "Concat" else
              helper.make_node(join, ["P", "Q"], ["J"]))
    nodes = [helper.make_node("MatMul", ["X", "Wt"], ["A"]),
             helper.make_node("MatMul", ["X", "Wnt"], ["B"]),
             helper.make_node(act, ["A"], ["P"], **attrs),
             helper.make_node(act, ["B"], ["Q"], **attrs),
             joined,
             helper.make_node("Gemm", ["J", "W2", "b2"], ["Y"], transB=1)]
    inits = {"Wt": W.T.copy(), "Wnt": (-W).T.copy(), "W2": w["W2"],
             "b2": w["b2"]}
    return save_onnx(tmp_path, nodes, [1, 3], [1, 2], inits=inits,
                     name=f"mirror-{act}-{join}")


@pytest.mark.parametrize("kind,form", [("Star", "box"), ("Star", "row"),
                                       ("Star", "several")])
@pytest.mark.parametrize("act,join", [("Sign", "Sub"), ("Sign", "Concat"),
                                      ("Relu", "Sub"), ("Relu", "Concat"),
                                      ("LeakyRelu", "Sub")])
def test_join_of_two_branches_with_equal_new_variables(tmp_path, act, join,
                                                       kind, form):
    """Sign(W x) - Sign(-W x) (which is 2 Sign(W x)), its Concat, and the
    same with Relu and LeakyRelu, on an input box symmetric about 0: the two
    activations append new predicate variables with equal bounds (for Sign
    a box-only variable in [-1, 1] on every neuron that crosses 0, with no
    row, so the two branches' systems are identical). A join that reads the
    two branches' variables as the same (because the graph was not seen to
    join two nodes, or the new variables were not tagged) collapses the
    difference to 0 and misses the outputs. Catches a join test that needs
    three sources and a tag that is skipped or inverted."""
    path = _mirror_model(tmp_path, act, join, np.random.default_rng(44))
    lo, hi = np.array([-1.0, -0.5, -0.75]), np.array([1.0, 0.5, 0.75])
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"{act}(Wx) {join} {act}(-Wx) on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, (1, 3)), what)


@pytest.mark.parametrize("form", ["box", "several"])
def test_join_of_sin_and_cos_of_one_value(tmp_path, form):
    """Gemm(Sin(x) - Cos(x)) on a Star whose range holds an extremum of both
    functions, so each appends a box-only variable in [-1, 1]: the two look
    alike and must be told apart at the join. Float64 graph; the reference
    is numpy in float64 (onnxruntime has no float64 Sin or Cos). Catches the
    join bugs of the mirror cell on a smooth pair."""
    W = np.array([[1.0, -0.5], [0.25, 2.0]])
    b = np.array([0.5, -1.0])
    nodes = [helper.make_node("Sin", ["X"], ["S"]),
             helper.make_node("Cos", ["X"], ["C"]),
             helper.make_node("Sub", ["S", "C"], ["T"]),
             helper.make_node("Gemm", ["T", "W", "b"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 2], inits={"W": W, "b": b},
                     in_type="DOUBLE", out_type="DOUBLE", name="sin-cos")
    lo, hi = np.array([-4.0, -5.0]), np.array([4.0, 3.0])
    sets, points = _flat_inputs("Star", form, lo, hi)
    what = f"Gemm(Sin(x) - Cos(x)) on a {form} Star"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, [W @ (np.sin(p) - np.cos(p)) + b for p in points],
                   what)


def _image_inputs(kind, form, lo, hi):
    """``(sets, member points in (C, H, W) order)`` of an image kind: "box",
    "row" (a binding row that is not a box row), "several" (three sets whose
    union is the box), or "rows-only" (the box as an ImageStar whose box
    rows are constraint rows and which has no predicate box)."""
    from tests.cell_support import constrained_image_star
    if form == "box":
        return [image_set(kind, lo, hi)], samples(lo, hi)
    if form == "row":
        s, points = constrained_image_star(lo, hi)
        return [s], points
    if form == "rows-only":
        s = image_set("ImageStar", lo, hi)
        k = int(s.nVar)
        bare = ImageStar(s.V, np.vstack([np.eye(k), -np.eye(k)]),
                         np.ones((2 * k, 1)), None, None, height=s.height,
                         width=s.width, num_channels=s.num_channels)
        return [bare], samples(lo, hi)
    return image_split_cover(kind, lo, hi), samples(lo, hi)


_IMAGE_INPUTS = [("ImageStar", "box"), ("ImageStar", "row"),
                 ("ImageStar", "several"), ("ImageStar", "rows-only"),
                 ("ImageZono", "box"), ("ImageZono", "several")]


@pytest.mark.parametrize("join,kind,form", (
    [("Sub-Relu-LeakyRelu", kind, form) for kind, form in _IMAGE_INPUTS]
    + [("Add-Relu-input", kind, form) for kind, form in (
        ("ImageStar", "row"), ("ImageStar", "rows-only"),
        ("ImageZono", "several"))]))
def test_onnx_join_of_two_image_branches_holds_every_output(tmp_path, join,
                                                            kind, form):
    """A (2, 3, 5) image through a 1x1 Conv, then Relu and LeakyRelu (or
    Relu and the Conv output itself) joined by Sub (or Add), flattened and a
    Gemm. Catches the plumbing and tagging bugs of the flat join on image
    sets, including the tag variable appended to an ImageStar, with and
    without a predicate box (the tag's bounds then go in two rows)."""
    rng = np.random.default_rng(43)
    second = ("L" if join.startswith("Sub") else "A")
    nodes = [helper.make_node("Conv", ["X", "K", "kb"], ["A"]),
             helper.make_node("Relu", ["A"], ["R"]),
             helper.make_node("LeakyRelu", ["A"], ["L"], alpha=0.25),
             helper.make_node(join.split("-")[0], ["R", second], ["J"]),
             helper.make_node("Reshape", ["J", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W", "b"], ["Y"], transB=1)]
    inits = {"K": _dyadic(rng, 2, 2, 1, 1), "kb": _dyadic(rng, 2),
             "F": np.array([1, -1], dtype=np.int64),
             "W": _dyadic(rng, 3, 30), "b": _dyadic(rng, 3)}
    path = save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, 3], inits=inits)
    lo, hi = pixel_box(2, 3, 5, seed=11)
    sets, points = _image_inputs(kind, form, lo, hi)
    what = f"Conv/{join}/Gemm on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, (1, 2, 3, 5)), what)


# --------------------------------------------------------------------------- #
# 6. The outputs of a Split, each read by the node that names it               #
# --------------------------------------------------------------------------- #
# name: (opset, split sizes, how the parts are used). ``perm`` lists, for each
# entry of the concatenation, the entry of H it is; ``relu`` the positions of
# the concatenation that pass a ReLU.
_SPLITS = {
    "sizes-1-2-3": (13, [1, 2, 3], ["Q:P2", "P0", "P1"],
                    [3, 4, 5, 0, 1, 2], {0, 1, 2}),
    "even-2-2-2": (13, [2, 2, 2], ["P2", "Q:P0", "P1"],
                   [4, 5, 0, 1, 2, 3], {2, 3}),
    "attribute-3-1-2": (11, [3, 1, 2], ["P1", "Q:P2", "P0"],
                        [3, 4, 5, 0, 1, 2], {1, 2}),
}


def _split_model(tmp_path, case, rng, name="split"):
    """X (1, 6) -> Gemm = H -> Split -> P0, P1, P2; the parts in the order of
    the case, one through a ReLU, concatenated, then a Gemm."""
    opset, sizes, use, _, _ = _SPLITS[case]
    w = {"W1": _dyadic(rng, 6, 6), "b1": _dyadic(rng, 6),
         "W2": _dyadic(rng, 3, 6), "b2": _dyadic(rng, 3)}
    inits = dict(w)
    if opset >= 13:
        inits["SP"] = np.array(sizes, dtype=np.int64)
        split = helper.make_node("Split", ["H", "SP"], ["P0", "P1", "P2"],
                                 axis=1)
    else:
        split = helper.make_node("Split", ["H"], ["P0", "P1", "P2"], axis=1,
                                 split=sizes)
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["H"], transB=1),
             split]
    parts = []
    for u in use:
        if u.startswith("Q:"):
            nodes.append(helper.make_node("Relu", [u[2:]], ["Q"]))
            parts.append("Q")
        else:
            parts.append(u)
    nodes += [helper.make_node("Concat", parts, ["J"], axis=1),
              helper.make_node("Gemm", ["J", "W2", "b2"], ["Y"], transB=1)]
    return save_onnx(tmp_path, nodes, [1, 6], [1, 3], inits=inits,
                     opset=opset, name=name), w


@pytest.mark.parametrize("kind,form", _JOIN_INPUTS)
@pytest.mark.parametrize("case", list(_SPLITS))
def test_each_split_output_reaches_the_node_that_names_it(tmp_path, case,
                                                          kind, form):
    """ONNX Split of a 6-entry vector into three parts (unequal sizes, equal
    sizes, sizes as an attribute of opset 11), the parts concatenated in
    another order with one of them through a ReLU, then a Gemm. Catches a
    getitem that gives a node the sets of another output of the Split
    (shifted index: equal sizes give a set of the right size with the wrong
    entries) and parts concatenated in the wrong order."""
    path, _ = _split_model(tmp_path, case, np.random.default_rng(51))
    lo, hi = flat_box(6, seed=8)
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"Split {case}/Concat/Gemm on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, (1, 6)), what)


@pytest.mark.parametrize("case", ["even-2-2-2"])
def test_exact_split_outputs_are_the_exact_image(tmp_path, case):
    """Exact method: the parts of the Split concatenated in the case's order
    with no layer in between, then a ReLU and a Gemm, on a Star with three
    predicate variables: both directions against the exact image in
    rationals. (A ReLU on one part before the Concat would join pieces of
    two predicate systems, which the exact method over-approximates:
    registry REACH-4, the pairing of operand lists, not this file's
    subject.) Catches a getitem that hands a node another output's sets;
    the parts have equal sizes, so a shifted index keeps every size. (The
    other split cases are covered by the approx cell; each exact case costs
    two seconds.)"""
    rng = np.random.default_rng(52)
    opset, sizes, use, perm, _ = _SPLITS[case]
    w = {"W1": _dyadic(rng, 6, 6), "b1": _dyadic(rng, 6),
         "W2": _dyadic(rng, 3, 6), "b2": _dyadic(rng, 3)}
    inits = dict(w)
    if opset >= 13:
        inits["SP"] = np.array(sizes, dtype=np.int64)
        split = helper.make_node("Split", ["H", "SP"], ["P0", "P1", "P2"],
                                 axis=1)
    else:
        split = helper.make_node("Split", ["H"], ["P0", "P1", "P2"], axis=1,
                                 split=sizes)
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["H"], transB=1),
             split,
             helper.make_node("Concat", [u.replace("Q:", "") for u in use],
                              ["C"], axis=1),
             helper.make_node("Relu", ["C"], ["J"]),
             helper.make_node("Gemm", ["J", "W2", "b2"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, [1, 6], [1, 3], inits=inits,
                     opset=opset, name="exact-split")
    star, x0, E = _low_rank_star(rng, 6)
    what = f"exact Split {case}/Concat/Relu/Gemm"
    out = reach_model(load_onnx(path), star, "exact", what)
    pieces = rational.affine(rational.box([-1] * 3, [1] * 3), E.tolist(),
                             x0.tolist())
    pieces = rational.affine(pieces, w["W1"].astype(float).tolist(),
                             w["b1"].astype(float).tolist())
    pieces = rational.relu(rational.affine(pieces, np.eye(6)[perm].tolist()))
    pieces = rational.affine(pieces, w["W2"].astype(float).tolist(),
                             w["b2"].astype(float).tolist())
    _assert_exact(out, pieces, what)
    assert_covered(out, replay_onnx(path, _low_rank_points(x0, E, 14),
                                    (1, 6)), what)


# Image splits: (axis, sizes, how the parts are used after the split).
_IMAGE_SPLITS = {
    "height-1-2-second-part": (2, [1, 2]),
    "channels-1-1-subtracted": (1, [1, 1]),
    "width-3-2-swapped": (3, [3, 2]),
}


@pytest.mark.parametrize("case,kind,form", (
    [(case, kind, "box") for case in _IMAGE_SPLITS
     for kind in ("ImageStar", "ImageZono")]
    + [("channels-1-1-subtracted", "ImageStar", "several")]))
def test_each_split_output_of_an_image_reaches_the_node_that_names_it(
        tmp_path, case, kind, form):
    """ONNX Split of a (2, 3, 5) image along the height (the second part
    through a ReLU), the channels (second part minus first) or the width
    (the parts concatenated in swapped order), then a Gemm. Catches a
    getitem that gives a node another output's sets, on image sets."""
    rng = np.random.default_rng(53)
    axis, sizes = _IMAGE_SPLITS[case]
    nodes = [helper.make_node("Split", ["X", "SP"], ["P0", "P1"], axis=axis)]
    if case.startswith("height"):
        nodes.append(helper.make_node("Relu", ["P1"], ["J"]))
        n_out = 2 * 2 * 5
    elif case.startswith("channels"):
        nodes.append(helper.make_node("Sub", ["P1", "P0"], ["J"]))
        n_out = 15
    else:
        nodes.append(helper.make_node("Concat", ["P1", "P0"], ["J"], axis=3))
        n_out = 30
    nodes += [helper.make_node("Reshape", ["J", "F"], ["G"]),
              helper.make_node("Gemm", ["G", "W", "b"], ["Y"], transB=1)]
    inits = {"SP": np.array(sizes, dtype=np.int64),
             "F": np.array([1, -1], dtype=np.int64),
             "W": _dyadic(rng, 3, n_out), "b": _dyadic(rng, 3)}
    path = save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, 3], inits=inits)
    lo, hi = pixel_box(2, 3, 5, seed=12)
    sets, points = _image_inputs(kind, form, lo, hi)
    what = f"Split {case}/Gemm on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, (1, 2, 3, 5)), what)


# --------------------------------------------------------------------------- #
# 7. Constant subgraphs, a second graph input, Pad's value, Resize's factors   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,form", [("Star", "box"), ("Star", "row"),
                                       ("Zono", "box"), ("Box", "several")])
def test_constant_subgraphs_are_folded_to_their_values(tmp_path, kind, form):
    """A weight given as Transpose of an initializer, a bias given as the
    second output of a Split of a constant times a constant, and a Reshape
    target given as a Concat of two constants. Catches a constant folded to
    another value (the wrong output of the constant Split, the operands of
    a constant op in the wrong order) and a constant subgraph taken for a
    set-valued node."""
    rng = np.random.default_rng(61)
    nodes = [helper.make_node("Transpose", ["Wt"], ["WT"], perm=[1, 0]),
             helper.make_node("MatMul", ["X", "WT"], ["M"]),
             helper.make_node("Split", ["Bc", "SP"], ["B0", "B1"], axis=0),
             helper.make_node("Mul", ["B1", "Two"], ["B2"]),
             helper.make_node("Add", ["M", "B2"], ["L"]),
             helper.make_node("Relu", ["L"], ["R"]),
             helper.make_node("Concat", ["C1", "C2"], ["SH"], axis=0),
             helper.make_node("Reshape", ["R", "SH"], ["G"]),
             helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"], transB=1)]
    inits = {"Wt": _dyadic(rng, 3, 6), "Bc": _dyadic(rng, 6),
             "SP": np.array([3, 3], dtype=np.int64),
             "Two": np.array([2.0], dtype=np.float32),
             "C1": np.array([1], dtype=np.int64),
             "C2": np.array([-1], dtype=np.int64),
             "W2": _dyadic(rng, 2, 3), "b2": _dyadic(rng, 2)}
    path = save_onnx(tmp_path, nodes, [1, 6], [1, 2], inits=inits)
    lo, hi = flat_box(6, seed=9)
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"constant subgraphs on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, (1, 6)), what)


def test_a_graph_with_two_inputs_is_refused(tmp_path):
    """One input set cannot stand for two graph inputs: reading the set for
    both makes X1 - X2 a single point. The graph must be refused with
    NotImplementedError."""
    graph = helper.make_graph(
        [helper.make_node("Sub", ["X1", "X2"], ["Y"])], "two",
        [helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 4]),
         helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = str(tmp_path / "two.onnx")
    onnx.save(model, path)
    lo, hi = flat_box(4)
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), flat_set("Star", lo, hi), "approx",
                    "two graph inputs")


def _pad_model(tmp_path, value, rng):
    """X (1, 2, 3, 5) -> Pad (one row on top, two columns on the right, the
    constant ``value``, or the input left out) -> Reshape -> Gemm."""
    inputs = ["X", "pads"] + ([] if value is None else ["v"])
    inits = {"pads": np.array([0, 0, 1, 0, 0, 0, 0, 2], dtype=np.int64),
             "F": np.array([1, -1], dtype=np.int64),
             "W": _dyadic(rng, 3, 2 * 4 * 7), "b": _dyadic(rng, 3)}
    if value is not None:
        inits["v"] = np.array(value, dtype=np.float32)
    nodes = [helper.make_node("Pad", inputs, ["P"], mode="constant"),
             helper.make_node("Reshape", ["P", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W", "b"], ["Y"], transB=1)]
    return save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, 3], inits=inits)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("value", [None, 0.0])
def test_zero_pad_holds_every_output(tmp_path, value, kind):
    """ONNX Pad with the constant value left out or given as 0, asymmetric
    pads on a (2, 3, 5) image: the passing neighbour of the refusal below,
    with the same model, replay and check. Catches pads read from the wrong
    input."""
    path = _pad_model(tmp_path, value, np.random.default_rng(62))
    lo, hi = pixel_box(2, 3, 5, seed=13)
    what = f"Pad value {value} on a {kind}"
    out = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                      what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), (1, 2, 3, 5)),
                   what)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("value", [0.5, -0.25])
def test_pad_with_a_nonzero_constant_is_refused(tmp_path, value, kind):
    """The Pad rule reaches only zero padding: a non-zero constant value must
    be refused with NotImplementedError, never answered as a zero pad.
    Catches a check of the value input that is skipped (an argument count
    off by one)."""
    path = _pad_model(tmp_path, value, np.random.default_rng(62))
    lo, hi = pixel_box(2, 3, 5, seed=13)
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                    f"Pad value {value}")


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("by", ["scales", "sizes"])
def test_resize_factors_reach_the_resize_rule(tmp_path, by, kind):
    """A nearest Resize (asymmetric coordinates, floor rounding: each pixel
    copied into a 2x2 block) of a (2, 3, 5) image to (2, 6, 10), the factors given
    by ``scales`` (with a region-of-interest tensor) or by ``sizes``, then
    Reshape and Gemm. The library must answer with a set that holds every
    onnxruntime output, or refuse with NotImplementedError (it refuses the
    ``sizes`` spelling today, in the Resize rule, after the walk has resolved
    the sizes, so the ``sizes`` cells check only that refusal). Catches scale
    factors read from the wrong input of the node."""
    rng = np.random.default_rng(63)
    if by == "scales":
        inputs = ["X", "roi", "sc"]
        inits = {"roi": np.zeros(0, dtype=np.float32),
                 "sc": np.array([1, 1, 2, 2], dtype=np.float32)}
    else:
        inputs = ["X", "", "", "sz"]
        inits = {"sz": np.array([1, 2, 6, 10], dtype=np.int64)}
    inits.update({"F": np.array([1, -1], dtype=np.int64),
                  "W": _dyadic(rng, 3, 120), "b": _dyadic(rng, 3)})
    nodes = [helper.make_node("Resize", inputs, ["R"], mode="nearest",
                              coordinate_transformation_mode="asymmetric",
                              nearest_mode="floor"),
             helper.make_node("Reshape", ["R", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W", "b"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, 3], inits=inits)
    lo, hi = pixel_box(2, 3, 5, seed=14)
    outputs = replay_onnx(path, samples(lo, hi), (1, 2, 3, 5))
    what = f"Resize by {by} on a {kind}"
    try:
        out = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                          what)
    except NotImplementedError:
        if by == "scales":
            raise                   # the scales spelling is answered today
        return
    assert_covered(out, outputs, what)


# --------------------------------------------------------------------------- #
# 8. ReduceSum and ReduceMean on a set that stands for a (2, 3, 5) tensor      #
# --------------------------------------------------------------------------- #
# name: (op, axes, keepdims, noop_with_empty_axes). Axes None: every axis.
_REDUCES = {
    "sum-channels": ("ReduceSum", [1], 1, 0),
    "sum-width-negative": ("ReduceSum", [-1], 0, 0),
    "sum-height-width": ("ReduceSum", [2, 3], 0, 0),
    "sum-empty-axes-noop": ("ReduceSum", [], 1, 1),
    "mean-channels-width": ("ReduceMean", [1, 3], 0, 0),
    "mean-height": ("ReduceMean", [2], 1, 0),
    "mean-negative-pair": ("ReduceMean", [-3, -2], 0, 0),
    "mean-every-axis": ("ReduceMean", None, 0, 0),
    # a mean over two entries: its divisor is exact in float64 (exact cell)
    "mean-channels": ("ReduceMean", [1], 0, 0),
}
_TENSOR = (2, 3, 5)


def _reduce_model(tmp_path, case, rng, tail_relu=False):
    """``(path, weights, n_reduced)``: a float64 graph X (1, 2, 3, 5) ->
    Reduce -> Reshape (1, -1) -> Gemm (3) [-> Relu -> Gemm (2)]. Float64, so
    that a mean over 3, 6, 10 or 30 entries is rounded at float64 precision
    in the replay."""
    op, axes, keep, noop = _REDUCES[case]
    attrs = {"keepdims": keep}
    inits = {"F": np.array([1, -1], dtype=np.int64)}
    inputs = ["X"]
    if op == "ReduceSum":
        attrs["noop_with_empty_axes"] = noop
        inits["A"] = np.array(axes, dtype=np.int64)
        inputs.append("A")
    elif axes is not None:
        attrs["axes"] = axes
    if axes == [] and noop:
        n = int(np.prod(_TENSOR))                   # the identity
    elif not axes:
        n = 1                                       # every axis reduced
    else:
        n = int(np.size(np.zeros((1,) + _TENSOR).sum(axis=tuple(axes))))
    w = {"W2": _dyadic(rng, 3, n).astype(np.float64),
         "b2": _dyadic(rng, 3).astype(np.float64)}
    nodes = [helper.make_node(op, inputs, ["R"], **attrs),
             helper.make_node("Reshape", ["R", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"], transB=1)]
    out = 3
    if tail_relu:
        w.update({"W3": _dyadic(rng, 2, 3).astype(np.float64),
                  "b3": _dyadic(rng, 2).astype(np.float64)})
        nodes[-1] = helper.make_node("Gemm", ["G", "W2", "b2"], ["P"],
                                     transB=1)
        nodes += [helper.make_node("Relu", ["P"], ["Q"]),
                  helper.make_node("Gemm", ["Q", "W3", "b3"], ["Y"],
                                   transB=1)]
        out = 2
    path = save_onnx(tmp_path, nodes, (1,) + _TENSOR, [1, out],
                     inits=dict(inits, **w), in_type="DOUBLE",
                     out_type="DOUBLE", name=f"reduce-{case}")
    return path, w, n


def _reduce_matrix(case):
    """The reduction as a matrix of Fractions, from the ONNX definition:
    output entry j sums (or averages) the input entries whose coordinates
    agree with j on the kept axes, in row-major order of both tensors."""
    op, axes, _, noop = _REDUCES[case]
    rank = len(_TENSOR)
    if axes == [] and noop:
        gone = []
    elif axes is None or axes == []:
        gone = list(range(rank))
    else:
        # an axis of the batch-prefixed tensor, as an axis of _TENSOR
        gone = sorted({a % (rank + 1) - 1 for a in axes})
    kept = [i for i in range(rank) if i not in gone]
    count = int(np.prod([_TENSOR[i] for i in gone])) if gone else 1
    weight = Fraction(1, count) if op == "ReduceMean" else Fraction(1)
    groups = {}
    for i, coords in enumerate(np.ndindex(*_TENSOR)):
        groups.setdefault(tuple(coords[k] for k in kept), []).append(i)
    rows = []
    for key in sorted(groups):
        row = [Fraction(0)] * int(np.prod(_TENSOR))
        for i in groups[key]:
            row[i] = weight
        rows.append(row)
    return rows


def _check_reduce_matrix(tmp_path, case, A):
    """Setup check of the reference: the reduction alone, as a float64 ONNX
    graph, must map each unit vector to the column of ``A`` (up to the
    rounding of a quotient such as 1/3)."""
    op, axes, keep, noop = _REDUCES[case]
    attrs, inits, inputs = {"keepdims": keep}, {}, ["X"]
    if op == "ReduceSum":
        attrs["noop_with_empty_axes"] = noop
        inits["A"] = np.array(axes, dtype=np.int64)
        inputs.append("A")
    elif axes is not None:
        attrs["axes"] = axes
    inits["F"] = np.array([1, -1], dtype=np.int64)
    nodes = [helper.make_node(op, inputs, ["R"], **attrs),
             helper.make_node("Reshape", ["R", "F"], ["Y"])]
    path = save_onnx(tmp_path, nodes, (1,) + _TENSOR, [1, len(A)],
                     inits=inits, in_type="DOUBLE", out_type="DOUBLE",
                     name=f"reduce-alone-{case}")
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for k in range(int(np.prod(_TENSOR))):
        e = np.zeros((1,) + _TENSOR)
        e.reshape(-1)[k] = 1.0
        got = session.run(None, {"X": e})[0].reshape(-1)
        want = np.array([float(row[k]) for row in A])
        if not np.allclose(got, want, rtol=4e-16, atol=0):
            pytest.fail(f"{case}: the reference reduction disagrees with "
                        f"onnxruntime on unit vector {k}: {got} vs {want}")


def _reduce_inputs(kind, form, lo, hi):
    if kind.startswith("Image"):
        sets, points = _image_inputs(kind, form, lo, hi)
    else:
        sets, points = _flat_inputs(kind, form, lo, hi)
    return sets, points


_REDUCE_CELLS = (
    [(case, kind, form) for case in _REDUCES if case != "mean-channels"
     for kind, form in (("Star", "row"), ("Zono", "box"))]
    + [(case, kind, form) for case in ("sum-width-negative",
                                       "sum-empty-axes-noop",
                                       "mean-channels-width",
                                       "mean-every-axis")
       for kind, form in (("Box", "several"), ("ImageZono", "box"))]
    + [("mean-height", "ImageStar", "box")])


@pytest.mark.parametrize("case,kind,form", _REDUCE_CELLS)
def test_reduce_on_a_set_with_a_tensor_shape_holds_every_output(
        tmp_path, case, kind, form):
    """ReduceSum or ReduceMean over one axis, two axes, a negative axis, every
    axis, or no axis with ``noop_with_empty_axes``, on a set that stands for
    a (2, 3, 5) tensor (``input_shape`` given), then Reshape and Gemm.
    Catches a sum divided like a mean, a divisor taken from the kept shape or
    summed instead of multiplied over several axes, a negative axis made
    positive by its absolute value, an image set reduced in its stored order
    by the flat rule, and a tensor shape recorded without the batch axis
    (the axis then names the wrong dimension)."""
    path, _, _ = _reduce_model(tmp_path, case, np.random.default_rng(71))
    lo, hi = pixel_box(*_TENSOR, seed=15)
    sets, points = _reduce_inputs(kind, form, lo, hi)
    what = f"{case}/Gemm on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what,
                      input_shape=_TENSOR)
    assert_covered(out, replay_onnx(path, points, (1,) + _TENSOR), what)


_MEAN_ROUNDING = known_bug(
    "7b", "registry ROUND-8: the flat Reduce rule divides by the count (6, "
          "10) in float64 rounded to nearest and the Star affine map has no "
          "error term, so vertices of the exact image of a mean lie outside "
          "the exact result by one or two units in the last place")


@pytest.mark.parametrize("case", [
    "sum-channels", "sum-width-negative", "mean-channels",
    pytest.param("mean-channels-width", marks=_MEAN_ROUNDING),
    pytest.param("mean-negative-pair", marks=_MEAN_ROUNDING)])
def test_exact_reduce_is_the_exact_image(tmp_path, case):
    """Exact method: Reduce -> Gemm -> ReLU -> Gemm on a flat Star of 30
    entries with three predicate variables and ``input_shape`` (2, 3, 5):
    both directions against the exact image in rationals, the reduction
    written out from the ONNX definition in fractions (checked against
    onnxruntime in float64 on the unit vectors). Catches every range and
    regime bug of the reduce rule in exact arithmetic, where a mean's
    divisor of 6 or 10 is not rounded away. The two sums and the mean over
    the two channels (dyadic, exact in float64) are the passing neighbours
    of the two marked means: a wrong divisor of the mean path (a sum, the
    kept shape) shows on ``mean-channels`` while the marked means are
    absorbed by their marker."""
    rng = np.random.default_rng(72)
    path, w, n = _reduce_model(tmp_path, case, rng, tail_relu=True)
    A = _reduce_matrix(case)
    if len(A) != n:
        pytest.fail(f"{case}: the reduction matrix has {len(A)} rows, the "
                    f"model reduces to {n} entries")
    _check_reduce_matrix(tmp_path, case, A)
    star, x0, E = _low_rank_star(rng, 30)
    pieces = rational.affine(rational.box([-1] * 3, [1] * 3), E.tolist(),
                             x0.tolist())
    pieces = rational.affine(pieces, A)
    pieces = rational.affine(pieces, w["W2"].tolist(), w["b2"].tolist())
    pieces = rational.affine(rational.relu(pieces), w["W3"].tolist(),
                             w["b3"].tolist())
    what = f"exact {case}/Gemm/Relu/Gemm"
    out = reach_model(load_onnx(path), star, "exact", what,
                      input_shape=_TENSOR)
    _assert_exact(out, pieces, what)
    assert_covered(out, replay_onnx(path, _low_rank_points(x0, E, 14),
                                    (1,) + _TENSOR), what)


# --------------------------------------------------------------------------- #
# 9. Transpose of a set that stands for a (2, 3, 5) tensor                     #
# --------------------------------------------------------------------------- #
# One permutation per kind of value of the one row-selection path: a cycle
# (not its own inverse, so an inverted permutation shows), a swap of two
# axes, and no perm (the default, every axis reversed, a branch of its own).
_PERMS = {
    "cycle-0231": [0, 2, 3, 1],
    "swap-height-width": [0, 1, 3, 2],
    "reverse-all-axes": None,
}


def _transpose_model(tmp_path, case, rng, tail_relu=False):
    perm = _PERMS[case]
    attrs = {} if perm is None else {"perm": perm}
    w = {"W2": _dyadic(rng, 3, 30), "b2": _dyadic(rng, 3)}
    nodes = [helper.make_node("Transpose", ["X"], ["T"], **attrs),
             helper.make_node("Reshape", ["T", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W2", "b2"],
                              ["P" if tail_relu else "Y"], transB=1)]
    out = 3
    if tail_relu:
        w.update({"W3": _dyadic(rng, 2, 3), "b3": _dyadic(rng, 2)})
        nodes += [helper.make_node("Relu", ["P"], ["Q"]),
                  helper.make_node("Gemm", ["Q", "W3", "b3"], ["Y"],
                                   transB=1)]
        out = 2
    path = save_onnx(tmp_path, nodes, (1,) + _TENSOR, [1, out],
                     inits=dict(w, F=np.array([1, -1], dtype=np.int64)),
                     name=f"transpose-{case}")
    return path, w


_TRANSPOSE_CELLS = (
    [(case, kind, form) for case in _PERMS
     for kind, form in (("Star", "row"), ("Zono", "box"))]
    + [(case, kind, "box" if kind.startswith("Image") else "several")
       for case in ("cycle-0231", "swap-height-width", "reverse-all-axes")
       for kind in ("Box", "ImageStar", "ImageZono")])


@pytest.mark.parametrize("case,kind,form", _TRANSPOSE_CELLS)
def test_transpose_of_a_set_with_a_tensor_shape_holds_every_output(
        tmp_path, case, kind, form):
    """ONNX Transpose of a (2, 3, 5) tensor by a cyclic permutation (not its
    own inverse), a swap of two axes, or no ``perm`` (every axis reversed,
    the batch axis included), then Reshape and Gemm, on flat sets with
    ``input_shape`` and on image sets. The library must answer with a set
    that holds every onnxruntime output; an image set may be refused
    (NotImplementedError) only for a permutation that moves the batch axis.
    Catches rows permuted by the inverse permutation, an index array built
    in the permuted shape, a Zono centre left unpermuted, a missing ``perm``
    read wrongly, and an image set sent to the flat rule."""
    path, _ = _transpose_model(tmp_path, case, np.random.default_rng(81))
    lo, hi = pixel_box(*_TENSOR, seed=16)
    sets, points = _reduce_inputs(kind, form, lo, hi)
    outputs = replay_onnx(path, points, (1,) + _TENSOR)
    what = f"Transpose {case}/Gemm on a {form} {kind}"
    options = {} if kind == "ImageStar" else {"input_shape": _TENSOR}
    try:
        out = reach_model(load_onnx(path), sets, "approx", what, **options)
    except NotImplementedError:
        if kind.startswith("Image") and _PERMS[case] is None:
            return                  # a clear refusal keeps the promise
        raise
    assert_covered(out, outputs, what)


@pytest.mark.parametrize("case", ["cycle-0231", "reverse-all-axes"])
def test_exact_transpose_is_the_exact_image(tmp_path, case):
    """Exact method: Transpose -> Gemm -> ReLU -> Gemm on a flat Star of 30
    entries with three predicate variables and ``input_shape`` (2, 3, 5):
    both directions against the exact image in rationals, the permutation
    taken from numpy's transpose of the index array. Catches a permutation
    applied inverted (the same for a swap, wrong for a cycle)."""
    rng = np.random.default_rng(82)
    path, w = _transpose_model(tmp_path, case, rng, tail_relu=True)
    perm = _PERMS[case]
    order = np.transpose(np.arange(30).reshape((1,) + _TENSOR),
                         perm).reshape(-1)
    P = np.eye(30)[order]
    star, x0, E = _low_rank_star(rng, 30)
    what = f"exact Transpose {case}/Gemm/Relu/Gemm"
    out = reach_model(load_onnx(path), star, "exact", what,
                      input_shape=_TENSOR)
    pieces = rational.affine(rational.box([-1] * 3, [1] * 3), E.tolist(),
                             x0.tolist())
    pieces = rational.affine(pieces, (w["W2"].astype(float) @ P).tolist(),
                             w["b2"].astype(float).tolist())
    pieces = rational.affine(rational.relu(pieces),
                             w["W3"].astype(float).tolist(),
                             w["b3"].astype(float).tolist())
    _assert_exact(out, pieces, what)
    assert_covered(out, replay_onnx(path, _low_rank_points(x0, E, 14),
                                    (1,) + _TENSOR), what)


@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_transpose_on_a_hexatope_or_octatope_is_refused(tmp_path, kind):
    """Hexatope and Octatope have no row permutation: a Transpose must be
    refused with NotImplementedError, never answered with the input set."""
    path, _ = _transpose_model(tmp_path, "cycle-0231",
                               np.random.default_rng(81))
    lo, hi = pixel_box(*_TENSOR, seed=16)
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                    f"Transpose on a {kind}", input_shape=_TENSOR)


# --------------------------------------------------------------------------- #
# 10. BatchNorm fusion, in exact arithmetic                                    #
# --------------------------------------------------------------------------- #
class _LinearBatchNorm(nn.Module):
    """Linear (2 -> 4) then BatchNorm1d in eval mode. ``eps`` = 2^-10 and the
    running variance 9/16 - eps, so the normalisation divides by exactly 3/4
    in real arithmetic; with ``dyadic_scale`` False, gamma / (3/4) is not a
    dyadic number and folding it into the Linear rounds. The fourth neuron
    is pruned (a zero weight row) and its BatchNorm terms nearly cancel
    (about 1830 against -1830, leaving 2^-10): a folded bias rounded to
    nearest misses its real value by about 1e-13, far more than any
    rounding margin taken relative to the small result."""

    def __init__(self, dyadic_scale):
        super().__init__()
        self.fc = nn.Linear(2, 4)
        self.bn = nn.BatchNorm1d(4, eps=2.0 ** -10)
        gamma = ([0.75, -1.5, 0.375, 0.75] if dyadic_scale
                 else [1.0, 0.75, -1.25, 1.0978])
        mean = [0.1875, -0.375, 0.5, -1234.5]
        # beta of the pruned neuron: minus its scaled term, plus 2^-10, as a
        # float64 number (the scaled term itself is not one).
        cancel = float(-(Fraction(gamma[3]) / Fraction(3, 4))
                       * (Fraction(0.25) - Fraction(mean[3]))
                       + Fraction(1, 1024))
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[0.75, -0.5], [0.25, 1.0],
                                               [-1.0, 0.125], [0.0, 0.0]]))
            self.fc.bias.copy_(torch.tensor([0.5, -0.25, 0.125, 0.25]))
            self.bn.weight.copy_(torch.tensor(gamma))
            self.bn.bias.copy_(torch.tensor([0.0625, -0.5, 0.25, cancel]))
            self.bn.running_mean.copy_(torch.tensor(mean))
            self.bn.running_var.fill_(0.5625 - 2.0 ** -10)
        self.double().eval()

    def forward(self, x):
        return self.bn(self.fc(x))

    def exact_map(self):
        """``(A, c)`` of the network in fractions: gamma (W x + b - mean) /
        (3/4) + beta, the square root taken exactly."""
        root = Fraction(3, 4)
        if (Fraction(float(self.bn.running_var[0])) + Fraction(self.bn.eps)
                != root * root):
            pytest.fail("var + eps is not (3/4)^2 exactly")
        W = rational.matrix(self.fc.weight.detach().numpy().tolist())
        b = rational.vector(self.fc.bias.detach().numpy().tolist())
        g = rational.vector(self.bn.weight.detach().numpy().tolist())
        mean = rational.vector(self.bn.running_mean.detach().numpy().tolist())
        beta = rational.vector(self.bn.bias.detach().numpy().tolist())
        A = [[g[i] / root * W[i][j] for j in range(2)] for i in range(4)]
        c = [g[i] / root * (b[i] - mean[i]) + beta[i] for i in range(4)]
        return A, c


_FOLD_ROUNDING = known_bug(
    "7b", "registry ROUND-22: BatchNorm is folded into the preceding Linear "
          "before the reach for every set type but Box, and the folded bias "
          "scale (b - mean) + beta is rounded to nearest in float64 with no "
          "error term; where those terms cancel (about 1830 against -1830 on "
          "the pruned neuron, leaving 2^-10) the result holds 1.0103384654e-3 "
          "where the real output is 12713/12582912, a miss of 7.6e-14, far "
          "beyond the rounding radius DESIGN section 1 gives the fused row "
          "(3.4e-19)")
# A Hexatope and an Octatope also round their own affine map to nearest with
# no radius at all (ROUND-24), so their cells pass only when PLAN step 9 has
# bounded that rounding as well; the step of the component that fails last.
_FOLD_ROUNDING_HEXOCT = known_bug(
    9, "registry ROUND-22 and ROUND-24: the BatchNorm fold rounds the folded "
       "bias to nearest (a miss of 7.6e-14 on the pruned neuron whose terms "
       "cancel), and the Hexatope and Octatope affine map that applies the "
       "folded layer rounds to nearest with no error term")


@pytest.mark.parametrize("kind,dyadic_scale", [
    pytest.param("Box", False, id="Box-rounded-fold"),
    pytest.param("Star", True, id="Star-exact-fold"),
    pytest.param("Zono", True, id="Zono-exact-fold"),
    pytest.param("Hexatope", True, id="Hexatope-exact-fold"),
    pytest.param("Octatope", True, id="Octatope-exact-fold"),
    pytest.param("Star", False, id="Star-rounded-fold", marks=_FOLD_ROUNDING),
    pytest.param("Zono", False, id="Zono-rounded-fold", marks=_FOLD_ROUNDING),
    pytest.param("Hexatope", False, id="Hexatope-rounded-fold",
                 marks=_FOLD_ROUNDING_HEXOCT),
    pytest.param("Octatope", False, id="Octatope-rounded-fold",
                 marks=_FOLD_ROUNDING_HEXOCT),
])
def test_linear_then_batchnorm_holds_the_exact_image(kind, dyadic_scale):
    """Linear then BatchNorm1d, every vertex of the exact image of two input
    boxes (one straddling zero, one far from it) checked for membership in
    exact arithmetic, with zero slack. A Box keeps the BatchNorm unfused and
    its rule rounds outward; the other kinds reach a fused Linear. The cells
    with a dyadic scale (the fold is exact, for every kind that is fused) and
    the Box are the passing neighbours with the same exact map and check.
    Catches a Box sent through
    the fusion (its fold rounds to nearest, which the pruned neuron with
    cancelling terms shows), a fold that drops the mean or the shift, and
    the fold's own rounding."""
    model = _LinearBatchNorm(dyadic_scale)
    A, c = model.exact_map()
    for lo, hi in (([-1.0, -0.5], [0.75, 1.0]), ([0.125, -3.0],
                                                  [0.375, -2.5])):
        what = f"Linear/BatchNorm1d on a {kind} [{lo}, {hi}]"
        out = reach_model(model, flat_set(kind, np.array(lo), np.array(hi)),
                          "approx", what)
        pieces = rational.affine(rational.box(lo, hi), A, c)
        escaped = rational.escapes(pieces, *[rational.star(s) for s in out])
        assert not escaped, (f"{what}: {len(escaped)} vertices of the exact "
                             f"image are in no returned set, first "
                             f"{[float(v) for v in escaped[0][1]]}")
        assert_covered(out, replay_torch(model, samples(np.array(lo),
                                                        np.array(hi)),
                                         (1, 2)), what)


def _within(computed, y, delta):
    """Is some point of the computed set within ``delta[i]`` of ``y[i]`` in
    every entry i? Exact feasibility in fractions of the predicate rows and
    bounds together with |c + G a - y| <= delta."""
    c, G, C, d, lb, ub = computed
    n = len(lb)
    A, b = [list(row) for row in C], list(d)
    for j in range(n):
        unit = [Fraction(int(k == j)) for k in range(n)]
        if ub[j] is not None:
            A.append(unit)
            b.append(ub[j])
        if lb[j] is not None:
            A.append([-v for v in unit])
            b.append(-lb[j])
    for i, (ci, row) in enumerate(zip(c, G)):
        A.append(list(row))
        b.append(y[i] - ci + delta[i])
        A.append([-v for v in row])
        b.append(ci - y[i] + delta[i])
    return rational.feasible(A_ub=A, b_ub=b) is not None


_FOLD_BEYOND_RADIUS = known_bug(
    "7b", "registry ROUND-22: the folded bias of the pruned neuron misses its "
          "real value by 7.6e-14, beyond the rounding radius DESIGN section 1 "
          "states for the affine map of a Star (3.4e-19 on that entry): the "
          "fold is rounded before the reach and enters no error term")


@pytest.mark.parametrize("dyadic_scale", [
    pytest.param(True, id="exact-fold"),
    pytest.param(False, id="rounded-fold", marks=_FOLD_BEYOND_RADIUS)])
def test_linear_then_batchnorm_on_a_star_holds_the_image_within_the_radius(
        dyadic_scale):
    """The narrowed claim of DESIGN section 1 for the model of the cell
    above: a Star result may miss the exact image by the rounding radius of
    its affine map, delta_i = gamma_3 (|A_i| T + |c_i|) with gamma_3 =
    3 u / (1 - 3 u), u = 2^-53, and T the term size |centre| + |half-width|
    of the input box. Every corner image of the two boxes, in fractions from
    the stored parameters, must be within delta of the result in every
    entry. The fold of the BatchNorm into the Linear is model preprocessing,
    not the affine map, so its rounding must not exceed that radius either.
    The dyadic scale (an exact fold) is the passing neighbour. Catches the
    fold's rounding where the zero-slack cells above cannot tell it from the
    affine map's own (ROUND-8)."""
    model = _LinearBatchNorm(dyadic_scale)
    A, c = model.exact_map()
    u = Fraction(1, 2 ** 53)
    gamma3 = 3 * u / (1 - 3 * u)
    for lo, hi in (([-1.0, -0.5], [0.75, 1.0]), ([0.125, -3.0],
                                                  [0.375, -2.5])):
        what = f"Linear/BatchNorm1d on a Star [{lo}, {hi}], within the radius"
        out = reach_model(model, flat_set("Star", np.array(lo), np.array(hi)),
                          "approx", what)
        T = [abs(Fraction(a) + Fraction(b)) / 2 + (Fraction(b) - Fraction(a))
             / 2 for a, b in zip(lo, hi)]
        delta = [gamma3 * (sum(abs(a) * t for a, t in zip(row, T)) + abs(ci))
                 for row, ci in zip(A, c)]
        computed = [rational.star(s) for s in out]
        for x in ((lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]),
                  (hi[0], hi[1])):
            y = [sum(a * Fraction(v) for a, v in zip(row, x)) + ci
                 for row, ci in zip(A, c)]
            assert any(_within(s, y, delta) for s in computed), (
                f"{what}: the exact output {[float(v) for v in y]} at the "
                f"corner {x} is farther than the radius from every result")


# --------------------------------------------------------------------------- #
# 11. The model entry point: method, set type and configuration                #
# --------------------------------------------------------------------------- #
def _small_relu_net():
    return _dyadic_module(nn.Sequential(nn.Linear(3, 3), nn.ReLU(),
                                        nn.Linear(3, 2)), seed=91)


@pytest.mark.parametrize("kind,method", [
    ("Zono", "exact"), ("Box", "exact"), ("Hexatope", "exact"),
    ("Octatope", "exact"), ("Star", "bogus"), ("Zono", "bogus"),
])
def test_entry_refuses_a_method_the_set_type_has_no_rule_for(kind, method):
    """``reach_pytorch_model`` refuses a method that is not valid for the
    set type: an exact reach of a Zono, Box, Hexatope or Octatope, and an
    unknown method name. Catches a method check that lets the call through
    (the approx rules would then answer as exact). An unknown name is an
    unreadable input, a ``ValueError``. For "exact" on a set type without
    exact rules this case holds only that the call is refused (today
    ``ValueError``, '... only supports approx'; the planned
    ``UnsupportedError`` subclasses ``NotImplementedError``): which of the
    two is right is decided in ``tests/contracts/test_reach_coverage.py::
    test_exact_on_a_set_type_without_exact_rules_is_refused``."""
    lo, hi = flat_box(3)
    refusal = (ValueError if method == "bogus"
               else (ValueError, NotImplementedError))
    with pytest.raises(refusal):
        reach_model(_small_relu_net(), flat_set(kind, lo, hi), method,
                    f"{method} on a {kind}")


def test_entry_refuses_an_exact_reach_of_an_image_zono():
    """An ImageZono is a Zono: an exact reach must be refused (today
    ``ValueError``; the type is decided in the contract file named in the
    test above)."""
    lo, hi = pixel_box(1, 2, 3)
    model = _dyadic_module(nn.Sequential(nn.Flatten(), nn.Linear(6, 2)), 92)
    with pytest.raises((ValueError, NotImplementedError)):
        reach_model(model, image_set("ImageZono", lo, hi), "exact",
                    "exact on an ImageZono")


@pytest.mark.parametrize("method", ["conformal", "flow_matching"])
def test_entry_refuses_a_probabilistic_method_on_a_star(method):
    """The probabilistic methods take a Box: a Star must raise
    ``TypeError`` before any sampling."""
    lo, hi = flat_box(3)
    with pytest.raises(TypeError):
        reach_model(_small_relu_net(), flat_set("Star", lo, hi), method,
                    f"{method} on a Star")


@pytest.mark.parametrize("bad", ["config-and-keyword", "config-of-another-type",
                                 "config-method-disagrees"])
def test_entry_refuses_a_contradictory_configuration(bad):
    """``config=`` and bare keywords together, a configuration of another
    method's type, and a configuration whose method disagrees with the
    ``method`` argument must each raise ``TypeError``. Catches a
    configuration whose options are dropped or silently mixed."""
    from n2v.probabilistic import ConformalReachConfig
    lo, hi = flat_box(3)
    options = {
        "config-and-keyword": {"config": ReachConfig(method="approx"),
                               "lp_solver": "default"},
        "config-of-another-type": {"config": ConformalReachConfig()},
        "config-method-disagrees": {"config": ReachConfig(method="exact")},
    }[bad]
    with pytest.raises(TypeError):
        reach_model(_small_relu_net(), flat_set("Star", lo, hi), "approx",
                    bad, **options)


@pytest.mark.parametrize("field,value", [("n_workers", 0),
                                         ("relax_factor", 1.5),
                                         ("relax_factor", -0.25),
                                         ("method", "bogus")])
def test_a_configuration_out_of_its_range_is_refused(field, value):
    """``ReachConfig`` checks its own fields: a worker count below one, a
    relaxation factor outside [0, 1] and a method that is neither exact nor
    approx are unreadable input and raise ValueError when the configuration
    is built, before any reach. Catches a check that is dropped or that lets
    a boundary value through (the factor's ends 0 and 1 are legal)."""
    fields = {field: value}
    if field == "relax_factor":
        fields["method"] = "approx"     # exact warns that it ignores it
    with pytest.raises(ValueError):
        ReachConfig(**fields)
    for legal in ({"relax_factor": 0.0, "method": "approx"},
                  {"relax_factor": 1.0, "method": "approx"},
                  {"n_workers": 1}):
        ReachConfig(**legal)


@pytest.mark.parametrize("method", ["approx", "exact"])
def test_entry_with_a_configuration_object_holds_every_output(method):
    """The documented ``config=ReachConfig(...)`` form, unpacked into the
    walk's keywords: the result must hold every replayed output (and, for
    the exact method, be the exact image). Catches a configuration whose
    unpacking hands the walk a wrong option."""
    model = _small_relu_net()
    lo, hi = flat_box(3, seed=10)
    what = f"config=ReachConfig(method={method!r})"
    out = reach_model(model, flat_set("Star", lo, hi), method, what,
                      config=ReachConfig(method=method))
    W1, b1 = _linear_of(model[0])
    W2, b2 = _linear_of(model[2])
    pieces = rational.affine(rational.relu(rational.affine(
        rational.box(lo.tolist(), hi.tolist()), W1, b1)), W2, b2)
    if method == "exact":
        _assert_exact(out, pieces, what)
    else:
        computed = [rational.star(s) for s in out]
        assert not rational.escapes(pieces, *computed), what
    assert_covered(out, replay_torch(model, samples(lo, hi, n=32), (1, 3)),
                   what)


# --------------------------------------------------------------------------- #
# 12. More of the walk: other set kinds with the pre-pass, an input Star given #
#     by rows only, the shape-aware flat BatchNorm, a join into a MaxPool,     #
#     the graph output as a tuple                                              #
# --------------------------------------------------------------------------- #
_PREPASS_HEXOCT = known_bug(
    16, "registry REACH-7: a reach with precompute_bounds on a Hexatope or "
        "an Octatope runs the pre-pass before any check of the set type, and "
        "the pre-pass raises TypeError ('Unsupported input_set type') from "
        "_extract_bounds (ibp) or _convert_to_zono (zono) where a refusal of "
        "the option is due", raises=TypeError)


# One marked cell per failing function of REACH-7: the ibp pre-pass fails in
# _extract_bounds and the zono pre-pass in _convert_to_zono, the same way for
# a Hexatope and an Octatope.
@pytest.mark.parametrize("precompute,kind", [
    pytest.param("ibp", "Zono", id="ibp-Zono"),
    pytest.param("zono", "Zono", id="zono-Zono"),
    pytest.param("ibp", "Box", id="ibp-Box"),
    pytest.param("zono", "Box", id="zono-Box"),
    pytest.param("ibp", "Hexatope", id="ibp-Hexatope", marks=_PREPASS_HEXOCT),
    pytest.param("zono", "Octatope", id="zono-Octatope",
                 marks=_PREPASS_HEXOCT)])
def test_reused_activation_with_the_pre_pass_on_other_flat_kinds(precompute,
                                                                  kind):
    """The reused-ReLU model of section 2 on a Zono, Box, Hexatope or
    Octatope with the bound pre-pass. A Zono and a Box must answer with a
    set that holds every replayed output; their ReLU rules take no
    precomputed bounds, so these cells check that the pre-pass runs on them
    and the reach stays sound, not the lookup of bounds by node name (the
    Star cells of section 2 do that). A Hexatope and an Octatope, which the
    pre-pass does not take, must be refused for the option (ValueError, as
    DESIGN section 7 names it, or NotImplementedError, the refusal of an
    unsupported combination; the owner settles which), never answered with
    bounds of another set and never with an accidental TypeError (REACH-7,
    marked; the ibp and the zono pre-pass fail in two different functions).
    Catches the option dropped before the pre-pass for these kinds (the
    reach then answers)."""
    model = _TwoCalls("nn.ReLU")
    lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    what = f"reused ReLU, precompute_bounds={precompute!r}, {kind}"
    if kind in ("Hexatope", "Octatope"):
        with pytest.raises((ValueError, NotImplementedError)):
            reach_model(model, flat_set(kind, lo, hi), "approx", what,
                        precompute_bounds=precompute)
        return
    out = reach_model(model, split_cover(kind, lo, hi), "approx", what,
                      precompute_bounds=precompute)
    assert_covered(out, replay_torch(model, samples(lo, hi), (1, 2)), what)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("precompute", ["ibp", "zono"])
def test_image_model_with_the_pre_pass_holds_every_output(tmp_path,
                                                          precompute, kind):
    """Conv 1x1 -> Relu -> Conv 1x1 -> Relu -> Reshape -> Gemm on a (2, 3, 5)
    image with the bound pre-pass: the two ReLU nodes have different input
    ranges and the same size, so bounds handed to the wrong node, or laid
    out in another order than the image set's, make the result miss
    outputs where they overlap the star's own enclosure (the crafted input
    of ``test_image_pre_pass_bounds_are_laid_out_like_the_image_set`` makes
    sure they do). The ImageZono ReLU rule takes no precomputed bounds, so
    the ImageZono cells check only that the pre-pass runs and the reach
    stays sound."""
    rng = np.random.default_rng(93)
    nodes = [helper.make_node("Conv", ["X", "K1", "k1"], ["A"]),
             helper.make_node("Relu", ["A"], ["R"]),
             helper.make_node("Conv", ["R", "K2", "k2"], ["B"]),
             helper.make_node("Relu", ["B"], ["Q"]),
             helper.make_node("Reshape", ["Q", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W", "b"], ["Y"], transB=1)]
    inits = {"K1": _dyadic(rng, 2, 2, 1, 1), "k1": _dyadic(rng, 2),
             "K2": _dyadic(rng, 2, 2, 1, 1), "k2": _dyadic(rng, 2) - 0.5,
             "F": np.array([1, -1], dtype=np.int64),
             "W": _dyadic(rng, 3, 30), "b": _dyadic(rng, 3)}
    path = save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, 3], inits=inits)
    lo, hi = pixel_box(2, 3, 5, seed=17)
    what = f"Conv/Relu/Conv/Relu with precompute_bounds={precompute!r}, {kind}"
    out = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                      what, precompute_bounds=precompute)
    assert_covered(out, replay_onnx(path, samples(lo, hi), (1, 2, 3, 5)),
                   what)


class _TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1, self.f2 = nn.Linear(2, 3), nn.Linear(3, 2)

    def forward(self, x):
        return self.f2(torch.relu(self.f1(x)))


@pytest.mark.parametrize("box", ["infinite", "absent"])
@pytest.mark.parametrize("method", ["approx", "exact"])
def test_input_star_given_by_rows_only_holds_every_output(box, method):
    """An input Star given only by its rows (the triangle a >= 0, a1 + a2 <=
    1, a binding row that is not a box row), with a predicate box of -inf
    and +inf or none at all: the entry point must make the box finite (or
    work from the rows) and the result must hold every replayed output, and
    under the exact method be the exact image. Catches an input conversion
    that is skipped or that keeps the box of another set."""
    model = _dyadic_module(_TwoLayer(), seed=94)
    V = np.array([[0.25, 1.0, 0.0], [-0.5, 0.0, 1.0]])
    C = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
    d = np.array([[0.0], [0.0], [1.0]])
    if box == "infinite":
        star = Star(V, C, d, np.full((2, 1), -np.inf), np.full((2, 1), np.inf))
    else:
        star = Star(V, C, d)
    what = f"{method} on a Star with rows only, predicate box {box}"
    out = reach_model(model, star, method, what)
    W1, b1 = _linear_of(model.f1)
    W2, b2 = _linear_of(model.f2)
    pieces = rational.polytope(C.tolist(), d.ravel().tolist())
    pieces = rational.affine(pieces, V[:, 1:].tolist(), V[:, 0].tolist())
    pieces = rational.affine(rational.relu(rational.affine(pieces, W1, b1)),
                             W2, b2)
    if method == "exact":
        _assert_exact(out, pieces, what)
    else:
        computed = [rational.star(s) for s in out]
        assert not rational.escapes(pieces, *computed), what
    grid = [np.array([a, b]) for a in np.arange(9) / 8 for b in np.arange(9) / 8
            if a + b <= 1]
    assert_covered(out, replay_torch(model, [V[:, 0] + V[:, 1:] @ t
                                             for t in grid], (1, 2)), what)


@pytest.mark.parametrize("kind,form", [("Star", "box"), ("Star", "row"),
                                       ("Zono", "box"), ("Box", "several"),
                                       ("Hexatope", "box")])
def test_batchnorm_over_the_channels_of_a_flat_tensor(tmp_path, kind, form):
    """ONNX BatchNormalization of a (1, 3, 4) tensor carried as a flat set
    (``input_shape`` (3, 4)), then Reshape and Gemm: the per-channel scale
    and shift broadcast over the last axis. The normalisation divides by
    exactly 1/2. Catches a BatchNorm on a flat set sent to the rule that
    reads the set as one value per channel, and channels broadcast along
    the wrong axis."""
    rng = np.random.default_rng(95)
    nodes = [helper.make_node("BatchNormalization",
                              ["X", "g", "be", "mu", "var"], ["B"],
                              epsilon=2.0 ** -10),
             helper.make_node("Reshape", ["B", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W", "b"], ["Y"], transB=1)]
    inits = {"g": np.array([1.0, -0.5, 0.75], dtype=np.float32),
             "be": np.array([0.25, 0.0, -0.5], dtype=np.float32),
             "mu": np.array([0.5, -0.25, 0.0], dtype=np.float32),
             "var": np.full(3, 0.25 - 2.0 ** -10, dtype=np.float32),
             "F": np.array([1, -1], dtype=np.int64),
             "W": _dyadic(rng, 3, 12), "b": _dyadic(rng, 3)}
    path = save_onnx(tmp_path, nodes, [1, 3, 4], [1, 3], inits=inits)
    lo, hi = flat_box(12, seed=12)
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"BatchNormalization over channels of (3, 4) on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what,
                      input_shape=(3, 4))
    assert_covered(out, replay_onnx(path, points, (1, 3, 4)), what)


@pytest.mark.parametrize("kind,form", [("ImageStar", "box"),
                                       ("ImageStar", "row"),
                                       ("ImageZono", "box")])
def test_join_of_two_image_branches_into_a_max_pool(tmp_path, kind, form):
    """Conv 1x1 -> (Relu, LeakyRelu) -> Sub -> MaxPool 2x2 with padding ->
    Reshape -> Gemm on a (2, 3, 5) image: the joined set, whose new
    predicate variables carry a tag, is read by a MaxPool, which uses the
    set's cached bounds. Catches a tagged set that keeps the bounds of
    another set and a join handed the wrong branch."""
    rng = np.random.default_rng(96)
    nodes = [helper.make_node("Conv", ["X", "K", "kb"], ["A"]),
             helper.make_node("Relu", ["A"], ["R"]),
             helper.make_node("LeakyRelu", ["A"], ["L"], alpha=0.25),
             helper.make_node("Sub", ["R", "L"], ["J"]),
             helper.make_node("MaxPool", ["J"], ["P"], kernel_shape=[2, 2],
                              strides=[2, 2], pads=[1, 1, 1, 1]),
             helper.make_node("Reshape", ["P", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W", "b"], ["Y"], transB=1)]
    inits = {"K": _dyadic(rng, 2, 2, 1, 1), "kb": _dyadic(rng, 2),
             "F": np.array([1, -1], dtype=np.int64),
             "W": _dyadic(rng, 3, 2 * 2 * 3), "b": _dyadic(rng, 3)}
    path = save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, 3], inits=inits)
    lo, hi = pixel_box(2, 3, 5, seed=18)
    sets, points = _image_inputs(kind, form, lo, hi)
    what = f"Conv/(Relu, LeakyRelu)/Sub/MaxPool/Gemm on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, (1, 2, 3, 5)), what)


class _Outputs(nn.Module):
    def __init__(self, how):
        super().__init__()
        self.f1, self.f2 = nn.Linear(4, 4), nn.Linear(4, 3)
        self.how = how

    def forward(self, x):
        h = self.f1(x)
        y = self.f2(F.relu(h))
        if self.how == "one-element tuple":
            return (y,)
        return y, h


def test_a_one_element_tuple_output_is_the_element():
    """``return (y,)``: the graph output names one node; the result must hold
    every replayed output of ``y``."""
    model = _dyadic_module(_Outputs("one-element tuple"), seed=97)
    lo, hi = flat_box(4, seed=13)
    out = reach_model(model, flat_set("Star", lo, hi), "approx", "(y,)")
    assert_covered(out, replay_torch(model, samples(lo, hi), (1, 4)), "(y,)")


def test_two_graph_outputs_are_refused():
    """``return y, h``: one list of sets cannot stand for two outputs, so the
    call must be refused with NotImplementedError, never answered with one
    of them."""
    model = _dyadic_module(_Outputs("two outputs"), seed=97)
    lo, hi = flat_box(4, seed=13)
    with pytest.raises(NotImplementedError):
        reach_model(model, flat_set("Star", lo, hi), "approx", "(y, h)")


def test_unknown_keyword_of_a_probabilistic_method_is_refused():
    """``method='conformal'`` builds its configuration from the keywords; an
    unknown keyword must raise ``TypeError`` before any sampling."""
    lo, hi = flat_box(3)
    with pytest.raises(TypeError):
        reach_model(_small_relu_net(), Box(lo, hi), "conformal",
                    "conformal with an unknown keyword", not_an_option=1)


_IMAGE_INF_BOX = known_bug(
    16, "registry REACH-5: the model entry point makes the predicate box of "
        "a Star finite but not that of an ImageStar, so an ImageStar whose "
        "box is -inf and +inf reaches the approx ReLU with an infinite box "
        "and the result holds NaN in its rows, which is not a set (under "
        "the suite's warnings policy numpy's invalid-value RuntimeWarning "
        "of that computation is raised first)", raises=RuntimeWarning)


@pytest.mark.parametrize("box,method", [
    ("absent", "approx"),
    pytest.param("infinite", "approx", marks=_IMAGE_INF_BOX)])
def test_input_image_star_given_by_rows_only_holds_every_output(tmp_path, box,
                                                                method):
    """Conv 1x1 -> Relu -> Reshape -> Gemm on a (2, 3, 5) ImageStar whose
    box rows are constraint rows, with a predicate box of -inf and +inf or
    none at all: every returned set must be a set (finite rows) and hold
    every onnxruntime output. The ImageStar without a box is the passing
    neighbour with the same model and check (the exact method passes too,
    but its bound LPs over 30 variables without a box take 20 seconds)."""
    rng = np.random.default_rng(98)
    nodes = [helper.make_node("Conv", ["X", "K", "kb"], ["A"]),
             helper.make_node("Relu", ["A"], ["R"]),
             helper.make_node("Reshape", ["R", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W", "b"], ["Y"], transB=1)]
    inits = {"K": _dyadic(rng, 2, 2, 1, 1), "kb": _dyadic(rng, 2),
             "F": np.array([1, -1], dtype=np.int64),
             "W": _dyadic(rng, 3, 30), "b": _dyadic(rng, 3)}
    path = save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, 3], inits=inits)
    lo, hi = pixel_box(2, 3, 5, seed=19)
    s = image_set("ImageStar", lo, hi)
    k = int(s.nVar)
    bounds = ((np.full((k, 1), -np.inf), np.full((k, 1), np.inf))
              if box == "infinite" else (None, None))
    image = ImageStar(s.V, np.vstack([np.eye(k), -np.eye(k)]),
                      np.ones((2 * k, 1)), *bounds, height=3, width=5,
                      num_channels=2)
    what = f"{method} on an ImageStar with rows only, predicate box {box}"
    out = reach_model(load_onnx(path), image, method, what)
    for j, o in enumerate(out):
        finite = all(np.all(np.isfinite(np.asarray(a, dtype=np.float64)))
                     for a in (o.V, o.C, o.d))
        assert finite, f"{what}: returned set {j} has NaN or inf in its rows"
    assert_covered(out, replay_onnx(path, samples(lo, hi), (1, 2, 3, 5)),
                   what)


# --------------------------------------------------------------------------- #
# 13. Row order and layout, checked on the whole tensor                        #
# --------------------------------------------------------------------------- #
# A Gemm to a few outputs after a step that moves entries projects a wrong
# placement away: an image Concat that reverses its operands left the
# width-swap Split cell of section 6 green, because the image of the swapped
# box under the 3-output Gemm holds the true outputs too. The cells here
# return the tensor the step produces, so that an entry put in the wrong
# place misses its own interval (every pixel has its own interval).
def _image_star_from(V_chw, k, lb=None, ub=None):
    """An ImageStar of a (C, H, W) tensor from its basis in the tensor's own
    order (``V_chw`` of shape (C, H, W, 1 + k)), stored (H, W, C), with no
    constraint row and the predicate box [-1, 1]^k."""
    V = np.transpose(V_chw, (1, 2, 0, 3)).copy()
    h, w, c = V.shape[:3]
    return ImageStar(V, np.zeros((0, k)), np.zeros((0, 1)),
                     -np.ones((k, 1)) if lb is None else lb,
                     np.ones((k, 1)) if ub is None else ub,
                     height=h, width=w, num_channels=c)


_GATHERS = {
    # name: (axis, indices, output shape)
    "height-2-0": (2, [2, 0], (1, 2, 2, 5)),
    "width-negative-4-1-1": (-1, [4, 1, 1], (1, 2, 3, 3)),
}


@pytest.mark.parametrize("case,kind,form", (
    [("height-2-0", kind, form) for kind, form in (
        ("Star", "row"), ("Zono", "box"), ("Box", "several"),
        ("ImageStar", "box"), ("ImageStar", "row"), ("ImageZono", "box"))]
    + [("width-negative-4-1-1", kind, form) for kind, form in (
        ("Star", "row"), ("ImageZono", "box"))]))
def test_gather_with_unsorted_indices_holds_every_output(tmp_path, case, kind,
                                                         form):
    """ONNX Gather with constant indices that are not ascending (rows [2, 0]
    of the height; columns [4, 1, 1] of the width by a negative axis, one
    taken twice) of a (2, 3, 5) tensor, the gathered tensor returned whole.
    Flat sets and the ImageZono get ``input_shape``; the ImageStar is given
    without it, so the tensor shapes come from its own height, width and
    channels. Catches row numbers sorted or made unique (the right rows in
    the wrong order), an image set flattened in its stored (height, width,
    channel) order before the channel-first row numbers are applied, a
    negative axis read as its absolute value, and an ImageStar whose
    placeholder for the shape pass swaps height and width (every recorded
    shape then has the two spatial axes exchanged)."""
    axis, indices, out_shape = _GATHERS[case]
    nodes = [helper.make_node("Gather", ["X", "I"], ["Y"], axis=axis)]
    path = save_onnx(tmp_path, nodes, (1,) + _TENSOR, list(out_shape),
                     inits={"I": np.array(indices, dtype=np.int64)})
    lo, hi = pixel_box(*_TENSOR, seed=21)
    sets, points = _reduce_inputs(kind, form, lo, hi)
    options = {} if kind == "ImageStar" else {"input_shape": _TENSOR}
    what = f"Gather {case} on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what, **options)
    assert_covered(out, replay_onnx(path, points, (1,) + _TENSOR), what,
                   out_chw=out_shape[1:])


@pytest.mark.parametrize("kind,form", [("Star", "row"), ("Zono", "box"),
                                       ("Box", "several")])
def test_reduce_over_the_first_of_two_equal_axes_with_input_shape(
        tmp_path, kind, form):
    """ReduceSum over axis 1 of a (1, 3, 3) tensor carried as a flat set with
    ``input_shape`` (3, 3), the (1, 3) result returned whole, bounds wider
    than 2 and beyond plus and minus 1. The model also runs on the (3, 3)
    tensor without the batch axis, where axis 1 is the other axis of the
    same length. Catches a shape pass that tries the caller's shape without
    the batch axis first: the recorded shapes then lack the batch axis, and
    the sum runs over the columns instead of the rows."""
    nodes = [helper.make_node("ReduceSum", ["X", "A"], ["Y"], keepdims=0)]
    path = save_onnx(tmp_path, nodes, [1, 3, 3], [1, 3],
                     inits={"A": np.array([1], dtype=np.int64)})
    lo, hi = pixel_box(1, 3, 3, seed=22)
    lo, hi = 4 * lo.reshape(3, 3), 4 * hi.reshape(3, 3)
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"ReduceSum axis 1 of (1, 3, 3) on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what,
                      input_shape=(3, 3))
    assert_covered(out, replay_onnx(path, points, (1, 3, 3)), what)


class _ViewConv(nn.Module):
    """x (1, 30) -> view (1, 2, 3, 5) -> Conv2d(2, 2, (1, 2))."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, (1, 2))

    def forward(self, x):
        return self.conv(x.view(1, 2, 3, 5))


@pytest.mark.parametrize("route", ["onnx-reshape", "torch-view"])
@pytest.mark.parametrize("kind,form", [("Star", "row"), ("Zono", "box")])
def test_flat_set_reshaped_into_a_convolution_keeps_the_channel_order(
        tmp_path, route, kind, form):
    """A flat set of 30 entries reshaped to a (2, 3, 5) image (an ONNX Reshape,
    or a torch ``view``) in front of a convolution that mixes the two
    channels, the (2, 3, 4) result returned whole. The flat rows are in
    channel-first order and the image set stores height, width, channel.
    Catches the rows put into the image without the channel-first to
    channel-last permutation (on a Star or a Zono), and a ``view`` that is
    not turned into an image set for its convolution."""
    lo, hi = pixel_box(2, 3, 5, seed=23)
    lo, hi = lo.reshape(-1), hi.reshape(-1)
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"{route} to (2, 3, 5) then Conv on a {form} {kind}"
    if route == "onnx-reshape":
        rng = np.random.default_rng(24)
        nodes = [helper.make_node("Reshape", ["X", "S"], ["R"]),
                 helper.make_node("Conv", ["R", "K", "kb"], ["Y"])]
        path = save_onnx(tmp_path, nodes, [1, 30], [1, 2, 3, 4],
                         inits={"S": np.array([1, 2, 3, 5], dtype=np.int64),
                                "K": _dyadic(rng, 2, 2, 1, 2),
                                "kb": _dyadic(rng, 2)})
        out = reach_model(load_onnx(path), sets, "approx", what)
        outputs = replay_onnx(path, points, (1, 30))
    else:
        model = _dyadic_module(_ViewConv(), seed=24)
        out = reach_model(model, sets, "approx", what)
        outputs = replay_torch(model, points, (1, 30))
    assert_covered(out, outputs, what, out_chw=(2, 3, 4))


class _ViewSoftmax(nn.Module):
    """x (1, 30) -> view (1, 2, 3, 5) -> Softmax over the last axis ->
    flatten."""

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(x.view(1, 2, 3, 5)).flatten(1)


@pytest.mark.parametrize("consumer,kind,form", (
    [(c, kind, form) for c in ("MatMul", "Softmax")
     for kind, form in (("Star", "row"), ("Zono", "box"), ("Box", "several"))]
    + [("torch-Softmax", "Star", "row")]))
def test_reshape_to_three_axes_without_a_spatial_consumer_keeps_the_set_flat(
        tmp_path, consumer, kind, form):
    """A flat set reshaped to (1, 2, 3, 5), a tensor that no convolution or
    pooling layer reads (the head split of a transformer), then a MatMul by a
    (5, 4) constant over the last axis or a Softmax over the last axis (ONNX,
    float64, so the replay is exact up to float64; or a torch ``view`` and
    ``nn.Softmax``), the result returned whole. The set must stay flat in
    the tensor's own order: read as a (C, H, W) image, the MatMul refuses
    and the Softmax normalises over every entry. Catches a reshape rule that
    is not told to keep the set flat (an ONNX Reshape or a torch view)."""
    lo, hi = pixel_box(2, 3, 5, seed=25)
    lo, hi = lo.reshape(-1), hi.reshape(-1)
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"Reshape (2, 3, 5) then {consumer} on a {form} {kind}"
    if consumer == "torch-Softmax":
        model = _ViewSoftmax().double().eval()
        out = reach_model(model, sets, "approx", what)
        assert_covered(out, replay_torch(model, points, (1, 30)), what)
        return
    rng = np.random.default_rng(26)
    inits = {"S": np.array([1, 2, 3, 5], dtype=np.int64)}
    if consumer == "MatMul":
        inits["W"] = _dyadic(rng, 5, 4).astype(np.float64)
        last = helper.make_node("MatMul", ["R", "W"], ["Y"])
        out_shape = [1, 2, 3, 4]
    else:
        last = helper.make_node("Softmax", ["R"], ["Y"], axis=-1)
        out_shape = [1, 2, 3, 5]
    nodes = [helper.make_node("Reshape", ["X", "S"], ["R"]), last]
    path = save_onnx(tmp_path, nodes, [1, 30], out_shape, inits=inits,
                     in_type="DOUBLE", out_type="DOUBLE",
                     name=f"flat-head-{consumer}")
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, (1, 30)), what)


def _image_and_flat_branch_model(tmp_path, join, rng):
    """X (1, 2, 3, 5) -> A = Conv 1x1 (an image set); B = Reshape(Gemm(
    Flatten(X)), (1, 2, 3, 5)) (a flat set: no convolution reads it);
    Y = join of A and B, returned whole."""
    first, second = ("B", "A") if join.endswith("flat-first") else ("A", "B")
    op = join.split("-")[0]
    nodes = [helper.make_node("Conv", ["X", "K", "kb"], ["A"]),
             helper.make_node("Flatten", ["X"], ["F"], axis=1),
             helper.make_node("Gemm", ["F", "W", "b"], ["G"], transB=1),
             helper.make_node("Reshape", ["G", "S"], ["B"]),
             helper.make_node(op, [first, second], ["Y"])]
    inits = {"K": _dyadic(rng, 2, 2, 1, 1), "kb": _dyadic(rng, 2),
             "W": _dyadic(rng, 30, 30) / 4, "b": _dyadic(rng, 30),
             "S": np.array([1, 2, 3, 5], dtype=np.int64)}
    return save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, 2, 3, 5],
                     inits=inits, name=f"image-flat-{join}")


@pytest.mark.parametrize("join", ["Add", "Sub-flat-first", "Mul"])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_join_of_an_image_branch_and_a_flat_branch_of_one_tensor(
        tmp_path, join, kind):
    """The same (2, 3, 5) tensor held as an image set on one branch (a 1x1
    Conv) and as a flat set on the other (Flatten, Gemm and a Reshape back to
    the image shape that no convolution reads), joined by Add, Sub (the flat
    operand first) or Mul, the joined tensor returned whole. The join must
    bring the image operand to the flat operand's channel-first order.
    Catches an image operand flattened in its stored (height, width,
    channel) order (``to_star`` or ``to_zono`` in place of the channel-first
    flattening), for the sum and for the product."""
    path = _image_and_flat_branch_model(tmp_path, join,
                                        np.random.default_rng(27))
    lo, hi = pixel_box(2, 3, 5, seed=27)
    what = f"Conv and Flatten/Gemm/Reshape joined by {join} on a {kind}"
    out = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                      what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), (1, 2, 3, 5)),
                   what, out_chw=(2, 3, 5))


@pytest.mark.parametrize("axis", [1, 3])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_concat_of_two_image_branches_keeps_the_operand_order(tmp_path, axis,
                                                              kind):
    """Conv 1x1 of a (2, 3, 5) image and Relu of the same image concatenated
    along the channels or the width, the (4, 3, 5) or (2, 3, 10) result
    returned whole. Catches operands concatenated in reverse order and an
    ONNX axis mapped to the wrong stored axis."""
    rng = np.random.default_rng(28)
    nodes = [helper.make_node("Conv", ["X", "K", "kb"], ["A"]),
             helper.make_node("Relu", ["X"], ["R"]),
             helper.make_node("Concat", ["A", "R"], ["Y"], axis=axis)]
    out_chw = (4, 3, 5) if axis == 1 else (2, 3, 10)
    path = save_onnx(tmp_path, nodes, [1, 2, 3, 5], (1,) + out_chw,
                     inits={"K": _dyadic(rng, 2, 2, 1, 1),
                            "kb": _dyadic(rng, 2)})
    lo, hi = pixel_box(2, 3, 5, seed=28)
    what = f"Concat(Conv, Relu) along axis {axis} on a {kind}"
    out = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                      what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), (1, 2, 3, 5)),
                   what, out_chw=out_chw)


@pytest.mark.parametrize("op", ["Add", "Mul", "Sub-constant-first"])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_image_set_with_a_constant_of_the_whole_tensor(tmp_path, op, kind):
    """An image set and a constant of the full (1, 2, 3, 5) shape: Add, Mul,
    or Sub with the constant first, the result returned whole. The constant
    is in the tensor's channel-first order, the set stores height, width,
    channel; the constant has both signs and magnitudes above and below one.
    Catches the constant read in the stored order without the permutation
    (the translation of an ImageStar, the scale of an ImageZono)."""
    rng = np.random.default_rng(29)
    k = (rng.integers(-12, 13, size=(1, 2, 3, 5)) / 4.0).astype(np.float32)
    inputs = ["K", "X"] if op.endswith("first") else ["X", "K"]
    nodes = [helper.make_node(op.split("-")[0], inputs, ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, 2, 3, 5],
                     inits={"K": k}, name=f"const-{op}")
    lo, hi = pixel_box(2, 3, 5, seed=29)
    what = f"{op} with a (1, 2, 3, 5) constant on a {kind}"
    out = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                      what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), (1, 2, 3, 5)),
                   what, out_chw=(2, 3, 5))


@pytest.mark.parametrize("kind,form", [("Star", "row"), ("Zono", "box"),
                                       ("Box", "several")])
def test_matmul_with_the_constant_on_the_left_of_a_matrix(tmp_path, kind,
                                                          form):
    """ONNX MatMul W (2, 3) times X (1, 3, 4), the constant on the left and
    four columns, on a flat set with ``input_shape`` (3, 4), the (1, 2, 4)
    result returned whole. On the flat set the map is kron(W, I_4); the
    other order kron(I_4, W) has the same size. Catches the Kronecker
    product taken in the wrong order."""
    rng = np.random.default_rng(30)
    nodes = [helper.make_node("MatMul", ["W", "X"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 3, 4], [1, 2, 4],
                     inits={"W": _dyadic(rng, 2, 3)})
    lo, hi = pixel_box(1, 3, 4, seed=30)
    lo, hi = lo.reshape(3, 4), hi.reshape(3, 4)
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"MatMul(W, X) with X (1, 3, 4) on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what,
                      input_shape=(3, 4))
    assert_covered(out, replay_onnx(path, points, (1, 3, 4)), what)


_LP_UNDERSHOOT = 5e-10      # HiGHS's largest measured undershoot


def _undershooting_lp(monkeypatch):
    """Make every LP a Star solves stop ``_LP_UNDERSHOOT`` inside its true
    optimum, as a solver within its feasibility tolerance may, at the two
    entry points every range query goes through. Returns the list of calls,
    so that a cell can check that the stub was reached."""
    calls = []
    single, batch = Star._solve_lp, Star._solve_lp_batch

    def inward(v, f, minimize):
        if v is None or not np.any(np.asarray(f) != 0):
            return v
        return v + _LP_UNDERSHOOT if minimize else v - _LP_UNDERSHOOT

    def one(self, f, minimize=True, lp_solver='default'):
        calls.append(1)
        return inward(single(self, f, minimize, lp_solver), f, minimize)

    def many(self, objectives, minimize_flags, lp_solver='default'):
        calls.append(len(objectives))
        values = batch(self, objectives, minimize_flags, lp_solver)
        return [inward(v, f, m)
                for v, f, m in zip(values, objectives, minimize_flags)]

    monkeypatch.setattr(Star, "_solve_lp", one)
    monkeypatch.setattr(Star, "_solve_lp_batch", many)
    return calls


@pytest.mark.parametrize("op", ["Mul", "Div"])
def test_product_and_quotient_of_two_branches_survive_an_lp_that_stops_inside(
        tmp_path, monkeypatch, op):
    """Two Gemm branches of one input multiplied or divided (float64 graph),
    on a Star, with every LP stopping 5e-10 inside its optimum. The McCormick
    rule takes its operand ranges from LPs and must pad them outward: Mul of
    a in [-2 + 2^-12, 2^-12] and b in [4095, 4097] reaches 4097/4096 at a
    corner, and an unpadded upper end of a misses it by about 2e-6; Div of b
    by e in [2^-13, 2^-12] reaches 4097 * 2^13, and an unpadded lower end of
    e misses it by about 137. Catches the outward pad of the operand ranges
    dropped (for the product and for the quotient)."""
    if op == "Mul":
        w = {"Wa": [[1.0, 0.0]], "ba": [2.0 ** -12 - 1.0],
             "Wb": [[0.0, 1.0]], "bb": [4096.0]}
    else:
        w = {"Wa": [[1.0, 0.0]], "ba": [4096.0],
             "Wb": [[0.0, 2.0 ** -14]], "bb": [3 * 2.0 ** -14]}
    nodes = [helper.make_node("Gemm", ["X", "Wa", "ba"], ["A"], transB=1),
             helper.make_node("Gemm", ["X", "Wb", "bb"], ["B"], transB=1),
             helper.make_node(op, ["A", "B"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 1],
                     inits={k: np.array(v) for k, v in w.items()},
                     in_type="DOUBLE", out_type="DOUBLE", name=f"mc-{op}")
    lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    model = load_onnx(path)
    calls = _undershooting_lp(monkeypatch)
    what = f"{op} of two Gemm branches, every LP 5e-10 inside"
    out = reach_model(model, flat_set("Star", lo, hi), "approx", what)
    monkeypatch.undo()
    if not calls:
        pytest.fail(f"{what}: the operand ranges did not go through the LP")
    assert_covered(out, replay_onnx(path, samples(lo, hi), (1, 2)), what)


# --------------------------------------------------------------------------- #
# 14. Handlers that take the pieces of an exact split, and exact image graphs  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", ["ReduceSum", "ReduceMean",
                                "BatchNormalization"])
def test_flat_reduce_and_batchnorm_take_every_piece_of_an_exact_split(
        tmp_path, op):
    """Exact method: a Star of 6 entries with three predicate variables ->
    Gemm -> Relu (an exact split into several pieces) -> Reshape (1, 2, 3)
    -> ReduceSum over the last axis, ReduceMean over the two channels, or a
    BatchNormalization over the channels -> Reshape -> Gemm. The shape-aware
    flat handler receives the list of pieces at once. Both directions
    against the exact image in rationals. Catches a handler that maps only
    the first piece, or reuses one piece's map for the others."""
    rng = np.random.default_rng(31)
    w = {"W1": _dyadic(rng, 6, 6), "b1": _dyadic(rng, 6)}
    inits = dict(w, S=np.array([1, 2, 3], dtype=np.int64),
                 F=np.array([1, -1], dtype=np.int64))
    if op == "ReduceSum":
        inits["A"] = np.array([2], dtype=np.int64)
        middle = helper.make_node("ReduceSum", ["T", "A"], ["M"], keepdims=0)
        M = [[Fraction(int(j // 3 == i)) for j in range(6)] for i in range(2)]
        c = [Fraction(0)] * 2
    elif op == "ReduceMean":
        middle = helper.make_node("ReduceMean", ["T"], ["M"], axes=[1],
                                  keepdims=0)
        M = [[Fraction(1, 2) * int(j % 3 == i) for j in range(6)]
             for i in range(3)]
        c = [Fraction(0)] * 3
    else:
        g, be = np.array([0.75, -1.5], np.float32), np.array([0.25, -0.5],
                                                               np.float32)
        mu = np.array([0.5, -0.25], np.float32)
        inits.update(g=g, be=be, mu=mu,
                     var=np.full(2, 0.25 - 2.0 ** -10, dtype=np.float32))
        middle = helper.make_node("BatchNormalization",
                                  ["T", "g", "be", "mu", "var"], ["M"],
                                  epsilon=2.0 ** -10)
        scale = [Fraction(float(v)) * 2 for v in g]       # sqrt(1/4) = 1/2
        M = [[scale[i // 3] * int(j == i) for j in range(6)]
             for i in range(6)]
        c = [Fraction(float(be[i // 3])) - scale[i // 3]
             * Fraction(float(mu[i // 3])) for i in range(6)]
    n_mid = len(M)
    inits.update(W2=_dyadic(rng, 2, n_mid), b2=_dyadic(rng, 2))
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["H"], transB=1),
             helper.make_node("Relu", ["H"], ["R"]),
             helper.make_node("Reshape", ["R", "S"], ["T"]), middle,
             helper.make_node("Reshape", ["M", "F"], ["G"]),
             helper.make_node("Gemm", ["G", "W2", "b2"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, [1, 6], [1, 2], inits=inits,
                     name=f"pieces-{op}")
    star, x0, E = _low_rank_star(rng, 6)
    what = f"exact Gemm/Relu/{op}/Gemm on the pieces of a split"
    out = reach_model(load_onnx(path), star, "exact", what)
    if len(out) < 2:
        pytest.fail(f"{what}: the ReLU did not split ({len(out)} piece)")
    pieces = rational.affine(rational.box([-1] * 3, [1] * 3), E.tolist(),
                             x0.tolist())
    pieces = rational.affine(pieces, w["W1"].astype(float).tolist(),
                             w["b1"].astype(float).tolist())
    pieces = rational.affine(rational.relu(pieces), M, c)
    pieces = rational.affine(pieces, inits["W2"].astype(float).tolist(),
                             inits["b2"].astype(float).tolist())
    _assert_exact(out, pieces, what)
    assert_covered(out, replay_onnx(path, _low_rank_points(x0, E, 14),
                                    (1, 6)), what)


def _image_graph_map(graph, rng):
    """``(nodes, inits, M, c)``: the affine part of a graph X (1, 2, 3, 5) ->
    <graph> -> Reshape (1, -1), and its map ``x -> M x + c`` on the flat
    channel-first tensor, from numpy index arrays and the constants (no
    n2v)."""
    index = np.arange(30).reshape(1, 2, 3, 5)
    eye = np.eye(30)
    if graph == "transpose":
        nodes = [helper.make_node("Transpose", ["X"], ["T"],
                                  perm=[0, 1, 3, 2])]
        order = np.transpose(index, (0, 1, 3, 2)).reshape(-1)
        return nodes, {}, eye[order], np.zeros(30)
    if graph == "split-swap":
        nodes = [helper.make_node("Split", ["X", "SP"], ["P0", "P1"], axis=2),
                 helper.make_node("Concat", ["P1", "P0"], ["T"], axis=2)]
        order = np.concatenate([index[:, :, 1:], index[:, :, :1]],
                               axis=2).reshape(-1)
        return nodes, {"SP": np.array([1, 2], dtype=np.int64)}, eye[order], \
            np.zeros(30)
    # join-before-relu: T = X - Conv1x1(X)
    K, kb = _dyadic(rng, 2, 2, 1, 1), _dyadic(rng, 2)
    nodes = [helper.make_node("Conv", ["X", "K", "kb"], ["A"]),
             helper.make_node("Sub", ["X", "A"], ["T"])]
    conv = np.kron(K[:, :, 0, 0].astype(np.float64), np.eye(15))
    return nodes, {"K": K, "kb": kb}, eye - conv, \
        -np.repeat(kb.astype(np.float64), 15)


@pytest.mark.parametrize("graph", ["transpose", "split-swap",
                                   "join-before-relu"])
def test_exact_image_star_graph_is_the_exact_image(tmp_path, graph):
    """Exact method on an ImageStar of a (2, 3, 5) tensor with three
    predicate variables (no ``input_shape``): a Transpose of height and width
    (the image path of the dispatcher), a Split of the height whose two
    outputs are concatenated in swapped order, or the image minus a 1x1
    Conv of itself (a join of one predicate system), then Reshape -> Gemm
    -> Relu -> Gemm. Both directions against the exact image in rationals,
    the affine part written from numpy index arrays and the constants.
    Catches entries moved to the wrong place or a getitem that hands a node
    the other output of the Split (on an image set, where the approx cells
    of sections 6 and 9 project such errors away), and a join that is not
    exact for one predicate system."""
    rng = np.random.default_rng(32)
    nodes, inits, M, c = _image_graph_map(graph, rng)
    inits.update(F=np.array([1, -1], dtype=np.int64),
                 W2=_dyadic(rng, 3, 30), b2=_dyadic(rng, 3),
                 W3=_dyadic(rng, 2, 3), b3=_dyadic(rng, 2))
    nodes = nodes + [
        helper.make_node("Reshape", ["T", "F"], ["G"]),
        helper.make_node("Gemm", ["G", "W2", "b2"], ["P"], transB=1),
        helper.make_node("Relu", ["P"], ["Q"]),
        helper.make_node("Gemm", ["Q", "W3", "b3"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, 2], inits=inits,
                     name=f"exact-image-{graph}")
    x0 = rng.integers(-8, 9, size=30) / 8.0
    E = rng.integers(-4, 5, size=(30, 3)) / 8.0
    V_chw = np.concatenate([x0[:, None], E], axis=1).reshape(2, 3, 5, 4)
    star = _image_star_from(V_chw, 3)
    what = f"exact {graph}/Gemm/Relu/Gemm on an ImageStar"
    out = reach_model(load_onnx(path), star, "exact", what)
    pieces = rational.affine(rational.box([-1] * 3, [1] * 3), E.tolist(),
                             x0.tolist())
    pieces = rational.affine(pieces, M.tolist(), c.tolist())
    pieces = rational.affine(pieces, inits["W2"].astype(float).tolist(),
                             inits["b2"].astype(float).tolist())
    pieces = rational.affine(rational.relu(pieces),
                             inits["W3"].astype(float).tolist(),
                             inits["b3"].astype(float).tolist())
    _assert_exact(out, pieces, what)
    points = [(x0 + E @ t).reshape(2, 3, 5)
              for t in np.vstack([np.eye(3), -np.eye(3),
                                  rng.integers(-4, 5, size=(10, 3)) / 4])]
    assert_covered(out, replay_onnx(path, points, (1, 2, 3, 5)), what)


# --------------------------------------------------------------------------- #
# 15. More of the walk: joins after an exact split, a value joined with        #
#     itself, calls that carry more than the set, state left by a reach, the   #
#     tensor shape each set type tracks, refusals                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("join", ["Sub", "Concat"])
def test_exact_join_after_a_relu_split_is_the_exact_image(tmp_path, join):
    """Exact method: X -> Gemm -> Relu = R (an exact split into several
    pieces); A = Gemm(R); J = R - A or Concat(R, A); Y = Gemm(J), on a Star
    with three predicate variables. Each piece of R must meet the piece of A
    that descends from it (the same predicate system), so the join stays
    exact. Both directions against the exact image in rationals. Catches a
    pairing of the two lists of pieces by position or by the first piece
    (a piece combined with another piece's image)."""
    rng = np.random.default_rng(33)
    w = {"W1": _dyadic(rng, 4, 4), "b1": _dyadic(rng, 4),
         "Wa": _dyadic(rng, 4, 4), "ba": _dyadic(rng, 4),
         "W2": _dyadic(rng, 3, 8 if join == "Concat" else 4),
         "b2": _dyadic(rng, 3)}
    joined = (helper.make_node("Concat", ["R", "A"], ["J"], axis=1)
              if join == "Concat" else helper.make_node("Sub", ["R", "A"],
                                                        ["J"]))
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["H"], transB=1),
             helper.make_node("Relu", ["H"], ["R"]),
             helper.make_node("Gemm", ["R", "Wa", "ba"], ["A"], transB=1),
             joined,
             helper.make_node("Gemm", ["J", "W2", "b2"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, [1, 4], [1, 3], inits=w,
                     name=f"split-join-{join}")
    star, x0, E = _low_rank_star(rng, 4)
    what = f"exact Gemm/Relu/(id, Gemm)/{join}/Gemm"
    out = reach_model(load_onnx(path), star, "exact", what)
    if len(out) < 2:
        pytest.fail(f"{what}: the ReLU did not split ({len(out)} piece)")
    f = {k: v.astype(np.float64) for k, v in w.items()}
    if join == "Sub":
        M, c = np.eye(4) - f["Wa"], -f["ba"]
    else:
        M = np.vstack([np.eye(4), f["Wa"]])
        c = np.concatenate([np.zeros(4), f["ba"]])
    pieces = rational.affine(rational.box([-1] * 3, [1] * 3), E.tolist(),
                             x0.tolist())
    pieces = rational.relu(rational.affine(pieces, f["W1"].tolist(),
                                           f["b1"].tolist()))
    pieces = rational.affine(rational.affine(pieces, M.tolist(), c.tolist()),
                             f["W2"].tolist(), f["b2"].tolist())
    _assert_exact(out, pieces, what)
    assert_covered(out, replay_onnx(path, _low_rank_points(x0, E, 14),
                                    (1, 4)), what)


@pytest.mark.parametrize("kind,form", [("Star", "row"), ("Zono", "box"),
                                       ("Box", "several")])
def test_a_value_multiplied_by_itself_holds_every_output(tmp_path, kind,
                                                         form):
    """X -> Gemm = H; Y = H * H returned whole: one node read twice by the
    same product, the square of a value that crosses zero (its true range
    starts at 0, below the square of either end of the value's range).
    Catches a product of a node with itself taken as the interval between
    the squares of its ends, a product that reads the second operand from
    another node, and a list of sets changed in place by the first
    reader."""
    rng = np.random.default_rng(34)
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["H"], transB=1),
             helper.make_node("Mul", ["H", "H"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 4], [1, 4],
                     inits={"W1": _dyadic(rng, 4, 4), "b1": _dyadic(rng, 4)})
    lo, hi = flat_box(4, seed=34)
    lo, hi = 4 * lo, 4 * hi
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"Gemm/H*H on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, (1, 4)), what)


class _Called(nn.Module):
    """fc1 -> <call> -> fc2, the middle call spelled with keywords or extra
    arguments."""

    def __init__(self, how):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(4, 4), nn.Linear(4, 3)
        self.mid, self.leaky = nn.Linear(4, 4), nn.LeakyReLU(0.25)
        self.how = how

    def forward(self, x):
        h = self.fc1(x)
        if self.how == "nn.Linear-by-keyword":
            h = self.mid(input=h)
        elif self.how == "nn.LeakyReLU-by-keyword":
            h = self.leaky(input=h)
        elif self.how == "F.relu-with-inplace":
            h = F.relu(h, inplace=False)
        elif self.how == "F.leaky_relu-all-positional":
            h = F.leaky_relu(h, 0.25, False)
        elif self.how == "torch.flatten-by-keyword":
            h = F.relu(torch.flatten(h, start_dim=1, end_dim=-1))
        else:                                       # flatten method, keyword
            h = F.relu(h.flatten(start_dim=1))
        return self.fc2(h)


_CALLED = ["nn.Linear-by-keyword", "nn.LeakyReLU-by-keyword",
           "F.relu-with-inplace", "F.leaky_relu-all-positional",
           "torch.flatten-by-keyword", "flatten-method-by-keyword"]


@pytest.mark.parametrize("how", _CALLED)
def test_a_call_that_carries_more_than_its_input_holds_every_output(how):
    """A module called with its input as a keyword, and function and method
    nodes whose further arguments are keywords or positions (an ``inplace``
    flag, a slope and a flag by position, flatten dims by keyword). The walk
    hands a module only its first positional argument (GRAPH-25, marked for
    ``nn.ConvTranspose2d`` with ``output_size`` in
    ``test_soundness_conv_transpose2d.py``), so a module called by keyword
    has no input it reads: the call must be refused with NotImplementedError
    or answered soundly, never answered for the sets of another node. Every
    other spelling must be answered with a set that holds every output.
    Catches a call that falls back to the last sets computed, and an
    argument read from the wrong place (an ``inplace`` flag taken for a
    slope)."""
    model = _dyadic_module(_Called(how), seed=35)
    lo, hi = flat_box(4, seed=35)
    what = f"{how} between two Linear layers"
    try:
        out = reach_model(model, flat_set("Star", lo, hi), "approx", what)
    except NotImplementedError:
        if how.endswith("-by-keyword") and how.startswith("nn."):
            return              # a clear refusal keeps the promise
        raise
    assert_covered(out, replay_torch(model, samples(lo, hi), (1, 4)), what)


def test_one_model_reached_again_with_another_input_and_pre_pass(tmp_path):
    """One loaded model reached three times in a row: a narrow Star with the
    ibp pre-pass, a wide Star with the ibp pre-pass, a wide Zono with the
    zono pre-pass. Each answer must hold every output of its own input, as a
    freshly loaded model's would. Catches state that a reach leaves on the
    model or the walk (bounds, shapes or sets of an earlier reach read by a
    later one)."""
    rng = np.random.default_rng(36)
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["H"], transB=1),
             helper.make_node("Relu", ["H"], ["R"]),
             helper.make_node("Gemm", ["R", "W2", "b2"], ["G"], transB=1),
             helper.make_node("Relu", ["G"], ["Q"]),
             helper.make_node("Gemm", ["Q", "W3", "b3"], ["Y"], transB=1)]
    path = save_onnx(tmp_path, nodes, [1, 3], [1, 2],
                     inits={"W1": _dyadic(rng, 4, 3), "b1": _dyadic(rng, 4),
                            "W2": _dyadic(rng, 4, 4), "b2": _dyadic(rng, 4),
                            "W3": _dyadic(rng, 2, 4), "b3": _dyadic(rng, 2)})
    model = load_onnx(path)
    lo, hi = flat_box(3, seed=36)
    narrow = (lo + hi) / 2, (lo + hi) / 2 + 0.125
    for kind, pre, (a, b) in (("Star", "ibp", narrow), ("Star", "ibp",
                                                        (4 * lo, 4 * hi)),
                              ("Zono", "zono", (4 * lo, 4 * hi))):
        what = f"reach of {kind} on [{a}, {b}] with {pre}, same model"
        out = reach_model(model, flat_set(kind, a, b), "approx", what,
                          precompute_bounds=pre)
        assert_covered(out, replay_onnx(path, samples(a, b), (1, 3)), what)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_inner_axis_concat_after_a_reshape_needs_a_tracked_shape(tmp_path,
                                                                 kind):
    """X (1, 2, 3, 5) -> Conv 1x1 and the input itself, each reshaped to
    (1, 6, 5) (no convolution reads them, so they become flat sets), then a
    Concat along axis 2, without ``input_shape``. An ImageStar gets its
    tensor shapes from its own height, width and channels and must answer
    with a set that holds every output; an ImageZono gets no tracked shape
    (registry SHAPE-3) and must refuse the inner-axis Concat with
    NotImplementedError, never stack the two operands as if the axis were
    the outer one. Catches either set type changing its side of this split
    without the other."""
    rng = np.random.default_rng(37)
    nodes = [helper.make_node("Conv", ["X", "K"], ["A"]),
             helper.make_node("Reshape", ["A", "S"], ["B"]),
             helper.make_node("Reshape", ["X", "S"], ["C"]),
             helper.make_node("Concat", ["B", "C"], ["Y"], axis=2)]
    path = save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, 6, 10],
                     inits={"K": _dyadic(rng, 2, 2, 1, 1),
                            "S": np.array([1, 6, 5], dtype=np.int64)})
    lo, hi = pixel_box(2, 3, 5, seed=37)
    what = f"Reshape then inner Concat on a {kind} without input_shape"
    if kind == "ImageZono":
        with pytest.raises(NotImplementedError):
            reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                        what)
        return
    out = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                      what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), (1, 2, 3, 5)),
                   what)


class _ViewPadConv(nn.Module):
    """x -> view (1, 2, 3, 5) -> nn.ZeroPad2d (left 1, bottom 2) ->
    Conv2d(2, 2, 2x2)."""

    def __init__(self):
        super().__init__()
        self.pad = nn.ZeroPad2d((1, 0, 0, 2))
        self.conv = nn.Conv2d(2, 2, (2, 2))

    def forward(self, x):
        return self.conv(self.pad(x.view(1, 2, 3, 5)))


_PAD_AFTER_VIEW = known_bug(
    7, "registry GRAPH-31: "
       "the walk decides whether a view or Reshape to an image feeds a "
       "convolution by walking past a fixed list of layout-keeping classes, "
       "which names the ONNX Pad modules but not nn.ZeroPad2d or "
       "nn.ConstantPad2d (nor nn.Dropout2d), so a flat set viewed as an "
       "image in front of torch padding and a convolution stays flat and the "
       "Pad kernel raises TypeError ('pad_star requires ImageStar', "
       "'pad_zono requires ImageZono') where the same model is answered on "
       "an image set", raises=TypeError)


@pytest.mark.parametrize("route,kind", [
    ("onnx-Pad", "Star"), ("onnx-Pad", "Zono"),
    ("torch-ZeroPad2d", "ImageStar"), ("torch-ZeroPad2d", "ImageZono"),
    pytest.param("torch-ZeroPad2d", "Star", marks=_PAD_AFTER_VIEW),
    pytest.param("torch-ZeroPad2d", "Zono", marks=_PAD_AFTER_VIEW)])
def test_padding_between_a_reshape_and_a_convolution(tmp_path, route, kind):
    """A flat input reshaped to a (2, 3, 5) image (an ONNX Reshape, or a
    torch ``view``), padded on the left by 1 and at the bottom by 2 (ONNX
    Pad, or ``nn.ZeroPad2d``) and convolved, the (2, 4, 5) result returned
    whole. The convolution needs the padded tensor as an image set, so the
    walk must look past the padding to make the reshaped set one. The ONNX
    route on a flat set and the torch route on an image set are the passing
    neighbours of the marked torch route on a flat set. Catches a
    layout-keeping layer missing from the walk's list (the set stays flat and
    the padding crashes) and padding applied in the stored order."""
    lo, hi = pixel_box(2, 3, 5, seed=38)
    if kind.startswith("Image"):
        sets, shape = image_set(kind, lo, hi), (1, 2, 3, 5)
    else:
        sets, shape = flat_set(kind, lo, hi), (1, 30)
    what = f"{route}/Conv2d on a {kind}"
    if route == "onnx-Pad":
        rng = np.random.default_rng(38)
        nodes = [helper.make_node("Reshape", ["X", "S"], ["R"]),
                 helper.make_node("Pad", ["R", "pads"], ["P"],
                                  mode="constant"),
                 helper.make_node("Conv", ["P", "K", "kb"], ["Y"])]
        path = save_onnx(tmp_path, nodes, [1, 30], [1, 2, 4, 5], inits={
            "S": np.array([1, 2, 3, 5], dtype=np.int64),
            "pads": np.array([0, 0, 0, 1, 0, 0, 2, 0], dtype=np.int64),
            "K": _dyadic(rng, 2, 2, 2, 2), "kb": _dyadic(rng, 2)})
        out = reach_model(load_onnx(path), sets, "approx", what)
        outputs = replay_onnx(path, samples(lo, hi), shape)
    else:
        model = _dyadic_module(_ViewPadConv(), seed=38)
        out = reach_model(model, sets, "approx", what)
        outputs = replay_torch(model, samples(lo, hi), shape)
    assert_covered(out, outputs, what, out_chw=(2, 4, 5))


@known_bug(16, "registry REACH-15: "
               "reach_pytorch_model reaches a torch model as it is "
               "given and lists nn.Dropout among the layers it passes "
               "through whatever the module's training flag, so a model in "
               "training mode, whose forward drops entries at random and "
               "scales the rest by 1 / (1 - p), is answered as the identity "
               "(NeuralNetwork sets eval mode; the entry itself does not)",
           raises=DID_NOT_RAISE)
def test_a_dropout_module_in_training_mode_is_refused():
    """``nn.Dropout`` in training mode is random, as ``F.dropout`` with a
    true flag is: the reach must be refused with NotImplementedError, like
    the functional spelling in ``test_value_changing_dropout_and_casts_are_
    refused``. The same module in eval mode is the passing neighbour
    (``test_value_keeping_dropout_and_casts_pass_the_sets_on[self.drop(h)-
    y-Star]``)."""
    model = _dyadic_module(_Middle("self.drop(h)", "y"), seed=31)
    model.train()
    assert model.drop.training, "setup: the module is not in training mode"
    lo, hi = flat_box(4, seed=5)
    with pytest.raises(NotImplementedError):
        reach_model(model, flat_set("Star", lo, hi), "approx",
                    "nn.Dropout in training mode")


@pytest.mark.parametrize("op", ["Gather", "ReduceSum", "ReduceMean"])
@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
def test_row_selection_and_reduce_on_a_hexatope_or_octatope_are_refused(
        tmp_path, op, kind):
    """Hexatope and Octatope have no row selection and no Reduce rule: a
    Gather, ReduceSum or ReduceMean of a (2, 3, 5) tensor (with
    ``input_shape``) must be refused with NotImplementedError, never
    answered with the input set or a guess."""
    if op == "Gather":
        node = helper.make_node("Gather", ["X", "I"], ["Y"], axis=2)
        inits, out_shape = {"I": np.array([2, 0], dtype=np.int64)}, [1, 2, 2, 5]
    elif op == "ReduceSum":
        node = helper.make_node("ReduceSum", ["X", "A"], ["Y"], keepdims=1)
        inits, out_shape = {"A": np.array([1], dtype=np.int64)}, [1, 1, 3, 5]
    else:
        node = helper.make_node("ReduceMean", ["X"], ["Y"], axes=[2],
                                keepdims=1)
        inits, out_shape = {}, [1, 2, 1, 5]
    path = save_onnx(tmp_path, [node], (1,) + _TENSOR, out_shape, inits=inits)
    lo, hi = pixel_box(*_TENSOR, seed=39)
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                    f"{op} on a {kind}", input_shape=_TENSOR)


class _TorchReshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(4, 4), nn.Linear(4, 3)

    def forward(self, x):
        return self.fc2(torch.reshape(self.fc1(x), (1, 4)))


def test_the_torch_reshape_function_is_refused():
    """``torch.reshape`` as a function node has no reach rule (the method
    spellings ``x.reshape`` and ``x.view`` have one): it must be refused with
    NotImplementedError, never skipped so that the next layer reads another
    node's sets."""
    model = _dyadic_module(_TorchReshape(), seed=40)
    lo, hi = flat_box(4, seed=40)
    with pytest.raises(NotImplementedError):
        reach_model(model, flat_set("Star", lo, hi), "approx",
                    "torch.reshape function")


@pytest.mark.parametrize("precompute", ["ibp", "zono"])
def test_image_pre_pass_bounds_are_laid_out_like_the_image_set(
        tmp_path, caplog, precompute):
    """Conv 1x1 identity -> Relu -> Conv 1x1 -> Relu on a (2, 3, 5)
    ImageStar whose channel 0 is in [-1, 1] and channel 1 in [-1/2, -1/4],
    with the bound pre-pass, the result returned whole. The pre-activation
    intervals of the two channels overlap, so bounds laid out channel-first
    and read in the image set's stored order give a neuron of channel 0 the
    bounds of channel 1 (stably inactive) without missing its own
    enclosure, which the ReLU rule would have caught and dropped with a
    warning. Catches bounds of the pre-pass recorded in the wrong layout;
    no 'precomputed bounds miss' warning may be logged."""
    import logging
    rng = np.random.default_rng(41)
    eye = np.eye(2, dtype=np.float32).reshape(2, 2, 1, 1)
    nodes = [helper.make_node("Conv", ["X", "K1", "k1"], ["A"]),
             helper.make_node("Relu", ["A"], ["R"]),
             helper.make_node("Conv", ["R", "K2", "k2"], ["B"]),
             helper.make_node("Relu", ["B"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, 2, 3, 5],
                     inits={"K1": eye, "k1": np.zeros(2, np.float32),
                            "K2": _dyadic(rng, 2, 2, 1, 1),
                            "k2": _dyadic(rng, 2)})
    lo = np.concatenate([-np.ones((1, 3, 5)), np.full((1, 3, 5), -0.5)])
    hi = np.concatenate([np.ones((1, 3, 5)), np.full((1, 3, 5), -0.25)])
    what = f"identity Conv/Relu/Conv/Relu with {precompute} on an ImageStar"
    with caplog.at_level(logging.WARNING):
        out = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                          "approx", what, precompute_bounds=precompute)
    missed = [r.getMessage() for r in caplog.records
              if "precomputed bounds miss" in r.getMessage()]
    assert not missed, f"{what}: {missed[0]}"
    assert_covered(out, replay_onnx(path, samples(lo, hi), (1, 2, 3, 5)),
                   what, out_chw=(2, 3, 5))


@pytest.mark.parametrize("join", ["Add-Relu-input", "Sub-Relu-LeakyRelu"])
def test_join_of_an_image_star_without_a_predicate_box_at_full_width(
        tmp_path, join):
    """The joins of ``test_onnx_join_of_two_image_branches_holds_every_
    output`` on an ImageStar given by its rows only (no predicate box), the
    joined (2, 3, 5) tensor returned whole instead of a 3-output Gemm of it.
    The tag variable of a joined set without a box is bounded by two rows;
    the last Gemm of the cells above projects a row put on the wrong
    variable away. Catches the tag's rows written on another variable."""
    rng = np.random.default_rng(43)
    second = "L" if join.startswith("Sub") else "A"
    nodes = [helper.make_node("Conv", ["X", "K", "kb"], ["A"]),
             helper.make_node("Relu", ["A"], ["R"]),
             helper.make_node("LeakyRelu", ["A"], ["L"], alpha=0.25),
             helper.make_node(join.split("-")[0], ["R", second], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 2, 3, 5], [1, 2, 3, 5],
                     inits={"K": _dyadic(rng, 2, 2, 1, 1),
                            "kb": _dyadic(rng, 2)})
    lo, hi = pixel_box(2, 3, 5, seed=11)
    sets, points = _image_inputs("ImageStar", "rows-only", lo, hi)
    what = f"Conv/{join} at full width on an ImageStar with rows only"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, (1, 2, 3, 5)), what,
                   out_chw=(2, 3, 5))


@pytest.mark.parametrize("act", ["LeakyRelu", "Relu"])
def test_join_of_two_relaxed_branches_on_a_zono(tmp_path, act):
    """Two relaxed branches of one flat Zono joined by Sub, the (1, 3)
    difference returned whole: act(W x) - act(W' x), where W' holds the rows
    of W in another order, so each branch appends one error generator per
    crossing neuron, the same number on both, for different neurons. A join
    that reads the two branches' error generators as the same noise symbols
    (column by column) cancels them and misses outputs. (The mirror model
    act(W x) - act(-W x) of the Star cells cannot show this on a Zono: its
    ReLU and LeakyReLU relaxations of t and -t differ by exactly t, so the
    cancelled difference is still the true one.)"""
    rng = np.random.default_rng(44)
    W = _dyadic(rng, 3, 3)
    attrs = {"alpha": 0.25} if act == "LeakyRelu" else {}
    nodes = [helper.make_node("MatMul", ["X", "Wt"], ["A"]),
             helper.make_node("MatMul", ["X", "Wpt"], ["B"]),
             helper.make_node(act, ["A"], ["P"], **attrs),
             helper.make_node(act, ["B"], ["Q"], **attrs),
             helper.make_node("Sub", ["P", "Q"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 3], [1, 3],
                     inits={"Wt": W.T.copy(), "Wpt": W[[1, 2, 0]].T.copy()},
                     name=f"zono-join-{act}")
    lo, hi = np.array([-1.0, -0.5, -0.75]), np.array([1.0, 0.5, 0.75])
    what = f"{act}(Wx) - {act}(W'x) on a Zono"
    out = reach_model(load_onnx(path), flat_set("Zono", lo, hi), "approx",
                      what)
    assert_covered(out, replay_onnx(path, samples(lo, hi), (1, 3)), what)


@pytest.mark.parametrize("kind,form", [("Star", "row"), ("Zono", "box"),
                                       ("Box", "several")])
def test_a_constant_node_operand_is_read_as_its_value(tmp_path, kind, form):
    """Two ONNX Constant nodes (not initializers) as operands: C1 - X with
    the constant first, then a division by C2, the (1, 4) result returned
    whole. Both constants have both signs and magnitudes above and below one.
    Catches the value of a Constant node read wrongly by the constant
    folding (``module.value``), and a Sub with its constant first read as X
    - C1."""
    c1 = np.array([[1.5, -0.25, 3.0, -2.0]], dtype=np.float32)
    c2 = np.array([[2.0, -0.5, 4.0, -0.25]], dtype=np.float32)
    nodes = [helper.make_node("Constant", [], ["C1"],
                              value=numpy_helper.from_array(c1)),
             helper.make_node("Constant", [], ["C2"],
                              value=numpy_helper.from_array(c2)),
             helper.make_node("Sub", ["C1", "X"], ["D"]),
             helper.make_node("Div", ["D", "C2"], ["Y"])]
    path = save_onnx(tmp_path, nodes, [1, 4], [1, 4], name="constant-nodes")
    lo, hi = flat_box(4, seed=42)
    sets, points = _flat_inputs(kind, form, 4 * lo, 4 * hi)
    what = f"(C1 - X) / C2 with Constant nodes on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, (1, 4)), what)


def test_leaky_relu_slope_given_by_position_in_a_built_graph():
    """A GraphModule built by hand (not traced) in which ``F.leaky_relu``
    carries its slope 1/4 as a positional argument, the only way to reach
    the positional branch of ``_function_node_to_module`` (torch.fx records
    the slope as a keyword when it traces a forward). Exact method on a
    Star with three predicate variables, both directions against the exact
    image in rationals. Catches the positional slope ignored (the default
    0.01) or read from the wrong position."""
    root = _dyadic_module(_TwoLayer(), seed=45)
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    h = graph.call_module("f1", (x,))
    r = graph.call_function(F.leaky_relu, (h, 0.25))
    graph.output(graph.call_module("f2", (r,)))
    model = torch.fx.GraphModule(root, graph)
    star, x0, E = _low_rank_star(np.random.default_rng(45), 2)
    what = "exact built graph with F.leaky_relu(h, 0.25)"
    out = reach_model(model, star, "exact", what)
    W1, b1 = _linear_of(root.f1)
    W2, b2 = _linear_of(root.f2)
    pieces = rational.affine(rational.box([-1] * 3, [1] * 3), E.tolist(),
                             x0.tolist())
    pieces = rational.leaky_relu(rational.affine(pieces, W1, b1),
                                 Fraction(1, 4))
    _assert_exact(out, rational.affine(pieces, W2, b2), what)
    assert_covered(out, replay_torch(model, _low_rank_points(x0, E, 14),
                                     (1, 2)), what)


def test_flat_batchnorm_of_a_box_holds_the_exact_value_of_cancelling_terms(
        tmp_path):
    """ONNX BatchNormalization (float64 graph) of a (1, 3, 4) tensor carried
    as a flat Box with ``input_shape`` (3, 4), the result returned whole. The
    normalisation divides by exactly 3/4; gamma / (3/4) is not a dyadic
    number for two channels, and on the third channel the terms cancel
    (mean -1234.5 against beta, leaving about 2^-10), so a scale and shift
    computed in float64 and applied to nearest miss the real value by about
    1e-13. Each returned interval must hold the exact interval of its entry,
    computed in fractions from the stored parameters, with zero slack.
    Catches the flat BatchNorm rule of a Box without the rounding margin of
    its folded scale and shift."""
    gamma = [1.0, 0.75, 1.0978]
    mean = [0.1875, -0.375, -1234.5]
    beta = [0.0625, -0.5,
            float(-(Fraction(gamma[2]) / Fraction(3, 4))
                  * (Fraction(0.25) - Fraction(mean[2])) + Fraction(1, 1024))]
    var = 0.5625 - 2.0 ** -10
    nodes = [helper.make_node("BatchNormalization",
                              ["X", "g", "be", "mu", "var"], ["Y"],
                              epsilon=2.0 ** -10)]
    path = save_onnx(tmp_path, nodes, [1, 3, 4], [1, 3, 4],
                     inits={"g": np.array(gamma), "be": np.array(beta),
                            "mu": np.array(mean), "var": np.full(3, var)},
                     in_type="DOUBLE", out_type="DOUBLE", name="bn-box")
    lo, hi = flat_box(12, seed=46)
    lo, hi = lo.reshape(3, 4), hi.reshape(3, 4)
    lo[2], hi[2] = [0.125, 0.25, -0.5, 0.25], [0.375, 0.25, 0.5, 0.75]
    what = "flat BatchNormalization of (3, 4) on a Box"
    out = reach_model(load_onnx(path), [Box(lo.reshape(-1), hi.reshape(-1))],
                      "approx", what, input_shape=(3, 4))
    if Fraction(var) + Fraction(2.0 ** -10) != Fraction(9, 16):
        pytest.fail("var + eps is not (3/4)^2 exactly")
    for k, (a, b) in enumerate(zip(lo.reshape(-1), hi.reshape(-1))):
        ch = k // 4
        scale = Fraction(gamma[ch]) / Fraction(3, 4)
        ends = [scale * (Fraction(float(v)) - Fraction(mean[ch]))
                + Fraction(beta[ch]) for v in (a, b)]
        got_lo = Fraction(float(np.asarray(out[0].lb).reshape(-1)[k]))
        got_hi = Fraction(float(np.asarray(out[0].ub).reshape(-1)[k]))
        assert got_lo <= min(ends) and got_hi >= max(ends), (
            f"{what}: entry {k} is [{float(got_lo)!r}, {float(got_hi)!r}], "
            f"the exact interval [{float(min(ends))!r}, "
            f"{float(max(ends))!r}]")
    assert_covered(out, replay_onnx(path, samples(lo, hi), (1, 3, 4)), what)


@pytest.mark.parametrize("axis", [1, -1])
@pytest.mark.parametrize("kind,form", [("Star", "row"), ("Box", "several")])
def test_opset11_softmax_over_the_flattened_tail(tmp_path, axis, kind, form):
    """ONNX Softmax of opset 11 (float64) on a (1, 2, 3) tensor carried as a
    flat set: with axis 1 it normalises over the flattened tail (all six
    entries), with axis -1 over each row of three, the result returned
    whole. Opset 13 normalises over the one axis only, so the two opsets
    differ for axis 1. Catches the opset-11 semantics read as the single-axis
    one, and the tail taken from the wrong axis."""
    nodes = [helper.make_node("Softmax", ["X"], ["Y"], axis=axis)]
    path = save_onnx(tmp_path, nodes, [1, 2, 3], [1, 2, 3], opset=11,
                     in_type="DOUBLE", out_type="DOUBLE",
                     name=f"softmax11-{axis}")
    lo, hi = flat_box(6, seed=47)
    lo, hi = lo.reshape(2, 3), hi.reshape(2, 3)
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"opset-11 Softmax(axis={axis}) of (1, 2, 3) on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what,
                      input_shape=(2, 3))
    assert_covered(out, replay_onnx(path, points, (1, 2, 3)), what)


@pytest.mark.parametrize("kind,form", [("Star", "row"), ("Zono", "box"),
                                       ("Box", "several")])
def test_concat_with_a_constant_operand_holds_every_output(tmp_path, kind,
                                                           form):
    """ONNX Concat of a constant (1, 2, 3) tensor (an initializer, as the
    class token of a vision transformer) and the input X (1, 2, 3) along
    axis 2, the (1, 2, 6) result returned whole, on a flat set with
    ``input_shape``. The constant must occupy its own entries, in its own
    row-major order, and the rows of X be interleaved after it in each row.
    Catches a constant operand dropped, read in column-major order, or put
    after the set operand."""
    rng = np.random.default_rng(48)
    nodes = [helper.make_node("Concat", ["C", "X"], ["Y"], axis=2)]
    path = save_onnx(tmp_path, nodes, [1, 2, 3], [1, 2, 6],
                     inits={"C": (rng.integers(-12, 13, size=(1, 2, 3)) / 4.0)
                            .astype(np.float32)})
    lo, hi = flat_box(6, seed=48)
    lo, hi = lo.reshape(2, 3), hi.reshape(2, 3)
    sets, points = _flat_inputs(kind, form, lo, hi)
    what = f"Concat(constant, X) along axis 2 on a {form} {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what,
                      input_shape=(2, 3))
    assert_covered(out, replay_onnx(path, points, (1, 2, 3)), what)


class _ChannelSums(nn.Module):
    """Flatten a (2, 3, 5) image and return the sum of each channel."""

    def __init__(self):
        super().__init__()
        self.flat, self.fc = nn.Flatten(), nn.Linear(30, 2)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.zero_()
            self.fc.weight[0, :15] = 1.0
            self.fc.weight[1, 15:] = 1.0

    def forward(self, x):
        return self.fc(self.flat(x))


_PROBABILISTIC_IMAGE_LAYOUT = known_bug(
    4, "registry PROB-15: method='probabilistic', the old alias of conformal "
       "reach that PLAN step 4 removes (registry REACH-2), builds its "
       "sampling box from estimate_ranges of an ImageStar, which is in the "
       "stored (height, width, channel) order, and reshapes the samples to "
       "(C, H, W), so it samples another input region and its box holds none "
       "of the true outputs")


@pytest.mark.parametrize("kind", [
    "Star", pytest.param("ImageStar", marks=_PROBABILISTIC_IMAGE_LAYOUT)])
def test_probabilistic_alias_samples_the_input_region_in_the_tensor_order(
        kind):
    """``method='probabilistic'`` on a (2, 3, 5) image whose channel 0 lies in
    [0, 1] and channel 1 in [10, 11], through a model that returns the sum
    of each channel, given as an ImageStar (stored height, width, channel)
    or as a flat Star in the tensor's order with ``input_shape``. The alias
    must be refused with ValueError naming its replacement ``conformal``
    (registry REACH-2, PLAN step 4); while it answers, its box must hold the
    true outputs at its stated coverage. With ``ell = m = 200`` the box
    covers at least the largest calibration score, so the coverage of a
    correct implementation is below 0.9 with probability 0.9^200 < 1e-9, and
    then more than 100 of 200 fresh true outputs outside has probability
    below 1e-40: the cell demands at most 100 outside (the seeds are fixed).
    Catches an input region read in the stored order of an image set and
    sampled as a channel-first tensor. The flat Star is the passing
    neighbour with the same model and check."""
    lo = np.concatenate([np.zeros((1, 3, 5)), np.full((1, 3, 5), 10.0)])
    hi = lo + 1.0
    model = _ChannelSums().eval()
    if kind == "ImageStar":
        sets, options = image_set("ImageStar", lo, hi), {}
    else:
        sets, options = flat_set("Star", lo, hi), {"input_shape": (2, 3, 5)}
    what = f"method='probabilistic' on a {kind}"
    try:
        box = reach_model(model, sets, "probabilistic", what, m=200, ell=200,
                          epsilon=0.05, surrogate="naive",
                          training_samples=100, seed=0, **options)[0]
    except ValueError as refusal:
        if "conformal" not in str(refusal):
            raise
        return                  # the designed refusal of the removed alias
    points = lo + (hi - lo) * np.random.default_rng(2).random((200, 2, 3, 5))
    ys = np.array(replay_torch(model, list(points), (1, 2, 3, 5)))
    lb = np.asarray(box.lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(box.ub, dtype=np.float64).reshape(-1)
    outside = int(np.sum(np.any((ys < lb) | (ys > ub), axis=1)))
    assert outside <= 100, (f"{what}: {outside} of 200 true outputs are "
                            f"outside the box [{lb}, {ub}]")


# --------------------------------------------------------------------------- #
# 16. Shape queries in a view target, the warning of the configuration,       #
#     dropout between a view and a convolution                                #
# --------------------------------------------------------------------------- #
class _BatchEntryView(nn.Module):
    """Conv 1x1 of a (2, 3, 5) image, flattened by a view or reshape whose
    batch entry is a traced shape query, then ReLU and Linear."""

    def __init__(self, how):
        super().__init__()
        self.conv, self.fc = nn.Conv2d(2, 2, 1), nn.Linear(30, 3)
        self.how = how

    def forward(self, x):
        h = self.conv(x)
        if self.how == "view(h.size(0), -1)":
            h = h.view(h.size(0), -1)
        elif self.how == "reshape(h.shape[0], -1)":
            h = h.reshape(h.shape[0], -1)
        else:
            h = h.view(h.size()[0], -1)
        return self.fc(F.relu(h))


@pytest.mark.parametrize("how,kind", [
    ("view(h.size(0), -1)", "ImageStar"),
    ("reshape(h.shape[0], -1)", "ImageStar"),
    ("view(h.size()[0], -1)", "ImageStar"),
    ("view(h.size(0), -1)", "ImageZono")])
def test_a_shape_query_in_the_batch_entry_of_a_view_holds_every_output(how,
                                                                      kind):
    """A torch forward that flattens with the batch entry read from the
    tensor (``h.size(0)``, ``h.shape[0]``, ``h.size()[0]``), the common
    spelling of a classifier head: the walk must keep these shape queries
    out of the set graph (they are tensor-shape values, not sets) and answer
    with a set that holds every replayed output, the image flattened in the
    tensor's channel-first order. Catches a shape query (the ``size``
    method, ``getattr`` of ``shape``, a ``getitem`` on either) that is
    refused or read as a set."""
    model = _dyadic_module(_BatchEntryView(how), seed=49)
    lo, hi = pixel_box(2, 3, 5, seed=49)
    what = f"Conv/{how}/Relu/Linear on a {kind}"
    out = reach_model(model, image_set(kind, lo, hi), "approx", what)
    assert_covered(out, replay_torch(model, samples(lo, hi), (1, 2, 3, 5)),
                   what)


def test_relax_options_under_the_exact_method_warn_that_they_are_ignored():
    """``ReachConfig`` documents that ``relax_factor`` and ``relax_method``
    are ignored by the exact method and that it warns when either is set:
    each of the two, set with ``method='exact'``, must warn, and the default
    configuration and an approx one must not (the suite turns any other
    warning into an error). Catches the warning dropped, or raised only
    when both options are set."""
    with pytest.warns(UserWarning, match="ignored"):
        ReachConfig(method="exact", relax_factor=0.25)
    with pytest.warns(UserWarning, match="ignored"):
        ReachConfig(method="exact", relax_method="area")
    ReachConfig(method="exact")
    ReachConfig(method="approx", relax_factor=0.25)


class _ViewDropoutConv(nn.Module):
    """x -> view (1, 2, 3, 5) -> a dropout in eval mode -> Conv2d(2, 2, 2x2)."""

    def __init__(self, drop):
        super().__init__()
        self.drop, self.conv = drop, nn.Conv2d(2, 2, (2, 2))

    def forward(self, x):
        return self.conv(self.drop(x.view(1, 2, 3, 5)))


_DROPOUT2D_AFTER_VIEW = known_bug(
    7, "registry GRAPH-31: "
       "the walk's list of layout-keeping classes, which it looks past "
       "to decide whether a view feeds a convolution, lacks nn.Dropout2d "
       "(as it lacks the torch padding classes), so a flat set viewed as an "
       "image and passed through an eval-mode Dropout2d stays flat and the "
       "convolution raises ValueError ('Conv2D requires ImageStar input') "
       "where the same model is answered on an image set", raises=ValueError)


@pytest.mark.parametrize("drop,kind", [
    ("Dropout", "Star"), ("Dropout2d", "ImageStar"),
    pytest.param("Dropout2d", "Star", marks=_DROPOUT2D_AFTER_VIEW)])
def test_dropout_between_a_view_and_a_convolution(drop, kind):
    """A flat input viewed as a (2, 3, 5) image, passed through a dropout in
    eval mode (the identity) and convolved, the (2, 2, 4) result returned
    whole. The walk must look past the dropout to make the viewed set an
    image set. ``nn.Dropout`` on a flat set and ``nn.Dropout2d`` on an image
    set are the passing neighbours of the marked ``nn.Dropout2d`` on a flat
    set. Catches a layout-keeping layer missing from the walk's list."""
    layer = nn.Dropout(0.5) if drop == "Dropout" else nn.Dropout2d(0.5)
    model = _dyadic_module(_ViewDropoutConv(layer), seed=50)
    lo, hi = pixel_box(2, 3, 5, seed=50)
    if kind == "Star":
        sets, shape = flat_set(kind, lo, hi), (1, 30)
    else:
        sets, shape = image_set(kind, lo, hi), (1, 2, 3, 5)
    what = f"view/{drop}/Conv2d on a {kind}"
    out = reach_model(model, sets, "approx", what)
    assert_covered(out, replay_torch(model, samples(lo, hi), shape), what,
                   out_chw=(2, 2, 4))

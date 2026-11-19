"""Soundness cells of the Linear layer on every set type.

``nn.Linear`` is reached through ``reach_layer``, through a torch model
(``reach_pytorch_model``; ``NeuralNetwork.reach`` calls it after
``model.eval()``) and through an ONNX file, whose Gemm nodes the loader turns
into ``nn.Linear``. Each kernel of ``n2v/nn/layer_ops/linear_reach.py`` reads
the stored weight and bias and calls the set's own ``affine_map`` once per
input set, with or without the bias; it does not read the method. The cells
follow tests/README.md "A soundness cell".

Oracle. Each input set is {c + G a : -1 <= a <= 1, r . a <= 1}, built here
from arrays written in this file, and its exact image is a union of
vertex-described pieces in fractions (tests/_oracles/rational.py). Every
vertex of every exact piece must be in one returned set, decided in exact
arithmetic with no slack (``escapes``, ``straddlers``); where the method is
'exact', every vertex of every returned set must also have a preimage
(``unreached``). On top, member points of the input are replayed through
the layer in torch float64 (or the model on onnxruntime) and must be members
of the returned sets by one LP on each set's own rows (``assert_covered``,
slack ``LP_TOL`` only). Weights, biases and inputs are dyadic, so float32,
float64 and the fractions agree, except in the rounding cells at the end,
whose data has full mantissas on purpose.

Inputs. Four entries: straddling zero, positive, of zero width, negative
(``flat_box``). Three families: one set; three sets that cut the box along
one coordinate at dyadic points (shares 1, 2 and 5 of 8, the shape of an
exact split); one set cut by a binding row that is not a box row (a general
row for Star, a difference row for Hexatope, a sum row for Octatope; for
Zono, which has no rows, generators with several entries and more generators
than entries, the shape a ReLU relaxation leaves). Layers: more outputs than
inputs with a bias and a zero row, fewer outputs without bias, a square one
without bias (where a kernel that returns its input unchanged still has the
right size), a single output, and one whose parameters are frozen
(``requires_grad`` False, the usual state of a network handed to a verifier).

Defects of today's library marked here (strict expected failures):

- The affine map of a Hexatope and of an Octatope rounds to nearest with no
  error term, so the set returned for a Linear layer can leave out the exact
  image of the stored set by units in the last place (registry ROUND-24; the
  Star and Zono form of the same rounding is registry ROUND-8, PLAN step 7b,
  marked in ``tests/contracts/test_verdicts.py`` and not repeated here).
- ONNX Gemm with an alpha or beta that is not a power of two: the loader
  folds them into the float32 weight and bias and rounds the products, so the
  reach set is that of another network (registry LOAD-8).
- ``reach_layer(..., 'exact')`` on a Box answers the outward bounding box of
  a map that mixes coordinates, not its image (registry REACH-8, of the kind of
  registry REACH-4).
- A flat set that stands for a (1, 2, 4) tensor (``input_shape``): a
  Linear(8, 3), which torch cannot apply to that tensor, is answered (registry
  AFFINE-6).

Image sets: ``tests/contracts/test_layer_refusals.py`` holds Linear handed to
a (2, 3, 4) image set (registry AFFINE-5, and the refusals of the legal
spellings); the image cell here is the one shape where torch's Linear maps the
whole image and the library answers, a 1 x 1 x W image.
"""

import itertools

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper

from n2v.nn import NeuralNetwork
from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.sets.hexatope import DifferenceConstraintSystem
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.sets.octatope import UTVPIConstraintSystem
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational as R
from tests.cell_support import (assert_covered, constrained_star, flat_box,
                                reach_model, reach_one_layer, replay_onnx,
                                replay_torch, samples, save_onnx)
from tests.known_bugs import DID_NOT_RAISE, known_bug

_N_IN = 4
_FAMILIES = ("one", "several", "row")
_KIND_METHODS = (("Box", "approx"), ("Zono", "approx"), ("Hexatope", "approx"),
                 ("Octatope", "approx"), ("Star", "approx"), ("Star", "exact"))
# One binding row r . a <= 1 per kind: each cuts one corner off the cube of
# predicate variables and leaves every variable's range at [-1, 1].
_ROW = {"Star": (1.0, -0.5, 0.25), "Hexatope": (1.0, -1.0, 0.0),
        "Octatope": (1.0, 1.0, 0.0)}
_REFUSALS = (NotImplementedError, ValueError)


def _box_generators(lo, hi):
    """``(c, G)`` of the box [lo, hi]: one generator, half the width, per
    coordinate of positive width; a zero-width coordinate is in ``c`` only."""
    live = np.flatnonzero(hi > lo)
    G = np.zeros((lo.size, live.size))
    G[live, np.arange(live.size)] = (hi - lo)[live] / 2
    return (lo + hi) / 2, G


def _full_row(row, k):
    return None if row is None else np.array(list(row) + [0.0] * (k - len(row)))


def _build(kind, c, G, row=None):
    """The set {c + G a : -1 <= a <= 1, row . a <= 1} as an n2v set of
    ``kind``, from the arrays given (no ``from_bounds``). A Hexatope gets an
    anchor variable 0 with a zero generator, as ``Hexatope.from_bounds``
    does; its rows bound every other variable against the anchor."""
    n, k = G.shape
    row = _full_row(row, k)
    if kind == "Box":
        if row is not None or np.count_nonzero(G, axis=1).max(initial=0) > 1:
            raise ValueError("a Box holds an axis-aligned box only")
        radius = np.abs(G).sum(axis=1)
        return Box(c - radius, c + radius)
    if kind == "Zono":
        if row is not None:
            raise ValueError("a Zono has no rows")
        return Zono(c.reshape(-1, 1), G.copy())
    if kind == "Star":
        C = np.vstack([np.eye(k), -np.eye(k)]
                      + ([] if row is None else [row[None, :]]))
        return Star(np.hstack([c.reshape(-1, 1), G]), C,
                    np.ones((C.shape[0], 1)), -np.ones((k, 1)),
                    np.ones((k, 1)))
    if kind == "Hexatope":
        dcs = DifferenceConstraintSystem(k + 1)
        for i in range(1, k + 1):
            dcs.add_constraint(i, 0, 1.0)
            dcs.add_constraint(0, i, 1.0)
        if row is not None:
            i, j = int(np.flatnonzero(row == 1)[0]), int(np.flatnonzero(row == -1)[0])
            dcs.add_constraint(i + 1, j + 1, 1.0)
        return Hexatope(c.copy(), np.hstack([np.zeros((n, 1)), G]), dcs)
    if kind == "Octatope":
        utvpi = UTVPIConstraintSystem(k)
        for i in range(k):
            utvpi.add_constraint(i, i, 1, 0, 1.0)
            utvpi.add_constraint(i, i, -1, 0, 1.0)
        if row is not None:
            i, j = (int(v) for v in np.flatnonzero(row))
            utvpi.add_constraint(i, j, int(row[i]), int(row[j]), 1.0)
        return Octatope(c.copy(), G.copy(), utvpi)
    raise ValueError(f"unknown kind {kind!r}")


def _exact_input(c, G, row=None):
    """The same set as rational pieces: the identity on the predicate
    polytope, followed by ``a -> c + G a``."""
    k = G.shape[1]
    row = _full_row(row, k)
    A = np.vstack([np.eye(k), -np.eye(k)]
                  + ([] if row is None else [row[None, :]]))
    return R.affine(R.polytope(A.tolist(), [1.0] * A.shape[0]),
                    G.tolist(), c.tolist())


def _member_points(c, G, row=None, levels=5):
    """Member points in the set's own order, computed here: the grid of
    ``levels`` values per predicate variable from -1 to 1 (every corner of
    the cube included); with a row, each grid point beyond it is moved along
    the first variable onto the cut face, or dropped when that leaves the
    cube."""
    k = G.shape[1]
    row = _full_row(row, k)
    points = []
    for a in itertools.product(np.linspace(-1.0, 1.0, levels), repeat=k):
        a = np.array(a)
        if row is not None and row @ a > 1:
            a[0] -= (row @ a - 1) / row[0]
            if abs(a[0]) > 1:
                continue
        points.append(c + G @ a)
    return points


def _family(kind, family, seed=0, shift=0.0):
    """``(sets, pieces, points)`` of one input family of ``_N_IN`` entries:
    the n2v input sets, their exact union as rational pieces, and member
    points covering every set (module docstring). ``shift`` (dyadic) is
    added to every bound, to place a family away from another one."""
    lo, hi = flat_box(_N_IN, seed=seed, zero_width=True)
    lo, hi = lo + shift, hi + shift
    if family == "several":
        ends = lo[0] + (hi[0] - lo[0]) * np.array([0, 1, 3, 8]) / 8
        parts = []
        for a, b in zip(ends[:-1], ends[1:]):
            part_lo, part_hi = lo.copy(), hi.copy()
            part_lo[0], part_hi[0] = a, b
            parts.append(_box_generators(part_lo, part_hi) + (None,))
        levels = 3
    elif family in ("one", "row"):
        c, G = _box_generators(lo, hi)
        row = None
        levels = 5
        if family == "row" and kind == "Zono":
            G[:, 0] += 0.5 * G[:, 1]           # a generator with two entries
            # Two more generators than the box has: five for four entries
            # (a kernel may not take one generator per entry for granted).
            # The entry of zero width keeps its zero row.
            more = np.array([[0.25, 0.0], [0.0, 0.125], [0.5, -0.25],
                             [-0.125, 0.25]])
            more[hi == lo] = 0.0
            G = np.hstack([G, more])
            levels = 3
        elif family == "row":
            row = _ROW[kind]
        parts = [(c, G, row)]
    else:
        raise ValueError(f"unknown family {family!r}")
    sets = [_build(kind, c, G, row) for c, G, row in parts]
    pieces = [p for c, G, row in parts for p in _exact_input(c, G, row)]
    points = [x for c, G, row in parts
              for x in _member_points(c, G, row, levels)]
    return sets, pieces, points


def _assert_holds_exact_image(sets, pieces, what, exact=False):
    """Every vertex of every exact piece is in one returned set, in exact
    arithmetic; with ``exact`` also every vertex of every returned set is in
    the exact image."""
    computed = [R.star(s) for s in sets]
    missed = R.escapes(pieces, *computed)
    assert not missed, (
        f"{what}: {len(missed)} vertices of the exact image are in no "
        f"returned set; first: piece {missed[0][0]}, vertex "
        f"{[float(v) for v in missed[0][1]]}")
    spread = R.straddlers(pieces, *computed)
    assert not spread, (
        f"{what}: the exact image of input piece(s) {spread} is not held by "
        f"any single returned set")
    if exact:
        for i, s in enumerate(computed):
            extra = R.unreached(pieces, R.set_vertices(s))
            assert not extra, (
                f"{what}: returned set {i} has {len(extra)} vertices that no "
                f"input point reaches; first {[float(v) for v in extra[0][1]]}")


def _dyadic_linear(n_in, n_out, bias, seed, zero_row=None, dtype=torch.float32,
                   frozen=False):
    rng = np.random.default_rng(seed)
    weight = rng.integers(-4, 5, size=(n_out, n_in)) / 4.0
    if zero_row is not None:
        weight[zero_row] = 0.0
    layer = nn.Linear(n_in, n_out, bias=bias, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight))
        if bias:
            layer.bias.copy_(torch.from_numpy(
                rng.integers(-8, 9, size=n_out) / 8.0))
    if frozen:
        layer.requires_grad_(False)
    return layer


_SHAPES = {
    "wide": dict(n_out=5, bias=True, zero_row=2),     # more outputs, a zero row
    "narrow": dict(n_out=3, bias=False),              # fewer outputs, no bias
    "square": dict(n_out=4, bias=False),              # an unmapped input fits
    "single": dict(n_out=1, bias=True),               # one output
    "frozen": dict(n_out=3, bias=True, frozen=True),  # requires_grad False
    "float64": dict(n_out=4, bias=True, dtype=torch.float64),
}


def _layer(shape, seed=11):
    return _dyadic_linear(_N_IN, seed=seed, **_SHAPES[shape])


def _affine_of(layer):
    W = layer.weight.detach().double().numpy().tolist()
    b = None if layer.bias is None else layer.bias.detach().double().numpy().tolist()
    return W, b


def _linear_cells():
    """``(kind, method, family, shape)`` of the one-layer cell. Every flat
    kind and method on every family (no row family for a Box) and the three
    layer shapes, except that Star approx keeps the row family only: the
    Star lane does not read the method for Linear, so Star approx runs the
    code of Star exact, whose cells check both directions. The single output
    and the layer with frozen parameters are checked once per kind (each
    kernel reads the weight and the bias itself)."""
    cells = []
    for kind, method in _KIND_METHODS:
        star_approx = kind == "Star" and method == "approx"
        for family in _FAMILIES:
            if (kind == "Box" and family == "row") or (
                    star_approx and family != "row"):
                continue
            for shape in ("wide", "narrow", "square"):
                cells.append(pytest.param(
                    kind, method, family, shape,
                    id=f"{kind}-{method}-{family}-{shape}"))
        if not star_approx:
            for shape in ("single", "frozen"):
                cells.append(pytest.param(kind, method, "one", shape,
                                          id=f"{kind}-{method}-one-{shape}"))
    return cells


@pytest.mark.parametrize("kind,method,family,shape", _linear_cells())
def test_linear_holds_the_exact_image_of_every_input_set(kind, method, family,
                                                          shape):
    """``reach_layer(nn.Linear)`` on every flat set type. Catches a kernel
    that maps without the bias or without the weights (a square layer without
    bias makes an unmapped input the right size), one that reads a parameter
    only when it requires a gradient, one that maps as many generators of a
    Zono as it has entries and drops the rest, one that maps only the
    first input set or the first set on every pass, one whose set-level
    ``affine_map`` puts the bias on the generators, and one that loses or
    changes a constraint row (the row family). For the exact method a result
    larger than the image fails too."""
    layer = _layer(shape)
    sets, pieces, points = _family(kind, family)
    what = f"Linear ({shape}) on {kind}, {method}, {family}"
    result = reach_one_layer(layer, sets, method, what)
    assert all(type(s) is type(sets[0]) for s in result), (
        f"{what}: returned {[type(s).__name__ for s in result]}")
    W, b = _affine_of(layer)
    _assert_holds_exact_image(result, R.affine(pieces, W, b), what,
                              exact=(method == "exact"))
    assert_covered(result, replay_torch(layer, points, (1, _N_IN)), what)


@pytest.mark.parametrize("kind,method", [("Star", "exact"), ("Zono", "approx"),
                                         ("Box", "approx"),
                                         ("Hexatope", "approx"),
                                         ("Octatope", "approx")])
def test_float64_linear_layer_holds_the_exact_image(kind, method):
    """A Linear layer stored in float64 (the dtype the BatchNorm folding and
    a model converted with ``.double()`` give): the weights are read in that
    dtype, not cast through float32. One case per kernel, since each kernel
    reads the weights itself."""
    layer = _layer("float64")
    with torch.no_grad():
        layer.weight[0, 0] += 2.0 ** -30        # not a float32 number
    sets, pieces, points = _family(kind, "several")
    what = f"float64 Linear on {kind}"
    result = reach_one_layer(layer, sets, method, what)
    W, b = _affine_of(layer)
    _assert_holds_exact_image(result, R.affine(pieces, W, b), what,
                              exact=(method == "exact"))
    assert_covered(result, replay_torch(layer, points, (1, _N_IN)), what)


# --------------------------------------------------------------------------- #
# Linear, then the next layer, through the model entry points                   #
# --------------------------------------------------------------------------- #
def _mlp(middle):
    """Linear(4, 3) with a bias, then ``middle``, then Linear(3, 2) without:
    the second layer reads the rows the first one wrote."""
    return nn.Sequential(_dyadic_linear(_N_IN, 3, True, seed=21), middle,
                         _dyadic_linear(3, 2, False, seed=22)).eval()


def _exact_mlp(model, pieces):
    first, middle, last = model
    W1, b1 = _affine_of(first)
    W2, b2 = _affine_of(last)
    hidden = R.affine(pieces, W1, b1)
    if isinstance(middle, nn.ReLU):
        hidden = R.relu(hidden)
    return R.affine(hidden, W2, b2)


# No ReLU behind a Zono, Hexatope or Octatope here: the Zono ReLU relaxation
# rounds to nearest (registry ROUND-10) and the Hexatope and Octatope split
# carries unpadded solver bounds (HEXOCT-4); both belong to other cells, and
# the exact check here has no slack.
_MODEL_CELLS = (("Star", "exact", "one", "ReLU"), ("Star", "exact", "row", "ReLU"),
                ("Star", "approx", "row", "ReLU"), ("Box", "approx", "one", "ReLU"),
                ("Zono", "approx", "row", "Dropout"),
                ("Hexatope", "approx", "row", "Dropout"),
                ("Octatope", "approx", "row", "Dropout"))


def _model_cells():
    """Every cell through ``reach_pytorch_model``. ``NeuralNetwork.reach``
    only calls ``model.eval()`` and then that entry point, so it is kept on
    the two Star row cells (both methods), not on every set type."""
    cells = []
    for kind, method, family, middle in _MODEL_CELLS:
        entries = ["reach_pytorch_model"]
        if kind == "Star" and family == "row":
            entries.append("NeuralNetwork")
        for entry in entries:
            cells.append(pytest.param(kind, method, family, middle, entry,
                                      id=f"{kind}-{method}-{family}-{entry}"))
    return cells


@pytest.mark.parametrize("kind,method,family,middle,entry", _model_cells())
def test_linear_layers_in_a_model_hold_the_exact_image(kind, method, family,
                                                        middle, entry):
    """Linear, ReLU (or a pass-through), Linear through the model entry
    points, so the second Linear consumes the rows and the basis the first
    one wrote. Catches what the one-layer cell catches, seen through the node
    loop of ``reach_pytorch_model`` and ``NeuralNetwork.reach``."""
    model = _mlp(nn.ReLU() if middle == "ReLU" else nn.Dropout(p=0.5))
    sets, pieces, points = _family(kind, family)
    what = f"Linear, {middle}, Linear on {kind} ({method}, {family}) by {entry}"
    if entry == "NeuralNetwork":
        network = NeuralNetwork(model)
        result = []
        for s in sets:
            result.extend(network.reach(s, method=method))
        if not result:
            pytest.fail(f"{what}: no sets returned")
    else:
        result = reach_model(model, sets, method, what)
    _assert_holds_exact_image(result, _exact_mlp(model, pieces), what,
                              exact=(method == "exact"))
    assert_covered(result, replay_torch(model, points, (1, _N_IN)), what)


def _gemm_model(tmp_path, bias_2d=False):
    """ONNX Gemm (transB=1, alpha=1/2, beta=2, bias C), Relu, Gemm
    (transB=0, no bias). With a 1-D C both Gemm nodes load as nn.Linear; a
    2-D C keeps the first one an OnnxGemm module."""
    rng = np.random.default_rng(31)
    W1 = (rng.integers(-4, 5, size=(3, _N_IN)) / 4.0).astype(np.float32)
    B1 = (rng.integers(-8, 9, size=3) / 8.0).astype(np.float32)
    W2 = (rng.integers(-4, 5, size=(3, 2)) / 4.0).astype(np.float32)
    nodes = [helper.make_node("Gemm", ["X", "W1", "B1"], ["A"], transB=1,
                              alpha=0.5, beta=2.0),
             helper.make_node("Relu", ["A"], ["H"]),
             helper.make_node("Gemm", ["H", "W2"], ["Y"])]
    inits = {"W1": W1, "B1": B1.reshape(1, 3) if bias_2d else B1, "W2": W2}
    path = save_onnx(tmp_path, nodes, [1, _N_IN], [1, 2], inits,
                     name="gemm2d" if bias_2d else "gemm")
    exact = (W1.astype(np.float64) * 0.5, B1.astype(np.float64) * 2.0,
             W2.astype(np.float64).T)
    return path, exact


@pytest.mark.parametrize("kind,method,family", [
    ("Star", "exact", "row"), ("Star", "approx", "several"),
    ("Zono", "approx", "row"), ("Box", "approx", "several")])
def test_onnx_gemm_layers_hold_the_exact_image(tmp_path, kind, method, family):
    """ONNX Gemm with transB=1, alpha and beta, then Relu, then Gemm with
    transB=0 and no bias, loaded by ``load_onnx`` (both Gemm nodes become
    nn.Linear) and replayed on onnxruntime. Catches a Linear kernel that
    drops the bias or the scale the conversion folded into the weights."""
    path, (W1, b1, W2) = _gemm_model(tmp_path)
    model = load_onnx(path)
    if sum(isinstance(m, nn.Linear) for m in model.modules()) != 2:
        pytest.fail("the two Gemm nodes did not load as nn.Linear")
    sets, pieces, points = _family(kind, family)
    what = f"Gemm, Relu, Gemm on {kind} ({method}, {family})"
    result = reach_model(model, sets, method, what)
    exact = R.affine(R.relu(R.affine(pieces, W1.tolist(), b1.tolist())),
                     W2.tolist())
    _assert_holds_exact_image(result, exact, what, exact=(method == "exact"))
    assert_covered(result, replay_onnx(path, points, [1, _N_IN]), what)


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_onnx_gemm_with_a_two_dimensional_bias_refuses_or_is_sound(tmp_path,
                                                                    kind):
    """A Gemm whose bias C has shape (1, n) loads as an OnnxGemm module, not
    as nn.Linear. It must be refused (no kernel reads OnnxGemm) or answered
    soundly, never passed through as another layer."""
    path, (W1, b1, W2) = _gemm_model(tmp_path, bias_2d=True)
    sets, pieces, points = _family(kind, "one")
    what = f"Gemm with a (1, 3) bias on {kind}"
    try:
        result = reach_model(load_onnx(path), sets, "approx", what)
    except NotImplementedError:
        return
    exact = R.affine(R.relu(R.affine(pieces, W1.tolist(), b1.tolist())),
                     W2.tolist())
    _assert_holds_exact_image(result, exact, what)
    assert_covered(result, replay_onnx(path, points, [1, _N_IN]), what)


_GEMM_SCALES_ROUNDED = known_bug(
    23, "the loader folds a Gemm's alpha into the float32 weight and beta "
        "into the float32 bias and rounds the products (vendored onnx2torch, "
        "node_converters/gemm.py), so the reach set is that of another "
        "network: true outputs of the stored network lie outside it, by "
        "about 1e-9 here")


@pytest.mark.parametrize("alpha,beta", [
    pytest.param(0.5, 2.0, id="powers-of-two"),
    pytest.param(0.1, 0.3, id="tenths", marks=_GEMM_SCALES_ROUNDED)])
def test_onnx_gemm_scales_hold_the_exact_output_of_the_stored_network(
        tmp_path, alpha, beta):
    """ONNX Gemm Y = alpha (X B^T) + beta C with a float32 weight B and bias C
    of full mantissas, on a Box (whose affine map is rounded outward, so the
    reach adds no rounding of its own). The stored network is alpha and beta
    as the file stores them (float32) times the stored arrays, in fractions;
    the returned box must hold its exact image. With powers of two the folded
    products are exact; with tenths they are not. Catches a loader that
    rounds the folded scales, and a Box kernel that drops what was folded."""
    rng = np.random.default_rng(41)
    weight = rng.uniform(-1, 1, (3, _N_IN)).astype(np.float32)
    bias = rng.uniform(-1, 1, 3).astype(np.float32)
    node = helper.make_node("Gemm", ["X", "W", "C"], ["Y"], transB=1,
                            alpha=alpha, beta=beta)
    path = save_onnx(tmp_path, [node], [1, _N_IN], [1, 3],
                     {"W": weight, "C": bias}, name="gemm_scaled")
    model = load_onnx(path)
    if sum(isinstance(m, nn.Linear) for m in model.modules()) != 1:
        pytest.fail("the Gemm node did not load as nn.Linear")
    a, b = R.frac(np.float32(alpha)), R.frac(np.float32(beta))
    stored_W = [[a * R.frac(v) for v in row] for row in weight]
    stored_b = [b * R.frac(v) for v in bias]
    lo, hi = flat_box(_N_IN, seed=3, zero_width=True)
    what = f"Gemm with alpha {alpha} and beta {beta} on a Box"
    result = reach_model(model, Box(lo, hi), "approx", what)
    _assert_holds_exact_image(
        result, R.affine(R.box(lo.tolist(), hi.tolist()), stored_W, stored_b),
        what)


# --------------------------------------------------------------------------- #
# Shapes of the input set that only some routes give                            #
# --------------------------------------------------------------------------- #
_RANK3_LINEAR_ANSWERED = known_bug(
    12, "a Linear layer whose in_features is the size of the whole flat set "
        "but not the last axis of the tensor the set stands for "
        "(input_shape=(1, 2, 4), Linear(8, 3)) is answered with a 3-entry "
        "set, where torch cannot apply the layer to that tensor: no Linear "
        "path reads the tracked tensor shape", raises=DID_NOT_RAISE)


@pytest.mark.parametrize("kind,n_in", [
    ("Star", 4), ("Zono", 4), ("Box", 4), ("Hexatope", 4), ("Octatope", 4),
    pytest.param("Star", 8, marks=_RANK3_LINEAR_ANSWERED, id="Star-8")])
def test_linear_on_a_flat_set_of_a_rank_three_tensor_refuses_or_is_sound(kind,
                                                                         n_in):
    """A flat set of 8 entries that stands for a (1, 2, 4) tensor
    (``input_shape=(1, 2, 4)``), through the torch model entry; the Star has
    a binding row that is not a box row. torch's Linear maps the last axis:
    Linear(4, 3) is legal and gives a (1, 2, 3) tensor; Linear(8, 3) raises.
    Where torch raises the library must refuse (``NotImplementedError`` or
    ``ValueError``); where torch answers the library may refuse, or must hold
    every torch output read row-major. Each set type's own affine map meets
    the legal layer (today each refuses it: the weight has 4 columns, the
    set 8 entries). Catches an answer for a network that does not run, and,
    for the legal layer, a map of the 8 entries as one vector or of the
    first 4 only (a weight padded with zero columns). The illegal layer is
    marked once, on the Star: no lane reads the tracked shape."""
    model = nn.Sequential(_dyadic_linear(n_in, 3, True, seed=61)).eval()
    lo, hi = flat_box(8, seed=6, zero_width=True)
    if kind == "Star":
        input_set, points = constrained_star(lo, hi, seed=6)
    else:
        input_set, points = _build(kind, *_box_generators(lo, hi)), samples(lo, hi)
    what = f"Linear({n_in}, 3) on a {kind} of a (1, 2, 4) tensor"
    try:
        outputs = replay_torch(model, points, (1, 2, 4))
    except RuntimeError:
        outputs = None
    if outputs is None:
        with pytest.raises(_REFUSALS):
            reach_model(model, input_set, "approx", what, input_shape=(1, 2, 4))
        return
    try:
        result = reach_model(model, input_set, "approx", what, input_shape=(1, 2, 4))
    except _REFUSALS:
        return
    assert_covered(result, outputs, what)


@pytest.mark.parametrize("kind,method,n_out", [("ImageStar", "exact", 4),
                                               ("ImageZono", "approx", 3)])
def test_linear_on_a_one_pixel_high_image_refuses_or_holds_every_output(
        kind, method, n_out):
    """A (C, H, W) = (1, 1, 4) image: torch's Linear maps the last axis,
    which is here the whole image, so Linear(4, n) is legal and gives the
    (1, 1, 1, n) tensor. This is the one image shape where the library
    answers a legal Linear (on larger images it refuses the legal spelling
    and answers the illegal one, registry AFFINE-5): ``ImageStar.affine_map``
    maps the flattened star with the full matrix, and an ImageZono takes the
    inherited ``Zono.affine_map`` and comes back as a flat Zono. A refusal
    (``NotImplementedError`` or ``ValueError``, which the planned refusal of
    Linear on every image set gives) keeps the promise; an answer must hold
    the exact image (for 'exact', be it) and every torch output, an image
    set storing height 1, width n and one channel. Catches the full-matrix
    ImageStar map dropping the bias or the weights, and the Zono map putting
    the bias on the generators."""
    layer = _dyadic_linear(_N_IN, n_out, True, seed=71)
    model = nn.Sequential(layer).eval()
    flat, pieces, points = _family("Star" if kind == "ImageStar" else "Zono",
                                   "row")
    if kind == "ImageStar":
        sets = [ImageStar(s.V, s.C, s.d, s.predicate_lb, s.predicate_ub,
                          1, _N_IN, 1) for s in flat]
    else:
        sets = [ImageZono(s.c, s.V, 1, _N_IN, 1) for s in flat]
    what = f"Linear(4, {n_out}) on a 1 x 1 x 4 {kind} ({method})"
    try:
        result = reach_model(model, sets, method, what)
    except _REFUSALS:
        return
    W, b = _affine_of(layer)
    _assert_holds_exact_image(result, R.affine(pieces, W, b), what,
                              exact=(method == "exact"))
    assert_covered(result, replay_torch(model, points, (1, 1, 1, _N_IN)),
                   what, out_chw=(1, 1, n_out))


# --------------------------------------------------------------------------- #
# The exact method on a Box                                                     #
# --------------------------------------------------------------------------- #
_EXACT_BOX_IS_ITS_BOUNDING_BOX = known_bug(
    18, "reach_layer with method='exact' on a Box does not read the method "
        "and returns the outward bounding box of the image, which is larger "
        "than the image of a Linear layer that mixes coordinates; the model "
        "entry refuses 'exact' on a Box before any layer")

# Boxes whose ends are all -1, 0 or 1: with dyadic weights every product is
# exact, and so is every bound of the Box rule on a row with at most one
# nonzero weight (a single rounding, corrected by TwoSum).
_UNIT_BOXES = ((np.array([0.0, -1.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0, 1.0])),
               (np.array([0.0, -1.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0, 1.0])),
               (np.array([-1.0, 0.0, -1.0, -1.0]), np.array([0.0, 1.0, 1.0, 0.0])))


@pytest.mark.parametrize("weights", [
    "per-coordinate",
    pytest.param("mixing", marks=_EXACT_BOX_IS_ITS_BOUNDING_BOX)])
def test_exact_linear_on_a_box_refuses_or_is_the_image(weights):
    """DESIGN section 2, rule 4: 'exact' is exact up to float64 rounding or
    raises. ``reach_layer(nn.Linear, boxes, 'exact')``: a refusal
    (``NotImplementedError`` or ``ValueError``, as the model entry gives)
    keeps the promise; an answer must be the image, in both directions. A
    Box is the image only of a map that does not mix coordinates: a diagonal
    layer with dyadic weights on boxes with ends -1, 0 and 1 (every bound
    exact) is answered exactly today; a layer that mixes coordinates gets the
    bounding box of a parallelotope. Catches the exact method returning an
    over-approximation under its name."""
    if weights == "per-coordinate":
        layer = nn.Linear(_N_IN, _N_IN)
        with torch.no_grad():
            layer.weight.copy_(torch.diag(torch.tensor([0.75, -1.25, 0.5, -2.0])))
            layer.bias.copy_(torch.tensor([0.125, -0.5, 1.0, 0.25]))
    else:
        layer = _dyadic_linear(_N_IN, _N_IN, True, seed=81)
    sets = [Box(lo, hi) for lo, hi in _UNIT_BOXES]
    pieces = [p for lo, hi in _UNIT_BOXES for p in R.box(lo.tolist(), hi.tolist())]
    what = f"exact Linear ({weights}) on three Boxes"
    try:
        result = reach_one_layer(layer, sets, "exact", what)
    except _REFUSALS:
        return
    W, b = _affine_of(layer)
    _assert_holds_exact_image(result, R.affine(pieces, W, b), what, exact=True)


@pytest.mark.parametrize("kind", ["Zono", "ImageZono", "Box", "Hexatope",
                                  "Octatope"])
def test_exact_linear_model_on_a_set_with_no_exact_method_is_refused(kind):
    """The rows (exact, Zono / ImageZono / Box / Hexatope / Octatope) of
    Linear: the model entry refuses ``method='exact'`` on these set types
    before any layer (today ``ValueError``, '... only supports approx'; the
    planned ``UnsupportedError`` subclasses ``NotImplementedError``). Catches
    the model entry passing 'exact' through to a lane that does not read it,
    which answers the approx result under the name exact (for a Box the
    bounding box, test above)."""
    model = nn.Sequential(_layer("wide")).eval()
    lo, hi = flat_box(_N_IN, seed=0, zero_width=True)
    if kind == "ImageZono":
        input_set = ImageZono.from_bounds(lo.reshape(1, _N_IN, 1),
                                          hi.reshape(1, _N_IN, 1), 1, _N_IN, 1)
    else:
        c, G = _box_generators(lo, hi)
        input_set = _build(kind, c, G)
    with pytest.raises(_REFUSALS):
        reach_model(model, input_set, "exact", f"exact Linear on {kind}")


# --------------------------------------------------------------------------- #
# The outward rounding of a Linear layer on a Box, and its absence elsewhere    #
# --------------------------------------------------------------------------- #
def _non_dyadic_boxes(seed):
    """Three Boxes of four entries whose bounds are float64 numbers with full
    mantissas (thirds, sevenths, random fractions): a box cut along its
    first coordinate, the shape of a split."""
    rng = np.random.default_rng(seed)
    lo = rng.uniform(-1.0, 1.0, _N_IN) / 3.0
    hi = lo + rng.uniform(0.1, 1.0, _N_IN) / 7.0
    cuts = lo[0] + (hi[0] - lo[0]) * np.array([0.0, 1.0 / 3.0, 0.6, 1.0])
    boxes = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        part_lo, part_hi = lo.copy(), hi.copy()
        part_lo[0], part_hi[0] = a, b
        boxes.append((part_lo, part_hi))
    return boxes


def _non_dyadic_layer(seed):
    """Linear(4, 3) whose weight and bias are float32 numbers with full
    mantissas."""
    rng = np.random.default_rng(100 + seed)
    layer = nn.Linear(_N_IN, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(rng.uniform(-1, 1, (3, _N_IN))))
        layer.bias.copy_(torch.from_numpy(rng.uniform(-1, 1, 3)))
    return layer


@pytest.mark.parametrize("seed", [0])
def test_linear_on_boxes_holds_the_exact_image_of_non_dyadic_data(seed):
    """``Box.affine_map`` promises bounds rounded outward, so Linear on a
    Box must hold the exact image of the box under the stored weights and
    bias even when every product and sum rounds. Weights and bias are
    float32 numbers with full mantissas, bounds are float64 thirds and
    sevenths; the reference is the image in rationals of exactly those
    numbers, with no slack. Catches a Box kernel that rounds to nearest (a
    centre and radius form, or bounds summed without the outward move, or
    moved the same way on both sides), which leaves about half of the bounds
    one unit inside."""
    layer = _non_dyadic_layer(seed)
    W, b = _affine_of(layer)
    parts = _non_dyadic_boxes(seed)
    sets = [Box(lo, hi) for lo, hi in parts]
    pieces = [p for lo, hi in parts for p in R.box(lo.tolist(), hi.tolist())]
    what = f"Linear on 3 Boxes with non-dyadic data (seed {seed})"
    result = reach_one_layer(layer, sets, "approx", what)
    _assert_holds_exact_image(result, R.affine(pieces, W, b), what)
    points = [x for lo, hi in parts for x in samples(lo, hi, n=43)]
    assert_covered(result, replay_torch(layer, points, (1, _N_IN)), what)


def _sparse_rows_layer():
    """Linear(4, 8) in float64 whose rows have no, one, two and four nonzero
    weights with full mantissas (thirds of random numbers), and a bias with
    full mantissas (sevenths)."""
    rng = np.random.default_rng(51)
    weight = np.zeros((8, _N_IN))
    for i, cols in enumerate(([], [0], [2], [3], [0, 1], [1, 2], [2, 3],
                              [0, 1, 2, 3])):
        weight[i, cols] = rng.uniform(-1, 1, len(cols)) / 3.0
    layer = nn.Linear(_N_IN, 8, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight))
        layer.bias.copy_(torch.from_numpy(rng.uniform(-1, 1, 8) / 7.0))
    return layer


def test_linear_on_boxes_with_unit_ends_holds_the_exact_image():
    """The branches of the outward Box rule that the cell above does not
    reach (``outward_interval_bounds``): box ends of -1, 0 and 1 make every
    product exact, so a row with one nonzero weight is one rounding (the
    bias plus the product), which the rule corrects exactly by TwoSum and
    moves to the next float only when the error points inward; a row whose
    products are all zero is the bias itself; a row with two nonzero weights
    has two roundings and must take the general bound. Reference: the image
    in rationals, no slack, plus 129 replayed points. Catches the single-term
    correction moved on the wrong side and two-term rows taken for single
    terms (both leave bounds one unit inside)."""
    layer = _sparse_rows_layer()
    W, b = _affine_of(layer)
    sets = [Box(lo, hi) for lo, hi in _UNIT_BOXES]
    pieces = [p for lo, hi in _UNIT_BOXES for p in R.box(lo.tolist(), hi.tolist())]
    what = "Linear with sparse rows on Boxes with ends -1, 0 and 1"
    result = reach_one_layer(layer, sets, "approx", what)
    _assert_holds_exact_image(result, R.affine(pieces, W, b), what)
    points = [x for lo, hi in _UNIT_BOXES for x in samples(lo, hi, n=43)]
    assert_covered(result, replay_torch(layer, points, (1, _N_IN)), what)


def test_linear_on_a_box_with_an_infinite_end_keeps_that_side_unbounded():
    """A Box may have infinite ends (``Box`` accepts them and the Box rule
    handles them): x1 in [0, inf) and x2 in (-inf, 1/2]. An output with a
    nonzero weight on such an end is unbounded on that side and must be
    returned so; an output whose weights on those coordinates are zero has a
    finite range, which the returned bounds must hold (a zero weight times an
    infinite end is no term, not NaN). Reference: each output's range in
    fractions, by the end values of each term. Catches a rule that zeroes an
    infinite end and forgets it, which returns a finite bound where true
    outputs grow without limit."""
    lo = np.array([-0.5, 0.0, -np.inf, 0.25])
    hi = np.array([0.75, np.inf, 0.5, 0.25])
    weight = [[1.0, 2.0, 0.0, -1.0], [0.5, 0.0, -1.0, 0.0],
              [1.0, 0.0, 0.0, 0.5], [0.0, -0.25, 0.5, 0.0]]
    bias = [0.25, -0.5, 0.125, 1.0]
    layer = nn.Linear(_N_IN, 4)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight))
        layer.bias.copy_(torch.tensor(bias))
    what = "Linear on a Box with infinite ends"
    (result,) = reach_one_layer(layer, [Box(lo, hi)], "approx", what)
    got_lo, got_hi = result.lb.reshape(-1), result.ub.reshape(-1)
    for i, (row, b) in enumerate(zip(weight, bias)):
        ends = [sorted((w * l, w * h)) if w else (0.0, 0.0)
                for w, l, h in zip(row, lo, hi)]
        low = sum(e[0] for e in ends) + b
        high = sum(e[1] for e in ends) + b
        if np.isinf(low):
            assert got_lo[i] == -np.inf, (
                f"{what}: output {i} is unbounded below, returned {got_lo[i]}")
        else:
            assert R.frac(got_lo[i]) <= sum(R.frac(e[0]) for e in ends) + R.frac(b), (
                f"{what}: output {i} lower bound {got_lo[i]} above {low}")
        if np.isinf(high):
            assert got_hi[i] == np.inf, (
                f"{what}: output {i} is unbounded above, returned {got_hi[i]}")
        else:
            assert R.frac(got_hi[i]) >= sum(R.frac(e[1]) for e in ends) + R.frac(b), (
                f"{what}: output {i} upper bound {got_hi[i]} below {high}")


_HEXOCT_AFFINE_ROUNDED = known_bug(
    9, "the affine map of a Hexatope and of an Octatope is computed in "
       "float64 rounded to nearest with no error term (Hexatope.affine_map, "
       "Octatope.affine_map), so the set returned for a Linear layer leaves "
       "out vertices of the exact image of the stored set under the stored "
       "weights by units in the last place; DESIGN section 1 narrows the "
       "claim for the Star and the Zono only")


@pytest.mark.parametrize("kind", [
    pytest.param("Hexatope", marks=_HEXOCT_AFFINE_ROUNDED),
    pytest.param("Octatope", marks=_HEXOCT_AFFINE_ROUNDED)])
def test_linear_on_hexatopes_and_octatopes_holds_the_exact_image_of_non_dyadic_data(kind):
    """The cell above for a Hexatope and an Octatope: the three parts of the
    non-dyadic split, each the stored set {c + G a} with the centre and
    half-widths as float64 computes them (full mantissas), under the same
    layer, against the image in rationals of exactly those numbers, with no
    slack. Its passing neighbour is
    ``test_linear_on_boxes_holds_the_exact_image_of_non_dyadic_data[0]``
    (same data, layer, reference and check, on a Box, whose rule is rounded
    outward). Catches an affine map rounded to nearest; the Star and the
    Zono do the same and are marked once, for PLAN step 7b, in
    ``tests/contracts/test_verdicts.py``."""
    layer = _non_dyadic_layer(0)
    W, b = _affine_of(layer)
    parts = [_box_generators(lo, hi) for lo, hi in _non_dyadic_boxes(0)]
    sets = [_build(kind, c, G) for c, G in parts]
    pieces = [p for c, G in parts for p in _exact_input(c, G)]
    what = f"Linear on 3 {kind}s with non-dyadic data"
    result = reach_one_layer(layer, sets, "approx", what)
    _assert_holds_exact_image(result, R.affine(pieces, W, b), what)

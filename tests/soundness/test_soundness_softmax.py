"""Soundness cells for Softmax: the torch layer and the graph operation.

What the file covers. Softmax reaches n2v in two ways.

- The torch layer ``nn.Softmax`` through the dispatcher (``reach_layer``)
  runs ``softmax_star`` or ``softmax_box`` (``n2v/nn/layer_ops/
  softmax_reach.py``): one softmax over the whole flat vector, for ``dim``
  -1, 1 or None; another axis is refused, and so are Zono, Hexatope and
  Octatope. Cells: Star, Box and a Star with a binding row that is not a box
  row; one input set and three at once; every accepted ``dim``; logits far
  apart (the box times 8) and a Star with more predicate variables than
  entries; the LP solver of the range LPs on a badly scaled Star; an
  ImageStar of one pixel (whose stored order is the channel order); Linear,
  Softmax, Linear as one ``nn.Sequential``; the refusals, among them
  ``method='exact'`` on a Box (strict expected failure).
- A Softmax node of a model (ONNX Softmax of opset 13, ONNX Softmax of
  opset 11, ``nn.Softmax`` in a traced torch model) on a flat set whose
  tensor shape is known goes to ``_handle_softmax_flat`` in
  ``n2v/nn/reach.py``. It bounds a softmax over one axis (opset 13 and
  torch) or over the flattened tail from the axis on (opset 11 and earlier)
  with ``softmax_attn_star``, ``softmax_attn_zono`` or ``softmax_attn_box``.
  Cells: shapes (1, 6), (1, 2, 3), (1, 3, 2) and (1, 2, 3, 2); the last
  axis, an earlier one, the batch axis, negative axes and the default axis
  of each opset (every case on a Star, the cases that reach another line of
  the handler on a Zono and a Box); logits far apart, a Zono with more
  generators than entries and a Star with more predicate variables than
  entries; the chain Gemm, Reshape, Softmax, Reshape, Gemm through the model
  entry for ONNX and torch models; ``nn.Softmax()`` with no ``dim`` in a
  torch model; the several pieces of an exact ReLU split; an image set
  flattened and grouped per channel or per pixel before the Softmax; an
  opset-11 Softmax on an image set (refused); the routes on which the walk
  has a wrong tensor shape or none (strict expected failures); functional
  softmax in a torch model (refused); the refusals of Hexatope, Octatope,
  LogSoftmax and of ``method='exact'`` on the set types without an exact
  method.

Strict expected failures of today's library marked here: ``nn.Softmax()``
with no ``dim`` in a traced model is read as the last axis, where torch takes
axis 0 of a rank-3 tensor and axis 1 of a rank-4 one (registry row ACT-6);
a flat set without ``input_shape`` on a
rank-3 model input is read as the tensor (1, n) (registry SHAPE-2); a model
whose shape record is lost sends the Softmax to the generic one-softmax rule
(registry SHAPE-8); ``reach_layer`` under ``method='exact'`` on a Box
answers with a box, which is not the image (registry REACH-8 and
REACH-4).

Not here: Softmax on an image set of more than one pixel at opset 13, and
the exact method on a Star (``tests/contracts/test_reach_coverage.py`` marks
both: ACT-1, ACT-2, REACH-4); the attention kernel called directly
(``test_soundness_softmax_attention.py``); the rounding of the bounds at one
unit in the last place (registry row ROUND-6, marked in
``tests/contracts/test_smooth_rounding.py`` and
``test_soundness_range_rounding.py``); a model input without a batch axis
(registry row SHAPE-1, ``test_soundness_tensor_shape.py``); the outward pad
of the range LPs of ``softmax_star`` (no
point and slack of these cells can see a pad of 1e-7, a rounding-size check
of the ROUND-6 family once ROUND-6 is fixed).

Oracle. True outputs are float64 replays that never touch n2v: torch in
float64 for the layer cells, onnxruntime on float64 graphs for the model
cells. Inputs and weights are dyadic. Membership is the LP oracle against the
rows of the result (``assert_covered``, slack ``LP_TOL``). The member points
include, for every output entry, the two corners of the input box at which
the entry takes its largest and its smallest value over the box: softmax
entry i rises with its own logit and falls with every other logit of its
group, so its largest value is at ``x_i = hi_i`` with every other logit at
its lower end, and its smallest value at the opposite corner. Every kernel
here returns one box per input set, computed from the bounding box of its
input, so on a box input these corners decide containment as completely as
a proof would, up to the slack. No interval proof is used: the bounds are the
extreme values rounded to nearest (registry row ROUND-6), so a proof with
zero slack fails at one unit in the last place on a defect another file
marks; ``LP_TOL`` (1e-7 against outputs in [0, 1]) is far below what a
wrong axis, a wrong bound or a dropped piece produces. The proof at every
vertex and near every corner of the input is in
``tests/contracts/test_smooth_rounding.py``.

Layout. A flat set holds the tensor in its row-major order. The image cell
flattens a (2, 3, 5) image first, so the Softmax groups entries that are far
apart in the stored (height, width, channel) order of the input set.
"""

import itertools

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper
from scipy.optimize import linprog

from n2v.sets import Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.utils.model_loader import load_onnx
from tests.cell_support import (assert_covered, constrained_image_star,
                                constrained_star, flat_set,
                                image_set, image_split_cover, pixel_box,
                                reach_model, reach_one_layer, replay_onnx,
                                replay_torch, samples, save_onnx, split_cover)
from tests.known_bugs import DID_NOT_RAISE, known_bug

# --------------------------------------------------------------------------- #
# Inputs                                                                        #
# --------------------------------------------------------------------------- #
# Six logits with their own widths (1/2 to 2), one of them fixed (index 2), so
# a bound that takes the wrong end of another logit, or of its own, is off.
_LO = np.array([-1.5, 0.25, -0.5, 1.0, -2.0, 0.5])
_HI = np.array([0.5, 1.75, -0.5, 2.0, -0.75, 2.5])
_N = _LO.size

# Twelve logits for the rank-4 shape.
_LO12 = np.concatenate([_LO, _LO[::-1] - 0.5])
_HI12 = np.concatenate([_HI, _HI[::-1] - 0.5])


def _logits(shape):
    """The logit box of a tensor of ``shape`` (six or twelve entries)."""
    n = int(np.prod(shape))
    return (_LO, _HI) if n == _N else (_LO12, _HI12)


def _extremes(lo, hi):
    """For every entry i, the corner with ``x_i`` at its upper end and every
    other entry at its lower end, and the opposite corner: where softmax
    entry i takes its largest and its smallest value over the box, whatever
    group i belongs to."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    points = []
    for i in range(lo.size):
        up, down = lo.copy(), hi.copy()
        up[i], down[i] = hi[i], lo[i]
        points += [up, down]
    return points


def _points(lo, hi, n=24):
    """Corners, quarter-grid points and the extreme corners of the box."""
    return samples(lo, hi, n=n) + _extremes(lo, hi)


def _binding_row_cover(lo, hi, parts):
    """``(stars, points)``: ``parts`` Stars, each with a binding row that is
    not a box row (``constrained_star``), whose boxes cut [lo, hi] along its
    widest coordinate at the dyadic shares 0, 1/8, 3/8, 1 (one part: the
    whole box). ``points`` are member points of the stars."""
    if parts == 1:
        star, points = constrained_star(lo, hi, n=32)
        return [star], points
    if parts != 3:
        raise ValueError("one or three parts")
    j = int(np.argmax(hi - lo))
    ends = lo[j] + (hi[j] - lo[j]) * np.array([0.0, 1.0, 3.0, 8.0]) / 8.0
    stars, points = [], []
    for k in range(3):
        part_lo, part_hi = lo.copy(), hi.copy()
        part_lo[j], part_hi[j] = ends[k], ends[k + 1]
        star, pts = constrained_star(part_lo, part_hi, n=16, seed=k)
        stars.append(star)
        points += pts
    return stars, points


def _more_variables_than_entries(kind, lo, hi, extra=3, n=64, seed=7):
    """``(set, points)``: the logits ``c + G a`` with ``a`` in [-1, 1]^k,
    ``k = len(lo) + extra``, ``G = [diag((hi - lo) / 4), K]`` with ``K`` a
    dyadic matrix of ``extra`` columns (entries -1/2 to 1/2): the shape a
    relaxation leaves, more variables than entries. A ``Star`` is cut by
    the row ``a_0 + a_k-1 <= 1/2``, which ties the first entry's variable to
    a variable of no entry of its own; a ``Zono`` has no row. ``points`` are
    member points (corners and half-grid points of ``a`` that keep the row),
    dyadic, computed without n2v."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    size = lo.size
    k = size + extra
    rng = np.random.default_rng(seed)
    c = (lo + hi) / 2.0
    K = rng.integers(-2, 3, size=(size, extra)) / 4.0
    G = np.hstack([np.diag((hi - lo) / 4.0), K])
    row = np.zeros(k)
    row[0], row[-1] = 1.0, 1.0
    points = []
    for attempt in range(50 * n):
        if len(points) >= n:
            break
        a = (rng.choice([-1.0, 1.0], size=k) if attempt % 2
             else rng.integers(-2, 3, size=k) / 2.0)
        if kind == "Star" and row @ a > 0.5:
            continue
        points.append(c + G @ a)
    if kind == "Zono":
        return Zono(c.reshape(-1, 1), G), points
    if kind != "Star":
        raise ValueError(kind)
    C = np.vstack([np.eye(k), -np.eye(k), row[None, :]])
    d = np.concatenate([np.ones(2 * k), [0.5]]).reshape(-1, 1)
    star = Star(np.hstack([c[:, None], G]), C, d,
                -np.ones((k, 1)), np.ones((k, 1)))
    return star, points


def _flat_inputs(kind, lo, hi, parts=1):
    """``(input sets, member points)``: the box [lo, hi] as ``parts`` sets of
    ``kind``, or ``parts`` Stars with a binding row; the points of a box
    kind include the extreme corners of every output entry."""
    if kind == "Star with a binding row":
        return _binding_row_cover(lo, hi, parts)
    sets = ([flat_set(kind, lo, hi)] if parts == 1
            else split_cover(kind, lo, hi, parts))
    return sets, _points(lo, hi)


def _other_input(kind, shape, lo, hi):
    """``(input sets, member points)`` of the input shapes the box cells do
    not have: ``"box times 8"`` (the logit box scaled by 8, so bounds go far
    beyond plus and minus 1 and the logits spread up to 36) and ``"more
    variables than entries"``."""
    if shape == "box times 8":
        return _flat_inputs(kind, 8.0 * lo, 8.0 * hi)
    if shape == "more variables than entries":
        input_set, points = _more_variables_than_entries(kind, lo, hi)
        return [input_set], points
    raise ValueError(shape)


# --------------------------------------------------------------------------- #
# The torch layer through the dispatcher                                        #
# --------------------------------------------------------------------------- #
# (kind, parts, dim): every accepted dim on one set of each kernel, the loop
# over three sets with dim -1, and the Star with a binding row once.
_LAYER_CELLS = [
    (kind, parts, dim)
    for kind, parts, dims in (("Star", 1, (-1, 1, None)),
                              ("Star", 3, (-1,)),
                              ("Box", 1, (-1, 1, None)),
                              ("Box", 3, (-1,)),
                              ("Star with a binding row", 1, (-1,)))
    for dim in dims]


@pytest.mark.parametrize("kind,parts,dim", [
    pytest.param(kind, parts, dim,
                 id=f"{kind}-{'one set' if parts == 1 else 'three sets'}"
                    f"-{dim}")
    for kind, parts, dim in _LAYER_CELLS])
def test_softmax_layer_holds_every_output(kind, parts, dim):
    """``nn.Softmax`` on a flat vector of six logits through ``reach_layer``,
    with one input set or three at once (pieces of distinct sizes, the
    smallest first). Catches a bound that takes the wrong end of a logit
    (the lower input bound used for the upper one, the own logit's upper end
    in its lower bound, the other logits' upper ends in an upper bound), an
    entry left out of the sum of the others or counted in it, a fixed logit
    mishandled, ranges read from the wrong coordinate, a loop over the input
    sets that stops after the first or reuses its bounds, and a refusal of
    an axis the layer accepts. ``dim=None`` is the feature axis of a (1, N)
    tensor, the axis torch itself picks for two dimensions. The dim is read
    only by the axis check, so the loop and the Star with a binding row run
    with dim -1."""
    sets, points = _flat_inputs(kind, _LO, _HI, parts)
    layer = nn.Softmax(dim=dim)
    what = f"Softmax(dim={dim}) on {parts} {kind}"
    out = reach_one_layer(layer, sets, "approx", what)
    reference = nn.Softmax(dim=1 if dim is None else dim)
    assert_covered(out, replay_torch(reference, points, [1, _N]), what)


def _scaled_star(seed=93, scale=1e4):
    """A valid, badly scaled Star of six logits: generators of size about
    ``scale`` over a predicate box of half-width ``0.09 / scale`` cut by
    three rows, every logit range under 1.3 wide. Returns ``(star, points)``:
    member points computed with scipy alone, the two LP extremes of every
    coordinate pulled by one part in a million towards the interior point
    ``a = 0``, and random members of the predicate polytope."""
    rng = np.random.default_rng(seed)
    G = scale * rng.normal(size=(_N, _N))
    c = rng.uniform(-1.0, 1.0, size=_N)
    w = 0.09 / scale
    H = rng.normal(size=(3, _N))
    C = np.vstack([np.eye(_N), -np.eye(_N), H])
    d = np.concatenate([w * np.ones(2 * _N), 0.3 * w * np.ones(3)])
    star = Star(np.hstack([c[:, None], G]), C, d[:, None],
                -w * np.ones((_N, 1)), w * np.ones((_N, 1)))
    points = [c.copy()]
    for i in range(_N):
        for sign in (1.0, -1.0):
            res = linprog(sign * G[i], A_ub=C, b_ub=d,
                          bounds=[(-w, w)] * _N, method="highs")
            if res.status != 0:
                raise RuntimeError(f"extreme point LP: {res.message}")
            a = res.x * (1.0 - 1e-6)
            if np.any(C @ a > d):
                raise RuntimeError("the pulled extreme point left the set")
            points.append(c + G @ a)
    while len(points) < 48:
        a = rng.uniform(-w, w, size=_N)
        if np.all(C @ a <= d):
            points.append(c + G @ a)
    return star, points


@pytest.mark.parametrize("lp_solver", [None, "linprog", "SCS"])
def test_softmax_layer_on_a_badly_scaled_star_under_each_lp_solver(lp_solver):
    """The range LPs of ``softmax_star`` are bounds: they must run on HiGHS
    whatever ``lp_solver`` asks for. On this star the first-order solver SCS
    misses the ends of the logit ranges by up to 0.5 (measured against
    scipy's HiGHS), so a kernel that hands the caller's solver to its range
    LPs loses the outputs of the extreme points the replay includes. The
    outward pad of those ranges is not seen here: the extreme points are
    pulled inward by one part in a million, more than the pad."""
    star, points = _scaled_star()
    options = {} if lp_solver is None else {"lp_solver": lp_solver}
    what = f"Softmax on a badly scaled Star, lp_solver={lp_solver}"
    out = reach_one_layer(nn.Softmax(dim=-1), [star], "approx", what,
                          **options)
    assert_covered(out, replay_torch(nn.Softmax(dim=-1), points, [1, _N]),
                   what)


@pytest.mark.parametrize("parts", [1, 3], ids=["one set", "three sets"])
def test_softmax_layer_on_a_one_pixel_image_star(parts):
    """An ImageStar of one pixel and four channels: its stored (height,
    width, channel) order is the channel order, so the softmax over the
    channels (``dim=1`` of a (1, 4, 1, 1) tensor) is one softmax over the
    stored vector, which the kernel computes after ``to_star``. Catches a
    conversion of the ImageStar that loses or reorders entries, and a loop
    over several ImageStars that stops after the first. The neighbour of the
    marked cells for larger images in ``test_reach_coverage.py``."""
    lo, hi = pixel_box(4, 1, 1, seed=3)
    sets = ([image_set("ImageStar", lo, hi)] if parts == 1
            else image_split_cover("ImageStar", lo, hi, parts))
    what = f"Softmax(dim=1) on {parts} one-pixel ImageStar"
    out = reach_one_layer(nn.Softmax(dim=1), sets, "approx", what)
    points = _points(lo.reshape(-1), hi.reshape(-1))
    assert_covered(out, replay_torch(nn.Softmax(dim=1), points, [1, 4, 1, 1]),
                   what)


# Linear (3 -> 6), Softmax, Linear (6 -> 6, invertible, so it hides nothing).
# The six logits share the three input predicate variables.
_IN_LO, _IN_HI = np.array([-1.0, -0.5, 0.25]), np.array([0.5, 1.0, 1.25])
_W1 = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, -0.5], [0.5, 0.0, 1.0],
                [-1.0, 0.25, 0.0], [0.25, 0.25, 0.25], [1.0, -1.0, 0.5]])
_B1 = np.array([0.0, 0.25, 0.5, -0.5, 1.0, 0.0])
_W2 = np.eye(_N) + 0.5 * np.roll(np.eye(_N), 1, axis=1)
_B2 = np.array([0.0, 0.5, -0.25, 0.0, 1.0, -0.5])


def _grid(lo, hi, steps=4):
    """Every point of the grid with ``steps`` equal steps per axis of the box
    [lo, hi], corners included; dyadic when the bounds are."""
    axes = [a + (b - a) * np.arange(steps + 1) / steps for a, b in zip(lo, hi)]
    return [np.array(x) for x in itertools.product(*axes)]


def _torch_chain():
    """Linear, Softmax, Linear in float64 with the weights above."""
    model = nn.Sequential(nn.Linear(3, _N), nn.Softmax(dim=-1),
                          nn.Linear(_N, _N)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(_W1))
        model[0].bias.copy_(torch.from_numpy(_B1))
        model[2].weight.copy_(torch.from_numpy(_W2))
        model[2].bias.copy_(torch.from_numpy(_B2))
    return model.eval()


# (kind, parts, entry). The model entry reaches each set of a list on its own,
# so three sets are run through the dispatcher only; Zono has no dispatcher
# rule (its refusal is test_softmax_layer_refuses_a_set_type_without_a_rule).
_CHAIN_LAYER_CELLS = [
    ("Star", 1, "reach_layer"), ("Star", 1, "model"),
    ("Star", 3, "reach_layer"),
    ("Box", 1, "reach_layer"), ("Box", 1, "model"), ("Box", 3, "reach_layer"),
    ("Zono", 1, "model"),
    ("Star with a binding row", 1, "reach_layer"),
    ("Star with a binding row", 1, "model"),
    ("Star with a binding row", 3, "reach_layer"),
]


@pytest.mark.parametrize("kind,parts,entry", [
    pytest.param(kind, parts, entry,
                 id=f"{kind}-{'one set' if parts == 1 else 'three sets'}"
                    f"-{entry}")
    for kind, parts, entry in _CHAIN_LAYER_CELLS])
def test_softmax_layer_between_two_linear_layers(kind, parts, entry):
    """Linear, Softmax, Linear as one torch model, through the dispatcher
    (which walks the ``nn.Sequential`` and runs ``softmax_star`` or
    ``softmax_box``) and through the model entry (which traces the model and
    sends the Softmax to ``_handle_softmax_flat``). The logits share the
    input's predicate variables, so the range LPs of ``softmax_star`` read
    the binding row, and the second Linear consumes the Softmax result.
    Catches a Softmax result that the next layer cannot hold the outputs in
    (a bound that uses the wrong logit, a box over the wrong entries), the
    sets of one node handed to another, and a loop over the input sets that
    stops after the first."""
    model = _torch_chain()
    if kind == "Star with a binding row":
        sets, points = _binding_row_cover(_IN_LO, _IN_HI, parts)
    else:
        sets = ([flat_set(kind, _IN_LO, _IN_HI)] if parts == 1
                else split_cover(kind, _IN_LO, _IN_HI, parts))
        points = samples(_IN_LO, _IN_HI, n=24) + _grid(_IN_LO, _IN_HI)
    what = f"Linear, Softmax, Linear on {parts} {kind} through {entry}"
    if entry == "reach_layer":
        out = reach_one_layer(model, sets, "approx", what)
    else:
        out = reach_model(model, sets, "approx", what)
    assert_covered(out, replay_torch(model, points, [1, 3]), what)


class _ImplicitAxisSoftmax(nn.Module):
    """Reshape the flat input to ``shape``, ``nn.Softmax()`` with no ``dim``,
    flatten. torch picks the axis itself (``F._get_softmax_dim``): axis 0
    for a tensor of rank 0, 1 or 3, axis 1 otherwise, with a deprecation
    warning."""

    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)
        self.softmax = nn.Softmax(dim=None)

    def forward(self, x):
        return torch.flatten(self.softmax(x.reshape(self.shape)), 1)


_IMPLICIT_AXIS_READ_AS_LAST = known_bug(
    7, "nn.Softmax with no dim in a traced model is bounded as a softmax over "
       "the last axis, while torch takes axis 0 of a rank-3 tensor and axis 1 "
       "of a rank-4 tensor; model outputs escape the result (every sampled "
       "one on the rank-3 tensor, 45 of 48 on the rank-4 one)")


@pytest.mark.filterwarnings("ignore:Implicit dimension choice:UserWarning")
@pytest.mark.parametrize("kind", ["Star"])
@pytest.mark.parametrize("shape", [
    (1, 6),
    pytest.param((1, 2, 3), marks=_IMPLICIT_AXIS_READ_AS_LAST),
    pytest.param((1, 2, 3, 2), marks=_IMPLICIT_AXIS_READ_AS_LAST),
], ids=lambda shape: "x".join(map(str, shape)))
def test_softmax_layer_with_the_implicit_axis_in_a_model(shape, kind):
    """``nn.Softmax()`` without ``dim`` inside a torch model, through the
    model entry: the handler must bound the softmax over the axis torch
    itself takes. On a (1, n) tensor that is the last axis; on (1, 2, 3) it
    is the batch axis (every output is 1) and on (1, 2, 3, 2) the axis of
    length 2 after the batch axis. Catches the implicit axis read as the last
    one, or as any fixed axis whatever the rank (axis 0 fails the (1, 6)
    case). The handler reads the axis before it picks the kernel of the set
    type, so a Star stands for every set type."""
    n = int(np.prod(shape))
    lo, hi = _logits(shape)
    model = _ImplicitAxisSoftmax(shape).double().eval()
    what = f"nn.Softmax() of a {shape} tensor on {kind}"
    out = reach_model(model, flat_set(kind, lo, hi), "approx", what)
    assert_covered(out, replay_torch(model, _points(lo, hi), [1, n]), what)


@pytest.mark.parametrize("dim,kind", [
    pytest.param(dim, kind, id=f"{dim}-{kind}")
    for dim, kind in ((0, "Star"), (2, "Star"), (-2, "Star"), (0, "Box"))])
def test_softmax_layer_refuses_another_axis(kind, dim):
    """The layer rule computes one softmax over the whole flat vector, which
    is the softmax of the feature axis of a (1, N) tensor only. A softmax
    over the batch axis or over an axis the flat vector does not have must
    be refused, not answered as a softmax over the wrong group. The Box
    kernel calls the axis check on its own (``softmax_box``)."""
    with pytest.raises(NotImplementedError):
        reach_one_layer(nn.Softmax(dim=dim), [flat_set(kind, _LO, _HI)],
                        "approx", f"Softmax(dim={dim}) on {kind}")


@pytest.mark.parametrize("kind", ["Zono", "Hexatope", "Octatope"])
def test_softmax_layer_refuses_a_set_type_without_a_rule(kind):
    """The dispatcher has no Softmax rule for these set types; it must raise
    rather than pass the set through or answer with another type's rule.
    The bounds are dyadic, so the Hexatope and Octatope are exact."""
    with pytest.raises(NotImplementedError):
        reach_one_layer(nn.Softmax(dim=-1), [flat_set(kind, _LO, _HI)],
                        "approx", f"Softmax on {kind}")


@pytest.mark.parametrize("kind", [
    pytest.param("Box", marks=known_bug(
        18, "reach_layer with method='exact' on a Box runs the Softmax "
            "interval rule and returns its box as the exact result; the "
            "image of a box under softmax is not a box (every output sums to "
            "1, the returned upper corner to 2.25), so the result holds "
            "points no input reaches (REACH-8, REACH-4)",
        raises=DID_NOT_RAISE)),
    "Zono",
])
def test_softmax_layer_under_exact_without_an_exact_rule_is_refused(
        kind):
    """``method='exact'`` through ``reach_layer`` on a set type with no exact
    Softmax rule must be refused, as the model entry refuses it before the
    walk (``test_exact_softmax_model_on_a_set_with_no_exact_method_is_refused``).
    The Box lane does not read the method and answers with the box of the
    approximate rule. Catches a lane that answers under the name exact; the
    Zono lane, which has no Softmax rule, is the passing neighbour."""
    with pytest.raises(NotImplementedError):
        reach_one_layer(nn.Softmax(dim=-1), [flat_set(kind, _LO, _HI)],
                        "exact", f"exact Softmax on {kind}")


class _FunctionalSoftmax(nn.Module):
    """The six logits viewed as (1, 2, 3), a softmax over axis 1 written as
    a function call or as a tensor method, flattened."""

    def __init__(self, spelling):
        super().__init__()
        self.spelling = spelling

    def forward(self, x):
        h = x.view(1, 2, 3)
        if self.spelling == "F.softmax":
            h = F.softmax(h, dim=1)
        elif self.spelling == "torch.softmax":
            h = torch.softmax(h, dim=1)
        else:
            h = h.softmax(1)
        return h.flatten(1)


@pytest.mark.parametrize("spelling,kind", [
    ("F.softmax", "Star"), ("torch.softmax", "Star"),
    ("Tensor.softmax", "Star"), ("F.softmax", "Zono"), ("F.softmax", "Box")])
def test_functional_softmax_in_a_torch_model_refuses_or_holds_every_output(
        spelling, kind):
    """A softmax written as ``F.softmax``, ``torch.softmax`` (function nodes
    of the traced graph) or ``Tensor.softmax`` (a method node) over axis 1
    of a (1, 2, 3) view, through the model entry. Today these nodes have no
    reach rule and the call is refused (``NotImplementedError``); an answer
    must hold every output. Catches a functional softmax mapped to a
    ``nn.Softmax`` without its ``dim``, which the generic rule bounds as one
    softmax over all six entries."""
    model = _FunctionalSoftmax(spelling).double().eval()
    what = f"{spelling}(dim=1) of a (1, 2, 3) view on {kind}"
    try:
        out = reach_model(model, flat_set(kind, _LO, _HI), "approx", what)
    except NotImplementedError:
        return
    assert_covered(out, replay_torch(model, _points(_LO, _HI), [1, _N]), what)


@pytest.mark.parametrize("kind", ["Zono", "ImageZono", "Box", "Hexatope",
                                  "Octatope"])
def test_exact_softmax_model_on_a_set_with_no_exact_method_is_refused(kind):
    """The rows (exact, Zono / ImageZono / Box / Hexatope / Octatope) of
    Softmax: the model entry refuses ``method='exact'`` on these set types
    before any layer (today ``ValueError``, '... only supports approx'; the
    planned ``UnsupportedError`` subclasses ``NotImplementedError``). Catches
    the model entry passing 'exact' through to the walk, where the Zono and
    Box branches of the handler would answer the approximate bound under the
    name exact. The ImageZono (six channels of one pixel) is flattened
    first, so that under approx the handler answers it."""
    if kind == "ImageZono":
        lo, hi = pixel_box(_N, 1, 1, seed=5)
        model = nn.Sequential(nn.Flatten(), nn.Softmax(dim=-1)).eval()
        input_set = image_set(kind, lo, hi)
    else:
        model = nn.Sequential(nn.Softmax(dim=-1)).eval()
        input_set = flat_set(kind, _LO, _HI)
    with pytest.raises((NotImplementedError, ValueError)):
        reach_model(model, input_set, "exact", f"exact Softmax on {kind}")


# --------------------------------------------------------------------------- #
# The graph operation: _handle_softmax_flat                                     #
# --------------------------------------------------------------------------- #
def _softmax_graph(tmp_path, shape, axis, opset, op="Softmax",
                   name="softmax"):
    """``Y = op(X)`` on a float64 tensor of ``shape``; no axis attribute when
    ``axis`` is None (the default of the opset: -1 for 13, 1 for 11)."""
    attrs = {} if axis is None else {"axis": axis}
    return save_onnx(tmp_path, [helper.make_node(op, ["X"], ["Y"], **attrs)],
                     list(shape), list(shape), opset=opset, name=name,
                     in_type="DOUBLE", out_type="DOUBLE")


# (shape, axis, opset). Opset 13 takes one axis, opset 11 the flattened tail
# from the axis on; the two agree only when the axis is the last one.
_CASES = [
    ((1, 6), None, 13), ((1, 6), None, 11), ((1, 6), 0, 13), ((1, 6), 0, 11),
    ((1, 2, 3), 1, 13), ((1, 2, 3), 2, 13), ((1, 2, 3), -1, 13),
    ((1, 2, 3), -2, 13), ((1, 2, 3), None, 11), ((1, 2, 3), 2, 11),
    ((1, 2, 3), -1, 11), ((1, 2, 3), -2, 11),
    # axis 0 of a rank 3 tensor under opset 11: one softmax over all six
    # entries. On (1, 6) an axis of 0 that is read as "no axis given" still
    # lands on the same six entries; here it lands on rows of three.
    ((1, 2, 3), 0, 11),
    ((1, 3, 2), 1, 13), ((1, 3, 2), -2, 11),
    ((1, 2, 3, 2), 2, 13), ((1, 2, 3, 2), -3, 13), ((1, 2, 3, 2), 2, 11),
    ((1, 2, 3, 2), -2, 11),
]

# The handler reads the axis, normalises it and builds the opset-11 shape once
# for every set type; only its last line, the kernel call, is the set type's
# own. Every case runs on a Star; on a Zono and a Box the cases that tell that
# call apart: the default and the batch axis of opset 13, an inner axis, a
# negative last axis, a negative inner axis of rank 4; the batch axis, the
# default and a negative last axis of opset 11.
_BRANCH_CASES = [((1, 6), None, 13), ((1, 6), 0, 13), ((1, 2, 3), 1, 13),
                 ((1, 2, 3), -1, 13), ((1, 2, 3, 2), -3, 13),
                 ((1, 6), 0, 11), ((1, 2, 3), None, 11), ((1, 2, 3), -1, 11)]


def _case_id(case):
    shape, axis, opset = case
    return f"{'x'.join(map(str, shape))}-axis{axis}-opset{opset}"


@pytest.mark.parametrize("case,kind", [
    pytest.param(case, kind, id=f"{_case_id(case)}-{kind}")
    for case, kind in ([(case, "Star") for case in _CASES]
                       + [(case, kind) for kind in ("Zono", "Box")
                          for case in _BRANCH_CASES])])
def test_onnx_softmax_holds_every_output(tmp_path, case, kind):
    """One Softmax node on a flat set of a tensor of rank 2 to 4, through the
    model entry. Catches an axis taken from the wrong attribute or not passed
    on (a softmax over the last axis where the node names an earlier one), a
    negative axis normalised other than by adding the rank, the opset 11
    flattened tail built as single-axis or with its two parts swapped, a
    reshape of the flat set in the wrong order, the default axis of an
    opset taken from the other, and an axis attribute of 0 read as not given
    (opset 11 on rank 3). The batch axis (length 1) gives outputs that
    are exactly 1 under opset 13 and a softmax over the whole tensor under
    opset 11. The tensor shape of the flat input set is passed as
    ``input_shape``: without it a model input of rank 3 or 4 is analysed as
    (1, n) (registry row SHAPE-2,
    ``test_onnx_softmax_without_input_shape_refuses_or_holds_every_output``)."""
    shape, axis, opset = case
    lo, hi = _logits(shape)
    path = _softmax_graph(tmp_path, shape, axis, opset)
    sets, points = _flat_inputs(kind, lo, hi)
    what = f"Softmax(axis={axis}, opset {opset}) of {shape} on {kind}"
    out = reach_model(load_onnx(path), sets, "approx", what,
                      input_shape=tuple(shape[1:]))
    assert_covered(out, replay_onnx(path, points, list(shape)), what)


# (route, kind, input shape)
_OTHER_INPUT_CELLS = [
    ("reach_layer", "Star", "box times 8"),
    ("reach_layer", "Box", "box times 8"),
    ("onnx13", "Star", "box times 8"),
    ("onnx13", "Zono", "box times 8"),
    ("onnx13", "Box", "box times 8"),
    ("reach_layer", "Star", "more variables than entries"),
    ("onnx13", "Star", "more variables than entries"),
    ("onnx13", "Zono", "more variables than entries"),
]


@pytest.mark.parametrize("route,kind,shape", [
    pytest.param(*cell, id="-".join(cell)) for cell in _OTHER_INPUT_CELLS])
def test_softmax_on_an_input_of_another_shape_holds_every_output(
        tmp_path, route, kind, shape):
    """The six logits as input sets the box cells do not build: the box
    times 8 (bounds from -16 to 20, logit differences up to 36, outputs
    near 0 and 1), and the shape a relaxation leaves, a Zono with more
    generators than entries and a Star with more predicate variables than
    entries, one of them tied to the first entry's variable by a binding
    row. Through ``reach_layer`` (``softmax_star``, ``softmax_box``) and
    through an opset-13 Softmax over axis 1 of (1, 2, 3) (the handler).
    Catches a bound helper that cuts the exponent differences (the box
    cells spread at most 4.5), a kernel that reads an entry's range from the
    predicate variables, which lie in [-1, 1], or from as many variables as
    the set has entries."""
    sets, points = _other_input(kind, shape, _LO, _HI)
    what = f"Softmax through {route} on a {kind}, {shape}"
    if route == "reach_layer":
        out = reach_one_layer(nn.Softmax(dim=-1), sets, "approx", what)
        outputs = replay_torch(nn.Softmax(dim=-1), points, [1, _N])
    else:
        path = _softmax_graph(tmp_path, (1, 2, 3), 1, 13)
        out = reach_model(load_onnx(path), sets, "approx", what,
                          input_shape=(2, 3))
        outputs = replay_onnx(path, points, [1, 2, 3])
    assert_covered(out, outputs, what)


def _chain_graph(tmp_path, shape, axis, opset, relu=False, name="chain"):
    """Gemm (3 -> 6), [Relu], Reshape to ``shape``, Softmax, Reshape to
    (1, 6), Gemm (6 -> 6, invertible); float64."""
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["A"], transB=1)]
    base = "A"
    if relu:
        nodes.append(helper.make_node("Relu", ["A"], ["R"]))
        base = "R"
    attrs = {} if axis is None else {"axis": axis}
    nodes += [helper.make_node("Reshape", [base, "s_in"], ["S"]),
              helper.make_node("Softmax", ["S"], ["P"], **attrs),
              helper.make_node("Reshape", ["P", "s_out"], ["F"]),
              helper.make_node("Gemm", ["F", "W2", "b2"], ["Y"], transB=1)]
    inits = {"W1": _W1, "b1": _B1, "W2": _W2, "b2": _B2,
             "s_in": np.array(shape, dtype=np.int64),
             "s_out": np.array([1, _N], dtype=np.int64)}
    return save_onnx(tmp_path, nodes, [1, 3], [1, _N], inits, opset=opset,
                     name=name, in_type="DOUBLE", out_type="DOUBLE")


_CHAIN_CASES = [((1, 6), -1, 13), ((1, 2, 3), 1, 13), ((1, 2, 3), -1, 13),
                ((1, 2, 3), 1, 11), ((1, 3, 2), -2, 11)]
# On the other set types the chain runs once per opset, over an inner axis.
_CHAIN_BRANCH_CASES = [((1, 2, 3), 1, 13), ((1, 2, 3), 1, 11)]


@pytest.mark.parametrize("case,kind", [
    pytest.param(case, kind, id=f"{_case_id(case)}-{kind}")
    for case, kind in ([(case, "Star") for case in _CHAIN_CASES]
                       + [(case, kind) for kind in
                          ("Zono", "Box", "Star with a binding row")
                          for case in _CHAIN_BRANCH_CASES])])
def test_onnx_softmax_between_two_affine_layers(tmp_path, case, kind):
    """Gemm, Reshape, Softmax, Reshape, Gemm: the logits share the input's
    predicate variables and the second Gemm consumes the Softmax result.
    Catches a handler that reads the shape of the wrong node, returns the
    sets of another node, or bounds a Softmax over the wrong group once the
    logits are correlated."""
    shape, axis, opset = case
    path = _chain_graph(tmp_path, shape, axis, opset)
    if kind == "Star with a binding row":
        star, points = constrained_star(_IN_LO, _IN_HI, n=48)
        sets = [star]
    else:
        sets = [flat_set(kind, _IN_LO, _IN_HI)]
        points = samples(_IN_LO, _IN_HI, n=24) + _grid(_IN_LO, _IN_HI)
    what = (f"Gemm, Softmax(axis={axis}, opset {opset}) of {shape}, Gemm on "
            f"{kind}")
    out = reach_model(load_onnx(path), sets, "approx", what)
    assert_covered(out, replay_onnx(path, points, [1, 3]), what)


@pytest.mark.parametrize("case", [((1, 6), -1, 13), ((1, 2, 3), 1, 13),
                                  ((1, 2, 3), 1, 11)], ids=_case_id)
def test_onnx_softmax_after_an_exact_relu_split_holds_every_piece(tmp_path,
                                                                  case):
    """Gemm, Relu, Reshape, Softmax, Reshape, Gemm under ``method='exact'``:
    four of the six ReLU neurons cross zero on the input box, so the ReLU
    splits the input star into pieces and the Softmax handler receives all
    of them in one list. Softmax has no exact rule, so the call may refuse,
    but only with the refusal REACH-4 promises: a ``NotImplementedError``
    whose message names the Softmax and the exact method; any other refusal
    (an exact ReLU that refuses) fails the cell, which would otherwise check
    nothing. An answer must hold the outputs of every piece. Catches a
    handler that bounds only the first star of the list, or reuses its
    bounds for the others. Once REACH-4 makes exact refuse here, the pieces
    reach the Softmax through the approximate split option that REACH-4's
    fix names (``split_piecewise_linear=True`` under approx), which this
    cell should then use."""
    shape, axis, opset = case
    path = _chain_graph(tmp_path, shape, axis, opset, relu=True, name="split")
    points = samples(_IN_LO, _IN_HI, n=24) + _grid(_IN_LO, _IN_HI)
    patterns = {tuple(_W1 @ x + _B1 > 0) for x in points}
    if len(patterns) < 4:
        pytest.fail(f"the points meet only {len(patterns)} activation "
                    f"patterns; the cell needs points in several pieces")
    what = f"Gemm, Relu, Softmax(axis={axis}, opset {opset}), Gemm, exact"
    try:
        out = reach_model(load_onnx(path), flat_set("Star", _IN_LO, _IN_HI),
                          "exact", what)
    except NotImplementedError as refusal:
        message = str(refusal).lower()
        if "softmax" in message and "exact" in message:
            return
        raise
    if len(out) < 2:
        pytest.fail(f"{what}: the model returned {len(out)} set(s); the cell "
                    f"needs several pieces at the Softmax (the exact ReLU did "
                    f"not split, or a node after it dropped pieces)")
    assert_covered(out, replay_onnx(path, points, [1, 3]), what)


_C, _H, _W = 2, 3, 5


def _has_a_finite_predicate_box(star):
    """Whether ``star`` stores finite lower and upper predicate bounds."""
    return (star.predicate_lb is not None and star.predicate_ub is not None
            and bool(np.all(np.isfinite(star.predicate_lb)))
            and bool(np.all(np.isfinite(star.predicate_ub))))


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono",
                                  "ImageStar without predicate bounds"])
@pytest.mark.parametrize("axis", [1, 2], ids=["per pixel", "per channel"])
def test_onnx_softmax_of_a_flattened_image(tmp_path, axis, kind):
    """A (1, 2, 3, 5) image flattened, reshaped to (1, 2, 15) and passed to
    a Softmax over axis 1 (the two channels of every pixel) or axis 2 (the
    fifteen pixels of every channel), then flattened. The image set stores
    (height, width, channel), the flat set after Flatten is in the tensor's
    (channel, height, width) order, and each pixel has its own interval.
    Catches a Softmax over entries grouped in the stored order instead of
    the tensor's, or over the wrong axis of the (1, 2, 15) tensor. The
    ImageStar with a binding row and no predicate bounds (bounded by its
    rows alone) reaches the handler as a flat Star whose logit ranges the
    handler must enclose by LPs. Every Star the Softmax returns must carry a
    finite predicate box, whatever the input had (``Star.fresh_from_bounds``
    gives it one):
    ``tests/soundness/test_soundness_set_arith_cells.py::test_concat_of_a_system_without_a_predicate_box_and_one_with``
    relies on it to reach the join of a system with a box and one without."""
    lo, hi = pixel_box(_C, _H, _W)
    nodes = [helper.make_node("Flatten", ["X"], ["F"], axis=1),
             helper.make_node("Reshape", ["F", "s"], ["R"]),
             helper.make_node("Softmax", ["R"], ["S"], axis=axis),
             helper.make_node("Flatten", ["S"], ["Y"], axis=1)]
    path = save_onnx(tmp_path, nodes, [1, _C, _H, _W], [1, _C * _H * _W],
                     {"s": np.array([1, _C, _H * _W], dtype=np.int64)},
                     name="image_softmax", in_type="DOUBLE",
                     out_type="DOUBLE")
    what = f"Flatten, Softmax(axis={axis}) of (1, 2, 15) on {kind}"
    if kind in ("ImageStar", "ImageZono"):
        input_set = image_set(kind, lo, hi)
        points = [p.reshape(_C, _H, _W)
                  for p in _points(lo.reshape(-1), hi.reshape(-1))]
    else:
        input_set, points = constrained_image_star(lo, hi, n=48)
        # The same set: its rows hold the box rows +-I.
        input_set = ImageStar(input_set.V, input_set.C, input_set.d,
                              None, None, _H, _W, _C)
    out = reach_model(load_onnx(path), input_set, "approx", what)
    assert_covered(out, replay_onnx(path, points, [1, _C, _H, _W]), what)
    without_box = [k for k, s in enumerate(out) if isinstance(s, Star)
                   and not _has_a_finite_predicate_box(s)]
    assert not without_box, (
        f"{what}: returned Stars {without_box} of {len(out)} have no finite "
        f"predicate box")


@pytest.mark.parametrize("axis", [1, -1])
@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_onnx_softmax_opset11_on_an_image_set_refuses_or_holds_every_output(
        tmp_path, kind, axis):
    """An opset-11 Softmax (``OnnxSoftmaxV1V11``) directly on a (1, 2, 3, 5)
    image, then Flatten, reached with ``input_shape``: over the flattened
    tail from axis 1 (all thirty entries) or over the last axis (the
    width). The handler leaves image sets to the dispatcher, which has no
    rule for this module, so the call is refused today; an answer must hold
    every output in the tensor's own order. Catches an image set taken by
    the handler as a flat set (its entries are stored height, width,
    channel, so the groups come out wrong) and a rule that returns the
    softmax of the stored vector as a flat set. The opset-13 form is in
    ``tests/contracts/test_reach_coverage.py`` (ACT-1)."""
    lo, hi = pixel_box(_C, _H, _W)
    nodes = [helper.make_node("Softmax", ["X"], ["S"], axis=axis),
             helper.make_node("Flatten", ["S"], ["Y"], axis=1)]
    path = save_onnx(tmp_path, nodes, [1, _C, _H, _W], [1, _C * _H * _W],
                     opset=11, name="image_softmax11", in_type="DOUBLE",
                     out_type="DOUBLE")
    what = f"Softmax(axis={axis}, opset 11) of (1, 2, 3, 5) on {kind}"
    try:
        out = reach_model(load_onnx(path), image_set(kind, lo, hi), "approx",
                          what, input_shape=(_C, _H, _W))
    except NotImplementedError:
        return
    points = [p.reshape(_C, _H, _W)
              for p in _points(lo.reshape(-1), hi.reshape(-1))]
    assert_covered(out, replay_onnx(path, points, [1, _C, _H, _W]), what)


_FLAT_SET_READ_AS_1_BY_N = known_bug(
    12, "a flat set reached without input_shape on a model whose input is "
        "(1, 2, 3) is read as the tensor (1, 6), so the Softmax over the "
        "last axis is bounded as one softmax over all six entries; every "
        "model output escapes (registry SHAPE-2)")


@pytest.mark.parametrize("axis", [
    pytest.param(-1, marks=_FLAT_SET_READ_AS_1_BY_N, id="axis-1"),
    pytest.param(2, id="axis2"),
])
def test_onnx_softmax_without_input_shape_refuses_or_holds_every_output(
        tmp_path, axis):
    """An opset-13 Softmax over the last axis of a (1, 2, 3) model input,
    reached on a flat Star without ``input_shape``: the walk has only a
    guessed tensor shape. It must refuse, or every output must be covered.
    Today the guess (1, 6) runs and the handler bounds one softmax over the
    six entries (SHAPE-2); the same graph and input with ``input_shape`` is
    ``test_onnx_softmax_holds_every_output[1x2x3-axis-1-opset13-Star]``.
    Axis 2 cannot run on the guess, so no shape is recorded and the generic
    rule refuses it: the passing neighbour on this route."""
    path = _softmax_graph(tmp_path, (1, 2, 3), axis, 13)
    sets, points = _flat_inputs("Star", _LO, _HI)
    what = f"Softmax(axis={axis}) of (1, 2, 3) on a Star without input_shape"
    try:
        out = reach_model(load_onnx(path), sets, "approx", what)
    except NotImplementedError:
        return
    assert_covered(out, replay_onnx(path, points, [1, 2, 3]), what)


def _lost_shape_graph(tmp_path, flatten_axis):
    """ONNX Flatten(``flatten_axis``) of a (1, 2, 3, 2) input, Reshape to
    (1, 2, 6), Softmax over the last axis, Reshape to (1, 12); float64. Both
    axes flatten to (1, 12) in ONNX; with axis 0 the converted module's
    forward raises in torch on the rank-4 placeholder, so the shape pass
    records no shape for any node (SHAPE-8)."""
    nodes = [helper.make_node("Flatten", ["X"], ["F"], axis=flatten_axis),
             helper.make_node("Reshape", ["F", "s"], ["R"]),
             helper.make_node("Softmax", ["R"], ["S"], axis=-1),
             helper.make_node("Reshape", ["S", "t"], ["Y"])]
    return save_onnx(tmp_path, nodes, [1, 2, 3, 2], [1, 12],
                     {"s": np.array([1, 2, 6], dtype=np.int64),
                      "t": np.array([1, 12], dtype=np.int64)},
                     name=f"lost_shape_{flatten_axis}", in_type="DOUBLE",
                     out_type="DOUBLE")


_SOFTMAX_WITHOUT_A_SHAPE = known_bug(
    12, "a model with an ONNX Flatten of axis 0 on its rank-4 input loses "
        "its whole tensor-shape record, so the Softmax over the last axis of "
        "a (1, 2, 6) tensor falls through to the generic rule, one softmax "
        "over all twelve entries; every model output escapes (SHAPE-8)")


@pytest.mark.parametrize("flatten_axis,kind", [
    pytest.param(0, "Star", marks=_SOFTMAX_WITHOUT_A_SHAPE,
                 id="flatten-axis0-Star"),
    pytest.param(0, "Box", marks=_SOFTMAX_WITHOUT_A_SHAPE,
                 id="flatten-axis0-Box"),
    pytest.param(0, "Zono", id="flatten-axis0-Zono"),
    pytest.param(1, "Star", id="flatten-axis1-Star"),
])
def test_onnx_softmax_with_no_shape_record_refuses_or_holds_every_output(
        tmp_path, flatten_axis, kind):
    """A Softmax over the last axis of a (1, 2, 6) tensor in a model whose
    tensor-shape record is lost (SHAPE-8), reached with ``input_shape``: the
    handler has no shape and leaves the node to the dispatcher. It must
    refuse, or every output must be covered. Today the generic rule of the
    Star and the Box (``softmax_star``, ``softmax_box``) bounds one softmax
    over all twelve entries; the Zono has no generic rule and is refused,
    and with Flatten axis 1 the shape is recorded and the cell is sound:
    the passing neighbours."""
    path = _lost_shape_graph(tmp_path, flatten_axis)
    sets, points = _flat_inputs(kind, _LO12, _HI12)
    what = (f"Flatten(axis={flatten_axis}), Softmax over the last axis of "
            f"(1, 2, 6) on {kind}")
    try:
        out = reach_model(load_onnx(path), sets, "approx", what,
                          input_shape=(2, 3, 2))
    except NotImplementedError:
        return
    assert_covered(out, replay_onnx(path, points, [1, 2, 3, 2]), what)


@pytest.mark.parametrize("kind", ["Hexatope", "Octatope"])
@pytest.mark.parametrize("opset", [13, 11])
def test_onnx_softmax_refuses_a_hexatope_or_octatope(tmp_path, opset, kind):
    """No Softmax rule exists for these set types, on either opset; the
    model entry must raise rather than pass the set through."""
    model = load_onnx(_softmax_graph(tmp_path, (1, _N), -1, opset))
    with pytest.raises(NotImplementedError):
        reach_model(model, flat_set(kind, _LO, _HI), "approx",
                    f"Softmax opset {opset} on {kind}")


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
@pytest.mark.parametrize("opset", [13, 11])
def test_onnx_log_softmax_is_refused(tmp_path, opset, kind):
    """LogSoftmax has no rule (its outputs are negative and unbounded below);
    the opset 11 form shares its torch module with Softmax and differs only
    in a flag, so a handler that ignores the flag would answer with softmax
    bounds, which hold none of the true outputs. The call must raise."""
    model = load_onnx(_softmax_graph(tmp_path, (1, _N), 1, opset,
                                     op="LogSoftmax", name="log_softmax"))
    with pytest.raises(NotImplementedError):
        reach_model(model, flat_set(kind, _LO, _HI), "approx",
                    f"LogSoftmax opset {opset} on {kind}")

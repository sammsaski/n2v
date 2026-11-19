"""Soundness cells for layers in combination: the places where two layers
that are each sound alone can be unsound together.

The component is the code between the layers in ``n2v/nn/reach.py``: the
joins of two operands (``_add_sets``, ``_mul_sets`` with
``_mul_stars_mccormick``, ``_div_sets`` with ``_div_stars_mccormick``,
``_concat_sets``, ``_handle_bilinear_matmul``), what they share
(``_coerce_set_types``, ``_same_predicate_system``, ``_join_predicates``,
``_join_star_systems``, ``_pair_set_lists``, the predicate tags of
``Star.fresh_from_bounds``) and the tensor shapes that travel beside the set
(``_propagate_shapes`` and the graph handlers that read them). The
argument "every layer is sound, so the network is" needs three things the
walk does not give: only the set travels from layer to layer, every layer
meets every kind of set an earlier layer hands it, and the network is a
chain. Each cell here is placed where one of the three fails:

- a join of an operand whose star was rebuilt from bounds (Softmax, the
  attention softmax and set-by-set MatMul call ``Star.fresh_from_bounds``)
  with one whose star was not, in both operand orders, on a Star with a
  predicate box and on an ImageStar given by its rows alone (one system
  then has a predicate box and the other has none);
- the exact method with several pieces on both branches of a product or a
  quotient;
- every graph handler that reads the tracked tensor shape, on the three
  routes on which no shape is tracked or a wrong one is: a flat set given
  without ``input_shape`` on a rank-4 input (registry SHAPE-2), an
  ImageZono given without it (SHAPE-3), and a model whose shape record is
  emptied by a node whose torch forward raises (SHAPE-8);
- the McCormick product of two Stars after a Pow chain;
- a join of an ImageStar with an ImageZono;
- a Reshape that turns a flat set into an image for a spatial layer, a
  Reshape whose first entry is not 1 in front of a layer that reads the
  shape, and a set of zero width through a MatMul.

The promise checked by every cell: every true output of every member point
of the input set is a member of a returned set (or the reach refuses with
``NotImplementedError`` where a cell says so). True outputs come from
onnxruntime on the ONNX file or from exact rational arithmetic on dyadic
data; membership is decided by one LP on the returned set's own rows
(``cell_support.assert_covered``) or in exact arithmetic
(``tests/_oracles/rational.py``). Layout: bounds, points and outputs are in
the network's order; image sets store height, width, channel.
"""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from onnx import helper

from n2v.nn.reach import _add_sets, _concat_sets, _div_sets, _mul_sets
from n2v.sets import Star
from n2v.sets.image_star import ImageStar
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational
from tests.cell_support import (assert_covered, constrained_image_star,
                                flat_box, flat_set, image_set, pixel_box,
                                reach_model, replay_onnx, samples, save_onnx)
from tests.known_bugs import known_bug


def _i64(*values):
    return np.array(values, dtype=np.int64)


def _rename_input(nodes, old, new):
    """``nodes`` with every input named ``old`` renamed ``new``."""
    out = []
    for n in nodes:
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        out.append(helper.make_node(n.op_type,
                                    [new if i == old else i for i in n.input],
                                    list(n.output), **attrs))
    return out


# =========================================================================== #
# 1. A join of an operand rebuilt from bounds and one that is not            #
# =========================================================================== #
# join -> the operation that rebuilds its operand from bounds
_REBUILT = {"Add": "Softmax", "Sub": "MatMul", "Mul": "Softmax",
            "Div": "Softmax", "Concat": "MatMul", "MatMul": "Softmax"}
_REBOX_FORMS = ("Star", "ImageStar-rows-only")


def _rebox_join_model(tmp_path, join, order, form):
    """``(path, input shape)``: H, the input as a (1, 2, 2) tensor (a Reshape
    of a (1, 1, 2, 2) image for the ImageStar); R, H through Softmax over its
    last axis or H @ H (both rebuild their star from bounds); Q = H + k (on
    H's predicate system); Y = join(R, Q) or join(Q, R). Float64."""
    nodes, inits = [], {}
    if form == "ImageStar-rows-only":
        in_shape = [1, 1, 2, 2]
        nodes.append(helper.make_node("Reshape", ["X", "shape"], ["H"]))
        inits["shape"] = _i64(1, 2, 2)
    else:
        in_shape = [1, 2, 2]
    if _REBUILT[join] == "Softmax":
        nodes.append(helper.make_node("Softmax", ["H"], ["R"], axis=2))
    else:
        nodes.append(helper.make_node("MatMul", ["H", "H"], ["R"]))
    nodes.append(helper.make_node("Add", ["H", "k"], ["Q"]))
    # Q is the divisor of R / Q: shifted away from zero over the input box
    inits["k"] = np.array(16.0 if join == "Div" else 0.5)
    a, b = ("R", "Q") if order == "rebuilt-first" else ("Q", "R")
    attrs = {"axis": 2} if join == "Concat" else {}
    nodes.append(helper.make_node(join, [a, b], ["Y"], **attrs))
    if form != "ImageStar-rows-only":
        nodes = _rename_input(nodes, "H", "X")
    out = [1, 2, 4] if join == "Concat" else [1, 2, 2]
    path = save_onnx(tmp_path, nodes, in_shape, out, inits=inits,
                     name=f"rebox-{join}-{order}", in_type="DOUBLE",
                     out_type="DOUBLE")
    return path, in_shape


@pytest.mark.parametrize("form", _REBOX_FORMS)
@pytest.mark.parametrize("order", ["rebuilt-first", "rebuilt-second"])
@pytest.mark.parametrize("join", sorted(_REBUILT))
def test_join_of_an_operand_rebuilt_from_bounds_and_one_that_is_not(
        tmp_path, join, order, form):
    """Add, Sub, Mul, Div, Concat and set-by-set MatMul of R, an operand
    rebuilt from bounds (Softmax, the attention softmax, or H @ H, whose
    stars are ``Star.fresh_from_bounds``), and Q = H + k, which keeps the
    input's predicate system, in both operand orders. Two inputs, each
    entry wider than 2 and beyond [-1, 1]: a flat Star given by its box
    (``from_bounds``), whose system is the one a plain ``from_bounds`` of
    the rebuilt operand would have, and an ImageStar given by its rows
    alone, cut by a binding row, so that Q has no predicate box while R has
    one (the join then takes the branch for mixed boxes of
    ``_join_predicates`` and ``_join_star_systems``). 128 onnxruntime
    outputs must be members of a returned set by LP. Catches a rebuilt star
    whose variables are read as the input's (a tag lost from
    ``fresh_from_bounds``, a system comparison that ignores the predicate
    box), a missing predicate box filled with finite values in a join, and
    a product, quotient or concatenation that lays the two systems out
    wrongly."""
    path, in_shape = _rebox_join_model(tmp_path, join, order, form)
    lo, hi = flat_box(4, seed=5)
    lo, hi = 4 * lo, 4 * hi
    if form == "ImageStar-rows-only":
        s, points = constrained_image_star(lo.reshape(1, 2, 2),
                                           hi.reshape(1, 2, 2))
        input_set = ImageStar(s.V, s.C, s.d, None, None, height=2, width=2,
                              num_channels=1)
        options = {}
    else:
        input_set, points = flat_set("Star", lo, hi), samples(lo, hi)
        options = {"input_shape": (2, 2)}
    what = f"{join} ({order}) of {_REBUILT[join]}(H) and H + k on a {form}"
    sets = reach_model(load_onnx(path), input_set, "approx", what, **options)
    assert_covered(sets, replay_onnx(path, points, in_shape), what)


# =========================================================================== #
# 2. The exact method: a product or quotient of two split branches          #
# =========================================================================== #
_LO3, _HI3 = np.array([-1.0, -0.75]), np.array([1.0, 1.25])
_WA3 = np.array([[1.0, -0.5], [0.5, 1.0]])
_BA3 = np.array([0.25, -0.25])
_WB3 = np.array([[-0.5, 1.0], [1.0, 0.5]])
_BB3 = np.array([-0.25, 0.5])


@pytest.mark.parametrize("op,order", [("Mul", "A-first"), ("Mul", "B-first"),
                                      ("Div", "A-first"), ("Div", "B-first")])
def test_exact_product_or_quotient_of_two_split_branches_holds_every_output(
        tmp_path, op, order):
    """method='exact': A = Relu(X WA + bA) and B = Relu(X WB + bB), each of
    which splits into several pieces on a different pair of input regions,
    then A * B or B * A, and A / (B + 4) or B / (A + 4). The two lists of
    pieces do not descend from one another, so every pair must be formed
    (``_pair_set_lists``) and each pair gets its McCormick product or
    quotient. 128 onnxruntime outputs must be members of a returned set by
    LP. Catches a product or quotient that zips the two lists, keeps only
    the pairs of one reference piece, or drops the pieces of the second
    operand. The result is not the exact image under method='exact'
    (REACH-4, marked in ``test_soundness_set_arith_cells.py``); only
    enclosure is checked. When PLAN step 18 makes method='exact' refuse a
    product or quotient, this cell must move to the approximate method with
    ``split_piecewise_linear=True``."""
    shift = 4.0 if op == "Div" else 0.0
    nodes = [helper.make_node("MatMul", ["X", "WA"], ["a0"]),
             helper.make_node("Add", ["a0", "BA"], ["a1"]),
             helper.make_node("Relu", ["a1"], ["A"]),
             helper.make_node("MatMul", ["X", "WB"], ["b0"]),
             helper.make_node("Add", ["b0", "BB"], ["b1"]),
             helper.make_node("Relu", ["b1"], ["B"])]
    first, second = ("A", "B") if order == "A-first" else ("B", "A")
    if op == "Div":
        nodes.append(helper.make_node("Add", [second, "S"], ["D"]))
        second = "D"
    nodes.append(helper.make_node(op, [first, second], ["Y"]))
    inits = {"WA": _WA3, "BA": _BA3, "WB": _WB3, "BB": _BB3}
    if op == "Div":
        inits["S"] = np.array(shift)
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 2], inits=inits,
                     name=f"exact-{op}-{order}", in_type="DOUBLE",
                     out_type="DOUBLE")
    what = f"exact {op} ({order}) of two split branches"
    sets = reach_model(load_onnx(path), flat_set("Star", _LO3, _HI3), "exact",
                       what)
    if len(sets) < 8:
        pytest.fail(f"{what}: {len(sets)} sets came back; the cell needs "
                    f"two operands split into several pieces each")
    assert_covered(sets, replay_onnx(path, samples(_LO3, _HI3), [1, 2]), what)


# =========================================================================== #
# 3. Every graph handler on the routes with no tracked tensor shape          #
# =========================================================================== #
_C, _H, _W = 2, 3, 4
_TENSOR = [1, _C, _H, _W]

# handler -> (nodes reading the (1, 2, 3, 4) tensor H, initializers, output
# shape). Every handler's output depends on the tensor shape: the axis it
# reads is an inner one, or the constant broadcasts along one.
_HANDLERS = {
    "reshape": ([helper.make_node("Reshape", ["H", "r"], ["Y"])],
                {"r": _i64(1, 6, 4)}, [1, 6, 4]),
    "flatten": ([helper.make_node("Flatten", ["H"], ["Y"], axis=2)], {},
                [2, 12]),
    "transpose": ([helper.make_node("Transpose", ["H"], ["Y"],
                                    perm=[0, 2, 3, 1])], {}, [1, 3, 4, 2]),
    "reduce": ([helper.make_node("ReduceMean", ["H"], ["Y"], axes=[3],
                                 keepdims=0)], {}, [1, 2, 3]),
    "softmax": ([helper.make_node("Softmax", ["H"], ["Y"], axis=2)], {},
                [1, 2, 3, 4]),
    "split": ([helper.make_node("Split", ["H", "sp"], ["P0", "P1"], axis=2),
               helper.make_node("Neg", ["P1"], ["Y"])], {"sp": _i64(1, 2)},
              [1, 2, 2, 4]),
    "slice": ([helper.make_node("Slice", ["H", "s", "e", "a"], ["Y"])],
              {"s": _i64(1), "e": _i64(3), "a": _i64(2)}, [1, 2, 2, 4]),
    "gather": ([helper.make_node("Gather", ["H", "g"], ["Y"], axis=2)],
               {"g": _i64(2, 0)}, [1, 2, 2, 4]),
    "concat": ([helper.make_node("Neg", ["H"], ["N"]),
                helper.make_node("Concat", ["H", "N"], ["Y"], axis=3)], {},
               [1, 2, 3, 8]),
    "constant-add-over-height": (
        [helper.make_node("Add", ["H", "k"], ["Y"])],
        {"k": np.array([0.5, -2.0, 3.25]).reshape(1, 1, 3, 1)},
        [1, 2, 3, 4]),
    "constant-mul-over-width": (
        [helper.make_node("Mul", ["H", "k"], ["Y"])],
        {"k": np.array([0.5, -2.0, 1.5, 4.0])}, [1, 2, 3, 4]),
    "matmul-constant": (
        [helper.make_node("MatMul", ["H", "w"], ["Y"])],
        {"w": np.array([[0.5, -1.0], [2.0, 0.25], [-0.75, 1.0],
                        [1.25, -0.5]])}, [1, 2, 3, 2]),
}

# route -> (input set kind, input_shape given, the nodes in front of H)
# "no-input-shape": a flat Star on the rank-4 input, no input_shape (SHAPE-2);
# "image-zono": an ImageZono, no input_shape, made flat behind a Conv, a
# Flatten and a Reshape (SHAPE-3; the Conv makes the flat placeholders fail);
# "lost-record": a flat Star with input_shape behind a Flatten with axis 0,
# whose torch forward raises on the rank-4 placeholder (SHAPE-8). The other
# three are the neighbours with the shape tracked.
_ROUTES = {
    "no-input-shape": ("Star", False, None),
    "input-shape": ("Star", True, None),
    "image-zono": ("ImageZono", False, "conv"),
    "image-star": ("ImageStar", False, "conv"),
    "lost-record": ("Star", True, 0),
    "flatten-axis1": ("Star", True, 1),
}

_SHAPE_3_SPLIT = known_bug(
    12, "SHAPE-3: an ImageZono given without input_shape gets no tracked "
        "tensor shape, so behind a Conv, a Flatten and a Reshape the Split "
        "along an inner axis of the flat set takes the fallback that cuts "
        "the rows as if the axis were the outermost one: 2 entries where "
        "the output has 16")
_SHAPE_3_SLICE = known_bug(
    12, "SHAPE-3: an ImageZono given without input_shape gets no tracked "
        "tensor shape, so behind a Conv, a Flatten and a Reshape the Slice "
        "along an inner axis of the flat set takes the fallback that slices "
        "only axis 0 of a flat set and returns it whole: 24 entries where "
        "the output has 16")
_SHAPE_8_SLICE = known_bug(
    12, "SHAPE-8: an ONNX Flatten with axis 0 on the rank-4 input makes the "
        "torch forward of the shape pass raise, so no tensor shape is "
        "recorded although input_shape is given, and the Slice along an "
        "inner axis of the flat set takes the fallback that slices only axis "
        "0 of a flat set and returns it whole: 24 entries where the output "
        "has 16 (a consumer the registry row does not list)")

# (handler, route, marks). Left out, because a marked case exists elsewhere
# for exactly that handler and route: split on no-input-shape
# (test_soundness_tensor_shape.py, SHAPE-2), slice on no-input-shape (the
# same file, SHAPE-2), split on lost-record (test_soundness_split.py,
# SHAPE-8), constant-add-over-height on no-input-shape
# (test_soundness_const_arith_cells.py, SHAPE-2).
_ROUTE_CELLS = (
    [(h, r, ()) for h in ("reshape", "flatten", "transpose", "reduce",
                          "softmax", "gather", "concat",
                          "constant-mul-over-width", "matmul-constant")
     for r in ("no-input-shape", "image-zono", "lost-record")]
    + [("split", "input-shape", ()), ("split", "image-zono", _SHAPE_3_SPLIT),
       ("split", "image-star", ()),
       ("slice", "image-zono", _SHAPE_3_SLICE), ("slice", "image-star", ()),
       ("slice", "lost-record", _SHAPE_8_SLICE),
       ("slice", "flatten-axis1", ()),
       ("constant-add-over-height", "input-shape", ()),
       ("constant-add-over-height", "image-zono", ()),
       ("constant-add-over-height", "lost-record", ())])

# The conv in front of the image routes: a 1x1 kernel of quarters.
_K = np.array([[[[0.5]], [[-1.0]]], [[[0.25]], [[0.75]]]])
_KB = np.array([0.25, -0.5])


def _route_model(tmp_path, handler, route):
    nodes, inits, out = _HANDLERS[handler]
    inits = dict(inits)
    front = _ROUTES[route][2]
    if front is None:
        nodes = _rename_input(nodes, "H", "X")
        pre = []
    elif front == "conv":
        pre = [helper.make_node("Conv", ["X", "K", "kb"], ["A"]),
               helper.make_node("Flatten", ["A"], ["F"], axis=1),
               helper.make_node("Reshape", ["F", "back"], ["H"])]
        inits.update({"K": _K, "kb": _KB, "back": _i64(*_TENSOR)})
    else:
        pre = [helper.make_node("Flatten", ["X"], ["F"], axis=front),
               helper.make_node("Reshape", ["F", "back"], ["H"])]
        inits["back"] = _i64(*_TENSOR)
    inits = {k: (v.astype(np.float32) if v.dtype.kind == "f" else v)
             for k, v in inits.items()}
    return save_onnx(tmp_path, pre + nodes, _TENSOR, out, inits=inits,
                     name=f"{handler}-{route}")


@pytest.mark.parametrize("handler,route", [
    pytest.param(h, r, marks=list(m) if isinstance(m, tuple) else [m],
                 id=f"{h}-{r}") for h, r, m in _ROUTE_CELLS])
def test_a_graph_handler_with_no_tracked_shape_refuses_or_holds_every_output(
        tmp_path, handler, route):
    """Each graph handler that reads the tracked tensor shape (Reshape,
    Flatten, Transpose, Reduce, Softmax, Split, Slice, Gather, Concat, the
    constant operand of Add and Mul, MatMul with a constant) on a
    (1, 2, 3, 4) tensor whose inner axes it uses, on the three routes that
    track no shape, or a wrong one: a flat Star on the rank-4 input with no
    ``input_shape`` (SHAPE-2); an ImageZono with no ``input_shape``, made
    flat behind a Conv, a Flatten and a Reshape (SHAPE-3); a flat Star with
    ``input_shape`` behind an ONNX Flatten with axis 0, whose torch forward
    raises in the shape pass so that the whole record is lost (SHAPE-8).
    The neighbours carry the shape: the same model with ``input_shape``, on
    an ImageStar, or with Flatten axis 1. Data are quarters (exact in
    float32) on intervals up to 5 wide and beyond [-1, 1]. The promise: a
    refusal with ``NotImplementedError`` (a ``ValueError`` for MatMul, from
    the affine map's dimension check, as ``test_soundness_matmul.py``
    accepts it), or every onnxruntime output of 128 member points in a
    returned set, with the output's number of entries. Catches a handler
    whose fallback guesses the missing shape (a constant tiled over the
    entries, an axis read as the outermost one, a Gather or Concat that
    reads a placeholder shape), and a shape pass that loses ``input_shape``
    (the neighbours fail then)."""
    kind, given, _ = _ROUTES[route]
    path = _route_model(tmp_path, handler, route)
    lo, hi = pixel_box(_C, _H, _W, seed=3)
    lo, hi = 4 * lo, 4 * hi
    points = samples(lo, hi)
    outputs = replay_onnx(path, points, _TENSOR)
    if kind == "Star":
        input_set = flat_set("Star", lo, hi)
    else:
        input_set = image_set(kind, lo, hi)
    options = {"input_shape": (_C, _H, _W)} if given else {}
    refusals = ((NotImplementedError, ValueError)
                if handler == "matmul-constant" else NotImplementedError)
    what = f"{handler} on the route {route} ({kind})"
    try:
        sets = reach_model(load_onnx(path), input_set, "approx", what,
                           **options)
    except refusals:
        return
    assert_covered(sets, outputs, what)


# =========================================================================== #
# 4. The McCormick product of two Stars after a Pow chain                    #
# =========================================================================== #
_PW1 = np.array([[1.0, 0.5], [-0.5, 1.0]])
_PW2 = np.array([[0.25, 1.0], [1.0, -0.75]])
_PLO, _PHI = np.array([-1.5, 0.5]), np.array([1.25, 2.25])


def _fractions(x):
    return [Fraction(float(v)) for v in np.asarray(x).reshape(-1)]


def _times(x, W):
    """``x @ W`` in fractions, ``x`` a vector, ``W`` a float matrix."""
    W = np.asarray(W)
    return [sum(xi * Fraction(float(W[i, j])) for i, xi in enumerate(x))
            for j in range(W.shape[1])]


# model -> (product, the second operand's nodes, its true value from U and V)
_POW_PRODUCTS = {
    # (X W1)^2 times X W2: the operands' systems differ (joined)
    "square-times-linear": (
        [helper.make_node("Mul", ["P", "V"], ["Y"])],
        lambda u, v: [a * a * b for a, b in zip(u, v)]),
    # (X W1)^2 times (X W2)^3: two Pow relaxations (joined)
    "square-times-cube": (
        [helper.make_node("Pow", ["V", "three"], ["T"]),
         helper.make_node("Mul", ["P", "T"], ["Y"])],
        lambda u, v: [a * a * b ** 3 for a, b in zip(u, v)]),
    # (X W1)^2 times its negation: one shared system
    "square-times-its-negation": (
        [helper.make_node("Neg", ["P"], ["N"]),
         helper.make_node("Mul", ["P", "N"], ["Y"])],
        lambda u, v: [-(a ** 4) for a in u]),
    # (X W1)^2 times itself: the same node twice
    "square-times-itself": (
        [helper.make_node("Mul", ["P", "P"], ["Y"])],
        lambda u, v: [a ** 4 for a in u]),
}


def _diagonal_points(lo, hi, steps=16):
    """The corners of the box [lo, hi] of a 2-vector, and the points at
    k / steps along its diagonal and its anti-diagonal."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    points = [np.array(c) for c in product(*zip(lo, hi))]
    for k in range(steps + 1):
        t = k / steps
        points.append(lo + t * (hi - lo))
        points.append(np.array([lo[0] + t * (hi[0] - lo[0]),
                                hi[1] - t * (hi[1] - lo[1])]))
    return points


@pytest.mark.parametrize("model", sorted(_POW_PRODUCTS))
def test_product_after_a_pow_chain_holds_every_output(tmp_path, model):
    """U = X W1, V = X W2, P = Pow(U, 2), then P times a second operand: X W2
    or Pow(X W2, 3) (operands on two predicate systems, joined), Neg(P) or P
    itself (one shared system). Both kernels of ``_mul_stars_mccormick``
    run on stars that carry the Pow relaxation's variables. Float64 graph;
    the box straddles zero in X0, so U straddles zero. The true outputs at
    the corners of the input box and at 17 points of each of its diagonals
    are computed in fractions and must be members of a returned set in
    exact arithmetic, with no slack; 128 onnxruntime outputs must be
    members by LP. Catches a McCormick envelope row with a wrong corner or
    sign, a product box missing a corner product, operand ranges taken
    from the wrong operand, and the shared and the joined systems laid out
    wrongly."""
    tail, value = _POW_PRODUCTS[model]
    nodes = [helper.make_node("MatMul", ["X", "W1"], ["U"]),
             helper.make_node("MatMul", ["X", "W2"], ["V"]),
             helper.make_node("Pow", ["U", "two"], ["P"])] + tail
    inits = {"W1": _PW1, "W2": _PW2, "two": np.array(2.0)}
    if model == "square-times-cube":
        inits["three"] = np.array(3.0)
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 2], inits=inits,
                     name=model, in_type="DOUBLE", out_type="DOUBLE")
    what = f"{model} after a Pow chain on a Star"
    sets = reach_model(load_onnx(path), flat_set("Star", _PLO, _PHI),
                       "approx", what)
    holders = [rational.Membership(rational.star(s)) for s in sets]
    for x in _diagonal_points(_PLO, _PHI):
        xf = _fractions(x)
        y = value(_times(xf, _PW1), _times(xf, _PW2))
        assert any(h(y) for h in holders), (
            f"{what}: the true output {[float(v) for v in y]} at the input "
            f"{x.tolist()} is in no returned set (exact arithmetic)")
    assert_covered(sets, replay_onnx(path, samples(_PLO, _PHI), [1, 2]),
                   what)


def _single_output_star(tmp_path, nodes, inits, name):
    path = save_onnx(tmp_path, nodes, [1, 2], [1, 1], inits=inits, name=name,
                     in_type="DOUBLE", out_type="DOUBLE")
    sets = reach_model(load_onnx(path), flat_set("Star", _PLO, _PHI),
                       "approx", name)
    if len(sets) != 1:
        pytest.fail(f"{name}: {len(sets)} sets; the cell needs one operand")
    return sets[0]


def _corner_range(w):
    """The exact range of ``x @ w`` (a column) over the input box."""
    values = [_times(_fractions(c), w)[0] for c in product(*zip(_PLO, _PHI))]
    return min(values), max(values)


@pytest.mark.parametrize("pairing", ["joined", "shared"])
def test_product_of_two_stars_holds_each_operand_pair_with_its_product(
        tmp_path, pairing):
    """``_mul_sets`` called on two operand Stars of one entry each, both
    reached through the model entry: x = P = Pow(X w1, 2) and, for "joined",
    y = X w2 (another predicate system: the kernel joins the two
    block-diagonally, so x and y range independently over their sets), for
    "shared", y = -P (the same system: y = -x). The triple (x, y, x y) must
    lie in the result read over its predicate variables
    [operand variables; z]: x and y through the operands' own bases, z
    through the new variable. Checked in exact arithmetic, with no slack,
    for the four corners of the box of the true ranges of x and y and for
    points along its diagonal and anti-diagonal ("joined"), and for points
    along x = -y ("shared"). The product of two values of a box is extreme
    at a corner, and a point on a diagonal lies strictly between the
    envelope planes, so a plane with the wrong corner or sign, or a box of
    z that misses a corner product, leaves a triple out while every true
    output may still be held. The cell assumes the layout the kernel
    documents (joined: the first operand's variables, then the second's,
    then z) and fails through ``pytest.fail`` if the variable count
    differs."""
    w1, w2 = _PW1[:, :1], _PW2[:, :1]
    P = _single_output_star(
        tmp_path, [helper.make_node("MatMul", ["X", "w1"], ["U"]),
                   helper.make_node("Pow", ["U", "two"], ["Y"])],
        {"w1": w1, "two": np.array(2.0)}, "pow-operand")
    if pairing == "joined":
        Q = _single_output_star(
            tmp_path, [helper.make_node("MatMul", ["X", "w2"], ["Y"])],
            {"w2": w2}, "linear-operand")
    else:
        Q = Star(-np.asarray(P.V), P.C, P.d, P.predicate_lb, P.predicate_ub)
    Z = _mul_sets([P], [Q])
    if len(Z) != 1:
        pytest.fail(f"{pairing}: {len(Z)} sets for one pair of operands")
    Z = Z[0]
    nP, nQ = int(P.nVar), int(Q.nVar)
    n = nP + nQ if pairing == "joined" else nP
    if int(Z.nVar) != n + 1:
        pytest.fail(f"{pairing}: the product has {Z.nVar} predicate "
                    f"variables; the cell reads {n} operand variables and one "
                    f"new one")
    _, _, C, d, lb, ub = rational.star(Z)
    cP, gP = _fractions(np.asarray(P.V)[:, 0]), _fractions(
        np.asarray(P.V)[:, 1:])
    cQ, gQ = _fractions(np.asarray(Q.V)[:, 0]), _fractions(
        np.asarray(Q.V)[:, 1:])
    zero = Fraction(0)
    if pairing == "joined":
        G = [gP + [zero] * nQ + [zero], [zero] * nP + gQ + [zero],
             [zero] * n + [Fraction(1)]]
    else:
        G = [gP + [zero], gQ + [zero], [zero] * n + [Fraction(1)]]
    holder = rational.Membership(([cP[0], cQ[0], zero], G, C, d, lb, ub))
    ul, uu = _corner_range(w1)
    xl = zero if ul < 0 < uu else min(ul * ul, uu * uu)
    xu = max(ul * ul, uu * uu)
    if pairing == "joined":
        yl, yu = _corner_range(w2)
        pairs = [(x, y) for x in (xl, xu) for y in (yl, yu)]
        for k in range(9):
            t = Fraction(k, 8)
            pairs.append((xl + t * (xu - xl), yl + t * (yu - yl)))
            pairs.append((xl + t * (xu - xl), yu - t * (yu - yl)))
    else:
        pairs = [(xl + Fraction(k, 8) * (xu - xl),
                  -(xl + Fraction(k, 8) * (xu - xl))) for k in range(9)]
    missed = [(x, y) for x, y in pairs if not holder([x, y, x * y])]
    assert not missed, (
        f"{pairing}: {len(missed)} of {len(pairs)} operand pairs with their "
        f"product are not in the result, for example "
        f"{[float(v) for v in missed[0]]}")


# =========================================================================== #
# 5. A join of an ImageStar with an ImageZono                                 #
# =========================================================================== #
_IMAGE_JOINS = {
    "Add": lambda a, b: _add_sets([a], [b], "add"),
    "Sub": lambda a, b: _add_sets([a], [b], "sub"),
    "Mul": lambda a, b: _mul_sets([a], [b]),
    "Div": lambda a, b: _div_sets([a], [b]),
    "Concat": lambda a, b: _concat_sets([[a], [b]], 2),
}
_CONCAT_STAR_ZONO = known_bug(
    23, "_concat_sets of an ImageStar and an ImageZono hands the ImageZono to "
        "_join_star_systems, which reads its nVar: AttributeError where a "
        "refusal is due (latent: the graph walk never builds the pair)",
    raises=AttributeError)
_CONCAT_ZONO_STAR = known_bug(
    23, "_concat_sets of an ImageZono and an ImageStar takes the ImageZono "
        "branch and reshapes the ImageStar's four-dimensional basis as a "
        "centre: numpy ValueError where a refusal is due (latent: the graph "
        "walk never builds the pair)", raises=ValueError)


@pytest.mark.parametrize("op,order", [
    pytest.param(op, order, id=f"{op}-{order}",
                 marks=([_CONCAT_STAR_ZONO] if (op, order) == (
                     "Concat", "ImageStar-first") else
                     [_CONCAT_ZONO_STAR] if op == "Concat" else []))
    for op in _IMAGE_JOINS for order in ("ImageStar-first",
                                         "ImageZono-first")])
def test_join_of_an_image_star_and_an_image_zono_is_refused(op, order):
    """The helpers that join two operands, handed an ImageStar and an
    ImageZono of one (2, 3, 5) image in either order. ``_coerce_set_types``
    brings an image set and a flat set to one type, but has no rule for
    these two families, so each helper must refuse with
    ``NotImplementedError`` rather than read one set as the other. Direct
    calls: the graph walk starts from one set and never builds the pair.
    Catches a helper that answers for a pair of families it has no rule
    for."""
    lo, hi = pixel_box(2, 3, 5)
    a, b = image_set("ImageStar", lo, hi), image_set("ImageZono", lo, hi)
    if order == "ImageZono-first":
        a, b = b, a
    with pytest.raises(NotImplementedError):
        _IMAGE_JOINS[op](a, b)


# =========================================================================== #
# 6. A Reshape in front of a layer that reads the layout                     #
# =========================================================================== #
_SPATIAL_LAYERS = {
    "Conv": ([helper.make_node("Conv", ["R", "K"], ["A"])],
             {"K": (np.arange(16).reshape(2, 2, 2, 2) % 5 - 2) / 4.0},
             [1, 2, 3, 5]),
    "ConvTranspose": ([helper.make_node("ConvTranspose", ["R", "K"], ["A"])],
                      {"K": (np.arange(16).reshape(2, 2, 2, 2) % 5 - 2)
                       / 4.0}, [1, 2, 5, 7]),
    "MaxPool": ([helper.make_node("MaxPool", ["R"], ["A"],
                                  kernel_shape=[2, 2])], {}, [1, 2, 3, 5]),
    "MaxPool-asymmetric-pads": (
        [helper.make_node("MaxPool", ["R"], ["A"], kernel_shape=[2, 2],
                          pads=[0, 0, 1, 1])], {}, [1, 2, 4, 6]),
    "AveragePool": ([helper.make_node("AveragePool", ["R"], ["A"],
                                      kernel_shape=[2, 2])], {}, [1, 2, 3, 5]),
}


@pytest.mark.parametrize("layer,kind", [
    ("Conv", "Star"), ("Conv", "Zono"), ("ConvTranspose", "Star"),
    ("MaxPool", "Star"), ("MaxPool-asymmetric-pads", "Star"),
    ("AveragePool", "Star")])
def test_reshape_of_a_flat_set_into_an_image_for_a_spatial_layer(tmp_path,
                                                                 layer, kind):
    """X (1, 48) reshaped to a (1, 2, 4, 6) image, then a spatial layer of
    each class ``load_onnx`` builds for Conv, ConvTranspose, MaxPool (with
    asymmetric pads it builds a Sequential of a Pad and a MaxPool) and
    AveragePool, then Flatten. The walk turns the flat set into an image
    set only when the Reshape feeds a class it lists as spatial, by the
    permutation from the tensor's (C, H, W) order to the stored
    (H, W, C) order. The promise: ``NotImplementedError``, or 128 onnxruntime
    outputs in a returned set. Catches a Reshape to an image that permutes
    the entries wrongly or keeps the tensor's order as the stored one.
    GlobalAveragePool, whose class the list misses, is marked in
    ``test_soundness_global_avgpool.py`` (GRAPH-28)."""
    nodes, inits, a_shape = _SPATIAL_LAYERS[layer]
    nodes = ([helper.make_node("Reshape", ["X", "shape"], ["R"])] + nodes
             + [helper.make_node("Flatten", ["A"], ["Y"], axis=1)])
    inits = {k: v.astype(np.float32) for k, v in inits.items()}
    inits["shape"] = _i64(1, 2, 4, 6)
    path = save_onnx(tmp_path, nodes, [1, 48], [1, int(np.prod(a_shape))],
                     inits=inits, name=f"reshape-{layer}")
    lo, hi = pixel_box(2, 4, 6, seed=2)
    lo, hi = lo.reshape(-1), hi.reshape(-1)
    what = f"Reshape to (1, 2, 4, 6), {layer}, Flatten on a {kind}"
    try:
        sets = reach_model(load_onnx(path), flat_set(kind, lo, hi), "approx",
                           what)
    except NotImplementedError:
        return
    assert_covered(sets, replay_onnx(path, samples(lo, hi), [1, 48]), what)


_BATCH_LAYERS = {
    "MatMul": ([helper.make_node("MatMul", ["R", "W"], ["Y"])],
               {"W": np.array([[0.5, -1.0], [0.25, 0.75], [-0.5, 1.0]])},
               [2, 2, 2]),
    "Softmax-axis0": ([helper.make_node("Softmax", ["R"], ["Y"], axis=0)],
                      {}, [2, 2, 3]),
    "BatchNormalization": (
        [helper.make_node("BatchNormalization", ["R", "sc", "bi", "mu", "va"],
                          ["Y"])],
        {"sc": np.array([2.0, -0.5]), "bi": np.array([0.25, 1.0]),
         "mu": np.array([0.5, -1.0]), "va": np.array([4.0, 0.25])},
        [2, 2, 3]),
}


@pytest.mark.parametrize("layer", sorted(_BATCH_LAYERS))
def test_reshape_whose_first_entry_is_not_one_then_a_layer_that_reads_it(
        tmp_path, layer):
    """X (1, 4, 3) reshaped to [2, -1, 3], a tensor whose axis 0 is a batch
    of two, then a layer whose result depends on that axis: MatMul by a
    constant (each of the four rows), Softmax over axis 0, BatchNormalization
    over the two channels of axis 1. The Reshape handler drops the first
    entry of its target as a batch axis of one (the root of GRAPH-23 for
    Conv1d, marked in ``test_soundness_conv1d.py``); these layers read the
    tracked tensor shape instead, which is the torch one. Float64 graph,
    a flat Star with ``input_shape`` (4, 3) and intervals beyond [-1, 1].
    The promise: ``NotImplementedError``, or 128 onnxruntime outputs in a
    returned set. Catches a layer that takes axis 0 of the tensor for a
    batch axis of one."""
    nodes, inits, out = _BATCH_LAYERS[layer]
    nodes = [helper.make_node("Reshape", ["X", "shape"], ["R"])] + nodes
    inits = dict(inits)
    inits["shape"] = _i64(2, -1, 3)
    path = save_onnx(tmp_path, nodes, [1, 4, 3], out, inits=inits,
                     name=f"batch-{layer}", in_type="DOUBLE",
                     out_type="DOUBLE")
    lo, hi = flat_box(12, seed=4)
    lo, hi = 4 * lo, 4 * hi
    what = f"Reshape to [2, -1, 3], then {layer}, on a Star"
    try:
        sets = reach_model(load_onnx(path), flat_set("Star", lo, hi),
                           "approx", what, input_shape=(4, 3))
    except NotImplementedError:
        return
    assert_covered(sets, replay_onnx(path, samples(lo, hi), [1, 4, 3]), what)


# =========================================================================== #
# 7. A set of zero width through a constant MatMul                           #
# =========================================================================== #
_POINT_W = np.array([[0.5, -1.0, 2.0], [0.25, 0.75, -0.5],
                     [-1.5, 1.0, 0.125], [1.0, 0.0, -0.25]])
_POINT = np.array([0.75, -1.25, 2.5, -0.5])


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box"])
def test_point_through_a_constant_matmul_holds_the_exact_image(tmp_path,
                                                               kind):
    """A point (every coordinate of zero width: a Star with no predicate
    variable, a Zono whose generators are zero, a Box with lb == ub) through
    X @ W on the ONNX MatMul route. The exact image, p @ W in fractions
    (dyadic data, so float32 and float64 are exact), must be a member of a
    returned set in exact arithmetic, and so must the onnxruntime output.
    Catches an affine map that drops the centre or loses the image of a set
    with no generator."""
    path = save_onnx(tmp_path, [helper.make_node("MatMul", ["X", "W"], ["Y"])],
                     [1, 4], [1, 3], inits={"W": _POINT_W.astype(np.float32)},
                     name=f"point-{kind}")
    what = f"a point through X @ W on a {kind}"
    sets = reach_model(load_onnx(path), flat_set(kind, _POINT, _POINT),
                       "approx", what)
    exact = _times(_fractions(_POINT), _POINT_W)
    assert any(rational.member(rational.star(s), exact) for s in sets), (
        f"{what}: the exact image {[float(v) for v in exact]} is in no "
        f"returned set")
    assert_covered(sets, replay_onnx(path, [_POINT], [1, 4]), what)

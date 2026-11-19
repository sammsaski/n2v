"""Soundness cells for the ONNX function nodes the dispatcher routes by the
function they wrap.

onnx2torch converts the ONNX nodes Abs, Acos, Asin, Atan, Cos, Exp, Log,
Sign, Sin, Tan and Tanh to one module class, ``OnnxFunction``, which holds
the torch function it applies; Sigmoid becomes ``nn.Sigmoid``. The
dispatcher (``n2v/nn/layer_ops/dispatcher.py``) tells the function nodes
apart by the wrapped function, in its Star, Zono and Box lanes:
``_is_sigmoid_layer`` (nn.Sigmoid or an ``OnnxFunction`` wrapping
torch.sigmoid, a form no converter makes and a hand-built module does),
``_is_tanh_layer`` (nn.Tanh or torch.tanh), ``_trig_kind`` (torch.sin,
torch.cos) and ``_is_sign_layer`` (torch.sign). The Hexatope and Octatope
lanes have no function branch. Every other function node falls through to
the refusal of its lane, ``NotImplementedError``. The kernels behind the
branches have their own files (Sign, Sin and Cos, Sigmoid, Tanh); the cells
here check the routing: each node reaches its own kernel with every input
set, and a node with no kernel, or on a set type with no rule, is refused.

Cells:

- The routed nodes Sign, Tanh, Sin, Cos and the function-node form of
  Sigmoid on every set type through ``reach_layer``: answered with every
  true output held on a Star, ImageStar, Zono, ImageZono and Box, refused on
  a Hexatope and an Octatope; three sets of different sizes at once on a
  Star, a Zono and a Box; Sign also under ``exact`` on a Star.
- The chain affine, node, affine through ``reach_pytorch_model`` on a model
  loaded with ``load_onnx`` (Gemm on flat sets, per-pixel Mul and Add on a
  (2, 3, 5) image), with the same promise. The Sigmoid chain uses the ONNX
  Sigmoid node, which loads as ``nn.Sigmoid``; its image cells are the ONNX
  image route of the Sigmoid rows. Under ``exact`` the model entry refuses a
  Zono, an ImageZono, a Box, a Hexatope and an Octatope before the walk.
- The same chain as a torch model that holds the ``OnnxFunction``: torch.fx
  traces through it to a call of the torch function, which the walker maps
  to ``nn.Tanh`` for Tanh and refuses for the others.
- The function nodes the dispatcher does not know (Abs, Acos, Asin, Atan,
  Exp, Log, Tan) through both entry points: ``NotImplementedError``. All
  seven on a Star, a Zono and a Box, where each passes the four tests above
  before the refusal (a test that also matched a look-alike, atan for tanh
  or asin for sine, would answer it); Abs on each other set type, whose lane
  takes the same path for every function.
- A known defect, a strict expected failure beside passing neighbours:
  ``method='exact'`` through ``reach_layer`` on a Box, which the lane
  answers with the interval rule for Sin and Cos, Tanh and Sigmoid (REACH-8).
  The same defect on the Zono lane and for Sign, and the mixed-family lists
  of the routed nodes (REACH-6), are marked in the kernels' own files.

Oracle: the true outputs of 128 sample points (the corners and the quarter
grid of the input box), each a member of a returned set by the LP on the
set's own rows. They come from onnxruntime on a float64 copy of the graph
(input, constants and output DOUBLE) for Sign, Tanh, Sin and Sigmoid, and
from torch in float64 for Cos, for which onnxruntime 1.26 has no float64
kernel; a float32 replay would put the reference a float32 rounding away
from the real-valued function that n2v encloses (DESIGN section 1). The
inputs are dyadic, cross zero, and hold a coordinate of zero width at 0
(Sign is 0 there) and one at a nonzero value; their ranges keep sine and
cosine, and tanh and the sigmoid, apart, so a node sent to the wrong kernel
misses outputs. The containment proofs of the smooth kernels are in the
kernels' own files.

Layout: flat sets are in the tensor's row-major order; an image set stores
height, width, channel, and image outputs are compared with ``out_chw``,
which also checks the stored height, width and channel count.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper
from onnx2torch.node_converters.functions import OnnxFunction

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.utils.model_loader import load_onnx
from tests.cell_support import (FLAT_KINDS, IMAGE_KINDS, assert_covered,
                                flat_set, image_set, pixel_box, reach_model,
                                reach_one_layer, replay_onnx, replay_torch,
                                samples, save_onnx, split_cover)
from tests.known_bugs import DID_NOT_RAISE, known_bug

ROUTED = ("Sign", "Tanh", "Sin", "Cos", "Sigmoid")
UNKNOWN = ("Abs", "Acos", "Asin", "Atan", "Exp", "Log", "Tan")
KINDS = FLAT_KINDS + IMAGE_KINDS
# The Hexatope and Octatope lanes have no branch for any function node.
NO_RULE = ("Hexatope", "Octatope")
_ORT64 = ("Sign", "Tanh", "Sin", "Sigmoid")

# x0 crosses zero, x1 is positive (sine and cosine are apart there: sin in
# [0.25, 0.69], cos in [0.73, 0.97]), x2 negative (tanh below -0.98, the
# sigmoid above 0), x3 has zero width at 0, x4 zero width at 3/2.
_LO = np.array([-1.5, 0.25, -3.0, 0.0, 1.5])
_HI = np.array([1.25, 0.75, -2.5, 0.0, 1.5])
_N = _LO.size

_C, _H, _W = 2, 3, 5


def _image_shape(name, kind):
    """(C, H, W) of the image cell. Sign on an ImageZono returns a flat Zono
    in the stored (height, width, channel) order, a known defect with its own
    marked test (registry ACT-3,
    ``contracts/test_reach_coverage.py::test_rounding_op_on_an_image_set_keeps_the_pixel_order``);
    that cell uses one channel, for which the stored order is the tensor's,
    so it checks the routing without depending on that defect."""
    return (1, _H, _W) if (name, kind) == ("Sign", "ImageZono") else (_C, _H, _W)


_TORCH = {"Sign": torch.sign, "Tanh": torch.tanh, "Sin": torch.sin,
          "Cos": torch.cos, "Sigmoid": torch.sigmoid}


def _layer(name):
    """The module a node of ``name`` is reached as through ``reach_layer``:
    the ``OnnxFunction`` the converter makes, and for Sigmoid the
    function-node form the dispatcher also accepts."""
    if name == "Sigmoid":
        module = OnnxFunction("Tanh")
        module.function = torch.sigmoid   # the function-node form of Sigmoid
        return module
    return OnnxFunction(name)


def _node(name, x, y):
    return helper.make_node(name, [x], [y])


# Constants of the chains, dyadic: Gemm 5 -> 4 and 4 -> 3 for flat sets;
# per-pixel scale and shift in front of and behind the node for an image.
_W1 = np.array([[1, -2, 1, 0, 2], [2, 1, 0, -1, 0], [-1, 1, 2, 1, -1],
                [0, -1, 1, 2, 1]]) / 4
_B1 = np.array([0.25, -0.5, 0.125, 0.75])
_W2 = np.array([[1, 0, -1, 2], [-2, 1, 1, 0], [0, 2, 1, -1]]) / 2
_B2 = np.array([0.5, -0.25, 0.0])
def _image_constants(chw):
    """Per-pixel scale and shift in front of (m1, a1) and behind (m2, a2) the
    node, (1, C, H, W) arrays of mixed signs."""
    g = np.arange(int(np.prod(chw))).reshape(1, *chw)
    return dict(m1=((g % 4 + 1) / 2) * (-1.0) ** (g // 7), a1=(g % 3 - 1) / 4,
                m2=(g % 3 + 1) / 4 * (-1.0) ** g, a2=(g % 5 - 2) / 8)


def _chain(name, chw):
    """Nodes, initializers (float64) and shapes of affine, node, affine; an
    image of shape ``chw``, or a flat vector when ``chw`` is None."""
    if chw is not None:
        nodes = [helper.make_node("Mul", ["X", "m1"], ["P"]),
                 helper.make_node("Add", ["P", "a1"], ["A"]), _node(name, "A", "Z"),
                 helper.make_node("Mul", ["Z", "m2"], ["U"]),
                 helper.make_node("Add", ["U", "a2"], ["Y"])]
        return nodes, _image_constants(chw), [1, *chw], [1, *chw]
    nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["A"], transB=1),
             _node(name, "A", "Z"),
             helper.make_node("Gemm", ["Z", "W2", "b2"], ["Y"], transB=1)]
    return nodes, dict(W1=_W1, b1=_B1, W2=_W2, b2=_B2), [1, _N], [1, 3]


def _torch_chain(name, chw, x):
    """The chain of :func:`_chain` in torch float64, written from its
    definition (no n2v, no converter)."""
    f = _TORCH[name]
    if chw is not None:
        k = {key: torch.as_tensor(v, dtype=torch.float64)
             for key, v in _image_constants(chw).items()}
        return f(x * k["m1"] + k["a1"]) * k["m2"] + k["a2"]
    a = F.linear(x, torch.as_tensor(_W1), torch.as_tensor(_B1))
    return F.linear(f(a), torch.as_tensor(_W2), torch.as_tensor(_B2))


def _true_outputs(tmp_path, name, nodes, inits, in_shape, out_shape, points,
                  torch_fn):
    """The outputs of the graph on ``points`` in float64: onnxruntime on the
    DOUBLE copy of the graph where it has the kernels, else ``torch_fn``."""
    if name in _ORT64:
        path = save_onnx(tmp_path, nodes, in_shape, out_shape, inits=inits,
                         name=f"ref_{name}", in_type="DOUBLE",
                         out_type="DOUBLE")
        return replay_onnx(path, points, in_shape)
    with torch.no_grad():
        return [torch_fn(torch.as_tensor(np.asarray(x, np.float64)
                                         .reshape(in_shape)))
                .numpy().reshape(-1) for x in points]


def _check(sets, outputs, chw, what):
    if chw is not None:
        assert_covered(sets, outputs, what, out_chw=chw)
    else:
        assert_covered(sets, outputs, what)


def _cells(entries):
    """(name, kind, method, inputs) over the routed nodes and the set types:
    a Star also with ``method='exact'`` for Sign, the piecewise-constant node
    (the smooth ones have no exact method; REACH-4 is marked in
    ``tests/contracts/test_reach_coverage.py``), three sets of different
    sizes at once for Star, Zono and Box at the layer entry, and at the
    model entry ``exact`` on each set type the entry refuses it for."""
    out = []
    for name in ROUTED:
        for kind in KINDS:
            out.append((name, kind, "approx", "one set"))
        if name == "Sign":
            out.append((name, "Star", "exact", "one set"))
        if entries == "layer":
            out += [(name, kind, "approx", "several sets")
                    for kind in ("Star", "Zono", "Box")]
    if entries == "model":
        out += [("Sign", kind, "exact", "one set")
                for kind in ("Zono", "ImageZono", "Box") + NO_RULE]
    return out


# --------------------------------------------------------------------------- #
# Routed function nodes                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,kind,method,inputs", _cells("layer"))
def test_routed_function_node_layer_refuses_or_holds_every_output(
        tmp_path, name, kind, method, inputs):
    """A routed function node through ``reach_layer``: refused with
    ``NotImplementedError`` on a Hexatope and an Octatope, whose lanes have no
    rule for it; on every other set type answered, with every true output of
    the node a member of the result. Catches a node sent to the kernel of
    another function (Sin as Cos, Cos as Sin, Tanh as the sigmoid, the
    sigmoid as tanh, a branch that calls the other S-curve kernel), a lane
    that passes on only the first of several input sets, a Hexatope or
    Octatope lane that answers a function node (as the identity, say), and a
    supported node that is no longer routed (refused)."""
    chw = _image_shape(name, kind) if kind in IMAGE_KINDS else None
    what = f"{name} on {inputs} of {kind} through reach_layer ({method})"
    if chw is not None:
        lo, hi = pixel_box(*chw, seed=31)
        sets = [image_set(kind, lo, hi)]
        shape = [1, *chw]
    else:
        lo, hi = _LO, _HI
        sets = ([flat_set(kind, lo, hi)] if inputs == "one set"
                else split_cover(kind, lo, hi, parts=3))
        shape = [1, _N]
    if kind in NO_RULE:
        with pytest.raises(NotImplementedError):
            reach_one_layer(_layer(name), sets, method, what)
        return
    result = reach_one_layer(_layer(name), sets, method, what)
    outputs = _true_outputs(tmp_path, name, [_node(name, "X", "Y")], {}, shape,
                            shape, samples(lo, hi), _TORCH[name])
    _check(result, outputs, chw, what)


@pytest.mark.parametrize("name,kind,method,inputs", _cells("model"))
def test_routed_function_node_in_a_chain_refuses_or_holds_every_output(
        tmp_path, name, kind, method, inputs):
    """Affine, the node, affine through ``reach_pytorch_model`` on the float32
    graph loaded with ``load_onnx`` (an image with ``input_shape``): the
    affine map behind the node consumes the rows the node's kernel emits.
    Refused on a Hexatope and an Octatope (the node has no rule there) and
    under ``exact`` on a set type with no exact method (the entry raises
    ``ValueError`` today, the planned ``UnsupportedError`` subclasses
    ``NotImplementedError``); answered with every true output held
    elsewhere. Catches the routing faults of the layer cell through the graph
    walker, a walker that passes a function node through or applies it to
    another node's sets, and a relaxation whose rows are wrong while its box
    is right."""
    chw = _image_shape(name, kind) if kind in IMAGE_KINDS else None
    nodes, inits, in_shape, out_shape = _chain(name, chw)
    path = save_onnx(tmp_path, nodes, in_shape, out_shape, name=f"m_{name}",
                     inits={k: v.astype(np.float32) for k, v in inits.items()})
    if chw is not None:
        lo, hi = pixel_box(*chw, seed=31)
        input_set, options = image_set(kind, lo, hi), dict(input_shape=chw)
    else:
        lo, hi = _LO, _HI
        input_set, options = flat_set(kind, lo, hi), {}
    what = f"affine, {name}, affine on {kind} through the model ({method})"
    if method == "exact" and kind not in ("Star", "ImageStar"):
        with pytest.raises((NotImplementedError, ValueError)):
            reach_model(load_onnx(path), input_set, method, what, **options)
        return
    if kind in NO_RULE:
        with pytest.raises(NotImplementedError):
            reach_model(load_onnx(path), input_set, method, what, **options)
        return
    sets = reach_model(load_onnx(path), input_set, method, what, **options)
    outputs = _true_outputs(tmp_path, name, nodes, inits, in_shape, out_shape,
                            samples(lo, hi),
                            lambda x: _torch_chain(name, chw, x))
    _check(sets, outputs, chw, what)


class _Holder(nn.Module):
    """Linear, an ``OnnxFunction``, Linear: the chain of the flat cells as a
    torch model that holds the function node as a submodule."""

    def __init__(self, name):
        super().__init__()
        self.first = nn.Linear(_N, 4, dtype=torch.float64)
        self.node = _layer(name)
        self.last = nn.Linear(4, 3, dtype=torch.float64)
        with torch.no_grad():
            self.first.weight.copy_(torch.as_tensor(_W1))
            self.first.bias.copy_(torch.as_tensor(_B1))
            self.last.weight.copy_(torch.as_tensor(_W2))
            self.last.bias.copy_(torch.as_tensor(_B2))

    def forward(self, x):
        return self.last(self.node(self.first(x)))


@pytest.mark.parametrize("name", ["Tanh", "Sigmoid", "Sin", "Abs"])
def test_function_node_in_a_torch_model_refuses_or_holds_every_output(name):
    """The chain as a torch model holding the function node, on a Star
    through ``reach_pytorch_model``. torch.fx traces through the
    ``OnnxFunction`` to a call of its torch function: the walker maps
    torch.tanh and torch.sigmoid to ``nn.Tanh`` and ``nn.Sigmoid`` and
    refuses torch.sin (Cos takes the same branch) and torch.abs with
    ``NotImplementedError``. The promise: refused, or every torch float64
    output held. Catches a walker that passes such a call through as the
    identity, and the two S-curve functions mapped to each other's module.
    Sign's call is held by ``test_soundness_sign.py`` (torch.sign and
    Tensor.sign)."""
    model = _Holder(name)
    what = f"Linear, {name}, Linear as a torch model on a Star"
    try:
        sets = reach_model(model, flat_set("Star", _LO, _HI), "approx", what)
    except NotImplementedError:
        return
    assert_covered(sets, replay_torch(model, samples(_LO, _HI), (1, _N)), what)


# --------------------------------------------------------------------------- #
# Function nodes the dispatcher does not know                                   #
# --------------------------------------------------------------------------- #
_ILO, _IHI = pixel_box(_C, _H, _W, seed=31)

# Every unknown function on the lanes that test the wrapped function (Star,
# Zono, Box); Abs on the others, whose lane takes one path for any function.
_UNKNOWN_CELLS = ([(name, kind) for kind in ("Box", "Star", "Zono")
                   for name in UNKNOWN]
                  + [("Abs", kind) for kind in NO_RULE + IMAGE_KINDS])


def _any_set(kind):
    if kind in IMAGE_KINDS:
        return image_set(kind, _ILO, _IHI)
    return flat_set(kind, _LO, _HI)


@pytest.mark.parametrize("name,kind", _UNKNOWN_CELLS)
def test_unknown_function_node_layer_is_refused(name, kind):
    """A function node with no kernel (Abs, Acos, Asin, Atan, Exp, Log, Tan)
    through ``reach_layer`` must raise ``NotImplementedError``. Catches a
    function-node test that accepts any wrapped function (an Abs answered by
    the Sign kernel, whose outputs lie in {-1, 0, 1}) or a look-alike (Atan
    by the Tanh kernel, Asin by the sine rule), and a router that returns the
    input for a function it does not know."""
    with pytest.raises(NotImplementedError):
        reach_one_layer(OnnxFunction(name), [_any_set(kind)], "approx",
                        f"{name} on {kind} through reach_layer")


@pytest.mark.parametrize("name,kind", _UNKNOWN_CELLS)
def test_unknown_function_node_in_a_model_is_refused(tmp_path, name, kind):
    """The same node as the only node of an ONNX graph, loaded with
    ``load_onnx`` and reached with ``reach_pytorch_model`` (an image set with
    ``input_shape``): the call must raise ``NotImplementedError``, not return
    a set for a function n2v has no kernel for."""
    image = kind in IMAGE_KINDS
    shape = [1, _C, _H, _W] if image else [1, _N]
    path = save_onnx(tmp_path, [_node(name, "X", "Y")], shape, shape,
                     name=f"u_{name}")
    options = dict(input_shape=(_C, _H, _W)) if image else {}
    with pytest.raises(NotImplementedError):
        reach_model(load_onnx(path), _any_set(kind), "approx",
                    f"{name} on {kind} through the model", **options)


# --------------------------------------------------------------------------- #
# Known defects of the routing                                                  #
# --------------------------------------------------------------------------- #
_EXACT_ANSWERED = known_bug(
    18, "reach_layer with method='exact' on a Box: the Sin and Cos, Tanh and "
        "Sigmoid branches of the Box lane do not read the method and return "
        "the interval rule under the name exact, although these functions "
        "have no exact rule on any set type (REACH-4); the model entry "
        "refuses exact on a Box before the walk (REACH-8, the Box lane)",
    raises=DID_NOT_RAISE)


@pytest.mark.parametrize("name,kind", [
    pytest.param("Sin", "Box", marks=_EXACT_ANSWERED),
    pytest.param("Tanh", "Box", marks=_EXACT_ANSWERED),
    pytest.param("Sigmoid", "Box", marks=_EXACT_ANSWERED),
    ("Abs", "Zono"),
    ("Abs", "Box"),
    ("Sin", "Hexatope"),
])
def test_function_node_layer_under_exact_without_an_exact_rule_is_refused(
        name, kind):
    """``method='exact'`` through ``reach_layer`` on a set type with no exact
    rule for the node must be refused, as the model entry refuses it before
    the walk (the chain cells ``[Sign-Zono-exact-one set]`` and the like).
    Sin and Cos (one branch), Tanh and Sigmoid are answered by the Box lane
    with the interval rule; none of them has an exact rule on any set type
    (REACH-4), so ``exact`` must be refused whatever the input. The Zono
    branches are marked in the kernels' files
    (``test_trig_exact_on_a_zono_is_refused``,
    ``test_tanh_exact_on_a_zono_is_refused``,
    ``test_sigmoid_exact_on_a_zono_is_refused``; an ImageZono takes the Zono
    branch), and both Sign branches, whose interval [-1, 1] holds values no
    input reaches, in ``test_soundness_sign.py``
    (``test_sign_exact_on_a_zono_or_a_box_refuses_or_is_the_image``).
    Neighbours that pass today with the same assertion: a node without a
    kernel on the Zono and Box lanes, and Sin on a Hexatope. The call goes
    to ``reach_layer`` directly, so the marker absorbs only the missing
    refusal."""
    with pytest.raises(NotImplementedError):
        reach_layer(_layer(name), [flat_set(kind, _LO, _HI)], "exact")

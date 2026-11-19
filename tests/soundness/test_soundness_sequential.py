"""Soundness cells for ``nn.Sequential`` given to the dispatcher as one
layer, and for the dispatcher's routing of a layer type to its kernel.

What is covered. ``reach_layer`` (``n2v/nn/layer_ops/dispatcher.py``) answers
an ``nn.Sequential`` by walking its children, each through ``reach_layer``
again with the same method and options, in every one of its five routers
(Star and ImageStar, Zono and ImageZono, Box, Hexatope, Octatope). The cells
give it Sequentials of several children (affine layers of equal and of
different widths, a ReLU, a pass-through layer, a nested Sequential, one
child, no child) on every set type and method, with four input sets at once;
image sets through Flatten and Linear; options handed to a child (a
relaxation factor of 0, precomputed bounds); a subclass of ``nn.Sequential``
with a forward of its own and one with the stock forward; and an empty
intermediate result. AvgPool2d, whose routing to the average-pooling kernel
(not the max-pooling one) is checked alone, inside a Sequential and in a
model. Last, the refusals of the routers: a layer no router knows, alone
and inside a Sequential, and an input that is no set.

Routes. A torch model never reaches the Sequential branch:
``torch.fx.symbolic_trace`` keeps no ``nn.Sequential`` as one node, so the
model cells here check that the traced walk of a nested Sequential is the
same composition. An ONNX file reaches it where the vendored onnx2torch
loads a Conv or MaxPool whose pads differ at the two ends of an axis as
``nn.Sequential(pad, op)``; those cells are in the Conv2d, MaxPool2d and Pad
files.

Oracle. The exact image of the Sequential is computed child by child in
rationals (``tests/_oracles/rational.py``): ``affine`` for a Linear layer
with the weights it stores, ``relu`` for a ReLU, nothing for a pass-through,
and for AvgPool2d the pooling matrix read off torch's ``avg_pool2d`` in
float64 on unit tensors (exact: every window has four entries, so every
weight is 1/4). Every vertex of every exact piece must lie in one returned
set with no slack (``escapes``, ``straddlers``), and for the exact method
every vertex of every returned set must have a preimage (``unreached``). On
top, member points of the inputs are replayed through the Sequential in
torch float64 (or the model on onnxruntime) and must be members by the LP
oracle on the sets' own rows (``assert_covered``, slack ``LP_TOL``); the
option, subclass and empty-result cells use that replay alone. Weights,
biases and inputs are dyadic.

Layout. Flat sets are in the tensor's row-major order. Image inputs are
(2, 3, 5) images built in (channel, height, width) order and stored in
(height, width, channel) order by the set; an image output (AvgPool2d alone)
is compared with ``out_chw`` and its exact pieces are permuted to the stored
order.

Defects of today's library marked here (strict expected failures):

- REACH-9 (PLAN step 7) on the Sequential branch of every router: a subclass
  of ``nn.Sequential`` whose forward is not the stock one is walked as if it
  were; ``test_a_sequential_subclass_refuses_or_holds_every_output``.
- REACH-10 (PLAN step 7): ``precomputed_bounds`` given for a Sequential reach
  every child, and a second ReLU reads them as its own;
  ``test_precomputed_bounds_given_to_a_sequential_refuse_or_hold_every_output``.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import Star
from tests._oracles import rational as R
from tests.cell_support import (assert_covered, flat_box, flat_set,
                                hwc_index, reach_model, reach_one_layer,
                                replay_torch, samples)
from tests.known_bugs import known_bug
from tests.soundness.test_soundness_flatten import _SHAPE, _picking_linear
from tests.soundness.test_soundness_identity_ops import (_flat_inputs,
                                                         _image_inputs)
from tests.soundness.test_soundness_linear import (_N_IN,
                                                   _assert_holds_exact_image,
                                                   _dyadic_linear)

_FLAT_KIND_METHODS = (("Box", "approx"), ("Zono", "approx"),
                      ("Hexatope", "approx"), ("Octatope", "approx"),
                      ("Star", "approx"), ("Star", "exact"))
_IMAGE_KIND_METHODS = (("ImageStar", "exact"), ("ImageStar", "approx"),
                       ("ImageZono", "approx"))
# One flat set type per router of the dispatcher (an image set goes to the
# router of its flat family).
_ROUTER_KINDS = ("Box", "Zono", "Hexatope", "Octatope", "Star")


# --------------------------------------------------------------------------- #
# The exact image of a Sequential, child by child                               #
# --------------------------------------------------------------------------- #
def _pool_matrix(layer, shape):
    """The matrix of a linear pooling layer over the row-major entries of an
    input of ``shape``, read off the layer's output on unit tensors in
    float64."""
    n = int(np.prod(shape))
    columns = []
    with torch.no_grad():
        for j in range(n):
            e = torch.zeros(n, dtype=torch.float64)
            e[j] = 1.0
            columns.append(layer(e.reshape(shape)).reshape(-1).numpy())
    return np.stack(columns, axis=1)


def _exact(module, pieces, shape):
    """``(pieces, shape)``: the exact image of ``pieces`` (row-major over a
    tensor of ``shape``) under ``module``, and the output tensor's shape."""
    if isinstance(module, nn.Sequential):
        for child in module:
            pieces, shape = _exact(child, pieces, shape)
        return pieces, shape
    if isinstance(module, nn.Linear):
        W = module.weight.detach().double().numpy()
        b = (None if module.bias is None
             else module.bias.detach().double().numpy().tolist())
        return R.affine(pieces, W.tolist(), b), (1, W.shape[0])
    if isinstance(module, nn.ReLU):
        return R.relu(pieces), shape
    if isinstance(module, (nn.Identity, nn.Dropout, nn.Dropout2d)):
        return pieces, shape
    if isinstance(module, nn.Flatten):
        return pieces, (1, int(np.prod(shape[1:])))
    if isinstance(module, nn.AvgPool2d):
        P = _pool_matrix(module.double(), shape)
        out_shape = tuple(module(torch.zeros(shape, dtype=torch.float64))
                          .shape)
        return R.affine(pieces, P.tolist()), out_shape
    raise ValueError(f"no exact rule for {type(module).__name__}")


def _to_stored(pieces, channels, height, width):
    """Pieces over a (C, H, W) tensor mapped to the (H, W, C) order an image
    set stores."""
    order = hwc_index(channels, height, width)
    P = np.zeros((order.size, order.size))
    P[np.arange(order.size), order] = 1.0
    return R.affine(pieces, P.tolist())


# --------------------------------------------------------------------------- #
# Sequentials of flat layers                                                    #
# --------------------------------------------------------------------------- #
def _sequential(shape, relu):
    """The Sequential named ``shape`` over inputs of four entries. ``relu``
    puts a ReLU in the middle of "narrowing"; otherwise an Identity sits
    there."""
    lin = _dyadic_linear
    if shape == "square":            # equal widths: a wrong walk still fits
        return nn.Sequential(lin(_N_IN, 4, True, seed=51), nn.Dropout(p=0.5),
                             lin(4, 4, False, seed=52))
    if shape == "narrowing":
        return nn.Sequential(lin(_N_IN, 3, True, seed=53),
                             nn.ReLU() if relu else nn.Identity(),
                             lin(3, 2, False, seed=54))
    if shape == "nested":
        return nn.Sequential(lin(_N_IN, 4, True, seed=55),
                             nn.Sequential(nn.Identity(), nn.Flatten(),
                                           lin(4, 4, True, seed=56)),
                             lin(4, 4, False, seed=57))
    if shape == "one":
        return nn.Sequential(lin(_N_IN, 3, True, seed=58))
    if shape == "empty":
        return nn.Sequential()
    raise ValueError(shape)


_SHAPES = ("square", "narrowing", "nested", "one", "empty")


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("kind,method", _FLAT_KIND_METHODS)
def test_sequential_as_one_layer_holds_the_exact_image(kind, method, shape):
    """``reach_layer(nn.Sequential(...))`` with four flat input sets (one with
    a binding row, three of a split) must hold the image of the composition
    of its children. Catches a walk that applies every child to the original
    input (the result is then the image under the last child: the "square"
    and "nested" Sequentials keep the widths equal so it fits), one that
    returns after the first child or skips the last, one that walks the
    children in another order, one that walks the first input set only, one
    that skips a nested Sequential (it holds a Flatten of the flat sets too,
    so the Flatten branch of every router sees four sets), and one that does
    not pass the method on (the exact ReLU in "narrowing" on a Star must give
    no point outside the image). The ReLU is used where its relaxation is
    exact or padded (Star, Box); the Zono, Hexatope and Octatope ReLU kernels
    round (registry ROUND-10, HEXOCT-4), so they get an Identity there."""
    model = _sequential(shape, relu=kind in ("Star", "Box")).eval()
    sets, pieces, points = _flat_inputs(kind)
    what = f"Sequential '{shape}' on {len(sets)} x {kind} ({method})"
    result = reach_one_layer(model, sets, method, what)
    assert all(type(s) is type(sets[0]) for s in result), (
        f"{what}: returned {[type(s).__name__ for s in result]}")
    exact, _ = _exact(model, pieces, (1, _N_IN))
    _assert_holds_exact_image(result, exact, what, exact=(method == "exact"))
    assert_covered(result, replay_torch(model, points, (1, _N_IN)), what)


@pytest.mark.parametrize("kind,method", _FLAT_KIND_METHODS)
def test_nested_sequential_through_the_model_entry(kind, method):
    """The nested Sequential as a model, through ``reach_pytorch_model``,
    which walks the traced graph instead of the dispatcher's Sequential
    branch: the same exact image must come out, so the two walks agree with
    the network and not merely with each other."""
    model = _sequential("nested", relu=False).eval()
    sets, pieces, points = _flat_inputs(kind)
    what = f"nested Sequential model on {kind} ({method})"
    result = reach_model(model, sets, method, what)
    exact, _ = _exact(model, pieces, (1, _N_IN))
    _assert_holds_exact_image(result, exact, what, exact=(method == "exact"))
    assert_covered(result, replay_torch(model, points, (1, _N_IN)), what)


# --------------------------------------------------------------------------- #
# Options handed to the children                                                #
# --------------------------------------------------------------------------- #
def test_relax_factor_zero_reaches_the_relu_inside_a_sequential():
    """``reach_layer(nn.Sequential(Linear, ReLU, Linear), sets, 'approx',
    relax_factor=0)`` on four flat Stars (one with a binding row, three of a
    split). At relaxation factor 0 the approx ReLU is the exact reach (its
    docstring), so the result must be the exact image of the composition in
    both directions. Catches a walk that does not hand its options to the
    children, or that replaces the factor 0 by its default: the ReLU then
    relaxes, and the returned sets hold points that no input reaches."""
    model = _sequential("narrowing", relu=True).eval()
    sets, pieces, points = _flat_inputs("Star")
    what = f"Sequential 'narrowing' on {len(sets)} x Star (approx, " \
           f"relax_factor=0)"
    result = reach_one_layer(model, sets, "approx", what, relax_factor=0.0)
    exact, _ = _exact(model, pieces, (1, _N_IN))
    _assert_holds_exact_image(result, exact, what, exact=True)
    assert_covered(result, replay_torch(model, points, (1, _N_IN)), what)


_BOUNDS_HANDED_TO_EVERY_CHILD = known_bug(
    7, "reach_layer of an nn.Sequential hands the precomputed_bounds option "
       "to every child: a second ReLU reads the bounds of the Sequential's "
       "input as its own pre-activation bounds, a neuron whose real range "
       "reaches above 0 is taken as inactive and set to 0, and true outputs "
       "lie outside the result")


def _bounds_cell(children):
    """``(model, lo, hi)``: the box x0 in [-1, 1], x1 in [-2, -1/2], and the
    Sequential ``ReLU`` or ``ReLU, Linear, ReLU`` whose Linear maps
    (r0, r1) to (r0 + 1/4, 2 r0 - 1). The real input range of the second
    ReLU is [1/4, 5/4] x [-1, 1]; the bounds of the Sequential's input, read
    as its own, cut the second neuron to [-1, -1/2]."""
    lo, hi = np.array([-1.0, -2.0]), np.array([1.0, -0.5])
    if children == "ReLU":
        return nn.Sequential(nn.ReLU()).eval(), lo, hi
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
        linear.bias.copy_(torch.tensor([0.25, -1.0]))
    return nn.Sequential(nn.ReLU(), linear, nn.ReLU()).eval(), lo, hi


@pytest.mark.parametrize("children,method", [
    pytest.param("ReLU", "approx", id="one-relu-approx"),
    pytest.param("ReLU", "exact", id="one-relu-exact"),
    pytest.param("ReLU-Linear-ReLU", "approx", id="two-relus-approx",
                 marks=_BOUNDS_HANDED_TO_EVERY_CHILD),
    pytest.param("ReLU-Linear-ReLU", "exact", id="two-relus-exact",
                 marks=_BOUNDS_HANDED_TO_EVERY_CHILD)])
def test_precomputed_bounds_given_to_a_sequential_refuse_or_hold_every_output(
        children, method):
    """``reach_layer(nn.Sequential(...), [star], method,
    precomputed_bounds=(lo, hi))`` on a flat Star with the true bounds of the
    Sequential's input, which are the pre-activation bounds of its first
    child, a ReLU. The option bounds a layer's input; a Sequential may hand
    them to the child whose input they bound, or refuse them, but every true
    output (torch float64 on 128 points, the corners among them) must be in
    the result. Catches bounds handed to a later child whose input they do
    not bound: the approx ReLU uses them as given and the exact one resets a
    neuron by them, so the second neuron of "ReLU, Linear, ReLU", whose real
    range is [-1, 1], is taken as inactive and every output above 0 there is
    lost. The one-ReLU Sequential, whose only child the bounds describe, is
    the passing neighbour."""
    model, lo, hi = _bounds_cell(children)
    what = (f"Sequential({children}) on a Star ({method}) with the "
            f"precomputed bounds of its input")
    try:
        result = reach_one_layer(
            model, [flat_set("Star", lo, hi)], method, what,
            precomputed_bounds=(lo.reshape(-1, 1), hi.reshape(-1, 1)))
    except (NotImplementedError, ValueError):
        return
    assert_covered(result, replay_torch(model, samples(lo, hi), (1, 2)), what)


# --------------------------------------------------------------------------- #
# A subclass of nn.Sequential                                                   #
# --------------------------------------------------------------------------- #
class _Residual(nn.Sequential):
    """A subclass of ``nn.Sequential`` with a forward of its own: its input
    plus the walk of its children (a residual block)."""

    def forward(self, x):
        return x + super().forward(x)


class _Block(nn.Sequential):
    """A subclass of ``nn.Sequential`` that keeps the stock forward."""


_SUBCLASS_FORWARD_IGNORED = known_bug(
    7, "reach_layer matches a subclass of nn.Sequential by isinstance and "
       "walks its children whatever its own forward computes: a residual "
       "block x + children(x) is answered as children(x), and the true "
       "outputs lie outside the result")


def _small_linear(seed):
    """Linear(4, 4) with a bias and dyadic weights in [-1/4, 1/4], steps of
    1/16: the children move the input by little, so in the residual block
    the input term, far from 0, decides where the output lies."""
    layer = _dyadic_linear(_N_IN, _N_IN, True, seed=seed)
    with torch.no_grad():
        layer.weight.mul_(0.25)
    return layer


def _subclass_cells():
    cells = []
    for kind in _ROUTER_KINDS:
        cells.append(pytest.param(kind, "own", id=f"{kind}-own-forward",
                                  marks=_SUBCLASS_FORWARD_IGNORED))
        cells.append(pytest.param(kind, "stock", id=f"{kind}-stock-forward"))
    return cells


@pytest.mark.parametrize("kind,forward", _subclass_cells())
def test_a_sequential_subclass_refuses_or_holds_every_output(kind, forward):
    """``reach_layer`` given a subclass of ``nn.Sequential`` of two Linear
    layers on a flat box far from 0 (entries from 56 to 80, intervals up to
    16 wide): one subclass computes x + children(x) (a residual block), the
    other keeps the stock forward. Each router may refuse a forward it
    cannot read, or must hold every true output of the module's own forward
    (torch float64 on 128 points, the corners among them). Catches a router
    that matches the subclass by ``isinstance`` and walks its children as if
    the forward were the stock one (REACH-9, on the Sequential branch
    of each router); the stock-forward subclass is the passing neighbour."""
    cls = _Residual if forward == "own" else _Block
    model = cls(_small_linear(63), _small_linear(64)).eval()
    lo, hi = flat_box(_N_IN, seed=2)
    lo, hi = 8 * lo + 64, 8 * hi + 64
    what = f"{cls.__name__}(Linear, Linear) on a {kind} (approx)"
    try:
        result = reach_one_layer(model, flat_set(kind, lo, hi), "approx",
                                 what)
    except NotImplementedError:
        return
    assert_covered(result, replay_torch(model, samples(lo, hi), (1, _N_IN)),
                   what)


# --------------------------------------------------------------------------- #
# An empty intermediate result                                                  #
# --------------------------------------------------------------------------- #
def test_an_empty_intermediate_result_ends_the_walk():
    """A Star whose predicate polytope is empty (a0 + a1 <= -3 on the box
    [-1, 1]^2), given to ``reach_layer`` of Sequential(Linear(2, 2), ReLU,
    Linear(2, 3)) under exact. The exact ReLU certifies the set empty and
    returns no set; the walk must stop there and return no set (DESIGN
    section 8: the loop over the nodes stops at the first empty result and
    returns it, and an empty result of a certified empty input keeps its
    proof). Catches a walk that carries on with the sets of the child before
    the empty result (it returns a set where none is due) and one that hands
    the empty list to the next child, which raises once ``reach_layer``
    refuses an empty list (registry EMPTY-2)."""
    first = nn.Linear(2, 2)
    with torch.no_grad():
        first.weight.copy_(torch.eye(2))
        first.bias.zero_()
    model = nn.Sequential(first, nn.ReLU(),
                          _dyadic_linear(2, 3, True, seed=61)).eval()
    empty = Star(np.hstack([np.zeros((2, 1)), np.eye(2)]),
                 np.array([[1.0, 1.0]]), np.array([[-3.0]]),
                 -np.ones((2, 1)), np.ones((2, 1)))
    result = reach_layer(model, [empty], "exact")
    assert isinstance(result, list) and not result, (
        f"an input certified empty by the exact ReLU gave "
        f"{[type(s).__name__ for s in result]} instead of no set")


# --------------------------------------------------------------------------- #
# Sequentials on image sets, and the routing of AvgPool2d                       #
# --------------------------------------------------------------------------- #
def _avgpool():
    """AvgPool2d with a 2 x 2 window and strides (1, 2): on a (2, 3, 5) image
    an output of (2, 2, 2), every weight 1/4."""
    return nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 2))


def _image_sequential(name):
    if name == "Dropout2d-Flatten-Linear":
        return nn.Sequential(nn.Dropout2d(p=0.5), nn.Flatten(),
                             _picking_linear())
    if name == "AvgPool-Flatten-Linear":
        return nn.Sequential(_avgpool(), nn.Flatten(),
                             _dyadic_linear(8, 3, True, seed=59))
    raise ValueError(name)


@pytest.mark.parametrize("name", ["Dropout2d-Flatten-Linear",
                                  "AvgPool-Flatten-Linear"])
@pytest.mark.parametrize("entry", ["reach_layer", "reach_pytorch_model"])
@pytest.mark.parametrize("kind,method", _IMAGE_KIND_METHODS)
def test_sequential_on_image_sets_holds_the_exact_image(kind, method, entry,
                                                        name):
    """A Sequential that starts on a (2, 3, 5) image set (a pass-through or
    an AvgPool2d, then Flatten, then Linear), given to ``reach_layer`` as one
    layer or to ``reach_pytorch_model`` as the model, with four image input
    sets. Catches the faults of the flat cell on the image routers, an
    AvgPool2d answered by another kernel (window maxima instead of means:
    the windows here are not constant), and a Flatten reading the pooled set
    in the stored order."""
    model = _image_sequential(name).eval()
    sets, pieces, points = _image_inputs(kind)
    what = f"Sequential {name} on {len(sets)} x {kind} ({method}) by {entry}"
    if entry == "reach_layer":
        result = reach_one_layer(model, sets, method, what)
    else:
        result = reach_model(model, sets, method, what)
    exact, _ = _exact(model, pieces, _SHAPE)
    _assert_holds_exact_image(result, exact, what, exact=(method == "exact"))
    assert_covered(result, replay_torch(model, points, _SHAPE), what)


@pytest.mark.parametrize("kind,method", _IMAGE_KIND_METHODS)
def test_avgpool_layer_is_answered_by_average_pooling(kind, method):
    """``reach_layer(nn.AvgPool2d)`` on four (2, 3, 5) image sets whose
    windows are not constant must return image sets of the pooled shape
    (2, 2, 2) holding the exact image of the average. Catches the routing of
    AvgPool2d to the max-pooling kernel (or any other kernel), on the Star
    and on the Zono router."""
    layer = _avgpool()
    sets, pieces, points = _image_inputs(kind)
    what = f"AvgPool2d on {len(sets)} x {kind} ({method})"
    result = reach_one_layer(layer, sets, method, what)
    exact, out_shape = _exact(layer, pieces, _SHAPE)
    _, channels, height, width = out_shape
    _assert_holds_exact_image(result, _to_stored(exact, channels, height,
                                                 width), what,
                              exact=(method == "exact"))
    assert_covered(result, replay_torch(layer, points, _SHAPE), what,
                   out_chw=(channels, height, width))


# --------------------------------------------------------------------------- #
# Refusals of the routers                                                       #
# --------------------------------------------------------------------------- #
class _Unknown(nn.Module):
    """A layer no router knows (it doubles its input)."""

    def forward(self, x):
        return 2.0 * x


def _one_set(kind):
    lo, hi = flat_box(_N_IN)
    return flat_set(kind, lo, hi)


@pytest.mark.parametrize("kind,method",
                         [(kind, "approx") for kind in _ROUTER_KINDS])
def test_a_layer_no_router_knows_is_refused(kind, method):
    """Every router ends in a refusal: a layer type it has no rule for must
    raise ``NotImplementedError``, never come back as the identity or as
    another layer's answer. One case per router: the last branch reads
    neither the method nor whether the set is an image set."""
    with pytest.raises(NotImplementedError):
        reach_layer(_Unknown(), [_one_set(kind)], method)


@pytest.mark.parametrize("kind", _ROUTER_KINDS)
def test_a_sequential_with_an_unknown_child_is_refused(kind):
    """A Sequential whose second child no router knows must raise, not return
    the image under the children before it."""
    model = nn.Sequential(_dyadic_linear(_N_IN, 4, True, seed=60), _Unknown())
    with pytest.raises(NotImplementedError):
        reach_layer(model, [_one_set(kind)], "approx")


@pytest.mark.parametrize("not_a_set", [
    pytest.param(np.zeros((_N_IN, 1)), id="ndarray")])
def test_an_input_that_is_no_set_is_a_type_error(not_a_set):
    """An input list whose first entry is no set of a supported type must
    raise ``TypeError``, whatever the layer."""
    with pytest.raises(TypeError):
        reach_layer(nn.Identity(), [not_a_set], "approx")

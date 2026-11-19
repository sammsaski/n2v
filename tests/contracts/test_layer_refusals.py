"""A layer handed directly to an image set answers only where torch answers.

An image set (``ImageStar``, ``ImageZono``) stands for the tensor
(1, C, H, W); here (C, H, W) = (2, 3, 4), 24 entries, stored in (height,
width, channel) order. A torch layer applied to that tensor either answers
or raises. When torch raises, the layer is not defined on that tensor, and
any set the library returns is the image of some other function: the
registry row AFFINE-5 measured up to 200 of 200 outputs outside under the
model's row-major reading, and an ``ImageStar`` that keeps the input's
height, width and channel count as the label of a different tensor.

The promise every cell checks, decided inside the test by running the torch
layer (a float64 copy in eval mode, ``replay_torch``) on member points of the
input set as (1, 2, 3, 4) tensors:

- torch raises: the library raises ``NotImplementedError`` (the planned
  ``UnsupportedError`` subclasses it) or ``ValueError`` (the planned
  ``ModelError`` subclasses it). Any other exception type is a crash and
  fails the cell; so does any return value, an empty list included.
- torch answers: the library raises one of those two types, or returns a
  non-empty list of sets that holds every torch output, read in the model's
  own (C, H, W) order: an image set must store the output's height, width
  and channel count and is compared through ``membership.hwc``; a flat set
  is compared with the output's row-major entries (``assert_covered``, LP
  on the set's own rows, slack ``LP_TOL``; the layers that answer have
  dyadic parameters, so the float64 reference is exact or within float64
  rounding).

Each layer is reached through ``reach_layer(layer, [set], method)`` and
through ``reach_pytorch_model`` without ``input_shape``, on an ``ImageStar``
with a binding constraint row that is not a box row (approx and exact) and on
an ``ImageZono`` (approx). The model handed to the model entry is a
``torch.fx`` graph with the layer as one ``call_module`` node, the form
``load_onnx`` builds (:func:`_graph_of`): a plain ``nn.Sequential(layer)``
is traced through the onnx2torch modules (Transpose, ReduceSum, Squeeze,
Unsqueeze) into ``call_function`` and ``call_method`` nodes that the graph
walk refuses for any axis, so the layer's own rule would never run.

Defects of today's library marked here:

- AFFINE-5 (PLAN step 7): Conv1d, Linear, BatchNorm1d, BatchNorm2d with a
  parameter count other than the channel count, MaxPool2d with a kernel
  larger than the image (on an ``ImageStar``) and ``Flatten(start_dim=4)``
  answer where torch refuses: one ``DID_NOT_RAISE`` marker per family;
  BatchNorm2d(1) crashes with ``IndexError`` instead and has its own marker.
- CONV-3 (PLAN step 23): a Conv2d, AvgPool2d or (on an ``ImageZono``)
  MaxPool2d kernel larger than the 3 by 4 image lets torch's ``RuntimeError``
  escape from the reach kernel instead of a refusal.

Refused today with ``ValueError``: Conv1d on an ``ImageStar`` whenever the
output count differs from 24 ("Non-square affine map changes dimensions"),
Conv1d(5, 2, k=1) ("not divisible by Conv1d in_channels 5"), Linear(24, 5)
on an ``ImageStar``, Linear(4, 3), Linear(4, 4) and Linear(7, 3) ("W has N
columns"), Conv2d and ConvTranspose2d with a channel mismatch ("Input has 2
channels"), ReduceSum over axis 4 ("axis 4 is out of range"). Refused with
``NotImplementedError``: Upsample linear and trilinear, Softmax(dim=4),
Transpose with a rank-3 perm, every Squeeze and Unsqueeze on an image set.
The passing neighbours of the markers are those refusals (same builder, same
torch decision) and the legal BatchNorm2d(2), which both answer and whose
result holds every torch output.

One configuration per code path of the kernel: Conv1d keeps a square map
(both image sets answer), a map that changes the entry count (the
``ImageStar`` refuses, the ``ImageZono`` answers) and a channel count that
does not divide the entries (both refuse); BatchNorm1d keeps a feature count
equal to the channel count and one equal to the entry count (the image-set
kernel reads the first C parameters whatever their number).
"""

import numpy as np
import pytest
import torch
import torch.fx as fx
import torch.nn as nn
from onnx2torch.node_converters.reduce import OnnxReduceSumStaticAxes
from onnx2torch.node_converters.squeeze import OnnxSqueezeStaticAxes
from onnx2torch.node_converters.transpose import OnnxTranspose
from onnx2torch.node_converters.unsqueeze import OnnxUnsqueezeStaticAxes

from tests.cell_support import (assert_covered, constrained_image_star,
                                image_set, pixel_box, reach_model,
                                reach_one_layer, replay_torch, samples)
from tests.known_bugs import DID_NOT_RAISE, known_bug

_C, _H, _W = 2, 3, 4
_TENSOR = (1, _C, _H, _W)
_REFUSALS = (NotImplementedError, ValueError)

_AFFINE_5 = ("answers when handed directly to an image set, where torch "
             "refuses the (1, C, H, W) tensor the set stands for; the answer "
             "is the image of another function (registry AFFINE-5)")
_CONV1D = known_bug(7, "Conv1d " + _AFFINE_5 + ": a flat Zono of the "
                    "stored-order reading, or an ImageStar labelled with the "
                    "input's shape", raises=DID_NOT_RAISE)
_LINEAR = known_bug(7, "Linear over all 24 entries " + _AFFINE_5,
                    raises=DID_NOT_RAISE)
_BATCHNORM1D = known_bug(7, "BatchNorm1d " + _AFFINE_5 + "; the result keeps "
                         "the input's image shape", raises=DID_NOT_RAISE)
_BATCHNORM_COUNT = known_bug(
    7, "BatchNorm2d with 3 or 4 features on a 2-channel image " + _AFFINE_5
    + "; the parameter count is never compared with the channel count",
    raises=DID_NOT_RAISE)
_MAXPOOL_EMPTY = known_bug(
    7, "MaxPool2d with a kernel larger than the 3 by 4 image " + _AFFINE_5
    + ": an ImageStar of height 0 and width 0", raises=DID_NOT_RAISE)
_FLATTEN_AXIS = known_bug(
    7, "Flatten(start_dim=4), an axis the rank-4 tensor does not have, "
    + _AFFINE_5, raises=DID_NOT_RAISE)
_BATCHNORM_ONE = known_bug(
    7, "BatchNorm2d(1) on a 2-channel image set crashes with IndexError (the "
    "channel loop indexes the one-entry parameters) instead of refusing "
    "(registry AFFINE-5)", raises=IndexError)
_KERNEL_TOO_LARGE = known_bug(
    23, "a Conv2d, AvgPool2d or (on an ImageZono) MaxPool2d kernel larger "
    "than the 3 by 4 image lets torch's RuntimeError ('Kernel size can't be "
    "greater than actual input size', 'Output size is too small') escape "
    "from the reach kernel instead of a clear refusal (registry row "
    "CONV-3)", raises=RuntimeError)


# --------------------------------------------------------------------------- #
# Layers, each built in float64 with dyadic parameters                         #
# --------------------------------------------------------------------------- #
def _dyadic_fill(layer, seed):
    """Every float parameter a multiple of 1/4 in [-1, 1], so that a float64
    replay of an answering layer is exact."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.as_tensor(rng.integers(-4, 5, size=tuple(p.shape))
                                    / 4.0, dtype=p.dtype))
    return layer


def _conv1d(cin, cout, k, pad=0):
    return lambda: _dyadic_fill(nn.Conv1d(cin, cout, k, padding=pad).double(),
                                cin * 10 + cout)


def _linear(fin, fout):
    return lambda: _dyadic_fill(nn.Linear(fin, fout).double(), fin + fout)


def _batchnorm(cls, features):
    """Eval-mode BatchNorm with dyadic running statistics chosen so that
    ``running_var + eps`` is a square of a dyadic number: scale and shift are
    exact."""
    def build():
        layer = cls(features, eps=2.0 ** -8).double()
        rng = np.random.default_rng(features)
        roots = rng.choice([0.5, 1.0, 2.0], size=features)
        with torch.no_grad():
            layer.running_mean.copy_(torch.as_tensor(
                rng.integers(-4, 5, size=features) / 4.0))
            layer.running_var.copy_(torch.as_tensor(roots ** 2 - 2.0 ** -8))
            layer.weight.copy_(torch.as_tensor(
                rng.choice([-1.5, -0.5, 0.75, 1.25], size=features)))
            layer.bias.copy_(torch.as_tensor(
                rng.integers(-4, 5, size=features) / 4.0))
        return layer.eval()
    return build


def _conv2d(cin, cout, k):
    return lambda: _dyadic_fill(nn.Conv2d(cin, cout, k).double(), cin + k)


def _conv_transpose2d(cin, cout, k):
    return lambda: _dyadic_fill(nn.ConvTranspose2d(cin, cout, k).double(),
                                cin + cout + k)


# id -> (builder, {set kind: marker or None})
def _same(marker):
    return {"ImageStar": marker, "ImageZono": marker}


def _zono_only(marker):
    return {"ImageStar": None, "ImageZono": marker}


_LAYERS = {
    # Conv1d: torch wants a rank-2 or rank-3 input.
    "Conv1d(2,3,k=2,pad=1)": (_conv1d(2, 3, 2, 1), _zono_only(_CONV1D)),
    "Conv1d(2,2,k=3,pad=1)": (_conv1d(2, 2, 3, 1), _same(_CONV1D)),
    "Conv1d(5,2,k=1)": (_conv1d(5, 2, 1), _same(None)),
    # Linear: torch maps the last axis (width 4).
    "Linear(24,5)": (_linear(24, 5), _zono_only(_LINEAR)),
    "Linear(24,24)": (_linear(24, 24), _same(_LINEAR)),
    "Linear(4,3)": (_linear(4, 3), _same(None)),
    "Linear(4,4)": (_linear(4, 4), _same(None)),
    "Linear(7,3)": (_linear(7, 3), _same(None)),
    # BatchNorm1d: torch wants a rank-2 or rank-3 input.
    "BatchNorm1d(24)": (_batchnorm(nn.BatchNorm1d, 24), _same(_BATCHNORM1D)),
    "BatchNorm1d(2)": (_batchnorm(nn.BatchNorm1d, 2), _same(_BATCHNORM1D)),
    # BatchNorm2d: as many features as channels, or torch refuses.
    "BatchNorm2d(3)": (_batchnorm(nn.BatchNorm2d, 3), _same(_BATCHNORM_COUNT)),
    "BatchNorm2d(4)": (_batchnorm(nn.BatchNorm2d, 4), _same(_BATCHNORM_COUNT)),
    "BatchNorm2d(1)": (_batchnorm(nn.BatchNorm2d, 1), _same(_BATCHNORM_ONE)),
    "BatchNorm2d(2)": (_batchnorm(nn.BatchNorm2d, 2), _same(None)),
    # Convolutions and pools that do not fit the image.
    "Conv2d(3,4,k=2)": (_conv2d(3, 4, 2), _same(None)),
    "Conv2d(1,3,k=2)": (_conv2d(1, 3, 2), _same(None)),
    "Conv2d(2,3,k=5)": (_conv2d(2, 3, 5), _same(_KERNEL_TOO_LARGE)),
    "ConvTranspose2d(3,2,k=2)": (_conv_transpose2d(3, 2, 2), _same(None)),
    "MaxPool2d(5)": (lambda: nn.MaxPool2d(5),
                     {"ImageStar": _MAXPOOL_EMPTY,
                      "ImageZono": _KERNEL_TOO_LARGE}),
    "AvgPool2d(5)": (lambda: nn.AvgPool2d(5), _same(_KERNEL_TOO_LARGE)),
    "Upsample(x2,linear)": (
        lambda: nn.Upsample(scale_factor=2, mode="linear"), _same(None)),
    "Upsample(x2,trilinear)": (
        lambda: nn.Upsample(scale_factor=2, mode="trilinear"), _same(None)),
    # Axes the rank-4 tensor does not have.
    "Softmax(dim=4)": (lambda: nn.Softmax(dim=4), _same(None)),
    "Transpose(perm=[0,2,1])": (lambda: OnnxTranspose([0, 2, 1]),
                                _same(None)),
    "ReduceSum(axes=[4])": (
        lambda: OnnxReduceSumStaticAxes(axes=[4], keepdims=0), _same(None)),
    "Flatten(start_dim=4)": (lambda: nn.Flatten(4), _same(_FLATTEN_AXIS)),
    "Squeeze(axes=[4])": (lambda: OnnxSqueezeStaticAxes(axes=[4]),
                          _same(None)),
    "Squeeze(axes=[-5])": (lambda: OnnxSqueezeStaticAxes(axes=[-5]),
                           _same(None)),
    "Unsqueeze(axes=[5])": (lambda: OnnxUnsqueezeStaticAxes(axes=[5]),
                            _same(None)),
    "Unsqueeze(axes=[-6])": (lambda: OnnxUnsqueezeStaticAxes(axes=[-6]),
                             _same(None)),
    # torch answers (a squeeze over an axis of size 2 is no change); the
    # library refuses today, which keeps the promise.
    "Squeeze(axes=[1])": (lambda: OnnxSqueezeStaticAxes(axes=[1]),
                          _same(None)),
}

_SETS = (("ImageStar", "approx"), ("ImageStar", "exact"),
         ("ImageZono", "approx"))
_ENTRIES = ("reach_layer", "reach_pytorch_model")


def _cells():
    cells = []
    for name, (_, marks) in _LAYERS.items():
        for kind, method in _SETS:
            for entry in _ENTRIES:
                marker = marks[kind]
                cells.append(pytest.param(
                    name, kind, method, entry,
                    marks=[marker] if marker is not None else [],
                    id=f"{name}-{kind}-{method}-{entry}"))
    return cells


def _input(kind):
    """The input set over a different dyadic interval per pixel and member
    points as (C, H, W) tensors; the ``ImageStar`` has a binding row that is
    not a box row."""
    lo, hi = pixel_box(_C, _H, _W, seed=3)
    if kind == "ImageStar":
        return constrained_image_star(lo, hi, seed=3)
    return image_set(kind, lo, hi), samples(lo, hi, seed=3)


class _EveryModuleALeaf(fx.Tracer):
    """Keeps every module as one ``call_module`` node, as ``load_onnx``
    does, so the graph walk hands the layer itself to its reach rule."""

    def is_leaf_module(self, module, qualified_name):
        return True


def _graph_of(layer):
    """``nn.Sequential(layer)`` as a ``torch.fx`` graph with the layer as one
    ``call_module`` node; the model entry takes a graph as given."""
    model = nn.Sequential(layer)
    return fx.GraphModule(model, _EveryModuleALeaf().trace(model))


def _torch_outputs(layer, points):
    """``(outputs, output tensor shape)`` of the layer on the points as
    (1, 2, 3, 4) tensors, or ``(None, error)`` when torch refuses."""
    try:
        outputs = replay_torch(layer, points, _TENSOR)
        with torch.no_grad():
            shape = tuple(layer(torch.zeros(_TENSOR, dtype=torch.float64))
                          .shape)
    except Exception as error:  # noqa: BLE001 - any torch error is a refusal
        return None, error
    return outputs, shape


@pytest.mark.parametrize("name,kind,method,entry", _cells())
def test_a_layer_on_an_image_set_answers_only_where_torch_answers(
        name, kind, method, entry):
    """Catches a reach kernel that answers for a layer torch cannot apply to
    the (1, C, H, W) tensor the image set stands for (AFFINE-5: Conv1d and
    Linear through ``affine_map`` with no image-set check, BatchNorm whose
    parameter count is never compared with the channel count, an empty
    MaxPool2d window grid, Flatten past the last axis; a Squeeze or Unsqueeze
    that returns the image set unchanged), a crash in place of the refusal
    (IndexError, RuntimeError), and, where torch answers, a result that misses
    a torch output or stores it in the wrong layout. Through the model entry
    it also catches a graph walk that hands the layer's node to the wrong
    rule."""
    build, _ = _LAYERS[name]
    layer = build()
    input_set, points = _input(kind)
    outputs, shape = _torch_outputs(layer, points)
    what = f"{name} on {kind} {method} via {entry}"

    def call():
        if entry == "reach_layer":
            return reach_one_layer(layer, [input_set], method, what)
        return reach_model(_graph_of(layer), input_set, method, what)

    if outputs is None:
        with pytest.raises(_REFUSALS):
            call()
        return
    try:
        sets = call()
    except _REFUSALS:
        return
    out_chw = shape[1:] if len(shape) == 4 else None
    assert_covered(sets, outputs, what, out_chw=out_chw)

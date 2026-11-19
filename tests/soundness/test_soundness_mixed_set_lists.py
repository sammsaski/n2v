"""Soundness cell of ``reach_layer`` on a list whose sets are of two families.

Component: ``n2v/nn/layer_ops/dispatcher.py::reach_layer``. It chooses the
lane of a layer by the family of the first set of its list (Star and
ImageStar; Zono and ImageZono; Box; Hexatope; Octatope) and checks no other
set, so a list of two families, which only a direct call can build, reaches a
kernel written for one family (registry REACH-6). The promise this file
checks, for every layer branch the dispatcher has: such a list is refused
(``NotImplementedError`` or ``ValueError``), or it is answered with sets that
hold every true output of every set of the list, in the tensor's own layout.
Anything else fails: an exception of another type (a crash where a refusal is
due, PLAN step 23) and an answer that leaves true outputs out (PLAN step 7).
The owner has not yet said whether every such list must be refused; either
answer keeps this promise.

Oracle. Each set of the list is the box of its own (dyadic) bounds in the
layer's input tensor; 64 points of each box (its two extreme corners, then
corners and quarter-grid points) are replayed through the layer in torch
float64 (``replay_torch``), and every output must be a member of some
returned set by the LP oracle on that set's own rows (``assert_covered``). An
image output is compared with an image set in its stored (height, width,
channel) order and with a flat set in the tensor's own row-major order
(``out_chw``); an image set returned for an output that is not an image fails,
because nothing says how its entries line up. ``reach_one_layer`` also fails
an empty result and a reach that changed an input set.

Inputs. One small instance per layer branch of the five lanes: an element-wise
tensor of 2 channels, height 1 and width 2; an image of 2 channels, height 2
and width 3; flat tensors of 4 entries, a (2, 3) signal for Conv1d and a
(4, 1, 1) tensor for Softmax over the channels. Every coordinate of the first
set has its own interval of width 9/4 to 3 that straddles zero, lies above it
or lies below it (in turn), so bounds go well beyond plus and minus 1; the
second set cycles the regimes the other way, so its outputs differ from the
first set's and a result that drops or merges a set misses outputs. Seven set
kinds (Star, ImageStar, Zono, ImageZono, Box, Hexatope, Octatope) where the
tensor is an image, the five flat kinds where it is not.

Two exclusions keep the cell on REACH-6. A kind that the layer already fails
on alone is left out for that layer, because the defect is registered and
marked in the layer's own file: a flat Star or Zono for the pooling layers
and Pad (POOL-6, CONV-2), an ImageZono for Sign and Round (ACT-3). A Hexatope
or Octatope in the first position gets a box with no coordinate that crosses
zero, because its own ReLU split escapes at the scale of its LP tolerance
(HEXOCT-4); the second set is given in a crossing and a non-crossing variant,
since the Hexatope and Octatope ReLU kernels take another path for each.

What is kept. The full grid (every layer, every ordered pair of two families,
both methods where the Star lane reads the method; 986 triples) was run
once. Its triples were grouped by the line that raises
and the exception type, or by the kernel that answers wrongly; each failing
group keeps one marked parameter whose marker gives the size of its group,
each layer keeps its passing neighbour (an order that is answered soundly
where one exists, otherwise one that is refused), and each further distinct
passing path (a lane that answers a set of another family soundly, a kernel
line that refuses one) keeps one parameter where a planted change of that
path makes it fail. Lanes that hand the list back by the same statement as a
kept lane (Cast, Identity, Squeeze and Unsqueeze, the pass-through of a flat
set by Flatten and by the identity Transpose, the affine maps of Linear and
Conv1d) are represented by the kept lanes. Marked cases of the same defect
in the per-layer files are duplicates of these and are listed in the change
requests of that work.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx2torch.node_converters.cast import OnnxCast
from onnx2torch.node_converters.functions import OnnxFunction
from onnx2torch.node_converters.neg import OnnxNeg
from onnx2torch.node_converters.reduce import (OnnxReduceStaticAxes,
                                               OnnxReduceSumStaticAxes)
from onnx2torch.node_converters.roundings import OnnxRound
from onnx2torch.node_converters.transpose import OnnxTranspose
from onnx2torch.node_converters.unsqueeze import OnnxUnsqueezeStaticAxes

from tests.cell_support import (assert_covered, flat_set, image_set,
                                reach_one_layer, replay_torch, samples)
from tests.known_bugs import known_bug

N_POINTS = 64
FAMILY = {"Star": "Star", "ImageStar": "Star", "Zono": "Zono",
          "ImageZono": "Zono", "Box": "Box", "Hexatope": "Hexatope",
          "Octatope": "Octatope"}
ALL_KINDS = ("Star", "ImageStar", "Zono", "ImageZono", "Box", "Hexatope",
             "Octatope")
FLAT_KINDS = ("Star", "Zono", "Box", "Hexatope", "Octatope")
_EL = (2, 1, 2)       # element-wise: 2 channels, height 1, width 2
_IM = (2, 2, 3)       # image: 2 channels, height 2, width 3


# --------------------------------------------------------------------------- #
# One small instance per layer branch of the dispatcher                       #
# --------------------------------------------------------------------------- #
def _dyadic(shape, seed, scale=8):
    """Nonzero dyadic numbers (multiples of 1/scale in [-1, 1]), the same
    whatever the test's seed: float32, float64 and the replay agree."""
    rng = np.random.default_rng(seed)
    values = rng.integers(1, scale + 1, size=shape) / scale
    return torch.tensor(values * rng.choice([-1.0, 1.0], size=shape),
                        dtype=torch.float32)


def _with_params(layer, seed):
    with torch.no_grad():
        for k, p in enumerate(layer.parameters()):
            p.copy_(_dyadic(tuple(p.shape), seed + k))
    return layer.eval()


def _batchnorm():
    layer = nn.BatchNorm1d(4)
    with torch.no_grad():
        layer.running_mean.copy_(torch.tensor([0.25, -0.5, 1.0, 0.0]))
        layer.running_var.copy_(torch.tensor([1.0, 4.0, 0.25, 1.0]))
        layer.weight.copy_(torch.tensor([1.5, -0.5, 2.0, 1.0]))
        layer.bias.copy_(torch.tensor([0.25, 1.0, -1.0, 0.5]))
    return layer.eval()


# name: (factory, input tensor shape without the batch axis, kinds, the
# methods of a list led by a Star or an ImageStar). The other lanes do not
# read the method (REACH-8), so their lists run under "approx" only.
_NO_IMAGEZONO = tuple(k for k in ALL_KINDS if k != "ImageZono")        # ACT-3
_NO_FLAT_STAR_ZONO = ("ImageStar", "ImageZono", "Box", "Hexatope",
                      "Octatope")                               # POOL-6, CONV-2
_BOTH = ("exact", "approx")
_APPROX = ("approx",)
LAYERS = {
    "linear": (lambda: _with_params(nn.Linear(4, 3), 10), (4,), FLAT_KINDS,
               _APPROX),
    "relu": (nn.ReLU, _EL, ALL_KINDS, _BOTH),
    "leakyrelu": (lambda: nn.LeakyReLU(0.25), _EL, ALL_KINDS, _BOTH),
    "sigmoid": (nn.Sigmoid, _EL, ALL_KINDS, _APPROX),
    "tanh": (nn.Tanh, _EL, ALL_KINDS, _APPROX),
    "sin": (lambda: OnnxFunction("Sin"), _EL, ALL_KINDS, _APPROX),
    "sign": (lambda: OnnxFunction("Sign"), _EL, _NO_IMAGEZONO, _BOTH),
    "round": (lambda: OnnxRound("Round"), _EL, _NO_IMAGEZONO, _APPROX),
    "neg": (OnnxNeg, _EL, ALL_KINDS, _APPROX),
    "cast": (lambda: OnnxCast(1), _EL, ALL_KINDS, _APPROX),
    "identity": (nn.Identity, _EL, ALL_KINDS, _APPROX),
    "softmax": (lambda: nn.Softmax(dim=1), (4, 1, 1), ALL_KINDS, _APPROX),
    "conv2d": (lambda: _with_params(nn.Conv2d(2, 2, 2), 20), _IM, ALL_KINDS,
               _BOTH),
    "conv1d": (lambda: _with_params(nn.Conv1d(2, 1, 2), 30), (2, 3),
               FLAT_KINDS, _APPROX),
    "conv_transpose2d": (lambda: _with_params(nn.ConvTranspose2d(2, 1, 2), 40),
                         _IM, ALL_KINDS, _BOTH),
    "batchnorm": (_batchnorm, (4,), FLAT_KINDS, _APPROX),
    "maxpool2d": (lambda: nn.MaxPool2d(2, 1), _IM, _NO_FLAT_STAR_ZONO, _BOTH),
    "avgpool2d": (lambda: nn.AvgPool2d(2, 1), _IM, _NO_FLAT_STAR_ZONO,
                  _APPROX),
    "global_avgpool": (lambda: nn.AdaptiveAvgPool2d(1), _IM,
                       _NO_FLAT_STAR_ZONO, _APPROX),
    "flatten": (nn.Flatten, _IM, ALL_KINDS, _APPROX),
    "pad": (lambda: nn.ZeroPad2d((1, 0, 0, 1)), _IM, _NO_FLAT_STAR_ZONO,
            _APPROX),
    "upsample": (lambda: nn.Upsample(scale_factor=2, mode="nearest"), _IM,
                 ALL_KINDS, _APPROX),
    "reduce": (lambda: OnnxReduceSumStaticAxes([2, 3], 1), _IM, ALL_KINDS,
               _APPROX),
    # axis 1 of a (1, 4) tensor, which every flat set answers alone
    "reduce_flat": (lambda: OnnxReduceStaticAxes("ReduceMean", [1], 1), (4,),
                    FLAT_KINDS, _APPROX),
    "transpose": (lambda: OnnxTranspose([0, 1, 3, 2]), _IM, ALL_KINDS,
                  _APPROX),
    "transpose_identity": (lambda: OnnxTranspose([0, 1, 2, 3]), _IM,
                           ALL_KINDS, _APPROX),
    "unsqueeze": (lambda: OnnxUnsqueezeStaticAxes([1]), _IM, ALL_KINDS,
                  _APPROX),
    "sequential": (lambda: nn.Sequential(nn.Identity(), nn.ReLU()), _EL,
                   ALL_KINDS, _BOTH),
}


# --------------------------------------------------------------------------- #
# The two input boxes                                                         #
# --------------------------------------------------------------------------- #
def _box(shape, regime, seed):
    """Dyadic bounds of a tensor of ``shape``: coordinate i straddles zero
    (regime 0), lies above it (1) or below it (2) by ``regime(i)``, with a
    width of 9/4 to 3 and ends up to 4.5 away from zero."""
    n = int(np.prod(shape))
    rng = np.random.default_rng(seed)
    lo, hi = np.zeros(n), np.zeros(n)
    for i in range(n):
        width = rng.integers(9, 13) / 4.0
        r = regime(i)
        if r == 0:
            lo[i] = -rng.integers(3, 9) / 4.0
            hi[i] = lo[i] + width
        elif r == 1:
            lo[i] = rng.integers(1, 7) / 4.0
            hi[i] = lo[i] + width
        else:
            hi[i] = -rng.integers(1, 7) / 4.0
            lo[i] = hi[i] - width
    return lo.reshape(shape), hi.reshape(shape)


def _boxes(shape, first, crossing):
    """The boxes of the first and the second set. A Hexatope or Octatope in
    the first position crosses zero nowhere (its own split escapes by its LP
    tolerance, HEXOCT-4); the second set crosses zero in some coordinate
    when ``crossing``, and in none otherwise."""
    lead = (lambda i: 1 + i % 2) if first in ("Hexatope", "Octatope") \
        else (lambda i: i % 3)
    second = (lambda i: (i + 1) % 3) if crossing else (lambda i: 1 + i % 2)
    return [_box(shape, lead, 1), _box(shape, second, 2)]


def _build(kind, lo, hi):
    if kind.startswith("Image"):
        return image_set(kind, lo, hi)
    return flat_set(kind, lo, hi)


def _out_chw(layer, shape):
    """(C, H, W) of the output when it is one image, else None."""
    with torch.no_grad():
        y = layer.double()(torch.zeros((1,) + tuple(shape),
                                       dtype=torch.float64))
    return tuple(y.shape[1:]) if y.dim() == 4 and y.shape[0] == 1 else None


def check_mixed_list(name, first, second, method, crossing=True):
    """Reach ``[first set, second set]`` through layer ``name`` and demand a
    refusal or every true output of both boxes in some returned set."""
    factory, shape, _, _ = LAYERS[name]
    layer = factory()
    boxes = _boxes(shape, first, crossing)
    sets = [_build(kind, lo, hi) for kind, (lo, hi) in zip((first, second),
                                                            boxes)]
    what = (f"{name} {method} on a list of a {first} and a {second}"
            f"{'' if crossing else ' (no crossing coordinate)'}")
    try:
        out = reach_one_layer(layer, sets, method, what)
    except (NotImplementedError, ValueError):
        return "refused"
    outputs = []
    for k, (lo, hi) in enumerate(boxes):
        outputs += replay_torch(layer, samples(lo, hi, N_POINTS, seed=k),
                                (1,) + tuple(shape))
    assert_covered(out, outputs, what, out_chw=_out_chw(factory(), shape))
    return "answered"


def all_triples():
    """Every (layer, first, second, method, crossing) of the full grid."""
    for name, (_, _, kinds, star_methods) in LAYERS.items():
        for first in kinds:
            for second in kinds:
                if FAMILY[first] == FAMILY[second]:
                    continue
                methods = star_methods if FAMILY[first] == "Star" else _APPROX
                for method in methods:
                    yield name, first, second, method, True
                    if first in ("Hexatope", "Octatope") and name in (
                            "relu", "sequential"):
                        yield name, first, second, method, False


# --------------------------------------------------------------------------- #
# The kept orders                                                             #
# --------------------------------------------------------------------------- #
def _crash(raises, text):
    """A crash where a refusal is due (PLAN step 23). ``text`` names the
    kernel, what it does with the set of the other family, and how many
    orders of the full grid take the same line."""
    return known_bug(23, f"{text}, where a refusal is due (registry REACH-6: "
                         f"reach_layer picks the kernel by the first set of "
                         f"the list and checks no other)", raises=raises)


def _wrong(text):
    """An answer that leaves true outputs out (PLAN step 7)."""
    return known_bug(7, f"{text} (registry REACH-6: reach_layer picks the "
                        f"kernel by the first set of the list and checks no "
                        f"other)")


_AE, _TE = AttributeError, TypeError
_ZONO_SECOND = "of a Zono or ImageZono that follows a Star or ImageStar"
_FLAT_SECOND = "of a Box, Hexatope or Octatope that follows a Star or ImageStar"
_IMAGE_KEPT = ("comes back unchanged, an image set stored (height, width, "
               "channel) for an output that is not an image, so the next "
               "layer reads its entries in the stored order where the tensor "
               "is (channel, height, width); 10 of 12 entries of the set lie "
               "at the wrong place")

# (layer, first, second, method, crossing, marker or None). Without a marker:
# the passing neighbour of the layer or a further passing path (a lane that
# answers a set of another family soundly, a line that refuses one).
_KEPT = [
    # Linear: every lane maps each set by its own affine map (the Hexatope
    # lane calls the same affine_map of the set as the others).
    ("linear", "Star", "Zono", "approx", True, None),
    ("linear", "Zono", "Box", "approx", True, None),
    ("linear", "Box", "Hexatope", "approx", True, None),
    ("linear", "Octatope", "Star", "approx", True, None),
    # ReLU (the Sequential's ReLU child takes the same lines).
    ("relu", "ImageStar", "ImageZono", "exact", True, _crash(
        _AE, "exact ReLU: _relu_single_star_exact reads the outer zonotope Z "
             f"{_ZONO_SECOND} and raises AttributeError (8 orders of the "
             "grid, ReLU and Sequential)")),
    ("relu", "Star", "Zono", "approx", True, _crash(
        _AE, "approx ReLU: _relu_single_star_approx reads the outer zonotope "
             f"Z {_ZONO_SECOND} and raises AttributeError (8 orders)")),
    ("relu", "Star", "Box", "exact", True, _crash(
        _AE, "exact ReLU: _relu_single_star_exact reads the basis V "
             f"{_FLAT_SECOND} and raises AttributeError (12 orders)")),
    ("relu", "ImageStar", "Octatope", "approx", True, _crash(
        _AE, "approx ReLU: _relu_single_star_approx reads the basis V "
             f"{_FLAT_SECOND} and raises AttributeError (12 orders)")),
    ("relu", "ImageZono", "ImageStar", "approx", True, _crash(
        _AE, "ReLU led by a Zono or ImageZono: _relu_single_zono calls "
             "get_bounds of a Star, ImageStar or Box that follows, which has "
             "none (AttributeError; 12 orders)")),
    ("relu", "Zono", "Hexatope", "approx", True, _crash(
        _TE, "ReLU led by a Zono or ImageZono: _relu_single_zono calls "
             "get_bounds() of a Hexatope or Octatope that follows, whose "
             "get_bounds needs a solver (TypeError; 8 orders)")),
    ("relu", "Box", "ImageZono", "approx", True, _crash(
        _AE, "ReLU led by a Box: relu_box reads lb of any set of another "
             "family that follows (AttributeError; 12 orders)")),
    ("relu", "Hexatope", "Star", "approx", True, _crash(
        _TE, "ReLU led by a Hexatope: _relu_single_hexatope splits a Star, "
             "ImageStar, Zono or ImageZono that follows and has a coordinate "
             "crossing zero by calling its intersect_half_space with the "
             "keyword solver, which those sets do not take (TypeError; 8 "
             "orders)")),
    ("relu", "Hexatope", "Box", "approx", True, _crash(
        _AE, "ReLU led by a Hexatope: _relu_single_hexatope splits a Box "
             "that follows and crosses zero by calling its "
             "intersect_half_space, which a Box has not (AttributeError; 2 "
             "orders)")),
    ("relu", "Octatope", "ImageZono", "approx", True, _crash(
        _TE, "ReLU led by an Octatope: _relu_single_octatope splits a Star, "
             "ImageStar, Zono or ImageZono that follows and has a coordinate "
             "crossing zero by calling its intersect_half_space with the "
             "keyword solver (TypeError; 8 orders)")),
    ("relu", "Octatope", "Box", "approx", True, _crash(
        _AE, "ReLU led by an Octatope: _relu_single_octatope splits a Box "
             "that follows by calling its intersect_half_space, which a Box "
             "has not (AttributeError; 2 orders)")),
    ("relu", "Hexatope", "ImageZono", "approx", False, _wrong(
        "ReLU led by a Hexatope: an ImageZono that follows and crosses zero "
        "nowhere goes through the kernel's affine map, which zeroes its "
        "negative entries and returns a flat Zono in the stored (height, "
        "width, channel) order, where a flat set stands for the tensor in "
        "(channel, height, width) order: 41 of 128 true outputs lie outside "
        "(2 orders of the grid, ReLU and Sequential)")),
    ("relu", "Octatope", "ImageZono", "approx", False, _wrong(
        "ReLU led by an Octatope: an ImageZono that follows and crosses zero "
        "nowhere comes back as a flat Zono in the stored (height, width, "
        "channel) order: 41 of 128 true outputs lie outside (2 orders)")),
    ("relu", "Hexatope", "Octatope", "approx", False, None),
    ("relu", "Octatope", "ImageStar", "approx", False, None),
    # LeakyReLU.
    ("leakyrelu", "Star", "ImageZono", "exact", True, _crash(
        _AE, "exact LeakyReLU: _leakyrelu_single_star_exact reads the outer "
             f"zonotope Z {_ZONO_SECOND} and raises AttributeError (4 "
             "orders)")),
    ("leakyrelu", "ImageStar", "Zono", "approx", True, _crash(
        _AE, "approx LeakyReLU: _leakyrelu_single_star_approx reads the "
             f"outer zonotope Z {_ZONO_SECOND} and raises AttributeError (4 "
             "orders)")),
    ("leakyrelu", "ImageStar", "Box", "exact", True, _crash(
        _AE, "exact LeakyReLU: _leakyrelu_single_star_exact reads the basis "
             f"V {_FLAT_SECOND} and raises AttributeError (6 orders)")),
    ("leakyrelu", "Star", "Hexatope", "approx", True, _crash(
        _AE, "approx LeakyReLU: _leakyrelu_single_star_approx reads the "
             f"basis V {_FLAT_SECOND} and raises AttributeError (6 orders)")),
    ("leakyrelu", "ImageZono", "Star", "approx", True, _crash(
        _AE, "LeakyReLU led by a Zono or ImageZono: _leakyrelu_single_zono "
             "calls get_bounds of a Star, ImageStar or Box that follows "
             "(AttributeError; 6 orders)")),
    ("leakyrelu", "Zono", "Octatope", "approx", True, _crash(
        _TE, "LeakyReLU led by a Zono or ImageZono: _leakyrelu_single_zono "
             "calls get_bounds() of a Hexatope or Octatope, which needs a "
             "solver (TypeError; 4 orders)")),
    ("leakyrelu", "Box", "ImageStar", "approx", True, _crash(
        _AE, "LeakyReLU led by a Box: leakyrelu_box reads lb of any set of "
             "another family (AttributeError; 6 orders)")),
    ("leakyrelu", "Hexatope", "Zono", "approx", True, None),
    # Sigmoid; its Star and Zono kernels are Tanh's as well.
    ("sigmoid", "Star", "Box", "approx", True, _crash(
        _AE, "Sigmoid or Tanh led by a Star or ImageStar: "
             "_s_curve_single_star_approx reads nVar of a Zono, ImageZono or "
             "Box that follows (AttributeError; 12 orders, Sigmoid and "
             "Tanh)")),
    ("sigmoid", "ImageZono", "ImageStar", "approx", True, _crash(
        _AE, "Sigmoid or Tanh led by a Zono or ImageZono: "
             "_s_curve_single_zono calls get_bounds of a Star, ImageStar or "
             "Box that follows (AttributeError; 12 orders, Sigmoid and "
             "Tanh)")),
    ("sigmoid", "Zono", "Hexatope", "approx", True, _crash(
        _TE, "Sigmoid or Tanh led by a Zono or ImageZono: "
             "_s_curve_single_zono calls get_bounds() of a Hexatope or "
             "Octatope, which needs a solver (TypeError; 8 orders, Sigmoid "
             "and Tanh)")),
    ("sigmoid", "Box", "Zono", "approx", True, _crash(
        _AE, "Sigmoid led by a Box: sigmoid_box reads lb of any set of "
             "another family (AttributeError; 6 orders)")),
    ("sigmoid", "ImageStar", "Hexatope", "approx", True, None),
    # Tanh.
    ("tanh", "Box", "ImageStar", "approx", True, _crash(
        _AE, "Tanh led by a Box: tanh_box reads lb of any set of another "
             "family (AttributeError; 6 orders)")),
    ("tanh", "Star", "Octatope", "approx", True, None),
    # Sin (Cos takes the same code).
    ("sin", "ImageStar", "ImageZono", "approx", True, _crash(
        _AE, "Sin led by a Star or ImageStar: _trig_single_star reads nVar "
             "of a Zono, ImageZono or Box that follows (AttributeError; 6 "
             "orders)")),
    ("sin", "ImageZono", "Octatope", "approx", True, _crash(
        _TE, "Sin led by a Zono or ImageZono: trig_zono calls get_ranges() "
             "of a Hexatope or Octatope without the solver it needs "
             "(TypeError; 4 orders)")),
    ("sin", "Box", "Star", "approx", True, _crash(
        _AE, "Sin led by a Box: trig_box reads lb of any set of another "
             "family (AttributeError; 6 orders)")),
    ("sin", "Zono", "ImageStar", "approx", True, _wrong(
        "Sin led by a Zono or ImageZono: trig_zono returns an ImageStar "
        "that follows as a flat Zono of its ranges in the stored (height, "
        "width, channel) order: 29 of 128 true outputs lie outside (2 "
        "orders)")),
    ("sin", "ImageZono", "Box", "approx", True, None),
    # Sign.
    ("sign", "ImageStar", "Zono", "exact", True, _crash(
        _TE, "exact Sign led by a Star or ImageStar: the phase helper "
             "_phase_range calls get_range(i, solver) of a Zono that "
             "follows, whose get_range takes one index (TypeError; 2 "
             "orders)")),
    ("sign", "Star", "Box", "approx", True, _crash(
        _AE, "approx Sign led by a Star or ImageStar: "
             "_sign_single_star_approx reads nVar of a Zono or Box that "
             "follows (AttributeError; 4 orders)")),
    ("sign", "Star", "Octatope", "exact", True, _crash(
        _AE, "Sign led by a Star or ImageStar: the phase helper _phase_range "
             "reads the basis V of a Box, Hexatope or Octatope that follows "
             "(AttributeError; 10 orders, exact for all three, approx for "
             "the Hexatope and the Octatope)")),
    ("sign", "Zono", "ImageStar", "approx", True, _crash(
        _AE, "Sign led by a Zono: sign_zono calls get_bounds of a Star, "
             "ImageStar or Box that follows (AttributeError; 3 orders)")),
    ("sign", "Zono", "Hexatope", "approx", True, _crash(
        _TE, "Sign led by a Zono: sign_zono calls get_bounds() of a "
             "Hexatope or Octatope, which needs a solver (TypeError; 2 "
             "orders)")),
    ("sign", "Box", "Zono", "approx", True, _crash(
        _AE, "Sign led by a Box: sign_box reads lb of any set of another "
             "family (AttributeError; 5 orders)")),
    ("sign", "Octatope", "ImageStar", "approx", True, None),
    # Round (Floor and Ceil take the same code).
    ("round", "ImageStar", "Box", "approx", True, _crash(
        _AE, "Round led by a Star or ImageStar: round_star reads nVar of a "
             "Zono or Box that follows (AttributeError; 4 orders)")),
    ("round", "Zono", "Star", "approx", True, _crash(
        _AE, "Round led by a Zono: round_zono reads the centre c of any set "
             "of another family (AttributeError; 5 orders)")),
    ("round", "Box", "Octatope", "approx", True, _crash(
        _AE, "Round led by a Box: round_box reads lb of any set of another "
             "family (AttributeError; 5 orders)")),
    ("round", "ImageStar", "Octatope", "approx", True, None),
    # Neg.
    ("neg", "ImageStar", "ImageZono", "approx", True, _crash(
        _AE, "Neg led by a Star or ImageStar: _neg_sets_star reads the rows "
             "C of a Zono or ImageZono that follows (AttributeError; 4 "
             "orders)")),
    ("neg", "Star", "Hexatope", "approx", True, _crash(
        _AE, "Neg led by a Star or ImageStar: _neg_sets_star reads the "
             "basis V of a Box, Hexatope or Octatope that follows "
             "(AttributeError; 6 orders)")),
    ("neg", "ImageZono", "Box", "approx", True, _crash(
        _AE, "Neg led by a Zono or ImageZono: _neg_sets_zono reads the "
             "centre c of any set of another family (AttributeError; 10 "
             "orders)")),
    ("neg", "Box", "ImageStar", "approx", True, _crash(
        _AE, "Neg led by a Box: _neg_sets_box reads ub of any set of another "
             "family (AttributeError; 6 orders)")),
    ("neg", "Hexatope", "ImageZono", "approx", True, _wrong(
        "Neg led by a Hexatope maps an ImageZono that follows through "
        "linear_hexatope with the matrix -I and returns a flat Zono in the "
        "stored (height, width, channel) order: 45 of 128 true outputs lie "
        "outside (1 order)")),
    ("neg", "Octatope", "ImageZono", "approx", True, _wrong(
        "Neg led by an Octatope maps an ImageZono that follows through "
        "linear_octatope with the matrix -I and returns a flat Zono in the "
        "stored (height, width, channel) order: 45 of 128 true outputs lie "
        "outside (1 order)")),
    ("neg", "Hexatope", "ImageStar", "approx", True, None),
    ("neg", "Octatope", "Star", "approx", True, None),
    # Cast and the pass-through layers: every lane hands the list back by the
    # same statement; the Star lane and one other lane are kept for each.
    ("cast", "Star", "ImageZono", "approx", True, None),
    ("cast", "Octatope", "Zono", "approx", True, None),
    ("identity", "Star", "Octatope", "approx", True, None),
    ("identity", "Zono", "ImageStar", "approx", True, None),
    # Softmax.
    ("softmax", "ImageStar", "ImageZono", "approx", True, _crash(
        _TE, "Softmax led by a Star or ImageStar: softmax_star calls "
             "get_range(i, solver) of a Zono, ImageZono or Box that follows "
             "(TypeError; 6 orders)")),
    ("softmax", "Box", "Zono", "approx", True, _crash(
        _AE, "Softmax led by a Box: softmax_box reads lb of any set of "
             "another family (AttributeError; 6 orders)")),
    ("softmax", "Zono", "Star", "approx", True, None),
    # Conv2d.
    ("conv2d", "ImageStar", "ImageZono", "exact", True, _crash(
        _TE, "Conv2d led by an ImageStar: the type check of conv2d_star "
             "raises TypeError ('conv2d_star expects Star or ImageStar') for "
             "any set of another family, where the declared refusal types "
             "are NotImplementedError and ValueError (10 orders, both "
             "methods)")),
    ("conv2d", "ImageZono", "ImageStar", "approx", True, None),
    # Conv1d: the Star, Zono and Box lanes answer a set of another family by
    # its own affine_map, as Linear does.
    ("conv1d", "Star", "Box", "approx", True, None),
    ("conv1d", "Box", "Hexatope", "approx", True, None),
    # ConvTranspose2d: its kernels refuse a set of another family.
    ("conv_transpose2d", "ImageStar", "ImageZono", "exact", True, None),
    ("conv_transpose2d", "ImageZono", "Box", "approx", True, None),
    # BatchNorm.
    ("batchnorm", "Star", "Zono", "approx", True, _crash(
        _TE, "BatchNorm led by a Star: the type check of batchnorm_star "
             "raises TypeError for a Zono, Box, Hexatope or Octatope that "
             "follows (4 orders)")),
    ("batchnorm", "Zono", "Box", "approx", True, _crash(
        _TE, "BatchNorm led by a Zono: the type check of batchnorm_zono "
             "raises TypeError for a Star, Box, Hexatope or Octatope that "
             "follows (4 orders)")),
    ("batchnorm", "Box", "Hexatope", "approx", True, _crash(
        _AE, "BatchNorm led by a Box: _batchnorm_single_box reads lb of a "
             "Star, Zono, Hexatope or Octatope that follows (AttributeError; "
             "4 orders)")),
    ("batchnorm", "Hexatope", "Star", "approx", True, None),
    ("batchnorm", "Octatope", "Zono", "approx", True, None),
    # MaxPool2d.
    ("maxpool2d", "ImageStar", "ImageZono", "exact", True, _crash(
        _TE, "exact MaxPool2d led by an ImageStar: the type check of the "
             "exact kernel raises TypeError ('MaxPool2D expects ImageStar "
             "input') for an ImageZono, Box, Hexatope or Octatope that "
             "follows (4 orders)")),
    ("maxpool2d", "ImageStar", "Box", "approx", True, _crash(
        _TE, "approx MaxPool2d led by an ImageStar: the type check of the "
             "approx kernel raises TypeError for an ImageZono, Box, Hexatope "
             "or Octatope that follows (4 orders)")),
    ("maxpool2d", "ImageZono", "ImageStar", "approx", True, _crash(
        _AE, "MaxPool2d led by an ImageZono: maxpool2d_zono calls get_bounds "
             "of an ImageStar or Box that follows (AttributeError; 2 "
             "orders)")),
    ("maxpool2d", "ImageZono", "Octatope", "approx", True, _crash(
        _TE, "MaxPool2d led by an ImageZono: maxpool2d_zono calls "
             "get_bounds() of a Hexatope or Octatope, which needs a solver "
             "(TypeError; 2 orders)")),
    ("maxpool2d", "Box", "ImageStar", "approx", True, None),
    # AvgPool2d.
    ("avgpool2d", "ImageStar", "Hexatope", "approx", True, _crash(
        _TE, "AvgPool2d led by an ImageStar: the type check of "
             "avgpool2d_star raises TypeError for an ImageZono, Box, "
             "Hexatope or Octatope that follows (4 orders)")),
    ("avgpool2d", "ImageZono", "ImageStar", "approx", True, _crash(
        _AE, "AvgPool2d led by an ImageZono: avgpool2d_zono reads the "
             "centre c of an ImageStar that follows (AttributeError; 1 "
             "order)")),
    ("avgpool2d", "ImageZono", "Box", "approx", True, _crash(
        _AE, "AvgPool2d led by an ImageZono: avgpool2d_zono reads height of "
             "a Box, Hexatope or Octatope that follows (AttributeError; 3 "
             "orders)")),
    ("avgpool2d", "Box", "ImageZono", "approx", True, None),
    # Global average pooling.
    ("global_avgpool", "ImageStar", "ImageZono", "approx", True, _crash(
        _TE, "global average pooling led by an ImageStar: the type check of "
             "global_avgpool_star raises TypeError for any set of another "
             "family (4 orders)")),
    ("global_avgpool", "ImageZono", "Box", "approx", True, _crash(
        _TE, "global average pooling led by an ImageZono: the type check of "
             "global_avgpool_zono raises TypeError for any set of another "
             "family (4 orders)")),
    ("global_avgpool", "Octatope", "ImageStar", "approx", True, None),
    # Flatten: the flat sets are answered, an image set of another family is
    # handed back unflattened.
    ("flatten", "ImageStar", "ImageZono", "approx", True, _wrong(
        f"Flatten led by a Star or ImageStar: an ImageZono that follows "
        f"{_IMAGE_KEPT} (2 orders)")),
    ("flatten", "Zono", "ImageStar", "approx", True, _wrong(
        f"Flatten led by a Zono or ImageZono: an ImageStar that follows "
        f"{_IMAGE_KEPT} (2 orders)")),
    ("flatten", "Box", "ImageZono", "approx", True, _wrong(
        f"Flatten led by a Box: an ImageStar or ImageZono that follows "
        f"{_IMAGE_KEPT} (2 orders)")),
    ("flatten", "Hexatope", "ImageStar", "approx", True, _wrong(
        f"Flatten led by a Hexatope: an ImageStar or ImageZono that follows "
        f"{_IMAGE_KEPT} (2 orders)")),
    ("flatten", "Octatope", "ImageZono", "approx", True, _wrong(
        f"Flatten led by an Octatope: an ImageStar or ImageZono that follows "
        f"{_IMAGE_KEPT} (2 orders)")),
    ("flatten", "ImageStar", "Zono", "approx", True, None),
    ("flatten", "ImageZono", "Box", "approx", True, None),
    ("flatten", "Hexatope", "Star", "approx", True, None),
    # Pad.
    ("pad", "ImageStar", "ImageZono", "approx", True, _crash(
        _TE, "Pad led by an ImageStar: the type check of pad_star raises "
             "TypeError for any set of another family (4 orders)")),
    ("pad", "ImageZono", "Hexatope", "approx", True, _crash(
        _TE, "Pad led by an ImageZono: the type check of pad_zono raises "
             "TypeError for any set of another family (4 orders)")),
    ("pad", "Hexatope", "ImageZono", "approx", True, None),
    # Upsample.
    ("upsample", "ImageStar", "ImageZono", "approx", True, _crash(
        _TE, "Upsample led by an ImageStar: the type check of upsample_star "
             "raises TypeError for any set of another family (5 orders)")),
    ("upsample", "ImageZono", "ImageStar", "approx", True, None),
    # Reduce.
    ("reduce", "ImageStar", "Zono", "approx", True, _crash(
        _TE, "Reduce led by a Star or ImageStar: the type check of "
             "reduce_star raises TypeError for any set of another family (9 "
             "orders, ReduceSum of an image and ReduceMean of a flat "
             "tensor)")),
    ("reduce", "ImageZono", "ImageStar", "approx", True, _crash(
        _TE, "Reduce led by a Zono or ImageZono: the type check of "
             "reduce_zono raises TypeError for any set of another family (9 "
             "orders, ReduceSum and ReduceMean)")),
    ("reduce", "Box", "ImageStar", "approx", True, None),
    ("reduce_flat", "Box", "Star", "approx", True, _crash(
        _TE, "ReduceMean over the entries of a flat tensor led by a Box: "
             "the type check of reduce_box raises TypeError for any set of "
             "another family (4 orders)")),
    # Transpose: an axis permutation refuses every set it cannot place; the
    # identity permutation hands every set back.
    ("transpose", "ImageStar", "ImageZono", "approx", True, None),
    ("transpose", "ImageZono", "Star", "approx", True, None),
    ("transpose", "Box", "ImageStar", "approx", True, None),
    ("transpose_identity", "Star", "ImageZono", "approx", True, None),
    ("transpose_identity", "ImageZono", "Star", "approx", True, None),
    # Squeeze and Unsqueeze: an image set must be refused (the dispatcher
    # refuses one that leads the list), and only the leading one is.
    ("unsqueeze", "Star", "ImageZono", "approx", True, _wrong(
        "Unsqueeze led by a Star: _shape_identity_star refuses an ImageStar "
        f"only, and an ImageZono that follows {_IMAGE_KEPT} (1 order)")),
    ("unsqueeze", "Zono", "ImageStar", "approx", True, _wrong(
        "Unsqueeze led by a Zono: _shape_identity_zono refuses an ImageZono "
        f"only, and an ImageStar that follows {_IMAGE_KEPT} (1 order)")),
    ("unsqueeze", "Box", "ImageStar", "approx", True, _wrong(
        "Unsqueeze led by a Box: the Box lane hands every set back, and an "
        f"ImageStar or ImageZono that follows {_IMAGE_KEPT} (2 orders)")),
    ("unsqueeze", "Hexatope", "ImageZono", "approx", True, _wrong(
        "Unsqueeze led by a Hexatope: the Hexatope lane hands every set "
        f"back, and an ImageStar or ImageZono that follows {_IMAGE_KEPT} (2 "
        "orders)")),
    ("unsqueeze", "Octatope", "ImageStar", "approx", True, _wrong(
        "Unsqueeze led by an Octatope: the Octatope lane hands every set "
        f"back, and an ImageStar or ImageZono that follows {_IMAGE_KEPT} (2 "
        "orders)")),
    ("unsqueeze", "Star", "Box", "approx", True, None),
    ("unsqueeze", "Box", "Octatope", "approx", True, None),
    ("unsqueeze", "Hexatope", "Star", "approx", True, None),
    ("unsqueeze", "ImageStar", "Box", "approx", True, None),
    ("unsqueeze", "ImageZono", "Star", "approx", True, None),
    # Sequential: each child gets the list as it is.
    ("sequential", "Box", "ImageStar", "approx", True, _crash(
        _AE, "Sequential(Identity, ReLU) led by a Box: the ReLU child gets "
             "the list as it is and relu_box reads lb of an ImageStar that "
             "follows (AttributeError; the path of the ReLU row)")),
    ("sequential", "Hexatope", "Octatope", "approx", False, None),
]


def _id(name, first, second, method, crossing):
    return (f"{name}-{first}-then-{second}-{method}"
            f"{'' if crossing else '-nocross'}")


_PARAMS = [pytest.param(*row[:5], marks=row[5] or (), id=_id(*row[:5]))
           for row in _KEPT]


@pytest.mark.parametrize("name,first,second,method,crossing", _PARAMS)
def test_reach_layer_on_a_list_of_two_set_families_refuses_or_holds_every_output(  # noqa: E501
        name, first, second, method, crossing):
    """One ``reach_layer`` call on a list of two sets of two families must
    refuse (``NotImplementedError`` or ``ValueError``) or return sets holding
    every true output of 64 points of each set's box, in the tensor's own
    layout (module docstring). Catches a kernel that is handed a set of
    another family and crashes on it, answers it as if it were of its own
    family (an image set returned unflattened or as a flat set in its stored
    order), drops it or merges it with the first set."""
    check_mixed_list(name, first, second, method, crossing)

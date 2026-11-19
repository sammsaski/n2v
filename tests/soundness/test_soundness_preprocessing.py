"""Soundness of model preprocessing: the BatchNorm fusion that runs before a
reach and the stripping of a final softmax (``n2v/utils/model_preprocessing.py``).

What the file promises:

- ``fuse_batchnorm`` returns a model that computes, in eval mode, the function
  of the model it was given: every Conv2d or Linear followed by its BatchNorm
  is folded, and every pair whose fold would change the function (a layer
  whose output has a second user, a layer or BatchNorm called twice, a
  subclass with its own forward, a BatchNorm without running statistics) is
  left as it is.
- ``reach_pytorch_model``, which fuses before every sound reach except on a
  Box, returns sets that hold every true output of the original model, and
  refuses a model it cannot reach.
- ``strip_final_softmax`` returns a model whose output is exactly the input of
  the trailing plain softmax, and leaves every model whose last operation is
  something else (a LogSoftmax, a softmax in the middle, an add after the
  softmax) computing what it computed.
- The fold is rounded once in float64 (``_fold_batchnorm`` docstring): a few
  units in the last place of each folded weight, and of the terms folded into
  each bias. Where that rounding escapes a zero-slack reach is registry
  ROUND-22, marked in ``test_soundness_graph_walk.py`` and not marked again
  here.

Oracles, never the fused or stripped model itself: the original model in
float64 (``replay_torch``) or onnxruntime for an ONNX file; exact equality on
dyadic data, where every parameter is a multiple of a small power of two and
every running variance plus eps is a power of four (eps 3/4, variances 1/4,
13/4 and 61/4), so that gamma / sqrt(var + eps) and both forward passes are
exact; the exact image in rationals (``tests/_oracles/rational``) for one
reach; a fold at 60 decimal digits for the precision of the folded
parameters; the original model's own layers for the logits of a stripped
model.

Inputs that change the code path: Linear and Conv2d with and without bias,
square and not; BatchNorm with affine=False, with an eps that is not
negligible against the variance, without running statistics, first in a
Sequential, after a layer it cannot be fused into, last in a Sequential and
twice in a row; Conv2d with stride, padding, a kernel that is not square,
groups, dilation and reflect or circular padding; nested Sequentials and
Sequential subclasses with their own forward; traced modules in which the
layer's output has a second user, the layer or the BatchNorm is called twice,
two BatchNorms follow one layer, the pair sits in a child module, the
BatchNorm reads a function's output, the BatchNorm is called by keyword; a
Sequential that holds one module object twice; subclasses that override
forward or ``_conv_forward``; a float64 network, also one whose parameters
are not float32 numbers; a model in training mode; ONNX models loaded by
onnx2torch. Input boxes reach well beyond plus and minus 1.
"""

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper

from n2v.utils.model_loader import load_onnx
from n2v.utils.model_preprocessing import (fuse_batchnorm, has_softmax,
                                           strip_final_softmax)
from tests._oracles import rational
from tests.cell_support import (assert_covered, flat_box, flat_set, image_set,
                                pixel_box, reach_model, replay_onnx,
                                replay_torch, samples, save_onnx)
from tests.known_bugs import known_bug

# eps and running variances whose sums are powers of four: the fold is exact.
_EPS = 0.75
_VARIANCES = (0.25, 3.25, 15.25)
_GAMMAS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def _dyadic(model, seed=0):
    """``model`` in eval mode with dyadic parameters: weights in multiples of
    1/4 in [-1, 1], biases, running means and shifts in multiples of 1/2 in
    [-2, 2], scales gamma of both signs, running variances from
    ``_VARIANCES``. Every BatchNorm must have eps ``_EPS``, so that every
    running variance plus eps is 1, 4 or 16 and the fold is exact."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                m.weight.copy_(torch.as_tensor(
                    rng.integers(-4, 5, tuple(m.weight.shape)) / 4.0))
                if m.bias is not None:
                    m.bias.copy_(torch.as_tensor(
                        rng.integers(-4, 5, tuple(m.bias.shape)) / 2.0))
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                if m.eps != _EPS:
                    raise ValueError(f"a dyadic BatchNorm needs eps {_EPS}")
                n = m.num_features
                if m.running_mean is not None:
                    m.running_mean.copy_(torch.as_tensor(
                        rng.integers(-4, 5, n) / 2.0))
                    m.running_var.copy_(torch.as_tensor(
                        rng.choice(_VARIANCES, n)))
                if m.weight is not None:
                    m.weight.copy_(torch.as_tensor(rng.choice(_GAMMAS, n)))
                    m.bias.copy_(torch.as_tensor(rng.integers(-4, 5, n) / 2.0))
    return model.eval()


def _bn1d(n, **kw):
    return nn.BatchNorm1d(n, eps=_EPS, **kw)


def _bn2d(n, **kw):
    return nn.BatchNorm2d(n, eps=_EPS, **kw)


# --------------------------------------------------------------------------- #
# Models                                                                        #
# --------------------------------------------------------------------------- #
class _Residual(nn.Sequential):
    """A Sequential subclass whose forward adds its input to its children's
    output: rebuilt as a plain Sequential it loses the addition."""

    def forward(self, x):
        return x + super().forward(x)


class _OutputUsedTwice(nn.Module):
    """The Linear's output feeds the BatchNorm and the subtraction: a fused
    Linear would hand BN(fc(x)) to both."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 3)
        self.bn = _bn1d(3)

    def forward(self, x):
        c = self.fc(x)
        return self.bn(c) - 2.0 * c


class _LinearCalledTwice(nn.Module):
    """One Linear module called twice: fusing it with the BatchNorm after its
    first call changes the second call too."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 3)
        self.bn = _bn1d(3)

    def forward(self, x):
        return self.fc(torch.relu(self.bn(self.fc(x))))


class _BatchNormCalledTwice(nn.Module):
    """One BatchNorm module after two Linears: fusing it into the first and
    replacing it by an identity drops the normalization at its second call."""

    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(3, 3)
        self.f2 = nn.Linear(3, 3)
        self.bn = _bn1d(3)

    def forward(self, x):
        return self.bn(self.f2(torch.relu(self.bn(self.f1(x)))))


class _TwoBatchNorms(nn.Module):
    """Two BatchNorms after one Linear: the second folds into the already
    fused layer."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 4)
        self.b1 = _bn1d(4)
        self.b2 = _bn1d(4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(torch.relu(self.b2(self.b1(self.fc(x)))))


class _Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.bn = _bn1d(n_out)

    def forward(self, x):
        return self.bn(self.fc(x))


class _PairInAChild(nn.Module):
    """The pair sits in a child module, so the traced graph names it by a
    dotted path (``block.fc``, ``block.bn``)."""

    def __init__(self):
        super().__init__()
        self.block = _Block(3, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(torch.relu(self.block(x)))


class _BatchNormCalledByKeyword(nn.Module):
    """The BatchNorm's input is passed as the keyword ``input=``, so its
    graph node has no positional argument: the graph fusion must skip it (or
    read the keyword), not index a missing first argument."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 4)
        self.bn = _bn1d(4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(torch.relu(self.bn(input=self.fc(x))))


def _one_module_object_twice():
    """One Linear object and one ReLU object each used twice in a Sequential
    (weight sharing): ``nn.Module.children()`` yields each object once, so a
    rebuild from it drops the second uses."""
    act, shared = nn.ReLU(), nn.Linear(3, 3)
    return nn.Sequential(shared, _bn1d(3), act, shared, act, nn.Linear(3, 2))


class _BatchNormAfterAFunction(nn.Module):
    """The BatchNorm reads ``torch.flatten(torch.relu(fc(x)))``: two function
    nodes stand between it and the Linear, so folding it into the Linear
    would drop the ReLU."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 4)
        self.bn = _bn1d(4)

    def forward(self, x):
        return self.bn(torch.flatten(torch.relu(self.fc(x)), 1))


class _DoubledLinear(nn.Linear):
    """A Linear subclass whose forward is twice the stock one."""

    def forward(self, x):
        return 2.0 * super().forward(x)


class _DoubledWeightConv(nn.Conv2d):
    """A Conv2d subclass that transforms its weight in ``_conv_forward`` (as
    weight standardization does), here by doubling it."""

    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(x, 2.0 * weight, bias)


class _ShiftedBatchNorm(nn.BatchNorm1d):
    """A BatchNorm subclass whose forward adds 1 to the stock one."""

    def forward(self, x):
        return super().forward(x) + 1.0


def _linear_pair():
    return nn.Sequential(nn.Linear(3, 4), _bn1d(4), nn.ReLU(), nn.Linear(4, 2))


def _conv_pair():
    return nn.Sequential(
        nn.Conv2d(2, 3, (2, 3), stride=(1, 2), padding=(1, 1)), _bn2d(3),
        nn.ReLU(), nn.Flatten())


def _conv_options():
    return nn.Sequential(
        nn.Conv2d(2, 4, (2, 2), padding=1, dilation=(1, 2),
                  padding_mode="reflect"), _bn2d(4),
        nn.Conv2d(4, 4, 2, groups=4, bias=False), _bn2d(4, affine=False))


FLAT = (1, 3)
IMAGE = (1, 2, 3, 5)

# name -> (builder, input shape). Every builder returns a fresh float32 model.
MODELS = {
    "linear-pair-in-the-middle": (_linear_pair, FLAT),
    "last-pair-square-no-bias-no-affine": (lambda: nn.Sequential(
        nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 3, bias=False),
        _bn1d(3, affine=False)), FLAT),
    "pair-then-a-second-batchnorm": (lambda: nn.Sequential(
        nn.Linear(3, 4), _bn1d(4), _bn1d(4), nn.Linear(4, 2)), FLAT),
    "nested-sequentials": (lambda: nn.Sequential(
        nn.Sequential(nn.Linear(3, 4), _bn1d(4)), nn.ReLU(),
        nn.Sequential(nn.Linear(4, 2), _bn1d(2))), FLAT),
    "sequential-subclass": (lambda: _Residual(nn.Linear(3, 3), _bn1d(3)), FLAT),
    "sequential-subclass-nested": (lambda: nn.Sequential(
        nn.Linear(3, 3), _Residual(nn.Linear(3, 3), _bn1d(3)),
        nn.Linear(3, 2)), FLAT),
    "float64-network": (lambda: _linear_pair().double(), FLAT),
    "output-used-twice": (_OutputUsedTwice, FLAT),
    "linear-called-twice": (_LinearCalledTwice, FLAT),
    "batchnorm-called-twice": (_BatchNormCalledTwice, FLAT),
    "two-batchnorms-after-one-linear": (_TwoBatchNorms, FLAT),
    "two-batchnorms-traced": (lambda: torch.fx.symbolic_trace(
        _TwoBatchNorms()), FLAT),
    "pair-in-a-child-module": (_PairInAChild, FLAT),
    "linear-subclass-own-forward": (lambda: nn.Sequential(
        _DoubledLinear(3, 4), _bn1d(4)), FLAT),
    "batchnorm-subclass-own-forward": (lambda: nn.Sequential(
        nn.Linear(3, 4), _ShiftedBatchNorm(4, eps=_EPS)), FLAT),
    "conv-pair": (_conv_pair, IMAGE),
    "conv-groups-dilation-reflect-no-bias-no-affine": (_conv_options, IMAGE),
    "conv-grouped-circular": (lambda: nn.Sequential(
        nn.Conv2d(2, 4, 3, padding=1, groups=2, padding_mode="circular"),
        _bn2d(4)), IMAGE),
    "conv-subclass-own-conv-forward": (lambda: nn.Sequential(
        _DoubledWeightConv(2, 3, 2), _bn2d(3), nn.Flatten()), IMAGE),
    "batchnorm-after-a-function": (_BatchNormAfterAFunction, FLAT),
    "batchnorm-called-by-keyword": (_BatchNormCalledByKeyword, FLAT),
    "linear-on-a-rank-four-tensor-then-batchnorm2d": (lambda: nn.Sequential(
        nn.Linear(5, 4), _bn2d(2)), IMAGE),
    "conv1d-then-batchnorm1d": (lambda: nn.Sequential(
        nn.Conv1d(2, 3, 2), _bn1d(3)), (1, 2, 6)),
    "one-module-object-twice-in-a-sequential": (_one_module_object_twice,
                                                FLAT),
}

_FUSION_DROPS_A_SECOND_USE = known_bug(
    12, "fuse_batchnorm rebuilds an nn.Sequential from children(), which "
        "yields a module object held twice only once, so the second use of a "
        "shared layer is dropped: the fused model, and every reach of the "
        "original that fuses it first, compute another function")
MODEL_CASES = [pytest.param(name, marks=_FUSION_DROPS_A_SECOND_USE)
               if name == "one-module-object-twice-in-a-sequential" else name
               for name in MODELS]


def _build(name, seed=0):
    build, in_shape = MODELS[name]
    return _dyadic(build(), seed), in_shape


def _points(in_shape, seed=0, n=128):
    """``n`` dyadic points of the box [-8, 8] of ``in_shape`` (batch axis
    dropped): its two extreme corners, then points in multiples of 1/2."""
    shape = tuple(in_shape[1:])
    rng = np.random.default_rng(seed)
    return ([np.full(shape, -8.0), np.full(shape, 8.0)]
            + [rng.integers(-16, 17, size=shape) / 2.0 for _ in range(n - 2)])


# --------------------------------------------------------------------------- #
# fuse_batchnorm keeps the function                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", MODEL_CASES)
def test_fused_model_computes_the_original_function(name):
    """The fused model and the original, both evaluated in float64 on 128
    dyadic points of [-8, 8], agree exactly; the fused model run in the
    network's own dtype takes and returns that dtype and agrees exactly with
    the original run the same way. The data makes every fold and both forward
    passes exact, so any change of the function shows. Catches a fold that
    drops or misplaces a term (the layer's bias, the running mean, eps put
    outside the square root or replaced by its default, beta = 1 for
    affine=False, the scale broadcast over the wrong weight axis); a fused
    Conv2d rebuilt without stride, groups, dilation or padding mode; a
    BatchNorm left in place after its fold; a pair that must stay unfused and
    is fused (a second user of the layer's output, a layer or BatchNorm called
    twice, a subclass whose forward or ``_conv_forward`` is not the stock one,
    a BatchNorm behind function nodes, after a Conv1d or after a Linear on a
    rank-4 tensor, which also must not crash the fusion); a
    Sequential subclass rebuilt as a plain Sequential; a second BatchNorm
    folded into the original layer instead of the fused one; a fused layer
    stored under the wrong name of a nested module; a graph fusion that
    reads the missing positional argument of a BatchNorm called by keyword;
    and a fused layer that returns float64 inside a float32 network. The
    Sequential that holds one module object twice is marked: today its
    rebuild drops the second uses; the other Sequential cases are its
    passing neighbours."""
    model, in_shape = _build(name)
    points = _points(in_shape)
    reference = replay_torch(model, points, in_shape)
    fused = fuse_batchnorm(model)
    got = replay_torch(fused, points, in_shape)
    wrong = [k for k, (a, b) in enumerate(zip(got, reference))
             if not np.array_equal(a, b)]
    assert not wrong, (
        f"{name}: the fused model differs from the original on {len(wrong)} "
        f"of {len(points)} points; first point {wrong[0]}: fused "
        f"{got[wrong[0]]}, original {reference[wrong[0]]}")

    dtype = next(model.parameters()).dtype
    x = torch.as_tensor(np.stack(points).reshape((-1,) + tuple(in_shape[1:])),
                        dtype=dtype)
    with torch.no_grad():
        own, expected = fused(x), model(x)
    assert own.dtype == dtype, (f"{name}: the fused model returns {own.dtype} "
                                f"for a {dtype} network")
    assert torch.equal(own, expected), (
        f"{name}: run in {dtype}, the fused model differs from the original "
        f"by up to {(own - expected).abs().max().item():.3g}")


@pytest.mark.parametrize("dtype", [
    pytest.param(torch.float32, id="float32-network"),
    pytest.param(torch.float64, id="float64-network-not-float32-numbers")])
def test_fold_is_rounded_once_in_float64(dtype):
    """The folded weights and biases are float64 and within the rounding of
    one float64 evaluation of gamma / sqrt(var + eps) times W and
    scale (b - mean) + beta, against a 60-digit fold of the stored
    parameters, on non-dyadic data: a Conv2d with BatchNorm2d, a Linear with
    BatchNorm1d of eps 1e-3, and a Linear without bias with an affine=False
    BatchNorm1d. The bound is 8 units of 2^-53 relative to the folded weight,
    and for the bias relative to |scale (b - mean)| + |beta| (the docstring's
    measure); the first-order rounding error of the code is 3.5 and 5.5 such
    units. The float64 network's parameters are full float64 numbers, not
    float32 ones, and the running means of the Linear's BatchNorm lie near
    plus and minus 1e4, so that a parameter read through float32 moves its
    fold by about 1e-3. Catches a fold computed or stored in float32 (an
    error near 2^-24 relative), a float64 parameter read through float32
    (which a float32 network cannot show), eps replaced by its default, and a
    fold whose rounding is larger than one evaluation."""
    rng = np.random.default_rng(11)
    model = nn.Sequential(
        nn.Conv2d(2, 3, 2), nn.BatchNorm2d(3), nn.Flatten(),
        nn.Linear(3 * 2 * 4, 4), nn.BatchNorm1d(4, eps=1e-3), nn.ReLU(),
        nn.Linear(4, 2, bias=False), nn.BatchNorm1d(2, affine=False)
    ).to(dtype)
    offset = {4: 1e4} if dtype == torch.float64 else {}
    with torch.no_grad():
        for k, m in enumerate(model):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.copy_(torch.as_tensor(
                    rng.uniform(-2, 2, tuple(m.weight.shape))))
                if m.bias is not None:
                    m.bias.copy_(torch.as_tensor(
                        rng.uniform(-3, 3, tuple(m.bias.shape))))
            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                n = m.num_features
                big = offset.get(k, 0.0) * np.where(np.arange(n) % 2, 1, -1)
                m.running_mean.copy_(torch.as_tensor(
                    rng.uniform(-5, 5, n) + big))
                m.running_var.copy_(torch.as_tensor(rng.uniform(0.01, 5, n)))
                if m.weight is not None:
                    m.weight.copy_(torch.as_tensor(rng.uniform(-3, 3, n)))
                    m.bias.copy_(torch.as_tensor(rng.uniform(-2, 2, n)))
    model.eval()
    if dtype == torch.float64 and all(
            float(np.float32(v)) == v
            for v in model[4].running_mean.tolist()):
        pytest.fail("setup: the float64 running means are float32 numbers")
    fused = fuse_batchnorm(model)
    unit = Fraction(1, 2 ** 53)
    for k_layer, k_bn in ((0, 1), (3, 4), (6, 7)):
        layer, bn, got = model[k_layer], model[k_bn], fused[k_layer]
        assert got.weight.dtype == torch.float64 and \
            got.bias.dtype == torch.float64, (
                f"layer {k_layer}: fused parameters are {got.weight.dtype}")
        dec = lambda t: [Decimal(float(v)) for v in t.detach().reshape(-1)]  # noqa: E731
        n = bn.num_features
        weight = dec(layer.weight)
        per_row = len(weight) // n
        bias = dec(layer.bias) if layer.bias is not None else [Decimal(0)] * n
        gamma = dec(bn.weight) if bn.weight is not None else [Decimal(1)] * n
        beta = dec(bn.bias) if bn.bias is not None else [Decimal(0)] * n
        mean, var = dec(bn.running_mean), dec(bn.running_var)
        got_w = got.weight.detach().reshape(-1).tolist()
        got_b = got.bias.detach().tolist()
        with localcontext() as ctx:
            ctx.prec = 60
            for i in range(n):
                scale = gamma[i] / (var[i] + Decimal(bn.eps)).sqrt()
                for j in range(per_row):
                    exact = Fraction(scale * weight[i * per_row + j])
                    miss = abs(Fraction(got_w[i * per_row + j]) - exact)
                    assert miss <= 8 * unit * abs(exact), (
                        f"layer {k_layer}, weight ({i}, {j}): off by "
                        f"{float(miss / abs(exact)):.3g} relative")
                folded = scale * (bias[i] - mean[i])
                exact = Fraction(folded + beta[i])
                miss = abs(Fraction(got_b[i]) - exact)
                size = Fraction(abs(folded)) + Fraction(abs(beta[i]))
                assert miss <= 8 * unit * size, (
                    f"layer {k_layer}, bias {i}: off by "
                    f"{float(miss / size):.3g} of the folded terms")


# --------------------------------------------------------------------------- #
# Reach through the model entry holds the outputs of the original model        #
# --------------------------------------------------------------------------- #
def _input_box(in_shape, seed=0):
    """Dyadic bounds of the input, scaled so that they reach well beyond
    plus and minus 1 and several intervals are wider than 2."""
    if len(in_shape) == 4:
        lo, hi = pixel_box(*in_shape[1:], seed=seed)
        return 4 * lo, 4 * hi
    lo, hi = flat_box(int(np.prod(in_shape[1:])), seed=seed)
    return 8 * lo, 8 * hi


def _callers_view(model):
    """What the caller can see of a model: module classes by name, training
    flags, hooks, parameters and buffers, and the graph's code."""
    return ([(n, type(m).__name__, m.training, len(m._forward_hooks),
              len(m._forward_pre_hooks)) for n, m in model.named_modules()],
            {k: v.clone() for k, v in model.state_dict().items()},
            getattr(model, "code", None))


def _assert_same_view(before, after, what):
    if before[0] != after[0] or before[2] != after[2] \
            or before[1].keys() != after[1].keys() \
            or not all(torch.equal(v, after[1][k])
                       for k, v in before[1].items()):
        pytest.fail(f"{what}: the reach changed the caller's model")


@pytest.mark.parametrize("name, kind, training", [
    ("linear-pair-in-the-middle", "Star", False),
    ("linear-pair-in-the-middle", "Zono", False),
    ("linear-pair-in-the-middle", "Hexatope", False),
    ("linear-pair-in-the-middle", "Octatope", False),
    ("linear-pair-in-the-middle", "Star", True),
    ("last-pair-square-no-bias-no-affine", "Star", False),
    ("nested-sequentials", "Zono", False),
    ("two-batchnorms-after-one-linear", "Star", False),
    ("two-batchnorms-traced", "Zono", False),
    ("batchnorm-called-twice", "Hexatope", False),
    ("pair-in-a-child-module", "Star", False),
    ("conv-pair", "ImageStar", False),
    ("conv-pair", "ImageZono", False),
    ("conv-groups-dilation-reflect-no-bias-no-affine", "ImageStar", False),
    pytest.param("one-module-object-twice-in-a-sequential", "Star", False,
                 marks=_FUSION_DROPS_A_SECOND_USE),
])
def test_reach_holds_every_output_of_the_original_model(name, kind, training):
    """``reach_pytorch_model`` (method approx), which fuses the model first
    for every set type but Box, on a box reaching well beyond plus and minus
    1: 128 points replayed on the original model in float64 are members of
    the returned sets by the LP on their rows, and the caller's model comes
    back as it was given (classes, training flags, hooks, parameters, graph).
    A model handed over in training mode is reached for its eval-mode
    function, which ``fuse_batchnorm`` documents. Catches every change of the
    function by the fusion on the reach route, a fusion that works on the
    caller's model in place, and a fused layer whose parameters the kernels
    read differently from its forward. The Sequential that holds one module
    object twice is marked (its fusion drops the second uses, so the reach
    answers for another network); ``linear-pair-in-the-middle`` on a Star is
    its passing neighbour."""
    model, in_shape = _build(name)
    if training:
        model.train()
    lo, hi = _input_box(in_shape)
    image = len(in_shape) == 4
    input_set = image_set(kind, lo, hi) if image else flat_set(kind, lo, hi)
    reference = replay_torch(model, samples(lo, hi), in_shape)
    before = _callers_view(model)
    what = f"{name} on a {kind}{' in training mode' if training else ''}"
    options = {"input_shape": in_shape} if image else {}
    sets = reach_model(model, input_set, "approx", what, **options)
    _assert_same_view(before, _callers_view(model), what)
    out_shape = tuple(model.eval()(torch.zeros(in_shape)).shape[1:])
    out_chw = out_shape if len(out_shape) == 3 else None
    assert_covered(sets, reference, what, out_chw=out_chw)


def _exact_sqrt(q):
    """The square root of a Fraction that is the square of one; a setup
    failure otherwise."""
    root = []
    for v in (q.numerator, q.denominator):
        r = int(np.sqrt(v))
        while r * r > v:
            r -= 1
        while (r + 1) * (r + 1) <= v:
            r += 1
        if r * r != v:
            pytest.fail(f"setup: {q} is not the square of a rational")
        root.append(r)
    return Fraction(root[0], root[1])


def test_exact_star_reach_of_a_fused_model_is_the_exact_image():
    """Exact Star reach of Linear(3, 4), BatchNorm1d, ReLU, Linear(4, 2) on a
    box of [-8, 8]-scale bounds, against the exact image of the original
    model in rationals (the BatchNorm as its own diagonal map, gamma /
    sqrt(var + eps) exact on this data): every vertex of every exact piece is
    in a returned set with zero slack, every vertex of every returned set has
    a preimage, and 128 points replayed in float64 are covered. Catches a fold
    whose error is too small for the replay's slack and a fused model whose
    exact reach returns points the original model never outputs."""
    model, in_shape = _build("linear-pair-in-the-middle", seed=3)
    lo, hi = _input_box(in_shape, seed=3)
    sets = reach_model(model, flat_set("Star", lo, hi), "exact",
                       "exact Star reach of a fused Linear and BatchNorm")
    if len(sets) > 16:
        pytest.fail(f"{len(sets)} sets for 4 ReLU neurons")
    fc, bn, head = model[0], model[1], model[3]
    frac = lambda t: [[Fraction(float(v)) for v in row]  # noqa: E731
                      for row in t.detach().double().reshape(t.shape[0], -1)]
    vec = lambda t: [Fraction(float(v)) for v in t.detach().double()]  # noqa: E731
    gamma, beta = vec(bn.weight), vec(bn.bias)
    mean, var = vec(bn.running_mean), vec(bn.running_var)
    scale = [g / _exact_sqrt(v + Fraction(bn.eps)) for g, v in zip(gamma, var)]
    n = len(scale)
    diagonal = [[scale[i] if i == j else Fraction(0) for j in range(n)]
                for i in range(n)]
    shift = [b - s * m for b, s, m in zip(beta, scale, mean)]
    pieces = rational.affine(rational.box(lo, hi), frac(fc.weight),
                             vec(fc.bias))
    pieces = rational.relu(rational.affine(pieces, diagonal, shift))
    pieces = rational.affine(pieces, frac(head.weight), vec(head.bias))
    computed = [rational.star(s) for s in sets]
    escaped = rational.escapes(pieces, *computed)
    assert not escaped, (f"{len(escaped)} vertices of the exact image are in "
                         f"no returned set, first "
                         f"{[float(v) for v in escaped[0][1]]}")
    assert not rational.straddlers(pieces, *computed), (
        "a piece of the exact image is spread over several returned sets")
    for k, s in enumerate(computed):
        extra = rational.unreached(pieces, rational.set_vertices(s))
        assert not extra, (f"set {k}: {len(extra)} vertices have no preimage, "
                           f"first {[float(v) for v in extra[0][1]]}")
    assert_covered(sets, replay_torch(model, samples(lo, hi), in_shape),
                   "exact Star reach of a fused Linear and BatchNorm")


def _onnx_residual(tmp_path):
    """Conv, BatchNormalization, Relu, Conv, BatchNormalization and an Add of
    the second BatchNormalization's output and the second Conv's output, all
    dyadic: the first pair fuses, the second conv has two users and must not.
    The second conv is 1x1 and the scales are at most 1 in magnitude, so
    every value stays below 2^24 in units of its own last place and
    onnxruntime's float32 evaluation is exact on dyadic points of the
    unscaled ``pixel_box``. Returns the file's path, the input shape and the
    output (C, H, W)."""
    rng = np.random.default_rng(7)
    c, h, w = 2, 3, 5
    inits = {"w1": rng.integers(-4, 5, (3, c, 2, 2)) / 4.0,
             "b1": rng.integers(-4, 5, 3) / 2.0,
             "w2": rng.integers(-4, 5, (3, 3, 1, 1)) / 4.0,
             "b2": rng.integers(-4, 5, 3) / 2.0}
    for k in ("1", "2"):
        inits["s" + k] = rng.choice((-1.0, -0.5, 0.5, 1.0), 3)
        inits["o" + k] = rng.integers(-4, 5, 3) / 2.0
        inits["m" + k] = rng.integers(-4, 5, 3) / 2.0
        inits["v" + k] = rng.choice(_VARIANCES, 3)
    inits = {k: np.asarray(v, dtype=np.float32) for k, v in inits.items()}
    nodes = [
        helper.make_node("Conv", ["X", "w1", "b1"], ["c1"], kernel_shape=[2, 2]),
        helper.make_node("BatchNormalization", ["c1", "s1", "o1", "m1", "v1"],
                         ["n1"], epsilon=_EPS),
        helper.make_node("Relu", ["n1"], ["r1"]),
        helper.make_node("Conv", ["r1", "w2", "b2"], ["c2"], kernel_shape=[1, 1]),
        helper.make_node("BatchNormalization", ["c2", "s2", "o2", "m2", "v2"],
                         ["n2"], epsilon=_EPS),
        helper.make_node("Add", ["n2", "c2"], ["Y"])]
    out = (3, h - 1, w - 1)
    path = save_onnx(tmp_path, nodes, [1, c, h, w], [1, *out], inits=inits)
    return path, (1, c, h, w), out


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_onnx_model_with_batchnorm_is_fused_and_reached_soundly(tmp_path, kind):
    """An ONNX model loaded by onnx2torch (a GraphModule, fused by the graph
    path) with one fusable pair and one conv whose output also feeds a
    residual Add: the fused model agrees exactly with onnxruntime on 128
    dyadic points, and the reach on the image set holds those 128 onnxruntime
    outputs. Catches a fusion of the residual conv (the Add then reads the
    normalized output twice) and every change of the fused function on the
    route a competition model takes."""
    path, in_shape, out_chw = _onnx_residual(tmp_path)
    lo, hi = pixel_box(*in_shape[1:])
    points = samples(lo, hi)
    reference = replay_onnx(path, points, in_shape)
    model = load_onnx(path)
    fused = fuse_batchnorm(model)
    with torch.no_grad():
        got = [fused(torch.as_tensor(p, dtype=torch.float32).reshape(in_shape))
               .double().numpy().reshape(-1) for p in points]
    wrong = sum(not np.array_equal(a, b) for a, b in zip(got, reference))
    assert wrong == 0, (f"the fused ONNX model differs from onnxruntime on "
                        f"{wrong} of {len(points)} points")
    what = f"ONNX Conv/BatchNormalization/Add on a {kind}"
    sets = reach_model(model, image_set(kind, lo, hi), "approx", what,
                       input_shape=in_shape)
    assert_covered(sets, reference, what, out_chw=out_chw)


# --------------------------------------------------------------------------- #
# Refusals on the reach route                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["Star", "ImageStar"])
def test_batchnorm_without_running_statistics_is_refused(kind):
    """A BatchNorm built with track_running_stats=False normalizes by the
    statistics of each batch even in eval mode, so it is no affine layer: the
    reach refuses it with NotImplementedError. Catches a fusion that folds it
    (it fails on the missing running variance with AttributeError)."""
    if kind == "Star":
        model = nn.Sequential(nn.Linear(3, 4),
                              nn.BatchNorm1d(4, track_running_stats=False))
        input_set = flat_set(kind, -np.ones(3), np.ones(3))
        options = {}
    else:
        model = nn.Sequential(nn.Conv2d(2, 3, 2),
                              nn.BatchNorm2d(3, track_running_stats=False))
        lo, hi = pixel_box(2, 3, 5)
        input_set = image_set(kind, lo, hi)
        options = {"input_shape": IMAGE}
    with pytest.raises(NotImplementedError):
        reach_model(model.eval(), input_set, "approx",
                    f"BatchNorm without running statistics on a {kind}",
                    **options)


_REACH_TRAINS_THE_CALLERS_BATCHNORM = known_bug(
    12, "registry SHAPE-9: the tensor-shape pass runs the caller's module "
        "forward on its placeholder; a BatchNorm in training mode updates its "
        "running statistics there, so a Box reach (BatchNorm kept unfused) "
        "answers for the drifted statistics and leaves the caller's model "
        "changed")


@pytest.mark.parametrize("training", [
    pytest.param(False, id="eval-mode"),
    pytest.param(True, id="training-mode",
                 marks=_REACH_TRAINS_THE_CALLERS_BATCHNORM)])
def test_box_reach_of_a_batchnorm_model_holds_its_outputs_and_keeps_it(
        training):
    """A Box reach keeps BatchNorm unfused (``reach.py``), so a model handed
    over in training mode reaches the caller's own BatchNorm2d. The reach
    must hold 128 outputs of the model as handed over (its eval-mode
    function, which ``fuse_batchnorm`` documents and the Box rule reads from
    the running statistics) and leave the model unchanged. The eval-mode case
    is the passing neighbour, and the training-mode Star cell of
    ``test_reach_holds_every_output_of_the_original_model`` passes because the
    fused copy is in eval mode. Catches a reach that runs the caller's
    BatchNorm in training mode."""
    model = _dyadic(nn.Sequential(_bn2d(2), nn.Flatten())).train(training)
    in_shape = (1, 2, 2, 3)
    lo, hi = _input_box(in_shape)
    reference = replay_torch(model, samples(lo, hi), in_shape)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    what = f"BatchNorm2d on a Box, model in {'training' if training else 'eval'} mode"
    sets = reach_model(model, flat_set("Box", lo, hi), "approx", what,
                       input_shape=in_shape)
    assert_covered(sets, reference, what)
    changed = [k for k, v in state.items()
               if not torch.equal(v, model.state_dict()[k])]
    assert not changed, f"{what}: the reach changed the caller's {changed}"
    assert model.training == training


@pytest.mark.parametrize("out_features", [
    pytest.param(2, id="channels-equal-features"),
    pytest.param(4, id="channels-differ-from-features", marks=known_bug(
        23, "fuse_batchnorm folds a BatchNorm1d whose channel count differs "
            "from the Linear's output features (legal on a rank-3 tensor, "
            "where the BatchNorm normalizes axis 1) and the fold raises a "
            "torch RuntimeError, so reach crashes where its docstring "
            "promises a refusal", raises=RuntimeError))])
def test_linear_then_batchnorm1d_on_a_rank_three_tensor_refuses_or_holds_every_output(
        out_features):
    """Linear(4, out_features) then BatchNorm1d(2) on a (1, 2, 4) tensor: the
    Linear maps the last axis and the BatchNorm normalizes axis 1, so the
    pair is not a fold of the Linear's output features. The ``fuse_batchnorm``
    docstring says reach refuses such a model; the promise checked is
    "refuses (NotImplementedError or ValueError), or every output replayed on
    the original model is covered". With 2 output features the fold is done
    and computes another function (documented) and the Linear kernel refuses
    the rank-3 set; with 4 the fold itself crashes. Catches a reach that
    answers for the folded function."""
    model = _dyadic(nn.Sequential(nn.Linear(4, out_features), _bn1d(2)))
    in_shape = (1, 2, 4)
    lo, hi = _input_box(in_shape)
    what = f"Linear(4, {out_features}) then BatchNorm1d(2) on a rank-3 tensor"
    try:
        sets = reach_model(model, flat_set("Star", lo, hi), "approx", what,
                           input_shape=in_shape)
    except (NotImplementedError, ValueError):
        return
    assert_covered(sets, replay_torch(model, samples(lo, hi), in_shape), what)


# --------------------------------------------------------------------------- #
# strip_final_softmax                                                           #
# --------------------------------------------------------------------------- #
class _Net(nn.Module):
    """Linear(4, 6), ReLU, Linear(6, 3), then ``tail(self, logits)``;
    ``logits`` is the original model's own computation of the logits."""

    def __init__(self, tail):
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 3)
        self.tail = tail

    def logits(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

    def forward(self, x):
        return self.tail(self, self.logits(x))


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.sm = nn.Softmax(dim=-1)


class _SoftmaxInAChild(nn.Module):
    """The trailing softmax module sits in a child (graph target
    ``head.sm``)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.head = _Head()

    def logits(self, x):
        return self.fc(x)

    def forward(self, x):
        return self.head.sm(self.fc(x))


class _SoftmaxInTheMiddle(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.sm = nn.Softmax(dim=-1)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x):
        return self.fc2(self.sm(self.fc1(x)))


def _graph(tail):
    """A traced ``_Net`` ending in ``tail`` and the original's logits."""
    net = _Net(tail)
    return torch.fx.symbolic_trace(net), net.logits


def _graph_module_tail(module):
    """A traced ``_Net`` whose tail is the module call ``module(logits)``."""
    net = _Net(lambda self, z: self.tail_module(z))
    net.tail_module = module
    return torch.fx.symbolic_trace(net), net.logits


def _graph_softmax_in_a_child():
    net = _SoftmaxInAChild()
    return torch.fx.symbolic_trace(net), net.logits


def _sequential_with(*tail):
    """``Sequential(Linear, ReLU, Linear, *tail)`` and the original's own
    first three layers as its logits."""
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3), *tail)
    return model, nn.Sequential(*list(model)[:3])


def _onnx_classifier(tmp_path, op, opset):
    """Gemm then an ONNX ``op`` (Softmax or LogSoftmax) over axis 1, with
    dyadic initializers, loaded by onnx2torch (opset 11 gives the
    OnnxSoftmaxV1V11 wrapper, opset 13 an nn.Softmax). The logits are the
    Gemm computed in numpy in float64 from the initializers, exact on dyadic
    points."""
    rng = np.random.default_rng(opset)
    w = (rng.integers(-8, 9, (4, 3)) / 4.0).astype(np.float32)
    b = (rng.integers(-4, 5, 3) / 2.0).astype(np.float32)
    path = save_onnx(tmp_path, [
        helper.make_node("Gemm", ["X", "W", "B"], ["H"]),
        helper.make_node(op, ["H"], ["Y"], axis=1)], [1, 4], [1, 3],
        inits={"W": w, "B": b}, name=f"{op}{opset}", opset=opset)

    def logits(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float64)
                               @ w.astype(np.float64) + b.astype(np.float64))
    return load_onnx(path), logits


# Models whose output is a plain softmax of logits: stripped, they must output
# exactly the logits. Each builder takes tmp_path and returns (model, logits).
STRIPPED = {
    "sequential-softmax-module": lambda tmp_path: _sequential_with(
        nn.Softmax(dim=-1)),
    "graph-softmax-module": lambda tmp_path: _graph_module_tail(
        nn.Softmax(dim=1)),
    "graph-softmax-module-in-a-child": lambda tmp_path:
        _graph_softmax_in_a_child(),
    "graph-torch-softmax": lambda tmp_path: _graph(
        lambda net, z: torch.softmax(z, -1)),
    "graph-torch-softmax-input-keyword": lambda tmp_path: _graph(
        lambda net, z: torch.softmax(input=z, dim=-1)),
    "graph-method-softmax": lambda tmp_path: _graph(
        lambda net, z: z.softmax(-1)),
    "onnx-opset11-softmax": lambda tmp_path: _onnx_classifier(
        tmp_path, "Softmax", 11),
    "onnx-opset13-softmax": lambda tmp_path: _onnx_classifier(
        tmp_path, "Softmax", 13),
}

# Models whose last operation is not a plain softmax: stripped, they must
# compute what they computed. Each builder takes tmp_path, returns the model.
KEPT = {
    "sequential-log-softmax": lambda tmp_path: _sequential_with(
        nn.LogSoftmax(dim=-1))[0],
    "sequential-softmax-in-the-middle": lambda tmp_path: nn.Sequential(
        nn.Linear(4, 6), nn.Softmax(dim=-1), nn.Linear(6, 3)),
    "sequential-empty": lambda tmp_path: nn.Sequential(),
    "graph-log-softmax-module": lambda tmp_path: _graph_module_tail(
        nn.LogSoftmax(dim=1))[0],
    "graph-functional-log-softmax": lambda tmp_path: _graph(
        lambda net, z: F.log_softmax(z, -1))[0],
    "graph-method-log-softmax": lambda tmp_path: _graph(
        lambda net, z: z.log_softmax(-1))[0],
    "graph-softmax-in-the-middle": lambda tmp_path: torch.fx.symbolic_trace(
        _SoftmaxInTheMiddle()),
    "graph-two-outputs": lambda tmp_path: _graph(
        lambda net, z: (z, torch.softmax(z, -1)))[0],
    "onnx-opset11-log-softmax": lambda tmp_path: _onnx_classifier(
        tmp_path, "LogSoftmax", 11)[0],
    "sequential-subclass-residual": lambda tmp_path: _Residual(
        nn.Linear(4, 4), nn.Softmax(dim=-1)),
}

_STRIP_DROPS_THE_FORWARD = known_bug(
    12, "strip_final_softmax rebuilds an nn.Sequential subclass that ends in "
        "a softmax child as a plain nn.Sequential without that child, which "
        "drops the subclass's own forward: x + softmax(f(x)) becomes f(x), "
        "so a verification of the stripped model answers for another "
        "function")
KEPT_CASES = [pytest.param(name, marks=_STRIP_DROPS_THE_FORWARD)
              if name == "sequential-subclass-residual" else name
              for name in KEPT]


def _strip_input(seed=0):
    """16 dyadic points of [-4, 4]^4 in multiples of 1/4."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-16, 17, (16, 4), generator=generator) / 4.0


@pytest.mark.parametrize("name", list(STRIPPED))
def test_strip_final_softmax_returns_exactly_the_logits(tmp_path, name):
    """A model whose output is a plain softmax of logits, as an
    nn.Sequential, a traced GraphModule (the softmax as a module at top level
    or in a child, as ``torch.softmax``, as ``F.softmax`` with its input
    given by keyword, as the tensor method) or an ONNX model loaded by
    onnx2torch (the opset 11 wrapper and the opset 13 module): the stripped
    model returns exactly the logits computed by the original model's own
    layers, and no softmax is left in it. fx records ``torch.softmax(input=
    z)`` with its input as a keyword (``F.softmax(input=z)`` is recorded
    positionally, so only the former reaches the keyword branch). Catches a
    softmax form that is not recognized (the model keeps returning
    probabilities where a logit specification is checked), the keyword input
    not read, and a softmax module left registered after its node is gone."""
    model, logits = STRIPPED[name](tmp_path)
    model.eval()
    x = _strip_input()
    with torch.no_grad():
        expected = logits(x)
        stripped = strip_final_softmax(model)
        got = stripped(x)
    assert torch.equal(got.to(expected.dtype), expected), (
        f"{name}: the stripped model differs from the logits by up to "
        f"{(got.double() - expected.double()).abs().max().item():.3g}")
    assert not has_softmax(stripped), f"{name}: a softmax is left"


@pytest.mark.parametrize("name", KEPT_CASES)
def test_strip_final_softmax_keeps_a_model_without_a_final_softmax(
        tmp_path, name):
    """A model whose last operation is not a plain softmax (a LogSoftmax as
    a module, a function, a tensor method or the opset 11 ONNX wrapper; a
    softmax in the middle; an empty Sequential; a graph with two outputs; a
    Sequential subclass that adds its input after its children) computes
    exactly what it
    computed before, on 16 dyadic points of [-4, 4]. Catches a LogSoftmax
    taken for a softmax (the model would return logits where log-probabilities
    are specified), a softmax dropped from the middle of a network, a crash
    on an empty Sequential, a graph that loses one of its outputs, and a
    Sequential subclass rebuilt as a plain one."""
    model = KEPT[name](tmp_path).eval()
    x = _strip_input(1)
    with torch.no_grad():
        expected = model(x)
        got = strip_final_softmax(model)(x)
    if isinstance(expected, tuple):
        same = len(got) == len(expected) and all(
            torch.equal(a, b) for a, b in zip(got, expected))
    else:
        same = torch.equal(got, expected)
    assert same, f"{name}: stripping changed a model without a final softmax"


def test_reach_of_a_stripped_onnx_model_holds_its_logits(tmp_path):
    """The workflow the docstring describes: strip the opset 11 softmax of an
    ONNX classifier, then reach the stripped model on a Star; 128 logits
    computed from the initializers in float64 are covered. Catches a stripped
    graph that still applies the softmax, or that the reach reads
    differently from its forward."""
    model, logits = _onnx_classifier(tmp_path, "Softmax", 11)
    lo, hi = _input_box((1, 4))
    points = samples(lo, hi)
    reference = [logits(p).numpy().reshape(-1) for p in points]
    sets = reach_model(strip_final_softmax(model), flat_set("Star", lo, hi),
                       "approx", "stripped ONNX classifier on a Star")
    assert_covered(sets, reference, "stripped ONNX classifier on a Star")

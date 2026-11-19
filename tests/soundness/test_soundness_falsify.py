"""Soundness cells for the counterexamples of ``n2v.utils.falsify``.

Component: ``n2v/utils/falsify.py``, the public ``falsify`` with its final
check of a counterexample, the random falsifier ``_falsify_random``, the
property readers ``_extract_halfspace_groups`` and
``_output_satisfies_property``, and ``_detect_model_device``. The attacks
(PGD, APGD, Square and their
cascades) are tested elsewhere; here they only produce the points that the
final check must judge.

The promise: a counterexample ``(x, y)`` that ``falsify`` returns is real.
``x`` lies in the float64 box ``[lb, ub]`` in every coordinate, compared
exactly; ``y`` is the model's own output at ``x`` at batch size 1; and the
model replayed on ``x`` in float64 (``replay_torch``) lies in the unsafe
region, decided by this file's own reading of the property in rationals at
zero tolerance (``_unsafe``), never by n2v's checker. When no point of the box
is unsafe the answer is ``(2, None)`` for every method and seed: ``falsify``
never claims safety.

Models: float32 models with dyadic weights in which every output reads one
input times a power of two (or a ReLU of it), so the float32 forward, the
float64 replay and the rational value coincide. Every attack builds float32
tensors, so a float64 model crashes inside the attacks (registry VERDICT-2,
marked in ``tests/contracts/test_verify_api.py``); the cells that need a
float64 model replace the attack by a stub of ``_falsify_impl``, which leaves
the final check, the code under test, to judge the point the stub hands it.
"""

import copy
import importlib
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import HalfSpace, Star
from n2v.utils import falsify
from n2v.utils.verify_specification import verify_specification
from tests.cell_support import pixel_box, replay_torch
from tests.known_bugs import known_bug

# a plain ``import n2v.utils.falsify`` resolves to the re-exported function
fz = importlib.import_module("n2v.utils.falsify")

# Every method name falsify accepts except the AutoAttack scaffold, which
# needs a package that is not installed.
METHODS = ["random", "pgd", "apgd", "square", "strong", "random+pgd",
           "random+pgd+apgd", "random+square", "random+apgd"]

# Small budgets: every cell here is about the judgement of a point, not about
# how hard an attack searches.
BUDGET = dict(n_samples=200, n_restarts=3, n_steps=30, n_iters=600, batch=64)


# --------------------------------------------------------------------------- #
# The property, written once as plain data and read independently            #
# --------------------------------------------------------------------------- #
# A property in plain form is a list of groups (AND); a group is a list of
# half-spaces (OR); a half-space is (rows, rhs) and holds when every row holds
# (AND): sum_k rows[i][k] * y[k] <= rhs[i].

def _hs(spec):
    rows, rhs = spec
    return HalfSpace(np.array(rows, dtype=np.float64),
                     np.array(rhs, dtype=np.float64).reshape(-1, 1))


def _holds(spec, y):
    rows, rhs = spec
    return all(sum(Fraction(a) * Fraction(float(v)) for a, v in zip(row, y))
               <= Fraction(b) for row, b in zip(rows, rhs))


def _unsafe(groups, y):
    """The meaning of the property at the point ``y``, in rationals, with no
    tolerance: every group holds, a group holds when one of its half-spaces
    holds, a half-space holds when every one of its rows holds."""
    return all(any(_holds(spec, y) for spec in group) for group in groups)


def _linear(weight, bias=None):
    """A float32 ``nn.Sequential(nn.Linear)`` with the given weights."""
    weight = np.asarray(weight, dtype=np.float64)
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
        if bias is not None:
            layer.bias.copy_(torch.as_tensor(np.asarray(bias),
                                             dtype=torch.float32))
    return nn.Sequential(layer)


def _own_output(model, x, in_shape):
    """The model's own output at ``x`` at batch size 1, in the dtype of its
    parameters (float32 when it has none), as ``falsify`` documents it."""
    params = list(model.parameters())
    dtype = params[0].dtype if params else torch.float32
    m = copy.deepcopy(model).eval()
    with torch.no_grad():
        y = m(torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dtype)
              .reshape(in_shape))
    return y.detach().cpu().numpy().reshape(-1)


def _assert_real(model, lb, ub, groups, result, cex, what):
    """The counterexample is real: ``x`` in the float64 box (exact), ``y`` the
    model's own batch-1 output at ``x``, and the float64 replay at ``x`` in
    the unsafe region by the plain reading of the property."""
    assert result == 0 and cex is not None, \
        f"{what}: expected a counterexample, got {(result, cex)}"
    lb64 = np.asarray(lb, dtype=np.float64)
    ub64 = np.asarray(ub, dtype=np.float64)
    in_shape = (1, *lb64.shape)
    x = np.asarray(cex[0], dtype=np.float64).reshape(-1)
    assert x.size == lb64.size, f"{what}: witness of size {x.size}"
    outside = np.nonzero((x < lb64.reshape(-1)) | (x > ub64.reshape(-1)))[0]
    assert outside.size == 0, (
        f"{what}: witness outside the float64 box at {outside.tolist()}: "
        f"x={x[outside].tolist()} lb={lb64.reshape(-1)[outside].tolist()} "
        f"ub={ub64.reshape(-1)[outside].tolist()}")
    y = np.asarray(cex[1]).reshape(-1)
    y_own = _own_output(model, x, in_shape)
    assert np.array_equal(y, y_own), \
        f"{what}: returned y {y.tolist()} is not the model's batch-1 output " \
        f"{y_own.tolist()} at the returned x"
    y64 = replay_torch(model, [x], in_shape)[0]
    assert _unsafe(groups, y64), \
        f"{what}: replayed output {y64.tolist()} at x={x.tolist()} is not " \
        f"in the unsafe region"


# --------------------------------------------------------------------------- #
# A box that touches the unsafe region in one corner                           #
# --------------------------------------------------------------------------- #
# y = (2 x0, x1 / 2). The unsafe region is y0 >= 2 a and y1 <= b / 2, which
# the box [l0, a] x [b, u1] meets only at its corner (a, b). "rounded": a = 0.3
# and b = 0.7 are no float32 numbers; float32(0.3) lies above 0.3 and
# float32(0.7) below 0.7, so the float32 attacks reach the corner from outside
# the user's box. "dyadic": the corner is exact, the output lies exactly on
# the boundary of the closed region.
_CORNERS = {
    "rounded": (np.array([0.0, 0.7]), np.array([0.3, 1.0])),
    "dyadic": (np.array([-1.0, -1.5]), np.array([1.25, 2.0])),
}


@pytest.mark.parametrize("method", ["apgd", "square"])
@pytest.mark.parametrize("corner", ["rounded", "dyadic"])
def test_a_box_that_touches_the_unsafe_region_in_one_corner(corner, method):
    """The only counterexample is the corner. Catches a final check that does
    not snap the attack's float32 point into the float64 box at either end
    (or returns the unsnapped point), and a check that reads the closed
    region as open (strict comparison) and so misses the exact corner."""
    lb, ub = _CORNERS[corner]
    a, b = ub[0], lb[1]
    model = _linear([[2.0, 0.0], [0.0, 0.5]])
    spec = ([[-1.0, 0.0], [0.0, 1.0]], [-2.0 * a, b / 2.0])
    result, cex = falsify(model, lb, ub, _hs(spec), method=method, seed=0,
                          **BUDGET)
    _assert_real(model, lb, ub, [[spec]], result, cex, f"{corner}/{method}")
    assert np.array_equal(np.asarray(cex[0], dtype=np.float64).reshape(-1),
                          np.array([a, b]))


# --------------------------------------------------------------------------- #
# A box with no unsafe point at all                                            #
# --------------------------------------------------------------------------- #
class _Complement(nn.Module):
    """y0 = relu(x0), y1 = 1 - relu(x0): y0 + y1 = 1 on the whole box."""

    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Linear(2, 2)
        with torch.no_grad():
            self.first.weight.copy_(torch.eye(2))
            self.second.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
            self.second.bias.copy_(torch.tensor([0.0, 1.0]))

    def forward(self, x):
        return self.second(torch.relu(self.first(x)))


# Group one: y1 >= 3/4. Group two: (y0 >= 3/4 and y1 >= 3/4) or y0 >= 5/4.
# Each group alone, each row of the two-row half-space alone, is met somewhere
# in the box; their conjunction nowhere, since y0 + y1 = 1 and y0 <= 1.
_Y1_HIGH = ([[0.0, -1.0]], [-0.75])
_BOTH_HIGH = ([[-1.0, 0.0], [0.0, -1.0]], [-0.75, -0.75])
_Y0_ABOVE_ONE = ([[-1.0, 0.0]], [-1.25])


@pytest.mark.parametrize("method", METHODS)
def test_a_box_with_no_unsafe_point_is_unknown_for_every_method(method):
    """No point of [-1, 1]^2 is unsafe, so the answer is (2, None) for every
    method and seed. Catches every reading that weakens the property: groups
    joined by OR, only the first group kept, the rows of a half-space joined
    by OR."""
    model = _Complement()
    lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    prop = [{"Hg": _hs(_Y1_HIGH)},
            {"Hg": [_hs(_BOTH_HIGH), _hs(_Y0_ABOVE_ONE)]}]
    for seed in (0, 1):
        result = falsify(model, lb, ub, prop, method=method, seed=seed,
                         **BUDGET)
        assert result == (2, None), \
            f"method {method} seed {seed}: {result} for a box with no " \
            f"unsafe point"


# --------------------------------------------------------------------------- #
# A sliver of the box, image-shaped bounds, a model without parameters          #
# --------------------------------------------------------------------------- #
def test_random_sampling_finds_a_sliver_at_the_far_end_of_a_box_away_from_zero():
    """y = (x0, 2 x1, -4 x2) on a box with both signs and magnitudes above
    one; the unsafe sliver is the top eighth of x1 in [2, 5] (one eighth of
    the box), so 500 uniform samples miss it with probability below 1e-28.
    Catches samples drawn from [lb, ub - lb] (the width read as the upper
    bound), which never reach x1 above 3."""
    model = _linear([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -4.0]])
    lb, ub = np.array([-3.0, 2.0, -0.5]), np.array([-1.0, 5.0, 0.25])
    spec = ([[0.0, -1.0, 0.0]], [-2.0 * (5.0 - 3.0 / 8.0)])
    result, cex = falsify(model, lb, ub, _hs(spec), method="random", seed=0,
                          n_samples=500)
    _assert_real(model, lb, ub, [[spec]], result, cex, "sliver")


def _picking_cnn():
    """A 1x1 convolution that reads channel 1 times 2, then Flatten: output k
    is 2 x[1, k // 5, k % 5] of a (2, 3, 5) input."""
    conv = nn.Conv2d(2, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([[[[0.0]], [[2.0]]]]))
    return nn.Sequential(conv, nn.Flatten())


def test_image_shaped_bounds_give_a_real_witness():
    """Bounds of shape (2, 3, 5), a different interval per pixel; the unsafe
    region is the top eighth of the range of output (2, 4). Catches a batch-1
    re-check or final recompute that reshapes the point to (1, 30) instead of
    (1, 2, 3, 5), which the convolution rejects."""
    model = _picking_cnn()
    lo, hi = pixel_box(2, 3, 5)
    k = 2 * 5 + 4
    top = 2.0 * (hi[1, 2, 4] - (hi[1, 2, 4] - lo[1, 2, 4]) / 8.0)
    row = [0.0] * 15
    row[k] = -1.0
    spec = ([row], [-top])
    result, cex = falsify(model, lo, hi, _hs(spec), method="random", seed=0,
                          n_samples=300)
    _assert_real(model, lo, hi, [[spec]], result, cex, "image (2, 3, 5)")


class _NoParameters(nn.Module):
    """y = (x0, -2 x1) with the weights held as a plain float32 tensor, as an
    ONNX-converted graph holds its constants: no parameter, no buffer; or,
    with ``as_buffer``, as a registered buffer: no parameter, one buffer."""

    def __init__(self, as_buffer=False):
        super().__init__()
        weight = torch.tensor([[1.0, 0.0], [0.0, -2.0]])
        if as_buffer:
            self.register_buffer("weight", weight)
        else:
            self.weight = weight

    def _apply(self, fn, *args, **kwargs):
        # ``.double()`` of the float64 replay converts a plain constant as
        # well; the float32 instance itself rejects float64 input.
        if "weight" not in self._buffers:
            self.weight = fn(self.weight)
        return super()._apply(fn, *args, **kwargs)

    def forward(self, x):
        return x @ self.weight.T


@pytest.mark.parametrize("as_buffer", [
    pytest.param(False, id="plain-constant"),
    pytest.param(True, id="buffer-only"),
])
def test_a_model_without_parameters_gives_a_real_witness(as_buffer):
    """The model has no parameters; its float32 weights are a plain tensor
    (no buffer either) or a registered buffer. The random falsifier and
    Square must both return a real witness. Catches a device lookup that does
    not fall back past the parameters and buffers (StopIteration) and a final
    recompute that feeds such a model float64 instead of float32."""
    model = _NoParameters(as_buffer)
    assert not list(model.parameters())
    assert len(list(model.buffers())) == int(as_buffer)
    lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    spec = ([[-1.0, 0.0], [0.0, -1.0]], [-0.5, -1.0])    # y0>=1/2, y1>=1
    for method in ("random", "square"):
        result, cex = falsify(model, lb, ub, _hs(spec), method=method,
                              seed=0, **BUDGET)
        _assert_real(model, lb, ub, [[spec]], result, cex, method)


def test_a_model_left_in_training_mode_is_falsified_as_its_inference_function():
    """A freshly built model with Dropout(0.5) is in training mode. Its
    inference function is y = x <= 1 on [0, 1], so y >= 3/2 has no
    counterexample; with Dropout active kept activations double and reach it.
    Catches a random falsifier that samples, re-checks and returns points of
    the training-mode forward."""
    layer = nn.Linear(1, 1)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.fill_(0.0)
    spec = ([[-1.0]], [-1.5])
    for seed in range(10):
        model = nn.Sequential(layer, nn.Dropout(0.5))
        assert model.training
        result = falsify(model, np.array([0.0]), np.array([1.0]), _hs(spec),
                         method="random", seed=seed, n_samples=200)
        assert result == (2, None), f"seed {seed}: {result}"


# --------------------------------------------------------------------------- #
# The final check on a point handed to it (the attack replaced by a stub)      #
# --------------------------------------------------------------------------- #
def _stub_attack(monkeypatch, x, y_claimed):
    """Replace every attack by one that reports ``(x, y_claimed)`` as SAT."""
    monkeypatch.setattr(
        fz, "_falsify_impl",
        lambda *a, **k: (0, (np.asarray(x), np.asarray(y_claimed))))


def _float64_identity(n):
    model = nn.Sequential(nn.Linear(n, n, bias=False)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(n, dtype=torch.float64))
    return model


def test_final_check_snaps_a_point_outside_the_box_at_both_ends(monkeypatch):
    """The attack reports float32(0.1), 1.5e-9 above the upper bound 0.1, and
    float32(0.7), 1.2e-8 below the lower bound 0.7 (a float32 attack's
    clamp). The point snapped into the box, (0.1, 0.7), is unsafe for
    y0 >= 1/16 and y1 <= 3/4, so it must be returned, inside the box. Catches
    a snap with either bound rounded to float32, no snap, and the unsnapped
    point returned."""
    model = _float64_identity(2)
    lb, ub = np.array([0.0, 0.7]), np.array([0.1, 1.0])
    x_attack = np.array([np.float32(0.1), np.float32(0.7)], dtype=np.float64)
    assert x_attack[0] > ub[0] and x_attack[1] < lb[1]
    _stub_attack(monkeypatch, x_attack, x_attack)
    spec = ([[-1.0, 0.0], [0.0, 1.0]], [-0.0625, 0.75])
    result, cex = falsify(model, lb, ub, _hs(spec))
    _assert_real(model, lb, ub, [[spec]], result, cex, "snap")
    assert np.array_equal(np.asarray(cex[0]).reshape(-1), np.array([0.1, 0.7]))


@pytest.mark.parametrize("excess, real", [
    pytest.param(0.0, True, id="on-the-boundary"),
    pytest.param(2.0 ** -30, False, id="2^-30-outside"),
])
def test_final_check_is_closed_and_at_zero_tolerance(monkeypatch, excess,
                                                     real):
    """A float64 model y = x and the point x = 0.1. The unsafe region is
    y >= 0.1 + excess: with excess 0 the point lies on the boundary of the
    closed region and is a counterexample; with excess 2^-30 (9.3e-10) it is
    not, by less than every tolerance a check might add (1e-9, 1e-8) and by
    less than float32 rounding of 0.1 (1.5e-9). Catches a strict comparison,
    an added tolerance, a recompute or a property check in float32."""
    model = _float64_identity(1)
    lb, ub = np.array([0.0]), np.array([0.5])
    x = np.array([0.1])
    _stub_attack(monkeypatch, x, x)
    spec = ([[-1.0]], [-(0.1 + excess)])
    result, cex = falsify(model, lb, ub, _hs(spec))
    if real:
        _assert_real(model, lb, ub, [[spec]], result, cex, "boundary")
    else:
        assert not _unsafe([[spec]], replay_torch(model, [x], (1, 1))[0])
        assert (result, cex) == (2, None)


# Plain specs over y in R^2 for the conjunction and disjunction cases below.
_Y0_GE_HALF = ([[-1.0, 0.0]], [-0.5])
_Y1_GE_HALF = ([[0.0, -1.0]], [-0.5])
_Y0_LE_MINUS_HALF = ([[1.0, 0.0]], [-0.5])
_ROWS_Y0_Y1_GE_HALF = ([[-1.0, 0.0], [0.0, -1.0]], [-0.5, -0.5])


@pytest.mark.parametrize("groups, as_given, x", [
    pytest.param([[_Y0_GE_HALF], [_Y1_GE_HALF]],
                 lambda: [{"Hg": _hs(_Y0_GE_HALF)}, {"Hg": _hs(_Y1_GE_HALF)}],
                 [0.75, 0.25], id="two-groups-first-met-only"),
    pytest.param([[_Y0_GE_HALF], [_Y1_GE_HALF]],
                 lambda: [{"Hg": _hs(_Y0_GE_HALF)}, {"Hg": _hs(_Y1_GE_HALF)}],
                 [0.75, 0.5], id="two-groups-both-met"),
    pytest.param([[_ROWS_Y0_Y1_GE_HALF]],
                 lambda: _hs(_ROWS_Y0_Y1_GE_HALF),
                 [0.25, 0.75], id="two-rows-second-met-only"),
    pytest.param([[_Y0_GE_HALF, _Y0_LE_MINUS_HALF]],
                 lambda: [_hs(_Y0_GE_HALF), _hs(_Y0_LE_MINUS_HALF)],
                 [-0.75, 0.0], id="disjunction-second-met-only"),
])
def test_final_check_reads_every_group_row_and_disjunct(monkeypatch, groups,
                                                        as_given, x):
    """The attack reports a point inside the box with a claimed output that
    meets the whole property; the final check must judge the true output
    instead: a point that meets one of two conjoined groups, or one row of a
    two-row half-space, is not a counterexample; a point that meets the
    second of two disjuncts is. Catches a final check that reads only the
    first group, groups joined by OR, rows joined by OR, and disjuncts joined
    by AND."""
    model = _linear(np.eye(2))
    lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    _stub_attack(monkeypatch, np.array(x), np.array([0.75, 0.75]))
    result, cex = falsify(model, lb, ub, as_given())
    if _unsafe(groups, x):
        _assert_real(model, lb, ub, groups, result, cex, "met")
    else:
        assert (result, cex) == (2, None)


class _BatchSkew(nn.Module):
    """y = x; the rows of a batch larger than one read 1/64 lower, as a
    converted graph that mishandles batches might."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x):
        y = self.scale * x.reshape(x.shape[0], -1)
        return y - 1.0 / 64.0 if x.shape[0] > 1 else y


def test_final_check_recomputes_the_output_at_batch_size_one(monkeypatch):
    """The attack reports x = 1/2 with the claimed output 1/2 - 1/64, which a
    batched forward gives; the batch-1 output, the one a replay sees, is 1/2.
    For the unsafe region y <= 1/2 - 1/128 the answer must be (2, None).
    Catches a final check that trusts the attack's output, concedes with the
    attack's result code, skips the recompute, or recomputes in a batch."""
    lb, ub = np.array([0.0]), np.array([1.0])
    _stub_attack(monkeypatch, np.array([0.5]), np.array([0.5 - 1 / 64]))
    spec = ([[1.0]], [0.5 - 1.0 / 128.0])
    assert falsify(_BatchSkew(), lb, ub, _hs(spec)) == (2, None)


# --------------------------------------------------------------------------- #
# The falsifier and the verifier read every accepted form alike               #
# --------------------------------------------------------------------------- #
# y0 >= 1/2 and y1 <= 1/4 (two rows); y0 <= -1/2; y1 >= 0.
_A = ([[-1.0, 0.0], [0.0, 1.0]], [-0.5, 0.25])
_B = ([[1.0, 0.0]], [-0.5])
_C = ([[0.0, -1.0]], [0.0])

_FORMS = {
    "HalfSpace": (lambda: _hs(_A), [[_A]]),
    "list-of-HalfSpace": (lambda: [_hs(_A), _hs(_B)], [[_A, _B]]),
    "dict-HalfSpace": (lambda: {"Hg": _hs(_A)}, [[_A]]),
    "dict-list": (lambda: {"Hg": [_hs(_A), _hs(_B)]}, [[_A, _B]]),
    "list-of-dict": (lambda: [{"Hg": _hs(_A)}, {"Hg": _hs(_C)}], [[_A], [_C]]),
    "list-of-dict-with-list": (
        lambda: [{"Hg": [_hs(_A), _hs(_B)]}, {"Hg": _hs(_C)}],
        [[_A, _B], [_C]]),
}

# Outputs on a grid in steps of 1/4 over [-1, 1] x [-3/4, 3/4], and points on
# the boundaries of A, B and C.
_OUTPUTS = ([(a / 4, b / 4) for a in range(-4, 5) for b in range(-3, 4)]
            + [(0.5, 0.25), (0.5, 0.0), (-0.5, 0.0), (0.5, -0.125),
               (0.625, 0.25), (-0.5, -0.5)])


@pytest.mark.parametrize("form", list(_FORMS))
def test_every_accepted_form_is_read_as_its_text_states_by_falsify_and_verify_specification(form):
    """The same property object goes to ``falsify`` (identity model on the
    one-point box {y}) and to ``verify_specification`` (the one-point Star
    {y}); both decisions must equal the plain reading of the property at y,
    inside, outside and on the boundary of every region. Catches the
    falsifier's reader splitting a list of disjuncts into conjoined groups
    (in each of the three list forms), keeping only the first of several
    groups, and a membership test that joins groups by OR, rows by OR,
    disjuncts by AND, compares g as a row, or excludes the boundary; and a
    divergence of the two readers."""
    make, groups = _FORMS[form]
    model = _linear(np.eye(2))
    wrong = []
    for y in _OUTPUTS:
        point = np.array(y)
        expected = _unsafe(groups, y)
        result, _ = falsify(model, point, point, make(), method="random",
                            seed=0, n_samples=1)
        if (result == 0) != expected:
            wrong.append(("falsify", y, result, expected))
        verdict = verify_specification([Star.from_bounds(point, point)],
                                       make()).verdict
        if (verdict != "UNSAT") != expected:
            wrong.append(("verify_specification", y, verdict, expected))
    assert not wrong, f"{form}: decisions that differ from the text " \
                      f"(reader, y, answer, unsafe): {wrong}"


# --------------------------------------------------------------------------- #
# The size guard of the final check                                            #
# --------------------------------------------------------------------------- #
@known_bug(19, "falsify's final check hands back, unclipped and unchecked, "
               "a point whose size differs from the box's (no attack returns "
               "one today), so the last check of every witness is skipped "
               "exactly where a point cannot be read as an input")
def test_final_check_refuses_a_point_that_does_not_fit_the_box(monkeypatch):
    """The attack reports a point of three entries for a box of two, with a
    claimed output in the unsafe region y >= 9; y = x0 + x1 <= 2 on [0, 1]^2,
    so there is no counterexample. Latent: only a stubbed attack reaches
    the guard."""
    model = _linear([[1.0, 1.0]])
    _stub_attack(monkeypatch, np.array([5.0, 5.0, 5.0]), np.array([10.0]))
    unsafe = _hs(([[-1.0]], [-9.0]))
    try:
        result = falsify(model, np.zeros(2), np.ones(2), unsafe)
    except ValueError:
        return
    assert result == (2, None)

"""The attacks behind ``falsify`` and their dispatch: PGD, the method names of
``_falsify_impl`` (single attacks and cascades), the AutoAttack scaffold, and
the promises every attack (PGD, APGD, Square) keeps alike.

Component: ``n2v/utils/falsify.py::_falsify_impl``, ``_falsify_pgd``,
``_falsify_autoattack``, and the shared regimes of ``_falsify_apgd`` and
``_falsify_square`` (APGD and Square have their own files,
``test_falsify_apgd.py`` and ``test_falsify_square.py``).

Promise: a method name runs exactly the attacks it names, in order, each with
the caller's seed and budget keywords, and returns the first counterexample;
an unknown name is refused. An attack finds a counterexample that its search
reaches by construction (a descent that walks to an unsafe corner, a step that
lands on it, independent restarts), evaluates the model only inside the box,
reads every group, row and disjunct of the property, keeps the tensor shape of
the bounds, and depends on its seed and on nothing else. ``falsify`` re-checks
every returned point, so a wrong attack misses counterexamples (unit harm, the
competition score) rather than returning a false one.

Oracles are independent of n2v: the models are linear or ReLU with dyadic
weights and dyadic boxes, so the unsafe set is known by hand (one corner of
the box, or a corner region), every witness is replayed in rationals
(``fractions.Fraction``) and checked against the property as written in the
test, and budgets are counted by a recording wrapper around the model. The
strength cases are exact, not statistical: each states the start-independent
reason why the unmodified attack reaches the counterexample, and fixes the
seed (the autouse fixture also seeds the global generators, and everything
runs on the CPU), so a case cannot flake. Every case runs well under a
second.
"""

import importlib
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import HalfSpace
from n2v.utils import falsify
from tests.known_bugs import known_bug

# ``import n2v.utils.falsify`` resolves to the re-exported function
fz = importlib.import_module("n2v.utils.falsify")

ATTACKS = ["pgd", "apgd", "square"]
# Written out, not read from METHODS: a name dropped from METHODS must fail
# its cases (falsify refuses it), not lose them.
ALL_METHODS = ["random", "pgd", "apgd", "square", "strong", "random+pgd",
               "random+pgd+apgd", "random+square", "random+apgd"]


# --------------------------------------------------------------------------- #
# Models, boxes and the independent replay                                      #
# --------------------------------------------------------------------------- #
def _linear(W, b):
    """A float32 ``nn.Linear`` holding exactly the dyadic ``W`` and ``b``."""
    W = np.asarray(W, dtype=np.float64)
    layer = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(W))
        layer.bias.copy_(torch.tensor(np.asarray(b, dtype=np.float64)))
    return layer


def _exact_affine(W, b, x):
    """``W x + b`` in rationals, for a point ``x`` read exactly."""
    xs = [Fraction(float(v)) for v in np.asarray(x, dtype=np.float64).ravel()]
    return [sum(Fraction(float(w)) * v for w, v in zip(row, xs))
            + Fraction(float(bb)) for row, bb in zip(W, b)]


def _assert_in_box(x, lb, ub):
    """The witness lies in the float64 box, exactly (no slack)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    lb = np.asarray(lb, dtype=np.float64).ravel()
    ub = np.asarray(ub, dtype=np.float64).ravel()
    assert x.shape == lb.shape, f"witness of {x.size} entries for a box of {lb.size}"
    assert np.all(lb <= x) and np.all(x <= ub), f"witness {x} outside [{lb}, {ub}]"


class _Recorder(nn.Module):
    """Runs ``inner`` and keeps every input row it is given (as float64), and
    the batch size of every forward. Holds ``inner`` as a submodule, so the
    model's parameters and device are those of ``inner``."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.rows = []
        self.batches = []

    def forward(self, x):
        flat = x.detach().reshape(x.shape[0], -1).to(torch.float64).cpu().numpy()
        self.rows.extend(flat.copy())
        self.batches.append(x.shape[0])
        return self.inner(x)


# The mixed box: intervals of different widths, both signs, magnitudes above
# and below one, and one degenerate coordinate (x3). y0 = W0 . x + 1/2 has
# weights of both signs and magnitudes above and below one; its maximum over
# the box, 11.125, is attained at exactly one point, the corner that takes ub
# where the weight is positive and lb where it is negative. y1 is unused by the
# property (more outputs than the property reads).
_W_MIX = [[1.0, -2.0, 0.5, 4.0, -1.0, 0.25],
          [0.5, 0.25, -1.0, 0.0, 2.0, -0.5]]
_B_MIX = [0.5, -1.0]
_LB_MIX = np.array([-2.0, -0.5, 1.0, 0.75, -3.0, 0.0])
_UB_MIX = np.array([2.0, 0.25, 3.0, 0.75, -1.0, 0.5])
_CORNER_MIX = np.array([2.0, -0.5, 3.0, 0.75, -3.0, 0.5])
_TOP_MIX = 11.125                                    # y0 at the corner


def _unsafe_top_corner():
    """Unsafe: y0 >= 11.125, met by the corner alone, with margin exactly 0."""
    return HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-_TOP_MIX]]))


def _assert_is_the_top_corner(result, cex):
    """SAT, and the witness is the one unsafe point: its exact y0 is 11.125."""
    assert result == 0, "the unsafe corner of the box was not found"
    x = np.asarray(cex[0], dtype=np.float64).ravel()
    _assert_in_box(x, _LB_MIX, _UB_MIX)
    assert _exact_affine(_W_MIX, _B_MIX, x)[0] >= Fraction(_TOP_MIX), (
        f"witness {x} is not unsafe: exact y0 = "
        f"{float(_exact_affine(_W_MIX, _B_MIX, x)[0])} < {_TOP_MIX}")
    np.testing.assert_array_equal(x, _CORNER_MIX)


# --------------------------------------------------------------------------- #
# Dispatch: the method names                                                    #
# --------------------------------------------------------------------------- #
# Budgets of the legs when nothing is found: the sampler evaluates n_samples
# points in one batch; PGD evaluates one point per step and checks the point
# its last step reaches, n_restarts * (n_steps + 1); APGD evaluates one point
# per step and has no check after its last step, n_restarts * n_steps; Square
# evaluates exactly n_iters points (the start point and clamped batches).
_N_SAMPLES, _N_RESTARTS, _N_STEPS, _N_ITERS, _BATCH = 7, 2, 3, 40, 16
_LEG_ROWS = {"random": _N_SAMPLES,
             "pgd": _N_RESTARTS * (_N_STEPS + 1),
             "apgd": _N_RESTARTS * _N_STEPS,
             "square": _N_ITERS}
_LEGS = {"random": ["random"], "pgd": ["pgd"], "apgd": ["apgd"],
         "square": ["square"], "random+pgd": ["random", "pgd"],
         "random+pgd+apgd": ["random", "pgd", "apgd"],
         "random+apgd": ["random", "apgd"],
         "random+square": ["random", "square"],
         "strong": ["random", "apgd", "square"]}


@pytest.mark.parametrize("method", ALL_METHODS)
def test_every_method_spends_exactly_the_budget_of_its_legs(method):
    """No counterexample exists (unsafe y0 >= 100, while y0 <= 5/2 on the
    box), so every leg the method names runs to
    the end of its budget, and the model sees exactly the sum of the legs'
    evaluations (see ``_LEG_ROWS``). Catches a leg that is skipped or replaced
    by another attack, a cascade whose test is inverted, keywords that do not
    reach a leg (its default budget is hundreds of evaluations), budgets
    swapped between restarts and steps, and a last Square batch that is not
    clamped (40 - 1 is no multiple of 16). The answer is (2, None)."""
    model = _Recorder(_linear([[1.0, 0.5], [-0.25, 1.0], [2.0, -1.0]],
                              [0.0, 0.5, -0.25]))
    lb, ub = np.array([-2.0, 0.5]), np.array([2.0, 1.0])
    unsafe = HalfSpace(np.array([[-1.0, 0.0, 0.0]]), np.array([[-100.0]]))
    result = falsify(model, lb, ub, unsafe, method=method, seed=5,
                     n_samples=_N_SAMPLES, n_restarts=_N_RESTARTS,
                     n_steps=_N_STEPS, n_iters=_N_ITERS, batch=_BATCH)
    assert result == (2, None)
    expected = sum(_LEG_ROWS[leg] for leg in _LEGS[method])
    assert len(model.rows) == expected, (
        f"{method}: the model saw {len(model.rows)} inputs, its legs "
        f"{_LEGS[method]} budget {expected}")


@pytest.mark.parametrize("method", ["random", "random+pgd", "random+pgd+apgd",
                                    "random+apgd", "random+square", "strong"])
def test_every_leg_of_a_method_draws_its_points_from_the_seed_alone(method):
    """The problem of the budget case above: no counterexample, so every leg
    runs to the end of its budget and the points the model sees split by leg
    (``_LEG_ROWS``). Run twice with seed 5, with the global numpy and torch
    generators set to other values before each run, every leg evaluates the
    same points; with seed 6 every leg evaluates other points (each leg's
    first point is a continuous draw). Catches a leg that the dispatcher
    calls without the seed (it then draws from fresh entropy) and a leg that
    draws from the global generator. The single attacks are the cases of
    ``test_attack_is_repeatable_for_its_seed_and_for_nothing_else``."""
    lb, ub = np.array([-2.0, 0.5]), np.array([2.0, 1.0])
    unsafe = HalfSpace(np.array([[-1.0, 0.0, 0.0]]), np.array([[-100.0]]))
    sizes = [_LEG_ROWS[leg] for leg in _LEGS[method]]

    def points_by_leg(seed, global_state):
        np.random.seed(global_state)
        torch.manual_seed(global_state)
        model = _Recorder(_linear([[1.0, 0.5], [-0.25, 1.0], [2.0, -1.0]],
                                  [0.0, 0.5, -0.25]))
        result = falsify(model, lb, ub, unsafe, method=method, seed=seed,
                         n_samples=_N_SAMPLES, n_restarts=_N_RESTARTS,
                         n_steps=_N_STEPS, n_iters=_N_ITERS, batch=_BATCH)
        if result != (2, None) or len(model.rows) != sum(sizes):
            pytest.fail(f"{method}: {result} after {len(model.rows)} "
                        f"evaluations, budget {sum(sizes)}")
        return np.split(np.array(model.rows), np.cumsum(sizes)[:-1])

    first, again, other = (points_by_leg(5, 11), points_by_leg(5, 12),
                           points_by_leg(6, 11))
    for leg, a, b, c in zip(_LEGS[method], first, again, other):
        np.testing.assert_array_equal(
            a, b, err_msg=f"{method}: the {leg} leg evaluated other points "
                          "for the same seed")
        assert not np.array_equal(a, c), (
            f"{method}: the {leg} leg evaluated the same points for seeds "
            "5 and 6")


class _ConstantOutput(nn.Module):
    """y = (0, 0) for every input, with zero gradient, and no parameters or
    buffers (the device falls back to the CPU)."""

    def forward(self, x):
        return x.reshape(x.shape[0], -1)[:, :2] * 0.0


@pytest.mark.parametrize("method", ALL_METHODS)
def test_every_method_returns_the_first_point_it_evaluates_when_every_output_is_unsafe(method):
    """The readers write the property of a disjunct that has no output
    constraint as ``0 . y <= 0``: every output is unsafe, with margin exactly
    0 (the unsafe region is closed) and zero gradient. Every method must
    answer SAT with the first point it evaluates, and run nothing after it: a
    leading sampler evaluates its one batch and re-runs its first point at
    batch size 1, an attack evaluates its start point, and ``falsify``
    re-checks the returned point once. Catches a hit test that needs a
    negative margin (APGD never moves on a zero gradient and has no final
    check), a Square that does not check its start point (no candidate can
    improve on margin 0), and a cascade that runs its next leg after a SAT."""
    model = _Recorder(_ConstantOutput())
    lb, ub = np.array([-1.5, 0.25, 2.0]), np.array([-0.5, 0.75, 6.0])
    every = HalfSpace(np.zeros((1, 2)), np.zeros((1, 1)))
    result, cex = falsify(model, lb, ub, every, method=method, seed=3,
                          n_samples=_N_SAMPLES, n_restarts=_N_RESTARTS,
                          n_steps=_N_STEPS, n_iters=_N_ITERS, batch=_BATCH)
    assert result == 0
    _assert_in_box(cex[0], lb, ub)
    first_leg_rows = _N_SAMPLES + 1 if _LEGS[method][0] == "random" else 1
    assert len(model.rows) == first_leg_rows + 1, (
        f"{method}: {len(model.rows)} evaluations, expected "
        f"{first_leg_rows} by the first leg and one re-check")
    np.testing.assert_array_equal(np.asarray(cex[0], dtype=np.float64),
                                  model.rows[0])


@pytest.mark.parametrize("method, kwargs", [
    pytest.param("random+pgd", dict(n_restarts=1, n_steps=1, step_size=4.0),
                 id="random+pgd-one-step-of-the-given-size"),
    pytest.param("random+pgd+apgd", dict(n_restarts=1, n_steps=25),
                 id="random+pgd+apgd-apgd-leg"),
    pytest.param("random+apgd", dict(n_restarts=1, n_steps=25),
                 id="random+apgd-apgd-leg"),
])
def test_cascade_finds_what_only_its_last_leg_reaches(method, kwargs):
    """The unsafe set is one corner of the mixed box, which the sampler hits
    with probability 0. In 'random+pgd' the PGD leg's one step of the given
    size 4 (the widest range) moves every coordinate onto its bound, and PGD
    checks the point its last step reaches. In the two APGD cascades APGD's
    auto step, 5% of the widest range (0.2), reaches the corner from any start
    within 20 steps, while PGD's 1% step (0.04) moves a coordinate at most 1
    in 25 steps, and PGD's seed-3 start lies 3.66 below the corner in x0.
    Catches a cascade that stops before its last leg, runs PGD where APGD is
    named, or does not hand the step keywords to its last leg."""
    result, cex = falsify(_linear(_W_MIX, _B_MIX), _LB_MIX, _UB_MIX,
                          _unsafe_top_corner(), method=method, seed=3,
                          n_samples=50, **kwargs)
    _assert_is_the_top_corner(result, cex)


@pytest.mark.parametrize("name", ["PGD", "random+square+pgd", ""])
def test_an_unknown_method_name_is_refused(name):
    """A name outside METHODS (another case, a cascade that does not exist,
    the empty string, which Python reads as false) is refused with
    ValueError, never answered by some default attack."""
    every = HalfSpace(np.zeros((1, 2)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="Unknown method"):
        falsify(_ConstantOutput(), np.zeros(2), np.ones(2), every,
                method=name, seed=0, n_samples=5)


@pytest.mark.parametrize("installed, refusal", [
    pytest.param(True, NotImplementedError, id="package-installed"),
    pytest.param(False, ImportError, id="package-missing"),
])
def test_autoattack_refuses_whether_or_not_its_package_is_installed(
        monkeypatch, installed, refusal):
    """'autoattack' is a scaffold: with the package it refuses with
    NotImplementedError (the loss is not wired), without it with ImportError
    and install instructions. Both branches are reached by setting the
    module's flag, whatever this environment has installed. Catches a
    scaffold that answers (2, None) as if it had searched (half of this box
    is unsafe) and swapped branches."""
    monkeypatch.setattr(fz, "_HAS_AUTOATTACK", installed)
    unsafe = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))
    with pytest.raises(refusal):
        falsify(_linear([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]), np.zeros(2),
                np.ones(2), unsafe, method="autoattack", seed=0)


# --------------------------------------------------------------------------- #
# Every attack                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ATTACKS)
def test_attack_evaluates_the_model_only_inside_the_box(method):
    """Projected search: every point an attack hands the model lies in the
    box. The unsafe set (y0 >= 100) lies beyond the box in the direction the
    descent pushes, so a step that is not projected leaves the box at once
    (the degenerate coordinate x3 moves on its first step). The bounds are
    dyadic, so the float32 box the attacks use is the box itself."""
    model = _Recorder(_linear(_W_MIX, _B_MIX))
    unsafe = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-100.0]]))
    result = falsify(model, _LB_MIX, _UB_MIX, unsafe, method=method, seed=1,
                     n_restarts=2, n_steps=10, n_iters=200, batch=32)
    assert result == (2, None)
    rows = np.array(model.rows)
    outside = ~np.all((_LB_MIX <= rows) & (rows <= _UB_MIX), axis=1)
    assert not outside.any(), (
        f"{method} evaluated {outside.sum()} of {len(rows)} points outside "
        f"the box, first {rows[outside][0]}")


def _image_model():
    """Conv2d(2 -> 1, kernel 2 x 3) on a (2, 3, 5) input, then Flatten: output
    (i, j) is x[0, i + 1, j + 2] - x[1, i, j] + 1/4, six outputs."""
    conv = nn.Conv2d(2, 1, kernel_size=(2, 3))
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 1, 2] = 1.0
        conv.weight[0, 1, 0, 0] = -1.0
        conv.bias.fill_(0.25)
    return nn.Sequential(conv, nn.Flatten())


def _pixel_box():
    """A different dyadic interval per pixel (lower ends in [-2, 1.75], widths
    in [1/4, 5/4]), as (C, H, W) arrays."""
    rng = np.random.default_rng(0)
    lo = rng.integers(-8, 8, size=(2, 3, 5)) / 4.0
    return lo, lo + rng.integers(1, 6, size=(2, 3, 5)) / 4.0


@pytest.mark.parametrize("method", ATTACKS)
def test_attack_reads_image_shaped_bounds(method):
    """Bounds of shape (C, H, W) = (2, 3, 5), different per pixel, for a
    convolution that needs its input as (N, C, H, W). Unsafe: output 5,
    x[0, 2, 4] - x[1, 1, 2] + 1/4, within 1/4 of its maximum over the box,
    which needs both pixels near their bounds. Catches a forward that hands
    the model flat candidates or reads the pixels in another order. The
    witness is replayed by hand, in rationals."""
    lo, hi = _pixel_box()
    top = hi[0, 2, 4] - lo[1, 1, 2] + 0.25
    unsafe = HalfSpace(np.eye(6)[[5]] * -1.0, np.array([[-(top - 0.25)]]))
    result, cex = falsify(_image_model(), lo, hi, unsafe, method=method,
                          seed=2, n_restarts=2, n_steps=120, n_iters=2000,
                          batch=64)
    assert result == 0, f"{method} found no point of the unsafe corner"
    x = np.asarray(cex[0], dtype=np.float64).reshape(2, 3, 5)
    _assert_in_box(x, lo, hi)
    y5 = Fraction(float(x[0, 2, 4])) - Fraction(float(x[1, 1, 2])) + Fraction(1, 4)
    assert y5 >= Fraction(float(top - 0.25))


class _TwoOutputs(nn.Module):
    """A = x and B = -x on two inputs, returned as a list of two (1, 2)
    tensors, as ``load_onnx`` returns the outputs of a graph with two output
    tensors; with ``joint=True`` the same function as one tensor (A, B)."""

    def __init__(self, joint):
        super().__init__()
        self.joint = joint
        self.lin = _linear(np.eye(2), np.zeros(2))

    def forward(self, x):
        a = self.lin(x)
        return torch.cat([a, -a], dim=1) if self.joint else [a, -a]


_TWO_OUTPUTS = known_bug(
    19, "the attacks read the model's output as one tensor, so a model with "
        "two output tensors (load_onnx returns them as a list) makes PGD, "
        "APGD and Square crash with AttributeError instead of searching the "
        "joint output or refusing the model (registry VERDICT-10, found in the "
        "sampler)", raises=AttributeError)


@pytest.mark.parametrize("method, joint", [
    *[pytest.param(m, False, id=f"{m}-two-output-tensors", marks=_TWO_OUTPUTS)
      for m in ATTACKS],
    *[pytest.param(m, True, id=f"{m}-one-joint-output-tensor")
      for m in ATTACKS]])
def test_attack_searches_the_joint_output_of_a_model_with_two_outputs(
        method, joint):
    """The joint output is (A0, A1, B0, B1) = (x0, x1, -x0, -x1) on [0, 1]^2;
    unsafe B0 <= -1/2, a row on the third entry of the joint output (so a
    reader of the first tensor alone cannot meet it), holds on half of the
    box. A model with two output tensors is either refused (ValueError or
    NotImplementedError) or searched over its joint output: SAT with a
    witness in the box whose B0, replayed by hand, is at most -1/2. The cases
    with one joint output tensor are the passing neighbours, held to SAT:
    descent raises x0 at once (gradient of B0 is -1), and Square's start or
    first move of x0 to its upper bound meets the row."""
    unsafe = HalfSpace(np.array([[0.0, 0.0, 1.0, 0.0]]), np.array([[-0.5]]))
    lb, ub = np.zeros(2), np.ones(2)
    try:
        result, cex = falsify(_TwoOutputs(joint), lb, ub, unsafe,
                              method=method, seed=0, n_restarts=2,
                              n_steps=20, n_iters=100, batch=8)
    except (ValueError, NotImplementedError):
        if joint:
            raise
        return
    assert result == 0, f"{method} found no point of half of the box"
    _assert_in_box(cex[0], lb, ub)
    assert -Fraction(float(np.asarray(cex[0], dtype=np.float64).ravel()[0])) \
        <= Fraction(-1, 2), f"witness {cex[0]} is not unsafe"


# The documented defaults: PGD 10 restarts of 50 steps, each restart ending
# with a check of the point its last step reaches; APGD 10 restarts of 50
# steps; Square n_iters = 8000 evaluations in forwards of at most 256 rows.
_DEFAULT_ROWS = {"pgd": 10 * (50 + 1), "apgd": 10 * 50, "square": 8000}


@pytest.mark.parametrize("method", ATTACKS)
def test_attack_runs_its_default_budget_to_the_end_on_image_shaped_bounds(
        method):
    """Called without budget keywords, as ``verify_bab_model`` and the
    runner without ``falsify_kwargs`` call it, on the (2, 3, 5) pixel box of
    the convolution above, with no counterexample (output 0 >= 100, while
    every output is at most 5.25 on the box): the attack answers (2, None)
    after exactly its documented default budget (``_DEFAULT_ROWS``), every
    forward in the (C, H, W) shape the convolution needs. Catches a default
    that is not the documented one, and a forward that hands the model flat
    inputs where only a search that finds nothing reaches it (PGD's check
    after the last step of a restart)."""
    lo, hi = _pixel_box()
    model = _Recorder(_image_model())
    unsafe = HalfSpace(-np.eye(6)[[0]], np.array([[-100.0]]))
    result = falsify(model, lo, hi, unsafe, method=method, seed=4)
    assert result == (2, None)
    assert len(model.rows) == _DEFAULT_ROWS[method], (
        f"{method}: {len(model.rows)} evaluations, the documented default "
        f"budget is {_DEFAULT_ROWS[method]}")
    assert max(model.batches) <= 256, f"{method}: a forward of {max(model.batches)} rows"


@pytest.mark.parametrize("method", ATTACKS)
def test_attack_is_repeatable_for_its_seed_and_for_nothing_else(method):
    """The same seed gives the same witness whatever the global numpy and
    torch generators hold (they are set to other values before each call),
    on the same model object used twice; another seed gives another witness
    (the starts are continuous draws). Seed 0, which Python reads as false,
    is one of the two. Catches an attack that draws from the global
    generator, ignores the seed, or does not receive it from the dispatcher.
    Unsafe: x0 + x1 >= 3/2 on [0, 1]^2, where the point returned depends on
    the start."""
    model = _linear([[1.0, 1.0]], [0.0])
    unsafe = HalfSpace(np.array([[-1.0]]), np.array([[-1.5]]))
    witnesses = {}
    for seed in (0, 7):
        runs = []
        for global_state in (11, 12):
            np.random.seed(global_state)
            torch.manual_seed(global_state)
            result, cex = falsify(model, np.zeros(2), np.ones(2), unsafe,
                                  method=method, seed=seed, n_restarts=3,
                                  n_steps=100, n_iters=500, batch=16)
            if result != 0:
                pytest.fail(f"{method} found no point of a quarter of the box")
            runs.append((np.asarray(cex[0]).copy(), np.asarray(cex[1]).copy()))
        np.testing.assert_array_equal(runs[0][0], runs[1][0])
        np.testing.assert_array_equal(runs[0][1], runs[1][1])
        witnesses[seed] = runs[0][0]
    assert not np.array_equal(witnesses[0], witnesses[7]), (
        f"{method}: seeds 0 and 7 gave the same witness {witnesses[0]}")


@pytest.mark.parametrize("method", ATTACKS)
def test_attack_refuses_an_empty_or_misshapen_box(method):
    """lb > ub in one coordinate holds no point, so even a property that every
    output meets has no counterexample: refused with ValueError (numpy's
    sampler raises it today) or answered (2, None), never SAT. Bounds of 1 and
    2 entries describe no box: refused with ValueError. Catches bounds that
    are sorted or broadcast into a box the caller did not give."""
    every = HalfSpace(np.zeros((1, 2)), np.zeros((1, 1)))
    model = _linear([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
    try:
        result = falsify(model, np.array([0.5, 0.0]), np.array([0.25, 1.0]),
                         every, method=method, seed=0, n_restarts=1,
                         n_steps=2, n_iters=10, batch=4)
    except ValueError:
        pass
    else:
        assert result == (2, None), f"{method} answered an empty box with SAT"
    with pytest.raises(ValueError):
        falsify(model, np.array([0.0]), np.array([1.0, 1.0]), every,
                method=method, seed=0, n_restarts=1, n_steps=2, n_iters=10,
                batch=4)


_INFINITE_BOUND = known_bug(
    23, "a box with an infinite bound makes the attack's start-point sampler "
        "raise OverflowError ('Range exceeds valid bounds') instead of a "
        "refusal or an answer (registry VERDICT-8)", raises=OverflowError)


@pytest.mark.parametrize("method, x0_low", [
    *[pytest.param(m, -np.inf, id=m, marks=_INFINITE_BOUND) for m in ATTACKS],
    *[pytest.param(m, -2.0 ** 20, id=f"{m}-wide-finite-bound")
      for m in ATTACKS]])
def test_attack_on_an_unbounded_box_is_refused_or_answered(method, x0_low):
    """x0 in [x0_low, 1], x1 in [0, 1] (a VNN-LIB 2.0 input without bounds is
    read as infinite); unsafe y1 >= 1/2 does not need x0. Either a refusal
    (ValueError, NotImplementedError) or an answer whose witness lies in the
    box. One marked case per attack: each draws its start point on its own
    line (PGD falsify.py:433, APGD :557, Square :686). The cases with the wide
    finite bound -2**20 are their passing neighbours: the same check, answered
    SAT today (x1 alone reaches the unsafe region)."""
    unsafe = HalfSpace(np.array([[0.0, -1.0]]), np.array([[-0.5]]))
    lb, ub = np.array([x0_low, 0.0]), np.array([1.0, 1.0])
    try:
        result, cex = falsify(_linear([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]),
                              lb, ub, unsafe, method=method, seed=0,
                              n_restarts=1, n_steps=5, n_iters=50, batch=8)
    except (ValueError, NotImplementedError):
        return
    if result == 0:
        x = np.asarray(cex[0], dtype=np.float64).ravel()
        assert np.all((lb <= x) & (x <= ub)) and x[1] >= 0.5
    else:
        assert (result, cex) == (2, None)


_NO_HALF_SPACE_CRASH = known_bug(
    6, "a property with no half-space must be refused with ValueError "
       "(registry VERDICT-7, after registry SPEC-12); PGD and APGD instead crash "
       "with RuntimeError, calling backward on the constant loss of the "
       "empty group", raises=RuntimeError)
_NO_ROWS_CRASH = known_bug(
    23, "PGD and APGD take the max over the rows of a half-space with a torch "
        "reduction that fails on a half-space without rows (an AND of "
        "nothing, met by every output) and crash with RuntimeError instead "
        "of answering SAT or refusing the property", raises=RuntimeError)


@pytest.mark.parametrize("method, prop, every_point_unsafe", [
    pytest.param(m, p, e, id=f"{m}-{pid}",
                 marks=mark if m != "square" else ())
    for m in ATTACKS
    for pid, p, e, mark in [
        ("no-half-space", [], False, _NO_HALF_SPACE_CRASH),
        ("half-space-without-rows",
         HalfSpace(np.zeros((0, 2)), np.zeros((0, 1))), True,
         _NO_ROWS_CRASH)]])
def test_attack_on_a_property_without_rows_is_refused_or_answered(
        method, prop, every_point_unsafe):
    """An OR of no half-space states nothing about what is unsafe: refused
    with ValueError, as ``verify_specification`` refuses the same object
    (registry SPEC-12) and as ``falsify`` must (registry VERDICT-7; marked for
    the sampler in ``test_falsify.py``). A half-space with no rows (an AND of
    nothing) is met by every output: refused with ValueError or answered SAT
    with a witness in the box. Never a crash. The Square cases pass today
    (numpy raises ValueError) and are the neighbours of the marked PGD and
    APGD cases."""
    lb, ub = np.array([-1.0, 0.5]), np.array([1.0, 2.0])
    model = _linear([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
    budget = dict(n_restarts=1, n_steps=3, n_iters=20, batch=4)
    if not every_point_unsafe:
        with pytest.raises(ValueError):
            falsify(model, lb, ub, prop, method=method, seed=0, **budget)
        return
    try:
        result, cex = falsify(model, lb, ub, prop, method=method, seed=0,
                              **budget)
    except ValueError:
        return
    assert result == 0
    _assert_in_box(cex[0], lb, ub)


# --------------------------------------------------------------------------- #
# PGD                                                                           #
# --------------------------------------------------------------------------- #
def test_pgd_walks_to_the_one_unsafe_corner_of_a_mixed_box():
    """The loss -y0 + 11.125 is linear, so every descent step moves each
    coordinate one auto step (1% of the widest range, 0.04) toward the bound
    its weight points to, and the projection holds it there: from any start
    the corner is reached within 100 steps and is a hit at margin exactly 0.
    n_restarts=1, n_steps=110. The degenerate coordinate x3 makes the
    smallest range 0. Catches an ascent step, an auto step taken from the
    smallest range (step 0, PGD never moves), budgets swapped between
    restarts and steps (110 restarts of one step), and an unprojected step
    (the hit lies outside the box and the re-check in the box fails)."""
    result, cex = falsify(_linear(_W_MIX, _B_MIX), _LB_MIX, _UB_MIX,
                          _unsafe_top_corner(), method="pgd", seed=3,
                          n_restarts=1, n_steps=110)
    _assert_is_the_top_corner(result, cex)


def test_pgd_checks_the_point_its_last_step_reaches():
    """One restart of one step of the given size 4, the widest range: the
    step moves every coordinate onto the bound its weight points to, so the
    point after the last step is the unsafe corner and only the check after
    the last step can see it. Catches a final check gated on the loss of the
    point before the step, an ignored ``step_size`` (the auto step moves
    0.04), and 'pgd' dispatched to APGD (no check after its last step)."""
    result, cex = falsify(_linear(_W_MIX, _B_MIX), _LB_MIX, _UB_MIX,
                          _unsafe_top_corner(), method="pgd", seed=3,
                          n_restarts=1, n_steps=1, step_size=4.0)
    _assert_is_the_top_corner(result, cex)


_LB_GROUPS = np.array([-2.0, -3.0, 0.0])
_UB_GROUPS = np.array([2.0, 1.0, 4.0])


def _groups_rows_and_disjuncts():
    """Two groups (AND): group A is the OR of A1: y0 >= 8 (no point of the
    box; its margin, at least 6, is always above A2's) and A2: y0 >= 7/4 AND
    y1 <= -11/4 (two rows); group B is y2 >= 15/4. For y = x the
    counterexamples are the corner region x0 >= 7/4, x1 <= -11/4,
    x2 >= 15/4."""
    a1 = HalfSpace(np.array([[-1.0, 0.0, 0.0]]), np.array([[-8.0]]))
    a2 = HalfSpace(np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                   np.array([[-1.75], [-2.75]]))
    b = HalfSpace(np.array([[0.0, 0.0, -1.0]]), np.array([[-3.75]]))
    return [{'Hg': [a1, a2]}, {'Hg': b}]


def _assert_meets_groups_rows_and_disjuncts(result, cex):
    assert result == 0, "the corner region of the property was not found"
    x = np.asarray(cex[0], dtype=np.float64).ravel()
    _assert_in_box(x, _LB_GROUPS, _UB_GROUPS)
    assert x[0] >= 1.75 and x[1] <= -2.75 and x[2] >= 3.75, (
        f"witness {x} is not in the unsafe region")


def test_pgd_reads_every_group_row_and_disjunct():
    """The loss is the max over groups of the min over a group's half-spaces
    of the max over a half-space's rows; its descent lowers the worst term
    until all of A2's rows and B hold (one step of 1/16 per step, at most
    3 * 60 steps). Each wrong reading leaves a coordinate without gradient:
    the min over groups stops once one group holds, the min over rows stops
    once one row of A2 holds, the max over disjuncts follows the infeasible
    A1 for ever. The region is 1/4096 of the box, so the random start almost
    never lies in it."""
    result, cex = falsify(_linear(np.eye(3), np.zeros(3)), _LB_GROUPS,
                          _UB_GROUPS, _groups_rows_and_disjuncts(),
                          method="pgd", seed=3, n_restarts=1, n_steps=200,
                          step_size=0.0625)
    _assert_meets_groups_rows_and_disjuncts(result, cex)


def _trap_net():
    """f(x) = 1 - 4x + 8 relu(x - 1/4) - 8 relu(x - 1/2) on [0, 1]: falls to a
    local minimum f(1/4) = 0, rises to f(1/2) = 1, falls to f(1) = -1. Unsafe
    f <= -3/4 holds for x >= 15/16 only. Descent from x < 1/2 is trapped at
    1/4, descent from x > 1/2 reaches the unsafe region."""
    net = nn.Sequential(nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 1))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0], [1.0], [1.0]]))
        net[0].bias.copy_(torch.tensor([0.0, -0.25, -0.5]))
        net[2].weight.copy_(torch.tensor([[-4.0, 8.0, -8.0]]))
        net[2].bias.copy_(torch.tensor([1.0]))
    return net


def _trap_unsafe():
    return HalfSpace(np.array([[1.0]]), np.array([[-0.75]]))


def _assert_escaped_the_trap(result, cex):
    assert result == 0, "no restart escaped the local minimum"
    x = Fraction(float(np.asarray(cex[0], dtype=np.float64).ravel()[0]))
    relu = lambda v: max(v, Fraction(0))                       # noqa: E731
    f = 1 - 4 * x + 8 * relu(x - Fraction(1, 4)) - 8 * relu(x - Fraction(1, 2))
    assert 0 <= x <= 1 and f <= Fraction(-3, 4), f"witness {x}: f = {f}"


def test_pgd_restarts_from_independent_points():
    """Seed 3 draws the restart points 0.09, 0.24, 0.80, ...: the first two
    are trapped at the local minimum, the third walks (1% steps, at most 44
    of them within n_steps=60) into the unsafe region. Catches restarts that
    all begin at the same point."""
    result, cex = falsify(_trap_net(), np.array([0.0]), np.array([1.0]),
                          _trap_unsafe(), method="pgd", seed=3, n_restarts=8,
                          n_steps=60)
    _assert_escaped_the_trap(result, cex)

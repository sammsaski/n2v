"""Verdicts: SAT comes with a witness that is one, UNSAT is a proof
(DESIGN sections 6 and 17).

A SAT witness is an input of the query: inside the input box, satisfying
every input constraint exactly, with an unsafe output. An UNSAT holds for the
network as real arithmetic defines it on the stored numbers, so a bound that
decides it is rounded outward, never to nearest.

Convention: SAT = unsafe (the unsafe region is ``G y <= g``), UNSAT = safe.
"""

from fractions import Fraction
from functools import lru_cache
from itertools import product

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx import helper

from n2v.nn import NeuralNetwork, vit_crown
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.reach import reach_pytorch_model
from n2v.nn.relational import (falsify_relational, solve_relational,
                               verify_relational)
from n2v.sets import Box, HalfSpace, ProbabilisticBox, Star, Zono
from n2v.utils.model_loader import load_onnx
from n2v.utils.verify_specification import verify_specification
from tests._oracles import interval_proof as proof
from tests._oracles import rational
from tests._oracles.membership import rows
from tests.cell_support import (assert_covered, constrained_star, flat_set,
                                hwc_index, image_set, replay_torch, samples,
                                save_onnx, split_cover)
from tests.known_bugs import DID_NOT_RAISE, known_bug


# --------------------------------------------------------------------------- #
# Relational queries: the witness of a SAT                                      #
# --------------------------------------------------------------------------- #
def _identity():
    layer = nn.Linear(1, 1)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.zero_()
    return nn.Sequential(layer)


def _relational_spec(lb, ub, coupling, unsafe_G, unsafe_g):
    """f and g are both y = x on one input each; the joint input is
    [x_f, x_g] in the box [lb, ub] with ``coupling @ x <= 0``; the unsafe
    region is ``unsafe_G @ [y_f, y_g] <= unsafe_g``."""
    networks = [
        {"name": "f", "relation": None, "input_offset": 0, "input_size": 1,
         "output_offset": 0, "output_size": 1},
        {"name": "g", "relation": ("equal-to", "f"), "input_offset": 1,
         "input_size": 1, "output_offset": 1, "output_size": 1},
    ]
    coupling = np.array(coupling, dtype=float)
    return {
        "format": "relational", "networks": networks,
        "lb": np.array(lb, dtype=float), "ub": np.array(ub, dtype=float),
        "input_coupling": HalfSpace(coupling, np.zeros(len(coupling))),
        "prop": [{"Hg": HalfSpace(np.array(unsafe_G, dtype=float),
                                  np.array(unsafe_g, dtype=float))}],
    }


_EQUAL = [[1, -1], [-1, 1]]                                   # x_f == x_g
_RELATIONAL = {
    # x_f in [0, 1], x_g in [0, 0.5], x_f == x_g: both inputs are in
    # [0, 0.5], so y_g >= 0.6 is unreachable. The sampler draws x_f = 0.77
    # and sets x_g = 0.77, outside x_g's box.
    "box": _relational_spec([0, 0], [1, 0.5], _EQUAL, [[0, -1]], [-0.6]),
    # x_f in [0, 1], x_g in [0.5, 1], x_g <= x_f: x_f >= 0.5, so y_f <= 0.3
    # is unreachable. The sampler draws x_f = 0.09 and sets x_g = 0.5.
    "coupling": _relational_spec([0, 0.5], [1, 1], [[-1, 1]], [[1, 0]], [0.3]),
    # x_f == x_g in [0, 1]: y_g >= 0.6 is reached on 40% of the inputs.
    "reachable": _relational_spec([0, 0], [1, 1], _EQUAL, [[0, -1]], [-0.6]),
}
_UNREACHABLE = [
    pytest.param("box", marks=known_bug(
        6, "the relational sampler sets a coupled input from the coupling "
           "alone, outside that input's own box")),
    pytest.param("coupling", marks=known_bug(
        6, "the relational sampler falls back to the box's lower bound when "
           "box and coupling leave it no interval, violating the coupling")),
]


def _witness_defects(spec, x):
    """What makes x not an input of the query (zero tolerance)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    coupling = spec["input_coupling"]
    defects = []
    if np.any(x < spec["lb"]) or np.any(x > spec["ub"]):
        defects.append(f"outside the box [{spec['lb']}, {spec['ub']}]")
    if np.any(coupling.G @ x > np.asarray(coupling.g).reshape(-1)):
        defects.append("violates the input coupling")
    return defects


@pytest.mark.parametrize("case", _UNREACHABLE + ["reachable"])
def test_relational_search_returns_only_inputs_of_the_query(case):
    spec = _RELATIONAL[case]
    found = falsify_relational(_identity(), _identity(), spec)
    if case == "reachable":
        assert found is not None
    if found is None:
        return
    x, y = found
    assert not _witness_defects(spec, x), (
        f"witness {x}: {'; '.join(_witness_defects(spec, x))}")
    unsafe = spec["prop"][0]["Hg"]
    assert np.all(unsafe.G @ np.asarray(y, dtype=np.float64).reshape(-1, 1)
                  <= unsafe.g)


@pytest.mark.parametrize("case", _UNREACHABLE)
def test_relational_verdict_is_not_sat_when_the_proof_says_unsat(case):
    """The sound half proves these two queries UNSAT; the full verdict, which
    runs the search first, answers SAT."""
    spec = _RELATIONAL[case]
    assert verify_relational(_identity(), _identity(), spec).verdict == "UNSAT"
    verdict, witness = solve_relational(_identity(), _identity(), spec)
    assert verdict != "sat", f"SAT with the witness {witness[0]}"


# --------------------------------------------------------------------------- #
# The LP-free CROWN lane: an UNSAT decided on a rounded margin                  #
# --------------------------------------------------------------------------- #
def _scalar_chain(weights):
    """y = w_k * ... * w_1 * x as one affine op per weight. Every op is a
    single product, so the result does not depend on a summation order."""
    ops = [dict(type="input", **{"in": []}, dim=1, mat=None)]
    for k, w in enumerate(weights):
        ops.append(vit_crown._affine_op(k, [[w]], [0.0], None))
    return ops


@pytest.mark.parametrize("crown_options", [
    {}, {"refine": False}, {"alpha": True}], ids=["default", "ibp", "alpha"])
@known_bug(7, "the CROWN lane decides 'safe' on margin > 0 with the margin "
              "rounded to nearest: a point whose exact output is in the "
              "unsafe region is reported safe")
def test_crown_margin_is_not_decided_by_rounding(crown_options):
    """Confirms the DESIGN section 17 row marked 'unconfirmed'. On a point
    box the relaxation has no gap, so the margin is the network's value. With
    y = 0.3 * 0.3 * 0.3 * 0.1 * x at x = 0.3, float64 evaluation gives
    0.0008100000000000001; the exact product of those five stored numbers
    is below the next float down, g. The point is therefore in the unsafe
    region y <= g, and 'safe' is wrong by about 1e-19."""
    weights, x = [0.1, 0.3, 0.3, 0.3], 0.3
    g = 0.00081
    exact = Fraction(x)
    for w in weights:
        exact *= Fraction(w)
    assert exact <= Fraction(g)                 # the point is unsafe, exactly
    point = np.array([x])
    safe, margin = vit_crown.verify_halfspace_group_safe(
        _scalar_chain(weights), point, point, [[1.0]], [g], **crown_options)
    assert not safe, (
        f"reported safe on a margin of {float(margin[0]):.3e}; the exact "
        f"margin is {float(exact - Fraction(g)):.3e}")


def test_crown_margin_decides_a_point_that_is_safe_by_a_wide_margin():
    point = np.array([0.3])
    safe, _ = vit_crown.verify_halfspace_group_safe(
        _scalar_chain([0.1, 0.3, 0.3, 0.3]), point, point, [[1.0]], [0.0005])
    assert safe
    safe, _ = vit_crown.verify_halfspace_group_safe(
        _scalar_chain([0.1, 0.3, 0.3, 0.3]), point, point, [[1.0]], [0.001])
    assert not safe


# --------------------------------------------------------------------------- #
# The set-based lane: an UNSAT decided on a set rounded to nearest              #
# --------------------------------------------------------------------------- #
# The final check of a reach set against the unsafe region is rounded outward.
# It answers for the set it is given. A set that was itself computed by
# round-to-nearest can leave out an output of the network by one unit in the
# last place, and an affine layer whose terms cancel makes that unit as large
# as the output. Each query below has an input whose exact output is in the
# unsafe region, shown in rationals; no reach set that holds that output can
# be disjoint from the region.
def _then_affine(first, weight, bias):
    """``first``, then y -> weight * y + bias. Returns the model with the two
    numbers as the float32 layer stores them."""
    last = nn.Linear(1, 1)
    with torch.no_grad():
        last.weight.fill_(weight)
        last.bias.fill_(bias)
    return (nn.Sequential(first, last),
            Fraction(last.weight.item()), Fraction(last.bias.item()))


def _set_verdict(model, kind, lo, hi, g):
    """The verdict on 'y <= g is unsafe' for the input [lo, hi] as ``kind``."""
    lo, hi = np.array([[lo]]), np.array([[hi]])
    if kind == "Box":
        input_set = Box(lo, hi)
    else:
        input_set = {"Star": Star, "Zono": Zono}[kind].from_bounds(lo, hi)
    sets = reach_pytorch_model(model, input_set, method="approx")
    unsafe = HalfSpace(np.array([[1.0]]), np.array([[g]]))
    return verify_specification(sets, unsafe).verdict


# activation: (module, the function for the exact value, a range on which
# float64 evaluates it to 1.0 at both ends)
_SATURATED = {
    "Tanh": (nn.Tanh, proof.TANH, 20.0, 40.0),
    "Sigmoid": (nn.Sigmoid, proof.SIGMOID, 40.0, 80.0),
}
_SATURATED_INTERVAL_IS_A_POINT = known_bug(
    7, "the Zono rule for Sigmoid and Tanh stores [f(lb), f(ub)] as float64 "
       "computes them: on a saturated range both are 1.0, the reach set is "
       "the point 1, and the next layer maps it to the point 0")
_CONSTANT_NEURON_IS_A_POINT = known_bug(
    7, "a neuron whose input is one value gets f of it as float64 computes "
       "it: 1.0 here, which the next layer maps to the point 0")


@pytest.mark.parametrize("activation", sorted(_SATURATED))
@pytest.mark.parametrize("inputs,kind", [
    ("range", "Star"),
    pytest.param("range", "Zono", marks=_SATURATED_INTERVAL_IS_A_POINT),
    ("range", "Box"),
    pytest.param("point", "Star", marks=_CONSTANT_NEURON_IS_A_POINT),
    pytest.param("point", "Zono", marks=_CONSTANT_NEURON_IS_A_POINT),
    ("point", "Box"),
])
def test_unsat_after_a_saturated_activation_is_not_decided_by_rounding(
        activation, inputs, kind):
    """y = W * f(x) - W with W = 1e17 as float32 stores it. In float64 f(x) is
    1.0 from the lower end of the range up, so every output computes to 0. In
    fact f is below 1 there, by 8.5e-18 for Tanh at 20 and 4.2e-18 for Sigmoid
    at 40, and the output at the lower end is -0.85 or -0.42: in the unsafe
    region y <= -0.25. The Star on a range answers UNKNOWN because it keeps
    -W and W * f apart until the final check, whose slack is relative to
    them; the Box because its bounds are rounded outward at every step."""
    module, function, lo, hi = _SATURATED[activation]
    model, weight, bias = _then_affine(module(), 1e17, -1e17)
    g = -0.25
    _, (upper,) = proof.value(
        proof.chain(function, proof.linear([[weight]], [bias])),
        [Fraction(lo)])
    if upper > Fraction(g):
        pytest.fail(f"the output at x = {lo} is not shown to be unsafe: it "
                    f"is only known to be at most {float(upper)!r}")
    verdict = _set_verdict(model, kind, lo, lo if inputs == "point" else hi, g)
    assert verdict != "UNSAT", (
        f"UNSAT, and the output at x = {lo} is at most {float(upper)!r}, "
        f"inside the unsafe region y <= {g}")


_AFFINE_MAP_ROUNDED_TO_NEAREST = known_bug(
    "7b",
    "the affine map of a Star and of a Zono is computed by round-to-nearest "
    "with no error term, so a Star or Zono UNSAT holds only beyond the "
    "rounding radius of DESIGN section 1, not for the real-valued network; "
    "with the error term carried to the final check an exact output in the "
    "unsafe region, as here, can no longer be answered UNSAT")


@pytest.mark.parametrize("kind", [
    pytest.param("Star", marks=_AFFINE_MAP_ROUNDED_TO_NEAREST),
    pytest.param("Zono", marks=_AFFINE_MAP_ROUNDED_TO_NEAREST),
    "Box",
])
def test_unsat_after_an_affine_map_is_not_decided_by_rounding(kind):
    """No activation at all: y = 2**58 * (w * x) + b at the one input x = 0.7,
    with w = 0.3 as float32 stores it and b the float32 number nearest to
    -2**58 * (w * x) as float64 computes that product. The product of the two
    stored numbers needs 77 bits; float64 rounds it up by 1.3e-17, the second
    layer multiplies the difference by 2**58, and b cancels the rest. The
    float64 output is 0 and the exact one is -3.84, in the unsafe region
    y <= -1. The input is a float64 number, as a decimal in a VNN-LIB file
    gives; with a float32 input the first product would be exact."""
    x, g = 0.7, -1.0
    first = nn.Linear(1, 1)
    with torch.no_grad():
        first.weight.fill_(0.3)
        first.bias.zero_()
    w = first.weight.item()
    model, weight, bias = _then_affine(
        first, 2.0 ** 58, np.float32(-(2.0 ** 58 * (w * x))).item())
    exact = weight * Fraction(w) * Fraction(x) + bias
    if exact > Fraction(g):
        pytest.fail(f"the exact output {float(exact)!r} is not in the unsafe "
                    f"region y <= {g}")
    verdict = _set_verdict(model, kind, x, x, g)
    assert verdict != "UNSAT", (
        f"UNSAT, and the exact output at x = {x} is {float(exact)!r}, inside "
        f"the unsafe region y <= {g}")


# --------------------------------------------------------------------------- #
# The Star and Zono lanes: what an UNSAT does claim (the rounding radius)       #
# --------------------------------------------------------------------------- #
# DESIGN section 1 (owner decision of 2026-11-13). The affine map of a Star and
# of a Zono is computed in float64 rounded to nearest and the rounding is not
# tracked, so an UNSAT from these two lanes claims less than "safe for the
# real-valued network": every exact output of the real-valued network (the
# stored weights in real arithmetic) lies within a rounding radius delta of a
# point of a returned set. delta is 0 at the input; an affine map with n
# inputs gives
#
#     delta_out = |W| delta_in + gamma_(n+1) (|W| T + |b|) + n eta A,
#
# with gamma_k = k u / (1 - k u), u = 2^-53, T = |c| + |V| max|alpha| of the
# set that enters the map, and the underflow term: a product that lands in the
# subnormal range is off by up to eta / 2, eta = 2^-1074, on top of the
# relative error, and A = 1 + sum_j max|alpha_j| weighs the centre and the
# generators. A ReLU leaves delta unchanged (it is 1-Lipschitz in each
# coordinate and its kernels are sound for the stored set). An UNSAT then
# proves that no input has an exact output y with G y <= g - |G| delta.
#
# The tests below check that claim and nothing more. They pass on lanes that
# are NOT exact for the real-valued network: the strict expected failures of
# ``test_unsat_after_an_affine_map_is_not_decided_by_rounding`` show an exact
# output in the unsafe region answered UNSAT. Carrying delta to the final
# check is PLAN step 7b. True outputs are exact fractions of the stored
# numbers, never n2v and never a float32 forward pass (owner decision 2).
_U = Fraction(1, 2 ** 53)
_ETA = Fraction(1, 2 ** 1074)


def _gamma(k):
    """Higham's ``gamma_k = k u / (1 - k u)``, exactly."""
    return k * _U / (1 - k * _U)


def _q(x):
    """The rational a float64 number is."""
    return Fraction(float(x))


def _qmatrix(W):
    return [[_q(w) for w in row]
            for row in np.atleast_2d(np.asarray(W, dtype=np.float64))]


def _qvector(b, size):
    if b is None:
        return [Fraction(0)] * size
    return [_q(v) for v in np.asarray(b, dtype=np.float64).reshape(-1)]


def _qdot(a, x):
    return sum((ai * xi for ai, xi in zip(a, x)), Fraction(0))


def _up(q):
    """The smallest float64 at or above the rational ``q``."""
    g = float(q)
    return g if Fraction(g) >= q else float(np.nextafter(g, np.inf))


def _magnitudes(sets):
    """``(T, A)`` of the sets that enter an affine map, read from their stored
    rows (``membership.rows``): ``T_i = |c_i| + sum_j |V_ij| max|alpha_j|``
    and ``A = 1 + sum_j max|alpha_j|``, each the largest over the sets."""
    T, A = None, Fraction(0)
    for s in sets:
        c, G, _, _, lb, ub = rows(s)
        amax = [max(abs(_q(lo)), abs(_q(hi))) for lo, hi in zip(lb, ub)]
        t = [abs(_q(ci)) + _qdot([abs(_q(g)) for g in row], amax)
             for ci, row in zip(c, G)]
        T = t if T is None else [max(x, y) for x, y in zip(T, t)]
        A = max(A, 1 + sum(amax, Fraction(0)))
    return T, A


def _rounding_radius(maps):
    """The rounding radius ``delta`` after a chain of affine maps, in exact
    fractions (a ReLU between two maps leaves it unchanged, so it has no
    entry). ``maps`` lists ``(W, b, T, A)`` per map: the weight (rows are
    outputs) and the bias (or None) as float64 arrays of the stored numbers,
    and the magnitudes of the set that enters the map (:func:`_magnitudes`
    or :func:`_apriori_magnitudes`). Larger ``T`` or ``A`` only enlarge
    ``delta``, so an upper bound of them gives a radius the claim covers."""
    delta = None
    for W, b, T, A in maps:
        W = _qmatrix(W)
        n = len(W[0])
        b = _qvector(b, len(W))
        if delta is None:
            delta = [Fraction(0)] * n
        g = _gamma(n + 1)
        delta = [_qdot([abs(w) for w in row], delta)
                 + g * (_qdot([abs(w) for w in row], T) + abs(bi))
                 + n * _ETA * A
                 for row, bi in zip(W, b)]
    return delta


def _layers_of(model):
    """``('affine', W, b)`` or ``('relu',)`` for each module of an
    ``nn.Sequential`` of Linear and ReLU; W and b as float64 arrays of the
    stored float32 numbers (exact)."""
    out = []
    for module in model:
        if isinstance(module, nn.Linear):
            b = (None if module.bias is None
                 else module.bias.detach().double().numpy())
            out.append(("affine", module.weight.detach().double().numpy(), b))
        elif isinstance(module, nn.ReLU):
            out.append(("relu",))
        else:
            raise TypeError(f"no rule for {module!r}")
    return out


def _apriori_magnitudes(layers, T0, A0):
    """Upper bounds of ``(T, A)`` of every set n2v can hand to each affine
    map of ``layers``, from those of the input set, without running n2v.

    An affine map stores each entry within gamma_(n+1) of ``|W| T + |b|``
    (plus the underflow term) and keeps the predicate, so ``T`` grows to at
    most ``(1 + gamma_(n+1)) (|W| T + |b|) + n eta A``. A ReLU, exact or
    relaxed, keeps or zeroes rows; a relaxed neuron gets one new predicate
    variable bounded by 1 (Zono) or by its padded upper bound (Star, at most
    ``T_i (1 + 1e-7) + 1e-7``), and its row has magnitude at most ``2 T_i``
    (Zono) or that bound (Star). ``2 T + 2^-20`` and ``A + m + sum_i (2 T_i +
    2^-20)`` are above all of these."""
    out = []
    T, A = list(T0), A0
    for layer in layers:
        if layer[0] == "relu":
            T = [2 * t + Fraction(1, 2 ** 20) for t in T]
            A = A + len(T) + sum(T, Fraction(0))
            continue
        _, W, b = layer
        out.append((T, A))
        W = _qmatrix(W)
        n = len(W[0])
        b = _qvector(b, len(W))
        T = [(1 + _gamma(n + 1)) * (_qdot([abs(w) for w in row], T) + abs(bi))
             + n * _ETA * A for row, bi in zip(W, b)]
    return out


# ----------------------------- the kernel level ---------------------------- #
def _centre_and_generators(s):
    """Centre and generator columns of a Star or Zono as float64 arrays."""
    if isinstance(s, Star):
        V = np.asarray(s.V, dtype=np.float64)
        return V[:, 0], V[:, 1:]
    return (np.asarray(s.c, dtype=np.float64).reshape(-1),
            np.asarray(s.V, dtype=np.float64).reshape(s.dim, -1))


def _with_generators(kind, c, G, cut=True):
    """A Zono, or a Star over the predicate box [-1, 1]^k, with centre ``c``
    and generator columns ``G``. With ``cut`` the Star also has a row that
    binds (half the sum of the variables at most a quarter of their count),
    which the affine map must keep."""
    c = np.asarray(c, dtype=np.float64).reshape(-1, 1)
    G = np.asarray(G, dtype=np.float64).reshape(c.shape[0], -1)
    if kind == "Zono":
        return Zono(c, G)
    k = G.shape[1]
    C, d = np.vstack([np.eye(k), -np.eye(k)]), np.ones((2 * k, 1))
    if cut:
        C = np.vstack([C, np.full((1, k), 0.5)])
        d = np.vstack([d, [[0.25 * k]]])
    return Star(np.hstack([c, G]), C, d, -np.ones((k, 1)), np.ones((k, 1)))


def _float32(a):
    """``a`` rounded to float32 and back: a weight as a float32 layer holds
    it."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def _the_2pow58_maps(kind):
    """The two layers of the 2^58 network of
    ``test_unsat_after_an_affine_map_is_not_decided_by_rounding`` as
    ``(input set, W, b)``: the point 0.7 through w = float32(0.3), then n2v's
    result of that map through 2^58 with the cancelling float32 bias."""
    x = 0.7
    w = float(_float32(0.3))
    first = {"Star": Star, "Zono": Zono}[kind].from_bounds(
        np.array([[x]]), np.array([[x]]))
    W1, b1 = np.array([[w]]), np.array([0.0])
    W2 = np.array([[2.0 ** 58]])
    b2 = _float32([-(2.0 ** 58 * (w * x))])
    return [(first, W1, b1), (first.affine_map(W1, b1), W2, b2)]


def _subnormal_case(kind):
    """Weights and set entries of order 2^-540: every product lands in the
    subnormal range, where rounding is absolute (row 0: the exact centre is
    45/64 eta and float64 gives eta)."""
    W = np.array([[3 * 2.0 ** -540, -5 * 2.0 ** -542],
                  [7 * 2.0 ** -541, 2.0 ** -539]])
    c = np.array([5 * 2.0 ** -538, 3 * 2.0 ** -538])
    G = np.array([[2.0 ** -537, 3 * 2.0 ** -539], [-7 * 2.0 ** -540, 2.0 ** -536]])
    return _with_generators(kind, c, G), W, None


def _random_case(kind, m, n, k, seed, mixed=False):
    """Float32 weights and bias of shape (m, n), a float64 centre and k
    generators; with ``mixed`` every entry is scaled by its own power of two
    in [2^-30, 2^30] (weights) or [2^-20, 2^20] (set entries)."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(-1, 1, (m, n))
    b = rng.uniform(-1, 1, m)
    c = rng.normal(size=n)
    G = rng.normal(size=(n, k))
    if mixed:
        W = W * 2.0 ** rng.integers(-30, 31, (m, n))
        b = b * 2.0 ** rng.integers(-30, 31, m)
        c = c * 2.0 ** rng.integers(-20, 21, n)
        G = G * 2.0 ** rng.integers(-20, 21, (n, k))
    return _with_generators(kind, c, G), _float32(W), _float32(b)


_KERNEL_CASES = {
    "2pow58-first-layer": lambda kind: _the_2pow58_maps(kind)[0],
    "2pow58-second-layer": lambda kind: _the_2pow58_maps(kind)[1],
    "cancelling": lambda kind: (
        _with_generators(kind, [1e16, 1.0], [[-1e16], [0.0]], cut=False),
        np.array([[0.1, 0.3]]), np.array([0.0])),
    "float32-7x13": lambda kind: _random_case(kind, 7, 13, 4, seed=1),
    "float32-wide-3x300": lambda kind: _random_case(kind, 3, 300, 5, seed=2),
    "float32-mixed-magnitude-5x9": lambda kind: _random_case(
        kind, 5, 9, 6, seed=3, mixed=True),
    "subnormal-products": _subnormal_case,
}


def _affine_map_excess(input_set, W, b, out, underflow=True):
    """Entries of ``out``, n2v's affine map of ``input_set``, that are further
    from the exact map of the stored input than the rounding bound, as
    ``(entry, |error|, bound)``. The bound is ``gamma_(n+1) (|W||c| + |b|)``
    for the centre and ``gamma_n |W||V|`` for a generator, each plus ``n
    eta`` (one underflow per product) unless ``underflow`` is False. Both
    sides are fractions of the stored float64 numbers."""
    c, G = _centre_and_generators(input_set)
    c2, G2 = _centre_and_generators(out)
    Wq = _qmatrix(W)
    n = len(Wq[0])
    bq = _qvector(b, len(Wq))
    eta = n * _ETA if underflow else Fraction(0)
    cq, Gq = [_q(v) for v in c], _qmatrix(G) if G.size else [[]] * len(c)
    excess = []
    for i, row in enumerate(Wq):
        exact = _qdot(row, cq) + bq[i]
        bound = _gamma(n + 1) * (_qdot([abs(w) for w in row],
                                       [abs(v) for v in cq]) + abs(bq[i])) + eta
        error = abs(_q(c2[i]) - exact)
        if error > bound:
            excess.append((f"centre[{i}]", float(error), float(bound)))
        for j in range(G.shape[1]):
            column = [Gq[r][j] for r in range(len(cq))]
            exact = _qdot(row, column)
            bound = _gamma(n) * _qdot([abs(w) for w in row],
                                      [abs(v) for v in column]) + eta
            error = abs(_q(G2[i, j]) - exact)
            if error > bound:
                excess.append((f"generator[{i}, {j}]", float(error),
                               float(bound)))
    return excess


@pytest.mark.parametrize("case", sorted(_KERNEL_CASES))
@pytest.mark.parametrize("kind", ["Star", "Zono"])
def test_affine_map_entries_are_within_the_rounding_bound(kind, case):
    """The per-map half of the narrowed claim: ``Star.affine_map`` and
    ``Zono.affine_map`` store every entry within the rounding bound of the
    exact map of the stored input (computed in fractions), keep every
    generator column, and (Star) keep the predicate rows and box as they
    are. It is a bound on rounding, not exactness: entries differ from the
    exact map. Catches a map that rounds its result to float32, drops or
    zeroes a generator column, leaves out or flips the bias, or uses |W| for
    the generators; the wide case (n = 300) and the mixed magnitudes keep the
    bound at gamma_301 and at cancelling sums of terms 2^60 apart."""
    input_set, W, b = _KERNEL_CASES[case](kind)
    out = input_set.affine_map(W, b)
    c, G = _centre_and_generators(input_set)
    m = np.atleast_2d(W).shape[0]
    assert type(out) is type(input_set)
    c2, G2 = _centre_and_generators(out)
    assert c2.shape == (m,) and G2.shape == (m, G.shape[1]), (
        f"{kind}.affine_map returned centre {c2.shape} and generators "
        f"{G2.shape}; the input has {G.shape[1]} generators and W has {m} "
        f"rows")
    if kind == "Star":
        assert out.nVar == input_set.nVar
        for name in ("C", "d", "predicate_lb", "predicate_ub"):
            assert np.array_equal(np.asarray(getattr(out, name)),
                                  np.asarray(getattr(input_set, name))), (
                f"Star.affine_map changed the predicate ({name})")
    excess = _affine_map_excess(input_set, W, b, out)
    assert not excess, (
        f"{kind}.affine_map ({case}): {len(excess)} entries outside the "
        f"rounding bound, first {excess[:3]} as (entry, |error|, bound)")


@pytest.mark.parametrize("kind", ["Star", "Zono"])
def test_the_rounding_bound_needs_the_set_magnitudes_not_the_point(kind):
    """Why the bound of the narrowed claim is written with the set's term
    magnitude ``T = |c| + |V| max|alpha|`` and not with the represented
    point ``x``: centre (1e16, 1) and generator (-1e16, 0) hold the point
    (0, 1) at alpha = 1; with W = (0.1, 0.3) the exact image is 0.3 and the
    mapped set holds 0.25 there. The per-point bound ``gamma_3 (|W||x| +
    |b|) + 2 eta`` (more generous than one with gamma_2) is
    violated by 0.05; the bound with T holds. Kept so that nobody tightens
    the radius of these tests to the per-point form."""
    input_set, W, b = _KERNEL_CASES["cancelling"](kind)
    out = input_set.affine_map(W, b)
    c, G = _centre_and_generators(input_set)
    c2, G2 = _centre_and_generators(out)
    x = [_q(ci) + _q(gi) for ci, gi in zip(c, G[:, 0])]       # alpha = 1
    if x != [0, 1]:
        pytest.fail(f"the represented input point is {x}, not (0, 1)")
    Wq, bq = _qmatrix(W)[0], _qvector(b, 1)[0]
    exact = _qdot(Wq, x) + bq
    held = _q(c2[0]) + _q(G2[0, 0])
    error = abs(held - exact)
    per_point = (_gamma(3) * (_qdot([abs(w) for w in Wq], [abs(v) for v in x])
                              + abs(bq)) + 2 * _ETA)
    T, A = _magnitudes([input_set])
    with_T = _gamma(3) * (_qdot([abs(w) for w in Wq], T) + abs(bq)) + 2 * _ETA * A
    assert error > per_point, (
        f"the per-point bound {float(per_point):.3g} covers the error "
        f"{float(error):.3g}: this case no longer shows why T is needed")
    assert error <= with_T


@pytest.mark.parametrize("kind", ["Star", "Zono"])
def test_the_rounding_bound_needs_the_underflow_term(kind):
    """The subnormal case: products of order 2^-1080 round to a multiple of
    2^-1074 with an absolute error that no multiple of u covers. The bound
    without the term ``n eta`` is violated there, the bound with it holds.
    Kept so that nobody drops the underflow term from the radius."""
    input_set, W, b = _subnormal_case(kind)
    out = input_set.affine_map(W, b)
    assert _affine_map_excess(input_set, W, b, out, underflow=False), (
        "no entry needs the underflow term: the case no longer shows it")
    assert not _affine_map_excess(input_set, W, b, out)


# -------------------------- the radius, as an oracle ------------------------ #
def test_rounding_radius_oracle_on_hand_computed_cases():
    """The recursion that every verdict test below uses, on numbers worked
    out by hand. (1) W = (1, -2), b = 3, T = (1, 1), A = 1: delta = gamma_3 *
    6 + 2 eta. Then W = (4; -1/2), b = (0, 1), T = 7, A = 2: delta = (4
    delta_1 + gamma_2 * 28 + 2 eta, delta_1 / 2 + gamma_2 * 9/2 + 2 eta). (2)
    The 2^58 network: delta = 40.32, which covers
    the float64 error 3.84 of that network and puts the tightest threshold
    the claim covers at 36.48."""
    g2 = Fraction(2, 2 ** 53 - 2)
    g3 = Fraction(3, 2 ** 53 - 3)
    eta = Fraction(1, 2 ** 1074)
    d1 = _rounding_radius([([[1.0, -2.0]], [3.0], [1, 1], 1)])
    assert d1 == [6 * g3 + 2 * eta]
    d2 = _rounding_radius([([[1.0, -2.0]], [3.0], [1, 1], 1),
                           ([[4.0], [-0.5]], [0.0, 1.0], [7], 2)])
    assert d2 == [4 * d1[0] + 28 * g2 + 2 * eta,
                  d1[0] / 2 + Fraction(9, 2) * g2 + 2 * eta]

    (s1, W1, b1), (_, W2, b2) = _the_2pow58_maps("Star")
    x, w = 0.7, W1[0, 0]
    p1 = w * x                                  # float64, as the map rounds
    delta = _rounding_radius([(W1, b1, [_q(x)], 1),
                              (W2, b2, [abs(_q(p1))], 1)])[0]
    exact = _q(W2[0, 0]) * _q(w) * _q(x) + _q(b2[0])
    float64 = _q(W2[0, 0] * p1 + b2[0])
    assert f"{float(delta):.4g}" == "40.32"
    assert abs(float64 - exact) <= delta
    assert f"{_up(exact + delta):.4g}" == "36.48"


# --------------------------------- verdicts --------------------------------- #
_ENTRIES = ("reach_layer", "NeuralNetwork.reach",
            "reach_pytorch_model(load_onnx)")


def _onnx_of(model, tmp_path, n_in, n_out):
    """The ``nn.Sequential`` of Linear and ReLU as an ONNX graph of Gemm and
    Relu nodes with its float32 numbers."""
    nodes, inits, current = [], {}, "X"
    for k, module in enumerate(model):
        out = "Y" if k == len(model) - 1 else f"h{k}"
        if isinstance(module, nn.Linear):
            inits[f"W{k}"] = module.weight.detach().numpy().astype(np.float32)
            inputs = [current, f"W{k}"]
            if module.bias is not None:
                inits[f"B{k}"] = module.bias.detach().numpy().astype(np.float32)
                inputs.append(f"B{k}")
            nodes.append(helper.make_node("Gemm", inputs, [out], transB=1))
        else:
            nodes.append(helper.make_node("Relu", [current], [out]))
        current = out
    return save_onnx(tmp_path, nodes, [1, n_in], [1, n_out], inits)


def _reach(entry, model, input_set, method, tmp_path):
    """Reach sets of ``model`` through ``entry`` and the magnitudes of the
    sets that enter each affine map: read from the sets themselves when the
    layers are composed with ``reach_layer``; bounded a priori
    (:func:`_apriori_magnitudes`) for the two model entries, which return
    only the last sets."""
    if entry == "reach_layer":
        sets, magnitudes = [input_set], []
        for module in model:
            if isinstance(module, nn.Linear):
                magnitudes.append(_magnitudes(sets))
            sets = reach_layer(module, sets, method)
            assert sets, f"reach_layer({module}) returned no sets"
        return sets, magnitudes
    magnitudes = _apriori_magnitudes(_layers_of(model),
                                     *_magnitudes([input_set]))
    if entry == "NeuralNetwork.reach":
        sets = NeuralNetwork(model).reach(input_set, method=method)
    else:
        n_in = model[0].in_features
        n_out = [m for m in model if isinstance(m, nn.Linear)][-1].out_features
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = _onnx_of(model, tmp_path, n_in, n_out)
        sets = reach_pytorch_model(load_onnx(path), input_set, method=method)
    assert sets, f"{entry} returned no sets for a non-empty input"
    return sets, magnitudes


def _not_unsat_within_the_radius(model, sets, magnitudes, lowest, G_rows):
    """For each row G, the unsafe region ``G y <= g`` with ``g`` the smallest
    float64 at or above ``lowest(G) + |G| delta`` (``lowest(G)`` is the least
    ``G z`` over exact outputs z of member points): the claim leaves an exact
    output in that region, so the verdict must not be UNSAT. Returns the rows
    answered UNSAT, with the threshold."""
    maps = [(W, b, T, A) for (_, W, b), (T, A) in zip(
        [layer for layer in _layers_of(model) if layer[0] == "affine"],
        magnitudes)]
    delta = _rounding_radius(maps)
    wrong = []
    for G in G_rows:
        Gq = [Fraction(v) for v in G]
        g = _up(lowest(Gq) + _qdot([abs(v) for v in Gq], delta))
        unsafe = HalfSpace(np.array([G], dtype=np.float64), np.array([[g]]))
        if verify_specification(sets, unsafe).verdict == "UNSAT":
            wrong.append((G, g, [float(d) for d in delta]))
    return wrong


def _input_point_set(kind, x):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    return {"Star": Star, "Zono": Zono}[kind].from_bounds(x, x.copy())


def _two_layer_model(first_weight, second_weight, second_bias, first_bias=None):
    """Linear(n, 1) then Linear(1, 1), float32 parameters."""
    first_weight = np.atleast_2d(first_weight)
    first = nn.Linear(first_weight.shape[1], 1, bias=first_bias is not None)
    last = nn.Linear(1, 1)
    with torch.no_grad():
        first.weight.copy_(torch.as_tensor(first_weight, dtype=torch.float32))
        if first_bias is not None:
            first.bias.copy_(torch.as_tensor(first_bias, dtype=torch.float32))
        last.weight.fill_(float(second_weight))
        last.bias.fill_(float(second_bias))
    return nn.Sequential(first, last)


def _exact_output(model, x):
    """The real-valued network on the stored numbers at the input ``x``, in
    fractions (Linear and ReLU only)."""
    v = [_q(t) for t in np.ravel(x)]
    for layer in _layers_of(model):
        if layer[0] == "relu":
            v = [max(t, Fraction(0)) for t in v]
            continue
        _, W, b = layer
        v = [_qdot(row, v) + bi for row, bi in zip(_qmatrix(W),
                                                     _qvector(b, len(W)))]
    return v


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("kind", ["Star", "Zono"])
def test_2pow58_network_is_not_unsat_within_the_rounding_radius(
        kind, entry, tmp_path):
    """The 2^58 network at x = 0.7: the exact output is -3.84 and float64
    gives 0, so the narrowed claim needs a radius of 40.32 there and the
    tightest threshold it covers is y <= 36.48 (and -y <= 44.16). Neither
    may be UNSAT. This is all the Star and Zono lanes claim: they are not
    exact for the real-valued network, and at y <= -1 they answer UNSAT
    (the expected failures above). Catches an affine map or a Linear kernel
    that leaves out or flips the bias, and a model entry that reaches a
    different network. It cannot see an affine map rounded to float32: the
    bias is float32(-2^58 times the float64 product), so a float32 product
    cancels it exactly (shown with a planted bug); the float32 family and
    the ReLU network below catch that."""
    w = float(_float32(0.3))
    model = _two_layer_model([[w]], 2.0 ** 58, _float32(-(2.0 ** 58 * (w * 0.7))),
                             first_bias=[0.0])
    z = _exact_output(model, [0.7])
    if z[0] > -3:
        pytest.fail(f"the exact output is {float(z[0])}, not -3.84")
    sets, magnitudes = _reach(entry, model, _input_point_set(kind, [0.7]),
                              "approx", tmp_path)
    wrong = _not_unsat_within_the_radius(
        model, sets, magnitudes, lambda G: _qdot(G, z), [[1.0], [-1.0]])
    assert not wrong, (
        f"UNSAT within the rounding radius of the exact output "
        f"{float(z[0])!r}: (row, threshold, delta) {wrong}")


def _float32_cancellations(count=24, seed=20261113):
    """The two-input float32 family: every input and
    weight a float32 number, w = (w0, w1 * 2^-9), then 2^58 and the float32
    bias nearest to minus 2^58 times the float64 product. On 400 measured
    cases the float64 output is off by up to 16 and the radius is up to about
    212; on these 24 (measured) the float64 error is up to 14 (15 cases at 1
    or more) and the radius runs from 5.3 to 161."""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(count):
        w = [float(_float32(rng.uniform(0.1, 1.0))),
             float(_float32(rng.uniform(0.1, 1.0))) * 2.0 ** -9]
        x = [float(_float32(rng.uniform(0.1, 1.0))),
             float(_float32(rng.uniform(0.1, 1.0)))]
        y64 = float(np.dot(w, x))
        cases.append((w, x, float(_float32(-(2.0 ** 58 * y64)))))
    return cases


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("kind", ["Star", "Zono"])
def test_float32_cancellations_are_not_unsat_within_the_rounding_radius(
        kind, entry, tmp_path):
    """24 networks of the two-input float32 family, whose old threshold at
    the exact output the Star and Zono lanes answer UNSAT in 152 of 400
    cases: with the threshold moved out by the
    rounding radius in either direction, none may be UNSAT. A float32
    input does not make a two-input product exact, which is why this
    family exists. Catches an affine map whose product or result is rounded
    to float32 and a dropped or flipped bias, for Star and Zono through all
    three entries (shown with planted bugs)."""
    wrong, errors = [], []
    for k, (w, x, b) in enumerate(_float32_cancellations()):
        model = _two_layer_model([w], 2.0 ** 58, b)
        z = _exact_output(model, x)
        y64 = float(np.float64(2.0 ** 58) * float(np.dot(w, x))
                    + b)
        errors.append(abs(float(_q(y64) - z[0])))
        sets, magnitudes = _reach(entry, model, _input_point_set(kind, x),
                                  "approx", tmp_path / f"case{k}")
        bad = _not_unsat_within_the_radius(
            model, sets, magnitudes, lambda G: _qdot(G, z), [[1.0], [-1.0]])
        wrong += [(k, row, g) for row, g, _ in bad]
    if max(errors) < 1:
        pytest.fail(f"no case has a float64 error of 1 or more "
                    f"({max(errors)}): the family lost its point")
    assert not wrong, (
        f"UNSAT within the rounding radius: (case, row, threshold) {wrong}")


def _relu_network():
    """Linear(3, 4), ReLU, Linear(4, 2) with float32 weights of mixed
    magnitude (2^-3 to 2^3) and signs, none of them dyadic, so every affine
    map rounds."""
    rng = np.random.default_rng(0)
    W1 = rng.uniform(-1, 1, (4, 3)) * 2.0 ** rng.integers(-3, 4, (4, 3))
    b1 = rng.uniform(-0.5, 0.5, 4)
    W2 = rng.uniform(-1, 1, (2, 4)) * 2.0 ** rng.integers(-3, 4, (2, 4))
    b2 = rng.uniform(-0.5, 0.5, 2)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    with torch.no_grad():
        for module, W, b in ((model[0], W1, b1), (model[2], W2, b2)):
            module.weight.copy_(torch.as_tensor(W, dtype=torch.float32))
            module.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    return model


# Input boxes of the ReLU network: every neuron crosses zero on "crossing";
# on "stable" two are active and two inactive, so the relaxations are exact
# and the threshold is tight for every method.
_RELU_BOXES = {
    "crossing": ([-1.0, -0.5, 0.25], [1.0, 0.75, 1.5]),
    "stable": ([0.53125, 0.65625, -0.78125], [0.625, 0.71875, -0.65625]),
}


@lru_cache(maxsize=None)
def _relu_network_image(box):
    """The exact image of the box under the ReLU network, as rational pieces
    (``tests/_oracles/rational.py``), and the number of neurons whose exact
    pre-activation range crosses zero."""
    lo, hi = _RELU_BOXES[box]
    (_, W1, b1), _, (_, W2, b2) = _layers_of(_relu_network())
    hidden = rational.affine(rational.box(lo, hi), W1, b1)
    crossing = 0
    for i in range(len(W1)):
        values = [piece.value(v)[i] for piece in hidden
                  for v in piece.corners()]
        crossing += min(values) < 0 < max(values)
    return rational.affine(rational.relu(hidden), W2, b2), crossing


_RELU_ROWS = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
              [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0, -2.0]]


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("box", sorted(_RELU_BOXES))
@pytest.mark.parametrize("kind,method", [
    ("Star", "exact"), ("Star", "approx"), ("Zono", "approx")])
def test_relu_network_is_not_unsat_within_the_rounding_radius(
        kind, method, box, entry, tmp_path):
    """A small ReLU network on a box, for eight directions G: with the
    threshold at the least exact ``G z`` over the box (the exact image in
    rationals) plus ``|G| delta``, the verdict must not be UNSAT. On the
    stable box and with the exact method the returned sets reach that least
    value up to rounding, so a kernel off by more than the radius shows;
    on the crossing box the relaxations are loose and the cell guards the
    composition. The Zono ReLU relaxation itself rounds to nearest by a few
    units in the last place (registry ROUND-10), which the radius does not
    cover: no neuron is relaxed on the stable box, and on the crossing box
    the relaxation's own slack is far larger. Every result also holds 128
    torch float64 outputs of the box (LP membership on its rows). Catches
    an affine map rounded to float32, a dropped or zeroed generator column,
    a dropped bias, |W| for the Zono generators or centre, and a Linear
    kernel that drops the bias (shown with planted bugs)."""
    pieces, crossing = _relu_network_image(box)
    if (box == "stable") != (crossing == 0) or (box == "crossing"
                                                 and crossing < 4):
        pytest.fail(f"the {box} box has {crossing} crossing neurons")
    model = _relu_network()
    lo, hi = (np.array(v) for v in _RELU_BOXES[box])
    input_set = {"Star": Star, "Zono": Zono}[kind].from_bounds(
        lo.reshape(-1, 1), hi.reshape(-1, 1))
    sets, magnitudes = _reach(entry, model, input_set, method, tmp_path)

    def lowest(G):
        return min(_qdot(G, y) for piece in pieces for y in piece.image())

    wrong = _not_unsat_within_the_radius(model, sets, magnitudes, lowest,
                                         _RELU_ROWS)
    assert not wrong, (
        f"UNSAT within the rounding radius: (row, threshold, delta) {wrong}")
    assert_covered(sets, replay_torch(model, samples(lo, hi), (1, 3)),
                   f"{kind} {method} {entry} on the {box} box")


# --------------------------------------------------------------------------- #
# verify_specification on given sets: the verdict against the exact answer      #
# --------------------------------------------------------------------------- #
# The reach sets here are given, not computed, so the cells check the final
# step alone: the reading of the property, the dispatch by set type and the
# disjointness checks, and the map from what they found to the verdict. The
# exact answer comes from the set's own stored rows (``rational.star``) in
# fractions, never from n2v: a set meets the unsafe region ``G y <= g`` exactly
# when its predicate rows together with ``G (c + V a) <= g`` have a point. An
# image set stores its entries in (height, width, channel) order while a
# property indexes the network's (channel, height, width) output, so the oracle
# permutes the columns of ``G`` into the stored order before it reads the rows.
# verify_specification never answers SAT, so a set that meets the region must
# be UNKNOWN; a set that misses it by a wide margin is UNSAT.
def _stored_order(s):
    """The permutation ``p`` with stored entry ``i`` = network entry ``p[i]``,
    or None for a flat set."""
    if hasattr(s, "num_channels"):
        return hwc_index(s.num_channels, s.height, s.width)
    return None


def _meeting_rows(s, rows_G, rows_g):
    """Predicate rows of ``s`` and the rows ``G (c + V a) <= g`` in fractions,
    for the half-space rows ``(G, g)`` written over the network's order."""
    c, V, C, d, lb, ub = rational.star(s)
    p = _stored_order(s)
    G = [rational.vector(row) for row in rows_G]
    if p is not None:
        G = [[row[k] for k in p] for row in G]
    n = len(lb)
    A, b = [list(row) for row in C], list(d)
    for j in range(n):
        unit = [Fraction(int(k == j)) for k in range(n)]
        if ub[j] is not None:
            A.append(unit)
            b.append(ub[j])
        if lb[j] is not None:
            A.append([-v for v in unit])
            b.append(-lb[j])
    for row, gi in zip(G, rational.vector(rows_g)):
        A.append([rational.dot(row, [V[i][j] for i in range(len(c))])
                  for j in range(n)])
        b.append(gi - rational.dot(row, c))
    return A, b


def _meets(sets, rows_G, rows_g):
    """Does some set have a point with every row ``G y <= g``? Exact."""
    return any(rational.feasible((), (), *_meeting_rows(s, rows_G, rows_g))
               is not None for s in sets)


def _box_least(lo, hi, e):
    """The exact least value of ``e . y`` over the box [lo, hi]."""
    return sum((min(Fraction(float(a)) * Fraction(float(x)),
                    Fraction(float(a)) * Fraction(float(y)))
                for a, x, y in zip(np.ravel(e), np.ravel(lo), np.ravel(hi))),
               Fraction(0))


# A flat box of three outputs with intervals on both sides of zero and of
# widths 2, 1 and 3/2, and an image of (C, H, W) = (2, 1, 3) with its own
# interval per pixel; all dyadic, so every set type below holds exactly the box.
_LO, _HI = np.array([-0.75, 0.5, -2.0]), np.array([1.25, 1.5, -0.5])
_E = np.array([0.5, -0.75, 1.0])
_LO_IMAGE = np.array([[[-0.5, 0.25, 1.0]], [[2.0, -1.5, 0.5]]])
_HI_IMAGE = _LO_IMAGE + np.array([[[0.5, 1.0, 0.25]], [[0.75, 0.5, 1.25]]])
# Network order (channel, height, width): channel 0 at pixel (0, 1), channel 1
# at pixels (0, 0) and (0, 2). Read in the stored order these columns land on
# other pixels whose least value is 3.25 instead of -2.25.
_E_IMAGE = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.5])

# Each entry: the sets, the boxes whose union they hold exactly, and e.
_GIVEN_SETS = {
    "Box": lambda: ([flat_set("Box", _LO, _HI)], [(_LO, _HI)], _E),
    "Star": lambda: ([flat_set("Star", _LO, _HI)], [(_LO, _HI)], _E),
    "Star-with-a-binding-row": lambda: (
        [constrained_star(_LO, _HI, n=8)[0]], [(_LO, _HI)], _E),
    "Zono": lambda: ([flat_set("Zono", _LO, _HI)], [(_LO, _HI)], _E),
    "Hexatope": lambda: ([flat_set("Hexatope", _LO, _HI)], [(_LO, _HI)], _E),
    "Octatope": lambda: ([flat_set("Octatope", _LO, _HI)], [(_LO, _HI)], _E),
    "ImageStar": lambda: ([image_set("ImageStar", _LO_IMAGE, _HI_IMAGE)],
                          [(_LO_IMAGE, _HI_IMAGE)], _E_IMAGE),
    "ImageZono": lambda: ([image_set("ImageZono", _LO_IMAGE, _HI_IMAGE)],
                          [(_LO_IMAGE, _HI_IMAGE)], _E_IMAGE),
    # Three stars that cover the box, cut along y0 into 1/8, 2/8 and 5/8 of
    # it; with e0 < 0 the least value is at y0 = 5/4, in the third.
    "three-Stars": lambda: (split_cover("Star", _LO, _HI), [(_LO, _HI)],
                            _E * np.array([-1.0, 1.0, 1.0])),
    # A Box far away first, then the Star that reaches the region.
    "Box-then-Star": lambda: ([flat_set("Box", _LO + 4, _HI + 4),
                               flat_set("Star", _LO, _HI)],
                              [(_LO + 4, _HI + 4), (_LO, _HI)], _E),
}

# The threshold relative to the least value of e . y over the sets, and the
# verdict a correct check gives.
_REGIMES = {
    "disjoint": (Fraction(-1, 4), "UNSAT"),
    "touching": (Fraction(0), "UNKNOWN"),        # one face of the set
    "inside": (Fraction(1, 2), "UNKNOWN"),
}


@pytest.mark.parametrize("regime", sorted(_REGIMES))
@pytest.mark.parametrize("kind", sorted(_GIVEN_SETS))
def test_the_verdict_on_a_given_set_is_its_exact_answer(kind, regime):
    """``verify_specification(sets, e . y <= g)`` with ``g`` a quarter below,
    at, and half above the least value of ``e . y`` over the sets, for every
    set type the function accepts, a set with a binding row that is not a box
    row, several sets, and a list that mixes types. The exact answer is read
    from the sets' own rows in fractions. A set that touches the closed region
    in one face meets it. Catches SAT and UNSAT swapped in the verdict map, a
    disjointness check that rounds a touching set into 'disjoint', an image
    set read in its stored order instead of the network's (the ImageStar and
    ImageZono cases are wrong in the touching regime then), a reach set other
    than the first one left unchecked, and a set type sent down the wrong
    path."""
    sets, boxes, e = _GIVEN_SETS[kind]()
    shift, expected = _REGIMES[regime]
    least = min(_box_least(lo, hi, e) for lo, hi in boxes)
    g = least + shift
    if Fraction(float(g)) != g:
        pytest.fail(f"the threshold {g} is not a float64 number")
    met = _meets(sets, [e], [g])
    if met != (expected == "UNKNOWN"):
        pytest.fail(f"the {kind} sets do not hold the box: exact meets={met}")
    unsafe = HalfSpace(np.array([e]), np.array([[float(g)]]))
    verdict = verify_specification(sets, unsafe).verdict
    assert verdict == expected, (
        f"{kind}, e . y <= {float(g)} ({regime}): {verdict}, the exact answer "
        f"is {'meets' if met else 'disjoint'}")


# Half-spaces over the box _LO, _HI (y0 in [-3/4, 5/4], y1 in [1/2, 3/2],
# y2 in [-2, -1/2]).
def _hs(rows_G, rows_g):
    return HalfSpace(np.array(rows_G, dtype=np.float64),
                     np.array(rows_g, dtype=np.float64).reshape(-1, 1))


_MISS_Y0 = lambda: _hs([[-1, 0, 0]], [-2.0])           # y0 >= 2      # noqa: E731
_MISS_Y1 = lambda: _hs([[0, 1, 0]], [0.25])            # y1 <= 1/4    # noqa: E731
_TOUCH_Y2 = lambda: _hs([[0, 0, -1]], [0.5])           # y2 >= -1/2   # noqa: E731
_IN_Y0 = lambda: _hs([[1, 0, 0]], [0.0])               # y0 <= 0      # noqa: E731
_IN_Y1 = lambda: _hs([[0, -1, 0]], [-1.0])             # y1 >= 1      # noqa: E731
# Two rows that each meet the box and together miss it (y0 >= 11/8 > 5/4).
_ROWS_MISS = lambda: _hs([[-1, -1, 0], [-1, 1, 0]], [-2.25, -0.5])  # noqa: E731
# Two rows that together leave one edge of the box: y0 = 5/4, y1 = 1/2.
_ROWS_TOUCH = lambda: _hs([[-1, -1, 0], [-1, 1, 0]], [-1.75, -0.75])  # noqa: E731

_FORMS = {
    "one-row": lambda: _TOUCH_Y2(),
    "one-row-missing": lambda: _MISS_Y0(),
    "two-rows-missing-together": lambda: _ROWS_MISS(),
    "two-rows-touching-together": lambda: _ROWS_TOUCH(),
    "list-reached-second": lambda: [_MISS_Y0(), _TOUCH_Y2()],
    "list-all-missing": lambda: [_MISS_Y0(), _MISS_Y1()],
    "dict": lambda: {"Hg": _TOUCH_Y2()},
    "dict-list-reached-second": lambda: {"Hg": [_MISS_Y0(), _TOUCH_Y2()]},
    "dicts-first-group-missing": lambda: [{"Hg": _MISS_Y1()},
                                          {"Hg": _IN_Y0()}],
    "dicts-last-group-missing": lambda: [{"Hg": _IN_Y0()},
                                         {"Hg": _MISS_Y1()}],
    "dicts-all-meeting": lambda: [{"Hg": _IN_Y0()}, {"Hg": _IN_Y1()}],
    "dicts-list-reached-second": lambda: [{"Hg": [_MISS_Y0(), _TOUCH_Y2()]},
                                          {"Hg": _IN_Y1()}],
    "dicts-list-all-missing": lambda: [{"Hg": [_MISS_Y0(), _MISS_Y1()]},
                                       {"Hg": _IN_Y0()}],
}


def _groups_by_the_docstring(prop):
    """The unsafe region of a property as an AND of groups, each an OR of
    half-spaces, read from the documented shapes (verify_specification's
    Notes), not by n2v's parser: a HalfSpace is one group of one; a list of
    HalfSpaces is one group (OR); a dict is one group, its ``Hg`` a HalfSpace
    or a list (OR); a list of dicts is an AND of those groups."""
    def group(hg):
        return [hg] if isinstance(hg, HalfSpace) else list(hg)
    if isinstance(prop, HalfSpace):
        return [[prop]]
    if isinstance(prop, dict):
        return [group(prop["Hg"])]
    if all(isinstance(p, dict) for p in prop):
        return [group(p["Hg"]) for p in prop]
    return [list(prop)]


@pytest.mark.parametrize("form", sorted(_FORMS))
@pytest.mark.parametrize("kind", ["Box", "Star"])
def test_every_form_of_a_property_is_read_as_its_unsafe_region(kind, form):
    """Every accepted shape of a property (one half-space with one or two
    rows, a list, a dict, a list of dicts, OR inside AND), on the Box path and
    the Star path. The exact answer: the region is reached when one choice of
    a half-space per AND group has all its rows met by a point of the set
    (fractions, on the set's rows); it is UNSAT when some group misses the
    set by itself. Catches an OR read as an AND in any of the three shapes
    that hold a list, only the first disjunct of a group checked, an AND
    whose earlier group is disjoint answered UNKNOWN (only the last group
    decides), the rows of one half-space checked one at a time but never
    together (no LP, or an LP without the box's bounds), and SAT and UNSAT
    swapped."""
    s = flat_set(kind, _LO, _HI)
    prop = _FORMS[form]()
    groups = _groups_by_the_docstring(prop)

    def rows_of(choice):
        G = np.vstack([hs.G for hs in choice])
        return G, np.vstack([hs.g for hs in choice]).reshape(-1)

    reached = any(_meets([s], *rows_of(choice)) for choice in product(*groups))
    a_group_misses = any(not any(_meets([s], *rows_of([hs])) for hs in group)
                         for group in groups)
    if reached == a_group_misses:
        pytest.fail(f"{form}: reached={reached}, a group misses="
                    f"{a_group_misses}; the case decides nothing")
    expected = "UNKNOWN" if reached else "UNSAT"
    verdict = verify_specification([s], prop).verdict
    assert verdict == expected, (
        f"{form} on a {kind}: {verdict}, the exact answer is "
        f"{'reached' if reached else 'a group misses the set'}")


def test_a_probabilistic_box_that_meets_the_region_is_never_unsat():
    """A ``ProbabilisticBox`` is a ``Box`` and goes down the Box path. Where
    it touches or enters the unsafe region, nothing is proved, so the verdict
    is UNKNOWN, as for the plain Box. (Whether a disjoint probabilistic box
    may say UNSAT without naming its guarantee is registry PROB-5, marked in
    ``tests/contracts/test_probabilistic_contract.py``.) Catches a dispatch
    that answers a probabilistic box as disjoint without checking it."""
    pbox = ProbabilisticBox(_LO, _HI, m=100, ell=99, epsilon=0.05)
    least = _box_least(_LO, _HI, _E)
    for shift in (Fraction(0), Fraction(1, 2)):
        g = float(least + shift)
        if not _meets([Box(_LO, _HI)], [_E], [g]):
            pytest.fail("the box does not meet the region")
        unsafe = HalfSpace(np.array([_E]), np.array([[g]]))
        assert verify_specification([pbox], unsafe).verdict == "UNKNOWN"


# ----------------------------- the Box path ------------------------------- #
def test_a_box_with_an_unbounded_side_is_not_disjoint_from_a_region_on_it():
    """y0 in [-inf, 1], y1 in [0, inf]; the unsafe region y0 <= -1 is reached
    (the member (-1, 0) is in it). The row's zero coefficient on the unbounded
    y1 makes the fast check's sum NaN (0 times inf), which it repairs by
    zeroing that one term; the nonzero coefficient on the unbounded side of y0
    must keep its minus infinity. Also y0 >= 2, which the box misses, is UNSAT.
    Catches a repair that zeroes every non-finite term (a false UNSAT)."""
    box = Box(np.array([-np.inf, 0.0]), np.array([1.0, np.inf]))
    member = (Fraction(-1), Fraction(0))
    if not (member[0] <= 1 and member[1] >= 0 and member[0] <= -1):
        pytest.fail("the witness is not a member in the region")
    reached = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-1.0]]))
    assert verify_specification([box], reached).verdict == "UNKNOWN"
    missed = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-2.0]]))
    assert verify_specification([box], missed).verdict == "UNSAT"


def test_a_wide_box_is_checked_row_by_row_against_its_own_thresholds():
    """A Box of 2049 outputs, so the fast check takes its rows three at a
    time, and four rows on four outputs: y0, y1, y2 <= 1/2 on [0, 1] and y3 <=
    11/2 on [5, 6]. The point with y0 = y1 = y2 = 0 and y3 = 5 (the rest
    anywhere in the box) meets all four, so the verdict is UNKNOWN; with y3 <=
    9/2 instead the fourth row misses the box and it is UNSAT. Catches a
    later block of rows compared with the thresholds of an earlier one (a
    false UNSAT)."""
    n = 2049
    lb, ub = np.zeros(n), np.ones(n)
    lb[3], ub[3] = 5.0, 6.0
    G = np.zeros((4, n))
    G[np.arange(4), np.arange(4)] = 1.0
    g = np.array([0.5, 0.5, 0.5, 5.5])
    member = [Fraction(0), Fraction(0), Fraction(0), Fraction(5)]
    if not all(Fraction(lb[i]) <= member[i] <= Fraction(ub[i])
               and member[i] <= Fraction(g[i]) for i in range(4)):
        pytest.fail("the witness is not a member in the region")
    box = Box(lb, ub)
    assert verify_specification([box], HalfSpace(G, g)).verdict == "UNKNOWN"
    g[3] = 4.5
    assert verify_specification([box], HalfSpace(G, g)).verdict == "UNSAT"


_SPEC_WIDTH_NOT_CHECKED_ON_A_BOX = known_bug(
    6, "the Box path of verify_specification broadcasts a property whose "
       "width differs from the box's (one column over two outputs, or "
       "several columns over one) instead of refusing it, and can answer "
       "UNSAT: a "
       "one-column 'y0 <= 1/2' over y0 in [0, 1], y1 in [5, 6] is answered "
       "UNSAT although y = (0, 5) is in it", raises=DID_NOT_RAISE)


@pytest.mark.parametrize("kind,outputs,width", [
    pytest.param("Box", 2, 1, marks=_SPEC_WIDTH_NOT_CHECKED_ON_A_BOX,
                 id="Box-of-2-property-of-1"),
    pytest.param("Box", 2, 3, id="Box-of-2-property-of-3"),
    pytest.param("Star", 2, 1, id="Star-of-2-property-of-1"),
])
def test_a_property_whose_width_is_not_the_sets_is_refused(kind, outputs,
                                                            width):
    """A property over another number of outputs than the reach set has
    cannot be read against it (DESIGN: a shape that does not fit raises
    ``ValueError``). The Star path refuses it, and so does the Box path when
    numpy cannot broadcast the two widths; where it can (one of them is 1),
    the Box path answers."""
    s = flat_set(kind, np.array([0.0, 5.0])[:outputs],
                 np.array([1.0, 6.0])[:outputs])
    G = np.zeros((1, width))
    G[0, 0] = 1.0
    with pytest.raises(ValueError):
        verify_specification([s], HalfSpace(G, np.array([[0.5]])))


# ----------------------------- the Star path ------------------------------ #
def test_an_empty_star_is_unsat_and_a_star_of_one_point_is_not():
    """Stars over [0, 1]^2 (from_bounds) with one more row on the predicate
    variables: a0 + a1 <= -3 leaves no point (UNSAT for any region, here one
    that holds every output); a0 + a1 <= -2 leaves the one corner (0, 0),
    which touches y0 <= 0, so that is UNKNOWN. And a star with no predicate
    variable, the point (1/4, -3/4): y0 <= 1/4 holds it (UNKNOWN), y0 <= 0
    misses it (UNSAT). The exact answer of each from the stored rows.
    Catches an emptiness test that is skipped or inverted, and a single
    member point lost by the check."""
    cases = []
    for rhs, region, expected in ((-3.0, ([[1.0, 0.0]], [100.0]), "UNSAT"),
                                  (-2.0, ([[1.0, 0.0]], [0.0]), "UNKNOWN")):
        base = Star.from_bounds(np.zeros((2, 1)), np.ones((2, 1)))
        C = np.vstack([base.C, [[1.0, 1.0]]])
        d = np.vstack([base.d, [[rhs]]])
        cases.append((Star(base.V, C, d, base.predicate_lb, base.predicate_ub),
                      region, expected))
    point = np.array([[0.25], [-0.75]])
    cases += [(Star.from_bounds(point, point.copy()), ([[1.0, 0.0]], [0.25]),
               "UNKNOWN"),
              (Star.from_bounds(point, point.copy()), ([[1.0, 0.0]], [0.0]),
               "UNSAT")]
    for star, (G, g), expected in cases:
        if _meets([star], G, g) != (expected == "UNKNOWN"):
            pytest.fail(f"the exact answer of {star!r} is not {expected}")
        verdict = verify_specification([star], _hs(G, g)).verdict
        assert verdict == expected, (f"{verdict} where the exact answer is "
                                     f"{expected}, rows {G} <= {g}")


def _rows_only_family(count=64, seed=0):
    """Stars ``y = V' a`` over one predicate variable ``a`` in [A, 2A] given
    by two constraint rows and no predicate box, with six outputs, and the
    region ``h . y <= g``, ``h`` a row of size 2^30 whose product with ``V'``
    nearly cancels (to between 1e-7 and 1e-11 of its terms), so that float64
    computes ``h V'`` with a large relative error. ``g`` is the smallest
    float64 at or above the exact ``(h V') A``: the member ``a = A`` is in the
    region, exactly. Yields ``(star, region, A)``."""
    rng = np.random.default_rng(seed)
    A = 1.0e6
    for _ in range(count):
        v = rng.uniform(0.5, 2.0, 6)
        h = rng.uniform(-1.0, 1.0, 6)
        h[-1] = (-float(np.dot(h[:-1], v[:-1])) / v[-1]
                 * (1 - 10.0 ** -float(rng.uniform(7, 11))))
        h = h * 2.0 ** 30
        hv = sum((Fraction(x) * Fraction(y) for x, y in zip(h, v)),
                 Fraction(0))
        if hv < 0:
            h, hv = -h, -hv
        g = _up(hv * Fraction(A))
        V = np.hstack([np.zeros((6, 1)), v.reshape(-1, 1)])
        star = Star(V, np.array([[1.0], [-1.0]]), np.array([[2 * A], [-A]]))
        yield star, HalfSpace(h.reshape(1, -1), np.array([[g]])), A


@known_bug(17, "on a Star with no finite predicate box (constraint rows "
               "only) the spec row pulled back to the predicate variables, h "
               "V', is rounded to nearest and moved outward only by the "
               "rounding of h c and g, not of h V' times the variables' "
               "magnitude; the emptiness LP then certifies a star that meets "
               "the unsafe region as disjoint: UNSAT although a member is in "
               "it (9 of these 64 cases today)")
def test_a_star_given_by_rows_only_is_not_unsat_when_a_member_is_unsafe():
    """64 stars of the family of ``_rows_only_family``, each with an exact
    member in the unsafe region (checked in fractions on the stored rows),
    none of which may be UNSAT."""
    wrong = []
    for k, (star, region, A) in enumerate(_rows_only_family()):
        if star.predicate_lb is not None:
            pytest.fail("the star has a predicate box")
        if not _meets([star], region.G, region.g.reshape(-1)):
            pytest.fail(f"case {k}: no member is in the region")
        if verify_specification([star], region).verdict == "UNSAT":
            wrong.append(k)
    assert not wrong, f"UNSAT with a member in the region: cases {wrong}"


def test_the_same_stars_with_a_finite_predicate_box_are_not_unsat():
    """The passing neighbour of the marked test above: the same 64 stars with
    the predicate box [A, 2A] stated, where the pre-check and the pulled-back
    rows carry the box's magnitude. Catches a regression of the outward
    rounding of the spec rows on the path that has a box."""
    wrong = []
    for k, (star, region, A) in enumerate(_rows_only_family()):
        boxed = Star(star.V, star.C, star.d, np.array([[A]]),
                     np.array([[2 * A]]))
        if not _meets([boxed], region.G, region.g.reshape(-1)):
            pytest.fail(f"case {k}: no member is in the region")
        if verify_specification([boxed], region).verdict == "UNSAT":
            wrong.append(k)
    assert not wrong, f"UNSAT with a member in the region: cases {wrong}"

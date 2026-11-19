"""Soundness of relational (two-network) verification, ``n2v/nn/relational.py``.

A relational specification couples two networks f and g over one joint input
``[x_f; x_g]``: each input lies in its own box (a side may be open), the pair
satisfies the coupling rows ``G x <= h`` (an equality is a pair of rows), and
the property is an unsafe region over the joint output ``[f(x_f); g(x_g)]``:
AND across groups, OR over the half-spaces of a group, every row of a
half-space. The promises checked here, each against an independent oracle:

- ``build_joint_input_star`` holds exactly the allowed pairs: every vertex of
  the allowed polytope is a member, and every vertex of the star is allowed,
  in rationals with no slack;
- ``relational_reach`` holds ``[f(x_f); g(x_g)]`` for every allowed pair:
  every vertex of the exact joint image (rationals, no slack) and every
  replayed allowed pair (at least 128 per instance, LP membership on the
  set's own rows) is a member; for both shared-prefix routes (torch
  Linear/ReLU and ONNX Gemm keep the input predicates, ONNX MatMul/Add and
  Dropout do not), equality and inequality couplings, boxes open on one or
  both sides, no coupling, a single allowed pair, approx and exact;
- ``verify_relational`` answers UNSAT only when no allowed pair violates, and
  ``solve_relational`` maps an undecided reach to 'unknown';
- ``falsify_relational`` and ``solve_relational`` return only
  counterexamples: inside both boxes, the coupling held in exact arithmetic,
  the outputs replayed on the model (onnxruntime for a file, torch in float64
  otherwise) inside the unsafe region, by the margin for the falsifier;
- ``seed``, ``n_rand``, ``margin`` and ``method`` are honoured, and inputs the
  engine does not support are refused.

Models have dyadic weights and replay points lie on a dyadic grid, so float32,
float64 and rational evaluation coincide.
"""

from fractions import Fraction

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
from onnx import TensorProto, helper

from n2v.nn.relational import (build_joint_input_star, falsify_relational,
                               relational_reach, solve_relational,
                               verify_relational)
from n2v.sets.halfspace import HalfSpace
from n2v.utils.model_loader import load_onnx
from tests._oracles import rational
from tests._oracles.membership import uncovered
from tests.cell_support import replay_onnx, replay_torch, save_onnx
from tests.known_bugs import known_bug

INF = np.inf


# --------------------------------------------------------------------------- #
# Models                                                                        #
# --------------------------------------------------------------------------- #
def _mlp(tmp_path, name, seed, n_in=2, n_hidden=4, n_out=2):
    """Tiny Gemm-Relu-Gemm ONNX model with random float32 weights; returns
    the loaded fx model."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((n_hidden, n_in)).astype(np.float32)
    b1 = rng.standard_normal(n_hidden).astype(np.float32)
    W2 = rng.standard_normal((n_out, n_hidden)).astype(np.float32)
    b2 = rng.standard_normal(n_out).astype(np.float32)

    def _t(nm, a):
        return helper.make_tensor(nm, TensorProto.FLOAT, list(a.shape),
                                  a.flatten().tolist())
    nodes = [
        helper.make_node("Gemm", ["x", "W1", "b1"], ["h"], transB=1),
        helper.make_node("Relu", ["h"], ["r"]),
        helper.make_node("Gemm", ["r", "W2", "b2"], ["y"], transB=1),
    ]
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, n_in])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, n_out])],
        initializer=[_t("W1", W1), _t("b1", b1), _t("W2", W2), _t("b2", b2)])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    path = str(tmp_path / f"{name}.onnx")
    onnx.save(model, path)
    return load_onnx(path), path


def _equal_input_spec(prop_G, prop_g, n=2):
    """Spec over [X_f(n); X_g(n)] with X_f boxed in [-1,1], coupling
    X_f == X_g, and a single output HalfSpace over [Y_f(n); Y_g(n)]."""
    lb = np.concatenate([-np.ones(n), np.full(n, -np.inf)])
    ub = np.concatenate([np.ones(n), np.full(n, np.inf)])
    rows = []
    for i in range(n):                       # X_f[i] == X_g[i]
        r = np.zeros(2 * n); r[i] = 1.0; r[n + i] = -1.0
        rows.append(r.copy())
        rows.append(-r)
    coupling = HalfSpace(np.array(rows), np.zeros(len(rows)))
    return {
        "format": "relational",
        "networks": [
            {"name": "f", "relation": None, "input_offset": 0,
             "input_size": n, "output_offset": 0, "output_size": n},
            {"name": "g", "relation": ("equal-to", "f"), "input_offset": n,
             "input_size": n, "output_offset": n, "output_size": n},
        ],
        "lb": lb, "ub": ub,
        "input_coupling": coupling,
        "prop": [{"Hg": HalfSpace(np.asarray(prop_G, dtype=float),
                                  np.asarray(prop_g, dtype=float))}],
    }


class _Net:
    """``y = W2 relu(W1 x + b1) + b2`` with dyadic weights (exact in float32),
    built for n2v by one route: ``torch`` (Linear, ReLU, Linear), ``dropout``
    (a Dropout after the first Linear: not a module that keeps the input
    predicates), ``gemm`` (ONNX Gemm, Relu, Gemm: loaded as Linear) or
    ``matmul`` (ONNX MatMul, Add, Relu, MatMul, Add: not Linear)."""

    def __init__(self, W1, b1, W2, b2, route="torch"):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        self.route = route
        self.n_in, self.n_out = len(W1[0]), len(W2)

    def build(self, tmp_path, name):
        """The model n2v receives and its replay: flat inputs to flat
        float64 outputs, by onnxruntime for a file, torch float64 otherwise."""
        W1, b1, W2, b2 = (np.asarray(a, dtype=np.float32)
                          for a in (self.W1, self.b1, self.W2, self.b2))
        shape = [1, self.n_in]
        if self.route in ("gemm", "matmul"):
            if self.route == "gemm":
                nodes = [helper.make_node("Gemm", ["X", "W1", "b1"], ["H"],
                                          transB=1),
                         helper.make_node("Relu", ["H"], ["R"]),
                         helper.make_node("Gemm", ["R", "W2", "b2"], ["Y"],
                                          transB=1)]
                inits = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
            else:
                nodes = [helper.make_node("MatMul", ["X", "W1t"], ["P"]),
                         helper.make_node("Add", ["P", "b1"], ["H"]),
                         helper.make_node("Relu", ["H"], ["R"]),
                         helper.make_node("MatMul", ["R", "W2t"], ["Q"]),
                         helper.make_node("Add", ["Q", "b2"], ["Y"])]
                inits = {"W1t": W1.T.copy(), "b1": b1, "W2t": W2.T.copy(),
                         "b2": b2}
            path = save_onnx(tmp_path, nodes, shape, [1, self.n_out],
                             inits=inits, name=name)
            return load_onnx(path), lambda xs: replay_onnx(path, xs, shape)
        first = nn.Linear(self.n_in, W1.shape[0])
        last = nn.Linear(W1.shape[0], self.n_out)
        with torch.no_grad():
            for layer, W, b in ((first, W1, b1), (last, W2, b2)):
                layer.weight[:] = torch.from_numpy(W)
                layer.bias[:] = torch.from_numpy(b)
        middle = [nn.Dropout(0.25)] if self.route == "dropout" else []
        model = nn.Sequential(first, *middle, nn.ReLU(), last).eval()
        return model, lambda xs: replay_torch(model, xs, shape)


def _affine(W, b):
    """A torch model ``y = W x + b`` (one Linear layer, float32)."""
    W = np.asarray(W, dtype=np.float32)
    layer = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        layer.weight[:] = torch.from_numpy(W)
        layer.bias[:] = torch.tensor(b, dtype=torch.float32)
    return nn.Sequential(layer).eval()


def _torch_replay(model, n_in):
    return lambda xs: replay_torch(model, xs, (1, n_in))


# --------------------------------------------------------------------------- #
# Specifications                                                                #
# --------------------------------------------------------------------------- #
def _spec(sizes, lb, ub, coupling, prop):
    """A relational spec in the form the VNN-LIB 2.0 loader returns.

    ``sizes`` is ``(nf, ng, of, og)``; ``coupling`` is ``(rows, rhs)`` of
    ``rows @ x <= rhs`` or None; ``prop`` is a list of groups, each a
    ``(G, g)`` pair or a list of such pairs (OR)."""
    nf, ng, of, og = sizes

    def hs(G, g):
        return HalfSpace(np.asarray(G, dtype=float),
                         np.asarray(g, dtype=float).reshape(-1, 1))
    return {
        "format": "relational",
        "networks": [
            {"name": "f", "relation": None, "input_offset": 0,
             "input_size": nf, "output_offset": 0, "output_size": of},
            {"name": "g", "relation": None, "input_offset": nf,
             "input_size": ng, "output_offset": of, "output_size": og}],
        "lb": np.array(lb, dtype=float), "ub": np.array(ub, dtype=float),
        "input_coupling": (None if coupling is None
                           else hs(*coupling)),
        "prop": [{"Hg": ([hs(*p) for p in grp] if isinstance(grp, list)
                         else hs(*grp))} for grp in prop],
    }


def _equal_rows(nf, pairs):
    """Rows of ``x_g[j] == x_f[i]`` for each ``(i, j)`` of ``pairs``."""
    rows = []
    for i, j in pairs:
        r = [0.0] * (nf + max(j for _, j in pairs) + 1)
        r[i], r[nf + j] = 1.0, -1.0
        rows += [r, [-v for v in r]]
    return rows, [0.0] * len(rows)


def _allowed_rows(spec):
    """The allowed set read from the spec arrays: finite box rows and
    coupling rows, as Fractions (``A x <= b``)."""
    lb, ub = spec["lb"], spec["ub"]
    n = lb.size
    A, b = [], []
    for i in range(n):
        unit = [Fraction(int(k == i)) for k in range(n)]
        if np.isfinite(ub[i]):
            A.append(unit); b.append(Fraction(float(ub[i])))
        if np.isfinite(lb[i]):
            A.append([-v for v in unit]); b.append(-Fraction(float(lb[i])))
    coupling = spec["input_coupling"]
    if coupling is not None:
        for row, rhs in zip(coupling.G, np.asarray(coupling.g).ravel()):
            A.append([Fraction(float(v)) for v in row])
            b.append(Fraction(float(rhs)))
    return A, b


def _is_allowed(spec, x):
    """Is the rational point ``x`` in both boxes and does it keep every
    coupling row, exactly?"""
    A, b = _allowed_rows(spec)
    x = [Fraction(v) for v in x]
    return all(sum(a * v for a, v in zip(row, x)) <= r for row, r in zip(A, b))


def _round_up(value):
    """The least float64 at or above the rational ``value``."""
    v = float(value)
    return v if Fraction(v) >= value else float(np.nextafter(v, np.inf))


# --------------------------------------------------------------------------- #
# Containment instances: two networks over one coupled input                   #
# --------------------------------------------------------------------------- #
# Each instance gives the networks, the spec's box and coupling, and the
# allowed set again as a polytope over free parameters t with x = E t + e:
# the independent description the exact image and the replay points use.
_F2 = ([[1, -0.5], [0.75, 1.25]], [0.25, -0.5], [[1, -1.5], [0.5, 2]],
       [0.125, -0.25])
_G2 = ([[-1, 0.5], [1.5, -0.25]], [0.5, 0.25], [[1.25, -0.75]], [-0.5])
_F3 = ([[1, 1], [-0.5, 1], [1, -2]], [0.5, -0.25, 0], [[1, -1, 0.5]], [0.25])
_G1 = ([[2], [-1]], [-1, 0.5], [[1, 0.5], [-0.75, 1]], [0, 0.25])
_FI = ([[1], [-1]], [0.5, 0.25], [[1, -0.5]], [0.25])
_GI = ([[2], [-0.5]], [-1, 0.5], [[-0.75, 1.5]], [0])
_FD = ([[1], [-2]], [0, 0.5], [[1.5, 1]], [-0.25])
_GS = ([[-1], [0.5]], [0.25, 1.5], [[1, -1]], [0.5])          # stable on C
_FS = ([[1, 0.5], [-1, 0.25]], [0.5, -1], [[2, 1]], [-0.5])   # stable on A
_GS2 = ([[0.5, -1], [1, 1]], [0.25, 1], [[1, -0.5], [-1.5, 1]], [0.25, 0])


def _box_rows(lo, hi):
    n = len(lo)
    A, b = [], []
    for i in range(n):
        unit = [int(k == i) for k in range(n)]
        A += [unit, [-v for v in unit]]
        b += [hi[i], -lo[i]]
    return A, b


def _instance(name):
    """``(f, g, spec_parts, method, (A_t, b_t, E, e, t_lo, t_hi))``."""
    if name == "onnx-gemm-equal":
        # x_g == x_f, x_g's box open on both sides (tightened by LP); both
        # networks keep the input predicates; equal own predicate counts.
        lo, hi = [-1.5, -0.75], [2, 0.5]
        coupling = _equal_rows(2, [(0, 0), (1, 1)])
        return (_Net(*_F2, route="gemm"), _Net(*_G2, route="gemm"),
                ((2, 2, 2, 1), lo + [-INF, -INF], hi + [INF, INF], coupling),
                "approx",
                (*_box_rows(lo, hi), [[1, 0], [0, 1], [1, 0], [0, 1]],
                 [0, 0, 0, 0], lo, hi))
    if name == "onnx-matmul-mixed":
        # g has one input, coupled to f's second; f (MatMul) does not keep
        # the input predicates, g (Gemm) does; own counts differ.
        lo, hi = [-2, -1.25], [1, 2.5]
        coupling = _equal_rows(2, [(1, 0)])
        return (_Net(*_F3, route="matmul"), _Net(*_G1, route="gemm"),
                ((2, 1, 1, 2), lo + [-INF], hi + [INF], coupling), "approx",
                (*_box_rows(lo, hi), [[1, 0], [0, 1], [0, 1]], [0, 0, 0],
                 lo, hi))
    if name == "dropout-no-coupling":
        # No coupling; neither network keeps the input predicates; g's
        # ReLUs are stable on its box, so its output star has no rows.
        lo, hi = [-1.25, -2], [2, -0.5]
        return (_Net(*_FD, route="dropout"), _Net(*_GS, route="dropout"),
                ((1, 1, 1, 1), lo, hi, None), "approx",
                (*_box_rows(lo, hi), [[1, 0], [0, 1]], [0, 0], lo, hi))
    if name == "exact-stable":
        # method='exact' where no ReLU splits (one star per network).
        lo, hi = [0.5, -1], [1.5, 0.25]
        coupling = _equal_rows(2, [(0, 0), (1, 1)])
        return (_Net(*_FS), _Net(*_GS2),
                ((2, 2, 1, 2), lo + [-INF, -INF], hi + [INF, INF], coupling),
                "exact",
                (*_box_rows(lo, hi), [[1, 0], [0, 1], [1, 0], [0, 1]],
                 [0, 0, 0, 0], lo, hi))
    if name == "inequality-offset":
        # x_g = x_f + d with d in [-1.25, 0.75]; x_g's box is open above.
        A, b = _box_rows([-2.5, -3], [1.5, 2.25])
        A += [[-1, 1], [1, -1]]
        b += [0.75, 1.25]
        return (_Net(*_FI), _Net(*_GI),
                ((1, 1, 1, 1), [-2.5, -3], [1.5, INF],
                 ([[-1, 1], [1, -1]], [0.75, 1.25])), "approx",
                (A, b, [[1, 0], [0, 1]], [0, 0], [-2.5, -3], [1.5, 2.25]))
    if name == "one-allowed-pair":
        # A in [0, 1], C in [1, 2], C <= A: only (1, 1) is allowed.
        A, b = _box_rows([0, 1], [1, 2])
        A.append([-1, 1]); b.append(0)
        return (_Net(*_FI), _Net(*_GI),
                ((1, 1, 1, 1), [0, 1], [1, 2], ([[-1, 1]], [0])), "approx",
                (A, b, [[1, 0], [0, 1]], [0, 0], [0, 1], [1, 2]))
    raise KeyError(name)


class _Case:
    """One built instance: models, replays, spec and the oracle's view."""

    def __init__(self, name, tmp_path):
        f, g, (sizes, lb, ub, coupling), method, param = _instance(name)
        self.nets = (f, g)
        self.f, self.f_replay = f.build(tmp_path, f"{name}-f")
        self.g, self.g_replay = g.build(tmp_path, f"{name}-g")
        self.sizes, self.lb, self.ub, self.coupling = sizes, lb, ub, coupling
        self.method, self.param = method, param

    def spec(self, prop=None):
        """The spec with ``prop`` (default: an unsafe region no output
        reaches, irrelevant for a containment check)."""
        nf, ng, of, og = self.sizes
        if prop is None:
            prop = [([[1.0] + [0.0] * (of + og - 1)], [-1e9])]
        return _spec(self.sizes, self.lb, self.ub, self.coupling, prop)

    def outputs(self, xs):
        """``[f(x_f); g(x_g)]`` replayed on the models."""
        nf, ng = self.sizes[:2]
        ys_f = self.f_replay([x[:nf] for x in xs])
        ys_g = self.g_replay([x[nf:nf + ng] for x in xs])
        return [np.concatenate([a, b]) for a, b in zip(ys_f, ys_g)]

    def allowed_points(self, n=160, seed=0):
        """Allowed pairs on a dyadic grid (2^-8) drawn in the parameter box
        and kept when they satisfy the parameter rows, plus the vertices of
        the parameter polytope: every one is checked against the spec text."""
        A_t, b_t, E, e, t_lo, t_hi = self.param
        rng = np.random.default_rng(seed)
        A_np, b_np = np.array(A_t, float), np.array(b_t, float)
        ts = [np.array([float(v) for v in vert])
              for vert in rational.vertices(rational.matrix(A_t),
                                            rational.vector(b_t))]
        for _ in range(40 * n):
            if len(ts) >= n:
                break
            t = np.round(rng.uniform(t_lo, t_hi) * 256) / 256
            if np.all(A_np @ t <= b_np):
                ts.append(t)
        xs = [np.array(E, float) @ t + np.array(e, float) for t in ts]
        spec = self.spec()
        bad = [x for x in xs if not _is_allowed(spec, x)]
        if bad:
            pytest.fail(f"oracle sample {bad[0]} is not an allowed pair")
        return xs

    def exact_image(self):
        """The exact joint image as pieces over t, in rationals: the allowed
        polytope mapped to x, then both networks side by side."""
        A_t, b_t, E, e, _, _ = self.param
        f, g = self.nets
        nf, ng = self.sizes[:2]
        hf, hg = len(f.W1), len(g.W1)
        W1 = ([list(r) + [0] * ng for r in f.W1]
              + [[0] * nf + list(r) for r in g.W1])
        W2 = ([list(r) + [0] * hg for r in f.W2]
              + [[0] * hf + list(r) for r in g.W2])
        pieces = rational.affine(rational.polytope(A_t, b_t), E, e)
        pieces = rational.relu(
            rational.affine(pieces, W1, list(f.b1) + list(g.b1)))
        return rational.affine(pieces, W2, list(f.b2) + list(g.b2))


_CONTAINMENT = ["onnx-gemm-equal", "onnx-matmul-mixed", "dropout-no-coupling",
                "exact-stable", "inequality-offset", "one-allowed-pair"]


def _check_join(case):
    """The joint reach set holds every vertex of the exact joint image
    (rationals, zero slack) and every replayed allowed pair (LP on its own
    rows); the spec it was given is unchanged."""
    spec = case.spec()
    before = (spec["lb"].copy(), spec["ub"].copy(),
              None if spec["input_coupling"] is None
              else (spec["input_coupling"].G.copy(),
                    spec["input_coupling"].g.copy()))
    joint = relational_reach(case.f, case.g, spec, method=case.method)
    nf, ng, of, og = case.sizes
    assert joint.dim == of + og, f"joint star has {joint.dim} entries"
    member = rational.Membership(rational.star(joint))
    escaped = [y for piece in case.exact_image() for y in piece.image()
               if not member(y)]
    assert not escaped, (f"{len(escaped)} vertices of the exact joint image "
                         f"are outside the joint star, e.g. "
                         f"{[float(v) for v in escaped[0]]}")
    xs = case.allowed_points()
    missed = uncovered([joint], case.outputs(xs))
    assert not missed, (f"{len(missed)} of {len(xs)} replayed allowed pairs "
                        f"are outside the joint star, e.g. x = "
                        f"{xs[missed[0][0]]} at distance {missed[0][1]:.3g}")
    after = spec["input_coupling"]
    assert np.array_equal(spec["lb"], before[0]) and np.array_equal(
        spec["ub"], before[1]), "the reach changed the spec's box"
    if after is not None:
        assert np.array_equal(after.G, before[2][0]) and np.array_equal(
            after.g, before[2][1]), "the reach changed the spec's coupling"


class TestRelationalSoundness:
    @pytest.mark.parametrize("name", ["onnx-gemm-equal", "onnx-matmul-mixed",
                                      "dropout-no-coupling", "exact-stable"])
    def test_join_encloses_true_joint_output(self, tmp_path, name):
        """Joint reach under equality coupling or none. Catches g reached
        with model_f, a wrong row or bound block in the join (V, C/d or
        predicate bounds swapped between f and g, own upper bounds read from
        the lower ones, input columns of a shared network's rows dropped),
        a non-shared network's first predicates dropped, a predicate box of
        an open coupled dim tightened wrongly, a sub-star with wrong bounds,
        and a method other than the one asked for."""
        _check_join(_Case(name, tmp_path))

    @pytest.mark.parametrize("name", ["inequality-offset", "one-allowed-pair"])
    def test_inequality_coupling_sound(self, tmp_path, name):
        """Joint reach under inequality couplings, a box open on one side
        and a coupling that leaves one allowed pair. Catches the coupling
        stored with the wrong sign, g reached on f's input slice, a one-sided
        box bound lost in the tightening LP, and the join errors above."""
        _check_join(_Case(name, tmp_path))

    @pytest.mark.parametrize("name", [
        "onnx-gemm-equal", "inequality-offset", "one-allowed-pair",
        pytest.param("non-dyadic-equality", marks=known_bug(
            6, "the coupled dims' predicate bounds are LP optima stored in "
               "round to nearest with no outward pad: for 3 C == A with "
               "A in [-1, 1] the bounds of C are fl(-1/3) > -1/3 and "
               "fl(1/3) < 1/3, so the joint input star leaves out the "
               "allowed pairs (-1, -1/3) and (1, 1/3)"))])
    def test_joint_input_star_holds_exactly_the_allowed_pairs(self, name):
        """``build_joint_input_star`` is the allowed set exactly: every vertex
        of the allowed polytope (box and coupling rows of the spec, in
        rationals) is a member, and every vertex of the star is allowed.
        Catches the coupling dropped (the star is the whole box), stored
        with the wrong sign, and a tightened bound that cuts off allowed
        pairs or is rounded inward."""
        if name == "non-dyadic-equality":
            spec = _spec((1, 1, 1, 1), [-1, -INF], [1, INF],
                         ([[-1, 3], [1, -3]], [0, 0]), [([[1, 0]], [-9])])
        else:
            _, _, (sizes, lb, ub, coupling), _, _ = _instance(name)
            spec = _spec(sizes, lb, ub, coupling, [([[1] + [0] * (
                sizes[2] + sizes[3] - 1)], [-9])])
        S = build_joint_input_star(spec)
        A, b = _allowed_rows(spec)
        computed = rational.star(S)
        member = rational.Membership(computed)
        missed = [v for v in rational.vertices(A, b) if not member(v)]
        assert not missed, (f"allowed pairs outside the joint input star: "
                            f"{[[str(c) for c in v] for v in missed]}")
        extra = [v for v in rational.set_vertices(computed)
                 if not _is_allowed(spec, v)]
        assert not extra, (f"joint input star vertices that are not allowed: "
                           f"{[[float(c) for c in v] for v in extra]}")


# --------------------------------------------------------------------------- #
# Witnesses                                                                     #
# --------------------------------------------------------------------------- #
def _witness_defects(spec, replay_f, replay_g, x, margin=0.0):
    """What makes ``x`` not a counterexample of the spec: outside a box, a
    coupling row broken in exact arithmetic, or outputs replayed on the
    models outside the unsafe region by ``margin`` (every group, any
    half-space of a group, every row of a half-space)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    nf = spec["networks"][0]["input_size"]
    ng = spec["networks"][1]["input_size"]
    defects = []
    if np.any(x < spec["lb"]) or np.any(x > spec["ub"]):
        defects.append(f"outside the box [{spec['lb']}, {spec['ub']}]")
    coupling = spec["input_coupling"]
    if coupling is not None:
        for row, rhs in zip(coupling.G, np.asarray(coupling.g).ravel()):
            lhs = sum(Fraction(float(a)) * Fraction(float(v))
                      for a, v in zip(row, x))
            if lhs > Fraction(float(rhs)):
                defects.append(f"breaks the coupling row {row} <= {rhs} "
                               f"by {float(lhs - Fraction(float(rhs))):.3g}")
    y = np.concatenate([replay_f([x[:nf]])[0], replay_g([x[nf:nf + ng]])[0]])
    for k, group in enumerate(spec["prop"]):
        hss = group["Hg"] if isinstance(group["Hg"], list) else [group["Hg"]]
        if not any(np.all(hs.G @ y <= np.asarray(hs.g).ravel() - margin)
                   for hs in hss):
            defects.append(f"replayed output {y} misses property group {k}")
    return defects


_ID = ([[1]], [0])        # y = x on one input

_RELATIONAL_TEXT = """\
(vnnlib-version <2.0>)
(declare-network f
    (declare-input X_f real [2])
    (declare-output Y_f real [1])
)
(declare-network g
    (equal-to f)
    (declare-input X_g real [2])
    (declare-output Y_g real [1])
)
(assert (and (<= X_f[0] 1.5) (>= X_f[0] -0.5)))
(assert (== X_f[1] 0.25))
(assert (>= X_g[0] -0.5))
(assert (>= X_f[0] X_g[0]))
(assert (== X_f[1] X_g[1]))
(assert (Y_f[0] < Y_g[0]))
"""


def _counterexample_case(name):
    """``(f, g, spec, replay_f, replay_g)`` of a falsifier instance."""
    ident = _affine(*_ID)
    if name == "output-order":
        # f(A) = A, g(C) = C + 1, C == A: B - D = -1 everywhere, so
        # B - D >= 0.5 is unreachable; read as [y_g; y_f] it holds.
        g = _affine([[1]], [1])
        spec = _spec((1, 1, 1, 1), [0, -INF], [1, INF],
                     _equal_rows(1, [(0, 0)]), [([[-1, 1]], [-0.5])])
        return ident, g, spec, _torch_replay(ident, 1), _torch_replay(g, 1)
    if name == "g-own-input":
        # C >= A + 0.25: D - B >= 0.25, so D - B <= 0.125 is unreachable.
        spec = _spec((1, 1, 1, 1), [0, 0], [1, 2], ([[1, -1]], [-0.25]),
                     [([[-1, 1]], [0.125])])
    elif name == "two-groups":
        # C == A: B >= 0.5 and D <= -0.5 each reachable, never together.
        spec = _spec((1, 1, 1, 1), [-1, -INF], [1, INF],
                     _equal_rows(1, [(0, 0)]),
                     [([[-1, 0]], [-0.5]), ([[0, 1]], [-0.5])])
    elif name == "two-rows":
        # One half-space with two rows: B - D >= 0.5 and B >= 0.75; the
        # second alone is reachable, B - D = 0 everywhere.
        spec = _spec((1, 1, 1, 1), [0, -INF], [1, INF],
                     _equal_rows(1, [(0, 0)]),
                     [([[-1, 1], [-1, 0]], [-0.5, -0.75])])
    elif name == "first-g-coordinate":
        # f = g = A0 + A1, C == A on both coordinates, A in [0.5, 1]^2:
        # |B - D| >= 0.25 (an OR group) is unreachable.
        f = _affine([[1, 1]], [0])
        spec = _spec((2, 2, 1, 1), [0.5, 0.5, -INF, -INF],
                     [1, 1, INF, INF], _equal_rows(2, [(0, 0), (1, 1)]),
                     [[([[-1, 1]], [-0.25]), ([[1, -1]], [-0.25])]])
        return f, f, spec, _torch_replay(f, 2), _torch_replay(f, 2)
    elif name == "coupling-over-f-inputs":
        # f = A0 - A1 under A0 <= A1 - 0.5 (a row over f's inputs alone),
        # g(C) = C with C == A0: B <= -0.5, so B >= 0 is unreachable.
        f = _affine([[1, -1]], [0])
        spec = _spec((2, 1, 1, 1), [0, 0, -INF], [1, 1, INF],
                     ([[1, -1, 0], [1, 0, -1], [-1, 0, 1]], [-0.5, 0, 0]),
                     [([[-1, 0]], [0])])
        return f, ident, spec, _torch_replay(f, 2), _torch_replay(ident, 1)
    elif name == "non-dyadic-equality":
        # 3 C == A with A in [0.5, 1]: every pair is unsafe (B >= 0), and
        # (0.75, 0.25) is an exact witness.
        spec = _spec((1, 1, 1, 1), [0.5, -INF], [1, INF],
                     ([[-1, 3], [1, -3]], [0, 0]), [([[-1, 0]], [0])])
    else:
        raise KeyError(name)
    return ident, ident, spec, _torch_replay(ident, 1), _torch_replay(ident, 1)


class TestRelationalVerdicts:
    def test_unsat_unsafe_region_far_from_outputs(self, tmp_path):
        # Unsafe region Y_f[0] <= -1e6 is far below the reachable outputs
        # (O(1) for a tiny net on a [-1,1] box), so the sound marginal
        # reach bound proves disjointness => UNSAT. (A tight relational
        # UNSAT like "Y_f==Y_g" is NOT provable here — the product
        # relaxation loses the cross-network correlation; that is a
        # documented precision limit, not a soundness one.)
        f, _ = _mlp(tmp_path, "f", 7)
        spec = _equal_input_spec([[1, 0, 0, 0]], [-1e6])
        res = verify_relational(f, f, spec, method="approx")
        assert res.verdict == "UNSAT"

    @pytest.mark.parametrize("name", _CONTAINMENT)
    def test_unsat_only_when_no_allowed_pair_violates(self, tmp_path, name):
        """Through the public entries: for four directions c over the first
        outputs B of f and D of g (B - D, D - B, B + D, -B - D), the unsafe
        region ``c . y <= min c . y`` over the exact joint image touches a
        true output, so ``verify_relational`` must not answer UNSAT and
        ``solve_relational`` without samples (``n_rand=0``) must not answer
        'unsat'. Catches every unsound join or input set at the verdict and
        an undecided reach mapped to 'unsat'."""
        case = _Case(name, tmp_path)
        of, og = case.sizes[2:]
        vertices = [y for piece in case.exact_image() for y in piece.image()]
        for sign_b, sign_d in ((1, -1), (-1, 1), (1, 1), (-1, -1)):
            c = [0] * (of + og)
            c[0], c[of] = sign_b, sign_d
            low = min(sum(ci * yi for ci, yi in zip(c, y)) for y in vertices)
            spec = case.spec([([c], [_round_up(low)])])
            verdict = verify_relational(case.f, case.g, spec,
                                        method=case.method).verdict
            assert verdict != "UNSAT", (
                f"UNSAT for c = {c}, c . y <= {float(low)}, which a true "
                f"output reaches")
            if (sign_b, sign_d) == (1, -1):
                answer, _ = solve_relational(case.f, case.g, spec,
                                             method=case.method, n_rand=0)
                assert answer != "unsat", f"'unsat' for c = {c}"

    @pytest.mark.parametrize("name", [
        "equal-mlp", "different-input-sizes", "no-coupling-boxes-differ",
        "or-group"])
    def test_sat_when_unsafe_region_reachable(self, tmp_path, name):
        """``solve_relational`` answers 'sat' on a reachable unsafe region,
        with a witness inside both boxes that keeps the coupling exactly and
        whose replayed outputs are unsafe. Catches g's input size taken from
        f's (a crash when they differ), g's inputs drawn from f's box when
        there is no coupling, and an OR group read as its first half-space."""
        ident = _affine(*_ID)
        if name == "equal-mlp":
            # f == g, X_f == X_g => Y_f == Y_g, so Y_f[0] - Y_g[0] <= 0.001
            # holds everywhere.
            f, path = _mlp(tmp_path, "f", 7)
            g, replay = f, (lambda xs: replay_onnx(path, xs, [1, 2]))
            spec = _equal_input_spec([[1, 0, -1, 0]], [0.001])
            replay_f = replay_g = replay
        elif name == "different-input-sizes":
            # f(A0, A1) = A0 + A1, g(C) = 2 C, C == A1: B - D = A0 - A1.
            f, g = _affine([[1, 1]], [0]), _affine([[2]], [0])
            spec = _spec((2, 1, 1, 1), [-1, -1, -INF], [1, 1, INF],
                         _equal_rows(2, [(1, 0)]), [([[-1, 1]], [-0.5])])
            replay_f, replay_g = _torch_replay(f, 2), _torch_replay(g, 1)
        elif name == "no-coupling-boxes-differ":
            # No coupling, A in [0, 1], C in [2, 3]: D <= 2.5 half the time.
            f = g = ident
            spec = _spec((1, 1, 1, 1), [0, 2], [1, 3], None,
                         [([[0, 1]], [2.5])])
            replay_f = replay_g = _torch_replay(ident, 1)
        else:
            # C == A in [-1, 1]; one group: B <= -2 (never) OR D >= 0.5.
            f = g = ident
            spec = _spec((1, 1, 1, 1), [-1, -INF], [1, INF],
                         _equal_rows(1, [(0, 0)]),
                         [[([[1, 0]], [-2]), ([[0, -1]], [-0.5])]])
            replay_f = replay_g = _torch_replay(ident, 1)
        verdict, cex = solve_relational(f, g, spec, method="approx",
                                        n_rand=100)
        assert verdict == "sat" and cex is not None, verdict
        defects = _witness_defects(spec, replay_f, replay_g, cex[0])
        assert not defects, f"witness {cex[0]}: {'; '.join(defects)}"

    @pytest.mark.parametrize("name", [
        "output-order", "g-own-input", "two-groups", "two-rows",
        "first-g-coordinate",
        pytest.param("coupling-over-f-inputs", marks=known_bug(
            6, "SPEC-13, second code path: the relational sampler draws f's "
               "inputs from their box alone and never applies a coupling row "
               "over f's inputs only (A0 <= A1 - 0.5), so solve_relational "
               "answers sat with a witness outside the query")),
        pytest.param("non-dyadic-equality", marks=known_bug(
            6, "SPEC-13, third code path: an equality coupling with a "
               "non-dyadic coefficient (3 C == A) sets C = fl(A / 3), which "
               "breaks the equality in exact arithmetic, and the candidate "
               "is not checked, so the witness is outside the query by a "
               "rounding error"))])
    def test_falsifier_returns_only_counterexamples(self, name):
        """Every pair ``falsify_relational`` returns, and every 'sat' witness
        of ``solve_relational``, is in both boxes, keeps every coupling row
        in exact arithmetic and has replayed outputs in the unsafe region.
        Catches the joint output read as [y_g; y_f], g evaluated on f's
        inputs, groups combined with OR, the rows of a half-space combined
        with OR, the first coupled coordinate never set, the inequality
        tightening inverted, and the inputs moved after sampling."""
        f, g, spec, replay_f, replay_g = _counterexample_case(name)
        found = falsify_relational(f, g, spec, n_samples=200)
        if found is not None:
            defects = _witness_defects(spec, replay_f, replay_g, found[0])
            assert not defects, f"falsify: {found[0]}: {'; '.join(defects)}"
        verdict, cex = solve_relational(f, g, spec, n_rand=200)
        if verdict == "sat":
            defects = _witness_defects(spec, replay_f, replay_g, cex[0])
            assert not defects, f"solve: {cex[0]}: {'; '.join(defects)}"

    @pytest.mark.parametrize("margin, unsafe", [
        (None, ([[1, -1]], [0])),         # D - B <= 0: only its boundary
        (0.5, ([[-1, -1]], [-1.5]))],     # B + D >= 1.5, by 0.5: only A = 1
        ids=["default-margin-boundary", "margin-0.5"])
    def test_falsifier_keeps_the_margin(self, margin, unsafe):
        """A pair the falsifier returns is inside the unsafe region by its
        margin (``G y <= g - margin`` on the replayed outputs). With f = g
        and C == A the joint outputs are equal, so ``D - B <= 0`` holds only
        on its boundary and ``B + D >= 1.5`` by 0.5 only at A = 1. Catches
        the margin added instead of subtracted and a margin not handed to
        the unsafe-region test."""
        ident = _affine(*_ID)
        spec = _spec((1, 1, 1, 1), [0, -INF], [1, INF],
                     _equal_rows(1, [(0, 0)]), [unsafe])
        options = {} if margin is None else {"margin": margin}
        found = falsify_relational(ident, ident, spec, n_samples=300,
                                   **options)
        if found is not None:
            replay = _torch_replay(ident, 1)
            defects = _witness_defects(spec, replay, replay, found[0],
                                       margin=1e-6 if margin is None
                                       else margin)
            assert not defects, f"{found[0]}: {'; '.join(defects)}"

    def test_seed_selects_the_random_stream(self):
        """The witness of a region every pair reaches is the first sample:
        the same seed gives the same witness, two seeds give two. Catches
        a seed that ``solve_relational`` or ``falsify_relational`` does not
        hand on (fixed at 42)."""
        ident = _affine(*_ID)
        spec = _spec((1, 1, 1, 1), [0, -INF], [1, INF],
                     _equal_rows(1, [(0, 0)]), [([[-1, 0]], [1])])

        def witness(seed):
            verdict, cex = solve_relational(ident, ident, spec, seed=seed)
            assert verdict == "sat", verdict
            return cex[0]
        first, again, other = witness(1), witness(1), witness(2)
        assert np.array_equal(first, again), (first, again)
        assert not np.array_equal(first, other), (
            f"seeds 1 and 2 give the same witness {first}")

    @pytest.mark.parametrize("n_rand", [0, 7])
    def test_n_rand_is_the_number_of_samples(self, n_rand):
        """On an unreachable region (f = g, C == A, B - D >= 0.5) the
        falsifier evaluates each network exactly ``n_rand`` times, then the
        reach proves it: ('unsat', None). Catches ``n_rand`` not handed on
        and 0 read as the default."""
        class Counting(nn.Module):
            # torch.fx traces with a Proxy, which is not a Tensor: only the
            # falsifier's real forward calls are counted.
            def __init__(self, inner):
                super().__init__()
                self.inner, self.calls = inner, 0

            def forward(self, x):
                if isinstance(x, torch.Tensor):
                    self.calls += 1
                return self.inner(x)
        f, g = Counting(_affine(*_ID)), Counting(_affine(*_ID))
        spec = _spec((1, 1, 1, 1), [0, -INF], [1, INF],
                     _equal_rows(1, [(0, 0)]), [([[-1, 1]], [-0.5])])
        assert solve_relational(f, g, spec, n_rand=n_rand) == ("unsat", None)
        assert (f.calls, g.calls) == (n_rand, n_rand)

    @pytest.mark.parametrize("g_weights, reachable", [
        ([[1, 1]], False), ([[2, 1]], True)], ids=["g-equal-to-f", "g-2x"])
    def test_loaded_vnnlib_spec_is_answered_by_its_text(self, tmp_path,
                                                        g_weights, reachable):
        """Through ``load_vnnlib`` on a VNN-LIB 2.0 relational text with a
        degenerate input (``== X_f[1] 0.25``), a box open on one side
        (``>= X_g[0] -0.5``), an inequality and an equality coupling, and a
        strict output comparison. f = X_f[0] + X_f[1]; with g = f the
        property ``Y_f[0] < Y_g[0]`` is unreachable (only its boundary is
        reached), with g = 2 X_g[0] + X_g[1] it is reachable. A 'sat'
        witness must satisfy the assertions as written, read by hand below
        (strict comparison included); a reachable property must not be
        UNSAT. Catches boundary witnesses of a strict comparison and the
        engine misreading the loader's box, coupling or property arrays."""
        from n2v.utils.load_vnnlib import load_vnnlib
        path = tmp_path / "relational.vnnlib"
        path.write_text(_RELATIONAL_TEXT)
        spec = load_vnnlib(str(path))
        f, g = _affine([[1, 1]], [0]), _affine(g_weights, [0])
        replay_f, replay_g = _torch_replay(f, 2), _torch_replay(g, 2)

        def text_holds(x):
            a0, a1, c0, c1 = (Fraction(float(v)) for v in x)
            y_f = replay_f([x[:2]])[0][0]
            y_g = replay_g([x[2:]])[0][0]
            return (Fraction(-1, 2) <= a0 <= Fraction(3, 2)
                    and a1 == Fraction(1, 4) and c0 >= Fraction(-1, 2)
                    and a0 >= c0 and a1 == c1 and y_f < y_g)
        verdict, cex = solve_relational(f, g, spec, n_rand=300)
        if verdict == "sat":
            assert text_holds(cex[0]), f"witness {cex[0]} breaks the text"
        if reachable:
            assert verdict == "sat", verdict
            assert verify_relational(f, g, spec).verdict != "UNSAT"

    @known_bug(5, "the relational falsifier feeds float32 tensors to the "
                  "models, so solve_relational on a float64 torch model "
                  "raises RuntimeError (dtype mismatch) instead of answering",
               raises=RuntimeError)
    def test_float64_models_are_answered(self):
        """``solve_relational`` on float64 models answers like on float32
        ones: here every pair is unsafe, so 'sat' with a valid witness."""
        ident = _affine(*_ID).double()
        spec = _spec((1, 1, 1, 1), [0, -INF], [1, INF],
                     _equal_rows(1, [(0, 0)]), [([[-1, 0]], [1])])
        verdict, cex = solve_relational(ident, ident, spec, n_rand=5)
        assert verdict == "sat", verdict
        replay = _torch_replay(ident, 1)
        assert not _witness_defects(spec, replay, replay, cex[0])


class TestRelationalRefusals:
    @pytest.mark.parametrize("side", ["f", "g"])
    def test_exact_reach_that_splits_is_refused(self, side):
        """method='exact' with a network whose ReLU splits must raise
        NotImplementedError, never answer from one of the pieces: f's side
        through ``verify_relational``, g's through ``solve_relational`` on an
        unreachable region (|relu(A) - A| <= 1 < 4). Catches a split reduced
        to its first star (false UNSAT), g reached with approx, and a method
        not handed from ``solve_relational`` to the reach."""
        split = _Net([[1]], [0], [[1]], [0]).build(None, "split")[0]
        ident = _affine(*_ID)
        f, g = (split, ident) if side == "f" else (ident, split)
        spec = _spec((1, 1, 1, 1), [-1, -INF], [1, INF],
                     _equal_rows(1, [(0, 0)]), [([[-1, 1]], [-4])])
        with pytest.raises(NotImplementedError):
            if side == "f":
                verify_relational(f, g, spec, method="exact")
            else:
                solve_relational(f, g, spec, method="exact", n_rand=4)

    @pytest.mark.parametrize("method, expected, options", [
        pytest.param("probabilistic", NotImplementedError,
                     {"m": 100, "training_samples": 50}, marks=known_bug(
            23, "relational_reach hands method='probabilistic' to the model "
                "reach, which returns a ProbabilisticBox, and the join then "
                "fails with AttributeError ('nVar') instead of refusing",
            raises=AttributeError), id="probabilistic"),
        pytest.param("flow_matching", (NotImplementedError, TypeError), {},
                     id="flow_matching"),
        pytest.param("no-such-method", ValueError, {}, id="unknown-name")])
    def test_reach_methods_without_a_star_are_refused(self, method, expected,
                                                      options):
        """The join needs one Star per network: a probabilistic method is
        not supported (NotImplementedError; for the Box-only flow_matching
        the model reach's deliberate TypeError "requires Box input" is
        accepted too) and an unknown method name cannot be read
        (ValueError). Catches a method silently read as approx. The
        conformal options only keep the probabilistic reach small (0.4 s
        instead of 130 s)."""
        ident = _affine(*_ID)
        spec = _spec((1, 1, 1, 1), [0, -INF], [1, INF],
                     _equal_rows(1, [(0, 0)]), [([[1, 0]], [-5])])
        with pytest.raises(expected):
            verify_relational(ident, ident, spec, method=method, **options)

    @known_bug(23, "a relational spec with three networks is reached for the "
                   "first two only and fails inside numpy (ValueError from "
                   "matmul) instead of being refused", raises=ValueError)
    def test_spec_with_three_networks_is_refused(self):
        """The engine joins two networks; a third must be refused, not
        dropped."""
        ident = _affine(*_ID)
        spec = {"format": "relational",
                "networks": [{"name": n, "relation": None, "input_offset": i,
                              "input_size": 1, "output_offset": i,
                              "output_size": 1}
                             for i, n in enumerate("fgh")],
                "lb": np.zeros(3), "ub": np.ones(3), "input_coupling": None,
                "prop": [{"Hg": HalfSpace(np.array([[0.0, 0.0, 1.0]]),
                                          np.array([[-5.0]]))}]}
        with pytest.raises(NotImplementedError):
            verify_relational(ident, ident, spec)

    def test_dim_neither_bounded_nor_coupled_is_refused(self):
        """A joint input dim with no finite bound that the coupling does not
        bound either (C >= A only) cannot be read as a finite set: ValueError
        from the joint input star, the reach and the falsifier."""
        ident = _affine(*_ID)
        spec = _spec((1, 1, 1, 1), [0, -INF], [1, INF], ([[1, -1]], [0]),
                     [([[1, 0]], [-5])])
        for call in (lambda: build_joint_input_star(spec),
                     lambda: verify_relational(ident, ident, spec),
                     lambda: falsify_relational(ident, ident, spec)):
            with pytest.raises(ValueError):
                call()

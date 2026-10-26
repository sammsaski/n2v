"""Regressions for the graph reach's dynamic Pad value and the McCormick
Mul/Div operand ranges (n2v/nn/reach.py).

- OnnxPadDynamic (ONNX Pad-11/13) takes its constant value as a runtime
  input. A value that could not be folded to a constant (e.g. one produced
  by a multi-output Split) was skipped, and the pad was reached as zero
  padding. A second reach of the same model also skipped the value check,
  since the pads were already stashed on the module, so a pad refused the
  first time was reached as zero padding the second time. A Pad of a
  constant tensor was refused for a non-zero value, although its consumers
  fold it exactly.
- Constant folding could not index a multi-output op on constants (the
  tuple a Split returns), so such a value was refused or skipped.
- A Concat operand with no sets that could not be folded was skipped, which
  undersized the output and shifted the operands after it.
- The McCormick Mul/Div of two stars raised ValueError when an operand LP
  returned no optimum (a solver failure, a presolve 'infeasible' that is not
  a certificate, or an empty operand). It now takes the rigorous interval
  estimate for that dimension, as the ReLU phase bounds do.

Convention: SAT = unsafe (the unsafe region is ``G y <= g``), UNSAT = safe.
"""

import numpy as np
import pytest
import torch

from n2v.nn.neural_network import NeuralNetwork
from n2v.nn.reach import _div_sets, _mul_sets
from n2v.sets import Star
from n2v.sets.image_star import ImageStar


# ---------------------------------------------------------------------------
# Dynamic Pad constant value
# ---------------------------------------------------------------------------

def _pad_model(value_nodes, value_name, initializers=()):
    """X (1, 1, 2, 2) -> Pad(X, pads = 1 on each spatial side, value)."""
    import onnx
    onnx2torch = pytest.importorskip('onnx2torch')
    from onnx import TensorProto, helper, numpy_helper
    inits = [numpy_helper.from_array(
        np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), 'pads')]
    inits += [numpy_helper.from_array(np.asarray(a), n) for n, a in initializers]
    nodes = list(value_nodes) + [
        helper.make_node('Pad', ['X', 'pads', value_name], ['Y'])]
    graph = helper.make_graph(
        nodes, 'g',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 4, 4])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    onnx.checker.check_model(model)
    return onnx2torch.convert(model).eval()


def _reach_point(gm, value=1.0):
    S = ImageStar.from_bounds(np.full((2, 2, 1), value), np.full((2, 2, 1), value),
                              height=2, width=2, num_channels=1)
    return NeuralNetwork(gm).reach(S, method='approx')


@pytest.mark.parametrize('value', [0.0, 5.0])
def test_pad_value_from_a_split_of_constants_is_folded(value):
    """The value comes out of Split(constant). Folding could not index the
    Split's tuple, so the value 5 was reached as zero padding (the true
    border is 5, the reach said 0) and the value 0 was refused."""
    from onnx import helper
    gm = _pad_model(
        [helper.make_node('Split', ['v2'], ['va', 'vb'], axis=0),
         helper.make_node('Squeeze', ['va', 'ax0'], ['v'])], 'v',
        [('v2', np.array([value, 6.0], np.float32)),
         ('ax0', np.array([0], np.int64))])
    with torch.no_grad():
        ref = gm(torch.ones(1, 1, 2, 2))[0, 0].numpy()
    assert ref[0, 0] == value
    if value != 0.0:
        with pytest.raises(NotImplementedError, match='non-zero constant'):
            _reach_point(gm)
        return
    lo, hi = _reach_point(gm)[0].estimate_ranges()
    np.testing.assert_array_equal(np.ravel(lo), ref.ravel())
    np.testing.assert_array_equal(np.ravel(hi), ref.ravel())


def test_pad_value_that_depends_on_the_input_is_refused():
    """The value is computed from X, so it cannot be folded to a constant;
    it must be refused, not taken as 0."""
    from onnx import helper
    gm = _pad_model(
        [helper.make_node('ReduceMean', ['X'], ['v'], keepdims=0)], 'v')
    with pytest.raises(NotImplementedError, match='could not be resolved'):
        _reach_point(gm)


def test_pad_of_a_constant_tensor_is_folded_with_its_value():
    """bias = Pad(B, pads, 5) is a constant; X + bias folds it with torch.
    The value check refused it although no set is padded."""
    import onnx
    onnx2torch = pytest.importorskip('onnx2torch')
    from onnx import TensorProto, helper, numpy_helper
    inits = [numpy_helper.from_array(np.array([1, 1], np.int64), 'pads'),
             numpy_helper.from_array(np.array(5.0, np.float32), 'v'),
             numpy_helper.from_array(np.array([1.0, 2.0], np.float32), 'B')]
    nodes = [helper.make_node('Pad', ['B', 'pads', 'v'], ['bias']),
             helper.make_node('Add', ['X', 'bias'], ['Y'])]
    graph = helper.make_graph(
        nodes, 'g', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [4])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    onnx.checker.check_model(model)
    gm = onnx2torch.convert(model).eval()
    S = Star.from_bounds(np.zeros((4, 1)), np.zeros((4, 1)))
    lo, hi = NeuralNetwork(gm).reach(S, method='approx')[0].estimate_ranges()
    np.testing.assert_array_equal(np.ravel(lo), [5.0, 1.0, 2.0, 5.0])
    np.testing.assert_array_equal(np.ravel(hi), [5.0, 1.0, 2.0, 5.0])


def test_nonzero_pad_value_is_refused_on_every_reach():
    """The first reach refused the pad value 5 after stashing the pads on the
    module; a second reach of the same model skipped the check and zero
    padded."""
    gm = _pad_model([], 'v', [('v', np.array(5.0, np.float32))])
    for _ in range(2):
        with pytest.raises(NotImplementedError):
            _reach_point(gm)


@pytest.mark.parametrize('case', ['initializer', 'constant_node', 'omitted'])
def test_zero_pad_value_is_reached_exactly(case):
    from onnx import helper, numpy_helper
    zero = np.array(0.0, np.float32)
    if case == 'initializer':
        gm = _pad_model([], 'v', [('v', zero)])
    elif case == 'constant_node':
        gm = _pad_model([helper.make_node(
            'Constant', [], ['v'], value=numpy_helper.from_array(zero))], 'v')
    else:
        gm = _pad_model([], '')
    out = _reach_point(gm, 1.0)
    with torch.no_grad():
        ref = gm(torch.ones(1, 1, 2, 2))[0, 0].numpy()
    lo, hi = out[0].estimate_ranges()
    np.testing.assert_array_equal(np.ravel(lo), ref.ravel())
    np.testing.assert_array_equal(np.ravel(hi), ref.ravel())


# ---------------------------------------------------------------------------
# Concat operands with no sets
# ---------------------------------------------------------------------------

def test_concat_operand_from_a_split_of_constants_is_kept():
    """Y = Relu(Concat(Split(const)[0], X)) with X in [0, 1]^2 is
    [10, X0, X1]. The constant operand could not be folded and was skipped:
    the reach returned 2 dimensions, the first one X0 in [0, 1]."""
    import onnx
    onnx2torch = pytest.importorskip('onnx2torch')
    from onnx import TensorProto, helper, numpy_helper
    nodes = [helper.make_node('Split', ['c2'], ['ca', 'cb'], axis=0),
             helper.make_node('Concat', ['ca', 'X'], ['Y0'], axis=1),
             helper.make_node('Relu', ['Y0'], ['Y'])]
    graph = helper.make_graph(
        nodes, 'g', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])],
        initializer=[numpy_helper.from_array(
            np.array([[10.0], [20.0]], np.float32), 'c2')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    onnx.checker.check_model(model)
    gm = onnx2torch.convert(model).eval()
    S = Star.from_bounds(np.zeros((2, 1)), np.ones((2, 1)))
    out = NeuralNetwork(gm).reach(S, method='approx')[0]
    assert out.dim == 3
    lo, hi = out.estimate_ranges()
    np.testing.assert_allclose(np.ravel(lo), [10.0, 0.0, 0.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.ravel(hi), [10.0, 1.0, 1.0], rtol=0, atol=1e-12)


def test_concat_operand_without_sets_or_constant_value_is_refused():
    """The second operand (x.size()) has no sets and is not a constant; the
    Concat returned only x's 2 dimensions."""
    import torch.fx as fx
    import torch.nn as nn
    from onnx2torch.node_converters.concat import OnnxConcat
    graph = fx.Graph()
    x = graph.placeholder('x')
    size = graph.call_method('size', (x,))
    graph.output(graph.call_module('cat', (x, size)))
    root = nn.Module()
    root.cat = OnnxConcat(axis=0)
    gm = fx.GraphModule(root, graph)
    S = Star.from_bounds(np.zeros((2, 1)), np.ones((2, 1)))
    with pytest.raises(NotImplementedError, match='could not be folded'):
        NeuralNetwork(gm).reach(S, method='approx')


# ---------------------------------------------------------------------------
# McCormick operand ranges when an LP returns no optimum
# ---------------------------------------------------------------------------

def _operands():
    """x, y over one shared predicate a in [-1, 1]^2; y > 0 for Div."""
    base = Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1)))
    x = base.affine_map(np.array([[1.5, 0.5], [0.5, -0.5]]),
                        np.array([[0.5], [1.0]]))
    y = base.affine_map(np.array([[0.5, 0.25], [-0.25, 0.5]]),
                        np.array([[2.0], [-1.5]]))
    return base, x, y


@pytest.mark.parametrize('op', ['mul', 'div'])
def test_mccormick_operand_without_lp_optimum_uses_interval_estimate(
        monkeypatch, op):
    """The LP of dimension 1 returns no optimum for both operands. The op
    raised ValueError; with the interval estimate every true (a, x*y) or
    (a, x/y) must satisfy the output star's constraints."""
    base, x, y = _operands()
    orig = Star._solve_lp_batch
    dropped = []

    def no_optimum_on_row_1(self, objectives, minimize_flags, lp_solver=None):
        res = orig(self, objectives, minimize_flags, lp_solver)
        row = np.ravel(self.V[1, 1:])
        out = []
        for f, r in zip(objectives, res):
            hit = self.dim > 1 and np.array_equal(np.ravel(f), row)
            dropped.append(hit)
            out.append(None if hit else r)
        return out

    monkeypatch.setattr(Star, '_solve_lp_batch', no_optimum_on_row_1)
    out = (_mul_sets if op == 'mul' else _div_sets)([x], [y])[0]
    monkeypatch.undo()
    assert any(dropped)

    rng = np.random.default_rng(0)
    C, d = out.C, out.d.ravel()
    plb, pub = out.predicate_lb.ravel(), out.predicate_ub.ravel()
    for _ in range(500):
        a = rng.uniform(-1.0, 1.0, base.nVar)
        xv = x.V[:, 0] + x.V[:, 1:] @ a
        yv = y.V[:, 0] + y.V[:, 1:] @ a
        z = xv * yv if op == 'mul' else xv / yv
        p = np.concatenate([a, z])
        assert np.all(C @ p <= d + 1e-9)
        assert np.all(p >= plb - 1e-12) and np.all(p <= pub + 1e-12)
        np.testing.assert_array_equal(out.V[:, 0] + out.V[:, 1:] @ p, z)


@pytest.mark.parametrize('op', ['mul', 'div'])
def test_mccormick_with_an_empty_operand_does_not_raise(op):
    """An empty operand (contradictory constraints) has no LP optimum; the
    op raised ValueError instead of returning a (sound, empty) star."""
    _, x, y = _operands()
    C = np.array([[1.0, 0.0], [-1.0, 0.0]])
    d = np.array([[-0.5], [-0.5]])                # a0 <= -0.5 and a0 >= 0.5
    empty = Star(x.V, C, d, x.predicate_lb, x.predicate_ub)
    out = (_mul_sets if op == 'mul' else _div_sets)([empty], [y])[0]
    assert isinstance(out, Star) and out.dim == x.dim

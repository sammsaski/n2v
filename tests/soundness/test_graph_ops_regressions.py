"""Soundness regressions for the graph executor and set-algebra ops in
``n2v/nn/reach.py``.

Each test pins one measured failure mode with a minimal reproduction:

  * fx ``call_function`` / ``call_method`` nodes without a reach rule
    (``x + f(x)``, scalar ``*``, ``torch.clamp``, ``.neg()``) were skipped,
    and their consumers silently reused the previous node's sets. They must
    raise; shape queries (``x.size(0)``, ``x.shape[0]``), eval-mode dropout
    and value-preserving methods (``.contiguous()``, ``.clone()``, float
    casts) stay supported. A Reshape of a constant fell back to the running
    sets.
  * Residual Add / Concat paired the pieces of two independently split
    branches by list index, so mixed input regions were never formed. Pieces
    that descend from one another (a piece and its sigmoid, a piece and its
    ReLU split children) are still combined one to one.
  * Div zipped its operand lists without broadcasting, dropping all but one
    piece of a split numerator.
  * ImageZono Add/Sub with a full (C, H, W) constant added it in CHW order to
    the HWC-stored center; ImageZono + flat Zono coerced in HWC order.
    Constant - set tiled the constant without its tensor shape.
  * McCormick Mul/Div built envelopes and the product box from raw LP
    optima, which can undershoot the true operand range.
  * The input-box finitization ignored the caller's LP solver; shape
    propagation used a float32 dummy on float64 models.
  * The zonotope bound pre-pass runs this executor on an ImageZono, a Zono
    subclass: the flat ReduceSum/ReduceMean path took it and applied a CHW
    reduction to its HWC rows, and Slice cut it as a flat Zono (the H slice
    only, on HWC rows). The misplaced bounds zeroed live ReLUs.
  * The pairing ops read identical predicate systems as the same variables.
    Ops that re-encode their output from bounds (Softmax, the attention
    softmax, set-by-set MatMul) built it with Star.from_bounds, whose system
    equals the input star's; and two nodes can append variables with
    identical rows and bounds for different functions (a box-only Sin and
    Cos, Sign or an odd Pow across 0; Round(x) and Floor(x + 1/2)). Sub and
    Concat then cancelled unrelated values: false UNSATs. Every placeholder
    of a two-input graph was bound to the one input set.
"""

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper, numpy_helper
from scipy.optimize import linprog

import n2v.sets.predicate_box as predicate_box
from n2v.nn.reach import (ReachConfig, _add_sets, _div_sets, _mul_sets,
                          _propagate_shapes, reach_pytorch_model)
from n2v.sets import Box, HalfSpace, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from n2v.utils.verify_specification import verify_specification


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _star_distance(s, y):
    """Inf-norm distance from point ``y`` to star ``s`` (one LP)."""
    V = np.asarray(s.V, dtype=np.float64)
    c, G = V[:, 0], V[:, 1:]
    m, n = G.shape
    y = np.asarray(y, dtype=np.float64).ravel()
    A = [np.hstack([G, -np.ones((m, 1))]), np.hstack([-G, -np.ones((m, 1))])]
    b = [y - c, c - y]
    if s.C is not None and np.asarray(s.C).size:
        C = np.asarray(s.C, dtype=np.float64).reshape(-1, n)
        A.append(np.hstack([C, np.zeros((C.shape[0], 1))]))
        b.append(np.asarray(s.d, dtype=np.float64).ravel())
    lb = np.asarray(s.predicate_lb, dtype=np.float64).ravel()
    ub = np.asarray(s.predicate_ub, dtype=np.float64).ravel()
    bounds = list(zip(lb, ub)) + [(0, None)]
    f = np.zeros(n + 1)
    f[-1] = 1.0
    r = linprog(f, A_ub=np.vstack(A), b_ub=np.concatenate(b),
                bounds=bounds, method='highs')
    return r.fun if r.status == 0 else np.inf


def _covered(sets, y, tol=1e-6):
    return any(_star_distance(s, y) <= tol for s in sets)


def _box_spec(y, r=1e-3):
    """Unsafe region: the box of half-width ``r`` around output ``y``."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    G = np.vstack([np.eye(n), -np.eye(n)])
    g = np.concatenate([y + r, -(y - r)]).reshape(-1, 1)
    return HalfSpace(G, g)


def _onnx_model(tmp_path, nodes, n_in, n_out, inits, name='m',
                in_shape=None, out_shape=None):
    """Save a float32 opset-13 ONNX graph and load it with load_onnx."""
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, in_shape or [1, n_in])],
        [helper.make_tensor_value_info(
            'Y', TensorProto.FLOAT, out_shape or [1, n_out])],
        initializer=[numpy_helper.from_array(np.asarray(a), k)
                     for k, a in inits.items()])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 10
    path = str(tmp_path / f'{name}.onnx')
    onnx.save(model, path)
    return load_onnx(path)


def _reach_or_none(model, input_set, method):
    """Reach sets, or None if the graph is rejected as unsupported."""
    try:
        return reach_pytorch_model(model, input_set, method=method)
    except NotImplementedError:
        return None


# ---------------------------------------------------------------------------
# fx nodes without a reach rule must not be skipped
# ---------------------------------------------------------------------------

class _Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(2))
            self.fc.bias.zero_()


class _ResAdd(_Residual):
    def forward(self, x):
        return x + torch.relu(self.fc(x))


class _Scale(_Residual):
    def forward(self, x):
        return self.fc(x) * 3.0


class _Clamp(_Residual):
    def forward(self, x):
        return torch.clamp(4.0 * self.fc(x), min=0.0)


class _Neg(_Residual):
    def forward(self, x):
        return self.fc(x).neg()


class _TorchCat(_Residual):
    def forward(self, x):
        return torch.cat((x, self.fc(x)), dim=1)


@pytest.mark.parametrize('cls', [_ResAdd, _Scale, _Clamp, _Neg, _TorchCat])
@pytest.mark.parametrize('set_kind,method', [
    ('star', 'exact'), ('star', 'approx'), ('box', 'approx')])
def test_unsupported_fx_nodes_raise_instead_of_dropping(cls, set_kind,
                                                        method):
    # x in [0.5, 1]^2 with fc = identity. The skipped node's consumers used
    # to reuse the Linear's sets, so e.g. x + relu(x) reached [0.5, 1] while
    # the true output at x = (1, 1) is (2, 2).
    model = cls().eval()
    lb, ub = np.full(2, 0.5), np.ones(2)
    S = Star.from_bounds(lb, ub) if set_kind == 'star' else Box(lb, ub)
    out = _reach_or_none(model, S, method)
    if out is None:
        return
    for x in ([0.5, 0.5], [1.0, 1.0], [0.5, 1.0]):
        with torch.no_grad():
            y = model(torch.tensor([x], dtype=torch.float32)).numpy().ravel()
        if set_kind == 'star':
            assert _covered(out, y, tol=1e-5), (cls.__name__, x, y)
        else:
            assert any(np.all(b.lb.ravel() - 1e-6 <= y)
                       and np.all(y <= b.ub.ravel() + 1e-6) for b in out)


class _ShapeOps(nn.Module):
    """Shape queries and layout-only methods that reach must still accept."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = h.view(h.size(0), -1)
        h = h.contiguous().reshape(h.shape[0], 4)
        h = torch.flatten(h, 1)
        return self.fc2(h)


def test_shape_query_and_layout_ops_are_supported():
    torch.manual_seed(0)
    model = _ShapeOps().eval()
    lb, ub = -np.ones(4), np.ones(4)
    out = reach_pytorch_model(model, Star.from_bounds(lb, ub), method='exact')
    rng = np.random.default_rng(0)
    for x in rng.uniform(lb, ub, size=(20, 4)):
        with torch.no_grad():
            y = model(torch.tensor(x[None], dtype=torch.float32)).numpy()
        assert _covered(out, y.ravel(), tol=1e-5)


class _EvalIdentityOps(_ShapeOps):
    """Eval-mode dropout and value-preserving tensor methods."""

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.dropout(h, p=0.5, training=self.training)
        h = torch.dropout(h, 0.2, False)
        h = h.clone().detach().double().float()
        h = h.to(torch.float64).to(dtype=torch.float32).type(torch.float32)
        return self.fc2(h)


class _TrainDropout(_ShapeOps):
    def forward(self, x):
        return self.fc2(F.dropout(F.relu(self.fc1(x)), p=0.5, training=True))


class _DefaultTrainingDropout(_ShapeOps):
    def forward(self, x):          # F.dropout defaults to training=True
        return self.fc2(F.dropout(F.relu(self.fc1(x)), 0.5))


class _DeviceMove(_ShapeOps):
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)).to('cpu'))


class _HalfCast(_ShapeOps):
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)).half().float())


@pytest.mark.parametrize('method', ['exact', 'approx'])
def test_eval_dropout_and_identity_methods_pass_sets_through(method):
    torch.manual_seed(0)
    model = _EvalIdentityOps().eval()
    plain = nn.Sequential(model.fc1, nn.ReLU(), model.fc2).eval()
    S = Star.from_bounds(-np.ones(4), np.ones(4))
    out = reach_pytorch_model(model, S, method=method)
    ref = reach_pytorch_model(plain, S, method=method)
    assert len(out) == len(ref)
    for a, b in zip(out, ref):
        np.testing.assert_array_equal(a.V, b.V)
    rng = np.random.default_rng(0)
    for x in rng.uniform(-1.0, 1.0, size=(10, 4)):
        with torch.no_grad():
            y = model(torch.tensor(x[None], dtype=torch.float32)).numpy()
        assert _covered(out, y.ravel(), tol=1e-5)


@pytest.mark.parametrize('cls', [_TrainDropout, _DefaultTrainingDropout,
                                 _DeviceMove, _HalfCast])
def test_random_dropout_and_value_changing_casts_raise(cls):
    model = cls().eval()
    with pytest.raises(NotImplementedError):
        reach_pytorch_model(model, Star.from_bounds(-np.ones(4), np.ones(4)),
                            method='approx')


def test_constant_reshape_is_folded_not_replaced_by_running_sets(tmp_path):
    # Add(X, Reshape(B)) with B an initializer. The Reshape of the constant
    # used to be computed on the running sets (X), giving X + X.
    model = _onnx_model(
        tmp_path,
        [helper.make_node('Reshape', ['B', 'shape'], ['Br']),
         helper.make_node('Add', ['X', 'Br'], ['Y'])],
        2, 2,
        {'B': np.array([[10.0], [20.0]], dtype=np.float32),
         'shape': np.array([1, 2], dtype=np.int64)})
    out = reach_pytorch_model(
        model, Star.from_bounds(np.zeros(2), np.ones(2)), method='approx')
    lo, hi = out[0].get_ranges()
    np.testing.assert_allclose(lo.ravel(), [10.0, 20.0], atol=1e-6)
    np.testing.assert_allclose(hi.ravel(), [11.0, 21.0], atol=1e-6)


@pytest.mark.parametrize('precompute', [False, 'ibp', 'zono'])
@pytest.mark.parametrize('method', ['approx', 'exact'])
def test_constant_reshape_under_precomputed_bounds(tmp_path, precompute,
                                                   method):
    # Relu(X + Reshape(B)), X in [-1, -0.5]^2, B = (2, 2): y in [1, 1.5]^2.
    # Pre-activation bounds of X + X ([-2, -1]) made ReLU zero y.
    model = _onnx_model(
        tmp_path,
        [helper.make_node('Reshape', ['B', 'shape'], ['Br']),
         helper.make_node('Add', ['X', 'Br'], ['P']),
         helper.make_node('Relu', ['P'], ['Y'])],
        2, 2,
        {'B': np.array([[2.0], [2.0]], dtype=np.float32),
         'shape': np.array([1, 2], dtype=np.int64)}, name='crelu')
    out = reach_pytorch_model(
        model, Star.from_bounds(-np.ones(2), -0.5 * np.ones(2)),
        method=method, precompute_bounds=precompute)
    assert _covered(out, [1.0, 1.0]) and _covered(out, [1.5, 1.5])
    spec = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.9]]))  # y0>=0.9
    assert verify_specification(out, spec).verdict != 'UNSAT'


# ---------------------------------------------------------------------------
# Independently split branches must meet in every combination
# ---------------------------------------------------------------------------

def _opposite_split_model(tmp_path, join):
    # h = (x, x + 2); a = relu(h); b = relu(h * (-1, 1)).
    # Add: y = (|x|, 2x + 4). Concat: y = (relu(x), x+2, relu(-x), x+2).
    nodes = [
        helper.make_node('Add', ['X', 'two'], ['X2']),
        helper.make_node('Concat', ['X', 'X2'], ['H'], axis=1),
        helper.make_node('Relu', ['H'], ['A']),
        helper.make_node('Mul', ['H', 'flip'], ['Hf']),
        helper.make_node('Relu', ['Hf'], ['Bb']),
    ]
    if join == 'add':
        nodes.append(helper.make_node('Add', ['A', 'Bb'], ['Y']))
        n_out = 2
    else:
        nodes.append(helper.make_node('Concat', ['A', 'Bb'], ['Y'], axis=1))
        n_out = 4
    return _onnx_model(
        tmp_path, nodes, 1, n_out,
        {'two': np.array([[2.0]], dtype=np.float32),
         'flip': np.array([[-1.0, 1.0]], dtype=np.float32)},
        name=f'opp_{join}')


def _opposite_split_truth(x, join):
    x = float(x)
    h = np.array([x, x + 2.0])
    a, b = np.maximum(h, 0), np.maximum(h * [-1.0, 1.0], 0)
    return a + b if join == 'add' else np.concatenate([a, b])


def _parallel_kwargs(parallel):
    # A parallel exact ReLU returns pieces in completion order.
    return {'parallel': True, 'n_workers': 4} if parallel else {}


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('join', ['add', 'concat'])
def test_opposite_splits_cover_every_input(tmp_path, join, parallel):
    model = _opposite_split_model(tmp_path, join)
    out = reach_pytorch_model(
        model, Star.from_bounds(np.array([-1.0]), np.array([1.0])),
        method='exact', **_parallel_kwargs(parallel))
    for x in np.linspace(-1.0, 1.0, 21):
        assert _covered(out, _opposite_split_truth(x, join)), x
    # x = -1 reaches this box; the old index pairing reported UNSAT
    y = _opposite_split_truth(-1.0, join)
    assert verify_specification(out, _box_spec(y)).verdict != 'UNSAT'


def _parallel_split_model(tmp_path, join):
    # a = relu(x0, x0 + 3), b = relu(x1, x1 + 3): each branch splits on a
    # different input, so their pieces are unrelated input regions.
    nodes = [
        helper.make_node('MatMul', ['X', 'Wa'], ['Ha']),
        helper.make_node('Add', ['Ha', 'ba'], ['Ga']),
        helper.make_node('Relu', ['Ga'], ['A']),
        helper.make_node('MatMul', ['X', 'Wb'], ['Hb']),
        helper.make_node('Add', ['Hb', 'bb'], ['Gb']),
        helper.make_node('Relu', ['Gb'], ['Bb']),
    ]
    if join == 'add':
        nodes.append(helper.make_node('Add', ['A', 'Bb'], ['Y']))
        n_out = 2
    else:
        nodes.append(helper.make_node('Concat', ['A', 'Bb'], ['Y'], axis=1))
        n_out = 4
    f32 = np.float32
    return _onnx_model(
        tmp_path, nodes, 2, n_out,
        {'Wa': np.array([[1.0, 1.0], [0.0, 0.0]], dtype=f32),
         'ba': np.array([0.0, 3.0], dtype=f32),
         'Wb': np.array([[0.0, 0.0], [1.0, 1.0]], dtype=f32),
         'bb': np.array([0.0, 3.0], dtype=f32)},
        name=f'par_{join}')


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('join', ['add', 'concat'])
def test_parallel_splits_cover_every_input(tmp_path, join, parallel):
    model = _parallel_split_model(tmp_path, join)
    out = reach_pytorch_model(
        model, Star.from_bounds(-np.ones(2), np.ones(2)), method='exact',
        **_parallel_kwargs(parallel))
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, size=(60, 2))
    for x0, x1 in pts:
        a = np.maximum([x0, x0 + 3.0], 0)
        b = np.maximum([x1, x1 + 3.0], 0)
        y = a + b if join == 'add' else np.concatenate([a, b])
        assert _covered(out, y), (x0, x1)
    # a mixed-sign input (x0 > 0 > x1): a region the index pairing dropped
    x0, x1 = 0.274, -0.46
    a = np.maximum([x0, x0 + 3.0], 0)
    b = np.maximum([x1, x1 + 3.0], 0)
    y = a + b if join == 'add' else np.concatenate([a, b])
    assert verify_specification(out, _box_spec(y)).verdict != 'UNSAT'


def test_reordered_identical_pieces_are_matched_exactly():
    # Same two input regions in both lists, in opposite order (a parallel
    # exact reach returns pieces in completion order). The pieces must be
    # matched by predicate system and added exactly, not crossed.
    lb, ub = np.array([[-1.0]]), np.array([[1.0]])
    C1, d1 = np.array([[1.0]]), np.array([[0.0]])     # alpha <= 0
    C2, d2 = np.array([[-1.0]]), np.array([[0.0]])    # alpha >= 0
    a1 = Star(np.array([[0.0, 1.0]]), C1, d1, lb, ub)
    a2 = Star(np.array([[0.0, 2.0]]), C2, d2, lb, ub)
    b1 = Star(np.array([[1.0, -1.0]]), C1, d1, lb, ub)
    b2 = Star(np.array([[1.0, 3.0]]), C2, d2, lb, ub)
    out = _add_sets([a1, a2], [b2, b1], 'add')
    assert len(out) == 2
    np.testing.assert_array_equal(out[0].V, [[1.0, 0.0]])
    np.testing.assert_array_equal(out[1].V, [[1.0, 5.0]])
    np.testing.assert_array_equal(out[0].C, C1)
    np.testing.assert_array_equal(out[1].C, C2)


def test_pieces_without_predicate_systems_meet_in_every_pair():
    # Boxes carry no region information, so no pairing can be assumed.
    a = [Box(np.array([[0.0]]), np.array([[1.0]])),
         Box(np.array([[10.0]]), np.array([[11.0]]))]
    b = [Box(np.array([[0.0]]), np.array([[1.0]])),
         Box(np.array([[100.0]]), np.array([[101.0]]))]
    out = _add_sets(a, b, 'add')
    got = sorted((float(s.lb[0, 0]), float(s.ub[0, 0])) for s in out)
    assert got == [(0.0, 2.0), (10.0, 12.0), (100.0, 102.0),
                   (110.0, 112.0)]


def _trunk_model(tmp_path, join):
    # t = relu(x) splits into 4 pieces; a = relu(t + 1) keeps every piece
    # (no neuron crosses zero) but a parallel reach may reorder them;
    # b = t * (2, -1) keeps t's order.
    nodes = [
        helper.make_node('Relu', ['X'], ['T']),
        helper.make_node('Add', ['T', 'one'], ['T1']),
        helper.make_node('Relu', ['T1'], ['A']),
        helper.make_node('Mul', ['T', 'w'], ['Bb']),
    ]
    if join == 'add':
        nodes.append(helper.make_node('Add', ['A', 'Bb'], ['Y']))
        n_out = 2
    else:
        nodes.append(helper.make_node('Concat', ['A', 'Bb'], ['Y'], axis=1))
        n_out = 4
    return _onnx_model(
        tmp_path, nodes, 2, n_out,
        {'one': np.array([1.0], dtype=np.float32),
         'w': np.array([2.0, -1.0], dtype=np.float32)},
        name=f'trunk_{join}')


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('join', ['add', 'concat'])
def test_pieces_of_a_common_split_are_matched_in_any_order(tmp_path, join,
                                                          parallel):
    model = _trunk_model(tmp_path, join)
    out = reach_pytorch_model(
        model, Star.from_bounds(-np.ones(2), np.ones(2)), method='exact',
        **_parallel_kwargs(parallel))
    assert len(out) == 4        # one per piece of t, not every combination
    rng = np.random.default_rng(2)
    for x in rng.uniform(-1.0, 1.0, size=(40, 2)):
        t = np.maximum(x, 0)
        a, b = t + 1.0, t * [2.0, -1.0]
        assert _covered(out, a + b if join == 'add'
                        else np.concatenate([a, b])), x


def _gate_model(tmp_path, tail):
    # g = W1 relu(W0 x + b0) + b1 has several pieces in exact mode; the
    # tail combines g with a function of g that keeps g's pieces as
    # prefixes of its own predicate systems.
    rng = np.random.default_rng(3)
    f32 = np.float32
    inits = {'W0': rng.normal(size=(2, 4)).astype(f32),
             'b0': (0.3 * rng.normal(size=4)).astype(f32),
             'W1': rng.normal(size=(4, 2)).astype(f32),
             'b1': (0.3 * rng.normal(size=2)).astype(f32)}
    nodes = [helper.make_node('MatMul', ['X', 'W0'], ['t0']),
             helper.make_node('Add', ['t0', 'b0'], ['t1']),
             helper.make_node('Relu', ['t1'], ['h']),
             helper.make_node('MatMul', ['h', 'W1'], ['t2'])]
    last = 'Y' if tail == 'g' else 'g'
    nodes.append(helper.make_node('Add', ['t2', 'b1'], [last]))
    if tail == 'mul_sigmoid':
        nodes += [helper.make_node('Sigmoid', ['g'], ['s']),
                  helper.make_node('Mul', ['g', 's'], ['Y'])]
    elif tail == 'add_relu':
        nodes += [helper.make_node('Relu', ['g'], ['r']),
                  helper.make_node('Add', ['g', 'r'], ['Y'])]
    elif tail == 'relu':
        nodes.append(helper.make_node('Relu', ['g'], ['Y']))
    model = _onnx_model(tmp_path, nodes, 2, 2, inits, name=f'gate_{tail}')
    return model, inits


def _gate_truth(x, inits, tail):
    h = np.maximum(x @ inits['W0'].astype(float) + inits['b0'], 0)
    g = h @ inits['W1'].astype(float) + inits['b1']
    if tail == 'mul_sigmoid':
        return g / (1.0 + np.exp(-g))
    return g + np.maximum(g, 0)


@pytest.mark.parametrize('tail,pieces_of', [('mul_sigmoid', 'g'),
                                            ('add_relu', 'relu')])
def test_descendant_pieces_are_paired_with_their_ancestor(tmp_path, tail,
                                                          pieces_of):
    # g * sigmoid(g): the sigmoid relaxation appends predicate variables to
    # each piece of g. g + relu(g): each piece of relu(g) is a split child of
    # a piece of g. Either way every piece meets only its own ancestor; the
    # cartesian product formed N^2 (or N*M) looser sets.
    S = Star.from_bounds(-np.ones(2), np.ones(2))
    ref, _ = _gate_model(tmp_path, pieces_of)
    n_expected = len(reach_pytorch_model(ref, S, method='exact'))
    model, inits = _gate_model(tmp_path, tail)
    out = reach_pytorch_model(model, S, method='exact')
    assert len(out) == n_expected
    rng = np.random.default_rng(4)
    for x in np.vstack([rng.uniform(-1.0, 1.0, size=(30, 2)),
                        [[1.0, 1.0], [-1.0, 1.0]]]):
        assert _covered(out, _gate_truth(x, inits, tail), tol=1e-5), x


def test_pieces_with_duplicate_systems_meet_every_same_system_piece():
    # Two pieces over the same region in each list: which one holds a given
    # input's value is unknown, so all four combinations are formed.
    lb, ub = np.array([[-1.0]]), np.array([[1.0]])
    C, d = np.zeros((0, 1)), np.zeros((0, 1))
    a = [Star(np.array([[0.0, 1.0]]), C, d, lb, ub),
         Star(np.array([[10.0, 1.0]]), C, d, lb, ub)]
    b = [Star(np.array([[0.0, 1.0]]), C, d, lb, ub),
         Star(np.array([[100.0, 1.0]]), C, d, lb, ub)]
    centers = sorted(float(s.V[0, 0]) for s in _add_sets(a, b, 'add'))
    assert centers == [0.0, 10.0, 100.0, 110.0]


def test_prefix_match_needs_a_constraint_row():
    # A system without constraint rows is only a predicate box, which stars
    # built independently of one another can share, so a piece that only
    # extends such a system is not taken to descend from it: every
    # combination is formed.
    C0, d0 = np.zeros((0, 1)), np.zeros((0, 1))
    o = [Star(np.array([[0.0, 1.0]]), C0, d0, np.array([[-1.0]]),
              np.array([[1.0]])),
         Star(np.array([[5.0, 1.0]]), C0, d0, np.array([[-1.0]]),
              np.array([[0.5]]))]
    lb, ub = np.array([[-1.0]]), np.array([[1.0]])
    p = [Star(np.array([[0.0, 1.0]]), np.array([[1.0]]), np.zeros((1, 1)),
              lb, ub),
         Star(np.array([[0.0, 1.0]]), np.array([[-1.0]]), np.zeros((1, 1)),
              lb, ub)]
    assert len(_add_sets(p, o, 'add')) == 4


# ---------------------------------------------------------------------------
# Div must not truncate a split operand
# ---------------------------------------------------------------------------

def test_div_keeps_every_piece_of_a_split_numerator(tmp_path):
    # y = relu(x) / (x + 3), x in [-1, 1]^2: exact reach splits the
    # numerator into 4 pieces against one denominator set. The old zip kept
    # only the first piece (range [0, 0]) though y(1, 1) = (0.25, 0.25).
    model = _onnx_model(
        tmp_path,
        [helper.make_node('Relu', ['X'], ['R']),
         helper.make_node('Add', ['X', 'three'], ['D']),
         helper.make_node('Div', ['R', 'D'], ['Y'])],
        2, 2, {'three': np.array([3.0], dtype=np.float32)}, name='div')
    out = reach_pytorch_model(
        model, Star.from_bounds(-np.ones(2), np.ones(2)), method='exact')
    assert len(out) == 4
    rng = np.random.default_rng(1)
    for x in np.vstack([rng.uniform(-1, 1, size=(12, 2)), [[1.0, 1.0]]]):
        assert _covered(out, np.maximum(x, 0) / (x + 3.0)), x
    spec = HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.1]]))  # y0>=0.1
    assert verify_specification(out, spec).verdict != 'UNSAT'


# ---------------------------------------------------------------------------
# ImageZono constant add/sub and coercion use the HWC layout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('op,mirrored', [
    ('Add', False), ('Add', True), ('Sub', False), ('Sub', True)])
def test_imagezono_full_constant_add_sub_layout(tmp_path, op, mirrored):
    # X (1, 2, 2, 2) op B, B = 10 * arange(8) in (C, H, W) order, then
    # Flatten (CHW). For X in [0, 1]^8 every output lies in
    # [x-free part, x-free part + 1] elementwise, in ONNX order.
    B = (10.0 * np.arange(8, dtype=np.float32)).reshape(1, 2, 2, 2)
    ins = ['B', 'X'] if mirrored else ['X', 'B']
    model = _onnx_model(
        tmp_path,
        [helper.make_node(op, ins, ['A']),
         helper.make_node('Flatten', ['A'], ['Y'])],
        8, 8, {'B': B}, name=f'iz_{op}_{mirrored}',
        in_shape=[1, 2, 2, 2])
    lo_img = np.zeros((2, 2, 2))    # (H, W, C) image box for ImageZono
    hi_img = np.ones((2, 2, 2))
    Z = ImageZono.from_bounds(lo_img, hi_img, 2, 2, 2)
    out = reach_pytorch_model(model, Z, method='approx')
    lo, hi = out[0].get_bounds()
    b = B.ravel().astype(np.float64)
    if op == 'Add':
        exp_lo, exp_hi = b, b + 1.0
    elif not mirrored:              # X - B
        exp_lo, exp_hi = -b, -b + 1.0
    else:                           # B - X
        exp_lo, exp_hi = b - 1.0, b
    np.testing.assert_allclose(np.asarray(lo).ravel(), exp_lo, atol=1e-6)
    np.testing.assert_allclose(np.asarray(hi).ravel(), exp_hi, atol=1e-6)


def test_imagezono_plus_flat_zono_aligns_in_chw_order():
    # ImageZono t (stored HWC) + the same tensor as a flat CHW Zono must
    # give 2t in CHW order (ImageStar + Star flattens in CHW order too).
    h, w, c = 2, 2, 2
    t_chw = np.arange(8, dtype=np.float64)                 # (C, H, W) flat
    t_hwc = t_chw.reshape(c, h, w).transpose(1, 2, 0)
    iz = ImageZono(t_hwc.reshape(-1, 1), np.zeros((8, 1)), h, w, c)
    z = Zono(t_chw.reshape(-1, 1), np.zeros((8, 1)))
    out = _add_sets([iz], [z], 'add')[0]
    np.testing.assert_allclose(np.asarray(out.c).ravel(), 2 * t_chw)
    out = _add_sets([z], [iz], 'sub')[0]
    np.testing.assert_allclose(np.asarray(out.c).ravel(), 0.0)


def _output_bounds(s):
    if isinstance(s, Box):
        return np.ravel(s.lb), np.ravel(s.ub)
    if isinstance(s, (Zono, ImageZono)):
        c = np.ravel(s.c)
        r = np.abs(np.asarray(s.V)).sum(axis=1).ravel()
        return c - r, c + r
    lo, hi = s.get_ranges()
    return np.ravel(lo), np.ravel(hi)


@pytest.mark.parametrize('b_shape', [(1, 2, 1, 1), (1, 1, 2, 2)])
@pytest.mark.parametrize('set_kind', ['star', 'box', 'zono', 'imagestar',
                                      'imagezono'])
def test_constant_minus_set_uses_the_constant_layout(tmp_path, b_shape,
                                                     set_kind):
    # Y = Flatten(B - X), X in [0, 1]^(2x2x2): Y lies in [B - 1, B] with B
    # broadcast to (1, 2, 2, 2) in CHW order. B - X tiled B over the flat
    # set instead (per-channel B = (0, 100) gave -1, 99, -1, 99, ...).
    B = (100.0 + 10.0 * np.arange(int(np.prod(b_shape)))).astype(
        np.float32).reshape(b_shape)
    model = _onnx_model(
        tmp_path,
        [helper.make_node('Sub', ['B', 'X'], ['A']),
         helper.make_node('Flatten', ['A'], ['Y'])],
        8, 8, {'B': B}, name='csub', in_shape=[1, 2, 2, 2])
    kwargs = {}
    if set_kind in ('star', 'box', 'zono'):
        lb, ub = np.zeros(8), np.ones(8)
        S = {'star': Star.from_bounds(lb, ub), 'box': Box(lb, ub),
             'zono': Zono.from_bounds(lb, ub)}[set_kind]
        kwargs['input_shape'] = (2, 2, 2)
    else:
        lo_img, hi_img = np.zeros((2, 2, 2)), np.ones((2, 2, 2))
        cls = ImageStar if set_kind == 'imagestar' else ImageZono
        S = cls.from_bounds(lo_img, hi_img, 2, 2, 2)
    try:
        out = reach_pytorch_model(model, S, method='approx', **kwargs)
    except NotImplementedError:
        # an image set cannot hold an (H, W) constant; it must not guess
        assert set_kind.startswith('image') and b_shape == (1, 1, 2, 2)
        return
    b = np.broadcast_to(B, (1, 2, 2, 2)).ravel().astype(np.float64)
    lo, hi = _output_bounds(out[0])
    np.testing.assert_allclose(lo, b - 1.0, atol=1e-6)
    np.testing.assert_allclose(hi, b, atol=1e-6)


# ---------------------------------------------------------------------------
# McCormick Mul/Div pad the LP operand ranges
# ---------------------------------------------------------------------------

_UNDERSHOOT = 5e-10     # measured HiGHS max undershoot on thin wedges


def _install_undershooting_lp(monkeypatch):
    """Every LP the Star solves stops 5e-10 inside its true optimum.

    Installed on ``Star._solve_lp`` and ``Star._solve_lp_batch``, the two
    entry points every range query goes through (``get_range``, ``get_max``,
    ``get_ranges`` on either the batch or the parallel path). An earlier
    version stubbed ``Star.get_range`` only, which the McCormick operand
    ranges never call on the default configuration: the stub was dead and the
    test passed with or without the pad. Returns the list of calls made.
    """
    calls = []
    orig_single, orig_batch = Star._solve_lp, Star._solve_lp_batch

    def shrink(v, f, minimize):
        if v is None or not np.any(np.asarray(f) != 0):
            return v                       # a constant row is exact
        return v + _UNDERSHOOT if minimize else v - _UNDERSHOOT

    def single(self, f, minimize=True, lp_solver='default'):
        calls.append('single')
        return shrink(orig_single(self, f, minimize, lp_solver), f, minimize)

    def batch(self, objectives, minimize_flags, lp_solver='default'):
        calls.append('batch')
        vals = orig_batch(self, objectives, minimize_flags, lp_solver)
        return [shrink(v, f, m) for v, f, m in zip(vals, objectives, minimize_flags)]

    monkeypatch.setattr(Star, '_solve_lp', single)
    monkeypatch.setattr(Star, '_solve_lp_batch', batch)
    return calls


@pytest.mark.parametrize('op', ['mul', 'div'])
def test_mccormick_survives_lp_undershoot(monkeypatch, op):
    # x = -1 + 1e-4 + a, a in [-1, 1] (true max 1e-4); y a constant set
    # (1e4 for Mul, 1e-4 for Div), so the true max of z is 1.0. With the
    # LP max 5e-10 low, the unpadded product box stopped at 0.999995 and
    # z >= 0.9999975 was reported UNSAT.
    lb, ub = np.array([[-1.0]]), np.array([[1.0]])
    C, d = np.zeros((0, 1)), np.zeros((0, 1))
    x = Star(np.array([[-1.0 + 1e-4, 1.0]]), C, d, lb, ub)
    k = 1e4 if op == 'mul' else 1e-4
    y = Star(np.array([[k, 0.0]]), C, d, lb, ub)
    calls = _install_undershooting_lp(monkeypatch)
    Z = (_mul_sets if op == 'mul' else _div_sets)([x], [y])[0]
    monkeypatch.undo()
    assert calls, 'the operand ranges did not go through the stubbed LP'
    assert Z.predicate_ub[-1, 0] >= 1.0
    spec = HalfSpace(np.array([[-1.0]]), np.array([[-0.9999975]]))
    assert verify_specification([Z], spec).verdict != 'UNSAT'


def test_mccormick_constant_operand_rows_stay_exact():
    # A dimension with no generator coefficients is exact; it is not padded.
    lb, ub = np.array([[-1.0]]), np.array([[1.0]])
    C, d = np.zeros((0, 1)), np.zeros((0, 1))
    x = Star(np.array([[2.0, 0.0]]), C, d, lb, ub)
    y = Star(np.array([[3.0, 0.0]]), C, d, lb, ub)
    Z = _mul_sets([x], [y])[0]
    lo, hi = Z.get_range(0)
    assert lo == pytest.approx(6.0, abs=1e-12)
    assert hi == pytest.approx(6.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Finitization LP solver and shape propagation dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('use_config', [False, True])
def test_finitization_uses_callers_lp_solver(monkeypatch, use_config):
    seen = []
    orig = predicate_box.ensure_finite_predicate_box

    def spy(star, lp_solver=None):
        seen.append(lp_solver)
        return orig(star, lp_solver=lp_solver)

    monkeypatch.setattr(predicate_box, 'ensure_finite_predicate_box', spy)
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    S = Star.from_bounds(-np.ones(2), np.ones(2))
    if use_config:
        reach_pytorch_model(model, S, method='approx',
                            config=ReachConfig(method='approx',
                                               lp_solver='linprog'))
    else:
        reach_pytorch_model(model, S, method='approx', lp_solver='linprog')
    assert seen == ['linprog']


def test_shape_propagation_matches_float64_models(capsys):
    model = torch.fx.symbolic_trace(
        nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double())
    shapes = _propagate_shapes(model, [Star.from_bounds(np.zeros(3),
                                                        np.ones(3))])
    assert shapes, "no shapes propagated for a float64 model"
    assert 'Traceback' not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Re-encoded outputs get fresh predicate variables
# ---------------------------------------------------------------------------

# Each case: nodes, model input/output shape, the batch-stripped input
# shape for the reach, an input x, and a spec (G, g) whose unsafe region
# G y <= g holds the model output at x.
_REBOX_CASES = {
    # Softmax over a (1, 2) row: x = (1, 1) gives y = -0.5.
    'softmax': (
        [helper.make_node('Softmax', ['X'], ['S'], axis=1),
         helper.make_node('Sub', ['S', 'X'], ['Y'])],
        [1, 2], (2,), [[1.0, 1.0]], [[1.0, 0.0]], -0.4),
    # The attention softmax: each row of a (1, 2, 2) tensor.
    'softmax_rows': (
        [helper.make_node('Softmax', ['X'], ['S'], axis=2),
         helper.make_node('Sub', ['S', 'X'], ['Y'])],
        [1, 2, 2], (2, 2), [[[1.0, 1.0], [1.0, 1.0]]], [[1.0, 0.0, 0.0, 0.0]],
        -0.4),
    # Set-by-set MatMul, re-encoded from interval bounds: Y = X @ X - X,
    # y00 = x00^2 + x01 x10 - x00 = 3 at x00 = -1, x01 = x10 = 1.
    'bilinear': (
        [helper.make_node('MatMul', ['X', 'X'], ['P']),
         helper.make_node('Sub', ['P', 'X'], ['Y'])],
        [2, 2], (2, 2), [[-1.0, 1.0], [1.0, 0.0]], [[-1.0, 0.0, 0.0, 0.0]],
        -2.5),
}


@pytest.mark.parametrize('case', sorted(_REBOX_CASES))
def test_reboxed_output_does_not_share_the_input_variables(tmp_path, case):
    """Softmax, the attention softmax and set-by-set MatMul re-encode their
    output from bounds. With Star.from_bounds the predicate system of that
    output (alpha in [-1, 1]^k, rows +-I) was identical to the input
    star's, so Sub(op(X), X) was computed as if both used the same
    variables: the specs below came out UNSAT though the model output at x
    is in the unsafe region (softmax: y0 = -0.5 <= -0.4; bilinear:
    y00 = 3 >= 2.5)."""
    nodes, shape, input_shape, x, G, g = _REBOX_CASES[case]
    n = int(np.prod(shape))
    model = _onnx_model(tmp_path, nodes, n, n, {}, name=case,
                        in_shape=shape, out_shape=shape)
    S = Star.from_bounds(-np.ones(n), np.ones(n))
    out = reach_pytorch_model(model, S, method='approx',
                              input_shape=input_shape)
    with torch.no_grad():
        y = model(torch.tensor(x, dtype=torch.float32)).numpy()
    y = y.astype(np.float64).ravel()
    spec = HalfSpace(np.array(G), np.array([[g]]))
    assert (np.array(G) @ y).item() <= g
    assert _covered(out, y)
    assert verify_specification(out, spec).verdict != 'UNSAT'


def test_fresh_from_bounds_systems_are_distinct():
    a = Star.fresh_from_bounds(-np.ones(2), np.ones(2))
    b = Star.fresh_from_bounds(-np.ones(2), np.ones(2))
    plain = Star.from_bounds(-np.ones(2), np.ones(2))
    from n2v.nn.reach import _same_predicate_system
    assert not _same_predicate_system(a, b)
    assert not _same_predicate_system(a, plain)
    lo, hi = a.estimate_ranges()
    np.testing.assert_array_equal(lo.ravel(), [-1.0, -1.0])
    np.testing.assert_array_equal(hi.ravel(), [1.0, 1.0])


# ---------------------------------------------------------------------------
# Predicate variables appended by different nodes are never shared
# ---------------------------------------------------------------------------

_HALF = np.array([[0.5]], dtype=np.float32)

# Each case: nodes, initializers, input range, an input x, and a spec
# (G, g) whose unsafe region G y <= g holds the model output at x. In each,
# two nodes append predicate variables with identical rows and bounds for
# different functions, and the Sub or Concat read them as one variable
# (reach [0, 0], verdict UNSAT).
_APPEND_CASES = {
    # Sign across 0 appends a box-only variable in [-1, 1] on both
    # branches: y = 2 at x = 0.25.
    'sign': (
        [helper.make_node('Sign', ['X'], ['A']),
         helper.make_node('Sub', ['X', 'h'], ['Xs']),
         helper.make_node('Sign', ['Xs'], ['B']),
         helper.make_node('Sub', ['A', 'B'], ['Y'])],
        {'h': _HALF}, (-1.0, 1.0), 0.25, -1.0, -1.0),
    # An odd Pow across 0 appends a box-only variable in [-1, 1]:
    # y = 2 x^3 = 2 at x = 1.
    'pow': (
        [helper.make_node('Pow', ['X', 'three'], ['A']),
         helper.make_node('Neg', ['X'], ['Xn']),
         helper.make_node('Pow', ['Xn', 'three'], ['B']),
         helper.make_node('Sub', ['A', 'B'], ['Y'])],
        {'three': np.array(3.0, dtype=np.float32)}, (-1.0, 1.0), 1.0,
        -1.0, -1.0),
    # Sin and Cos over a range holding an extremum: box-only in [-1, 1];
    # y = sqrt(2) at x = 3 pi / 4.
    'sin_cos': (
        [helper.make_node('Sin', ['X'], ['A']),
         helper.make_node('Cos', ['X'], ['B']),
         helper.make_node('Sub', ['A', 'B'], ['Y'])],
        {}, (-4.0, 4.0), 0.75 * np.pi, -1.0, -1.0),
    # The same through Concat (a block join) and a MatMul.
    'sin_cos_concat': (
        [helper.make_node('Sin', ['X'], ['A']),
         helper.make_node('Cos', ['X'], ['B']),
         helper.make_node('Concat', ['A', 'B'], ['AB'], axis=1),
         helper.make_node('MatMul', ['AB', 'W'], ['Y'])],
        {'W': np.array([[1.0], [-1.0]], dtype=np.float32)}, (-4.0, 4.0),
        0.75 * np.pi, -1.0, -1.0),
    # Round(x) and Floor(x + 1/2) both give z - x in [-1/2, 1/2], z in
    # [0, 1], but differ at the tie: y = 0 - 1 = -1 at x = 0.5 (Round is
    # half to even).
    'round_floor': (
        [helper.make_node('Round', ['X'], ['A']),
         helper.make_node('Add', ['X', 'h'], ['Xh']),
         helper.make_node('Floor', ['Xh'], ['B']),
         helper.make_node('Sub', ['A', 'B'], ['Y'])],
        {'h': _HALF}, (0.4, 0.6), 0.5, 1.0, -0.5),
}


@pytest.mark.parametrize('method', ['approx', 'exact'])
@pytest.mark.parametrize('case', sorted(_APPEND_CASES))
def test_predicates_of_two_nodes_are_not_shared(tmp_path, case, method):
    """The pairing ops read identical predicate systems as the same
    variables; the executor now tags the new variables of each node."""
    nodes, inits, (lo, hi), x, G, g = _APPEND_CASES[case]
    model = _onnx_model(tmp_path, nodes, 1, 1, inits, name=case)
    with torch.no_grad():
        y = model(torch.tensor([[x]], dtype=torch.float32)).numpy()
    y = y.astype(np.float64).ravel()
    assert G * y[0] <= g
    out = reach_pytorch_model(
        model, Star.from_bounds(np.array([lo]), np.array([hi])),
        method=method)
    assert _covered(out, y)
    spec = HalfSpace(np.array([[G]]), np.array([[g]]))
    assert verify_specification(out, spec).verdict != 'UNSAT'


def test_graph_with_two_inputs_is_refused(tmp_path):
    """Every placeholder was bound to the one input set, so Y = X1 - X2 was
    reached as a single point (y0 >= 0.5 UNSAT, though X1 = (1, .),
    X2 = (-1, .) gives y0 = 2)."""
    graph = helper.make_graph(
        [helper.make_node('Sub', ['X1', 'X2'], ['Y'])], 'two',
        [helper.make_tensor_value_info('X1', TensorProto.FLOAT, [1, 2]),
         helper.make_tensor_value_info('X2', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])])
    onnx_model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)])
    onnx_model.ir_version = 10
    path = str(tmp_path / 'two.onnx')
    onnx.save(onnx_model, path)
    model = load_onnx(path)
    with pytest.raises(NotImplementedError, match='2 inputs'):
        reach_pytorch_model(model, Star.from_bounds(-np.ones(2), np.ones(2)),
                            method='approx')


@pytest.mark.parametrize('first', ['Softmax', 'Sigmoid'])
def test_descendants_of_new_predicates_still_pair(tmp_path, caplog, first):
    """S = first(X) brings new predicate variables (Softmax re-encodes its
    output; Sigmoid appends relaxation variables and the executor tags
    them). A = Relu(S - 0.5) and B = Relu(A - 0.1) split in exact mode, and
    each piece of B must still meet only the piece of A it came from in
    Y = A + B, with the true outputs covered."""
    c5 = np.full((1, 3), 0.5, dtype=np.float32)
    c1 = np.full((1, 3), 0.1, dtype=np.float32)
    model = _onnx_model(
        tmp_path,
        [helper.make_node(first, ['X'], ['S'],
                          **({'axis': 1} if first == 'Softmax' else {})),
         helper.make_node('Sub', ['S', 'c5'], ['S5']),
         helper.make_node('Relu', ['S5'], ['A']),
         helper.make_node('Sub', ['A', 'c1'], ['A1']),
         helper.make_node('Relu', ['A1'], ['B']),
         helper.make_node('Add', ['A', 'B'], ['Y'])],
        3, 3, {'c5': c5, 'c1': c1}, name=first.lower())
    with caplog.at_level('WARNING', logger='n2v.nn.reach'):
        out = reach_pytorch_model(
            model, Star.from_bounds(-2 * np.ones(3), 2 * np.ones(3)),
            method='exact')
    assert not [r for r in caplog.records
                if 'do not descend' in r.getMessage()]
    xs = np.random.default_rng(0).uniform(-2, 2, size=(40, 3))
    with torch.no_grad():
        ys = model(torch.tensor(xs, dtype=torch.float32)).numpy()
    for y in ys.astype(np.float64):
        assert _covered(out, y, tol=1e-5), y


def test_only_graphs_that_join_branches_tag_new_predicates():
    """Tags cost a basis column per node, and only a node that takes the
    sets of two different nodes compares predicate systems."""
    from n2v.nn.reach import _constant_node_names, _graph_joins_branches

    class _Flat(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return torch.relu(self.fc(x.view(x.size(0), -1)))

    for module, joins in [
            (nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2)),
             False),
            (_Flat(), False),
            (_ResAdd(), True),
            (_TorchCat(), True)]:
        gm = torch.fx.symbolic_trace(module)
        assert _graph_joins_branches(gm, _constant_node_names(gm)) is joins

    # Without a join the reach output keeps its untagged variables: one
    # per ReLU neuron that crosses 0 (both do here).
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU()).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
        model[0].bias.zero_()
    out = reach_pytorch_model(model, Star.from_bounds(-np.ones(2),
                                                      np.ones(2)),
                              method='approx')
    assert [s.nVar for s in out] == [4]


# ---------------------------------------------------------------------------
# Zonotope bound pre-pass through image Reduce / Slice
# ---------------------------------------------------------------------------

def _chw_image_star(lb_chw, ub_chw):
    C, H, W = lb_chw.shape
    return ImageStar.from_bounds(lb_chw.transpose(1, 2, 0),
                                 ub_chw.transpose(1, 2, 0),
                                 height=H, width=W, num_channels=C)


@pytest.mark.parametrize('method', ['approx', 'exact'])
@pytest.mark.parametrize('op', ['reduce', 'slice'])
def test_zono_prepass_image_reduce_and_slice_no_false_unsat(tmp_path, op,
                                                            method):
    """Channel 0 in [1, 2], the other channels in [-5, -4]. The Relu input is
    channel 0 (the H/W mean, or a slice of it), so y0 = 2 is reachable. The
    pre-pass gave that neuron channel-1 bounds and the verdict for y0 >= 1.5
    was UNSAT with precompute_bounds='zono' (UNKNOWN without it)."""
    if op == 'reduce':
        C, H, W = 2, 2, 2
        nodes = [helper.make_node('ReduceMean', ['X'], ['P'], axes=[2, 3],
                                  keepdims=0)]
        inits, n_out = {}, 2
    else:
        C, H, W = 3, 4, 4
        nodes = [helper.make_node('Slice', ['X', 'st', 'en', 'ax'], ['S']),
                 helper.make_node('Flatten', ['S'], ['P'])]
        inits = {'st': np.array([0, 1, 2], dtype=np.int64),
                 'en': np.array([1, 3, 3], dtype=np.int64),
                 'ax': np.array([1, 2, 3], dtype=np.int64)}
        n_out = 2
    model = _onnx_model(
        tmp_path, nodes + [helper.make_node('Relu', ['P'], ['Y'])],
        C * H * W, n_out, inits, name=f'zp_{op}', in_shape=[1, C, H, W])
    lb = np.full((C, H, W), -5.0)
    lb[0] = 1.0
    S = _chw_image_star(lb, lb + 1.0)
    out = reach_pytorch_model(model, S, method=method,
                              precompute_bounds='zono', input_shape=(C, H, W))
    y = model(torch.tensor((lb + 1.0)[None], dtype=torch.float32))
    y = y.detach().numpy().ravel()
    assert y[0] == 2.0
    assert _covered(out, y)
    G = np.zeros((1, n_out))
    G[0, 0] = -1.0
    spec = HalfSpace(G, np.array([[-1.5]]))           # unsafe: y0 >= 1.5
    assert verify_specification(out, spec).verdict != 'UNSAT'


_IMAGE_OPS = {
    'slice_c': ([helper.make_node('Slice', ['A', 's', 'e', 'x'], ['P'])],
                {'s': [1], 'e': [3], 'x': [1]}),
    'slice_h': ([helper.make_node('Slice', ['A', 's', 'e', 'x'], ['P'])],
                {'s': [1], 'e': [3], 'x': [2]}),
    'slice_w': ([helper.make_node('Slice', ['A', 's', 'e', 'x'], ['P'])],
                {'s': [0], 'e': [2], 'x': [3]}),
    'slice_hw_step': ([helper.make_node('Slice', ['A', 's', 'e', 'x', 'p'],
                                        ['P'])],
                      {'s': [0, 1], 'e': [4, 5], 'x': [2, 3], 'p': [2, 3]}),
    'reducemean_hw': ([helper.make_node('ReduceMean', ['A'], ['P'],
                                        axes=[2, 3], keepdims=1)], {}),
    'reducemean_w': ([helper.make_node('ReduceMean', ['A'], ['P'],
                                       axes=[3], keepdims=1)], {}),
    'reducesum_c': ([helper.make_node('ReduceSum', ['A', 'r'], ['P'],
                                      keepdims=1)], {'r': [1]}),
    'sub_channel_mean': ([helper.make_node('ReduceMean', ['A'], ['M'],
                                           axes=[1], keepdims=1),
                          helper.make_node('Sub', ['A', 'M'], ['P'])], {}),
}


@pytest.mark.parametrize('op', sorted(_IMAGE_OPS))
def test_zono_prepass_bounds_match_the_reach_layout(tmp_path, op):
    """Conv(1x1) -> op -> Relu on an image box. Every op here is linear, so
    the reach's own set entering the Relu is exact, and the pre-pass bounds
    for that node must have its size and enclose its range row by row (the
    Relu consumer indexes them in the reach's row order)."""
    from n2v.nn.reach import _handle_graphmodule
    from n2v.utils.bounds_precomputation import compute_intermediate_bounds
    C, H, W = 3, 4, 5
    rng = np.random.default_rng(0)
    nodes, ints = _IMAGE_OPS[op]
    inits = {'cw': rng.normal(size=(4, C, 1, 1)).astype(np.float32),
             'cb': rng.normal(size=4).astype(np.float32)}
    inits.update({k: np.array(v, dtype=np.int64) for k, v in ints.items()})
    graph = helper.make_graph(
        [helper.make_node('Conv', ['X', 'cw', 'cb'], ['A'])] + nodes
        + [helper.make_node('Relu', ['P'], ['Y'])], op,
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT,
                                       [1, 'c', 'h', 'w'])],
        initializer=[numpy_helper.from_array(a, k) for k, a in inits.items()])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 10
    path = str(tmp_path / f'{op}.onnx')
    onnx.save(m, path)
    model = load_onnx(path)
    lb = rng.uniform(-1.0, 0.5, (C, H, W))
    S = _chw_image_star(lb, lb + rng.uniform(0.05, 0.5, (C, H, W)))

    seen = {}

    def record(name, module, sets):
        if isinstance(module, nn.ReLU):
            seen[name] = sets[0].estimate_ranges()

    _handle_graphmodule(model, [S], 'approx', input_shape=(C, H, W),
                        _on_layer_input=record)
    pre = compute_intermediate_bounds(model, S, method='zono',
                                      input_shape=(C, H, W))
    assert seen and set(seen) == set(pre)
    for name, (lo, hi) in seen.items():
        lo, hi = np.ravel(lo), np.ravel(hi)
        plo, phi = np.ravel(pre[name][0]), np.ravel(pre[name][1])
        assert plo.size == lo.size
        tol = 1e-9 * (1.0 + np.abs(lo) + np.abs(hi))
        assert np.all(plo <= lo + tol) and np.all(phi >= hi - tol)

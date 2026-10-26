"""Regressions for ReduceSum / ReduceMean reach on image sets
(n2v/nn/layer_ops/reduce_reach.py).

- A partial reduction with keepdims=0 on an ImageStar / ImageZono flattened
  the remaining axes in the set's (H, W, C) order, not the tensor's
  row-major (C, H, W) order, so the output coordinates were permuted:
  ReduceMean over H of arange(12) as a (1, 2, 2, 3) input gave
  [1.5 7.5 2.5 8.5 3.5 9.5] for torch's [1.5 2.5 3.5 7.5 8.5 9.5], and a
  verdict of UNSAT for a reachable output.
- Negative axes were mapped onto the wrong (H, W, C) axis (axis -1, the
  width, reduced the channels of an ImageStar), and an ImageZono reduced its
  center and its generators over different axes.
- Reducing only the batch axis reduced every axis, a ReduceSum with empty
  axes and noop_with_empty_axes=1 reduced every axis instead of passing its
  input through, and a full reduction with keepdims=0 on an ImageStar built
  a malformed Star.
- ReduceMax, ReduceMin, ReduceProd, ReduceL1, ReduceL2, ReduceLogSum,
  ReduceLogSumExp and ReduceSumSquare were reached as a ReduceSum. They are
  now refused, except ReduceMax, ReduceMin, ReduceProd and ReduceLogSumExp
  over size-1 axes of an image set, which are the identity.
- The ImageZono reduce joined its center and generators into one array,
  copying the whole generator matrix (widened to float64).

The checks evaluate each output set at sampled predicate points and compare
it with torch on the matching input point, so a permuted coordinate, a wrong
axis or a wrong operation shows up as a mismatch.

Convention: SAT = unsafe (the unsafe region is ``G y <= g``), UNSAT = safe.
"""

import itertools

import numpy as np
import pytest
import torch

import onnx
reduce_mod = pytest.importorskip('onnx2torch.node_converters.reduce')

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from n2v.nn import NeuralNetwork  # noqa: E402
from n2v.nn.layer_ops.dispatcher import reach_layer  # noqa: E402
from n2v.sets import Box, HalfSpace, Star, Zono  # noqa: E402
from n2v.sets.image_star import ImageStar  # noqa: E402
from n2v.sets.image_zono import ImageZono  # noqa: E402
from n2v.utils.model_loader import load_onnx  # noqa: E402
from n2v.utils.verify_specification import verify_specification  # noqa: E402

OnnxReduceStaticAxes = reduce_mod.OnnxReduceStaticAxes
OnnxReduceSumStaticAxes = reduce_mod.OnnxReduceSumStaticAxes

C, H, W = 2, 3, 4
_rng = np.random.default_rng(3)
LB = _rng.uniform(-3.0, 3.0, size=(C, H, W))
UB = LB + _rng.uniform(0.1, 2.0, size=(C, H, W))

# Every subset of the (N, C, H, W) axes, spelled with positive and with
# negative indices, plus the attribute left out.
AXES = [None] + [list(s) for r in range(1, 5)
                 for s in itertools.combinations(range(4), r)]
AXES += [[a - 4 for a in s] for s in AXES[1:]]


def _image_star():
    return ImageStar.from_bounds(LB.transpose(1, 2, 0), UB.transpose(1, 2, 0),
                                 height=H, width=W, num_channels=C)


def _image_zono():
    return ImageZono.from_bounds(LB.transpose(1, 2, 0), UB.transpose(1, 2, 0),
                                 height=H, width=W, num_channels=C)


def _at(S, p):
    """The point of S at predicate point ``p``, flat in row-major tensor
    order ((C, H, W) for the image sets, whose rows are (H, W, C))."""
    if isinstance(S, ImageStar):
        return (S.V[..., 0] + S.V[..., 1:] @ p).transpose(2, 0, 1).ravel()
    if isinstance(S, Star):
        return S.V[:, 0] + S.V[:, 1:] @ p
    v = S.c[:, 0] + S.V @ p
    if isinstance(S, ImageZono):
        return v.reshape(S.height, S.width, S.num_channels).transpose(
            2, 0, 1).ravel()
    return v


def _predicate_points(S, n=12):
    if isinstance(S, Zono):
        return [_rng.uniform(-1.0, 1.0, S.V.shape[1]) for _ in range(n)]
    lo, hi = S.predicate_lb.ravel(), S.predicate_ub.ravel()
    return [lo, hi] + [_rng.uniform(lo, hi) for _ in range(n)]


def _assert_matches(out, S_in, f):
    """``out`` is exactly ``f`` applied to ``S_in``: equal at sampled
    predicate points (the reduction does not add predicate variables)."""
    for p in _predicate_points(S_in):
        x = _at(S_in, p).reshape(C, H, W)
        want, got = f(x), _at(out, p)
        assert got.shape == want.shape, (got.shape, want.shape)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def _torch_reduce(op, axes, keepdims):
    fn = torch.mean if op == 'ReduceMean' else torch.sum
    dims = list(range(4)) if axes is None else list(axes)

    def f(x):
        t = torch.tensor(x[None], dtype=torch.float64)
        return fn(t, dim=dims, keepdim=bool(keepdims)).numpy().ravel()
    return f


def _layer(op, axes, keepdims):
    if op == 'ReduceSum':
        return OnnxReduceSumStaticAxes(axes=axes, keepdims=keepdims)
    return OnnxReduceStaticAxes(op, axes=axes, keepdims=keepdims)


# ---------------------------------------------------------------------------
# Layer level (the dispatcher path, which image sets always take)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('keepdims', [0, 1])
@pytest.mark.parametrize('axes', AXES, ids=str)
@pytest.mark.parametrize('op', ['ReduceMean', 'ReduceSum'])
def test_image_reduce_layer_matches_torch(op, axes, keepdims):
    f = _torch_reduce(op, axes, keepdims)
    for S in (_image_star(), _image_zono()):
        out = reach_layer(_layer(op, axes, keepdims), [S], 'approx')
        assert len(out) == 1
        if keepdims:
            assert isinstance(out[0], type(S))
        else:
            assert type(out[0]) in (Star, Zono)
        _assert_matches(out[0], S, f)


def test_reduce_sum_noop_with_empty_axes_passes_input_through():
    layer = OnnxReduceSumStaticAxes(axes=[], keepdims=0,
                                    noop_with_empty_axes=1)
    for S in (_image_star(), _image_zono(),
              Star.from_bounds(LB.ravel(), UB.ravel()),
              Box(LB.reshape(-1, 1), UB.reshape(-1, 1))):
        out = reach_layer(layer, [S], 'approx')[0]
        if isinstance(S, Box):
            np.testing.assert_array_equal(out.lb, S.lb)
            np.testing.assert_array_equal(out.ub, S.ub)
        else:
            _assert_matches(out, S, lambda x: x.ravel())


@pytest.mark.parametrize('keepdims', [0, 1])
def test_flat_reduce_over_batch_axis_only_is_identity(keepdims):
    """ReduceMean over axis 0 of a (1, N) tensor leaves its N values; the
    flat path reduced all of them."""
    lb, ub = LB.ravel()[:5], UB.ravel()[:5]
    layer = OnnxReduceStaticAxes('ReduceMean', axes=[0], keepdims=keepdims)
    S = Star.from_bounds(lb, ub)
    out = reach_layer(layer, [S], 'approx')[0]
    assert out.dim == 5
    for p in _predicate_points(S):
        np.testing.assert_array_equal(_at(out, p), _at(S, p))
    B = reach_layer(layer, [Box(lb.reshape(-1, 1), ub.reshape(-1, 1))],
                    'approx')[0]
    np.testing.assert_array_equal(B.lb.ravel(), lb)
    np.testing.assert_array_equal(B.ub.ravel(), ub)


NON_AFFINE = ['ReduceMax', 'ReduceMin', 'ReduceProd', 'ReduceL1', 'ReduceL2',
              'ReduceLogSum', 'ReduceLogSumExp', 'ReduceSumSquare']


@pytest.mark.parametrize('op', NON_AFFINE)
def test_non_affine_reduce_is_refused(op):
    flat = (LB.ravel(), UB.ravel())
    sets = [_image_star(), _image_zono(), Star.from_bounds(*flat),
            Zono.from_bounds(*flat),
            Box(flat[0].reshape(-1, 1), flat[1].reshape(-1, 1))]
    for axes in ([2, 3], [1]):
        layer = OnnxReduceStaticAxes(op, axes=axes, keepdims=0)
        for S in sets:
            with pytest.raises(NotImplementedError):
                reach_layer(layer, [S], 'approx')


# ---------------------------------------------------------------------------
# End to end through an ONNX graph
# ---------------------------------------------------------------------------

def _reduce_model(tmp_path, op, axes, keepdims, tail='Reshape'):
    """X (1, C, H, W) -> Reduce -> Reshape to (1, n), or -> Flatten."""
    kept = [s for i, s in enumerate((1, C, H, W))
            if axes is not None and i not in {a % 4 for a in axes}]
    n_out = int(np.prod(kept))
    inits = [numpy_helper.from_array(np.array([1, -1], np.int64), 'shp')]
    ins, attrs = ['X'], {'keepdims': keepdims}
    if op == 'ReduceSum':
        inits.append(numpy_helper.from_array(np.array(axes, np.int64), 'ax'))
        ins.append('ax')
    elif axes is not None:
        attrs['axes'] = list(axes)
    nodes = [helper.make_node(op, ins, ['R'], **attrs)]
    if tail == 'Reshape':
        nodes.append(helper.make_node('Reshape', ['R', 'shp'], ['Y']))
    else:
        nodes.append(helper.make_node('Flatten', ['R'], ['Y']))
    graph = helper.make_graph(
        nodes, 'reduce',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, C, H, W])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, n_out])],
        initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    path = str(tmp_path / 'reduce.onnx')
    onnx.save(model, path)
    return load_onnx(path)


def _model_fn(gm):
    def f(x):
        with torch.no_grad():
            return gm(torch.tensor(x[None], dtype=torch.float64)).numpy().ravel()
    return f


# A ReduceSum without an axes input converts to a dynamic-axes module, which
# the reach does not take; the model cases give ReduceSum explicit axes.
E2E = ([('ReduceMean', a) for a in AXES]
       + [('ReduceSum', a) for a in AXES if a is not None])


@pytest.mark.parametrize('keepdims', [0, 1])
@pytest.mark.parametrize('op,axes', E2E, ids=str)
def test_reduce_model_reach_matches_torch(tmp_path, op, axes, keepdims):
    gm = _reduce_model(tmp_path, op, axes, keepdims)
    f = _model_fn(gm)
    net = NeuralNetwork(gm)
    samples = [LB, UB] + [_rng.uniform(LB, UB) for _ in range(40)]
    ys = np.array([f(x) for x in samples])

    for S, kw in ((_image_star(), dict(precompute_bounds=False)),
                  (_image_star(), dict(precompute_bounds='zono')),
                  (_image_zono(), {})):
        out = net.reach(S, method='approx', input_shape=(C, H, W), **kw)
        assert len(out) == 1
        _assert_matches(out[0], S, f)
        if isinstance(out[0], Star) and kw['precompute_bounds'] is False:
            lo, hi = (np.ravel(v) for v in out[0].get_ranges())
            assert np.all(ys >= lo - 1e-9) and np.all(ys <= hi + 1e-9)

    B = net.reach(Box(LB.reshape(-1, 1), UB.reshape(-1, 1)), method='approx',
                  input_shape=(C, H, W))[0]
    assert np.all(ys >= B.lb.ravel()) and np.all(ys <= B.ub.ravel())


def test_partial_reduce_keepdims0_no_false_unsat(tmp_path):
    """The reviewer's case: ReduceMean over H with keepdims=0, then Flatten,
    on a point input. The unsafe region is a small box around the torch
    output, so the verdict must not be UNSAT; the permuted rows made it
    UNSAT."""
    gm = _reduce_model(tmp_path, 'ReduceMean', [2], 0, tail='Flatten')
    x = np.arange(C * H * W, dtype=np.float64).reshape(C, H, W)
    y = _model_fn(gm)(x)
    S = ImageStar.from_bounds(x.transpose(1, 2, 0), x.transpose(1, 2, 0),
                              height=H, width=W, num_channels=C)
    n = y.size
    G = np.vstack([np.eye(n), -np.eye(n)])
    g = np.concatenate([y + 1e-3, -(y - 1e-3)]).reshape(-1, 1)
    for pc in (False, 'zono'):
        out = NeuralNetwork(gm).reach(S, method='approx', precompute_bounds=pc,
                                      input_shape=(C, H, W))
        assert verify_specification(out, HalfSpace(G, g)).verdict != 'UNSAT'


# ---------------------------------------------------------------------------
# ReduceMax / ReduceMin / ReduceProd / ReduceLogSumExp over size-1 axes
# ---------------------------------------------------------------------------

# (C, H, W) shape of the input, and axes whose (C, H, W) sizes are all 1:
# a single-channel map, a (C, 1, 1) map after a global average pool, a
# 1x1x1 map, and the batch axis alone.
SIZE_ONE_CASES = [((1, 3, 4), [1]), ((1, 3, 4), [-3]), ((1, 3, 4), [0, 1]),
                  ((2, 3, 4), [0]), ((2, 1, 1), [2, 3]),
                  ((2, 1, 1), [-2, -1]), ((1, 1, 1), None)]


def _image_sets(shape):
    c, h, w = shape
    lb = _rng.uniform(-2.0, 2.0, size=(h, w, c))
    ub = lb + 0.5
    return [ImageStar.from_bounds(lb, ub, height=h, width=w, num_channels=c),
            ImageZono.from_bounds(lb, ub, height=h, width=w, num_channels=c)]


@pytest.mark.parametrize('keepdims', [0, 1])
@pytest.mark.parametrize('shape,axes', SIZE_ONE_CASES, ids=str)
@pytest.mark.parametrize('op', ['ReduceMax', 'ReduceMin', 'ReduceProd',
                                'ReduceLogSumExp'])
def test_identity_non_affine_reduce_over_size_one_axes(op, shape, axes,
                                                       keepdims):
    """Over axes of size 1 these reductions return their input, so the
    reach is exact; refusing every non-affine reduce had refused these."""
    for S in _image_sets(shape):
        out = reach_layer(OnnxReduceStaticAxes(op, axes=axes,
                                               keepdims=keepdims),
                          [S], 'approx')[0]
        if keepdims:
            assert isinstance(out, type(S))
        # The oracle takes the positive spelling of the axes: onnx2torch's
        # ReduceMax/Min/Prod with keepdims=0 shifts negative axes as it
        # removes them, so [-2, -1] reduces C and H (onnxruntime reduces H, W)
        pos_axes = None if axes is None else [a % 4 for a in axes]
        for p in _predicate_points(S):
            x = torch.tensor(_at(S, p).reshape(shape)[None])
            layer = OnnxReduceStaticAxes(op, axes=pos_axes, keepdims=keepdims)
            want = layer(x).numpy().ravel()
            np.testing.assert_allclose(_at(out, p), want,
                                       rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('op', ['ReduceL1', 'ReduceL2', 'ReduceLogSum',
                                'ReduceSumSquare'])
def test_non_identity_reduce_over_size_one_axes_is_refused(op):
    """|x|, sqrt(x^2), log x and x^2 are not the identity, even over one
    element."""
    for S in _image_sets((1, 3, 4)):
        for kd in (0, 1):
            with pytest.raises(NotImplementedError):
                reach_layer(OnnxReduceStaticAxes(op, axes=[1], keepdims=kd),
                            [S], 'approx')


def test_image_zono_reduce_does_not_copy_generators():
    """The center and the generators are reduced separately: the output
    keeps the generators' dtype and is contiguous, and no copy of the input
    generator matrix is made (joining [c | V] copied it, widened to the
    center's float64)."""
    import tracemalloc
    h, w, c, n_gen = 8, 8, 4, 1024
    Z = ImageZono(_rng.normal(size=(h * w * c, 1)),
                  _rng.normal(size=(h * w * c, n_gen)).astype(np.float32),
                  h, w, c)
    for kd in (0, 1):
        layer = OnnxReduceStaticAxes('ReduceMean', axes=[2, 3], keepdims=kd)
        tracemalloc.start()
        try:
            out = reach_layer(layer, [Z], 'approx')[0]
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < Z.V.nbytes / 2, (peak, Z.V.nbytes)
        assert out.V.dtype == np.float32 and out.V.flags['C_CONTIGUOUS']
        np.testing.assert_allclose(
            out.V, Z.V.reshape(h, w, c, n_gen).mean(axis=(0, 1)),
            rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            out.c.ravel(), Z.c.reshape(h, w, c).mean(axis=(0, 1)),
            rtol=1e-12, atol=1e-12)

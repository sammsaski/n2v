"""
Regression tests for the IBP pre-pass and the conv / conv-transpose /
avgpool set arithmetic.

Each test is derived from a false-UNSAT repro:

- The IBP pre-pass treated OnnxTranspose as the identity, so bounds
  recorded after a permuting Transpose belonged to the wrong neurons.
- A module fed by a node the IBP pre-pass did not propagate (a
  call_function such as ``torch.relu``, a call_method such as
  ``x.reshape``, or a constant subgraph such as ``Cast(Constant)``) got the
  network input box as its bounds, and a binary op was trusted on its first
  operand alone. The IBP LeakyReLU also assumed a nonnegative slope.
- AvgPool reach (star, zono) and its IBP divided border windows by the
  full kernel area, ignoring ``count_include_pad=False`` (the ONNX default).
  With ``ceil_mode`` and ``count_include_pad`` torch and ONNX divide an
  overhanging window differently, so such a layer is refused.
- Conv / conv-transpose / avgpool reach rounded the float64 set to
  float32, which moved the set by more than the float64-level padding.
- BatchNorm fusion, MaxPool and Pad: fusion changed the network (a shared
  BN, a Sequential or conv subclass with its own forward) or crashed (no
  gamma/beta, no running statistics); every reach that carried lp_solver
  raised at the first MaxPool; the approx MaxPool star started its new
  variable at the smallest lower bound; pad_star / pad_zono reached any
  pad (a -inf or non-zero constant, reflect, a channel pad) as zero
  padding, and read onnx2torch's OnnxPadStatic pads in ONNX order.

Convention: SAT = unsafe (the unsafe region is ``G y <= g``), UNSAT = safe.
"""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.neural_network import NeuralNetwork
from n2v.nn.layer_ops import avgpool2d_reach, conv2d_reach, conv_transpose2d_reach
from n2v.sets import HalfSpace, ImageZono, Star
from n2v.sets.image_star import ImageStar
from n2v.utils import bounds_precomputation as bp
from n2v.utils.verify_specification import verify_specification


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _node(op, inputs, outputs, **attrs):
    from onnx import helper
    return helper.make_node(op, inputs, outputs, **attrs)


def _onnx_graph_module(nodes, in_shape, out_shape, initializers=()):
    """Build an ONNX model from nodes and convert it with onnx2torch."""
    import onnx
    onnx2torch = pytest.importorskip('onnx2torch')
    from onnx import TensorProto, helper, numpy_helper
    graph = helper.make_graph(
        nodes, 'g',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, list(in_shape))],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, list(out_shape))],
        initializer=[numpy_helper.from_array(np.asarray(a, dtype=np.float32), n)
                     for n, a in initializers])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    onnx.checker.check_model(model)
    return onnx2torch.convert(model).eval()


def _verdict(out, G, g):
    return verify_specification(
        out, HalfSpace(np.atleast_2d(G), np.reshape(g, (-1, 1)))).verdict


def _output_range(out, i=0):
    lo = min(float(np.ravel(s.get_ranges()[0])[i]) for s in out)
    hi = max(float(np.ravel(s.get_ranges()[1])[i]) for s in out)
    return lo, hi


def _check_image_map_exact(layer, star_fn, zono_fn, H, W, C, scale=1.0,
                           atol=1e-12, n_points=20):
    """c' + V' a (star and zono reach) equals the float64 forward of the
    layer at x = c + V a for random predicate points a."""
    layer = layer.eval()
    ref = copy.deepcopy(layer).double()
    rng = np.random.default_rng(0)
    lb = rng.uniform(0.0, scale, (H, W, C))
    ub = lb + rng.uniform(0.0, 0.05 * scale + 1e-3, (H, W, C))

    S = ImageStar.from_bounds(lb, ub, height=H, width=W, num_channels=C)
    out = star_fn(layer, [S])[0]
    Z = ImageZono.from_bounds(lb, ub, H, W, C)
    zout = zono_fn(layer, [Z])[0]
    for _ in range(n_points):
        a = rng.uniform(S.predicate_lb.ravel(), S.predicate_ub.ravel())
        x = S.V[..., 0] + S.V[..., 1:] @ a                  # (H, W, C)
        y = out.V[..., 0] + out.V[..., 1:] @ a              # (H', W', C')
        with torch.no_grad():
            y_ref = ref(torch.from_numpy(x.transpose(2, 0, 1)[None].copy()))
        y_ref = y_ref[0].numpy().transpose(1, 2, 0)
        assert y.shape == y_ref.shape
        np.testing.assert_allclose(y, y_ref, rtol=0, atol=atol)

        b = rng.uniform(-1.0, 1.0, Z.V.shape[1])
        xz = (Z.c.ravel() + Z.V @ b).reshape(H, W, C)
        yz = (zout.c.ravel() + zout.V @ b).reshape(
            zout.height, zout.width, zout.num_channels)
        with torch.no_grad():
            yz_ref = ref(torch.from_numpy(xz.transpose(2, 0, 1)[None].copy()))
        np.testing.assert_allclose(yz, yz_ref[0].numpy().transpose(1, 2, 0),
                                   rtol=0, atol=atol)


# ---------------------------------------------------------------------------
# IBP through OnnxTranspose
# ---------------------------------------------------------------------------

def test_ibp_transpose_on_flat_set_is_not_identity():
    """Transpose(0,2,1) -> Flatten -> Gemm -> Relu on a flat Star. The Gemm
    reads flat entry 1 of the transposed tensor, which is input entry 2 in
    [-6, -5], so the pre-activation -t1 - 5.5 lies in [-0.5, 0.5] and y = 0.5
    is reachable. Treating the Transpose as the identity recorded
    [-11.5, -10.5] and made the ReLU stably inactive (false UNSAT)."""
    gm = _onnx_graph_module(
        [_node('Transpose', ['X'], ['T'], perm=[0, 2, 1]),
         _node('Flatten', ['T'], ['F'], axis=1),
         _node('Gemm', ['F', 'W', 'B'], ['H'], transB=1),
         _node('Relu', ['H'], ['Y'])],
        (1, 2, 2), (1, 1),
        [('W', [[0.0, -1.0, 0.0, 0.0]]), ('B', [-5.5])])
    lb = np.array([0.0, 5.0, -6.0, 0.0])
    ub = np.array([0.0, 6.0, -5.0, 0.0])
    with torch.no_grad():
        y_wit = gm(torch.tensor([0.0, 5.0, -6.0, 0.0]).reshape(1, 2, 2))
    assert float(y_wit) == pytest.approx(0.5)

    bounds = bp.compute_intermediate_bounds(gm, Star.from_bounds(lb, ub), 'ibp')
    for lo, hi in bounds.values():   # any recorded bound must hold the truth
        assert lo.min() <= -0.5 + 1e-9 and hi.max() >= 0.5 - 1e-9

    net = NeuralNetwork(gm)
    for method in ('exact', 'approx'):
        out = net.reach(Star.from_bounds(lb, ub), method=method,
                        input_shape=(2, 2), precompute_bounds='ibp')
        assert _output_range(out)[1] >= 0.5 - 1e-9
        assert _verdict(out, [[-1.0]], [-0.25]) != 'UNSAT'


@pytest.mark.parametrize('perm', [[0, 1, 3, 2], [0, 2, 3, 1], [0, 3, 1, 2]])
def test_ibp_transpose_on_image_matches_reach(perm):
    """A batch-preserving Transpose of an image is propagated exactly: the
    recorded ReLU bounds equal the ranges of the reach set's rows (every
    entry has distinct bounds, so a misplaced bound is detected)."""
    C, H, W = 2, 3, 4
    t_shape = tuple(int(d) for d in np.array([1, C, H, W])[perm])
    gm_t = _onnx_graph_module([_node('Transpose', ['X'], ['Y'], perm=perm)],
                              (1, C, H, W), t_shape)
    gm = _onnx_graph_module([_node('Transpose', ['X'], ['T'], perm=perm),
                             _node('Relu', ['T'], ['Y'])],
                            (1, C, H, W), t_shape)
    rng = np.random.default_rng(0)
    lb = rng.uniform(-1.0, 1.0, (H, W, C))
    ub = lb + rng.uniform(0.1, 1.0, (H, W, C))
    S = ImageStar.from_bounds(lb, ub, height=H, width=W, num_channels=C)

    bounds = bp.compute_intermediate_bounds(gm, S, 'ibp')
    assert len(bounds) == 1
    ibp_lb, ibp_ub = next(iter(bounds.values()))

    # The reach set of the Transpose alone, in its own (HWC) row order
    T = NeuralNetwork(gm_t).reach(S, method='approx')[0]
    r_lb, r_ub = T.to_star().estimate_ranges()
    np.testing.assert_allclose(ibp_lb.ravel(), np.ravel(r_lb), atol=1e-12)
    np.testing.assert_allclose(ibp_ub.ravel(), np.ravel(r_ub), atol=1e-12)


# ---------------------------------------------------------------------------
# IBP trust through unpropagated nodes and binary ops
# ---------------------------------------------------------------------------

class _FunctionalRelu(nn.Module):
    """act(fc2(torch.relu(fc1(x)))), fc1 = x + 1, fc2 = z - 0.5."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)
        self.act = nn.ReLU()
        with torch.no_grad():
            self.fc1.weight.fill_(1.0)
            self.fc1.bias.fill_(1.0)
            self.fc2.weight.fill_(1.0)
            self.fc2.bias.fill_(-0.5)

    def forward(self, x):
        return self.act(self.fc2(torch.relu(self.fc1(x))))


class _MethodReshape(nn.Module):
    """act(fc2(fc1(x).reshape(-1, 2))), fc1 = x + 3, fc2 = sum - 4."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.act = nn.ReLU()
        with torch.no_grad():
            self.fc1.weight.copy_(torch.eye(2))
            self.fc1.bias.fill_(3.0)
            self.fc2.weight.fill_(1.0)
            self.fc2.bias.fill_(-4.0)

    def forward(self, x):
        return self.act(self.fc2(self.fc1(x).reshape(-1, 2)))


@pytest.mark.parametrize('model_cls, lb, ub, x_wit, y_wit, threshold', [
    # true act pre-activation in [-0.5, 0.5]; the input box read as trusted
    # gave [-1.5, -0.5] (ReLU stably inactive)
    (_FunctionalRelu, [-1.0], [0.0], [0.0], 0.5, 0.25),
    # true act pre-activation in [0, 2]; the input box gave [-6, -4]
    (_MethodReshape, [-1.0, -1.0], [0.0, 0.0], [0.0, 0.0], 2.0, 1.0),
])
def test_unpropagated_producer_is_not_trusted(model_cls, lb, ub, x_wit,
                                                  y_wit, threshold):
    model = model_cls().eval()
    with torch.no_grad():
        assert float(model(torch.tensor([x_wit]))) == pytest.approx(y_wit)
    lb, ub = np.array(lb), np.array(ub)

    gm = torch.fx.symbolic_trace(model)
    bounds = bp.compute_intermediate_bounds(gm, Star.from_bounds(lb, ub), 'ibp')
    if 'act' in bounds:   # the act pre-activation at the witness is y_wit
        lo, hi = bounds['act']
        lo, hi = np.asarray(lo).item(), np.asarray(hi).item()
        assert lo <= y_wit + 1e-9 and hi >= y_wit - 1e-9

    out = NeuralNetwork(model).reach(Star.from_bounds(lb, ub), method='approx',
                                     precompute_bounds='ibp')
    assert _output_range(out)[1] >= y_wit - 1e-9
    assert _verdict(out, [[-1.0]], [-threshold]) != 'UNSAT'


def test_functional_relu_bounds_are_propagated():
    """The pre-pass follows the functional ReLU (the reach converts it the
    same way), so both ReLUs still get correct bounds."""
    gm = torch.fx.symbolic_trace(_FunctionalRelu().eval())
    bounds = bp.compute_intermediate_bounds(
        gm, Star.from_bounds(np.array([-1.0]), np.array([0.0])), 'ibp')
    assert set(bounds) == {'relu', 'act'}
    np.testing.assert_allclose(np.ravel(bounds['relu']), [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(np.ravel(bounds['act']), [-0.5, 0.5], atol=1e-12)


@pytest.mark.parametrize('cast', [False, True])
def test_constant_operand_is_a_constant(cast):
    """Add(X, K) -> Relu with X in [-1, 0] and K = 0.5 an ONNX Constant,
    optionally behind a Cast. The true pre-activation is [-0.5, 0.5] and
    y = 0.5 is reached at x = 0. The constant node used to carry the input
    box as bounds and the Add was trusted on X alone, so the pre-pass
    recorded X + [-1, 0] = [-2, 0], the ReLU looked stably inactive, and
    y >= 0.25 came back UNSAT on the runner's ONNX path."""
    import onnx  # noqa: F401
    from onnx import TensorProto, helper
    dtype = TensorProto.DOUBLE if cast else TensorProto.FLOAT
    nodes = [helper.make_node(
        'Constant', [], ['K'],
        value=helper.make_tensor('k', dtype, [1, 1], [0.5]))]
    k = 'K'
    if cast:
        nodes.append(_node('Cast', ['K'], ['KF'], to=TensorProto.FLOAT))
        k = 'KF'
    nodes += [_node('Add', ['X', k], ['M']), _node('Relu', ['M'], ['Y'])]
    gm = _onnx_graph_module(nodes, (1, 1), (1, 1))
    S = Star.from_bounds(np.array([-1.0]), np.array([0.0]))

    bounds = bp.compute_intermediate_bounds(gm, S, 'ibp')
    assert len(bounds) == 1   # the constant is folded, as the reach folds it
    lo, hi = next(iter(bounds.values()))
    np.testing.assert_allclose([np.asarray(lo).item(), np.asarray(hi).item()],
                               [-0.5, 0.5], atol=1e-12)

    for method in ('exact', 'approx'):
        out = NeuralNetwork(gm).reach(S, method=method, precompute_bounds='ibp')
        assert _output_range(out)[1] >= 0.5 - 1e-6
        assert _verdict(out, [[-1.0]], [-0.25]) != 'UNSAT'


def test_binary_op_needs_every_operand_trusted():
    """Add(X, Transpose(X)) -> Flatten -> Gemm -> Relu on a flat Star. The
    Transpose cannot be propagated on a flat set, so its entry keeps the
    untransposed (untrusted) bounds. The Gemm reads -(x1 + x2) with x1 in
    [5, 6] and x2 in [-6, -5], so the pre-activation lies in [-1, 1] and
    y = 1 at x = (0, 5, -6, 0). Trusting the Add on X alone recorded
    -(x1 + x1) in [-12, -10], and the ReLU looked stably inactive (false
    UNSAT for y >= 0.5)."""
    gm = _onnx_graph_module(
        [_node('Transpose', ['X'], ['T'], perm=[0, 2, 1]),
         _node('Add', ['X', 'T'], ['A']),
         _node('Flatten', ['A'], ['F'], axis=1),
         _node('Gemm', ['F', 'W'], ['H'], transB=1),
         _node('Relu', ['H'], ['Y'])],
        (1, 2, 2), (1, 1), [('W', [[0.0, -1.0, 0.0, 0.0]])])
    lb = np.array([0.0, 5.0, -6.0, 0.0])
    ub = np.array([0.0, 6.0, -5.0, 0.0])
    with torch.no_grad():
        y_wit = gm(torch.tensor([0.0, 5.0, -6.0, 0.0]).reshape(1, 2, 2))
    assert float(y_wit) == pytest.approx(1.0)

    bounds = bp.compute_intermediate_bounds(gm, Star.from_bounds(lb, ub), 'ibp')
    for lo, hi in bounds.values():   # the witness pre-activation is 1
        assert lo.min() <= 1.0 + 1e-9 and hi.max() >= 1.0 - 1e-9

    net = NeuralNetwork(gm)
    for method in ('exact', 'approx'):
        out = net.reach(Star.from_bounds(lb, ub), method=method,
                        input_shape=(2, 2), precompute_bounds='ibp')
        assert _output_range(out)[1] >= 1.0 - 1e-9
        assert _verdict(out, [[-1.0]], [-0.5]) != 'UNSAT'


def test_ibp_pool_without_image_shape_raises():
    """Pooling cannot be propagated without the image shape; passing the
    bounds through unchanged would misplace them."""
    lb, ub = np.zeros(16), np.ones(16)
    for layer in (nn.AvgPool2d(2), nn.MaxPool2d(2)):
        with pytest.raises(NotImplementedError):
            bp._ibp_propagate_layer(layer, lb, ub, None)


@pytest.mark.parametrize('gamma', [-2.0, -0.5, 0.0, 0.01, 1.0, 3.0])
def test_ibp_leakyrelu_holds_for_any_slope(gamma):
    """ONNX LeakyRelu allows any alpha. With a negative slope the function
    is not monotone: mapping lb to lb and ub to ub gave [0, 0.5] for
    [-1, 0.5] (true [0, 2]) and the inverted [4, 2] for [-2, -1]."""
    lb = np.array([-1.0, -2.0, 0.5, -3.0, 0.0, -1.0])
    ub = np.array([0.5, -1.0, 2.0, 4.0, 0.0, 0.0])
    lo, hi = bp._ibp_leakyrelu(lb, ub, gamma)
    # grid plus the kink clipped into each interval: the extremes are exact
    xs = np.vstack([np.linspace(lb, ub, 201), np.clip(0.0, lb, ub)[None]])
    f = np.where(xs >= 0, xs, gamma * xs)
    np.testing.assert_allclose(lo, f.min(0), atol=1e-12)
    np.testing.assert_allclose(hi, f.max(0), atol=1e-12)


def test_ibp_sigmoid_upper_bound_survives_expit_underflow():
    """scipy's expit is exactly 0 once exp(-x) overflows (x below about
    -709.78), while sigmoid(x) is positive there; the relative pad left the
    upper bound at 2^-1074, e.g. below sigmoid(-741.19) = 1.3e-322 and
    sigmoid(-716.74) = 5.3e-312."""
    import mpmath
    xs = np.array([-745.5, -744.0, -741.19, -730.0, -716.74, -710.0, -709.9,
                   -709.5, -708.0, -700.0, -30.0, 0.0, 30.0, 800.0])
    lo, hi = bp._ibp_sigmoid(xs, xs)
    with mpmath.workdps(60):
        for x, a, b in zip(xs, lo, hi):
            s = 1 / (1 + mpmath.exp(-mpmath.mpf(float(x))))
            assert mpmath.mpf(float(a)) <= s <= mpmath.mpf(float(b)), x


# ---------------------------------------------------------------------------
# AvgPool count_include_pad / ceil_mode / divisor_override
# ---------------------------------------------------------------------------

def test_onnx_average_pool_default_excludes_padding():
    """ONNX AveragePool 3x3, stride 1, pads 1, default count_include_pad=0,
    input in [0.9, 1.0]: every output is in [0.9, 1.0]. Dividing border
    windows by 9 gave corners in [0.4, 0.444] and a false UNSAT for
    y_corner >= 0.8."""
    gm = _onnx_graph_module(
        [_node('AveragePool', ['X'], ['Y'], kernel_shape=[3, 3],
               pads=[1, 1, 1, 1], strides=[1, 1])],
        (1, 1, 3, 3), (1, 1, 3, 3))
    pool = [m for m in gm.modules() if isinstance(m, nn.AvgPool2d)][0]
    assert pool.count_include_pad is False

    lb, ub = np.full((3, 3, 1), 0.9), np.full((3, 3, 1), 1.0)
    S = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=1)
    out = NeuralNetwork(gm).reach(S, method='approx')
    lo, hi = out[0].to_star().estimate_ranges()
    np.testing.assert_allclose(np.ravel(lo), 0.9, atol=1e-9)
    np.testing.assert_allclose(np.ravel(hi), 1.0, atol=1e-9)
    G = np.zeros((1, 9))
    G[0, 0] = -1.0
    assert _verdict(out, G, [-0.8]) != 'UNSAT'

    Z = ImageZono.from_bounds(lb, ub, 3, 3, 1)
    zlo, zhi = NeuralNetwork(gm).reach(Z, method='approx')[0].get_bounds()
    np.testing.assert_allclose(np.ravel(zlo), 0.9, atol=1e-9)
    np.testing.assert_allclose(np.ravel(zhi), 1.0, atol=1e-9)

    ilo, ihi, _ = bp._ibp_pool(np.full(9, 0.9), np.full(9, 1.0), pool, (1, 3, 3))
    np.testing.assert_allclose(ilo, 0.9, atol=1e-12)
    np.testing.assert_allclose(ihi, 1.0, atol=1e-12)


def test_avgpool_ibp_feeds_relu_correctly():
    """AvgPool(count_include_pad=False) -> Linear(I, -0.5) -> ReLU: the true
    y0 range is [0.4, 0.5]; wrong IBP pool bounds made the ReLU look stably
    inactive (reach y0 = 0)."""
    model = nn.Sequential(nn.AvgPool2d(3, 1, 1, count_include_pad=False),
                          nn.Flatten(), nn.Linear(9, 9), nn.ReLU()).eval()
    with torch.no_grad():
        model[2].weight.copy_(torch.eye(9))
        model[2].bias.fill_(-0.5)
    S = ImageStar.from_bounds(np.full((3, 3, 1), 0.9), np.full((3, 3, 1), 1.0),
                              height=3, width=3, num_channels=1)
    for method in ('exact', 'approx'):
        out = NeuralNetwork(model).reach(S, method=method, precompute_bounds='ibp')
        lo, hi = _output_range(out, 0)
        assert lo <= 0.4 + 1e-9 and hi >= 0.5 - 1e-9


_POOL_CONFIGS = [
    dict(kernel_size=3, stride=1, padding=1, count_include_pad=False),
    dict(kernel_size=3, stride=1, padding=1, count_include_pad=True),
    dict(kernel_size=3, stride=2, padding=1, count_include_pad=False),
    dict(kernel_size=2, stride=2, padding=1, count_include_pad=False),
    dict(kernel_size=3, stride=2, padding=1, ceil_mode=True,
         count_include_pad=False),
    dict(kernel_size=2, stride=2, padding=0, ceil_mode=True,
         count_include_pad=False),
    # ceil_mode with count_include_pad and no overhanging window (torch
    # drops the window that would start in the right padding)
    dict(kernel_size=2, stride=2, padding=1, ceil_mode=True),
    dict(kernel_size=3, stride=1, padding=1, divisor_override=4),
    dict(kernel_size=2, stride=2, padding=0, ceil_mode=True,
         divisor_override=3),
]


@pytest.mark.parametrize('cfg', _POOL_CONFIGS)
def test_avgpool_star_and_zono_match_layer(cfg):
    """Star and zono avgpool reach equal the layer's own forward on every
    point of the set (same output shape, float64-level error)."""
    _check_image_map_exact(nn.AvgPool2d(**cfg), avgpool2d_reach.avgpool2d_star,
                           avgpool2d_reach.avgpool2d_zono, H=5, W=6, C=2)


def test_avgpool_ibp_margin_uses_the_kernel_area():
    """A kernel size given as a list is the same kernel as the tuple. The
    IBP rounding margin read [2, 2] as 16 entries per window instead of 4
    (sound, but wider than the rounding it covers)."""
    rng = np.random.default_rng(0)
    lb = rng.uniform(-1.0, 1.0, 16)
    ub = lb + rng.uniform(0.0, 1.0, 16)
    tup = bp._ibp_pool(lb, ub, nn.AvgPool2d((2, 2)), (1, 4, 4))
    lst = bp._ibp_pool(lb, ub, nn.AvgPool2d([2, 2]), (1, 4, 4))
    np.testing.assert_array_equal(lst[0], tup[0])
    np.testing.assert_array_equal(lst[1], tup[1])


def test_ceil_mode_overhang_with_count_include_pad_is_refused():
    """ONNX AveragePool k2 s2 ceil_mode=1 count_include_pad=1 on a 3x3
    input. onnxruntime divides the windows that overhang the input by the
    kernel area (x = 1 gives [1, 0.5, 0.5, 0.25]); the converted
    nn.AvgPool2d divides them by their in-bounds size (all 1). The reach
    followed torch and proved y11 <= 0.5 unreachable for x in [1, 1.01],
    a false UNSAT for the ONNX model. Star, zono and IBP now refuse it."""
    gm = _onnx_graph_module(
        [_node('AveragePool', ['X'], ['P'], kernel_shape=[2, 2],
               strides=[2, 2], ceil_mode=1, count_include_pad=1),
         _node('Relu', ['P'], ['Y'])],
        (1, 1, 3, 3), (1, 1, 2, 2))
    pool = [m for m in gm.modules() if isinstance(m, nn.AvgPool2d)][0]
    assert pool.ceil_mode and pool.count_include_pad

    lb, ub = np.full((3, 3, 1), 1.0), np.full((3, 3, 1), 1.01)
    S = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=1)
    Z = ImageZono.from_bounds(lb, ub, 3, 3, 1)
    with pytest.raises(NotImplementedError):
        avgpool2d_reach.avgpool2d_star(pool, [S])
    with pytest.raises(NotImplementedError):
        avgpool2d_reach.avgpool2d_zono(pool, [Z])
    with pytest.raises(NotImplementedError):
        bp._ibp_pool(np.full(9, 1.0), np.full(9, 1.01), pool, (1, 3, 3))
    # The graph pass stops trusting the bounds after the pool
    assert bp.compute_intermediate_bounds(gm, S, 'ibp') == {}
    with pytest.raises(NotImplementedError):
        NeuralNetwork(gm).reach(S, method='approx')


# ---------------------------------------------------------------------------
# float64 set arithmetic in conv / conv-transpose / avgpool
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['exact', 'approx'])
@pytest.mark.parametrize('precompute', [False, 'ibp'])
def test_conv_large_center_no_false_unsat(method, precompute):
    """y = relu(x - 1000) with x in [1000.00001, 1000.00005]: y0 = 5e-5 at
    the upper corner. In float32 the center rounds to 1000.0 and the reach
    set was y0 <= 2e-5, so y0 >= 3e-5 came back UNSAT."""
    conv = nn.Conv2d(1, 1, 1)
    with torch.no_grad():
        conv.weight.fill_(1.0)
        conv.bias.fill_(-1000.0)
    model = nn.Sequential(conv, nn.ReLU()).eval()
    S = ImageStar.from_bounds(np.full((2, 2, 1), 1000.00001),
                              np.full((2, 2, 1), 1000.00005),
                              height=2, width=2, num_channels=1)
    out = NeuralNetwork(model).reach(S, method=method, precompute_bounds=precompute)
    stars = [s.flatten_to_star() for s in out]
    hi = max(float(np.ravel(s.get_ranges()[1])[0]) for s in stars)
    assert hi >= 1000.00005 - 1000.0 - 1e-9
    assert _verdict(out, [[-1.0, 0.0, 0.0, 0.0]], [-3e-5]) != 'UNSAT'


def _image_ops():
    torch.manual_seed(0)
    return [
        (nn.Conv2d(2, 3, 3, stride=1, padding=1), conv2d_reach.conv2d_star,
         conv2d_reach.conv2d_zono),
        (nn.Conv2d(2, 4, 2, stride=2, groups=2, bias=False),
         conv2d_reach.conv2d_star, conv2d_reach.conv2d_zono),
        (nn.ConvTranspose2d(2, 3, 3, stride=2, padding=1, output_padding=1),
         conv_transpose2d_reach.conv_transpose2d_star,
         conv_transpose2d_reach.conv_transpose2d_zono),
        (nn.AvgPool2d(2, 2), avgpool2d_reach.avgpool2d_star,
         avgpool2d_reach.avgpool2d_zono),
    ]


@pytest.mark.parametrize('idx', range(4))
def test_image_ops_are_float64_exact(idx):
    """At raw-pixel magnitude (0..255) the star and zono images of a conv,
    conv-transpose or avgpool equal the float64 forward of the same layer
    to float64 rounding; float32 set arithmetic was off by ~1e-5."""
    layer, star_fn, zono_fn = _image_ops()[idx]
    _check_image_map_exact(layer, star_fn, zono_fn, H=6, W=5, C=2,
                           scale=255.0, atol=1e-9)


@pytest.mark.parametrize('mode', ['reflect', 'replicate', 'circular'])
def test_conv_padding_mode_matches_layer(mode):
    """A Conv2d with a non-zero padding mode pads by copying entries; the
    reach and the IBP pre-pass used to zero-pad instead, i.e. they verified
    a different function."""
    torch.manual_seed(2)
    layer = nn.Conv2d(2, 3, 3, padding=1, padding_mode=mode)
    _check_image_map_exact(layer, conv2d_reach.conv2d_star,
                           conv2d_reach.conv2d_zono, H=4, W=5, C=2, atol=1e-12)

    ref = copy.deepcopy(layer).double()
    rng = np.random.default_rng(2)
    lb = rng.uniform(-1.0, 1.0, (2, 4, 5))                  # (C, H, W)
    ub = lb + 0.3
    ilo, ihi, _ = bp._ibp_conv(lb.ravel(), ub.ravel(), layer, (2, 4, 5))
    for _ in range(50):
        x = rng.uniform(lb, ub)
        with torch.no_grad():
            y = ref(torch.from_numpy(x[None])).numpy().ravel()
        assert np.all(ilo <= y + 1e-12) and np.all(y <= ihi + 1e-12)


def test_column_batches_match_single_batch(monkeypatch):
    """Pushing the basis through in many small batches (the memory cap on
    float64 temporaries) gives the same set as one batch."""
    torch.manual_seed(1)
    layer = nn.Conv2d(2, 3, 3, padding=1)
    rng = np.random.default_rng(1)
    lb = rng.uniform(0, 1, (4, 5, 2))
    S = ImageStar.from_bounds(lb, lb + 0.1, height=4, width=5, num_channels=2)
    whole = conv2d_reach.conv2d_star(layer, [S])[0].V
    monkeypatch.setattr(conv2d_reach, '_COLUMN_BATCH_BYTES', 1)
    batched = conv2d_reach.conv2d_star(layer, [S])[0].V
    np.testing.assert_allclose(batched, whole, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# IBP bounds are rounded outward
# ---------------------------------------------------------------------------

def test_ibp_rounds_outward_on_a_cancelling_neuron():
    """W = [1, 1], b = -0.7999999999999999 over [0, 0.1] x [0, 0.7]: the true
    max is +2.78e-17, but the round-to-nearest IBP gave an upper bound of
    exactly 0.0. The approximate ReLU trusts precomputed bounds as given, so
    it zeroed the neuron and a 1e17 gain made the reach miss y = 2.78."""
    from fractions import Fraction
    net = nn.Sequential(nn.Linear(2, 1), nn.ReLU(), nn.Linear(1, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0, 1.0]], dtype=torch.float64))
        net[0].bias.fill_(-0.7999999999999999)
        net[2].weight.fill_(1e17)
        net[2].bias.zero_()
    S = Star.from_bounds(np.zeros((2, 1)), np.array([[0.1], [0.7]]))
    true_max = (Fraction(0.1) + Fraction(0.7) + Fraction(-0.7999999999999999))
    assert true_max > 0
    _, ub = bp.compute_intermediate_bounds(net, S)[1]
    assert Fraction(float(ub[0, 0])) >= true_max
    for kw in ({}, {'relax_method': 'area', 'relax_factor': 0.5}):
        out = NeuralNetwork(net).reach(S, method='approx',
                                       precompute_bounds='ibp', **kw)
        hi = max(r.estimate_ranges()[1][0, 0] for r in out)
        assert Fraction(float(hi)) >= true_max * Fraction(1e17)


def test_ibp_affine_ops_enclose_the_exact_range():
    """Linear, Conv2d and AvgPool IBP bounds against the exact rational range
    of the first layer, on inputs built so the range touches 0 (where
    round-to-nearest lands inside it)."""
    from fractions import Fraction
    rng = np.random.default_rng(0)

    def exact(W, b, lb, ub):
        lo = [Fraction(b[i]) + sum(min(Fraction(w) * Fraction(l), Fraction(w) * Fraction(u))
                                   for w, l, u in zip(W[i], lb, ub)) for i in range(len(W))]
        hi = [Fraction(b[i]) + sum(max(Fraction(w) * Fraction(l), Fraction(w) * Fraction(u))
                                   for w, l, u in zip(W[i], lb, ub)) for i in range(len(W))]
        return lo, hi

    for _ in range(40):
        lb = rng.uniform(-1, 1, 4) * 10.0 ** rng.integers(-3, 3)
        ub = lb + rng.uniform(0, 1, 4)
        W = rng.normal(size=(3, 4)) * 10.0 ** rng.integers(-2, 3)
        b = -(W @ np.where(rng.random(4) < 0.5, lb, ub))
        lo, hi = bp._ibp_linear(lb, ub, W, b)
        elo, ehi = exact(W, b, lb, ub)
        assert all(Fraction(float(x)) <= e for x, e in zip(lo, elo))
        assert all(Fraction(float(x)) >= e for x, e in zip(hi, ehi))

        layer = nn.Conv2d(1, 1, 2).double()
        lbi = rng.uniform(-1, 1, (1, 2, 2))
        ubi = lbi + rng.uniform(0, 0.5, (1, 2, 2))
        Wc = layer.weight.detach().numpy().reshape(1, -1)
        with torch.no_grad():
            layer.bias.fill_(float(-(Wc @ np.where(rng.random(4) < 0.5, lbi.ravel(), ubi.ravel()))[0]))
        clo, chi, _ = bp._ibp_conv(lbi.ravel(), ubi.ravel(), layer, (1, 2, 2))
        elo, ehi = exact(Wc, layer.bias.detach().numpy(), lbi.ravel(), ubi.ravel())
        assert Fraction(float(clo[0])) <= elo[0] and Fraction(float(chi[0])) >= ehi[0]

        plo, phi, _ = bp._ibp_pool(lbi.ravel(), ubi.ravel(), nn.AvgPool2d(2), (1, 2, 2))
        assert Fraction(float(plo[0])) <= sum(Fraction(x) for x in lbi.ravel()) / 4
        assert Fraction(float(phi[0])) >= sum(Fraction(x) for x in ubi.ravel()) / 4


def test_ibp_keeps_exact_zero_bounds_exact():
    """Outward rounding must not turn an exactly computed bound into a
    crossing one: a zero row and a ReLU output fed through nonnegative
    weights with zero bias keep lower bound exactly 0."""
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU()).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0, -1.0], [0.0, 0.0]], dtype=torch.float64))
        net[0].bias.zero_()
        net[2].weight.copy_(torch.tensor([[0.5, 2.0], [1.0, 0.0]], dtype=torch.float64))
        net[2].bias.zero_()
    S = Star.from_bounds(-np.ones((2, 1)), np.ones((2, 1)))
    bounds = bp.compute_intermediate_bounds(net, S)
    lb1, ub1 = bounds[1]
    assert lb1[1, 0] == 0.0 and ub1[1, 0] == 0.0
    lb3, _ = bounds[3]
    assert np.all(lb3 == 0.0)


# ---------------------------------------------------------------------------
# MaxPool padding and BatchNorm fusion
# ---------------------------------------------------------------------------

def test_maxpool_padding_is_never_the_max():
    """MaxPool pads with -inf. The star reach padded with 0, so a window over
    negative inputs got a max of at least 0: with x in [-1, -0.9], reach gave
    y0 in [0, 0] and UNSAT for y0 <= -0.5, though y0 <= -0.9 always."""
    from n2v.nn.layer_ops import maxpool2d_reach
    layer = nn.MaxPool2d(2, stride=1, padding=1)
    lb, ub = np.full((2, 2, 1), -1.0), np.full((2, 2, 1), -0.9)
    S = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)
    Z = ImageZono.from_bounds(lb, ub, 2, 2, 1)
    rng = np.random.default_rng(0)
    xs = rng.uniform(lb, ub, (200, 2, 2, 1))
    ys = layer(torch.from_numpy(xs.transpose(0, 3, 1, 2))).numpy().transpose(0, 2, 3, 1)
    ys = ys.reshape(len(xs), -1)
    for method in ('exact', 'approx'):
        out = maxpool2d_reach.maxpool2d_star(layer, [S], method=method)
        lo = np.min([o.estimate_ranges()[0].ravel() for o in out], axis=0)
        hi = np.max([o.estimate_ranges()[1].ravel() for o in out], axis=0)
        assert np.all(ys >= lo - 1e-12) and np.all(ys <= hi + 1e-12), method
        G = np.zeros((1, ys.shape[1]))
        G[0, 0] = 1.0
        assert _verdict(out, G, [-0.5]) != 'UNSAT', method
    zlo, zhi = maxpool2d_reach.maxpool2d_zono(layer, [Z])[0].get_bounds()
    assert np.all(ys >= zlo.ravel() - 1e-12) and np.all(ys <= zhi.ravel() + 1e-12)


@pytest.mark.parametrize('kw', [dict(ceil_mode=True), dict(dilation=2)])
def test_maxpool_star_refuses_windows_it_does_not_build(kw):
    """The star reach builds floor-mode windows from kernel, stride and
    padding only; ceil_mode is accepted where it adds no window."""
    from n2v.nn.layer_ops import maxpool2d_reach
    S = ImageStar.from_bounds(np.zeros((5, 5, 1)), np.ones((5, 5, 1)),
                              height=5, width=5, num_channels=1)
    with pytest.raises(NotImplementedError):
        maxpool2d_reach.maxpool2d_star(nn.MaxPool2d(2, 2, **kw), [S], method='approx')
    if 'ceil_mode' in kw:
        S4 = ImageStar.from_bounds(np.zeros((4, 4, 1)), np.ones((4, 4, 1)),
                                   height=4, width=4, num_channels=1)
        out = maxpool2d_reach.maxpool2d_star(nn.MaxPool2d(2, 2, **kw), [S4],
                                             method='approx')
        assert (out[0].height, out[0].width) == (2, 2)


class _ConvUsedTwice(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        c = self.conv(x)
        return torch.flatten(self.bn(c) - c, 1)


class _BnOnInput(nn.Module):
    """conv and bn are registered next to each other, but bn normalizes x."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        return torch.flatten(self.conv(x) + self.bn(x), 1)


class _SharedBn(nn.Module):
    """One BN module called after two different convs."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 1, 1)
        self.c2 = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        return torch.flatten(self.bn(self.c2(torch.relu(self.bn(self.c1(x))))), 1)


class _ReluBlock(nn.Sequential):
    """A Sequential subclass with its own forward."""
    def forward(self, x):
        return torch.relu(super().forward(x))


def _steep_conv():
    """1x1 conv x -> 10 x: with the BN statistics drawn below, the BN output
    changes sign over x in [-1, 1], so a dropped ReLU is detected."""
    conv = nn.Conv2d(1, 1, 1)
    with torch.no_grad():
        conv.weight.fill_(10.0)
        conv.bias.zero_()
    return conv


@pytest.mark.parametrize('make', [
    _ConvUsedTwice, _BnOnInput, _SharedBn,
    lambda: nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1),
                          nn.BatchNorm2d(1), nn.Flatten()),
    lambda: _ReluBlock(_steep_conv(), nn.BatchNorm2d(1)),
    lambda: nn.Sequential(_ReluBlock(_steep_conv(), nn.BatchNorm2d(1)),
                          nn.Flatten())])
def test_batchnorm_fusion_keeps_the_function(make):
    """Fusion must not change the network: a conv whose output also feeds
    another op, a BN registered after a conv it never follows, chained BNs
    (the second must fuse into the already fused conv), a BN called after
    two convs (fusing one pair turned the BN into an identity at the other
    call), and a Sequential subclass that overrides forward (it was rebuilt
    as a plain Sequential, at top level and nested)."""
    from n2v.utils.model_preprocessing import fuse_batchnorm
    torch.manual_seed(0)
    model = make().eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-1, 1)
                m.running_var.uniform_(0.5, 2)
                m.weight.uniform_(0.5, 3)
                m.bias.uniform_(-1, 1)
    x = torch.linspace(-1, 1, 7).reshape(7, 1, 1, 1)
    for m in (model, torch.fx.symbolic_trace(model)):
        with torch.no_grad():
            torch.testing.assert_close(fuse_batchnorm(m)(x), model(x))
    if not isinstance(model, nn.Sequential):
        return  # native torch add/sub have no reach rule
    S = ImageStar.from_bounds(-np.ones((1, 1, 1)), np.ones((1, 1, 1)),
                              height=1, width=1, num_channels=1)
    out = NeuralNetwork(model).reach(S, method='approx')
    lo = min(o.estimate_ranges()[0].min() for o in out)
    hi = max(o.estimate_ranges()[1].max() for o in out)
    with torch.no_grad():
        y = model(x).numpy()
    assert lo <= y.min() + 1e-6 and y.max() - 1e-6 <= hi


class _DoubledConv(nn.Conv2d):
    """A Conv2d subclass whose forward is not the stock one."""
    def forward(self, x):
        return 2.0 * super().forward(x)


class _StandardizedConv(nn.Conv2d):
    """Weight standardization through _conv_forward."""
    def _conv_forward(self, x, weight, bias):
        w = (weight - weight.mean()) / (weight.std() + 1e-5)
        return super()._conv_forward(x, w, bias)


class _ShiftedBn(nn.BatchNorm2d):
    def forward(self, x):
        return super().forward(x) + 1.0


@pytest.mark.parametrize('conv_cls, bn_cls', [
    (_DoubledConv, nn.BatchNorm2d), (_StandardizedConv, nn.BatchNorm2d),
    (nn.Conv2d, _ShiftedBn)])
def test_batchnorm_fusion_leaves_overriding_subclasses_alone(conv_cls, bn_cls):
    """Fusion rebuilds the pair as a plain Conv2d from its parameters, which
    dropped a subclass's own forward; such a pair must stay unfused."""
    from n2v.utils.model_preprocessing import fuse_batchnorm
    torch.manual_seed(0)
    model = nn.Sequential(conv_cls(1, 2, 3), bn_cls(2), nn.Flatten()).eval()
    with torch.no_grad():
        model[1].running_mean.uniform_(-1, 1)
        model[1].running_var.uniform_(0.5, 2)
    x = torch.randn(4, 1, 5, 5)
    with torch.no_grad():
        torch.testing.assert_close(fuse_batchnorm(model)(x), model(x))


def test_batchnorm_without_affine_or_running_stats():
    """affine=False (no gamma/beta) fuses with gamma = 1, beta = 0; a BN
    without running statistics normalizes by batch statistics even in eval
    mode, so it is not fused and the reach refuses it. Both made fusion
    raise a TypeError on None arithmetic."""
    from n2v.utils.model_preprocessing import fuse_batchnorm, has_batchnorm
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2, affine=False),
                          nn.ReLU(), nn.Flatten()).eval()
    with torch.no_grad():
        model[1].running_mean.uniform_(-1, 1)
        model[1].running_var.uniform_(0.5, 2)
    x = torch.rand(20, 1, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(fuse_batchnorm(model)(x), model(x))
    S = ImageStar.from_bounds(np.zeros((4, 4, 1)), np.ones((4, 4, 1)),
                              height=4, width=4, num_channels=1)
    out = NeuralNetwork(model).reach(S, method='approx')
    lo = np.min([o.estimate_ranges()[0].ravel() for o in out], axis=0)
    hi = np.max([o.estimate_ranges()[1].ravel() for o in out], axis=0)
    with torch.no_grad():
        y = model(x).reshape(20, 2, 2, 2).permute(0, 2, 3, 1).reshape(20, -1).numpy()
    assert np.all(y >= lo - 1e-6) and np.all(y <= hi + 1e-6)

    batch_stats = nn.Sequential(
        nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2, track_running_stats=False),
        nn.ReLU(), nn.Flatten()).eval()
    assert has_batchnorm(fuse_batchnorm(batch_stats))
    with pytest.raises(NotImplementedError):
        NeuralNetwork(batch_stats).reach(S, method='approx')


def test_maxpool_star_reach_takes_lp_solver_and_verbose():
    """The MaxPool branch passed lp_solver and verbose by name and again
    through **kwargs, so every reach that carried them (the config= form
    always does) raised a TypeError at the first MaxPool."""
    from n2v.nn import ReachConfig
    from n2v.nn.reach import reach_pytorch_model
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Flatten(), nn.Linear(8, 2)).eval()
    S = ImageStar.from_bounds(np.zeros((6, 6, 1)), np.full((6, 6, 1), 0.1),
                              height=6, width=6, num_channels=1)
    x = 0.1 * torch.rand(50, 1, 6, 6)
    with torch.no_grad():
        y = model(x).numpy()
    for out in (NeuralNetwork(model).reach(S, 'approx',
                                           config=ReachConfig(method='approx')),
                reach_pytorch_model(model, S, 'approx', verbose=False)):
        lo = np.min([o.estimate_ranges()[0].ravel() for o in out], axis=0)
        hi = np.max([o.estimate_ranges()[1].ravel() for o in out], axis=0)
        assert np.all(y >= lo - 1e-6) and np.all(y <= hi + 1e-6)


def test_maxpool_star_approx_starts_at_the_largest_lower_bound():
    """For an uncertain window the approx star's new variable spanned
    [min of the lower bounds, max of the upper bounds], though the max is at
    least the largest lower bound. The set must stay sound: the true max,
    with the new variable at max(x) - center, satisfies every constraint."""
    from n2v.nn.layer_ops import maxpool2d_reach
    layer = nn.MaxPool2d(2)
    lb = np.array([0.0, 1.0, -5.0, -3.0]).reshape(2, 2, 1)
    ub = np.array([2.0, 1.5, -4.0, 3.0]).reshape(2, 2, 1)
    S = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)
    out = maxpool2d_reach.maxpool2d_star(layer, [S], method='approx')[0]
    lo, hi = out.estimate_ranges()
    assert abs(float(np.ravel(lo)[0]) - 1.0) <= 1e-12   # max lb, not min lb (-5)
    assert abs(float(np.ravel(hi)[0]) - 3.0) <= 1e-12

    rng = np.random.default_rng(0)
    V_in = S.V[..., 1:].reshape(4, -1)
    center = out.V[0, 0, 0, 0]
    for x in rng.uniform(lb.ravel(), ub.ravel(), (200, 4)):
        a = np.linalg.lstsq(V_in, x - S.V[..., 0].ravel(), rcond=None)[0]
        p = np.append(a, x.max() - center)
        assert np.all(out.C @ p <= out.d.ravel() + 1e-12)
        assert np.all(p >= out.predicate_lb.ravel() - 1e-12)
        assert np.all(p <= out.predicate_ub.ravel() + 1e-12)
        assert abs(out.V[0, 0, 0, 0] + out.V[0, 0, 0, 1:] @ p - x.max()) <= 1e-12


# ---------------------------------------------------------------------------
# Pad: only zero padding of the spatial axes, onnx2torch pad order
# ---------------------------------------------------------------------------

def _onnx_pad_static(onnx_pads, mode='constant', value=0.0):
    from onnx2torch.node_converters.pad import OnnxPadStatic
    layer = OnnxPadStatic.create_from_onnx_params(
        onnx_pads=onnx_pads, onnx_mode=mode, constant_value=value)
    layer.constant_value = value
    return layer


def _onnx_pad_dynamic(onnx_pads, mode='constant'):
    from onnx2torch.node_converters.pad import OnnxPadDynamic
    layer = OnnxPadDynamic(mode=mode)
    layer.pads = list(onnx_pads)   # as n2v.nn.reach resolves them
    return layer


@pytest.mark.parametrize('make', [
    lambda: nn.ConstantPad2d((0, 1, 0, 1), float('-inf')),
    lambda: nn.ConstantPad2d(1, 5.0),
    # the module onnx2torch puts in front of an asymmetrically padded MaxPool
    lambda: _onnx_pad_static([0, 0, 1, 1], value=float('-inf')),
    lambda: _onnx_pad_static([0, 0, 1, 1, 0, 0, 1, 1], mode='reflect'),
    lambda: _onnx_pad_static([0, 1, 0, 0, 0, 0, 1, 1]),       # channel pad
    lambda: _onnx_pad_static([0, 0, -1, 0, 0, 0, 0, 1]),      # cropping
    lambda: _onnx_pad_dynamic([0, 0, 1, 1, 0, 0, 1, 1], mode='reflect'),
    lambda: _onnx_pad_dynamic([0, 1, 1, 1, 0, 0, 1, 1])])     # channel pad
def test_pad_other_than_spatial_zero_padding_is_refused(make):
    """pad_star / pad_zono pad the spatial axes with zeros. Another value
    (e.g. -inf in front of a MaxPool made the padding a max candidate), mode,
    or axis was reached as that zero padding, or gave a wrong shape."""
    from n2v.nn.layer_ops.pad_reach import pad_star, pad_zono
    layer = make()
    lb, ub = np.full((3, 3, 2), -2.0), np.full((3, 3, 2), -1.0)
    with pytest.raises(NotImplementedError):
        pad_star(layer, [ImageStar.from_bounds(lb, ub, height=3, width=3,
                                               num_channels=2)])
    with pytest.raises(NotImplementedError):
        pad_zono(layer, [ImageZono.from_bounds(lb, ub, 3, 3, 2)])


def test_minus_inf_pad_before_maxpool_is_refused_end_to_end():
    """The 'same' max-pool idiom: every true output was outside the reach."""
    model = nn.Sequential(nn.ConstantPad2d((0, 1, 0, 1), float('-inf')),
                          nn.MaxPool2d(2, 2)).eval()
    S = ImageStar.from_bounds(np.full((3, 3, 1), -2.1), np.full((3, 3, 1), -1.7),
                              height=3, width=3, num_channels=1)
    with pytest.raises(NotImplementedError):
        NeuralNetwork(model).reach(S, method='approx')


@pytest.mark.parametrize('onnx_pads', [
    [0, 0, 1, 2, 0, 0, 0, 1],   # Pad at opset 2-10: top 1, left 2, right 1
    [0, 1, 2, 1],               # asymmetric conv padding: [top, left, bottom, right]
    [1, 1, 1, 1]])
def test_onnx_pad_static_is_read_in_torch_order(onnx_pads):
    """onnx2torch stores OnnxPadStatic.pads in torch order (last axis first);
    they were read in ONNX order, so the padded image had the wrong shape
    and misplaced entries (and the 4-entry form was refused)."""
    from n2v.nn.layer_ops.pad_reach import pad_star, pad_zono
    _check_image_map_exact(_onnx_pad_static(onnx_pads), pad_star, pad_zono,
                           H=3, W=4, C=2)

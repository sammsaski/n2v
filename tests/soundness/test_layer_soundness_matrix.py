"""Authoritative per-layer-type soundness matrix.

Goal: prove that for every reach-SUPPORTED ONNX op that appears in the
VNN-COMP benchmark corpus, the reachability output is a SOUND
over-approximation on every set-type path it implements, across
edge-case input regimes.

Soundness check (the house pattern): build an input set over a box,
reach through the op, then sample many points from that box, push each
through an INDEPENDENT ground truth, and assert every true output lies
inside the reach enclosure's ranges.

Ground truth (per the agreed split):
  * single ONNX-node ops -> onnxruntime on a one-node model (independent
    of BOTH n2v and onnx2torch). Exercised end-to-end through the real
    load + reach pipeline.
  * set-set composition primitives (Add/Mul/Div/Concat of two computed
    sets) -> torch, in the dedicated files test_soundness_residual_add
    and test_soundness_mul_div_concat (not duplicated here).

Set paths: Star (flat) / ImageStar (spatial), Zono / ImageZono, Box.
A box is encoded into each: Star/ImageStar via from_bounds, Zono/
ImageZono via center + diagonal generators, Box directly.
"""

import numpy as np
import onnx
import pytest
import onnxruntime as ort
from onnx import TensorProto, helper

from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.reach import reach_pytorch_model
from n2v.utils.model_loader import load_onnx

TOL = 1e-5


# --------------------------------------------------------------------------
# ONNX single-op model + onnxruntime ground truth
# --------------------------------------------------------------------------

def _single_op(tmp_path, op_type, attrs, in_shape, out_shape, extra_inits=()):
    node_inputs = ["x"] + [i.name for i in extra_inits]
    node = helper.make_node(op_type, node_inputs, ["y"], **attrs)
    graph = helper.make_graph(
        [node], f"op_{op_type}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(in_shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))],
        initializer=list(extra_inits),
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    path = str(tmp_path / f"{op_type}.onnx")
    onnx.save(model, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return load_onnx(path), sess


def _ort_forward(sess, x, in_shape):
    y = sess.run(None, {"x": np.asarray(x, dtype=np.float32).reshape(in_shape)})
    return np.asarray(y[0], dtype=np.float64).flatten()


# --------------------------------------------------------------------------
# Build each set-path input from a flat box, and read the reach enclosure
# --------------------------------------------------------------------------

def _flat_inputs(lb, ub):
    """Star, Zono, Box encodings of the flat box [lb, ub]."""
    lb = np.asarray(lb, dtype=np.float64).flatten()
    ub = np.asarray(ub, dtype=np.float64).flatten()
    c = (lb + ub) / 2.0
    rad = (ub - lb) / 2.0
    return {
        "Star": Star.from_bounds(lb, ub),
        "Zono": Zono(c.reshape(-1, 1), np.diag(rad)),
        "Box": Box(lb, ub),
    }


def _spatial_inputs(lb, ub, C, H, W):
    """ImageStar, ImageZono encodings of a box, given the (C,H,W) the
    ONNX op expects. Bounds arrive in (C,H,W) flat order and are stored
    HWC, matching create_input_set."""
    lb = np.asarray(lb, dtype=np.float64).flatten()
    ub = np.asarray(ub, dtype=np.float64).flatten()
    lb_hwc = lb.reshape(C, H, W).transpose(1, 2, 0).reshape(-1, 1)
    ub_hwc = ub.reshape(C, H, W).transpose(1, 2, 0).reshape(-1, 1)
    c = (lb_hwc + ub_hwc) / 2.0
    rad = (ub_hwc - lb_hwc) / 2.0
    n = lb.size
    return {
        "ImageStar": ImageStar.from_bounds(lb_hwc, ub_hwc, height=H,
                                           width=W, num_channels=C),
        "ImageZono": ImageZono(c, np.diag(rad.flatten()), H, W, C),
    }


def _enclosure(out_list):
    s = out_list[0]
    lo, hi = s.get_ranges()
    return np.asarray(lo).flatten(), np.asarray(hi).flatten()


def _assert_sound(tmodel, sess, lb, ub, in_shape, set_inputs, n=120,
                  seed=0):
    """For each set-path input, reach the op and assert every sampled
    onnxruntime output is enclosed."""
    feat = tuple(in_shape[1:])
    lb = np.asarray(lb, dtype=np.float64).flatten()
    ub = np.asarray(ub, dtype=np.float64).flatten()
    rng = np.random.default_rng(seed)
    # corner + center + random samples
    samples = [lb, ub, (lb + ub) / 2.0]
    samples += [rng.uniform(lb, ub) for _ in range(n - 3)]
    truth = [_ort_forward(sess, x, in_shape) for x in samples]

    for path, sset in set_inputs.items():
        out = reach_pytorch_model(tmodel, sset, method="approx",
                                  input_shape=feat)
        lo, hi = _enclosure(out)
        for y in truth:
            assert y.shape == lo.shape, (
                f"{path}: shape {y.shape} vs enclosure {lo.shape}")
            below = y < lo - TOL
            above = y > hi + TOL
            assert not below.any() and not above.any(), (
                f"[{path}] soundness violated: y={y}\n lo={lo}\n hi={hi}\n"
                f" below={np.where(below)[0]} above={np.where(above)[0]}")


# ==========================================================================
# Activations — relaxations, where over-approximation soundness can fail
# ==========================================================================

# regime edge cases over a 4-vector
_ACT_BOXES = {
    "all_neg":   (np.array([-3., -2., -1.5, -0.5]), np.array([-2., -1., -0.5, -0.1])),
    "all_pos":   (np.array([0.1, 0.5, 1., 2.]),     np.array([0.5, 1., 2., 3.])),
    "mixed":     (np.array([-2., -1., 0.2, 1.]),    np.array([-0.5, 1., 1.5, 3.])),
    "spans_zero":(np.array([-1., -1., -1., -1.]),   np.array([1., 1., 1., 1.])),
    "wide":      (np.array([-6., -6., -6., -6.]),   np.array([6., 6., 6., 6.])),
    "degenerate":(np.array([0.3, -0.7, 1.2, -2.1]), np.array([0.3, -0.7, 1.2, -2.1])),
}

_ACT_OPS = ["Relu", "Sigmoid", "Tanh", "LeakyRelu"]


class TestActivationSoundness:
    @pytest.mark.parametrize("op", _ACT_OPS)
    @pytest.mark.parametrize("regime", list(_ACT_BOXES))
    def test_all_set_paths(self, tmp_path, op, regime):
        lb, ub = _ACT_BOXES[regime]
        in_shape = (1, 4)
        attrs = {"alpha": 0.1} if op == "LeakyRelu" else {}
        tmodel, sess = _single_op(tmp_path, op, attrs, in_shape, in_shape)
        _assert_sound(tmodel, sess, lb, ub, in_shape, _flat_inputs(lb, ub))


class TestSignSoundness:
    @pytest.mark.parametrize("regime", list(_ACT_BOXES))
    def test_all_set_paths(self, tmp_path, regime):
        lb, ub = _ACT_BOXES[regime]
        in_shape = (1, 4)
        tmodel, sess = _single_op(tmp_path, "Sign", {}, in_shape, in_shape)
        # Sign zono/box use interval bounds; star uses the relaxation
        _assert_sound(tmodel, sess, lb, ub, in_shape, _flat_inputs(lb, ub))


_SOFTMAX_BOXES = {
    "saturated":  (np.array([-1., -1., -1., 4.]),  np.array([-0.5, -0.5, -0.5, 5.])),
    "near_uniform":(np.array([-0.1, -0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1, 0.1])),
    "wide":       (np.array([-3., -3., -3., -3.]), np.array([3., 3., 3., 3.])),
    "degenerate": (np.array([1., -2., 0.5, 3.]),   np.array([1., -2., 0.5, 3.])),
}


class TestSoftmaxSoundness:
    @pytest.mark.parametrize("regime", list(_SOFTMAX_BOXES))
    def test_star_and_box(self, tmp_path, regime):
        lb, ub = _SOFTMAX_BOXES[regime]
        in_shape = (1, 4)
        tmodel, sess = _single_op(tmp_path, "Softmax", {"axis": -1},
                                  in_shape, in_shape)
        # softmax implements star + box (no zono path)
        inputs = _flat_inputs(lb, ub)
        inputs.pop("Zono")
        _assert_sound(tmodel, sess, lb, ub, in_shape, inputs)


# --------------------------------------------------------------------------
# Spatial ops: ImageStar / ImageZono paths. The reach enclosure is HWC-flat
# while onnxruntime returns NCHW-flat, so the truth is permuted to HWC.
# --------------------------------------------------------------------------

def _ort_full(sess, x, in_shape):
    return np.asarray(
        sess.run(None, {"x": np.asarray(x, dtype=np.float32).reshape(in_shape)})[0],
        dtype=np.float64)


def _assert_sound_spatial(tmodel, sess, lb, ub, C, H, W, n=80, seed=0):
    in_shape = (1, C, H, W)
    feat = (C, H, W)
    lb = np.asarray(lb, dtype=np.float64).flatten()
    ub = np.asarray(ub, dtype=np.float64).flatten()
    rng = np.random.default_rng(seed)
    samples = [lb, ub, (lb + ub) / 2.0] + [rng.uniform(lb, ub)
                                           for _ in range(n - 3)]
    # NCHW output -> HWC-flat truth (matches Imagestar get_ranges order)
    truth = []
    for x in samples:
        y = _ort_full(sess, x, in_shape)[0]          # (Co, Ho, Wo)
        truth.append(y.transpose(1, 2, 0).flatten())

    for path, sset in _spatial_inputs(lb, ub, C, H, W).items():
        out = reach_pytorch_model(tmodel, sset, method="approx",
                                  input_shape=feat)
        lo, hi = _enclosure(out)
        for y in truth:
            assert y.shape == lo.shape, (
                f"{path}: shape {y.shape} vs enclosure {lo.shape}")
            assert not (y < lo - TOL).any() and not (y > hi + TOL).any(), (
                f"[{path}] spatial soundness violated:\n y={y}\n lo={lo}\n hi={hi}")


def _w(name, arr):
    a = np.asarray(arr, dtype=np.float32)
    return helper.make_tensor(name, TensorProto.FLOAT, list(a.shape),
                              a.flatten().tolist())


class TestConvSoundness:
    @pytest.mark.parametrize("attrs", [
        {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]},
        {"kernel_shape": [3, 3], "strides": [2, 2]},
        {"kernel_shape": [2, 2], "dilations": [2, 2]},
    ])
    def test_imagestar_imagezono(self, tmp_path, attrs):
        rng = np.random.default_rng(0)
        C_in, C_out, H, W = 3, 2, 6, 6
        k = attrs["kernel_shape"]
        Wt = rng.standard_normal((C_out, C_in, k[0], k[1]))
        b = rng.standard_normal(C_out)
        inits = [_w("W", Wt), _w("B", b)]
        # out_shape isn't checked by ort; give a permissive value
        tmodel, sess = _single_op(
            tmp_path, "Conv", attrs, (1, C_in, H, W), (1, C_out, 1, 1),
            extra_inits=inits)
        center = rng.uniform(-1, 1, C_in * H * W)
        _assert_sound_spatial(tmodel, sess, center - 0.1, center + 0.1,
                              C_in, H, W)


class TestConvTransposeSoundness:
    @pytest.mark.parametrize("attrs", [
        {"kernel_shape": [3, 3]},
        {"kernel_shape": [4, 4], "strides": [2, 2], "pads": [1, 1, 1, 1]},
        {"kernel_shape": [3, 3], "strides": [2, 2], "output_padding": [1, 1]},
    ])
    def test_imagestar_imagezono(self, tmp_path, attrs):
        rng = np.random.default_rng(1)
        C_in, C_out, H, W = 2, 3, 5, 5
        k = attrs["kernel_shape"]
        Wt = rng.standard_normal((C_in, C_out, k[0], k[1]))  # ConvT layout
        b = rng.standard_normal(C_out)
        tmodel, sess = _single_op(
            tmp_path, "ConvTranspose", attrs, (1, C_in, H, W),
            (1, C_out, 1, 1), extra_inits=[_w("W", Wt), _w("B", b)])
        center = rng.uniform(-1, 1, C_in * H * W)
        _assert_sound_spatial(tmodel, sess, center - 0.1, center + 0.1,
                              C_in, H, W)


class TestPoolSoundness:
    def test_maxpool(self, tmp_path):
        # MaxPool is nonlinear -> the star path is a relaxation
        C, H, W = 2, 6, 6
        tmodel, sess = _single_op(
            tmp_path, "MaxPool",
            {"kernel_shape": [2, 2], "strides": [2, 2]},
            (1, C, H, W), (1, C, 3, 3))
        rng = np.random.default_rng(2)
        center = rng.uniform(-1, 1, C * H * W)
        _assert_sound_spatial(tmodel, sess, center - 0.2, center + 0.2,
                              C, H, W)

    def test_averagepool(self, tmp_path):
        C, H, W = 2, 6, 6
        tmodel, sess = _single_op(
            tmp_path, "AveragePool",
            {"kernel_shape": [2, 2], "strides": [2, 2]},
            (1, C, H, W), (1, C, 3, 3))
        rng = np.random.default_rng(3)
        center = rng.uniform(-1, 1, C * H * W)
        _assert_sound_spatial(tmodel, sess, center - 0.2, center + 0.2,
                              C, H, W)

    def test_globalaveragepool(self, tmp_path):
        C, H, W = 3, 5, 5
        tmodel, sess = _single_op(
            tmp_path, "GlobalAveragePool", {}, (1, C, H, W), (1, C, 1, 1))
        rng = np.random.default_rng(4)
        center = rng.uniform(-1, 1, C * H * W)
        _assert_sound_spatial(tmodel, sess, center - 0.2, center + 0.2,
                              C, H, W)


# ==========================================================================
# Affine flat ops, reductions, rounding, constant-operand arithmetic
# ==========================================================================

_AFFINE_BOXES = {
    "small":   (np.array([-1., -0.5, 0.2, 1.]),  np.array([0., 0.5, 1.2, 2.])),
    "wide":    (np.array([-4., -4., -4., -4.]),  np.array([4., 4., 4., 4.])),
    "degenerate": (np.array([0.3, -0.7, 1.2, -2.1]), np.array([0.3, -0.7, 1.2, -2.1])),
}


class TestMatMulSoundness:
    @pytest.mark.parametrize("regime", list(_AFFINE_BOXES))
    def test_x_at_W(self, tmp_path, regime):
        lb, ub = _AFFINE_BOXES[regime]
        rng = np.random.default_rng(0)
        Wt = rng.standard_normal((4, 3))          # x(1,4) @ W(4,3) -> (1,3)
        tmodel, sess = _single_op(
            tmp_path, "MatMul", {}, (1, 4), (1, 3),
            extra_inits=[_w("W", Wt)])
        _assert_sound(tmodel, sess, lb, ub, (1, 4), _flat_inputs(lb, ub))

    def test_batched(self, tmp_path):
        # (1, R, K) @ W(K, M) applies W per row
        rng = np.random.default_rng(1)
        R, K, M = 2, 3, 2
        Wt = rng.standard_normal((K, M))
        tmodel, sess = _single_op(
            tmp_path, "MatMul", {}, (1, R, K), (1, R, M),
            extra_inits=[_w("W", Wt)])
        lb = -np.ones(R * K)
        ub = np.ones(R * K)
        _assert_sound(tmodel, sess, lb, ub, (1, R, K), _flat_inputs(lb, ub))


class TestGemmSoundness:
    @pytest.mark.parametrize("regime", list(_AFFINE_BOXES))
    def test_all_set_paths(self, tmp_path, regime):
        lb, ub = _AFFINE_BOXES[regime]
        rng = np.random.default_rng(2)
        Wt = rng.standard_normal((3, 4))          # Gemm: y = x W^T + b
        b = rng.standard_normal(3)
        tmodel, sess = _single_op(
            tmp_path, "Gemm", {"transB": 1}, (1, 4), (1, 3),
            extra_inits=[_w("W", Wt), _w("B", b)])
        _assert_sound(tmodel, sess, lb, ub, (1, 4), _flat_inputs(lb, ub))


class TestNegSoundness:
    @pytest.mark.parametrize("regime", list(_AFFINE_BOXES))
    def test_all_set_paths(self, tmp_path, regime):
        lb, ub = _AFFINE_BOXES[regime]
        tmodel, sess = _single_op(tmp_path, "Neg", {}, (1, 4), (1, 4))
        _assert_sound(tmodel, sess, lb, ub, (1, 4), _flat_inputs(lb, ub))


class TestBatchNormSoundness:
    def test_spatial(self, tmp_path):
        rng = np.random.default_rng(3)
        C, H, W = 3, 5, 5
        inits = [_w("scale", rng.uniform(0.5, 1.5, C)),
                 _w("B", rng.standard_normal(C)),
                 _w("mean", rng.standard_normal(C)),
                 _w("var", rng.uniform(0.5, 2.0, C))]
        tmodel, sess = _single_op(
            tmp_path, "BatchNormalization", {"epsilon": 1e-5},
            (1, C, H, W), (1, C, H, W), extra_inits=inits)
        center = rng.uniform(-1, 1, C * H * W)
        _assert_sound_spatial(tmodel, sess, center - 0.2, center + 0.2,
                              C, H, W)


class TestReduceSoundness:
    @pytest.mark.parametrize("op", ["ReduceSum", "ReduceMean"])
    def test_flat_last_axis(self, tmp_path, op):
        # reduce over the feature axis -> (1, 1)
        lb = np.array([-1., -0.5, 0.2, 1.])
        ub = np.array([0., 0.5, 1.2, 2.])
        # opset-13 ReduceSum takes axes as an input tensor; ReduceMean
        # still takes it as an attribute.
        if op == "ReduceSum":
            axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
            tmodel, sess = _single_op(
                tmp_path, op, {"keepdims": 1}, (1, 4), (1, 1),
                extra_inits=[axes])
        else:
            tmodel, sess = _single_op(
                tmp_path, op, {"axes": [1], "keepdims": 1}, (1, 4), (1, 1))
        _assert_sound(tmodel, sess, lb, ub, (1, 4), _flat_inputs(lb, ub))


class TestRoundingSoundness:
    @pytest.mark.parametrize("op", ["Round", "Floor", "Ceil"])
    @pytest.mark.parametrize("regime", list(_AFFINE_BOXES))
    def test_all_set_paths(self, tmp_path, op, regime):
        lb, ub = _AFFINE_BOXES[regime]
        tmodel, sess = _single_op(tmp_path, op, {}, (1, 4), (1, 4))
        _assert_sound(tmodel, sess, lb, ub, (1, 4), _flat_inputs(lb, ub))


class TestConstArithmeticSoundness:
    """Add/Sub/Mul/Div by a constant, in both operand orders (the
    mirrored path: Add(c,x), Sub(c,x), Mul(c,x))."""

    @pytest.mark.parametrize("op", ["Add", "Sub", "Mul", "Div"])
    @pytest.mark.parametrize("mirror", [False, True])
    def test_all_set_paths(self, tmp_path, op, mirror):
        if op == "Div" and mirror:
            pytest.skip("c / set is division by a set, not supported")
        lb = np.array([0.5, 1., -0.5, 2.])      # positive-safe for Div
        ub = np.array([1.5, 2., 0.5, 3.])
        cst = np.array([2.0, -1.0, 0.5, 3.0])
        c_init = _w("c", cst)
        node_inputs = (["c", "x"] if mirror else ["x", "c"])
        node = helper.make_node(op, node_inputs, ["y"])
        graph = helper.make_graph(
            [node], f"op_{op}",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
            initializer=[c_init])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10
        path = str(tmp_path / f"{op}_{mirror}.onnx")
        onnx.save(model, path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        tmodel = load_onnx(path)
        _assert_sound(tmodel, sess, lb, ub, (1, 4), _flat_inputs(lb, ub))


# ==========================================================================
# Structural / exact spatial ops across image set paths (Pad, Transpose).
# Exactness is entry-pinned in test_shape_ops_onnx_oracle; here we add the
# all-image-set-path + non-degenerate-box dimension where zono gaps hide.
# ==========================================================================

class TestPadSoundness:
    def test_spatial_pad(self, tmp_path):
        C, H, W = 2, 4, 4
        pads = helper.make_tensor("pads", TensorProto.INT64, [8],
                                  [0, 0, 1, 1, 0, 0, 1, 1])  # NCHW begin/end
        tmodel, sess = _single_op(
            tmp_path, "Pad", {"mode": "constant"}, (1, C, H, W),
            (1, C, H + 2, W + 2), extra_inits=[pads])
        rng = np.random.default_rng(7)
        center = rng.uniform(-1, 1, C * H * W)
        _assert_sound_spatial(tmodel, sess, center - 0.1, center + 0.1,
                              C, H, W)


class TestTransposeSpatialSoundness:
    def test_nchw_perm(self, tmp_path):
        C, H, W = 2, 3, 4
        # permute (N,C,H,W)->(N,H,W,C) then back is covered by exactness;
        # here check a within-spatial swap stays sound on image set paths
        tmodel, sess = _single_op(
            tmp_path, "Transpose", {"perm": [0, 1, 3, 2]},
            (1, C, H, W), (1, C, W, H))
        rng = np.random.default_rng(8)
        center = rng.uniform(-1, 1, C * H * W)
        # output is (C, W, H) in NCHW -> truth permute handled by helper
        in_shape = (1, C, H, W)
        lb = center - 0.1
        ub = center + 0.1
        truth = []
        for x in [lb, ub, center]:
            y = _ort_full(sess, x, in_shape)[0]   # (C, W, H)
            truth.append(y.transpose(1, 2, 0).flatten())
        for path, sset in _spatial_inputs(lb, ub, C, H, W).items():
            out = reach_pytorch_model(tmodel, sset, method="approx",
                                      input_shape=(C, H, W))
            lo, hi = _enclosure(out)
            for y in truth:
                assert y.size == lo.size and not (y < lo - TOL).any() \
                    and not (y > hi + TOL).any(), f"[{path}] transpose"


# ==========================================================================
# Pow(x, p) with constant integer exponent (p=2 convex, p=3 monotonic).
# Edge regimes exercise convex / concave / sign-spanning per neuron.
# ==========================================================================

_POW_BOXES = {
    "all_neg":    (np.array([-3., -2., -1.5, -0.5]), np.array([-2., -1., -0.6, -0.2])),
    "all_pos":    (np.array([0.2, 0.5, 1., 2.]),     np.array([0.6, 1., 2., 3.])),
    "spans_zero": (np.array([-1.5, -1., -2., -0.5]), np.array([1., 1.5, 0.5, 2.])),
    "wide":       (np.array([-3., -3., -3., -3.]),   np.array([3., 3., 3., 3.])),
    "degenerate": (np.array([-1.5, -0.3, 0.7, 2.0]), np.array([-1.5, -0.3, 0.7, 2.0])),
}


class TestPowSoundness:
    @pytest.mark.parametrize("p", [2, 3, 4, 5])
    @pytest.mark.parametrize("regime", list(_POW_BOXES))
    def test_all_set_paths(self, tmp_path, p, regime):
        lb, ub = _POW_BOXES[regime]
        exp = helper.make_tensor("e", TensorProto.FLOAT, [], [float(p)])
        tmodel, sess = _single_op(
            tmp_path, "Pow", {}, (1, 4), (1, 4), extra_inits=[exp])
        _assert_sound(tmodel, sess, lb, ub, (1, 4), _flat_inputs(lb, ub))


# ==========================================================================
# Sin / Cos: bounded periodic; relaxation must handle interior extrema.
# ==========================================================================

_TRIG_BOXES = {
    "small_arc":     (np.array([0.1, -1.0, 0.5, -0.5]), np.array([0.5, -0.3, 1.0, 0.2])),
    "spans_extremum":(np.array([-3., 1., -0.5, 2.]),    np.array([3., 2.5, 4., 5.])),
    "multi_period":  (np.array([-7., -7., -7., -7.]),   np.array([7., 7., 7., 7.])),
    "degenerate":    (np.array([-2.0, -0.3, 0.7, 2.5]), np.array([-2.0, -0.3, 0.7, 2.5])),
}


class TestTrigSoundness:
    @pytest.mark.parametrize("op", ["Sin", "Cos"])
    @pytest.mark.parametrize("regime", list(_TRIG_BOXES))
    def test_all_set_paths(self, tmp_path, op, regime):
        lb, ub = _TRIG_BOXES[regime]
        tmodel, sess = _single_op(tmp_path, op, {}, (1, 4), (1, 4))
        _assert_sound(tmodel, sess, lb, ub, (1, 4), _flat_inputs(lb, ub))

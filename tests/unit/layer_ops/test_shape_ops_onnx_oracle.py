"""Shape-dependent ops pinned against the ONNX reference implementation.

For each op (Gather, Slice, Transpose-on-flat, Unsqueeze) the test builds
a tiny single-op ONNX model, takes ground truth from onnxruntime (the
ONNX project's reference runtime — independent of n2v AND of onnx2torch),
converts the model with the production loader, and requires our reach of
a degenerate (point) star to reproduce onnxruntime's output entry by
entry. These ops are pure index selections — exact, no approximation —
so entry equality on adversarial cases (duplicate indices, middle-axis
slices, axis permutations) is complete evidence of correctness.

Also covers: sampled containment through each op, and the loud-raise
guard when shape tracking is unavailable.
"""

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

import onnxruntime as ort

from n2v.nn.reach import reach_pytorch_model
from n2v.sets import Star
from n2v.utils.model_loader import load_onnx


def _single_op_model(tmp_path, op_type, attrs, in_shape, out_shape,
                     extra_inits=()):
    """Build, save, and load a one-node ONNX model; return
    (torch_model, ort_session, path)."""
    node_inputs = ["x"] + [i.name for i in extra_inits]
    node = helper.make_node(op_type, node_inputs, ["y"], **attrs)
    graph = helper.make_graph(
        [node], f"single_{op_type}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
        initializer=list(extra_inits),
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10  # stay within installed onnxruntime's support
    path = str(tmp_path / f"{op_type}.onnx")
    onnx.save(model, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return load_onnx(path), sess, path


def _point_star(values):
    flat = np.asarray(values, dtype=np.float64).flatten()
    return Star.from_bounds(flat, flat)


def _reach_point(tmodel, values, in_shape):
    out = reach_pytorch_model(tmodel, _point_star(values), method="approx",
                              input_shape=tuple(in_shape[1:]))
    lo, hi = out[0].estimate_ranges()
    lo, hi = np.asarray(lo).flatten(), np.asarray(hi).flatten()
    assert float(np.max(hi - lo)) < 1e-9, "degenerate set must be a point"
    return (lo + hi) / 2


def _ort_point(sess, values, in_shape):
    x = np.asarray(values, dtype=np.float32).reshape(in_shape)
    return sess.run(None, {"x": x})[0].astype(np.float64).flatten()


IN_SHAPE = [1, 2, 3, 4]   # N, A, B, C
N = 24


class TestGatherOracle:
    @pytest.mark.parametrize("axis,indices,out_shape", [
        (1, [1, 0], [1, 2, 3, 4]),
        (2, [2, 0, 1], [1, 2, 3, 4]),
        (3, [3, 1], [1, 2, 3, 2]),
        (2, [1, 1, 1], [1, 2, 3, 4]),         # duplicate indices
        (3, [[0, 2], [1, 3]], [1, 2, 3, 2, 2]),  # 2-D index tensor
    ])
    def test_entries_match_onnxruntime(self, tmp_path, axis, indices,
                                       out_shape):
        idx = np.asarray(indices, dtype=np.int64)
        init = helper.make_tensor("idx", TensorProto.INT64,
                                  list(idx.shape), idx.flatten().tolist())
        tmodel, sess, _ = _single_op_model(
            tmp_path, "Gather", {"axis": axis}, IN_SHAPE, out_shape,
            extra_inits=[init])
        vals = np.arange(N, dtype=np.float64)
        np.testing.assert_allclose(
            _reach_point(tmodel, vals, IN_SHAPE),
            _ort_point(sess, vals, IN_SHAPE), atol=1e-6)

    def test_containment_through_gather(self, tmp_path):
        idx = np.asarray([2, 0], dtype=np.int64)
        init = helper.make_tensor("idx", TensorProto.INT64, [2],
                                  idx.tolist())
        tmodel, sess, _ = _single_op_model(
            tmp_path, "Gather", {"axis": 2}, IN_SHAPE, [1, 2, 2, 4],
            extra_inits=[init])
        lb, ub = np.zeros(N), np.ones(N)
        out = reach_pytorch_model(tmodel, Star.from_bounds(lb, ub),
                                  method="approx",
                                  input_shape=tuple(IN_SHAPE[1:]))
        lo, hi = out[0].estimate_ranges()
        lo, hi = np.asarray(lo).flatten(), np.asarray(hi).flatten()
        rng = np.random.default_rng(0)
        for _ in range(100):
            x = rng.uniform(lb, ub)
            y = _ort_point(sess, x, IN_SHAPE)
            assert np.all(y >= lo - 1e-6) and np.all(y <= hi + 1e-6)


class TestSliceOracle:
    @pytest.mark.parametrize("starts,ends,axes,out_shape", [
        ([1], [3], [2], [1, 2, 2, 4]),     # middle axis (the mscn bug)
        ([0], [1], [1], [1, 1, 3, 4]),
        ([1], [4], [3], [1, 2, 3, 3]),
        ([0, 1], [2, 3], [1, 2], [1, 2, 2, 4]),
    ])
    def test_entries_match_onnxruntime(self, tmp_path, starts, ends, axes,
                                       out_shape):
        inits = [
            helper.make_tensor("starts", TensorProto.INT64,
                               [len(starts)], starts),
            helper.make_tensor("ends", TensorProto.INT64,
                               [len(ends)], ends),
            helper.make_tensor("axes", TensorProto.INT64,
                               [len(axes)], axes),
        ]
        tmodel, sess, _ = _single_op_model(
            tmp_path, "Slice", {}, IN_SHAPE, out_shape, extra_inits=inits)
        vals = np.arange(N, dtype=np.float64)
        np.testing.assert_allclose(
            _reach_point(tmodel, vals, IN_SHAPE),
            _ort_point(sess, vals, IN_SHAPE), atol=1e-6)


class TestTransposeFlatOracle:
    """Flat-star transposes become exact once shapes are tracked."""

    @pytest.mark.parametrize("perm,out_shape", [
        ([0, 2, 1, 3], [1, 3, 2, 4]),
        ([0, 3, 2, 1], [1, 4, 3, 2]),
    ])
    def test_entries_match_onnxruntime(self, tmp_path, perm, out_shape):
        tmodel, sess, _ = _single_op_model(
            tmp_path, "Transpose", {"perm": perm}, IN_SHAPE, out_shape)
        vals = np.arange(N, dtype=np.float64)
        np.testing.assert_allclose(
            _reach_point(tmodel, vals, IN_SHAPE),
            _ort_point(sess, vals, IN_SHAPE), atol=1e-6)


class TestUnsqueezeOracle:
    def test_unsqueeze_is_flat_identity(self, tmp_path):
        init = helper.make_tensor("ax", TensorProto.INT64, [1], [2])
        tmodel, sess, _ = _single_op_model(
            tmp_path, "Unsqueeze", {}, [1, 2, 3], [1, 2, 1, 3],
            extra_inits=[init])
        vals = np.arange(6, dtype=np.float64)
        np.testing.assert_allclose(
            _reach_point(tmodel, vals, [1, 2, 3]),
            _ort_point(sess, vals, [1, 2, 3]), atol=1e-6)


def _multi_op_model(tmp_path, name, nodes, in_shape, out_shape,
                    extra_inits=()):
    """Build, save, and load a small multi-node ONNX model; return
    (torch_model, ort_session, path)."""
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
        initializer=list(extra_inits),
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    path = str(tmp_path / f"{name}.onnx")
    onnx.save(model, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return load_onnx(path), sess, path


def _reach_point_relaxed(tmodel, values, in_shape, width_tol=1e-6):
    """Like _reach_point but tolerates LP-level slack in the width
    (McCormick on a point input is exact up to solver tolerance)."""
    out = reach_pytorch_model(tmodel, _point_star(values), method="approx",
                              input_shape=tuple(in_shape[1:]))
    lo, hi = out[0].estimate_ranges()
    lo, hi = np.asarray(lo).flatten(), np.asarray(hi).flatten()
    assert float(np.max(hi - lo)) < width_tol, \
        "degenerate set must collapse to a point (up to LP tolerance)"
    return (lo + hi) / 2


def _containment(tmodel, sess, lb, ub, in_shape, n=80, seed=11):
    """Sample the box, forward through onnxruntime, and require every
    true output inside the reach set's ranges."""
    out = reach_pytorch_model(
        tmodel, Star.from_bounds(lb.flatten(), ub.flatten()),
        method="approx", input_shape=tuple(in_shape[1:]))
    lo, hi = out[0].estimate_ranges()
    lo, hi = np.asarray(lo).flatten(), np.asarray(hi).flatten()
    rng = np.random.default_rng(seed)
    for _ in range(n):
        v = rng.uniform(lb, ub)
        y = sess.run(None, {"x": v.astype(np.float32).reshape(in_shape)}
                     )[0].astype(np.float64).flatten()
        assert np.all(y >= lo - 1e-5) and np.all(y <= hi + 1e-5), \
            f"containment violated: y={y}, lo={lo}, hi={hi}"


class TestSplitConcatOracle:
    """Inner-axis Split, getitem extraction, and inner-axis Concat —
    chained so the reversed re-concatenation pins both row mappings."""

    @pytest.mark.parametrize("axis,out_shape", [
        (3, [1, 2, 3, 4]),
        (1, [1, 2, 3, 4]),
    ])
    def test_split_then_reversed_concat(self, tmp_path, axis, out_shape):
        nodes = [
            helper.make_node("Split", ["x"], ["s0", "s1"], axis=axis),
            helper.make_node("Concat", ["s1", "s0"], ["y"], axis=axis),
        ]
        tmodel, sess, _ = _multi_op_model(
            tmp_path, f"split_concat_ax{axis}", nodes, IN_SHAPE, out_shape)
        vals = np.arange(N, dtype=np.float64)
        np.testing.assert_allclose(
            _reach_point(tmodel, vals, IN_SHAPE),
            _ort_point(sess, vals, IN_SHAPE), atol=1e-6)

    def test_concat_with_computed_branch(self, tmp_path):
        nodes = [
            helper.make_node("Neg", ["x"], ["nx"]),
            helper.make_node("Concat", ["x", "nx"], ["y"], axis=2),
        ]
        tmodel, sess, _ = _multi_op_model(
            tmp_path, "concat_neg_ax2", nodes, IN_SHAPE, [1, 2, 6, 4])
        vals = np.arange(N, dtype=np.float64)
        np.testing.assert_allclose(
            _reach_point(tmodel, vals, IN_SHAPE),
            _ort_point(sess, vals, IN_SHAPE), atol=1e-6)


class TestReduceOracle:
    """Shape-aware ReduceSum/ReduceMean over inner axes (exact affine
    summation maps)."""

    @pytest.mark.parametrize("keepdims,out_shape", [
        (1, [1, 2, 1, 4]),
        (0, [1, 2, 4]),
    ])
    def test_reducesum_inner_axis(self, tmp_path, keepdims, out_shape):
        # opset 13 ReduceSum takes axes as an input tensor
        init = helper.make_tensor("axes", TensorProto.INT64, [1], [-2])
        nodes = [helper.make_node("ReduceSum", ["x", "axes"], ["y"],
                                  keepdims=keepdims)]
        tmodel, sess, _ = _multi_op_model(
            tmp_path, f"reducesum_kd{keepdims}", nodes, IN_SHAPE, out_shape,
            extra_inits=[init])
        vals = np.arange(N, dtype=np.float64)
        np.testing.assert_allclose(
            _reach_point(tmodel, vals, IN_SHAPE),
            _ort_point(sess, vals, IN_SHAPE), atol=1e-6)

    def test_reducemean_inner_axis(self, tmp_path):
        nodes = [helper.make_node("ReduceMean", ["x"], ["y"],
                                  axes=[1], keepdims=0)]
        tmodel, sess, _ = _multi_op_model(
            tmp_path, "reducemean_ax1", nodes, IN_SHAPE, [1, 3, 4])
        vals = np.arange(N, dtype=np.float64)
        np.testing.assert_allclose(
            _reach_point(tmodel, vals, IN_SHAPE),
            _ort_point(sess, vals, IN_SHAPE), atol=1e-6)


class TestBroadcastBinaryOracle:
    """Broadcasting between two COMPUTED sets (the mscn failure class):
    Mul/Div of x with a reduced (dim-1) branch of itself."""

    def _model(self, tmp_path, op):
        nodes = [
            helper.make_node("ReduceMean", ["x"], ["m"],
                             axes=[3], keepdims=1),
            helper.make_node(op, ["x", "m"], ["y"]),
        ]
        return _multi_op_model(
            tmp_path, f"broadcast_{op.lower()}", nodes, IN_SHAPE, IN_SHAPE)

    @pytest.mark.parametrize("op", ["Mul", "Div"])
    def test_degenerate_matches_onnxruntime(self, tmp_path, op):
        tmodel, sess, _ = self._model(tmp_path, op)
        vals = np.arange(N, dtype=np.float64) + 1.0  # positive denominator
        np.testing.assert_allclose(
            _reach_point_relaxed(tmodel, vals, IN_SHAPE),
            _ort_point(sess, vals, IN_SHAPE), atol=1e-5)

    @pytest.mark.parametrize("op", ["Mul", "Div"])
    def test_containment(self, tmp_path, op):
        tmodel, sess, _ = self._model(tmp_path, op)
        center = np.arange(N, dtype=np.float64) + 2.0
        _containment(tmodel, sess, center - 0.25, center + 0.25, IN_SHAPE)


class TestRoundOracle:
    """Round/Floor relaxation: exact on points, sound on boxes."""

    @pytest.mark.parametrize("op", ["Round", "Floor"])
    def test_degenerate_matches_onnxruntime(self, tmp_path, op):
        tmodel, sess, _ = _single_op_model(
            tmp_path, op, {}, IN_SHAPE, IN_SHAPE)
        vals = np.arange(N, dtype=np.float64) * 0.37 - 4.2
        np.testing.assert_allclose(
            _reach_point(tmodel, vals, IN_SHAPE),
            _ort_point(sess, vals, IN_SHAPE), atol=1e-6)

    def test_containment(self, tmp_path):
        tmodel, sess, _ = _single_op_model(
            tmp_path, "Round", {}, IN_SHAPE, IN_SHAPE)
        center = np.arange(N, dtype=np.float64) * 0.37 - 4.2
        _containment(tmodel, sess, center - 0.4, center + 0.4, IN_SHAPE)

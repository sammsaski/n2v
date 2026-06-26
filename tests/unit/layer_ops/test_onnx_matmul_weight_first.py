"""Regression for the weight-first ``MatMul(W, x)`` form (issue #49).

The VNN-COMP ``test`` benchmark's ``test_small`` model writes each layer as
``y = W @ x`` — the weight is the FIRST MatMul operand and ``x`` is a bare
1-D vector — followed by a bias ``Add``. Broadcasting in shape inference
inflates the post-Add shape (``(M, 1) + (M,) -> (M, M)``), so the recorded
``node_shapes`` reports ``M*M`` elements for an ``M``-vector. The MatMul
handler must NOT trust that hint over the running set's own ``dim``;
doing so built a ``kron``-expanded weight and raised
``ValueError: Matrix W has 4 columns, expected 2``.

Ground truth comes from onnxruntime (independent of n2v and onnx2torch):
a point star must reproduce ort's output exactly, and an interval box's
reach must soundly contain sampled outputs.
"""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from n2v.nn.reach import reach_pytorch_model
from n2v.sets import Star
from n2v.utils.model_loader import load_onnx


def _weight_first_mlp(tmp_path):
    """Build test_small's structure: 1 -> 2 -> 2 -> 1 with every layer a
    weight-first ``MatMul(W, x)`` + bias ``Add`` and ReLU activations."""
    W0 = np.array([[1.5], [-2.0]], dtype=np.float32)        # (2, 1)
    B0 = np.array([0.5, 1.0], dtype=np.float32)             # (2,)
    W1 = np.array([[0.5, -1.0], [2.0, 0.25]], dtype=np.float32)  # (2, 2)
    B1 = np.array([-0.5, 0.75], dtype=np.float32)           # (2,)
    W2 = np.array([[1.0, -1.5]], dtype=np.float32)          # (1, 2)
    B2 = np.array([3.0], dtype=np.float32)                  # (1,)
    inits = [numpy_helper.from_array(a, n) for a, n in
             [(W0, "W0"), (B0, "B0"), (W1, "W1"),
              (B1, "B1"), (W2, "W2"), (B2, "B2")]]
    nodes = [
        helper.make_node("MatMul", ["W0", "X"], ["M0"]),
        helper.make_node("Add", ["M0", "B0"], ["H0"]),
        helper.make_node("Relu", ["H0"], ["R0"]),
        helper.make_node("MatMul", ["W1", "R0"], ["M1"]),
        helper.make_node("Add", ["M1", "B1"], ["H1"]),
        helper.make_node("Relu", ["H1"], ["R1"]),
        helper.make_node("MatMul", ["W2", "R1"], ["M2"]),
        helper.make_node("Add", ["M2", "B2"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "weight_first_mlp",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        initializer=inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    path = str(tmp_path / "weight_first_mlp.onnx")
    onnx.save(model, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return load_onnx(path), sess


def test_weight_first_matmul_point_matches_onnxruntime(tmp_path):
    tmodel, sess = _weight_first_mlp(tmp_path)
    for x in (-1.0, -0.3, 0.0, 0.7, 1.0):
        out = reach_pytorch_model(
            tmodel, Star.from_bounds([x], [x]),
            method="approx", input_shape=(1,))
        lo, hi = out[0].estimate_ranges()
        got = ((np.asarray(lo) + np.asarray(hi)) / 2).flatten()
        ref = sess.run(None, {"X": np.array([x], np.float32)})[0].flatten()
        assert np.allclose(got, ref, atol=1e-6), (x, got, ref)


def test_weight_first_matmul_box_is_sound(tmp_path):
    tmodel, sess = _weight_first_mlp(tmp_path)
    out = reach_pytorch_model(
        tmodel, Star.from_bounds([-1.0], [1.0]),
        method="approx", input_shape=(1,))
    lo, hi = out[0].estimate_ranges()
    lo, hi = float(np.asarray(lo).flatten()[0]), float(np.asarray(hi).flatten()[0])
    xs = np.linspace(-1.0, 1.0, 401, dtype=np.float32)
    ys = np.array([sess.run(None, {"X": np.array([x], np.float32)})[0].flatten()[0]
                   for x in xs])
    assert lo - 1e-6 <= ys.min() and ys.max() <= hi + 1e-6, (lo, hi, ys.min(), ys.max())

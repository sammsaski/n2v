"""Regression test for rank-1 axis-0 Slice (issue #46, Part 1).

Some VNN-COMP models (e.g. cctsdb_yolo_2023) take a RANK-1 input that packs
a flattened image plus a few spec parameters into a single vector and use an
axis-0 ``Slice`` to split them, then reshape the image part to ``[1,C,H,W]``.

These models have no batch dimension. ``_propagate_shapes`` used to probe the
graph with a batch-prefixed dummy (``(1, N)``), whose forward crashes on such
a model — so shape metadata came back empty and the axis-0 ``Slice`` handler
raised ``NotImplementedError`` ("cannot tell batch from data axis"). The fix
falls back to an un-prefixed (rank-1) dummy, restoring shape metadata so the
slice resolves exactly.
"""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnx2torch")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from n2v.nn import NeuralNetwork  # noqa: E402
from n2v.sets import Star  # noqa: E402
from n2v.utils.model_loader import load_onnx  # noqa: E402


def _build_rank1_axis0_slice_model(path, n_total=6, n_data=4):
    """A rank-1-input model mirroring the cctsdb pattern:

    input [n_total] -> Slice axis0 [0:n_data] -> Reshape [1,n_data] -> Gemm.

    The leading 1 in the reshape makes the batched dummy ``(1, n_total)``
    fail (a size-n_total tensor cannot reshape to [1, n_data]), so this only
    passes via the rank-1 fallback in ``_propagate_shapes``.
    """
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), "starts")
    ends = numpy_helper.from_array(np.array([n_data], dtype=np.int64), "ends")
    axes = numpy_helper.from_array(np.array([0], dtype=np.int64), "axes")
    steps = numpy_helper.from_array(np.array([1], dtype=np.int64), "steps")
    rshape = numpy_helper.from_array(
        np.array([1, n_data], dtype=np.int64), "rshape")
    W = numpy_helper.from_array(
        np.arange(n_data * 2, dtype=np.float32).reshape(n_data, 2), "W")
    B = numpy_helper.from_array(np.zeros(2, dtype=np.float32), "B")

    nodes = [
        helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"],
                         ["sliced"], name="slice_data"),
        helper.make_node("Reshape", ["sliced", "rshape"], ["img"],
                         name="reshape_img"),
        helper.make_node("Gemm", ["img", "W", "B"], ["y"], name="gemm_out"),
    ]
    graph = helper.make_graph(
        nodes, "rank1_axis0_slice",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [n_total])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        initializer=[starts, ends, axes, steps, rshape, W, B],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_rank1_axis0_slice_reach(tmp_path):
    n_total, n_data = 6, 4
    onnx_path = tmp_path / "rank1.onnx"
    _build_rank1_axis0_slice_model(onnx_path, n_total, n_data)

    model = load_onnx(str(onnx_path))

    lb = -np.ones((n_total, 1))
    ub = np.ones((n_total, 1))
    input_set = Star.from_bounds(lb, ub)

    # Must not raise NotImplementedError about axis-0 slicing.
    out = NeuralNetwork(model).reach(
        input_set, method="approx", input_shape=(n_total,))

    assert len(out) >= 1
    # Gemm maps the 4 sliced + reshaped values to 2 outputs.
    assert out[0].dim == 2


def test_rank1_axis0_slice_matches_concrete(tmp_path):
    """Reach bounds must soundly contain the concrete forward output."""
    import torch

    n_total, n_data = 6, 4
    onnx_path = tmp_path / "rank1.onnx"
    _build_rank1_axis0_slice_model(onnx_path, n_total, n_data)
    model = load_onnx(str(onnx_path))

    lb = -np.ones((n_total, 1))
    ub = np.ones((n_total, 1))
    out = NeuralNetwork(model).reach(
        Star.from_bounds(lb, ub), method="approx", input_shape=(n_total,))
    rl, ru = out[0].get_ranges()
    rl = np.asarray(rl).flatten()
    ru = np.asarray(ru).flatten()

    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.uniform(-1.0, 1.0, size=n_total).astype(np.float32)
        y = model(torch.from_numpy(x)).detach().numpy().flatten()
        assert np.all(y >= rl - 1e-5)
        assert np.all(y <= ru + 1e-5)

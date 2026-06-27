"""Tests for QuantizeLinear / DequantizeLinear support (issue #47).

onnx2torch ships no Q/DQ converter, so QDQ-quantized models fail to load.
``n2v.utils.quant_converters`` registers spec-exact converters and folds the
weight-side (all-constant) Q/DQ into float initializers so weight-consuming
ops (Conv/Gemm) still see initializers. These tests pin:
  1. the per-op math against onnxruntime (per-tensor + per-axis), and
  2. that a small QDQ graph loads via load_onnx and forwards to onnxruntime.
"""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
import torch  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from n2v.utils.model_loader import load_onnx  # noqa: E402
from n2v.utils.quant_converters import (  # noqa: E402
    OnnxDequantizeLinear,
    OnnxQuantizeLinear,
    _np_dequantize,
    _np_quantize,
)

_TPROTO = {
    np.dtype("int8"): TensorProto.INT8,
    np.dtype("uint8"): TensorProto.UINT8,
    np.dtype("int32"): TensorProto.INT32,
    np.dtype("float32"): TensorProto.FLOAT,
}
_QRANGE = {np.dtype("int8"): (-128.0, 127.0), np.dtype("uint8"): (0.0, 255.0)}


def _ort_single(op, x, scale, zp, axis):
    attrs = {} if axis is None else {"axis": axis}
    node = helper.make_node(op, ["x", "scale", "zp"], ["y"], **attrs)
    out_t = (TensorProto.FLOAT if op == "DequantizeLinear"
             else _TPROTO[zp.dtype])
    g = helper.make_graph(
        [node], "g", [],
        [helper.make_tensor_value_info("y", out_t, None)],
        initializer=[numpy_helper.from_array(x, "x"),
                     numpy_helper.from_array(scale, "scale"),
                     numpy_helper.from_array(zp, "zp")],
    )
    m = helper.make_model(g, opset_imports=[helper.make_operatorsetid("", 13)])
    m.ir_version = 8
    s = ort.InferenceSession(m.SerializeToString(),
                             providers=["CPUExecutionProvider"])
    return np.asarray(s.run(None, {})[0])


_CASES = [
    ("pertensor_uint8",
     np.array([[0.31, -0.22, 1.7], [-3.0, 0.05, 0.0]], np.float32),
     np.float32(0.05), np.uint8(124), None),
    ("peraxis_int8_ax0",
     np.arange(15).reshape(5, 3).astype(np.float32) * 0.1 - 0.7,
     np.array([0.01, 0.02, 0.03, 0.04, 0.05], np.float32),
     np.zeros(5, np.int8), 0),
    ("peraxis_int8_ax1",
     np.arange(24).reshape(4, 6).astype(np.float32) * 0.07 - 0.5,
     (np.arange(6) + 1).astype(np.float32) * 0.01,
     np.zeros(6, np.int8), 1),
]


@pytest.mark.parametrize("name,xf,scale,zp,axis", _CASES)
def test_quantize_matches_onnxruntime(name, xf, scale, zp, axis):
    ax = axis if axis is not None else 1
    ref = _ort_single("QuantizeLinear", xf, scale, zp, axis).astype(np.float32)

    yp = _np_quantize(xf, np.asarray(scale), np.asarray(zp), ax)
    np.testing.assert_allclose(yp.astype(np.float32), ref, atol=0)

    mod = OnnxQuantizeLinear(ax, *_QRANGE[np.dtype(zp.dtype)])
    mod.scale = torch.tensor(scale)
    mod.zero_point = torch.tensor(zp)
    yt = mod(torch.from_numpy(xf)).numpy()
    np.testing.assert_allclose(yt, ref, atol=0)


@pytest.mark.parametrize("name,xf,scale,zp,axis", _CASES)
def test_dequantize_matches_onnxruntime(name, xf, scale, zp, axis):
    ax = axis if axis is not None else 1
    # Quantize first to get a valid integer tensor, then dequantize it.
    xi = _ort_single("QuantizeLinear", xf, scale, zp, axis).astype(zp.dtype)
    ref = _ort_single("DequantizeLinear", xi, scale, zp, axis)

    yp = _np_dequantize(xi, np.asarray(scale), np.asarray(zp), ax)
    np.testing.assert_allclose(yp, ref, atol=0)

    mod = OnnxDequantizeLinear(ax)
    mod.scale = torch.tensor(scale)
    mod.zero_point = torch.tensor(zp)
    yt = mod(torch.from_numpy(xi)).numpy()
    np.testing.assert_allclose(yt, ref, atol=0)


def _build_qdq_gemm_model(path):
    """X -> Q -> DQ -> MatMul(W via int8->DQ) -> Add(bias) -> Y.

    Mirrors a QDQ model: activation Q/DQ stay dynamic; the weight is an int8
    initializer behind a DequantizeLinear and must be folded to load."""
    in_features, out_features = 4, 3
    a_scale = numpy_helper.from_array(np.float32(0.05), "a_scale")
    a_zp = numpy_helper.from_array(np.uint8(128), "a_zp")
    w_int = (np.random.RandomState(0).randint(-127, 127,
             size=(in_features, out_features))).astype(np.int8)
    w_q = numpy_helper.from_array(w_int, "w_q")
    w_scale = numpy_helper.from_array(
        np.array([0.01, 0.02, 0.03], np.float32), "w_scale")
    w_zp = numpy_helper.from_array(np.zeros(3, np.int8), "w_zp")
    bias = numpy_helper.from_array(
        np.array([0.1, -0.2, 0.3], np.float32), "bias")

    nodes = [
        helper.make_node("QuantizeLinear", ["X", "a_scale", "a_zp"], ["xq"]),
        helper.make_node("DequantizeLinear", ["xq", "a_scale", "a_zp"], ["xd"]),
        helper.make_node("DequantizeLinear", ["w_q", "w_scale", "w_zp"],
                         ["wd"], axis=1),
        helper.make_node("MatMul", ["xd", "wd"], ["mm"]),
        helper.make_node("Add", ["mm", "bias"], ["Y"]),
    ]
    g = helper.make_graph(
        nodes, "qdq_gemm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                       [1, in_features])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                       [1, out_features])],
        initializer=[a_scale, a_zp, w_q, w_scale, w_zp, bias],
    )
    m = helper.make_model(g, opset_imports=[helper.make_operatorsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def test_qdq_model_loads_and_matches_onnxruntime(tmp_path):
    onnx_path = tmp_path / "qdq.onnx"
    _build_qdq_gemm_model(onnx_path)

    # Must not raise "Converter is not implemented (... DequantizeLinear ...)".
    model = load_onnx(str(onnx_path))
    model.eval()

    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 4)).astype(np.float32)
    with torch.no_grad():
        yt = np.asarray(model(torch.from_numpy(x))).flatten()

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    yo = np.asarray(sess.run(None, {"X": x})[0]).flatten()

    # Spec-exact float QDQ: single-node ops aren't int8-fused, so this matches
    # onnxruntime tightly.
    np.testing.assert_allclose(yt, yo, atol=1e-5)


def _build_const_qdq_chain_model(path):
    """A CONSTANT Quantize->Dequantize chain (exercises the fold-Q path)
    added to a runtime input: Y = X + DQ(Q(const))."""
    const = numpy_helper.from_array(
        np.array([[1.3, -2.1, 0.4]], np.float32), "C")
    scale = numpy_helper.from_array(np.float32(0.05), "sc")
    zp = numpy_helper.from_array(np.uint8(128), "zp")
    nodes = [
        helper.make_node("QuantizeLinear", ["C", "sc", "zp"], ["cq"]),
        helper.make_node("DequantizeLinear", ["cq", "sc", "zp"], ["cd"]),
        helper.make_node("Add", ["X", "cd"], ["Y"]),
    ]
    g = helper.make_graph(
        nodes, "const_qdq",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])],
        initializer=[const, scale, zp],
    )
    m = helper.make_model(g, opset_imports=[helper.make_operatorsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def test_constant_qdq_chain_folds_and_matches(tmp_path):
    """fold_constant_qdq must fold a constant Q->DQ chain (the _np_quantize
    fold path) and produce onnxruntime-exact results."""
    onnx_path = tmp_path / "const_qdq.onnx"
    _build_const_qdq_chain_model(onnx_path)
    model = load_onnx(str(onnx_path))
    model.eval()

    x = np.array([[0.5, 0.5, 0.5]], np.float32)
    with torch.no_grad():
        yt = np.asarray(model(torch.from_numpy(x))).flatten()
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    yo = np.asarray(sess.run(None, {"X": x})[0]).flatten()
    np.testing.assert_allclose(yt, yo, atol=1e-5)

"""Parity: the ported ViT_BN model must match the benchmark ONNX (via ORT).

This guards that benchmarks/vit_vnncomp2023/model.py + the ONNX weight loader
reproduce the deployed VNN-COMP 2023 ViT to floating-point tolerance, so that
verifying the torch model is verifying the benchmark model.
"""

import os
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX_DIR = os.path.join(REPO, "benchmarks", "vit_vnncomp2023", "onnx")

ort = pytest.importorskip("onnxruntime")
import sys
sys.path.insert(0, os.path.join(REPO, "benchmarks", "vit_vnncomp2023"))


@pytest.mark.parametrize("name", ["pgd_2_3_16", "ibp_3_3_8"])
def test_ported_model_matches_onnx(name):
    import torch
    from model import build_model

    onnx_path = os.path.join(ONNX_DIR, f"{name}.onnx")
    if not os.path.exists(onnx_path):
        pytest.skip(f"vendored ONNX missing: {onnx_path}")

    m = build_model(name, onnx_path)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(8):
        x = rng.standard_normal((1, 3, 32, 32)).astype(np.float32)
        y_ort = sess.run(None, {iname: x})[0]
        with torch.no_grad():
            y_t = m(torch.from_numpy(x)).numpy()
        max_err = max(max_err, float(np.abs(y_ort - y_t).max()))
    assert max_err < 1e-4, f"{name}: torch vs ORT disagree by {max_err:.2e}"

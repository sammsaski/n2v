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

import onnxruntime as ort
import sys
sys.path.insert(0, os.path.join(REPO, "benchmarks", "vit_vnncomp2023"))


@pytest.mark.parametrize("name", ["pgd_2_3_16", "ibp_3_3_8"])
def test_ported_model_matches_onnx(name):
    import torch
    from model import build_model

    onnx_path = os.path.join(ONNX_DIR, f"{name}.onnx")
    # Tracked file: a missing one is a broken checkout, never a skip.
    assert os.path.exists(onnx_path), f"vendored ONNX missing: {onnx_path}"

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


@pytest.mark.parametrize("name", ["pgd_2_3_16", "ibp_3_3_8"])
def test_runner_reconstruction_matches_onnx(name):
    """The competition runner does not use model.py: it rebuilds the ViT
    with n2v's ``load_vit_onnx`` and lowers it with ``to_ops``. Both must
    compute the deployed network. The rebuilt model is compared with
    onnxruntime on the file (the 1e-4 of the case above: the float32 replay
    agrees to about 1e-6, and every listed reconstruction error moves a
    logit by 1e-2 or more), and the rebuilt model and the lowered operation
    list with the ported model in float64 (1e-9 of the logit scale; float64
    evaluations of one network agree to about 1e-15). Catches the loader's
    mapping (query and key order, a later block reading block 0's matrices,
    the feed-forward BatchNorm), the attention scale and the BatchNorm shift
    of n2v's own ``ViT_BN``, and every lowering error, on the two real files
    (two and three blocks, 2x2 and 4x4 patch grids)."""
    import torch
    from model import build_model
    from n2v.nn import crown_reach as cr
    from n2v.nn import vit_crown as vc

    onnx_path = os.path.join(ONNX_DIR, f"{name}.onnx")
    # Tracked file: a missing one is a broken checkout, never a skip.
    assert os.path.exists(onnx_path), f"vendored ONNX missing: {onnx_path}"

    model = vc.load_vit_onnx(onnx_path)
    ops = vc.to_ops(model)
    ported = build_model(name, onnx_path).double()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    rng = np.random.default_rng(1)
    for _ in range(4):
        x = rng.standard_normal((1, 3, 32, 32)).astype(np.float32)
        y_ort = sess.run(None, {iname: x})[0].reshape(-1)
        x64 = torch.from_numpy(x.astype(np.float64))
        with torch.no_grad():
            y_ref = ported(x64).numpy().reshape(-1)
            y_loaded = model(x64).numpy().reshape(-1)
        y_ops = cr.eval_ops(ops, x.astype(np.float64).reshape(-1))[0]
        pad = 1e-9 * (1.0 + float(np.max(np.abs(y_ref))))
        assert float(np.abs(y_loaded - y_ort).max()) < 1e-4, f"{name}: rebuilt vs ORT"
        assert float(np.abs(y_loaded - y_ref).max()) <= pad, f"{name}: rebuilt vs ported"
        assert float(np.abs(y_ops - y_ref).max()) <= pad, f"{name}: lowered vs ported"

"""Box and Zonotope reach are SOUND end-to-end on the ViT layers (coverage).

For each set representation the ViTReacher output must enclose the true logits
of sampled inputs in the eps-box. Box/Zono are looser than Star (and won't
verify the hard benchmark) but must be sound — this is layer coverage.
"""
import os
import sys
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "benchmarks", "vit_vnncomp2023")
ONNX = os.path.join(BENCH, "onnx", "pgd_2_3_16.onnx")
NPZ = os.path.join(BENCH, "instances", "pgd_2_3_16.npz")

if not (os.path.exists(ONNX) and os.path.exists(NPZ)):
    pytest.skip("vendored ViT model/instances missing", allow_module_level=True)
sys.path.insert(0, BENCH)

import torch                                  # noqa: E402
from model import build_model                 # noqa: E402
from reach import ViTReacher                   # noqa: E402
from n2v.sets import Star, Zono, Box           # noqa: E402

MEAN = np.array([0.4914, 0.4822, 0.4465])
STD = np.array([0.2023, 0.1994, 0.2010])


def _bounds(o):
    if isinstance(o, Box):
        return np.asarray(o.lb).reshape(-1), np.asarray(o.ub).reshape(-1)
    if isinstance(o, Zono):
        lo, hi = o.get_bounds()
        return np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1)
    lo, hi = o.estimate_ranges()
    return np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1)


@pytest.mark.parametrize("set_type", [Box, Zono, Star])
def test_vit_reach_sound(set_type):
    z = np.load(NPZ)
    x = z["images"][0].astype(np.float64)
    m = build_model("pgd_2_3_16", ONNX)
    eps = 1.0 / 255
    lb = (np.clip(x - eps, 0, 1) - MEAN[:, None, None]) / STD[:, None, None]
    ub = (np.clip(x + eps, 0, 1) - MEAN[:, None, None]) / STD[:, None, None]

    out = ViTReacher(m, mode="concretize", set_type=set_type).reach(lb, ub)
    lo, hi = _bounds(out)
    assert np.all(lo <= hi + 1e-9)

    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.uniform(lb, ub).astype(np.float32)
        with torch.no_grad():
            y = m(torch.from_numpy(s).unsqueeze(0)).numpy().reshape(-1)
        assert np.all(y >= lo - 1e-3) and np.all(y <= hi + 1e-3), set_type.__name__

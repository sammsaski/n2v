"""The runner gates multimodal / multi-input specs (>1 declared input tensor,
e.g. smart_turn's X1 + X2) to a sound ``unknown`` from the spec header, BEFORE
loading the model: sound reach over the concatenated joint input is intractable
(a ~1.27M-dim dense generator OOMs at 11.7 TiB). This pins the gate, that it is
precise (single-input specs are NOT gated), and that it fires without touching
the (124 MB) model. Mirrors NNV's header-level multimodal gate.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'examples', 'Submission', 'VNN_COMP2026'))
import vnncomp_runner as R  # noqa: E402

CAT = "smart_turn_multimodal_2026"


def _prop(n_inputs):
    return {
        "lb": [0.0, 0.0], "ub": [1.0, 1.0], "pairs": [],
        "input_tensors": [{"name": f"X{i}", "shape": (1, 2)}
                          for i in range(n_inputs)],
        "output_tensors": [{"name": "Y", "shape": (1, 1)}],
    }


def test_multi_input_gated_to_unknown_without_loading_model(monkeypatch):
    monkeypatch.setattr(R, "load_vnnlib", lambda p: _prop(2))

    def _boom(*a, **k):
        raise AssertionError("load_onnx must NOT be called for a gated "
                             "multi-input spec")
    monkeypatch.setattr(R, "load_onnx", _boom)

    res = R.verify_instance("nonexistent.onnx", "spec.vnnlib", CAT)
    assert res["result"] == R.RESULT_UNKNOWN
    assert res.get("counterexample") is None


def test_single_input_not_gated(monkeypatch):
    """One declared input tensor must NOT trip the gate; the runner proceeds to
    load the model (proven here by load_onnx being reached)."""
    monkeypatch.setattr(R, "load_vnnlib", lambda p: _prop(1))

    reached = {"loaded": False}

    def _mark(*a, **k):
        reached["loaded"] = True
        raise RuntimeError("stop after gate check")
    monkeypatch.setattr(R, "load_onnx", _mark)

    with pytest.raises(RuntimeError, match="stop after gate"):
        R.verify_instance("x.onnx", "spec.vnnlib", CAT)
    assert reached["loaded"] is True

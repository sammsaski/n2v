"""Tests for examples/VNN-COMP/run_instance.py"""

import sys
import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'VNN-COMP'))


class TestFormatCounterexample:
    """Test VNN-COMP counterexample formatting."""

    def test_basic_format(self):
        """Counterexample should use VNN-COMP format with X_ and Y_ variables."""
        from run_instance import format_counterexample

        inp = np.array([1.0, 2.0])
        out = np.array([3.0])
        result = format_counterexample(inp, out)
        assert "(X_0" in result
        assert "(X_1" in result
        assert "(Y_0" in result
        assert "1.0" in result or "1." in result
        assert "3.0" in result or "3." in result

    def test_output_format_lowercase(self):
        """Output result strings should be lowercase for VNN-COMP compliance."""
        from run_instance import RESULT_SAT, RESULT_UNSAT, RESULT_UNKNOWN, RESULT_TIMEOUT, RESULT_ERROR

        assert RESULT_SAT == "sat"
        assert RESULT_UNSAT == "unsat"
        assert RESULT_UNKNOWN == "unknown"
        assert RESULT_TIMEOUT == "timeout"
        assert RESULT_ERROR == "error"


class TestVerifyInstance:
    """Test the core verification strategy."""

    def _export_onnx(self, model, input_shape, path):
        dummy = torch.randn(1, *input_shape)
        torch.onnx.export(model, dummy, path, input_names=['input'],
                          output_names=['output'], opset_version=13)

    def _write_vnnlib_sat(self, path, n_inputs, n_outputs, lb, ub):
        """Write a VNNLIB that should be SAT (easy to violate).

        Property: Y_0 <= 1000 (unsafe region is Y_0 <= 1000, almost always true)
        """
        lines = []
        for i in range(n_inputs):
            lines.append(f"(declare-const X_{i} Real)")
        for i in range(n_outputs):
            lines.append(f"(declare-const Y_{i} Real)")
        for i in range(n_inputs):
            lines.append(f"(assert (>= X_{i} {lb[i]}))")
            lines.append(f"(assert (<= X_{i} {ub[i]}))")
        lines.append(f"(assert (<= Y_0 1000.0))")
        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def _write_vnnlib_unsat(self, path, n_inputs, n_outputs, lb, ub):
        """Write a VNNLIB that should be UNSAT (impossible to violate).

        Property: Y_0 <= -1e10 (unsafe region is Y_0 <= -1e10, impossible for bounded network)
        """
        lines = []
        for i in range(n_inputs):
            lines.append(f"(declare-const X_{i} Real)")
        for i in range(n_outputs):
            lines.append(f"(declare-const Y_{i} Real)")
        for i in range(n_inputs):
            lines.append(f"(assert (>= X_{i} {lb[i]}))")
            lines.append(f"(assert (<= X_{i} {ub[i]}))")
        lines.append(f"(assert (<= Y_0 -1e10))")
        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def test_falsification_catches_sat(self, tmp_path):
        """Falsification should find counterexample for trivially SAT property."""
        from run_instance import verify_instance

        # Simple model: output = sum(inputs) which is in [0, 3] for inputs in [0, 1]
        model = nn.Sequential(nn.Linear(3, 1, bias=False))
        model.eval()
        with torch.no_grad():
            model[0].weight.fill_(1.0)

        onnx_path = str(tmp_path / "model.onnx")
        self._export_onnx(model, (3,), onnx_path)

        lb = [0.0, 0.0, 0.0]
        ub = [1.0, 1.0, 1.0]
        vnnlib_path = str(tmp_path / "prop.vnnlib")
        self._write_vnnlib_sat(vnnlib_path, 3, 1, lb, ub)

        result = verify_instance(onnx_path, vnnlib_path)
        assert result['result'] == 'sat'

    def test_approx_proves_unsat(self, tmp_path):
        """Approx reachability should prove UNSAT for impossible property."""
        from run_instance import verify_instance

        model = nn.Sequential(nn.Linear(3, 1, bias=False))
        model.eval()
        with torch.no_grad():
            model[0].weight.fill_(1.0)

        onnx_path = str(tmp_path / "model.onnx")
        self._export_onnx(model, (3,), onnx_path)

        lb = [0.0, 0.0, 0.0]
        ub = [1.0, 1.0, 1.0]
        vnnlib_path = str(tmp_path / "prop.vnnlib")
        self._write_vnnlib_unsat(vnnlib_path, 3, 1, lb, ub)

        result = verify_instance(onnx_path, vnnlib_path)
        assert result['result'] == 'unsat'

    def test_result_has_time(self, tmp_path):
        """Result should include wall-clock time."""
        from run_instance import verify_instance

        model = nn.Sequential(nn.Linear(3, 1, bias=False))
        model.eval()
        with torch.no_grad():
            model[0].weight.fill_(1.0)

        onnx_path = str(tmp_path / "model.onnx")
        self._export_onnx(model, (3,), onnx_path)

        lb = [0.0, 0.0, 0.0]
        ub = [1.0, 1.0, 1.0]
        vnnlib_path = str(tmp_path / "prop.vnnlib")
        self._write_vnnlib_unsat(vnnlib_path, 3, 1, lb, ub)

        result = verify_instance(onnx_path, vnnlib_path)
        assert 'time' in result
        assert result['time'] >= 0.0

    def test_sat_result_has_counterexample(self, tmp_path):
        """SAT result should include counterexample string."""
        from run_instance import verify_instance

        model = nn.Sequential(nn.Linear(3, 1, bias=False))
        model.eval()
        with torch.no_grad():
            model[0].weight.fill_(1.0)

        onnx_path = str(tmp_path / "model.onnx")
        self._export_onnx(model, (3,), onnx_path)

        lb = [0.0, 0.0, 0.0]
        ub = [1.0, 1.0, 1.0]
        vnnlib_path = str(tmp_path / "prop.vnnlib")
        self._write_vnnlib_sat(vnnlib_path, 3, 1, lb, ub)

        result = verify_instance(onnx_path, vnnlib_path)
        assert result['result'] == 'sat'
        assert 'counterexample' in result
        assert result['counterexample'] is not None

    def test_no_falsify_flag(self, tmp_path):
        """--no-falsify should skip falsification stage."""
        from run_instance import verify_instance

        model = nn.Sequential(nn.Linear(3, 1, bias=False))
        model.eval()
        with torch.no_grad():
            model[0].weight.fill_(1.0)

        onnx_path = str(tmp_path / "model.onnx")
        self._export_onnx(model, (3,), onnx_path)

        lb = [0.0, 0.0, 0.0]
        ub = [1.0, 1.0, 1.0]
        vnnlib_path = str(tmp_path / "prop.vnnlib")
        self._write_vnnlib_unsat(vnnlib_path, 3, 1, lb, ub)

        result = verify_instance(onnx_path, vnnlib_path, no_falsify=True)
        assert result['result'] == 'unsat'
        assert result['method'] != 'falsification'

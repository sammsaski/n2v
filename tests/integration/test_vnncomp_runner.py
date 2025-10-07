"""
Integration tests for the generic VNN-COMP runner.

Tests the full pipeline: ONNX model -> VNNLIB spec -> verification result.
Uses synthetic models saved as ONNX and hand-written VNNLIB specs.
"""

import sys
import os
import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'vnncomp'))


class TestVNNCOMPRunnerEndToEnd:
    """End-to-end tests for the generic VNN-COMP runner."""

    def _export_onnx(self, model, input_shape, path):
        dummy = torch.randn(1, *input_shape)
        torch.onnx.export(model, dummy, path, input_names=['input'],
                          output_names=['output'], opset_version=13)

    def _write_vnnlib(self, path, n_inputs, n_outputs, lb, ub, output_constraints):
        """Write a VNNLIB file with custom output constraints.

        output_constraints: list of strings like "(assert (<= Y_0 0.5))"
        """
        lines = []
        for i in range(n_inputs):
            lines.append(f"(declare-const X_{i} Real)")
        for i in range(n_outputs):
            lines.append(f"(declare-const Y_{i} Real)")
        for i in range(n_inputs):
            lines.append(f"(assert (>= X_{i} {lb[i]}))")
            lines.append(f"(assert (<= X_{i} {ub[i]}))")
        for constraint in output_constraints:
            lines.append(constraint)
        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def test_fc_sat(self, tmp_path):
        """FC model with trivially SAT property (falsification should catch it)."""
        from run_instance import verify_instance

        # Model: output = x0 + x1 + x2, range [0, 3] for inputs in [0, 1]
        model = nn.Sequential(nn.Linear(3, 1, bias=False))
        model.eval()
        with torch.no_grad():
            model[0].weight.fill_(1.0)

        onnx_path = str(tmp_path / "model.onnx")
        self._export_onnx(model, (3,), onnx_path)

        lb = [0.0, 0.0, 0.0]
        ub = [1.0, 1.0, 1.0]
        # Unsafe region: Y_0 <= 100 (always true for output in [0,3])
        vnnlib_path = str(tmp_path / "prop.vnnlib")
        self._write_vnnlib(vnnlib_path, 3, 1, lb, ub, ["(assert (<= Y_0 100.0))"])

        result = verify_instance(onnx_path, vnnlib_path)
        assert result['result'] == 'sat'

    def test_fc_unsat(self, tmp_path):
        """FC model with trivially UNSAT property (approx should prove it)."""
        from run_instance import verify_instance

        # Model: output = x0 + x1 + x2, range [0, 3]
        model = nn.Sequential(nn.Linear(3, 1, bias=False))
        model.eval()
        with torch.no_grad():
            model[0].weight.fill_(1.0)

        onnx_path = str(tmp_path / "model.onnx")
        self._export_onnx(model, (3,), onnx_path)

        lb = [0.0, 0.0, 0.0]
        ub = [1.0, 1.0, 1.0]
        # Unsafe region: Y_0 <= -100 (never true for output in [0,3])
        vnnlib_path = str(tmp_path / "prop.vnnlib")
        self._write_vnnlib(vnnlib_path, 3, 1, lb, ub, ["(assert (<= Y_0 -100.0))"])

        result = verify_instance(onnx_path, vnnlib_path)
        assert result['result'] == 'unsat'

    def test_fc_with_relu(self, tmp_path):
        """FC model with ReLU -- approx should handle this."""
        from run_instance import verify_instance

        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
        model.eval()

        onnx_path = str(tmp_path / "model.onnx")
        self._export_onnx(model, (2,), onnx_path)

        # Tight bounds -> should be provable
        lb = [0.5, 0.5]
        ub = [0.6, 0.6]
        # Compute actual output range
        with torch.no_grad():
            corners = torch.tensor([[0.5, 0.5], [0.5, 0.6], [0.6, 0.5], [0.6, 0.6]])
            outputs = model(corners)
            out_max = outputs.max().item()
        # Unsafe region: Y_0 <= out_max - 1000 (way below actual range)
        vnnlib_path = str(tmp_path / "prop.vnnlib")
        self._write_vnnlib(vnnlib_path, 2, 1, lb, ub,
                           [f"(assert (<= Y_0 {out_max - 1000}))"])

        result = verify_instance(onnx_path, vnnlib_path)
        assert result['result'] == 'unsat'

    def test_cnn_model(self, tmp_path):
        """CNN model should use ImageStar auto-detection."""
        from run_instance import verify_instance

        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(1, 2, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 4 * 4, 1),
        )
        model.eval()

        onnx_path = str(tmp_path / "cnn.onnx")
        self._export_onnx(model, (1, 4, 4), onnx_path)

        n = 16
        lb = [0.0] * n
        ub = [0.01] * n  # very tight bounds
        # Unsafe region: Y_0 <= -1e6
        vnnlib_path = str(tmp_path / "prop.vnnlib")
        self._write_vnnlib(vnnlib_path, n, 1, lb, ub, ["(assert (<= Y_0 -1e6))"])

        result = verify_instance(onnx_path, vnnlib_path)
        assert result['result'] == 'unsat'

    def test_error_handling(self, tmp_path):
        """Bad ONNX path should return error, not crash."""
        from run_instance import verify_instance

        result = verify_instance(
            str(tmp_path / "nonexistent.onnx"),
            str(tmp_path / "nonexistent.vnnlib"),
        )
        assert result['result'] == 'error'
        assert result['time'] >= 0.0

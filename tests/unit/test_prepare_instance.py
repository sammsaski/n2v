"""Tests for examples/VNN-COMP/prepare_instance.py"""

import sys
import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

# Add examples/VNN-COMP to path so we can import prepare_instance
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'VNN-COMP'))


class TestGetInputShape:
    """Test ONNX input shape detection."""

    def _export_onnx(self, model, input_shape, path):
        """Helper: export a PyTorch model to ONNX."""
        dummy = torch.randn(1, *input_shape)
        torch.onnx.export(model, dummy, path, input_names=['input'],
                          output_names=['output'], opset_version=13)

    def test_fc_model_shape(self, tmp_path):
        """FC model should return flat shape (n,)."""
        from prepare_instance import get_input_shape

        model = nn.Sequential(nn.Linear(5, 3))
        model.eval()
        onnx_path = str(tmp_path / "fc.onnx")
        self._export_onnx(model, (5,), onnx_path)

        shape = get_input_shape(onnx_path)
        assert shape == (5,)

    def test_cnn_model_shape(self, tmp_path):
        """CNN model should return spatial shape (C, H, W)."""
        from prepare_instance import get_input_shape

        model = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 3),
        )
        model.eval()
        onnx_path = str(tmp_path / "cnn.onnx")
        self._export_onnx(model, (1, 8, 8), onnx_path)

        shape = get_input_shape(onnx_path)
        assert shape == (1, 8, 8)

    def test_2d_input_shape(self, tmp_path):
        """2D input (e.g. sequence) should return shape as-is."""
        from prepare_instance import get_input_shape

        model = nn.Sequential(nn.Flatten(), nn.Linear(6, 3))
        model.eval()
        onnx_path = str(tmp_path / "seq.onnx")
        self._export_onnx(model, (2, 3), onnx_path)

        shape = get_input_shape(onnx_path)
        assert shape == (2, 3)


from n2v.sets import Star
from n2v.sets.image_star import ImageStar


class TestCreateInputSet:
    """Test input set creation from bounds and shape."""

    def test_flat_returns_star(self):
        """1D shape should produce a Star."""
        from prepare_instance import create_input_set

        lb = np.array([0.0, -1.0, 0.5])
        ub = np.array([1.0, 1.0, 1.5])
        result = create_input_set(lb, ub, (3,))
        assert isinstance(result, Star)
        assert result.dim == 3

    def test_image_returns_imagestar(self):
        """3D shape (C, H, W) should produce an ImageStar."""
        from prepare_instance import create_input_set

        shape = (1, 4, 4)
        n = 1 * 4 * 4
        lb = np.zeros(n)
        ub = np.ones(n)
        result = create_input_set(lb, ub, shape)
        assert isinstance(result, ImageStar)
        assert result.height == 4
        assert result.width == 4
        assert result.num_channels == 1

    def test_2d_flattened_to_star(self):
        """2D shape should be flattened to 1D Star."""
        from prepare_instance import create_input_set

        lb = np.zeros(6)
        ub = np.ones(6)
        result = create_input_set(lb, ub, (2, 3))
        assert isinstance(result, Star)
        assert result.dim == 6

    def test_star_bounds_correct(self):
        """Star bounds should match input lb/ub."""
        from prepare_instance import create_input_set

        lb = np.array([-1.0, 0.0, 0.5])
        ub = np.array([1.0, 2.0, 1.5])
        star = create_input_set(lb, ub, (3,))
        lb_out, ub_out = star.get_ranges()
        np.testing.assert_allclose(lb_out.flatten(), lb, atol=1e-6)
        np.testing.assert_allclose(ub_out.flatten(), ub, atol=1e-6)

    def test_imagestar_bounds_correct(self):
        """ImageStar bounds should match input lb/ub."""
        from prepare_instance import create_input_set

        shape = (1, 2, 2)
        n = 4
        lb = np.array([0.1, 0.2, 0.3, 0.4])
        ub = np.array([0.5, 0.6, 0.7, 0.8])
        imgstar = create_input_set(lb, ub, shape)
        lb_out, ub_out = imgstar.estimate_ranges()
        # ImageStar flattened bounds in HWC order
        assert lb_out.size == n
        assert ub_out.size == n


class TestLoadAndPrepare:
    """Test the full load_and_prepare pipeline."""

    def _export_onnx(self, model, input_shape, path):
        """Helper: export a PyTorch model to ONNX."""
        dummy = torch.randn(1, *input_shape)
        torch.onnx.export(model, dummy, path, input_names=['input'],
                          output_names=['output'], opset_version=13)

    def _write_vnnlib(self, path, n_inputs, n_outputs, lb, ub):
        """Helper: write a minimal VNNLIB file."""
        lines = []
        for i in range(n_inputs):
            lines.append(f"(declare-const X_{i} Real)")
        for i in range(n_outputs):
            lines.append(f"(declare-const Y_{i} Real)")
        for i in range(n_inputs):
            lines.append(f"(assert (>= X_{i} {lb[i]}))")
            lines.append(f"(assert (<= X_{i} {ub[i]}))")
        # Output property: Y_0 <= 0 (simple specification)
        lines.append(f"(assert (<= Y_0 0.0))")
        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def test_fc_load_and_prepare(self, tmp_path):
        """Full pipeline with FC model."""
        from prepare_instance import load_and_prepare

        model = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
        model.eval()
        onnx_path = str(tmp_path / "fc.onnx")
        self._export_onnx(model, (3,), onnx_path)

        lb = [0.0, 0.0, 0.0]
        ub = [1.0, 1.0, 1.0]
        vnnlib_path = str(tmp_path / "prop.vnnlib")
        self._write_vnnlib(vnnlib_path, 3, 1, lb, ub)

        result = load_and_prepare(onnx_path, vnnlib_path)

        assert 'model' in result
        assert 'input_shape' in result
        assert 'regions' in result
        assert 'property_spec' in result
        assert result['input_shape'] == (3,)
        assert len(result['regions']) >= 1
        assert 'lb' in result['regions'][0]
        assert 'ub' in result['regions'][0]
        assert 'input_set' in result['regions'][0]
        assert isinstance(result['regions'][0]['input_set'], Star)

    def test_cnn_load_and_prepare(self, tmp_path):
        """Full pipeline with CNN model produces ImageStar."""
        from prepare_instance import load_and_prepare

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
        ub = [1.0] * n
        vnnlib_path = str(tmp_path / "prop.vnnlib")
        self._write_vnnlib(vnnlib_path, n, 1, lb, ub)

        result = load_and_prepare(onnx_path, vnnlib_path)

        assert result['input_shape'] == (1, 4, 4)
        assert isinstance(result['regions'][0]['input_set'], ImageStar)

"""
Soundness tests for Upsample/Resize layer reachability.

Nearest-neighbor upsampling is a linear operation (pixel replication).
Tests verify that random samples from the input set always land in the output set.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.layer_ops.upsample_reach import (
    upsample_star,
    upsample_zono,
    _scale_factors_from_onnx,
)


class TestUpsampleImageStarSoundness:
    """Soundness tests for Upsample with ImageStar sets."""

    def test_output_dimensions(self):
        """Verify output spatial dimensions are scaled correctly."""
        layer = nn.Upsample(scale_factor=2, mode='nearest')

        lb = np.zeros((4, 5, 3))
        ub = np.ones((4, 5, 3))
        istar = ImageStar.from_bounds(lb, ub, height=4, width=5, num_channels=3)

        out = upsample_star(layer, [istar])
        assert out[0].height == 8
        assert out[0].width == 10
        assert out[0].num_channels == 3

    def test_scale_factor_3(self):
        """Verify 3x upsampling."""
        layer = nn.Upsample(scale_factor=3, mode='nearest')

        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1))
        istar = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        out = upsample_star(layer, [istar])
        assert out[0].height == 6
        assert out[0].width == 6

    def test_pixel_replication(self):
        """Each output 2x2 block should have bounds matching the source pixel."""
        layer = nn.Upsample(scale_factor=2, mode='nearest')

        np.random.seed(42)
        lb = np.random.rand(3, 3, 2) * 0.5
        ub = lb + 0.3
        istar = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=2)

        out = upsample_star(layer, [istar])
        lb_out, ub_out = out[0].estimate_ranges()
        lb_in, ub_in = istar.estimate_ranges()

        lb_out_4d = lb_out.reshape(6, 6, 2)
        lb_in_3d = lb_in.reshape(3, 3, 2)

        for h in range(3):
            for w in range(3):
                block = lb_out_4d[h*2:(h+1)*2, w*2:(w+1)*2, :]
                expected = lb_in_3d[h, w, :]
                assert np.allclose(block, expected, atol=1e-10), \
                    f"Block ({h},{w}) doesn't match source pixel"

    def test_random_samples_in_bounds(self):
        """Random samples from input ImageStar produce outputs within bounds."""
        layer = nn.Upsample(scale_factor=2, mode='nearest')

        np.random.seed(0)
        lb = np.random.rand(3, 4, 2) * 0.5
        ub = lb + 0.2
        istar = ImageStar.from_bounds(lb, ub, height=3, width=4, num_channels=2)

        out = upsample_star(layer, [istar])
        lb_out, ub_out = out[0].estimate_ranges()

        for _ in range(200):
            # Sample from input set
            alpha = np.random.uniform(
                istar.predicate_lb.flatten(),
                istar.predicate_ub.flatten()
            )
            V_flat = istar.V.reshape(-1, istar.V.shape[-1])
            x_flat = V_flat[:, 0:1] + V_flat[:, 1:] @ alpha.reshape(-1, 1)
            x_img = x_flat.reshape(3, 4, 2)

            # Apply PyTorch upsample (NCHW format)
            x_torch = torch.tensor(x_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                y_torch = layer(x_torch).squeeze(0).permute(1, 2, 0).numpy()

            y_flat = y_torch.reshape(-1, 1)
            assert np.all(y_flat >= lb_out - 1e-6), "Below lower bound"
            assert np.all(y_flat <= ub_out + 1e-6), "Above upper bound"

    def test_constraints_preserved(self):
        """Upsample should preserve constraints (C, d, predicate bounds)."""
        layer = nn.Upsample(scale_factor=2, mode='nearest')

        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1))
        istar = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        out = upsample_star(layer, [istar])

        assert np.array_equal(out[0].C, istar.C)
        assert np.array_equal(out[0].d, istar.d)
        assert np.array_equal(out[0].predicate_lb, istar.predicate_lb)
        assert np.array_equal(out[0].predicate_ub, istar.predicate_ub)


class TestUpsampleImageZonoSoundness:
    """Soundness tests for Upsample with ImageZono sets."""

    def test_output_dimensions(self):
        """Verify output spatial dimensions."""
        layer = nn.Upsample(scale_factor=2, mode='nearest')

        lb = np.zeros((3, 3, 2))
        ub = np.ones((3, 3, 2))
        izono = ImageZono.from_bounds(lb, ub, height=3, width=3, num_channels=2)

        out = upsample_zono(layer, [izono])
        assert out[0].height == 6
        assert out[0].width == 6
        assert out[0].num_channels == 2

    def test_random_samples_in_bounds(self):
        """Random samples from input ImageZono produce outputs within bounds."""
        layer = nn.Upsample(scale_factor=2, mode='nearest')

        np.random.seed(0)
        lb = np.random.rand(3, 3, 2) * 0.5
        ub = lb + 0.2
        izono = ImageZono.from_bounds(lb, ub, height=3, width=3, num_channels=2)

        out = upsample_zono(layer, [izono])
        lb_out, ub_out = out[0].get_bounds()

        for _ in range(200):
            x = np.random.uniform(lb.flatten(), ub.flatten())
            x_torch = torch.tensor(x.reshape(3, 3, 2), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                y_torch = layer(x_torch).squeeze(0).permute(1, 2, 0).numpy()

            y_flat = y_torch.reshape(-1, 1)
            assert np.all(y_flat >= lb_out - 1e-6)
            assert np.all(y_flat <= ub_out + 1e-6)


class TestOnnxResizeGraphScales:
    """OnnxResize (opset-9 Upsample upgraded to Resize) takes its scales as
    forward inputs, not module attributes. The reach driver resolves the
    real scales from the ONNX graph and passes them via resize_scales — so
    the op must use those, not probe the bare module (which cannot recover
    them) and must never assume a scale factor."""

    def test_scale_factors_from_onnx_scales(self):
        """ONNX scales [n, c, h, w] map to integer (scale_h, scale_w)."""
        assert _scale_factors_from_onnx([1, 1, 2, 2], None, 4, 4) == (2, 2)
        assert _scale_factors_from_onnx([1.0, 1.0, 3.0, 3.0], None, 5, 5) == (3, 3)

    def test_scale_factors_from_onnx_sizes(self):
        """ONNX sizes [n, c, H_out, W_out] divide the input dims."""
        assert _scale_factors_from_onnx(None, [1, 1, 8, 8], 4, 4) == (2, 2)

    def test_scale_factors_reject_noninteger_and_bc_scaling(self):
        """Non-integer scales and batch/channel scaling are not exactly
        representable as nearest-neighbor pixel replication."""
        with pytest.raises(NotImplementedError):
            _scale_factors_from_onnx([1, 1, 2.5, 2.5], None, 4, 4)
        with pytest.raises(NotImplementedError):
            _scale_factors_from_onnx([1, 2, 2, 2], None, 4, 4)
        with pytest.raises(NotImplementedError):
            _scale_factors_from_onnx(None, [1, 1, 7, 7], 4, 4)

    def test_onnx_resize_uses_graph_scales_and_is_sound(self):
        """upsample_star on a real OnnxResize, given resize_scales from the
        graph, scales correctly and contains every concrete sample — without
        probing the module."""
        from onnx2torch.node_converters.resize import OnnxResize

        layer = OnnxResize(mode='nearest')
        np.random.seed(7)
        lb = np.random.rand(3, 3, 2) * 0.5
        ub = lb + 0.4
        istar = ImageStar.from_bounds(lb, ub, height=3, width=3, num_channels=2)

        out = upsample_star(layer, [istar], resize_scales=[1, 1, 2, 2])
        assert out[0].height == 6 and out[0].width == 6 and out[0].num_channels == 2

        lb_out, ub_out = out[0].estimate_ranges()
        scales = torch.tensor([1.0, 1.0, 2.0, 2.0])
        for _ in range(200):
            x = np.random.uniform(lb.flatten(), ub.flatten())
            x_t = torch.tensor(x.reshape(3, 3, 2), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                y = layer(x_t, None, scales).squeeze(0).permute(1, 2, 0).numpy()
            y_flat = y.reshape(-1, 1)
            assert np.all(y_flat >= lb_out - 1e-6)
            assert np.all(y_flat <= ub_out + 1e-6)

    def test_onnx_resize_without_scales_fails_loudly(self):
        """With no graph scales, OnnxResize must raise rather than silently
        assume a scale factor (which would be unsound)."""
        from onnx2torch.node_converters.resize import OnnxResize

        layer = OnnxResize(mode='nearest')
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        istar = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        with pytest.raises(NotImplementedError):
            upsample_star(layer, [istar])

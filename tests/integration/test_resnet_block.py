"""
Integration test: ResNet-style block end-to-end reachability.

Tests that NeuralNetwork.reach() handles auto-fusion, dispatching,
and layer-by-layer processing correctly for models with BatchNorm and Pad.
Note: True residual connections (skip connections) require ONNX graph module
tracing, which is tested separately. These tests focus on the sequential
Conv→BN→ReLU pipeline and Pad+Conv pipeline.

Oracle: the models take a (2, 3, 5) image with its own
interval per pixel (``pixel_box``), so a result that mixes up channels,
height and width misses outputs; 128 member points of the input box, corners
included, are replayed on the original torch model (BatchNorm unfused) in
float64, and each true output must be a member of a returned set by an LP on
that set's own rows (``assert_covered``). The reach goes through the public
entry ``NeuralNetwork.reach``. The input boxes are narrow (widths of 1/16 to
5/16), so that the exact method, which a reach that loses its ``method``
argument would run, ends in a few pieces instead of hanging.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, ImageStar, ImageZono
from n2v.nn import NeuralNetwork
from tests.cell_support import (assert_covered, image_set, pixel_box,
                                replay_torch, samples)


def _narrow_box(seed):
    """A (2, 3, 5) box with its own dyadic interval per pixel, widths 1/16 to
    5/16, in the network's (channel, height, width) order."""
    lo, hi = pixel_box(2, 3, 5, seed=seed)
    return lo, lo + (hi - lo) / 4


def _assert_one_set_holding_every_output(result, model, lo, hi, what,
                                         out_chw=None):
    """The approx method relaxes and never splits, so one input set gives one
    output set (a reach that runs the exact method instead gives several);
    every replayed output of the original model must be in it."""
    assert len(result) == 1, (f"{what}: the approx method returned "
                              f"{len(result)} sets for one input set")
    assert_covered(result, replay_torch(model, samples(lo, hi), (1, 2, 3, 5)),
                   what, out_chw=out_chw)


class TestResNetBlockReachability:
    """End-to-end reachability through models with BatchNorm."""

    def _make_conv_bn_model(self):
        """Create and prepare a Conv+BN+ReLU model."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 3 * 5, 2),
        )
        model.eval()
        model[1].running_mean.data = torch.randn(4) * 0.1
        model[1].running_var.data = torch.rand(4) + 0.1
        return model

    def test_sequential_model_approx_star(self):
        """BatchNorm fusion of a float32 Conv+BN+ReLU model, then an approx
        Star reach of it. The fused model must compute the original model's
        function (float64, 128 points, within 1e-12 relative) and keep the
        network's dtype (a float32 input gives a float32 output); the reach
        must give one set that holds every replayed output of the original
        model. Catches a fold that uses the wrong statistic, drops or flips
        the mean of a layer without bias, scales the wrong weight axis, keeps
        the BatchNorm after folding it (normalised twice), or returns the
        fused layer's output in float64."""
        model = self._make_conv_bn_model()

        from n2v.utils.model_preprocessing import fuse_batchnorm
        fused = fuse_batchnorm(model)
        fused.eval()
        x = torch.randn(1, 2, 3, 5)
        with torch.no_grad():
            y32 = fused(x)
        assert y32.dtype == torch.float32, (
            f"the fused float32 model returns {y32.dtype}")

        lo, hi = _narrow_box(1)
        model64 = copy.deepcopy(model).double().eval()
        fused64 = fuse_batchnorm(model64).eval()
        with torch.no_grad():
            for p in samples(lo, hi):
                t = torch.as_tensor(p.reshape(1, 2, 3, 5))
                want, got = model64(t), fused64(t)
                assert torch.allclose(got, want, rtol=1e-12, atol=1e-12), (
                    f"the fused model computes {got} where the model "
                    f"computes {want}")

        result = NeuralNetwork(model).reach(
            image_set("ImageStar", lo, hi), method='approx')
        _assert_one_set_holding_every_output(
            result, model, lo, hi, "approx Conv/BatchNorm/ReLU on an ImageStar")

    def test_soundness_with_sampling(self):
        """Fused Conv+BN model, approx reach of an ImageStar whose input box
        has a binding row that is not a box row: every replayed output of the
        unfused model is in the result. Catches a fused bias shifted by the
        wrong statistic or sign and a result laid out in the wrong order."""
        from tests.cell_support import constrained_image_star
        model = self._make_conv_bn_model()
        lo, hi = _narrow_box(2)
        star, points = constrained_image_star(lo, hi)
        result = NeuralNetwork(model).reach(star, method='approx')
        assert len(result) == 1, (
            f"the approx method returned {len(result)} sets for one input")
        assert_covered(result, replay_torch(model, points, (1, 2, 3, 5)),
                       "approx Conv/BatchNorm/ReLU on a constrained ImageStar")

    def _make_pad_conv_model(self, bn):
        torch.manual_seed(42)
        layers = [nn.ZeroPad2d((1, 0, 2, 1)),
                  nn.Conv2d(2, 4 if bn else 2, kernel_size=3, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(4))
        layers += [nn.ReLU(), nn.Flatten(),
                   nn.Linear((4 if bn else 2) * 4 * 4, 3)]
        model = nn.Sequential(*layers).eval()
        if bn:
            model[2].running_mean.data = torch.randn(4) * 0.1
            model[2].running_var.data = torch.rand(4) + 0.1
        return model

    def test_pad_conv_pipeline(self):
        """ZeroPad2d (left 1, top 2, bottom 1) + Conv2d on a (2, 3, 5)
        ImageStar through NeuralNetwork.reach(): one set that holds every
        replayed output. Catches pads read in the wrong order (an asymmetric
        pad then moves the image) and a padded set laid out wrongly."""
        model = self._make_pad_conv_model(bn=False)
        lo, hi = _narrow_box(3)
        result = NeuralNetwork(model).reach(image_set("ImageStar", lo, hi),
                                            method='approx')
        _assert_one_set_holding_every_output(
            result, model, lo, hi, "approx ZeroPad2d/Conv on an ImageStar")

    def test_pad_conv_bn_pipeline(self):
        """ZeroPad2d + Conv2d (no bias) + BatchNorm2d through
        NeuralNetwork.reach(): the BatchNorm is folded into a convolution
        without bias that follows a pad. Catches a fold of a layer without
        bias that adds the mean instead of subtracting it, and a fold that
        uses the variance as the mean."""
        model = self._make_pad_conv_model(bn=True)
        lo, hi = _narrow_box(4)
        result = NeuralNetwork(model).reach(image_set("ImageStar", lo, hi),
                                            method='approx')
        _assert_one_set_holding_every_output(
            result, model, lo, hi,
            "approx ZeroPad2d/Conv/BatchNorm on an ImageStar")

    def test_imagezono_pad_conv(self):
        """ZeroPad2d + Conv pipeline with ImageZono (approx method): one set
        that holds every replayed output. Catches the method not handed on
        (an ImageZono under the default 'exact' is refused) and an ImageZono
        padded in the wrong order."""
        model = self._make_pad_conv_model(bn=False)
        lo, hi = _narrow_box(5)
        result = NeuralNetwork(model).reach(image_set("ImageZono", lo, hi),
                                            method='approx')
        _assert_one_set_holding_every_output(
            result, model, lo, hi, "approx ZeroPad2d/Conv on an ImageZono")

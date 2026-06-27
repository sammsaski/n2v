"""Regression test for MaxPool approx predicate explosion (issue #50).

The approx MaxPool over-approximation introduces one new predicate per
*uncertain* pooling window. A window was previously deemed uncertain whenever
two or more pixels could tie for the max (``ub >= max_lb_val``). But an
all-equal window — e.g. the many all-zero windows produced after ReLU on a
sparsely-perturbed input — has a fully determined max (lb == ub) and needs no
predicate. The old test flagged every such window, spawning ~280k degenerate
predicates and a multi-TiB dense basis matrix on ImageNet-scale VGG-16.

The fix counts a window as uncertain only when some pixel can *strictly*
exceed the best guaranteed value. These tests pin both the no-blowup behavior
and the soundness of the result.
"""

import numpy as np
import torch
import torch.nn as nn

from n2v.sets import ImageStar
from n2v.nn.layer_ops import maxpool2d_reach


def _sparse_image_star(H, W, C, base_value, perturbed, eps):
    """An ImageStar fixed at ``base_value`` everywhere except a few perturbed
    (h, w, c) pixels, each given an interval [v-eps, v+eps]."""
    lb = np.full((H, W, C), base_value, dtype=np.float64)
    ub = np.full((H, W, C), base_value, dtype=np.float64)
    for (h, w, c, v) in perturbed:
        lb[h, w, c] = v - eps
        ub[h, w, c] = v + eps
    return ImageStar.from_bounds(lb, ub, height=H, width=W, num_channels=C)


def test_all_equal_window_adds_no_predicate():
    """A constant (all-equal) input must not spawn MaxPool predicates."""
    H = W = 4
    C = 2
    istar = _sparse_image_star(H, W, C, base_value=0.0, perturbed=[], eps=0.0)
    assert istar.nVar == 0

    layer = nn.MaxPool2d(2, 2)
    out = maxpool2d_reach.maxpool2d_star(layer, [istar], method="approx")

    assert len(out) == 1
    # Constant input -> constant output -> still zero predicates (no blowup).
    assert out[0].nVar == 0


def test_sparse_perturbation_adds_few_predicates():
    """Only windows overlapping a perturbed pixel may add a predicate."""
    H = W = 8
    C = 3
    # One perturbed pixel; its 2x2 window is the only uncertain one.
    istar = _sparse_image_star(
        H, W, C, base_value=0.5, perturbed=[(3, 5, 1, 0.9)], eps=0.3)
    assert istar.nVar == 1

    layer = nn.MaxPool2d(2, 2)
    out = maxpool2d_reach.maxpool2d_star(layer, [istar], method="approx")

    # nVar must stay tiny: 1 input predicate + at most a handful of windows
    # that genuinely straddle the max (here: a single window over one channel).
    assert out[0].nVar <= 3


def test_maxpool_approx_is_sound_over_samples():
    """Approx MaxPool output bounds must contain the concrete MaxPool."""
    H = W = 6
    C = 2
    rng = np.random.default_rng(0)
    perturbed = [(1, 1, 0, 0.7), (4, 2, 1, -0.2), (2, 4, 0, 0.1)]
    istar = _sparse_image_star(H, W, C, base_value=0.3, perturbed=perturbed,
                               eps=0.4)

    layer = nn.MaxPool2d(2, 2)
    out = maxpool2d_reach.maxpool2d_star(layer, [istar], method="approx")[0]
    rl, ru = out.estimate_ranges()
    rl = np.asarray(rl).flatten()
    ru = np.asarray(ru).flatten()

    # Reconstruct the concrete forward by sampling the perturbed pixels.
    base = np.full((H, W, C), 0.3, dtype=np.float64)
    torch_pool = nn.MaxPool2d(2, 2)
    for _ in range(300):
        img = base.copy()
        for (h, w, c, v) in perturbed:
            img[h, w, c] = v + rng.uniform(-0.4, 0.4)
        # ImageStar is HWC; torch MaxPool wants NCHW.
        t = torch.from_numpy(img.transpose(2, 0, 1)[None])
        y = torch_pool(t).numpy()[0].transpose(1, 2, 0).flatten()
        assert np.all(y >= rl - 1e-7)
        assert np.all(y <= ru + 1e-7)

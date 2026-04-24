"""Rotated linear golden-path at production-scale flow training.

Same closed-form benchmark as test_rotated_linear.py, but with the
``flow_config='production'`` training config (hidden=128, n_layers=4,
standardize_outputs=True, Sinkhorn OT coupling on GPU).

Uses 10k training samples × 2000 epochs × batch 2048 — the training-
quality sweet spot on a 3D affine target. Per-benchmark flow training
lands at ~170s on an A30; inference (MC volume) is ~1-2s under the
rk4/30-step FlowScore.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from examples.FlowConformal.benchmarks._common_analytical import (
    print_report, run_pipeline,
)
from examples.FlowConformal.benchmarks.test_rotated_linear import (
    RotatedLinear, _rot_2d, _rot_3d,
)


def main():
    alpha = 0.01

    print('=' * 60)
    print("RotatedLinear 2D (production-scale flow training)")
    print('=' * 60)
    R = _rot_2d(math.pi / 6)
    net = RotatedLinear(R)
    exact = (2.0 ** 2) * (1.0 - alpha)
    bundle = run_pipeline(
        net, x_center=np.zeros(2), radius=1.0, output_dim=2,
        exact_volume=exact, alpha=alpha,
        n_train=10_000, flow_epochs=2000, flow_config='production',
        n_mc_volume=400_000,
    )
    print_report(bundle)

    print('=' * 60)
    print("RotatedLinear 3D (production-scale flow training)")
    print('=' * 60)
    R = _rot_3d()
    net = RotatedLinear(R)
    exact = (2.0 ** 3) * (1.0 - alpha)
    bundle = run_pipeline(
        net, x_center=np.zeros(3), radius=1.0, output_dim=3,
        exact_volume=exact, alpha=alpha,
        n_train=10_000, flow_epochs=2000, flow_config='production',
        n_mc_volume=400_000,
    )
    print_report(bundle)


if __name__ == '__main__':
    main()

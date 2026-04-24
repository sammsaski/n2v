"""Rotated linear network f(x) = R @ x. f_#P_X = Uniform on rotated cube.

Rotation preserves volume so exact volume is still 2^d, but the reach
set is no longer axis-aligned: hyperrect is loose because it pads
around the rotated cube; flow should learn the rotation and match the
true shape; ball is anisotropically loose.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from examples.FlowConformal.benchmarks._common_analytical import (
    print_report, run_pipeline,
)


def _rot_2d(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s], [s, c]])


def _rot_3d() -> torch.Tensor:
    # Arbitrary non-axis rotation: compose two planar rotations.
    a, b = math.pi / 5, math.pi / 7
    Ryz = torch.tensor([[1.0, 0.0, 0.0],
                        [0.0, math.cos(a), -math.sin(a)],
                        [0.0, math.sin(a),  math.cos(a)]])
    Rxz = torch.tensor([[math.cos(b), 0.0, math.sin(b)],
                        [0.0, 1.0, 0.0],
                        [-math.sin(b), 0.0, math.cos(b)]])
    return Rxz @ Ryz


class RotatedLinear(torch.nn.Module):
    def __init__(self, R: torch.Tensor):
        super().__init__()
        self.register_buffer('R', R)

    def forward(self, x):
        return x @ self.R.T


def main():
    alpha = 0.01

    print('=' * 60)
    print("RotatedLinear 2D, angle = 30 deg")
    print('=' * 60)
    R = _rot_2d(math.pi / 6)
    net = RotatedLinear(R)
    exact = (2.0 ** 2) * (1.0 - alpha)
    bundle = run_pipeline(
        net, x_center=np.zeros(2), radius=1.0, output_dim=2,
        exact_volume=exact, alpha=alpha,
    )
    print_report(bundle)

    print('=' * 60)
    print("RotatedLinear 3D")
    print('=' * 60)
    R = _rot_3d()
    net = RotatedLinear(R)
    exact = (2.0 ** 3) * (1.0 - alpha)
    bundle = run_pipeline(
        net, x_center=np.zeros(3), radius=1.0, output_dim=3,
        exact_volume=exact, alpha=alpha,
    )
    print_report(bundle)


if __name__ == '__main__':
    main()

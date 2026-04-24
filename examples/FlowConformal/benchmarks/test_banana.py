"""RotatedBananaNet PoC: tightness vs Star-union ground truth (2D).

Banana net input: uniform on [0, 1]^2 (centered at 0.5, radius 0.5).
Output: 2D curved strip. Hyperrect pads the strip's corners -> loose.
Flow should hug the curve.
"""
from __future__ import annotations

import numpy as np
import torch

from examples.FlowConformal.networks import RotatedBananaNet
from examples.FlowConformal.benchmarks._common import (
    exact_star_union_volume, print_report, run_pipeline,
)


def main():
    torch.manual_seed(0)
    net = RotatedBananaNet().eval()

    x_center = np.array([0.5, 0.5])
    radius = 0.5
    print('Computing Star-union ground truth...')
    star_vol, stars = exact_star_union_volume(
        net, x_center=x_center, radius=radius, output_dim=2,
    )
    print(f'  n_stars = {len(stars)}  Star-union area = {star_vol:.4f}')

    print('=' * 60)
    print('RotatedBananaNet, center=(0.5, 0.5), radius=0.5')
    print('=' * 60)
    bundle = run_pipeline(
        net, x_center=x_center, radius=radius, output_dim=2,
        star_union_volume=star_vol, alpha=0.01,
    )
    print_report(bundle)


if __name__ == '__main__':
    main()

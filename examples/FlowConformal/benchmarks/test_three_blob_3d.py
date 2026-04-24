"""ThreeBlobClassifier3D PoC: tightness vs Star-union ground truth.

Runs the full flow-conformal pipeline on `ThreeBlobClassifier3D` with
`(x_0 = 0, eps = 1.0)` and compares hyperrect / ball / flow conformal
reachsets against the Star-union ground truth (cached MC value 213.72,
re-verified via `verify_exact_caches.py`).

The previous baseline (§9 of the project description) reported flow at
`4.95x exact` on this benchmark; reproducing that number here is the
immediate goal, with the new fast training path (GPU-end-to-end, rk4
inference, 2000 epochs x batch 2048) landing training within
120-180s.
"""
from __future__ import annotations

import numpy as np
import torch

from examples.FlowConformal.networks import ThreeBlobClassifier3D
from examples.FlowConformal.benchmarks._common import (
    exact_star_union_volume, print_report, run_pipeline,
)


def main():
    torch.manual_seed(0)
    net = ThreeBlobClassifier3D().eval()

    print('Computing Star-union ground truth (~30s)...')
    star_vol, stars = exact_star_union_volume(
        net, x_center=np.zeros(3), radius=1.0, output_dim=3,
    )
    print(f'  n_stars = {len(stars)}  Star-union volume = {star_vol:.2f}')

    print('=' * 60)
    print('ThreeBlobClassifier3D, center=(0,0,0), radius=1.0')
    print('=' * 60)
    bundle = run_pipeline(
        net, x_center=np.zeros(3), radius=1.0, output_dim=3,
        star_union_volume=star_vol, alpha=0.01,
    )
    print_report(bundle)


if __name__ == '__main__':
    main()

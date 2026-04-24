"""Identity network f(x) = x. f_#P_X = Uniform([-1, 1]^d).

Analytical 1 - alpha probabilistic reach set volume: approximately
(1 - alpha) * 2^d for axis-aligned scores (hyperrect); for an L2 ball,
the minimum-volume 1 - alpha set of Uniform on a cube is a ball
inscribed / chi-square-adjacent — in practice hyperrect is near-exact
here, flow should match, ball should be looser.
"""

from __future__ import annotations

import numpy as np
import torch

from examples.FlowConformal.benchmarks._common_analytical import (
    print_report, run_pipeline,
)


class IdentityNet(torch.nn.Module):
    def forward(self, x):
        return x


def main():
    alpha = 0.01
    for d in (2, 3):
        print('=' * 60)
        print(f"Identity network, dim = {d}")
        print('=' * 60)
        net = IdentityNet()
        exact = (2.0 ** d) * (1.0 - alpha)
        bundle = run_pipeline(
            net, x_center=np.zeros(d), radius=1.0, output_dim=d,
            exact_volume=exact, alpha=alpha,
        )
        print_report(bundle)


if __name__ == '__main__':
    main()

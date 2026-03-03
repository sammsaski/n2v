"""
Soundness tests for set concatenation.

Tests verify that for random points sampled from the input set,
forwarding through separate PyTorch layers and concatenating produces
outputs contained in the reachable set obtained via _concat_sets.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.nn.reach import _concat_sets
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.sets import Star, Zono, Box


class TestConcatLinearBranchesStarSoundness:
    """Soundness: concat of two linear branches with Star sets."""

    def test_concat_linear_branches_star(self):
        """Two linear layers, forward a Star through each, concat results.
        Sample 200 points, verify np.concatenate([W1(x), W2(x)]) is within bounds."""
        torch.manual_seed(42)

        W1 = nn.Linear(4, 3, bias=True)
        W2 = nn.Linear(4, 2, bias=True)

        # Create Star from bounds [0, 1] dim=4
        lb = np.zeros(4)
        ub = np.ones(4)
        star = Star.from_bounds(lb, ub)

        # Forward through each layer via reach_layer
        reach_W1 = reach_layer(W1, [star], 'approx')
        reach_W2 = reach_layer(W2, [star], 'approx')

        # Concat the two reach sets
        result_sets = _concat_sets([reach_W1, reach_W2], axis=0)
        result = result_sets[0]

        # Get bounds from the result Star
        lb_out, ub_out = result.estimate_ranges()

        # Sample 200 random points from [0, 1]^4 and verify containment
        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(0.0, 1.0, size=(4,))
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out1 = W1(pt_input).numpy().flatten()
                out2 = W2(pt_input).numpy().flatten()
                pt_output = np.concatenate([out1, out2])

            assert np.all(pt_output >= lb_out.flatten() - 1e-5), (
                f"Output {pt_output} below lower bound {lb_out.flatten()}"
            )
            assert np.all(pt_output <= ub_out.flatten() + 1e-5), (
                f"Output {pt_output} above upper bound {ub_out.flatten()}"
            )


class TestConcatLinearBranchesZonoSoundness:
    """Soundness: concat of two linear branches with Zono sets."""

    def test_concat_linear_branches_zono(self):
        """Two linear layers, forward a Zono through each, concat results.
        Sample 200 points, verify containment."""
        torch.manual_seed(99)

        W1 = nn.Linear(5, 3, bias=True)
        W2 = nn.Linear(5, 4, bias=True)

        # Create Zono from bounds [-1, 1] dim=5
        lb = -np.ones(5)
        ub = np.ones(5)
        zono = Zono.from_bounds(lb, ub)

        # Forward through each layer via reach_layer
        reach_W1 = reach_layer(W1, [zono], 'approx')
        reach_W2 = reach_layer(W2, [zono], 'approx')

        # Concat the two reach sets
        result_sets = _concat_sets([reach_W1, reach_W2], axis=0)
        result = result_sets[0]

        # Get bounds from the result Zono
        lb_out, ub_out = result.get_bounds()

        # Sample 200 random points from [-1, 1]^5 and verify containment
        np.random.seed(99)
        for _ in range(200):
            point = np.random.uniform(-1.0, 1.0, size=(5,))
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out1 = W1(pt_input).numpy().flatten()
                out2 = W2(pt_input).numpy().flatten()
                pt_output = np.concatenate([out1, out2])

            assert np.all(pt_output >= lb_out.flatten() - 1e-5), (
                f"Output {pt_output} below lower bound {lb_out.flatten()}"
            )
            assert np.all(pt_output <= ub_out.flatten() + 1e-5), (
                f"Output {pt_output} above upper bound {ub_out.flatten()}"
            )


class TestConcatBoxesSoundness:
    """Soundness: concat two Stars from bounds, sample points, verify."""

    def test_concat_boxes_soundness(self):
        """Concat two Stars from bounds. Sample 200 points, verify containment."""
        lb1 = np.array([0.0, -1.0, 0.5])
        ub1 = np.array([1.0, 1.0, 2.0])
        lb2 = np.array([-2.0, 3.0])
        ub2 = np.array([0.0, 5.0])

        star1 = Star.from_bounds(lb1, ub1)
        star2 = Star.from_bounds(lb2, ub2)

        # Concat
        result_sets = _concat_sets([[star1], [star2]], axis=0)
        result = result_sets[0]

        # Get bounds
        lb_out, ub_out = result.estimate_ranges()

        # Sample 200 random points and verify containment
        np.random.seed(7)
        for _ in range(200):
            # Sample from first input bounds
            alpha1 = np.random.uniform(0.0, 1.0, size=(3,))
            point1 = lb1 + alpha1 * (ub1 - lb1)

            # Sample from second input bounds
            alpha2 = np.random.uniform(0.0, 1.0, size=(2,))
            point2 = lb2 + alpha2 * (ub2 - lb2)

            pt_output = np.concatenate([point1, point2])

            assert np.all(pt_output >= lb_out.flatten() - 1e-5), (
                f"Output {pt_output} below lower bound {lb_out.flatten()}"
            )
            assert np.all(pt_output <= ub_out.flatten() + 1e-5), (
                f"Output {pt_output} above upper bound {ub_out.flatten()}"
            )

"""
Soundness tests for residual (element-wise) set addition.

Tests verify that for random points sampled from the input set,
forwarding through two PyTorch layers and summing produces outputs
contained in the reachable set obtained via _add_sets.
"""

import numpy as np
import torch
import torch.nn as nn
from n2v.nn.reach import _add_sets
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.layer_ops.flatten_reach import flatten_zono
from n2v.sets import Star, ImageStar, ImageZono


class TestResidualAddStarSoundness:
    """Soundness tests for residual add with Star sets."""

    def test_linear_residual_star(self):
        """Two linear layers added: W1(x) + W2(x) should be within reach set bounds."""
        torch.manual_seed(42)

        W1 = nn.Linear(4, 4, bias=False)
        W2 = nn.Linear(4, 4, bias=False)

        # Create Star from bounds [0, 1] dim=4
        lb = np.zeros(4)
        ub = np.ones(4)
        star = Star.from_bounds(lb, ub)

        # Forward through each layer via reach_layer
        reach_W1 = reach_layer(W1, [star], 'approx')
        reach_W2 = reach_layer(W2, [star], 'approx')

        # Add the two reach sets
        result_sets = _add_sets(reach_W1, reach_W2, 'add')
        result = result_sets[0]

        # Get bounds from the result Star
        lb_out, ub_out = result.estimate_ranges()

        # Sample 200 random points from [0, 1]^4 and verify containment
        np.random.seed(42)
        for _ in range(200):
            point = np.random.uniform(0.0, 1.0, size=(4,))
            pt_input = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out1 = W1(pt_input)
                out2 = W2(pt_input)
                pt_output = (out1 + out2).numpy().flatten()

            assert np.all(pt_output >= lb_out.flatten() - 1e-5), (
                f"Output {pt_output} below lower bound {lb_out.flatten()}"
            )
            assert np.all(pt_output <= ub_out.flatten() + 1e-5), (
                f"Output {pt_output} above upper bound {ub_out.flatten()}"
            )


class TestResidualAddImageStarSoundness:
    """Soundness tests for residual add with ImageStar sets."""

    def test_conv_residual_imagestar(self):
        """Two conv layers added: conv1(x) + conv2(x) should be within reach set bounds."""
        torch.manual_seed(42)

        conv1 = nn.Conv2d(1, 2, 3, padding=1)
        conv2 = nn.Conv2d(1, 2, 1)

        # Create ImageStar from bounds [0, 0.5] for 4x4x1 image
        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1)) * 0.5
        img_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        # Forward through each conv layer via reach_layer
        reach_conv1 = reach_layer(conv1, [img_star], 'approx')
        reach_conv2 = reach_layer(conv2, [img_star], 'approx')

        # Add the two reach sets
        result_sets = _add_sets(reach_conv1, reach_conv2, 'add')
        result = result_sets[0]

        # Get bounds from the result ImageStar
        lb_out, ub_out = result.estimate_ranges()

        # Sample 200 random points and verify containment
        np.random.seed(42)
        for _ in range(200):
            # Random point in HWC format within bounds
            point = np.random.uniform(0.0, 0.5, size=(4, 4, 1))

            # Convert to NCHW for PyTorch
            pt_input = torch.tensor(
                point.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32
            )

            with torch.no_grad():
                out1 = conv1(pt_input)  # (1, 2, 4, 4)
                out2 = conv2(pt_input)  # (1, 2, 4, 4)
                pt_output_nchw = (out1 + out2).numpy()

            # Convert PyTorch NCHW output to HWC for comparison with ImageStar bounds
            pt_output_hwc = pt_output_nchw[0].transpose(1, 2, 0)  # (4, 4, 2)
            pt_output_flat = pt_output_hwc.flatten()

            assert np.all(pt_output_flat >= lb_out.flatten() - 1e-5), (
                f"Output below lower bound. "
                f"Min diff: {np.min(pt_output_flat - lb_out.flatten())}"
            )
            assert np.all(pt_output_flat <= ub_out.flatten() + 1e-5), (
                f"Output above upper bound. "
                f"Max diff: {np.max(pt_output_flat - ub_out.flatten())}"
            )


def _assert_contained(model_fns, combine, result, lb_in, ub_in, n=300,
                      input_shape=None, seed=7):
    """House soundness pattern: sample inputs, forward through torch,
    assert every true output lies in the reach set's ranges."""
    lo, hi = result.estimate_ranges()
    lo, hi = lo.flatten(), hi.flatten()
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = rng.uniform(lb_in, ub_in)
        t = torch.tensor(x, dtype=torch.float32)
        if input_shape is not None:
            t = t.reshape(input_shape)
        t = t.unsqueeze(0)
        with torch.no_grad():
            y = combine(*[f(t) for f in model_fns]).numpy().flatten()
        assert np.all(y >= lo - 1e-5) and np.all(y <= hi + 1e-5), (
            f"containment violated: y={y}, lo={lo}, hi={hi}")


class TestResidualAddMismatchedPredicates:
    """Residual adds where the branches do NOT share identical predicate
    systems — the cersyve/cifar100/tinyimagenet/yolo failure class.

    Before task 2.1 these either crashed (different predicate counts) or
    silently took the shared-predicate path on shape alone (equal counts,
    different constraints — unsound)."""

    def test_relu_on_one_branch(self):
        """cersyve shape: relu(W1 x) + W2 x — approx-star ReLU appends
        predicates on one branch only. Used to crash with a broadcast
        error."""
        torch.manual_seed(1)
        W1 = nn.Linear(4, 4)
        W2 = nn.Linear(4, 4)
        relu = nn.ReLU()
        star = Star.from_bounds(np.full(4, -1.0), np.ones(4))

        branch_a = reach_layer(relu, reach_layer(W1, [star], 'approx'),
                               'approx')
        branch_b = reach_layer(W2, [star], 'approx')
        result = _add_sets(branch_a, branch_b, 'add')[0]

        _assert_contained(
            [lambda t: relu(W1(t)), W2], lambda a, b: a + b, result,
            np.full(4, -1.0), np.ones(4))

    def test_relu_on_both_branches(self):
        """ResNet-trunk shape: relu(W1 x) + relu(W2 x) — both branches
        append their own (different) relaxation predicates."""
        torch.manual_seed(2)
        W1 = nn.Linear(3, 5)
        W2 = nn.Linear(3, 5)
        relu = nn.ReLU()
        star = Star.from_bounds(np.full(3, -1.0), np.ones(3))

        branch_a = reach_layer(relu, reach_layer(W1, [star], 'approx'),
                               'approx')
        branch_b = reach_layer(relu, reach_layer(W2, [star], 'approx'),
                               'approx')
        result = _add_sets(branch_a, branch_b, 'add')[0]

        _assert_contained(
            [lambda t: relu(W1(t)), lambda t: relu(W2(t))],
            lambda a, b: a + b, result,
            np.full(3, -1.0), np.ones(3))

    def test_equal_counts_different_constraints(self):
        """relu(x) + relu(-x) = |x| on x in [-1,1]: both branches gain
        exactly ONE relaxation predicate (same nVar) but with DIFFERENT
        constraints. Shape-based predicate sharing is unsound here; the
        constraint systems must be compared, not the shapes."""
        x_id = nn.Linear(1, 1, bias=False)
        x_neg = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            x_id.weight.copy_(torch.tensor([[1.0]]))
            x_neg.weight.copy_(torch.tensor([[-1.0]]))
        relu = nn.ReLU()
        star = Star.from_bounds(np.array([-1.0]), np.array([1.0]))

        branch_a = reach_layer(relu, reach_layer(x_id, [star], 'approx'),
                               'approx')
        branch_b = reach_layer(relu, reach_layer(x_neg, [star], 'approx'),
                               'approx')
        assert branch_a[0].nVar == branch_b[0].nVar  # the trap
        result = _add_sets(branch_a, branch_b, 'add')[0]

        _assert_contained(
            [lambda t: relu(x_id(t)), lambda t: relu(x_neg(t))],
            lambda a, b: a + b, result,
            np.array([-1.0]), np.array([1.0]), n=500)
        # |x| reaches 1.0 at x=+/-1: the set must cover it
        lo, hi = result.estimate_ranges()
        assert hi.flatten()[0] >= 1.0 - 1e-5

    def test_identical_constraints_stay_exact(self):
        """Two LINEAR branches share the ancestor's predicate system
        exactly -> the join must remain the exact shared-predicate sum
        (W1+W2)x, not a loose Minkowski sum."""
        torch.manual_seed(3)
        W1 = nn.Linear(2, 2, bias=False)
        W2 = nn.Linear(2, 2, bias=False)
        star = Star.from_bounds(np.full(2, -1.0), np.ones(2))

        result = _add_sets(reach_layer(W1, [star], 'approx'),
                           reach_layer(W2, [star], 'approx'), 'add')[0]
        lo, hi = result.estimate_ranges()

        Wsum = (W1.weight + W2.weight).detach().numpy()
        exact_hi = np.abs(Wsum).sum(axis=1)   # max of Wsum @ x over the box
        np.testing.assert_allclose(hi.flatten(), exact_hi, atol=1e-6)
        np.testing.assert_allclose(lo.flatten(), -exact_hi, atol=1e-6)

    def test_mismatched_imagestar(self):
        """yolo/cifar shape: conv->relu branch + conv skip on ImageStars."""
        torch.manual_seed(4)
        conv1 = nn.Conv2d(1, 2, 3, padding=1)
        conv2 = nn.Conv2d(1, 2, 1)
        relu = nn.ReLU()
        lb = np.zeros((3, 3, 1))
        ub = np.ones((3, 3, 1)) * 0.5
        img = ImageStar.from_bounds(lb, ub, height=3, width=3,
                                    num_channels=1)

        branch_a = reach_layer(relu, reach_layer(conv1, [img], 'approx'),
                               'approx')
        branch_b = reach_layer(conv2, [img], 'approx')
        result = _add_sets(branch_a, branch_b, 'add')[0]

        lo, hi = result.estimate_ranges()
        lo, hi = lo.flatten(), hi.flatten()
        rng = np.random.default_rng(11)
        for _ in range(200):
            point = rng.uniform(0.0, 0.5, size=(3, 3, 1))
            t = torch.tensor(point.transpose(2, 0, 1)[np.newaxis],
                             dtype=torch.float32)
            with torch.no_grad():
                y = (relu(conv1(t)) + conv2(t)).numpy()
            y = y[0].transpose(1, 2, 0).flatten()
            assert np.all(y >= lo - 1e-5) and np.all(y <= hi + 1e-5)


class TestResidualAddZonoSoundness:
    """Soundness tests for residual add with Zono sets."""

    def test_linear_residual_zono(self):
        """Two linear layers added on flattened ImageZono: W1(x) + W2(x) within bounds."""
        torch.manual_seed(42)

        W1 = nn.Linear(9, 9, bias=True)
        W2 = nn.Linear(9, 9, bias=True)

        # Create ImageZono from bounds [0, 1] for 3x3x1 image
        lb = np.zeros((3, 3, 1))
        ub = np.ones((3, 3, 1))
        img_zono = ImageZono.from_bounds(lb, ub, height=3, width=3, num_channels=1)

        # Flatten via flatten_zono
        flat_zonos = flatten_zono(nn.Flatten(), [img_zono])

        # Forward through each linear layer via reach_layer
        reach_W1 = reach_layer(W1, flat_zonos, 'approx')
        reach_W2 = reach_layer(W2, flat_zonos, 'approx')

        # Add the two reach sets
        result_sets = _add_sets(reach_W1, reach_W2, 'add')
        result = result_sets[0]

        # Get bounds from the result Zono
        lb_out, ub_out = result.get_bounds()

        # Sample 200 random points and verify containment
        np.random.seed(42)
        for _ in range(200):
            # Random point in HWC [0, 1]
            point_hwc = np.random.uniform(0.0, 1.0, size=(3, 3, 1))

            # Flatten in CHW order to match nn.Flatten behavior
            point_chw = point_hwc.transpose(2, 0, 1)  # (1, 3, 3)
            point_flat = point_chw.flatten()

            pt_input = torch.tensor(point_flat, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out1 = W1(pt_input)
                out2 = W2(pt_input)
                pt_output = (out1 + out2).numpy().flatten()

            assert np.all(pt_output >= lb_out.flatten() - 1e-5), (
                f"Output {pt_output} below lower bound {lb_out.flatten()}"
            )
            assert np.all(pt_output <= ub_out.flatten() + 1e-5), (
                f"Output {pt_output} above upper bound {ub_out.flatten()}"
            )

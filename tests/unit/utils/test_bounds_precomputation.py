"""Tests for Zonotope pre-pass bounds precomputation."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono


class TestComputeIntermediateBounds:
    """Tests for compute_intermediate_bounds on Sequential models."""

    def _make_fc_model(self):
        """Create a small FC model: Linear(3,5) -> ReLU -> Linear(5,3) -> ReLU -> Linear(3,2)."""
        model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        model.eval()
        return model

    def test_returns_bounds_at_each_relu(self):
        """Pre-pass should return bounds keyed by layer index for each ReLU."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = self._make_fc_model()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Star.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)

        # model layers: 0=Linear, 1=ReLU, 2=Linear, 3=ReLU, 4=Linear
        # Should have bounds at layer indices 1 and 3 (the two ReLUs)
        assert 1 in layer_bounds, "Should have bounds for first ReLU (layer 1)"
        assert 3 in layer_bounds, "Should have bounds for second ReLU (layer 3)"
        assert len(layer_bounds) == 2

    def test_bounds_shapes_match_layer_dims(self):
        """Bounds should have shape matching the layer input dimension."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = self._make_fc_model()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Star.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)

        # First ReLU input has dim 5 (output of Linear(3,5))
        pre_lb, pre_ub = layer_bounds[1]
        assert pre_lb.shape == (5, 1)
        assert pre_ub.shape == (5, 1)

        # Second ReLU input has dim 3 (output of Linear(5,3))
        pre_lb, pre_ub = layer_bounds[3]
        assert pre_lb.shape == (3, 1)
        assert pre_ub.shape == (3, 1)

    def test_bounds_contain_sampled_points(self):
        """Pre-pass bounds must contain all concrete activations (soundness)."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = self._make_fc_model()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Star.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)

        # Sample 500 points from input range
        np.random.seed(42)
        n_samples = 500
        samples = lb + (ub - lb) * np.random.rand(3, n_samples)

        # Forward through model layer by layer, check each ReLU input
        layers = list(model.children())
        for s in range(n_samples):
            x = torch.tensor(samples[:, s:s+1].T, dtype=torch.float32)
            for i, layer in enumerate(layers):
                if i in layer_bounds:
                    # Check: activation at this point is within bounds
                    activation = x.detach().numpy().flatten()
                    pre_lb, pre_ub = layer_bounds[i]
                    assert np.all(activation >= pre_lb.flatten() - 1e-6), \
                        f"Sample {s} below lower bound at layer {i}"
                    assert np.all(activation <= pre_ub.flatten() + 1e-6), \
                        f"Sample {s} above upper bound at layer {i}"
                x = layer(x)

    def test_lb_leq_ub(self):
        """Lower bounds must be <= upper bounds everywhere."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = self._make_fc_model()
        lb = np.array([[-0.5], [-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5], [0.5]])
        input_set = Star.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)

        for layer_id, (pre_lb, pre_ub) in layer_bounds.items():
            assert np.all(pre_lb <= pre_ub + 1e-10), \
                f"lb > ub at layer {layer_id}"

    def test_accepts_box_input(self):
        """Should accept Box as input set."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = self._make_fc_model()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Box(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)
        assert len(layer_bounds) == 2

    def test_accepts_zono_input(self):
        """Should accept Zono as input set."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = self._make_fc_model()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Zono.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)
        assert len(layer_bounds) == 2

    def test_leakyrelu_detected(self):
        """Should also store bounds before LeakyReLU layers."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        model = nn.Sequential(
            nn.Linear(3, 4),
            nn.LeakyReLU(0.1),
            nn.Linear(4, 2),
        )
        model.eval()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Star.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)
        assert 1 in layer_bounds  # LeakyReLU at index 1
        pre_lb, pre_ub = layer_bounds[1]
        assert pre_lb.shape == (4, 1)

    def test_stable_neurons_identified(self):
        """With biased weights, some neurons should be provably stable."""
        from n2v.utils.bounds_precomputation import compute_intermediate_bounds

        # Create model where first Linear has large positive bias
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )
        model.eval()
        # Set bias so that output of Linear is always positive for small inputs
        with torch.no_grad():
            model[0].weight.copy_(torch.eye(3, 2) * 0.1)
            model[0].bias.copy_(torch.tensor([10.0, -10.0, 0.0]))

        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        input_set = Star.from_bounds(lb, ub)

        layer_bounds = compute_intermediate_bounds(model, input_set)
        pre_lb, pre_ub = layer_bounds[1]

        # Neuron 0: bias=10, weight*input in [-0.1, 0.1] -> always active (lb >= 0)
        assert pre_lb[0, 0] >= 0, "Neuron 0 should be provably active"
        # Neuron 1: bias=-10, weight*input in [-0.1, 0.1] -> always inactive (ub <= 0)
        assert pre_ub[1, 0] <= 0, "Neuron 1 should be provably inactive"


# --------------------------------------------------------------------------- #
# Helpers of the pass, with values written down by hand                         #
# --------------------------------------------------------------------------- #
from fractions import Fraction  # noqa: E402

import torch.fx as fx  # noqa: E402

from n2v.utils import bounds_precomputation as bp  # noqa: E402

_ETA = 2.0 ** -1074


class TestPrePassHelpers:

    def test_recorded_bounds_of_an_image_layer_are_reordered_to_its_rows(self):
        """A (C, H, W) = (2, 2, 3) layer's channel-major bounds come back in
        (height, width, channel) order; a flat layer (shape None) and a
        (C, L) layer of a Conv1d come back as they are; both as columns,
        lower first. Catches a wrong permutation, a reorder applied to a
        rank-2 shape, and the ends swapped or duplicated."""
        lb = np.arange(12, dtype=np.float64)          # value = c*6 + h*3 + w
        ub = lb + 100.0
        lo, hi = bp._record_spatial_bounds(lb, ub, (2, 2, 3))
        expected = [c * 6 + h * 3 + w for h in range(2) for w in range(3)
                    for c in range(2)]
        assert lo.shape == hi.shape == (12, 1)
        assert lo.ravel().tolist() == expected
        assert hi.ravel().tolist() == [v + 100.0 for v in expected]
        for shape in (None, (2, 6)):
            lo, hi = bp._record_spatial_bounds(lb, ub, shape)
            assert lo.ravel().tolist() == lb.tolist()
            assert hi.ravel().tolist() == ub.tolist()

    def test_layer_bounds_hull_every_set_and_widen_by_the_allowance(self):
        """Two sets entering one layer: the recorded interval holds both
        sets' row bounds; with an allowance of the layer's size it is
        widened outward by at least that much on each side; an allowance
        that is None or of another size is not applied. Catches a hull of
        the wrong ends or of the first set only, and an allowance applied
        inward or from one side only."""
        b1 = (np.array([-1.0, 0.5, 2.0]), np.array([1.0, 0.75, 3.0]))
        b2 = (np.array([-0.5, -4.0, 2.5]), np.array([2.0, 0.25, 2.75]))
        err = np.array([2.0 ** -40, 0.0, 0.5])
        hull_lo = np.array([-1.0, -4.0, 2.0])
        hull_hi = np.array([2.0, 0.75, 3.0])
        lo, hi = bp._zono_layer_bounds([b1, b2], err)
        assert lo.shape == hi.shape == (3, 1)
        for i in range(3):
            assert Fraction(lo[i, 0]) <= Fraction(hull_lo[i]) - Fraction(err[i])
            assert Fraction(hi[i, 0]) >= Fraction(hull_hi[i]) + Fraction(err[i])
        for e in (None, np.zeros(2)):
            lo, hi = bp._zono_layer_bounds([b1, b2], e)
            assert np.all(lo.ravel() <= hull_lo) and np.all(hi.ravel() >= hull_hi)

    def test_row_bounds_hold_a_zonotope_and_a_set_that_is_not_one(self):
        """The row bounds of a Zono hold c +- sum |V| (exact for these
        dyadic numbers); of a Star (a set the zonotope reach does not
        produce today) they hold its range. Catches a generator range of
        [0, 1] instead of [-1, 1] and ends swapped."""
        z = Zono(np.array([[0.5], [-2.0]]),
                 np.array([[1.0, -0.25, 0.0], [0.5, 0.5, -3.0]]))
        lo, hi = bp._zono_row_bounds(z)
        assert lo[0] <= -0.75 and hi[0] >= 1.75
        assert lo[1] <= -6.0 and hi[1] >= 2.0
        s = Star.from_bounds(np.array([[-3.0], [1.0]]), np.array([[2.0], [5.0]]))
        lo, hi = bp._zono_row_bounds(s)
        assert lo[0] <= -3.0 and hi[0] >= 2.0 and lo[1] <= 1.0 and hi[1] >= 5.0

    def test_magnitude_is_the_largest_over_the_sets(self):
        """Per row the largest |c| + sum |V| over the sets: (3, 9) here.
        Catches a magnitude without the centre or of the first set only."""
        z1 = Zono(np.array([[-1.0], [2.0]]),
                  np.array([[0.5, -1.5], [1.0, 0.0]]))
        z2 = Zono(np.array([[0.5], [-4.0]]),
                  np.array([[2.5, 0.0], [-1.0, 4.0]]))
        mag = bp._zono_magnitude([z1, z2])
        assert mag[0] >= 3.0 and mag[1] >= 9.0

    def test_underflow_allowance_counts_every_product(self):
        """``_SAFETY * (n_gens + 1) * count`` half-subnormals per row,
        computed exactly: 0, 4 * 4 * 1 and 4 * 4 * 5 subnormals for counts
        0, 1 and 5 with three generators. Catches one allowance per inner
        product instead of one per product."""
        out = bp._underflow_error(3, np.array([0.0, 1.0, 5.0]))
        assert out[0] == 0.0
        assert out[1] >= 16 * _ETA and out[2] >= 80 * _ETA

    def test_nonzero_products_count_the_live_columns(self):
        """Per row the nonzero weights on live columns, counted exactly;
        none live gives zeros. Catches the dead columns counted instead and
        a count over every column."""
        W = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]])
        live = np.array([True, False, True])
        assert bp._nonzero_products(W, live).tolist() == [2.0, 1.0, 1.0]
        assert bp._nonzero_products(W, np.zeros(3, bool)).tolist() == [0.0] * 3

    def test_error_step_keeps_the_allowance_through_a_pass_through_layer(self):
        """Output sets that are the input sets (an Identity or Dropout) keep
        the allowance as it is; an unknown allowance stays unknown. Catches
        an allowance dropped or reset at such a layer."""
        z = Zono(np.array([[1.0], [2.0]]), np.array([[1.0], [0.5]]))
        err = np.array([2.0 ** -50, 2.0 ** -48])
        out = bp._zono_error_step(nn.Identity(), [z], [z], err)
        assert out is not None and np.array_equal(out, err)
        assert bp._zono_error_step(nn.Linear(2, 2).double(), [z], [z], None) \
            is None

    def test_zono_method_runs_the_zonotope_pass(self):
        """On ``y = x - x`` (Linear(1, 2) of (1, 1), Linear(2, 1) of
        (1, -1), ReLU) over x in [-1, 1] the zonotope pass keeps the
        correlation and records an interval within 1e-12 of [0, 0], where
        intervals give [-2, 2]: the observable of the option. Catches
        ``method='zono'`` running the interval pass (an ignored option)."""
        l1 = nn.Linear(1, 2, bias=False).double()
        l2 = nn.Linear(2, 1, bias=False).double()
        with torch.no_grad():
            l1.weight.copy_(torch.tensor([[1.0], [1.0]], dtype=torch.float64))
            l2.weight.copy_(torch.tensor([[1.0, -1.0]], dtype=torch.float64))
        model = nn.Sequential(l1, l2, nn.ReLU())
        S = Star.from_bounds(np.array([[-1.0]]), np.array([[1.0]]))
        for target in (model, fx.symbolic_trace(model)):
            zono = list(bp.compute_intermediate_bounds(target, S, 'zono')
                        .values())[0]
            ibp = list(bp.compute_intermediate_bounds(target, S, 'ibp')
                       .values())[0]
            assert zono[0][0, 0] <= 0.0 <= zono[1][0, 0]
            assert zono[1][0, 0] - zono[0][0, 0] <= 1e-12
            assert ibp[0][0, 0] <= -2.0 and ibp[1][0, 0] >= 2.0

    def test_zonotope_pass_keeps_the_bounds_before_a_node_it_cannot_reach(
            self):
        """A traced model with ``torch.cumsum`` (no reach rule) after the
        first ReLU: the zonotope pass records the first ReLU, whose bounds
        hold its input, and nothing behind the cumsum. Catches a pass that
        drops everything when it stops, and one that records layers behind
        a node it could not reach."""
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1, self.fc2 = nn.Linear(2, 2), nn.Linear(2, 2)
                self.act1, self.act2 = nn.ReLU(), nn.ReLU()

            def forward(self, x):
                return self.act2(self.fc2(torch.cumsum(self.act1(
                    self.fc1(x)), 1)))
        net = Net().double()
        with torch.no_grad():
            net.fc1.weight.copy_(torch.tensor([[1.0, -1.0], [0.5, 2.0]]))
            net.fc1.bias.copy_(torch.tensor([0.25, -0.5]))
        S = Star.from_bounds(np.array([[-1.0], [-2.0]]),
                             np.array([[3.0], [1.0]]))
        bounds = bp.compute_intermediate_bounds(fx.symbolic_trace(net), S,
                                                method='zono')
        assert list(bounds) == ['act1']
        lo, hi = bounds['act1']
        # fc1 over the box: row 0 in [-1.75, 5.25], row 1 in [-5, 3]
        assert lo[0, 0] <= -1.75 and hi[0, 0] >= 5.25
        assert lo[1, 0] <= -5.0 and hi[1, 0] >= 3.0

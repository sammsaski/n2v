"""
Integration tests for probabilistic verification with n2v.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from n2v import ProbabilisticBox, probabilistic, Box, Star
from n2v.nn import NeuralNetwork
from n2v.probabilistic import ConformalReachConfig
from n2v.sets.image_star import ImageStar
from n2v.sets.probabilistic_box import ProbabilisticBox as ProbabilisticBoxDirect
from tests.known_bugs import known_bug


class TestImports:
    """Tests for module imports."""

    def test_probabilistic_box_importable_from_n2v_sets(self):
        """Test that ProbabilisticBox can be imported from n2v.sets."""
        from n2v.sets import ProbabilisticBox as PB
        assert PB is not None

    def test_probabilistic_box_importable_from_n2v(self):
        """Test that ProbabilisticBox can be imported from n2v."""
        from n2v import ProbabilisticBox as PB
        assert PB is not None

    def test_probabilistic_module_importable_from_n2v(self):
        """Test that probabilistic module can be imported from n2v."""
        from n2v import probabilistic
        assert hasattr(probabilistic, 'conformal_reach')

    def test_conformal_reach_importable_from_probabilistic(self):
        """Test that conformal_reach can be imported from n2v.probabilistic."""
        from n2v.probabilistic import conformal_reach
        assert callable(conformal_reach)


class TestNeuralNetworkReach:
    """Tests for NeuralNetwork.reach() with probabilistic methods."""

    def test_reach_probabilistic_method(self):
        """Test NeuralNetwork.reach() with method='probabilistic'."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

        # Create verifier
        net = NeuralNetwork(model)

        # Create input set
        lb = np.zeros(3)
        ub = np.ones(3)
        input_box = Box(lb, ub)

        # Run probabilistic verification
        result = net.reach(
            input_box,
            method='probabilistic',
            m=50,
            epsilon=0.1,
            seed=42
        )

        # Should return list with ProbabilisticBox
        assert len(result) == 1
        assert isinstance(result[0], ProbabilisticBox)
        assert result[0].dim == 2  # Output dimension
        assert result[0].epsilon == 0.1

    def test_reach_probabilistic_with_star_input(self):
        """Test that probabilistic method accepts Star input."""
        model = nn.Sequential(nn.Linear(2, 2))

        net = NeuralNetwork(model)

        # Create Star input
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        input_star = Box(lb, ub).to_star()

        # Should convert Star to Box internally and work
        result = net.reach(
            input_star,
            method='probabilistic',
            m=30,
            epsilon=0.1,
            seed=42
        )

        assert len(result) == 1
        assert isinstance(result[0], ProbabilisticBox)

    def test_reach_probabilistic_preserves_params(self):
        """Test that m, ell, epsilon are correctly passed through."""
        model = nn.Linear(2, 2)
        net = NeuralNetwork(model)
        input_box = Box(np.zeros(2), np.ones(2))

        result = net.reach(
            input_box,
            method='probabilistic',
            m=100,
            ell=95,
            epsilon=0.05,
            seed=42
        )

        pbox = result[0]
        assert pbox.m == 100
        assert pbox.ell == 95
        assert pbox.epsilon == 0.05


# method='hybrid' was covered here by two tests that asserted only
# ``len(result) >= 1``. The method drops skip connections and is to be removed;
# the contract is tests/contracts/test_verify_api.py
# (``test_hybrid_reach_method_is_removed``).


# verify_specification on a ProbabilisticBox was covered here by a test that
# put the whole box inside the unsafe region, silenced every warning and
# accepted any of the three verdicts, and ``isinstance(pbox, Box)`` was
# asserted by a second one. A probabilistic box is not a sound reach set; the
# contract is tests/contracts/test_probabilistic_contract.py
# (``test_probabilistic_box_that_meets_the_unsafe_region_is_not_unsat``,
# ``test_unsat_from_a_probabilistic_box_says_it_is_probabilistic`` and
# ``test_probabilistic_box_is_not_a_box``).


class TestProbabilisticBoxOperations:
    """Tests for ProbabilisticBox operations in the context of n2v."""

    def test_sample_from_probabilistic_box(self):
        """Test sampling from ProbabilisticBox."""
        lb = np.array([0.0, 0.0, 0.0])
        ub = np.array([1.0, 1.0, 1.0])
        pbox = ProbabilisticBox(lb, ub, m=100, ell=99, epsilon=0.01)

        samples = pbox.sample(100)

        assert samples.shape == (3, 100)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_complete_probabilistic_workflow(self):
        """Test complete probabilistic verification workflow."""
        # Create model
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
        model.eval()

        # Create verifier
        net = NeuralNetwork(model)

        # Define input region
        center = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        epsilon = 0.1
        input_box = Box(center - epsilon, center + epsilon)

        # Run probabilistic verification
        result = net.reach(
            input_box,
            method='probabilistic',
            m=100,
            epsilon=0.05,
            surrogate='naive',
            seed=42
        )

        # Verify result
        pbox = result[0]
        assert isinstance(pbox, ProbabilisticBox)
        assert pbox.dim == 3
        assert pbox.coverage == 0.95  # 1 - epsilon

        # Check bounds make sense
        lb, ub = pbox.get_range()
        assert np.all(ub >= lb)

    def test_probabilistic_then_deterministic_comparison(self):
        """Compare probabilistic bounds with deterministic bounds."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        model.eval()

        net = NeuralNetwork(model)
        input_box = Box(np.zeros(2), np.ones(2) * 0.5)

        # Probabilistic
        prob_result = net.reach(
            input_box,
            method='probabilistic',
            m=500,
            epsilon=0.001,
            seed=42
        )

        # Deterministic (approx)
        det_result = net.reach(input_box.to_star(), method='approx')

        # Both should give valid results
        assert len(prob_result) >= 1
        assert len(det_result) >= 1

        # Note: We can't directly compare bounds since deterministic
        # is sound overapproximation while probabilistic is empirical
        # with coverage guarantee



class _BudgetModule(nn.Module):
    """y = W x + b on 3 inputs, 2 outputs; keeps every input and output batch
    and fails the test once it is asked for more rows than ``budget`` (a run
    that drops the requested sizes then fails at its first batch)."""

    def __init__(self, budget):
        super().__init__()
        self.lin = nn.Linear(3, 2)
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor([[1.5, -0.75, 2.0],
                                                [-0.5, 2.0, 1.25]]))
            self.lin.bias.copy_(torch.tensor([4.0, -6.0]))
        self.budget = budget
        self.inputs, self.outputs = [], []

    def forward(self, x):
        done = sum(len(b) for b in self.inputs)
        if done + len(x) > self.budget:
            pytest.fail(f"the model was asked for {done + len(x)} rows, more "
                        f"than the {self.budget} the requested sizes need")
        y = self.lin(x)
        self.inputs.append(x.detach().clone())
        self.outputs.append(y.detach().clone())
        return y


_BOX3 = Box(np.array([-3.0, 0.5, -1.5]), np.array([1.0, 4.0, -0.25]))


class TestConformalMethod:
    """NeuralNetwork.reach(method='conformal') runs conformal_reach."""

    @pytest.mark.parametrize("route", ["config", "keywords"])
    def test_reach_conformal_returns_the_calibrated_box(self, route):
        """The public route of conformal_reach, with config= and with keyword
        options: the module is evaluated on 10 training and 30 calibration
        draws of the box (no more: it fails the test when asked for more), the
        result is one ProbabilisticBox stating (30, 28, 0.25), and with the
        naive surrogate it holds exactly ell = 28 of the 30 calibration
        outputs the module returned (a pad of 1e-12 of the output scale for
        float64 arithmetic). Catches a method name or options that are not
        handed on to conformal_reach (the default config draws 4000 training
        rows; a sound method returns a list of sets), and every wrong rank,
        inflation or centre of the box on this route."""
        options = dict(m=30, ell=28, epsilon=0.25, surrogate="naive",
                       training_samples=10, seed=4)
        module = _BudgetModule(40)
        net = NeuralNetwork(module)
        if route == "config":
            result = net.reach(_BOX3, method="conformal",
                               config=ConformalReachConfig(**options))
        else:
            result = net.reach(_BOX3, method="conformal", **options)
        assert isinstance(result, ProbabilisticBox)
        assert (result.m, result.ell, result.epsilon) == (30, 28, 0.25)
        x = torch.cat(module.inputs).numpy()
        y = torch.cat(module.outputs).numpy().astype(np.float64)
        assert x.shape == (40, 3)
        lb, ub = result.lb.ravel(), result.ub.ravel()
        pad = 1e-12 * (1.0 + np.max(np.abs(y)))
        held = lambda lo, hi: int(np.sum(np.all((y[10:] >= lo) & (y[10:] <= hi),
                                                axis=1)))
        assert held(lb - pad, ub + pad) >= 28 and held(lb + pad, ub - pad) <= 28

    def test_reach_conformal_refuses_a_star(self):
        """method='conformal' draws inputs from a Box; a Star input is refused
        with TypeError before any model call, not converted silently."""
        net = NeuralNetwork(_BudgetModule(0))
        with pytest.raises(TypeError, match="requires Box input"):
            net.reach(_BOX3.to_star(), method="conformal", m=30)


class _RecordingFlatten(nn.Module):
    """Flatten that keeps every input tensor it is evaluated on."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return x.flatten(1)


class TestProbabilisticAliasOnImageSets:
    """The legacy alias method='probabilistic' on an image input set."""

    @pytest.mark.parametrize("chw", [
        pytest.param((2, 3, 5), id="2x3x5", marks=known_bug(
            4, "reach(method='probabilistic') of an ImageStar samples the box "
               "of estimate_ranges, which is in the stored (H, W, C) order, and "
               "reshapes the samples to (C, H, W), so the model is evaluated on "
               "permuted pixels and the stated box misses the true outputs "
               "(registry row PROB-15; REACH-2 says the alias must refuse)")),
        pytest.param((1, 3, 5), id="1x3x5-one-channel"),
    ])
    def test_alias_evaluates_the_model_on_the_image_region(self, chw):
        """The promise of the alias for an ImageStar: either it refuses (REACH-2
        says the alias must raise ValueError naming method='conformal'), or the
        model is evaluated only on members of the input region, as (C, H, W)
        tensors, and the stated box meets the true interval of every output.
        Model: Flatten, so output k is entry k of the (C, H, W) tensor and its
        true interval is the input interval of that entry, written down here.
        Every pixel has its own interval, of width 0.125 on a grid of step
        0.25 from -3.75 to 3.5 (both signs, beyond plus and minus 1), so a
        pixel read from the stored (H, W, C) order lies outside its own
        interval. Deterministic: inputs drawn from the region always lie in it,
        and a box that holds at least one calibration output meets every true
        interval. Catches samples drawn in the stored order and fed in the
        tensor order (pixels permuted between channels). The one-channel case
        is the passing neighbour: there the two orders coincide."""
        C, H, W = chw
        lo = (np.arange(C * H * W) * 0.25 - 3.75).reshape(C, H, W)
        hi = lo + 0.125
        image = ImageStar.from_bounds(lo.transpose(1, 2, 0), hi.transpose(1, 2, 0),
                                      H, W, C)
        module = _RecordingFlatten()
        try:
            result = NeuralNetwork(module).reach(
                image, method="probabilistic", m=30, ell=29, epsilon=0.25,
                surrogate="naive", training_samples=10, seed=0)
        except NotImplementedError:
            return  # a refusal keeps the promise
        except ValueError as refusal:
            if "conformal" not in str(refusal):
                raise
            return  # the refusal that REACH-2 asks for keeps the promise
        if len(module.inputs) == 0 or not len(result) == 1:
            pytest.fail("the alias answered without evaluating the model once, "
                        "or with more than one set")
        x = torch.cat(module.inputs).numpy().astype(np.float64)
        assert x.shape == (40, C, H, W), f"model inputs of shape {x.shape}"
        outside = int(np.sum(np.any((x < lo) | (x > hi), axis=(1, 2, 3))))
        assert outside == 0, (
            f"{outside} of {len(x)} model inputs lie outside the input region")
        lb, ub = result[0].lb.ravel(), result[0].ub.ravel()
        disjoint = int(np.sum((ub < lo.ravel()) | (lb > hi.ravel())))
        assert disjoint == 0, (
            f"{disjoint} of {lb.size} outputs have a true interval disjoint from "
            f"the stated box")

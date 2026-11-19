"""Tests for reach pipeline with precompute_bounds option."""

import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star
from n2v.nn import NeuralNetwork
from n2v.nn.reach import reach_pytorch_model


class TestReachWithPrecomputeBounds:

    def _make_model(self):
        model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        model.eval()
        return model

    def test_precompute_bounds_produces_valid_output(self):
        """reach with precompute_bounds=True should produce valid Star sets."""
        model = self._make_model()
        lb = np.array([[-1.0], [-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_set = Star.from_bounds(lb, ub)

        result = reach_pytorch_model(
            model, input_set, method='approx', precompute_bounds=True
        )
        assert len(result) >= 1
        assert result[0].V is not None

    def test_precompute_bounds_is_sound(self):
        """Output with precompute_bounds must contain all concrete outputs."""
        model = self._make_model()
        lb = np.array([[-0.5], [-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5], [0.5]])
        input_set = Star.from_bounds(lb, ub)

        result = reach_pytorch_model(
            model, input_set, method='approx', precompute_bounds=True
        )

        # Sample and verify containment
        np.random.seed(42)
        n_samples = 500
        samples = lb + (ub - lb) * np.random.rand(3, n_samples)

        out_lb, out_ub = result[0].estimate_ranges()

        for i in range(n_samples):
            x = torch.tensor(samples[:, i:i+1].T, dtype=torch.float32)
            with torch.no_grad():
                y = model(x).numpy().flatten()
            assert np.all(y >= out_lb.flatten() - 1e-5), \
                f"Sample {i} output below lower bound"
            assert np.all(y <= out_ub.flatten() + 1e-5), \
                f"Sample {i} output above upper bound"

    def test_precompute_false_is_default(self):
        """precompute_bounds=False should match default behavior exactly."""
        model = self._make_model()
        lb = np.array([[-0.5], [-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5], [0.5]])
        input_set = Star.from_bounds(lb, ub)

        result_default = reach_pytorch_model(model, input_set, method='approx')
        result_false = reach_pytorch_model(
            model, input_set, method='approx', precompute_bounds=False
        )

        lb_d, ub_d = result_default[0].estimate_ranges()
        lb_f, ub_f = result_false[0].estimate_ranges()
        np.testing.assert_allclose(lb_d, lb_f, atol=1e-10)
        np.testing.assert_allclose(ub_d, ub_f, atol=1e-10)

    def test_exact_with_precompute(self):
        """Exact method with precompute_bounds should also work."""
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )
        model.eval()
        lb = np.array([[-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5]])
        input_set = Star.from_bounds(lb, ub)

        result = reach_pytorch_model(
            model, input_set, method='exact', precompute_bounds=True
        )
        assert len(result) >= 1

    def test_neural_network_reach_forwards_precompute(self):
        """NeuralNetwork.reach() should forward precompute_bounds."""
        model = self._make_model()
        net = NeuralNetwork(model)
        lb = np.array([[-0.5], [-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5], [0.5]])
        input_set = Star.from_bounds(lb, ub)

        # Should not raise
        result = net.reach(input_set, method='approx', precompute_bounds=True)
        assert len(result) >= 1


# --------------------------------------------------------------------------- #
# What reach hands the pre-pass, and what it hands each layer                  #
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402
import torch.nn.functional as TF  # noqa: E402

import n2v.nn.reach as reach_module  # noqa: E402
from n2v.sets.image_star import ImageStar  # noqa: E402


def _spy_prepass(monkeypatch):
    """Record every call of the pre-pass that reach makes (it still runs)."""
    calls = []
    real = reach_module.compute_intermediate_bounds

    def spy(model, input_set, method='ibp', input_shape=None):
        bounds = real(model, input_set, method=method, input_shape=input_shape)
        calls.append({'input_set': input_set, 'method': method,
                      'input_shape': input_shape, 'bounds': bounds})
        return bounds
    monkeypatch.setattr(reach_module, 'compute_intermediate_bounds', spy)
    return calls


def _spy_layers(monkeypatch):
    """Record ``(layer, precomputed_bounds)`` for every layer the main reach
    hands a Star list (the zonotope pass's own calls carry zonotopes)."""
    seen = []
    real = reach_module.reach_layer

    def spy(layer, input_sets, method='exact', **kwargs):
        if isinstance(input_sets[0], Star):
            seen.append((layer, kwargs.get('precomputed_bounds')))
        return real(layer, input_sets, method, **kwargs)
    monkeypatch.setattr(reach_module, 'reach_layer', spy)
    return seen


class _Functional(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = (nn.Linear(3, 4), nn.Linear(4, 4),
                                        nn.Linear(4, 2))

    def forward(self, x):
        x = TF.relu(self.fc1(x))
        return self.fc3(TF.leaky_relu(self.fc2(x), 0.25))


class TestPrecomputeHandover:

    @pytest.mark.parametrize('option, expected', [
        (True, 'ibp'), ('ibp', 'ibp'), ('zono', 'zono'), (1, 'ibp'),
        (False, None), (None, None), (0, None), ('', None)])
    def test_the_option_runs_the_named_pass_once_on_the_input_set(
            self, option, expected, monkeypatch):
        """``precompute_bounds`` True (and any other value that is not a
        string, such as 1) runs the interval pass, a string names the pass,
        and a value Python reads as false (False, None, 0, '') runs none.
        The pass runs once per reach, on the reach's own input set (a Star
        with a finite predicate box is handed over as it is), with the
        ``input_shape`` the caller gave.

        Catches an option spelling mapped to the wrong pass or not
        switching it on or off, a pass that runs per layer or on another
        set, and an ``input_shape`` not handed on."""
        calls = _spy_prepass(monkeypatch)
        model = TestReachWithPrecomputeBounds()._make_model()
        input_set = Star.from_bounds(np.array([[-0.5], [-2.0], [0.25]]),
                                     np.array([[0.5], [1.5], [3.0]]))
        reach_pytorch_model(model, input_set, method='approx',
                            precompute_bounds=option, input_shape=(3,))
        if expected is None:
            assert calls == []
            return
        assert len(calls) == 1
        assert calls[0]['method'] == expected
        assert calls[0]['input_set'] is input_set
        assert calls[0]['input_shape'] == (3,)

    def test_an_image_input_and_its_shape_reach_the_pass(self, monkeypatch):
        """An ImageStar is handed to the pass as it is, with the caller's
        ``input_shape`` (the zonotope pass reads tensor shapes from it).
        Catches a reach that hands the pass a flattened or converted set
        or drops the shape."""
        calls = _spy_prepass(monkeypatch)
        model = nn.Sequential(nn.Conv2d(2, 2, (1, 2)), nn.ReLU(),
                              nn.Flatten(), nn.Linear(8, 2)).eval()
        lo = np.zeros((2, 3, 2))
        S = ImageStar.from_bounds(lo, lo + 1.0, height=2, width=3,
                                  num_channels=2)
        reach_pytorch_model(model, S, method='approx',
                            precompute_bounds='zono', input_shape=(2, 2, 3))
        assert len(calls) == 1
        assert calls[0]['input_set'] is S
        assert calls[0]['input_shape'] == (2, 2, 3)
        assert calls[0]['method'] == 'zono'

    @pytest.mark.parametrize('pre', ['ibp', 'zono'])
    @pytest.mark.parametrize('method', ['approx', 'exact'])
    @pytest.mark.parametrize('kind', ['sequential', 'functional'])
    def test_each_activation_is_handed_the_bounds_recorded_for_its_node(
            self, kind, method, pre, monkeypatch):
        """Linear, ReLU, Linear, LeakyReLU, Linear as an ``nn.Sequential``
        (call_module nodes) and with ``F.relu`` and ``F.leaky_relu``
        (call_function nodes): the pass records one entry per activation,
        and reach hands each activation exactly the entry recorded under
        its own node, the very arrays, and hands no bounds to the Linear
        layers.

        Catches bounds handed to the wrong node or under the producer's
        name, bounds dropped for functional activations, bounds handed to
        every layer, and bounds copied or recomputed on the way."""
        calls = _spy_prepass(monkeypatch)
        seen = _spy_layers(monkeypatch)
        model = (_Functional() if kind == 'functional' else nn.Sequential(
            nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 4), nn.LeakyReLU(0.25),
            nn.Linear(4, 2))).eval()
        input_set = Star.from_bounds(-np.ones((3, 1)), np.ones((3, 1)))
        reach_pytorch_model(model, input_set, method=method,
                            precompute_bounds=pre)
        assert len(calls) == 1
        recorded = list(calls[0]['bounds'].values())
        assert len(recorded) == 2, sorted(calls[0]['bounds'])
        assert [type(layer).__name__ for layer, _ in seen] == [
            'Linear', 'ReLU', 'Linear', 'LeakyReLU', 'Linear']
        handed = [bounds for _, bounds in seen]
        assert handed[0] is None and handed[2] is None and handed[4] is None
        assert handed[1] is recorded[0] and handed[3] is recorded[1]

    @pytest.mark.parametrize('option', ['IBP', 'interval', 'zonotope'])
    def test_an_unknown_pass_name_is_refused(self, option):
        """A ``precompute_bounds`` string that names no pass raises
        ``ValueError`` through reach. Catches an unknown name silently
        read as one of the passes or as off."""
        model = TestReachWithPrecomputeBounds()._make_model()
        input_set = Star.from_bounds(-np.ones((3, 1)), np.ones((3, 1)))
        with pytest.raises(ValueError, match='precompute'):
            reach_pytorch_model(model, input_set, method='approx',
                                precompute_bounds=option)

    @pytest.mark.parametrize('option', ['ibp', 'zono'])
    def test_neural_network_reach_hands_the_option_to_the_pass(
            self, option, monkeypatch):
        """``NeuralNetwork.reach`` hands ``precompute_bounds`` on: the pass
        named by the option runs once, on the input set. Catches the entry
        route dropping or renaming the option (the case above only checks
        that some result comes back)."""
        calls = _spy_prepass(monkeypatch)
        net = NeuralNetwork(TestReachWithPrecomputeBounds()._make_model())
        input_set = Star.from_bounds(-np.ones((3, 1)), np.ones((3, 1)))
        net.reach(input_set, method='approx', precompute_bounds=option)
        assert [c['method'] for c in calls] == [option]
        assert calls[0]['input_set'] is input_set

"""Unit tests for synthetic test networks.

Each network trains in its constructor, about three seconds. The tests only
read the trained network, so each network is built once for the module, from
seed 0, the seed the accuracy checks below already used."""

import pytest
import torch


def _trained(name):
    """The example network ``name`` as its default constructor trains it,
    from seed 0. The global RNG is put back afterwards so the construction
    leaks into no test."""
    from examples.FlowConformal import networks

    state = torch.get_rng_state()
    torch.manual_seed(0)
    try:
        return getattr(networks, name)()
    finally:
        torch.set_rng_state(state)


@pytest.fixture(scope="module")
def banana_net():
    return _trained("RotatedBananaNet")


@pytest.fixture(scope="module")
def blob_net():
    return _trained("ThreeBlobClassifier")


@pytest.fixture(scope="module")
def blob_net_3d():
    return _trained("ThreeBlobClassifier3D")


class TestRotatedBananaNet:
    """Tests for RotatedBananaNet."""

    def test_output_shape(self, banana_net):
        """Network should map R^2 -> R^2."""
        x = torch.rand(10, 2)
        y = banana_net(x)
        assert y.shape == (10, 2)

    def test_produces_banana_shape(self, banana_net):
        """Output of unit square should have parabolic structure.

        y1 ≈ x1, y2 ≈ x1^2 + 0.3*x2.
        So for inputs in [0,1]^2:
        - y1 range should be approximately [0, 1]
        - y2 should have a quadratic relationship with y1
        """
        torch.manual_seed(0)
        x = torch.rand(5000, 2)
        with torch.no_grad():
            y = banana_net(x)

        # y1 should span roughly [0, 1]
        assert y[:, 0].min() < 0.2
        assert y[:, 0].max() > 0.8

        # y2 should correlate nonlinearly with y1 (not just linear)
        # Check that y2 at high y1 is significantly larger than at low y1
        low_y1 = y[y[:, 0] < 0.3]
        high_y1 = y[y[:, 0] > 0.7]
        assert high_y1[:, 1].mean() > low_y1[:, 1].mean()

    def test_output_is_deterministic(self, banana_net):
        """Same input should give same output."""
        x = torch.tensor([[0.5, 0.5]])
        y1 = banana_net(x)
        y2 = banana_net(x)
        torch.testing.assert_close(y1, y2)


class TestThreeBlobClassifier:
    """Tests for ThreeBlobClassifier."""

    def test_output_shape(self, blob_net):
        """Network should map R^2 -> R^3 logits."""
        x = torch.randn(10, 2)
        logits = blob_net(x)
        assert logits.shape == (10, 3)

    def test_accuracy_on_training_distribution(self, blob_net):
        """Trained classifier should achieve >= 90% on its own distribution."""
        x, y = blob_net.sample_data(1000, seed=42)
        with torch.no_grad():
            preds = blob_net(x).argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
        assert accuracy >= 0.9

    def test_sample_data_shapes(self, blob_net):
        """sample_data should return (x, y) of matching shapes."""
        x, y = blob_net.sample_data(100, seed=0)
        assert x.shape == (100, 2)
        assert y.shape == (100,)


class TestThreeBlobClassifier3D:
    """Tests for ThreeBlobClassifier3D."""

    def test_forward_shape(self, blob_net_3d):
        """Network should map R^3 -> R^3 logits."""
        x = torch.randn(4, 3)
        y = blob_net_3d(x)
        assert y.shape == (4, 3)

    def test_classifies_near_blobs_correctly(self, blob_net_3d):
        """Classifier should return the right class for points near each blob."""
        net = blob_net_3d
        # Near blob 0 at [1, 0, 0]
        x = torch.tensor([[0.9, 0.0, 0.0]])
        with torch.no_grad():
            logits = net(x)
        assert logits.argmax(dim=1).item() == 0
        # Near blob 1 at [0, 1, 0]
        x = torch.tensor([[0.0, 0.9, 0.0]])
        with torch.no_grad():
            logits = net(x)
        assert logits.argmax(dim=1).item() == 1
        # Near blob 2 at [0, 0, 1]
        x = torch.tensor([[0.0, 0.0, 0.9]])
        with torch.no_grad():
            logits = net(x)
        assert logits.argmax(dim=1).item() == 2


class TestNetworkForwardSpotChecks:
    def test_banana_predicts_target(self, banana_net):
        """RotatedBananaNet should approximate (x1, x1^2 + 0.3 x2)."""
        torch.manual_seed(0)
        x = torch.rand(200, 2)
        with torch.no_grad():
            y_pred = banana_net(x)
        y_true = torch.stack([x[:, 0], x[:, 0] ** 2 + 0.3 * x[:, 1]], dim=1)
        err = (y_pred - y_true).norm(dim=1).mean().item()
        assert err < 0.05, f"banana net mean error {err} — training regressed"

    def test_three_blob_2d_train_accuracy(self, blob_net):
        x, y = blob_net.sample_data(2000, seed=1)
        with torch.no_grad():
            pred = blob_net(x).argmax(dim=1)
        acc = (pred == y).float().mean().item()
        assert acc > 0.80, f"ThreeBlobClassifier train-dist accuracy {acc}"

    def test_three_blob_3d_train_accuracy(self, blob_net_3d):
        x, y = blob_net_3d.sample_data(2000, seed=1)
        with torch.no_grad():
            pred = blob_net_3d(x).argmax(dim=1)
        acc = (pred == y).float().mean().item()
        assert acc > 0.80, f"ThreeBlobClassifier3D train-dist accuracy {acc}"

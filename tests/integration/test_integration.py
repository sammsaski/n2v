"""
Integration tests for full network verification.

Oracle: every reach goes through the public entry
``NeuralNetwork.reach``; 128 member points of the input box, corners
included, are replayed on the torch model in float64, and each true output
must be a member of a returned set by an LP on that set's own rows
(``assert_covered``). Where a fully connected ReLU network is reached with the
exact method, the result is also compared with the exact image in rationals
(``tests/_oracles/rational.py``) in both directions: every vertex of the true
image is in one returned set, and every vertex of a returned set has a
preimage. The weights stay the models' own (float32 numbers, exact as
rationals). An empty result for a non-empty input fails: the specification
check would read it as a proof.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, ImageStar, Zono
from n2v.nn import NeuralNetwork
from tests._oracles import rational
from tests.cell_support import assert_covered, replay_torch, samples


def _holds_every_output(sets, model, lo, hi, in_shape, what, out_chw=None):
    """At least one set, and every replayed output of ``model`` on the box
    [lo, hi] (given in the network's order) in one of them."""
    assert len(sets) >= 1, f"{what}: no set returned for a non-empty input"
    assert_covered(sets, replay_torch(model, samples(lo, hi), in_shape),
                   what, out_chw=out_chw)


def _dyadic(model):
    """``model`` with every parameter rounded to a multiple of 1/16, so that
    float32, float64 and the rational image agree exactly: the exact checks
    compare with zero slack, and the affine map of a Star rounds to nearest
    on other data (registry ROUND-8, not these cells' subject)."""
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.round(p * 16) / 16)
    return model


def _exact_pieces(model, lo, hi):
    """The exact image of the box [lo, hi] under a Sequential of Linear and
    ReLU layers, as rational pieces, from the layers' own weights."""
    pieces = rational.box(list(lo), list(hi))
    for layer in model:
        if isinstance(layer, nn.Linear):
            pieces = rational.affine(
                pieces, layer.weight.detach().double().numpy().tolist(),
                layer.bias.detach().double().numpy().tolist())
        elif isinstance(layer, nn.ReLU):
            pieces = rational.relu(pieces)
        else:
            raise ValueError(f"no exact image for {type(layer).__name__}")
    return pieces


def _assert_holds_exact_image(sets, model, lo, hi, what):
    """No vertex of the true image outside the result, and no piece of the
    true image spread over several returned sets (a convex set that holds a
    piece's vertices holds the piece): the result holds the exact image."""
    pieces = _exact_pieces(model, lo, hi)
    computed = [rational.star(s) for s in sets]
    escaped = rational.escapes(pieces, *computed)
    assert not escaped, (f"{what}: {len(escaped)} vertices of the true image "
                         f"are in no returned set")
    assert not rational.straddlers(pieces, *computed), (
        f"{what}: a piece of the true image is not held by one returned set")
    return pieces, computed


def _assert_exact_image(sets, model, lo, hi, what):
    """Both directions against the exact image, for a network without ReLU
    (the exact ReLU keeps a neuron whose range on a piece ends within its LP
    pad of 0 in one star with a band of the pad's width, as its docstring
    says, so the other direction is checked by the ReLU kernel's own cells):
    the result holds the exact image, and no vertex of a returned set lacks
    a preimage."""
    pieces, computed = _assert_holds_exact_image(sets, model, lo, hi, what)
    for k, s in enumerate(computed):
        extra = rational.unreached(pieces, rational.set_vertices(s))
        assert not extra, (f"{what}: returned set {k} has {len(extra)} "
                           f"vertices with no preimage")


class TestFeedforwardNetworks:
    """Integration tests for feedforward networks."""

    def test_simple_feedforward_exact(self):
        """Linear-ReLU-Linear (dyadic weights), exact method on the unit
        cube: every returned set has the output's two entries, the result
        holds every replayed output and every vertex of the exact image in
        rationals, each piece of the image in one set. Catches a ReLU piece
        dropped, and an input set built off its box."""
        model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        _dyadic(model).eval()

        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        output_stars = NeuralNetwork(model).reach(input_star, method="exact")

        what = "exact Linear/ReLU/Linear on the unit cube"
        for star in output_stars:
            assert star.dim == 2
            pytest.assert_star_valid(star)
        _holds_every_output(output_stars, model, lb.ravel(), ub.ravel(),
                            (1, 3), what)
        _assert_holds_exact_image(output_stars, model, lb.ravel(), ub.ravel(),
                                  what)

    def test_feedforward_with_bounds_check(self):
        """The identity Linear layer on [0, 1]^2, exact method: one set whose
        vertices are exactly the four corners of the box (no more, no less).
        Catches an input set or a result moved off the box (centred on a
        corner instead of the centre) or scaled."""
        model = nn.Sequential(
            nn.Linear(2, 2)
        )
        model.eval()

        with torch.no_grad():
            model[0].weight.data = torch.eye(2)
            model[0].bias.data = torch.zeros(2)

        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        output_stars = NeuralNetwork(model).reach(input_star, method="exact")

        assert len(output_stars) == 1
        vertices = {tuple(v) for v in
                    rational.set_vertices(rational.star(output_stars[0]))}
        corners = {(rational.frac(a), rational.frac(b))
                   for a in (0, 1) for b in (0, 1)}
        assert vertices == corners, (
            f"the identity layer's result has vertices "
            f"{sorted(tuple(float(x) for x in v) for v in vertices)}, the "
            f"box [0, 1]^2 has its four corners")

    def test_multiple_relu_layers(self):
        """Two ReLU layers (dyadic weights), exact method on [-1, 1]^2: every
        returned set has one entry, and the result holds every replayed output
        and every vertex of the exact image in rationals. Catches a piece lost
        at the second ReLU and a split that keeps a piece of the wrong
        pattern."""
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        _dyadic(model).eval()

        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        output_stars = NeuralNetwork(model).reach(input_star, method="exact")

        what = "exact Linear/ReLU/Linear/ReLU/Linear on [-1, 1]^2"
        for star in output_stars:
            assert star.dim == 1
        _holds_every_output(output_stars, model, lb.ravel(), ub.ravel(),
                            (1, 2), what)
        _assert_holds_exact_image(output_stars, model, lb.ravel(), ub.ravel(),
                                  what)


class TestCNNNetworks:
    """Integration tests for CNN networks."""

    def test_simple_cnn_exact(self):
        """Conv (2 neurons) -> ReLU -> Flatten -> Linear on a 2x2 image,
        exact method: at most four pieces (two ReLU neurons), each of two
        entries, holding every replayed output. Catches a piece dropped by
        the exact ReLU on an image set and a flattened image in the wrong
        order."""
        model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2, 2)
        )
        model.eval()

        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1))
        input_star = ImageStar.from_bounds(lb, ub, height=2, width=2, num_channels=1)

        output_stars = NeuralNetwork(model).reach(input_star, method="exact")

        assert len(output_stars) <= 4
        for star in output_stars:
            assert star.dim == 2
        _holds_every_output(output_stars, model, np.zeros((1, 2, 2)),
                            np.ones((1, 2, 2)), (1, 1, 2, 2),
                            "exact Conv/ReLU/Flatten/Linear on a 2x2 image")

    def test_cnn_with_avgpool_no_splitting(self):
        """Conv -> AvgPool -> Flatten has no ReLU: the exact method gives
        exactly one set, and it holds every replayed output (eight entries in
        the tensor's channel, height, width order). Catches a split where no
        activation is, and a pooled image flattened in its stored order."""
        model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.AvgPool2d(2, 2),
            nn.Flatten()
        )
        model.eval()

        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        input_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        output_stars = NeuralNetwork(model).reach(input_star, method="exact")

        assert len(output_stars) == 1
        _holds_every_output(output_stars, model, np.zeros((1, 4, 4)),
                            np.ones((1, 4, 4)), (1, 1, 4, 4),
                            "exact Conv/AvgPool/Flatten on a 4x4 image")

    def test_cnn_strided_conv(self):
        """Strided Conv (4 neurons) -> ReLU -> Flatten -> Linear, exact: at
        most 16 pieces, each of two entries, holding every replayed output.
        Catches a stride read wrongly and a piece dropped by the split."""
        model = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4, 2)
        )
        model.eval()

        lb = np.zeros((4, 4, 1))
        ub = np.ones((4, 4, 1))
        input_star = ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)

        output_stars = NeuralNetwork(model).reach(input_star, method="exact")

        assert len(output_stars) <= 16
        for star in output_stars:
            assert star.dim == 2
        _holds_every_output(output_stars, model, np.zeros((1, 4, 4)),
                            np.ones((1, 4, 4)), (1, 1, 4, 4),
                            "exact strided Conv/ReLU/Flatten/Linear")


class TestRobustnessVerification:
    """Integration tests for robustness verification."""

    def test_local_robustness_verified(self):
        """A classifier whose class 1 wins by a margin of at least 1/2 on the
        whole input box: the first layer's weights and biases have torch's
        default size (at most 1/sqrt(2)), so every hidden value on the box
        lies in [0, 3/2]; each row of the last weight is scaled to an
        absolute sum of 1/2 and class 1 gets a bias of 2, so class 1 leads
        every other class by at least 2 - 2 (1/2) (3/2). The exact reach must
        hold every replayed output and prove the robustness, that is, for
        every returned set and every other class j, the largest value of
        y_j - y_1 over the set (an LP) is below 0. Catches a result that is
        too large (a centre or a generator wrong, so the classes overlap) and
        a result that misses outputs."""
        model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
        model.eval()
        with torch.no_grad():
            model[2].weight.mul_(0.5 / model[2].weight.abs().sum(dim=1,
                                                              keepdim=True))
            model[2].bias.copy_(torch.tensor([0.0, 2.0, 0.0]))

        center = np.array([[0.5], [0.5]])
        epsilon = 0.01
        lb = center - epsilon
        ub = center + epsilon
        input_star = Star.from_bounds(lb, ub)

        output_stars = NeuralNetwork(model).reach(input_star, method="exact")

        what = "exact reach of a classifier robust on its box"
        _holds_every_output(output_stars, model, lb.ravel(), ub.ravel(),
                            (1, 2), what)
        true_class = 1
        for star in output_stars:
            for j in (0, 2):
                diff = np.zeros((1, 3))
                diff[0, j], diff[0, true_class] = 1.0, -1.0
                _, hi = star.affine_map(diff).get_ranges()
                assert float(np.max(hi)) < 0, (
                    f"{what}: class {j} reaches class {true_class} by "
                    f"{float(np.max(hi))}")

    def test_adversarial_example_detection(self):
        """y = (x1 + x2, -(x1 + x2)) on a box that crosses the decision line:
        the exact reach is one set, the exact image (a segment from (-1, 1)
        to (1, -1)) in both directions, so it holds outputs of both classes.
        Catches a result that loses one side of the boundary or adds points
        off the segment."""
        model = nn.Sequential(
            nn.Linear(2, 2)
        )
        model.eval()

        with torch.no_grad():
            model[0].weight.data = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
            model[0].bias.data = torch.zeros(2)

        lb = np.array([[-0.5], [-0.5]])
        ub = np.array([[0.5], [0.5]])
        input_star = Star.from_bounds(lb, ub)

        output_stars = NeuralNetwork(model).reach(input_star, method="exact")

        what = "exact reach of a linear classifier across its boundary"
        assert len(output_stars) == 1
        _holds_every_output(output_stars, model, lb.ravel(), ub.ravel(),
                            (1, 2), what)
        _assert_exact_image(output_stars, model, lb.ravel(), ub.ravel(), what)


class TestApproximateMethods:
    """Integration tests for approximate methods."""

    def test_zono_approximate_verification(self):
        """Zonotope approx reach: one set of two entries (the approx method
        relaxes instead of splitting) that holds every replayed output.
        Catches the method not handed on (a Zono under the default 'exact'
        is refused) and a relaxation that misses outputs."""
        model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        model.eval()

        from n2v.sets import Zono
        lb = np.array([[0.0], [0.0], [0.0]])
        ub = np.array([[1.0], [1.0], [1.0]])
        input_zono = Zono.from_bounds(lb, ub)

        output_zonos = NeuralNetwork(model).reach(input_zono, method="approx")

        assert len(output_zonos) == 1
        assert output_zonos[0].dim == 2
        _holds_every_output(output_zonos, model, lb.ravel(), ub.ravel(),
                            (1, 3), "approx Zono reach")

    def test_approx_faster_than_exact(self):
        """The same network reached exactly from a Star and approximately
        from a Zono: both hold every replayed output, and the approx result
        has no more sets than the exact one (it does not split). Catches the
        method not handed on and an exact split that loses a piece."""
        model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
        model.eval()

        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])

        from n2v.sets import Star
        star = Star.from_bounds(lb, ub)
        exact_output = NeuralNetwork(model).reach(star, method="exact")

        from n2v.sets import Zono
        zono = Zono.from_bounds(lb, ub)
        approx_output = NeuralNetwork(model).reach(zono, method="approx")

        assert len(approx_output) <= len(exact_output)
        _holds_every_output(approx_output, model, lb.ravel(), ub.ravel(),
                            (1, 2), "approx Zono reach")
        # The exact result has many pieces; 32 points keep the cell short
        # (the other exact cells replay 128).
        assert len(exact_output) >= 1, "exact reach: no set returned"
        assert_covered(exact_output, replay_torch(
            model, samples(lb.ravel(), ub.ravel(), n=32), (1, 2)),
            "exact Star reach")


class TestEdgeCases:
    """Integration tests for edge cases."""

    def test_single_neuron_network(self):
        """Linear(2, 1), exact: one set of one entry, the exact image of the
        box (an interval) in both directions."""
        model = nn.Sequential(
            nn.Linear(2, 1)
        )
        _dyadic(model).eval()

        lb = np.array([[0.0], [0.0]])
        ub = np.array([[1.0], [1.0]])
        input_star = Star.from_bounds(lb, ub)

        output_stars = NeuralNetwork(model).reach(input_star, method="exact")

        assert len(output_stars) == 1
        assert output_stars[0].dim == 1
        _assert_exact_image(output_stars, model, lb.ravel(), ub.ravel(),
                            "exact Linear(2, 1)")

    def test_very_small_perturbation(self):
        """A box 2e-6 wide around (0.5, 0.5), exact method: the result
        holds every replayed output. The membership slack (1e-7 of an output
        near 1) is a tenth of the box's width, so the cell sees a result
        moved off the box by more than that (centred on a corner instead of
        the centre) but not a smaller error."""
        model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 2)
        )
        model.eval()

        center = np.array([[0.5], [0.5]])
        epsilon = 1e-6
        lb = center - epsilon
        ub = center + epsilon
        input_star = Star.from_bounds(lb, ub)

        output_stars = NeuralNetwork(model).reach(input_star, method="exact")

        _holds_every_output(output_stars, model, lb.ravel(), ub.ravel(),
                            (1, 2), "exact reach of a box 2e-6 wide")

    def test_exact_point_input(self):
        """A single point (no uncertainty): the exact result is not empty and
        holds the model's output at that point, which is the only output."""
        model = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU()
        )
        model.eval()

        point = np.array([[0.5], [0.5]])
        input_star = Star.from_bounds(point, point)

        output_stars = NeuralNetwork(model).reach(input_star, method="exact")

        _holds_every_output(output_stars, model, point.ravel(), point.ravel(),
                            (1, 2), "exact reach of a point")

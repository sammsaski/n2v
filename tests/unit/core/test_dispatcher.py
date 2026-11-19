"""
Tests for layer dispatcher system.
"""

import pytest
import numpy as np
import torch.nn as nn
from n2v.sets import Star
from n2v.nn.layer_ops.dispatcher import reach_layer


class TestDispatcherStar:
    """Tests for Star set dispatcher."""

    def test_dispatch_linear(self, simple_star):
        """Test dispatching Linear layer with Star."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = reach_layer(layer, [simple_star], method='exact')

        assert len(result) == 1
        assert result[0].dim == 2

    def test_dispatch_relu(self, simple_star):
        """Test dispatching ReLU layer with Star."""
        layer = nn.ReLU()

        result = reach_layer(layer, [simple_star], method='exact')

        assert len(result) >= 1
        for star in result:
            pytest.assert_star_valid(star)

    def test_dispatch_conv2d(self, simple_image_star):
        """Test dispatching Conv2D layer with ImageStar."""
        layer = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        layer.eval()

        result = reach_layer(layer, [simple_image_star], method='exact')

        assert len(result) == 1
        assert result[0].num_channels == 2

    def test_dispatch_maxpool2d(self, simple_image_star):
        """Test dispatching MaxPool2D layer with ImageStar."""
        layer = nn.MaxPool2d(2, 2)

        result = reach_layer(layer, [simple_image_star], method='exact')

        assert len(result) >= 1
        assert result[0].height == 2

    def test_dispatch_avgpool2d(self, simple_image_star):
        """Test dispatching AvgPool2D layer with ImageStar."""
        layer = nn.AvgPool2d(2, 2)

        result = reach_layer(layer, [simple_image_star], method='exact')

        # AvgPool should never split
        assert len(result) == 1
        assert result[0].height == 2

    def test_dispatch_flatten(self, simple_image_star):
        """Test dispatching Flatten layer with ImageStar."""
        layer = nn.Flatten()

        result = reach_layer(layer, [simple_image_star], method='exact')

        assert len(result) == 1
        assert result[0].dim == 16  # 4*4*1

    def test_dispatch_sequential(self, simple_star):
        """Test dispatching Sequential container."""
        layer = nn.Sequential(
            nn.Linear(3, 2),
            nn.ReLU()
        )
        layer.eval()

        result = reach_layer(layer, [simple_star], method='exact')

        assert len(result) >= 1

    def test_unknown_layer_error(self, simple_star):
        """Test error on unknown layer type."""
        class CustomLayer(nn.Module):
            def forward(self, x):
                return x

        layer = CustomLayer()

        with pytest.raises((ValueError, NotImplementedError)):
            reach_layer(layer, [simple_star], method='exact')


class TestDispatcherZono:
    """Tests for Zonotope dispatcher."""

    def test_dispatch_linear(self, simple_zono):
        """Test dispatching Linear layer with Zono."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = reach_layer(layer, [simple_zono])

        assert len(result) == 1
        assert result[0].dim == 2

    def test_dispatch_relu(self, simple_zono):
        """Test dispatching ReLU layer with Zono."""
        layer = nn.ReLU()

        result = reach_layer(layer, [simple_zono])

        assert len(result) == 1
        pytest.assert_zono_valid(result[0])

    def test_dispatch_conv2d(self, simple_image_zono):
        """Conv2d on an ImageZono routes to ``conv2d_zono`` (soundness of
        that kernel is ``tests/soundness/test_soundness_conv2d.py::
        test_conv2d_layer_is_the_exact_image_in_rationals``)."""
        layer = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        layer.eval()

        result = reach_layer(layer, [simple_image_zono])

        assert len(result) == 1
        assert result[0].num_channels == 2

    def test_dispatch_maxpool2d(self, simple_image_zono):
        """Test dispatching MaxPool2D layer with ImageZono."""
        layer = nn.MaxPool2d(2, 2)

        result = reach_layer(layer, [simple_image_zono])

        assert len(result) == 1
        assert result[0].height == 2

    def test_dispatch_avgpool2d(self, simple_image_zono):
        """Test dispatching AvgPool2D layer with ImageZono."""
        layer = nn.AvgPool2d(2, 2)

        result = reach_layer(layer, [simple_image_zono])

        assert len(result) == 1
        assert result[0].height == 2

    def test_dispatch_flatten(self, simple_image_zono):
        """Test dispatching Flatten layer with ImageZono."""
        layer = nn.Flatten()

        result = reach_layer(layer, [simple_image_zono])

        assert len(result) == 1
        assert result[0].dim == 16


class TestDispatcherBox:
    """Tests for Box dispatcher."""

    def test_dispatch_linear(self, simple_box):
        """Test dispatching Linear layer with Box."""
        layer = nn.Linear(3, 2)
        layer.eval()

        result = reach_layer(layer, [simple_box])

        assert len(result) == 1
        assert result[0].dim == 2

    def test_dispatch_relu(self, simple_box):
        """Test dispatching ReLU layer with Box."""
        layer = nn.ReLU()

        result = reach_layer(layer, [simple_box])

        assert len(result) == 1
        assert result[0].dim == simple_box.dim

    def test_dispatch_flatten(self):
        """Test dispatching Flatten layer with Box."""
        from n2v.sets import Box
        import numpy as np

        # Create a simple 2D image box
        lb = np.zeros((2, 2, 1))
        ub = np.ones((2, 2, 1))
        box = Box(lb.flatten().reshape(-1, 1), ub.flatten().reshape(-1, 1))

        layer = nn.Flatten()

        result = reach_layer(layer, [box])

        assert len(result) == 1


class TestDispatcherOptions:
    """Test dispatcher with different options."""

    def test_exact_vs_approx_relu(self, simple_star):
        """Test exact vs approx method for ReLU."""
        layer = nn.ReLU()

        exact_result = reach_layer(layer, [simple_star], method='exact')
        approx_result = reach_layer(layer, [simple_star], method='approx')

        # Approx should not split
        assert len(approx_result) <= len(exact_result)

    def test_maxpool_exact_vs_approx(self, simple_image_star):
        """Test exact vs approx method for MaxPool2D."""
        layer = nn.MaxPool2d(2, 2)

        exact_result = reach_layer(layer, [simple_image_star], method='exact')
        approx_result = reach_layer(layer, [simple_image_star], method='approx')

        # Approx should not split
        assert len(approx_result) <= len(exact_result)

    def test_display_option(self, caplog):
        """Test verbose=True produces log output."""
        import logging
        layer = nn.ReLU()

        # Create a star that crosses zero to guarantee splitting (and thus output)
        lb = np.array([[-1.0], [-1.0]])
        ub = np.array([[1.0], [1.0]])
        star_crossing_zero = Star.from_bounds(lb, ub)

        with caplog.at_level(logging.DEBUG):
            reach_layer(layer, [star_crossing_zero], method='exact', verbose=True)

        # Should log something about ReLU processing
        assert len(caplog.records) > 0


# --------------------------------------------------------------------------- #
# The dispatcher's own refusals and its method argument                       #
# --------------------------------------------------------------------------- #
from tests.cell_support import (assert_covered, flat_set, replay_torch,  # noqa: E402
                                samples)
from tests.known_bugs import DID_NOT_RAISE, known_bug  # noqa: E402

_LANE_KINDS = ("Star", "Zono", "Box", "Hexatope", "Octatope")
# Every coordinate in its own regime: straddling zero, above it, below it,
# straddling again; widths above 2 and ends beyond plus and minus 1.
_LO = np.array([-1.5, 0.25, -3.0, -0.75])
_HI = np.array([1.25, 2.75, -0.5, 2.0])
# The Hexatope ReLU splits a crossing coordinate with an LP whose optimum can
# miss by its tolerance (registry HEXOCT-4); its box crosses zero nowhere.
_BOXES = {"Hexatope": (np.array([0.25, 0.5, -3.0, -2.75]),
                       np.array([2.75, 3.0, -0.5, -0.25]))}


class _Doubling(nn.Module):
    """A layer type the dispatcher has no rule for (its output is not its
    input, so an identity answer would be wrong as well)."""

    def forward(self, x):
        return 2.0 * x


class TestDispatcherRefusals:
    """``reach_layer`` refuses what it has no rule for, on every lane."""

    @pytest.mark.parametrize("kind", _LANE_KINDS)
    def test_unknown_layer_type_is_refused_on_every_lane(self, kind):
        """A layer type without a rule is refused with
        ``NotImplementedError`` (the settled type of an unsupported
        combination) on each of the five lanes, which have their own final
        branch. Catches a lane whose final branch answers (an identity, or
        no set at all) or raises another type, and a profiling region that
        swallows the refusal (the reach then returns nothing)."""
        with pytest.raises(NotImplementedError):
            reach_layer(_Doubling(), [flat_set(kind, _LO, _HI)], "approx")

    @pytest.mark.parametrize("item", [
        pytest.param(np.zeros((4, 1)), id="ndarray"),
        pytest.param(None, id="none")])
    def test_an_object_that_is_not_a_set_is_refused(self, item):
        """A list whose first item is not a set is refused. Today the type is
        ``TypeError`` with a message naming the supported set types (a
        deliberate check); ``ValueError`` (input that cannot be read) and
        ``NotImplementedError`` are the settled types, so any of the three
        passes. Catches a lane chooser whose final branch answers, for
        example with the list as given or with no set."""
        with pytest.raises((TypeError, ValueError, NotImplementedError)):
            reach_layer(nn.ReLU(), [item], "approx")


_UNKNOWN_METHOD = known_bug(
    16, "reach_layer accepts any method name: in the Star lane every name "
        "other than 'exact' (a misspelling, 'Exact', the empty string) runs "
        "the approx rule, so a request for the exact image is answered with "
        "an over-approximation, and the Zono, Box, Hexatope and Octatope "
        "lanes never read the method; the model entry reach_pytorch_model "
        "refuses the same names with ValueError (DESIGN section 2 rule 4: "
        "unknown options raise)", raises=DID_NOT_RAISE)


class TestDispatcherMethodName:
    """The method argument of ``reach_layer``."""

    @pytest.mark.parametrize("kind,method", [
        pytest.param("Star", "exact", id="star-exact"),
        pytest.param("Star", "approx", id="star-approx"),
        pytest.param("Star", "Exact", marks=_UNKNOWN_METHOD,
                     id="star-Exact"),
        pytest.param("Zono", "approx", id="zono-approx"),
        pytest.param("Zono", "exakt", marks=_UNKNOWN_METHOD, id="zono-exakt"),
        pytest.param("Hexatope", "approx", id="hexatope-approx"),
        pytest.param("Hexatope", "", marks=_UNKNOWN_METHOD,
                     id="hexatope-empty")])
    def test_a_method_name_is_read_or_refused(self, kind, method):
        """``'exact'`` and ``'approx'`` are answered with sets that hold
        every true output of 64 points of the box (ReLU replayed in torch
        float64, LP membership on the sets' own rows); any other name is
        refused with ``ValueError``, as the model entry refuses it. The
        Star lane reads the name in each layer branch, the other lanes do
        not read it; the passing neighbours are the valid names on the same
        lanes. Catches a dispatcher that maps an unknown name to a rule (a
        misspelled ``'exact'`` silently answered with an over-approximation)
        and a valid name that is no longer answered or answered wrongly."""
        lo, hi = _BOXES.get(kind, (_LO, _HI))
        star = flat_set(kind, lo, hi)
        if method in ("exact", "approx"):
            out = reach_layer(nn.ReLU(), [star], method)
            assert_covered(out, replay_torch(nn.ReLU(), samples(lo, hi, 64),
                                             (1, 4)),
                           f"ReLU {method} on a {kind}")
        else:
            with pytest.raises(ValueError):
                reach_layer(nn.ReLU(), [star], method)

    def test_exact_reaches_the_exact_rule_in_the_star_lane(self):
        """x = (t, -t) with t in [-1, 1]: the ReLU image is two segments on
        the axes, and (1/4, 1/4) has no preimage (relu(t) = 1/4 forces
        t = 1/4, and then relu(-t) = 0). ``reach_layer(..., 'exact')`` must
        hold the image (33 points of the segments, the LP oracle on the
        returned rows) and leave that point out; the approx triangle holds
        it. Catches a dispatcher that hands the Star lane another method
        than the one asked for, which no box input shows (on a box the
        approx and the exact image have the same bounding box)."""
        from tests._oracles.membership import distance
        star = Star(np.array([[0.0, 1.0], [0.0, -1.0]]),
                    np.array([[1.0], [-1.0]]), np.ones((2, 1)),
                    -np.ones((1, 1)), np.ones((1, 1)))
        out = reach_layer(nn.ReLU(), [star], method="exact")
        image = [np.array([max(t, 0.0), max(-t, 0.0)])
                 for t in np.linspace(-1.0, 1.0, 33)]
        assert_covered(out, image, "exact ReLU of (t, -t)")
        gap = min(distance(s, np.array([0.25, 0.25])) for s in out)
        assert gap > 1e-3, (
            f"exact ReLU of (t, -t) holds (1/4, 1/4), which no input "
            f"reaches (distance {gap:.3g}): the result is not the exact image")

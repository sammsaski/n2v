"""Instrumentation tests for the convolution / pooling Star path.

Covers the MaxPool split counting (the pooling analog of ReLU's classification)
and the Conv2d / AvgPool / GlobalAvgPool operation regions.

NB: the full-CNN tests use **approx** reach. Exact reach on a conv net is
exponential (the post-conv ReLU splits 2^k over ~100 neurons) and OOMs in CI;
approx exercises the same conv/pool instrumentation with a bounded (population-1)
working set. Exact MaxPool *splitting* is tested in isolation on a tiny,
single-window input whose split count is bounded and deterministic.
"""

import numpy as np
import pytest
import torch.nn as nn

from n2v.sets import ImageStar
from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.profiling import profile, OPERATION, LAYER


def _as_list(out):
    return out if isinstance(out, list) else [out]


def _ambiguous_maxpool_input():
    """A 2x2x1 ImageStar (one 2x2 pooling window) whose argmax is ambiguous
    between the two high pixels -> exact MaxPool splits into exactly 2 stars."""
    lb = np.array([[[0.4], [0.4]], [[-1.0], [-1.0]]])
    ub = np.array([[[0.6], [0.6]], [[-0.8], [-0.8]]])
    return ImageStar.from_bounds(lb, ub, 2, 2, 1)


# --------------------------------------------------------------------------- #
# Non-interference: conv/pool output is bit-identical with profiling on vs off.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("level", ["run", "layer", "operation"])
def test_conv_pool_non_interference(cnn_net, level):
    net, X = cnn_net

    off = _as_list(net.reach(X, method="approx"))
    with profile(level=level):
        on = _as_list(net.reach(X, method="approx"))

    assert len(off) == len(on)
    for a, b in zip(off, on):
        assert np.array_equal(a.V, b.V), "V perturbed by profiling"
        assert np.array_equal(a.C, b.C), "C perturbed by profiling"
        assert np.array_equal(a.d, b.d), "d perturbed by profiling"


def test_maxpool_exact_non_interference():
    """Exact MaxPool splitting must also be non-perturbing (bounded input)."""
    X = _ambiguous_maxpool_input()
    off = reach_layer(nn.MaxPool2d(2), [X], "exact")
    with profile(level="operation"):
        on = reach_layer(nn.MaxPool2d(2), [X], "exact")
    assert len(off) == len(on) == 2
    for a, b in zip(off, on):
        assert np.array_equal(a.V, b.V)
        assert np.array_equal(a.C, b.C)
        assert np.array_equal(a.d, b.d)


# --------------------------------------------------------------------------- #
# Conv2d / AvgPool get OPERATION regions (parity with Linear's affine_map).
# --------------------------------------------------------------------------- #
def test_conv_avgpool_operation_regions(cnn_net):
    net, X = cnn_net
    with profile(level="operation") as p:
        net.reach(X, method="approx")

    op_names = {r.name for r in p.records() if r.level == OPERATION}
    assert "conv2d" in op_names
    assert "avgpool2d" in op_names

    # an operation region nests inside its layer region
    conv = p.find("conv2d")[0]
    assert conv.parent is not None
    assert conv.parent.level == LAYER
    assert conv.parent.meta.get("layer_type") == "Conv2d"


def test_operation_regions_skipped_at_layer_level(cnn_net):
    """At level='layer', deeper OPERATION regions must not be recorded."""
    net, X = cnn_net
    with profile(level="layer") as p:
        net.reach(X, method="approx")
    assert p.find("conv2d") == []
    assert p.find("avgpool2d") == []
    # but the layers themselves are still present
    assert p.find("Conv2d") and p.find("MaxPool2d") and p.find("AvgPool2d")


# --------------------------------------------------------------------------- #
# MaxPool exact: static window count + uncertain/split classification, on a
# tiny single-window input with a deterministic, bounded 2-way split.
# --------------------------------------------------------------------------- #
def test_maxpool_exact_counters():
    X = _ambiguous_maxpool_input()
    with profile(level="layer") as p:
        out = reach_layer(nn.MaxPool2d(2), [X], "exact")

    mp = p.find("MaxPool2d")
    assert len(mp) == 1
    c = p.subtree_counters(mp[0])

    assert c["n_pool_windows"] == 1     # 2x2 -> 1x1, 1 channel
    assert c["n_uncertain"] == 1        # one ambiguous window
    assert c["n_split"] == 1            # 2 candidates -> +1 branch
    assert len(out) == 2
    assert mp[0].counters["n_sets_out"] == mp[0].counters["n_sets_in"] + c["n_split"]


# --------------------------------------------------------------------------- #
# MaxPool approx (bounded full-CNN): relaxes, never splits.
# --------------------------------------------------------------------------- #
def test_maxpool_approx_relaxes_not_splits(cnn_net):
    net, X = cnn_net
    with profile(level="layer") as p:
        net.reach(X, method="approx")

    mp = p.find("MaxPool2d")
    assert len(mp) == 1
    c = p.subtree_counters(mp[0])
    # approx introduces a new predicate per uncertain window, never splits
    assert c.get("n_split", 0) == 0
    assert c["n_pool_windows"] == 4 * 4 * 2
    # uncertain windows (if any) are all relaxed
    assert c.get("n_relaxed", 0) == c.get("n_uncertain", 0)
    assert mp[0].counters["n_sets_out"] == mp[0].counters["n_sets_in"]

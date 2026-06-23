"""Instrumentation tests for the convolution / pooling Star path.

Covers the MaxPool split counting (the pooling analog of ReLU's classification)
and the Conv2d / AvgPool / GlobalAvgPool operation regions.
"""

import numpy as np
import pytest

from n2v.profiling import profile, OPERATION, LAYER


def _as_list(out):
    return out if isinstance(out, list) else [out]


# --------------------------------------------------------------------------- #
# Non-interference: conv/pool output is bit-identical with profiling on vs off.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("level", ["run", "layer", "operation"])
def test_conv_pool_non_interference(cnn_net, level):
    net, X = cnn_net

    off = _as_list(net.reach(X, method="exact"))
    with profile(level=level):
        on = _as_list(net.reach(X, method="exact"))

    assert len(off) == len(on)
    for a, b in zip(off, on):
        assert np.array_equal(a.V, b.V), "V perturbed by profiling"
        assert np.array_equal(a.C, b.C), "C perturbed by profiling"
        assert np.array_equal(a.d, b.d), "d perturbed by profiling"


# --------------------------------------------------------------------------- #
# Conv2d / AvgPool get OPERATION regions (parity with Linear's affine_map).
# --------------------------------------------------------------------------- #
def test_conv_avgpool_operation_regions(cnn_net):
    net, X = cnn_net
    with profile(level="operation") as p:
        net.reach(X, method="exact")

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
        net.reach(X, method="exact")
    assert p.find("conv2d") == []
    assert p.find("avgpool2d") == []
    # but the layers themselves are still present
    assert p.find("Conv2d") and p.find("MaxPool2d") and p.find("AvgPool2d")


# --------------------------------------------------------------------------- #
# MaxPool: static window count + uncertain/split classification.
# --------------------------------------------------------------------------- #
def test_maxpool_counters(cnn_net):
    net, X = cnn_net
    with profile(level="layer") as p:
        out = net.reach(X, method="exact")

    mp = p.find("MaxPool2d")
    assert len(mp) == 1
    c = p.subtree_counters(mp[0])

    # static output window count: 8x8x2 conv -> maxpool(2) -> 4x4x2 = 32
    assert c["n_pool_windows"] == 4 * 4 * 2

    # n_split is the population growth across the MaxPool (branches).
    # out population == input population (1) + total splits across the run's
    # MaxPool; for a single input star, growth == n_split.
    n_split = c.get("n_split", 0)
    assert n_split >= 0
    assert mp[0].counters["n_sets_out"] == mp[0].counters["n_sets_in"] + n_split

    # if anything split, there were uncertain windows to cause it
    if n_split > 0:
        assert c["n_uncertain"] >= 1

    # whole-run sanity: reach produced the reported number of output stars
    assert len(out) >= 1


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

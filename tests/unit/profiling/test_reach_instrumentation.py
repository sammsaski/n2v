"""Reach-path instrumentation tests (Linear + ReLU).

T2 — counter accuracy (classification + split/relaxed + LP count).
T3 — non-interference (output bit-identical with profiling off vs on).

See plans/2026-06-22-profiler-test-spec.md.
"""

import numpy as np
import pytest

from n2v.profiling import profile, LAYER


def _as_list(out):
    """Normalize a reach result to a list of sets."""
    return out if isinstance(out, list) else [out]


# --------------------------------------------------------------------------- #
# T2 — counter accuracy
# --------------------------------------------------------------------------- #
def test_t2_counters_exact(tiny_split_net, lp_oracle):
    net, X = tiny_split_net
    with profile(level="operation") as p:
        net.reach(X, method="exact")

    relu = p.find("ReLU")
    assert len(relu) == 1, "expected exactly one ReLU layer region"
    c = p.subtree_counters(relu[0])

    # classification
    assert c["n_neurons"] == 3
    assert c["n_stable_active"] == 1
    assert c["n_stable_inactive"] == 1
    assert c["n_unstable"] == 1
    # handling of the unstable neuron
    assert c["n_split"] == 1
    assert c.get("n_relaxed", 0) == 0
    assert c.get("n_resolved", 0) == 0
    # population growth across the ReLU (the split)
    assert c["n_sets_in"] == 1
    assert c["n_sets_out"] == 2

    # invariants
    assert (
        c["n_stable_active"] + c["n_stable_inactive"] + c["n_unstable"]
        == c["n_neurons"]
    )
    assert (
        c["n_split"] + c.get("n_relaxed", 0) + c.get("n_resolved", 0)
        == c["n_unstable"]
    )

    # LP count matches the independent oracle (and equals 2: one min+max
    # get_range for the single crossing neuron)
    assert c["n_lp_solves"] == lp_oracle["n"]
    assert c["n_lp_solves"] == 2


def test_t2_counters_approx(tiny_split_net, lp_oracle):
    net, X = tiny_split_net
    with profile(level="operation") as p:
        net.reach(X, method="approx")

    c = p.subtree_counters(p.find("ReLU")[0])

    assert c["n_neurons"] == 3
    assert c["n_stable_active"] == 1
    assert c["n_stable_inactive"] == 1
    assert c["n_unstable"] == 1
    # approx relaxes, never splits
    assert c.get("n_split", 0) == 0
    assert c["n_relaxed"] == 1
    assert c.get("n_resolved", 0) == 0
    # no population growth (no split)
    assert c["n_sets_in"] == 1
    assert c["n_sets_out"] == 1
    # approx classification is LP-free
    assert c.get("n_lp_solves", 0) == lp_oracle["n"]
    assert c.get("n_lp_solves", 0) == 0


# --------------------------------------------------------------------------- #
# T3 — non-interference (the load-bearing guarantee)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["exact", "approx"])
@pytest.mark.parametrize("level", ["run", "phase", "layer", "operation"])
def test_t3_non_interference(tiny_split_net, method, level):
    net, X = tiny_split_net

    off = _as_list(net.reach(X, method=method))
    with profile(level=level):
        on = _as_list(net.reach(X, method=method))

    assert len(off) == len(on)
    for a, b in zip(off, on):
        assert np.array_equal(a.V, b.V), "V perturbed by profiling"
        assert np.array_equal(a.C, b.C), "C perturbed by profiling"
        assert np.array_equal(a.d, b.d), "d perturbed by profiling"


# --------------------------------------------------------------------------- #
# Activation coverage (LeakyReLU split, Sigmoid relax)
# --------------------------------------------------------------------------- #
def test_leakyrelu_counters_exact(leaky_net):
    net, X = leaky_net
    with profile(level="layer") as p:
        net.reach(X, method="exact")
    c = p.subtree_counters(p.find("LeakyReLU")[0])
    assert c["n_neurons"] == 3
    assert c["n_stable_active"] == 1
    assert c["n_stable_inactive"] == 1
    assert c["n_unstable"] == 1
    assert c["n_split"] == 1
    assert c["n_sets_out"] == 2


def test_sigmoid_counters_approx(sigmoid_net):
    net, X = sigmoid_net
    with profile(level="layer") as p:
        net.reach(X, method="approx")
    c = p.subtree_counters(p.find("Sigmoid")[0])
    assert c["n_neurons"] == 3
    assert c["n_relaxed"] == 3
    assert c.get("n_constant", 0) == 0


# --------------------------------------------------------------------------- #
# Relax-method ReLU variants (range / area / bound): classification + outcome
# counters fire, sum to n_unstable, never split, and don't perturb output.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("relax_method", ["range", "area", "bound"])
def test_relax_variant_counters(tiny_split_net, relax_method):
    net, X = tiny_split_net

    with profile(level="layer") as p:
        on = _as_list(
            net.reach(X, method="approx",
                      relax_method=relax_method, relax_factor=0.5)
        )
    c = p.subtree_counters(p.find("ReLU")[0])

    # classification (same hand-checkable structure as tiny_split_net)
    assert c["n_neurons"] == 3
    assert c["n_stable_inactive"] == 1
    assert c["n_unstable"] == 1
    assert c["n_stable_active"] == 1
    # the estimated-unstable neuron is partitioned into relaxed + resolved
    assert c.get("n_relaxed", 0) + c.get("n_resolved", 0) == c["n_unstable"]
    # relax variants over-approximate, never split
    assert c.get("n_split", 0) == 0
    assert c["n_sets_out"] == 1

    # non-interference: identical output with profiling off
    off = _as_list(
        net.reach(X, method="approx",
                  relax_method=relax_method, relax_factor=0.5)
    )
    assert len(off) == len(on)
    for a, b in zip(off, on):
        assert np.array_equal(a.V, b.V)
        assert np.array_equal(a.C, b.C)
        assert np.array_equal(a.d, b.d)


# --------------------------------------------------------------------------- #
# Phase region: layers nest under "reach"
# --------------------------------------------------------------------------- #
def test_reach_phase_region(tiny_split_net):
    from n2v.profiling import PHASE

    net, X = tiny_split_net
    with profile(level="layer") as p:
        net.reach(X, method="exact")
    reach = p.find("reach")
    assert len(reach) == 1
    assert reach[0].level == PHASE
    assert reach[0].parent is p.root
    layer_names = [ch.name for ch in reach[0].children]
    assert "Linear" in layer_names and "ReLU" in layer_names


# --------------------------------------------------------------------------- #
# T5 — self-time non-negative; children wall <= parent wall
# --------------------------------------------------------------------------- #
def test_t5_self_time_and_nesting(multi_layer_net):
    net, X = multi_layer_net
    with profile(level="operation") as p:
        net.reach(X, method="exact")
    for r in p.records():
        assert r.self_time >= -1e-9
        for ch in r.children:
            assert ch.wall_time <= r.wall_time + 1e-9


# --------------------------------------------------------------------------- #
# T6 — distinct layer-type regions with metadata
# --------------------------------------------------------------------------- #
def test_t6_layer_types(multi_layer_net):
    from n2v.profiling import LAYER

    net, X = multi_layer_net
    with profile(level="layer") as p:
        net.reach(X, method="exact")
    types = [r.meta.get("layer_type") for r in p.records() if r.level == LAYER]
    assert types.count("Linear") == 2
    assert types.count("ReLU") == 2


# --------------------------------------------------------------------------- #
# n_neurons is the STATIC layer size (recorded once), not summed over the
# per-star population; the work counters stay per-star totals.
# --------------------------------------------------------------------------- #
def test_n_neurons_static_per_layer(multi_layer_net):
    net, X = multi_layer_net
    with profile(level="layer") as p:
        net.reach(X, method="exact")

    relus = p.find("ReLU")
    assert len(relus) == 2
    first, second = relus

    # first ReLU sees one input star: n_neurons == its 4 neurons
    c1 = p.subtree_counters(first)
    assert c1["n_neurons"] == 4
    assert c1["n_sets_in"] == 1

    # second ReLU has 3 neurons but processes a 7-star population.
    # n_neurons must stay 3 (static), NOT 3 * 7.
    c2 = p.subtree_counters(second)
    assert c2["n_neurons"] == 3
    assert c2["n_sets_in"] == 7
    # the classification work IS summed across the 7 stars (per-star totals)
    assert (
        c2["n_stable_active"] + c2["n_stable_inactive"] + c2["n_unstable"]
        == 3 * c2["n_sets_in"]
    )


# --------------------------------------------------------------------------- #
# Run-level rollup: per-layer-type aggregate at the end.
# --------------------------------------------------------------------------- #
def test_rollup_aggregates_by_layer_type(multi_layer_net):
    net, X = multi_layer_net
    with profile(level="layer") as p:
        net.reach(X, method="exact")

    ro = p.rollup()
    assert ro["n_layers"] == 4
    assert ro["peak_population"] == 7

    by_type = ro["by_layer_type"]
    assert by_type["Linear"]["count"] == 2
    assert by_type["ReLU"]["count"] == 2
    # static neuron counts summed over the two ReLU instances: 4 + 3
    assert by_type["ReLU"]["counters"]["n_neurons"] == 7

    # grand totals match the sum of per-layer-type counters
    assert ro["totals"]["n_neurons"] == 7
    # rollup text renders and is appended to summary()
    assert "=== run rollup ===" in p.summary()
    assert "=== run rollup ===" not in p.summary(rollup=False)


# --------------------------------------------------------------------------- #
# Memory: set-representation byte volume per layer + rollup peak (T12).
# --------------------------------------------------------------------------- #
def test_set_bytes_out(tiny_split_net):
    from n2v.nn.layer_ops.dispatcher import _set_nbytes

    net, X = tiny_split_net
    with profile(level="layer") as p:
        out = net.reach(X, method="exact")

    # ReLU is the last layer: its set_bytes_out == bytes of the returned sets
    relu = p.find("ReLU")[0]
    expected = sum(_set_nbytes(s) for s in out)
    assert relu.counters["set_bytes_out"] == expected
    assert relu.counters["set_bytes_out"] > 0

    # rollup exposes the memory high-water mark across layers
    ro = p.rollup()
    assert ro["peak_set_bytes"] == max(
        r.counters.get("set_bytes_out", 0)
        for r in p.records()
        if r.level == LAYER
    )
    assert ro["peak_set_bytes"] > 0


# --------------------------------------------------------------------------- #
# Counter ACCURACY (not just non-perturbation): counts == independent ground
# truth. A wrong counter would silently mislead development, so these are
# load-bearing.
# --------------------------------------------------------------------------- #
def test_rollup_totals_equal_raw_counter_sum(multi_layer_net):
    """rollup totals must equal the raw sum of every region's own counters
    (validates the subtree/rollup aggregation math)."""
    net, X = multi_layer_net
    with profile(level="operation") as p:
        net.reach(X, method="exact")

    raw = {}
    for r in p.records():
        for k, v in r.counters.items():
            raw[k] = raw.get(k, 0) + v
    assert p.rollup()["totals"] == raw


def test_classification_and_split_accuracy(multi_layer_net):
    """For exact Star ReLU: the per-star classification sums to
    n_neurons * n_input_stars, and n_split equals the population growth."""
    net, X = multi_layer_net
    with profile(level="layer") as p:
        net.reach(X, method="exact")

    for relu in p.find("ReLU"):
        c = relu.counters
        assert (
            c["n_stable_active"] + c["n_stable_inactive"] + c["n_unstable"]
            == c["n_neurons"] * c["n_sets_in"]
        )
        assert c.get("n_split", 0) == c["n_sets_out"] - c["n_sets_in"]


def test_lp_count_matches_independent_oracle(multi_layer_net, lp_oracle):
    """Total n_lp_solves across the run equals the independent monkeypatch
    count of solve_lp_batch objectives."""
    net, X = multi_layer_net
    with profile(level="operation") as p:
        net.reach(X, method="exact")

    total_lp = sum(r.counters.get("n_lp_solves", 0) for r in p.records())
    assert total_lp == lp_oracle["n"]
    assert total_lp > 0


# --------------------------------------------------------------------------- #
# T10 — FLOPs counting + roofline arithmetic intensity.
# --------------------------------------------------------------------------- #
def test_t10_flops_and_intensity(tiny_split_net):
    net, X = tiny_split_net  # Linear(2->3) then ReLU
    with profile(level="operation") as p:
        net.reach(X, method="exact")

    # Linear maps W(3x2) @ V(2 x ncols): 2 mul-adds per output element.
    linear = p.find("Linear")[0]
    expected = 2 * 3 * 2 * X.V.shape[1]
    assert p.subtree_counters(linear)["flops"] == expected

    ro = p.rollup()
    assert ro["totals"]["flops"] >= expected
    # both flops and carried bytes are present -> intensity is finite and > 0
    assert ro["arithmetic_intensity"] > 0.0
    assert ro["arithmetic_intensity"] == (
        ro["totals"]["flops"] / ro["totals"]["set_bytes_out"]
    )


def test_t10_intensity_zero_without_flops():
    # A pure-region run with no flops -> intensity is 0.0, not a div-by-zero.
    with profile(level="layer") as p:
        from n2v.profiling import region, count, LAYER as _L
        with region("X", _L):
            count("set_bytes_out", 100)
    assert p.rollup()["arithmetic_intensity"] == 0.0


# --------------------------------------------------------------------------- #
# T16 — overhead guard: catastrophic-regression tripwire (very generous bound,
# min-of-repeats to resist scheduler noise; never a tight assertion).
# --------------------------------------------------------------------------- #
def test_t16_overhead_guard(multi_layer_net):
    from time import perf_counter

    net, X = multi_layer_net
    net.reach(X, method="exact")  # warm-up

    def t_off():
        s = perf_counter(); net.reach(X, method="exact"); return perf_counter() - s

    def t_on():
        s = perf_counter()
        with profile(level="operation"):
            net.reach(X, method="exact")
        return perf_counter() - s

    off = min(t_off() for _ in range(5))
    on = min(t_on() for _ in range(5))
    # tripwire only: flags an accidental order-of-magnitude (e.g. quadratic)
    # regression, not normal single-digit-% overhead.
    assert on < max(off * 25.0, off + 0.05)


# --------------------------------------------------------------------------- #
# T14 — deterministic region structure
# --------------------------------------------------------------------------- #
def test_t14_deterministic_structure(tiny_split_net):
    net, X = tiny_split_net

    def structure(p):
        return [(r.name, r.level) for r in p.records()]

    with profile(level="operation") as p1:
        net.reach(X, method="exact")
    with profile(level="operation") as p2:
        net.reach(X, method="exact")
    assert structure(p1) == structure(p2)

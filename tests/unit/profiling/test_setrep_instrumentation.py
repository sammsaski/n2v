"""Instrumentation tests for the non-Star set representations.

Zono (LP-free over-approx), Hexatope / Octatope (split + MCF). All share the
hand-checkable tiny_split_net structure: neuron 0 stable-active, neuron 1
stable-inactive, neuron 2 unstable (crosses zero).
"""

import numpy as np
import pytest

from n2v.profiling import profile, LAYER


def _as_list(out):
    return out if isinstance(out, list) else [out]


def _repr_arrays(s):
    """Representation arrays present on a set (Star V/C/d, Zono c/V,
    Hexatope/Octatope center/generators) for bit-identity comparison."""
    out = []
    for attr in ("c", "V", "C", "d", "center", "generators"):
        v = getattr(s, attr, None)
        if isinstance(v, np.ndarray):
            out.append((attr, v))
    return out


# --------------------------------------------------------------------------- #
# Zono: classification + relaxation (no split, LP-free).
# --------------------------------------------------------------------------- #
def test_zono_relu_classification(zono_split_net):
    net, X = zono_split_net
    with profile(level="layer") as p:
        net.reach(X, method="approx")

    c = p.subtree_counters(p.find("ReLU")[0])
    assert c["n_neurons"] == 3
    assert c["n_stable_active"] == 1
    assert c["n_stable_inactive"] == 1
    assert c["n_unstable"] == 1
    assert c["n_relaxed"] == 1          # crossing neuron over-approximated
    assert c.get("n_split", 0) == 0     # Zono never splits
    assert c.get("n_lp_solves", 0) == 0  # LP-free path


# --------------------------------------------------------------------------- #
# Hexatope / Octatope: classification + split + n_mcf_solves.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture", ["hexa_split_net", "octa_split_net"])
def test_hexa_octa_relu_classification(fixture, request):
    net, X = request.getfixturevalue(fixture)
    with profile(level="layer") as p:
        out = _as_list(net.reach(X, method="approx"))

    c = p.subtree_counters(p.find("ReLU")[0])
    assert c["n_neurons"] == 3
    assert c["n_stable_active"] == 1
    assert c["n_stable_inactive"] == 1
    assert c["n_unstable"] == 1
    # one crossing neuron -> one split -> population 1 -> 2
    assert c["n_split"] == 1
    relu = p.find("ReLU")[0]
    assert relu.counters["n_sets_out"] == relu.counters["n_sets_in"] + 1
    assert len(out) == 2


@pytest.mark.parametrize(
    "fixture", ["zono_split_net", "hexa_split_net", "octa_split_net"]
)
def test_setrep_set_bytes_nonzero(fixture, request):
    """Memory (set_bytes_out) must be measured for every representation -- in
    particular Hexatope/Octatope, whose center/generators arrays were omitted
    from _SET_ARRAY_ATTRS, so peak_set_bytes silently read 0."""
    net, X = request.getfixturevalue(fixture)
    with profile(level="layer") as p:
        net.reach(X, method="approx")
    assert p.rollup()["peak_set_bytes"] > 0
    for r in p.records():
        if r.level == LAYER:
            assert r.counters.get("set_bytes_out", 0) > 0


@pytest.mark.parametrize("fixture", ["hexa_split_net", "octa_split_net"])
def test_hexa_octa_mcf_solves_counted(fixture, request):
    net, X = request.getfixturevalue(fixture)
    # solver='mcf' forces the network-simplex path (both reps route through
    # Hexatope._optimize_dcs_mcf, where n_mcf_solves is counted).
    with profile(level="layer") as p:
        net.reach(X, method="approx", solver="mcf")

    total_mcf = p.subtree_counters(p.root).get("n_mcf_solves", 0)
    assert total_mcf > 0


# --------------------------------------------------------------------------- #
# Non-interference holds for every representation (profiling never perturbs).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fixture", ["zono_split_net", "hexa_split_net", "octa_split_net"]
)
def test_setrep_non_interference(fixture, request):
    net, X = request.getfixturevalue(fixture)

    off = _as_list(net.reach(X, method="approx"))
    with profile(level="operation"):
        on = _as_list(net.reach(X, method="approx"))

    assert len(off) == len(on)
    for a, b in zip(off, on):
        arrs_a, arrs_b = _repr_arrays(a), _repr_arrays(b)
        assert arrs_a and [k for k, _ in arrs_a] == [k for k, _ in arrs_b]
        for (ka, va), (kb, vb) in zip(arrs_a, arrs_b):
            assert np.array_equal(va, vb), f"{ka} perturbed by profiling"

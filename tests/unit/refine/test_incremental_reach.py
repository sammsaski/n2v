"""
Unit tests for incremental (shared-prefix) reach.

The load-bearing invariant: resuming from a parent checkpoint and reprocessing
only the layers at/after the split must produce a star *identical* to running
``relaxed_reach`` from scratch with the same fixed set. If that holds, incremental
reach is a pure speedup with no behavioural change -- the verdict suite
(test_refine_cpu_sound) then confirms it end-to-end.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets.star import Star
from n2v.refine.reach_relaxed import (
    extract_layers, relaxed_reach, resume_reach, relu_positions,
)
from n2v.refine.types import NeuronKey, Phase


def _net(seed, dims):
    rng = np.random.default_rng(seed)
    mods = []
    for i in range(len(dims) - 2):
        mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    mods += [nn.Linear(dims[-2], dims[-1])]
    net = nn.Sequential(*mods).double()
    with torch.no_grad():
        for m in net:
            if isinstance(m, nn.Linear):
                m.weight.copy_(torch.tensor(rng.uniform(-1.4, 1.4, m.weight.shape)))
                m.bias.copy_(torch.tensor(rng.uniform(-0.4, 0.4, m.bias.shape)))
    return net


def _assert_star_equal(a: Star, b: Star, tol=1e-9):
    assert a.V.shape == b.V.shape, f"V shape {a.V.shape} vs {b.V.shape}"
    assert np.allclose(a.V, b.V, atol=tol)
    assert a.C.shape == b.C.shape, f"C shape {a.C.shape} vs {b.C.shape}"
    assert np.allclose(a.C, b.C, atol=tol)
    assert np.allclose(a.d, b.d, atol=tol)
    assert np.allclose(a.predicate_lb, b.predicate_lb, atol=tol)
    assert np.allclose(a.predicate_ub, b.predicate_ub, atol=tol)


def _assert_meta_equal(ma, mb, tol=1e-9):
    assert len(ma) == len(mb), f"meta len {len(ma)} vs {len(mb)}"
    for x, y in zip(ma, mb):
        assert x.key == y.key
        assert x.pred_col == y.pred_col
        assert x.preact_center == pytest.approx(y.preact_center, abs=tol)
        assert np.allclose(x.preact_gens, y.preact_gens, atol=tol)
        assert x.l == pytest.approx(y.l, abs=tol) and x.u == pytest.approx(y.u, abs=tol)


@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu"])
@pytest.mark.parametrize("seed", range(4))
def test_resume_equals_from_scratch(bound_mode, seed):
    """resume_reach(parent ckpt, child_fixed) == relaxed_reach(child_fixed)."""
    net = _net(seed, dims=(2, 6, 6, 6, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    parent, pmeta = relaxed_reach(S_in, layers, {}, bound_mode=bound_mode)
    assert parent.checkpoints is not None and len(parent.checkpoints) == len(relu_positions(layers))

    # Fix one relaxed neuron at each layer that has candidates, both phases.
    by_layer = {}
    for m in pmeta:
        by_layer.setdefault(m.key.layer, m.key)
    assert by_layer, "expected relaxed neurons to split on"
    for L, key in by_layer.items():
        for phase in (Phase.ACTIVE, Phase.INACTIVE):
            child_fixed = {key: phase}
            scratch, smeta = relaxed_reach(S_in, layers, child_fixed, bound_mode=bound_mode)
            resumed = resume_reach(parent, layers, child_fixed, bound_mode, L)
            _assert_star_equal(resumed, scratch)
            _assert_meta_equal(resumed.relax_meta, smeta)


@pytest.mark.parametrize("bound_mode", ["box", "lp_cpu"])
def test_resume_from_multifix_parent_nonmonotonic(bound_mode):
    """
    The case the BaB actually produces (depth > 1): resume from a parent that
    *already* carries a fix, splitting at a SHALLOWER layer than that fix
    (non-monotonic order). Must still equal a from-scratch reach with the full
    fixed set -- exact V/C/d/meta. This is the coverage the single-fix-from-root
    test misses.
    """
    net = _net(3, dims=(2, 6, 6, 6, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    root, rmeta = relaxed_reach(S_in, layers, {}, bound_mode=bound_mode)

    deep_L = max(m.key.layer for m in rmeta)
    deep_key = next(m.key for m in rmeta if m.key.layer == deep_L)
    # Parent already has a DEEP fix.
    parent = resume_reach(root, layers, {deep_key: Phase.ACTIVE}, bound_mode, deep_L)
    shallow = [m.key for m in parent.relax_meta if m.key.layer < deep_L]
    assert shallow, "need a relaxed neuron shallower than the deep fix"
    shallow_key = shallow[0]

    # Now split the parent at the SHALLOWER layer (non-monotonic).
    combined = {deep_key: Phase.ACTIVE, shallow_key: Phase.INACTIVE}
    child = resume_reach(parent, layers, combined, bound_mode, shallow_key.layer)
    scratch, smeta = relaxed_reach(S_in, layers, combined, bound_mode=bound_mode)
    _assert_star_equal(child, scratch)
    _assert_meta_equal(child.relax_meta, smeta)


def test_resume_rejects_prefix_disagreement():
    """The load-bearing precondition is enforced: a child_fixed that contradicts
    the parent below the split layer raises rather than silently producing an
    unsound star."""
    net = _net(0, dims=(2, 5, 5, 5, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    root, rmeta = relaxed_reach(S_in, layers, {})
    L0_key = next(m.key for m in rmeta if m.key.layer == 0)
    split_L = max(m.key.layer for m in rmeta)
    assert split_L > 0
    # child_fixed adds a fix at layer 0 (< split_L) that the parent lacks.
    with pytest.raises(ValueError, match="below split"):
        resume_reach(root, layers, {L0_key: Phase.ACTIVE}, "box", split_L)
    # out-of-range split layer is also rejected clearly.
    with pytest.raises(ValueError, match="out of range"):
        resume_reach(root, layers, {}, "box", len(root.checkpoints))


def test_checkpoint_prefix_is_shared_by_reference():
    """Child checkpoints/meta below the split layer are the parent's (shared)."""
    net = _net(1, dims=(2, 5, 5, 5, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    parent, pmeta = relaxed_reach(S_in, layers, {})
    L = max(m.key.layer for m in pmeta)  # deepest relaxed layer -> max reuse
    key = next(m.key for m in pmeta if m.key.layer == L)
    child = resume_reach(parent, layers, {key: Phase.ACTIVE}, "box", L)

    # checkpoints strictly below L are the same objects as the parent's.
    for i in range(L):
        assert child.checkpoints[i] is parent.checkpoints[i]
    # meta below L is reused verbatim (same keys/pred_cols as parent's prefix).
    pre_parent = [m for m in pmeta if m.key.layer < L]
    pre_child = [m for m in child.relax_meta if m.key.layer < L]
    _assert_meta_equal(pre_child, pre_parent)


@pytest.mark.parametrize("seed", range(4))
def test_zono_bounds_sound_and_tighter_than_box(seed):
    """
    The "zono" bound mode is a sound over-approximation (its output star encloses
    every exact model output) and tighter-or-equal to the box mode (the DeepZ
    zonotope tracks correlations the predicate box loses).
    """
    net = _net(seed, dims=(3, 8, 8, 8, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(-np.ones(3), np.ones(3))
    out_box, _ = relaxed_reach(S_in, layers, {}, bound_mode="box")
    out_zono, _ = relaxed_reach(S_in, layers, {}, bound_mode="zono")
    lbz, ubz = out_zono.estimate_ranges()
    lbb, ubb = out_box.estimate_ranges()
    # tighter-or-equal to box (intersecting box with the zonotope bound)
    assert np.all(lbz >= lbb - 1e-9) and np.all(ubz <= ubb + 1e-9)
    # sound: every exact output lies inside the zono-mode output bbox
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(20000, 3))
    with torch.no_grad():
        Y = net(torch.as_tensor(X, dtype=torch.float64)).numpy()
    assert np.all(Y >= lbz.flatten() - 1e-6) and np.all(Y <= ubz.flatten() + 1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_zono_lp_identical_to_lp_cpu(seed):
    """
    zono_lp (zono pre-classify, LP only the zono-unstable) is RESULT-IDENTICAL to
    full lp_cpu: same output star V/C/d and same relaxed-neuron metadata. Only the
    number of LPs solved differs (zono-stable neurons skip the LP). This is the
    pure-speedup guarantee.
    """
    net = _net(seed, dims=(3, 7, 7, 7, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(-np.ones(3), np.ones(3))
    out_lp, meta_lp = relaxed_reach(S_in, layers, {}, bound_mode="lp_cpu")
    out_zlp, meta_zlp = relaxed_reach(S_in, layers, {}, bound_mode="zono_lp")
    _assert_star_equal(out_zlp, out_lp, tol=1e-6)
    _assert_meta_equal(meta_zlp, meta_lp, tol=1e-6)


def test_resume_does_less_work_than_scratch():
    """
    Resuming at L>0 actually REUSES the prefix rather than recomputing: the
    child's checkpoints below L are the parent's exact objects (identity), and
    only the n_relu-L layers at/after L are freshly built. Identity (not just
    length) is what proves work was saved, not merely relabelled.
    """
    net = _net(2, dims=(2, 5, 5, 5, 2))
    layers = extract_layers(net)
    S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    parent, pmeta = relaxed_reach(S_in, layers, {})
    n_relu = len(relu_positions(layers))
    L = max(m.key.layer for m in pmeta)
    assert L >= 1, "need a non-root split layer for this test"
    key = next(m.key for m in pmeta if m.key.layer == L)
    child = resume_reach(parent, layers, {key: Phase.ACTIVE}, "box", L)
    # prefix [0, L) reused verbatim (same Star objects); suffix [L, n) rebuilt.
    assert all(child.checkpoints[i] is parent.checkpoints[i] for i in range(L))
    assert len(child.checkpoints) == n_relu
    assert child.checkpoints[L] is parent.checkpoints[L]  # entry star at L shared
    # the freshly built suffix is a strict subset of all layers (work saved)
    assert n_relu - L < n_relu

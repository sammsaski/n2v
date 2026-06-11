"""Phase 1 sanity tests: activations and normalisations ported from nnVLA.

These tests validate shape, dtype, and basic soundness via random-sample
pushforward containment. They are deliberately small; richer per-layer
oracles live under ``tests/oracles``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Box, Star
from n2v.nn.layer_ops import (
    relu6_reach,
    elu_reach,
    gelu_reach,
    quickgelu_reach,
    silu_reach,
    hardswish_reach,
    layernorm_reach,
    groupnorm_reach,
    rmsnorm_reach,
    grn_reach,
)
from n2v.nn.layers import RMSNorm, GRN

# pylint: disable=missing-function-docstring

# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


@pytest.fixture
def small_box():
    lb = np.array([[-2.0], [-1.0], [0.0], [1.0]])
    ub = np.array([[-1.0], [0.0], [1.0], [2.0]])
    return Box(lb, ub)


@pytest.fixture
def small_star(small_box):
    return Star.from_bounds(small_box.lb, small_box.ub)


def _activation_box_passes(reach_fn, torch_fn, small_box):
    out = reach_fn([small_box])
    assert len(out) == 1
    samples = small_box.sample(64)
    pushed = torch_fn(torch.from_numpy(samples).double()).numpy()
    out_lb = out[0].lb.reshape(-1)
    out_ub = out[0].ub.reshape(-1)
    assert np.all(pushed >= out_lb[:, None] - 1e-5)
    assert np.all(pushed <= out_ub[:, None] + 1e-5)


def test_relu6_box(small_box):
    _activation_box_passes(relu6_reach.relu6_box, nn.ReLU6(), small_box)


def test_elu_box(small_box):
    _activation_box_passes(lambda b: elu_reach.elu_box(b, alpha=1.0), nn.ELU(alpha=1.0), small_box)


def test_gelu_box(small_box):
    _activation_box_passes(gelu_reach.gelu_box, nn.GELU(), small_box)


def test_gelu_tanh_box(small_box):
    """T0-3 (audit C5): the tanh-approximation GELU dips lower than the erf
    form (~-0.170041 vs ~-0.169971). gelu_tanh_box must be a sound box reach
    for nn.GELU(approximate='tanh') -- the GPT-2 / HF default. With the
    previous erf-form routing the box floor was strictly above the tanh-form
    minimum and true outputs at x = -0.7517 fell BELOW the reach floor.
    """
    _activation_box_passes(
        gelu_reach.gelu_tanh_box, nn.GELU(approximate="tanh"), small_box,
    )


def test_gelu_tanh_dip_is_contained():
    """Box reach must contain the true tanh-GELU minimum at x ≈ -0.7525.

    Regression: pre-fix the dispatcher routed nn.GELU(approximate='tanh')
    to gelu_box (erf form) whose F_MIN -0.16997 lies ABOVE the tanh dip
    -0.170040750, so a true output exactly at the dip was excluded from
    the reach.

    Note (audit I1 follow-up): the F_MIN constant tightened from -0.170041
    to -0.17004075058 once the buggy x_min point check was fixed (the
    old looser floor compensated for the rounded x_min). True tanh-GELU
    minimum to 16 digits is -0.17004075057125403 (scipy-verified).
    """
    box = Box(np.array([[-2.0]]), np.array([[1.0]]))
    out = gelu_reach.gelu_tanh_box([box])
    assert len(out) == 1
    # True minimum at high precision (scipy bounded-Brent, xatol=1e-12).
    true_tanh_min = -0.17004075057125403
    assert out[0].lb.flatten()[0] <= true_tanh_min + 1e-12, (
        f"gelu_tanh_box lower bound {out[0].lb.flatten()[0]} is above the "
        f"true tanh-GELU minimum {true_tanh_min} -- box reach is unsound."
    )
    # Cross-check that the erf box would have been unsound for tanh.
    true_erf_min = -0.16997120747990369
    erf_box = gelu_reach.gelu_box([box])
    assert erf_box[0].lb.flatten()[0] <= true_erf_min + 1e-12
    # Box floor for erf form is strictly above the tanh dip, demonstrating
    # the unsoundness window the dispatcher branch closes.
    assert erf_box[0].lb.flatten()[0] > true_tanh_min


def test_quickgelu_box(small_box):
    quickgelu = lambda x: x * torch.sigmoid(1.702 * x)
    _activation_box_passes(quickgelu_reach.quickgelu_box, quickgelu, small_box)


def test_silu_box(small_box):
    _activation_box_passes(silu_reach.silu_box, nn.SiLU(), small_box)


def test_hardswish_box(small_box):
    _activation_box_passes(hardswish_reach.hardswish_box, nn.Hardswish(), small_box)


def test_activation_star_returns_star(small_star):
    for fn in [
        relu6_reach.relu6_star_approx,
        lambda s: elu_reach.elu_star_approx(s, alpha=1.0),
        gelu_reach.gelu_star_approx,
        quickgelu_reach.quickgelu_star_approx,
        silu_reach.silu_star_approx,
        hardswish_reach.hardswish_star_approx,
    ]:
        out = fn([small_star])
        assert len(out) == 1
        assert isinstance(out[0], Star)


# ---------------------------------------------------------------------------
# Normalisations
# ---------------------------------------------------------------------------


def test_layernorm_box():
    layer = nn.LayerNorm(4, eps=1e-5, elementwise_affine=True)
    layer.eval()
    lb = np.array([[-1.0], [-0.5], [0.0], [0.5]])
    ub = np.array([[0.0], [0.5], [1.0], [1.5]])
    out = layernorm_reach.layernorm_box(layer, [Box(lb, ub)])
    assert len(out) == 1
    assert out[0].dim == 4


def test_rmsnorm_box():
    layer = RMSNorm(4, eps=1e-6)
    lb = np.array([[-1.0], [-0.5], [0.0], [0.5]])
    ub = np.array([[0.0], [0.5], [1.0], [1.5]])
    out = rmsnorm_reach.rmsnorm_box(layer, [Box(lb, ub)])
    assert len(out) == 1
    assert out[0].dim == 4


def test_groupnorm_box():
    layer = nn.GroupNorm(num_groups=2, num_channels=4)
    layer.eval()
    lb = np.array([[-1.0], [-0.5], [0.0], [0.5]])
    ub = np.array([[0.0], [0.5], [1.0], [1.5]])
    out = groupnorm_reach.groupnorm_box(layer, [Box(lb, ub)])
    assert len(out) == 1
    assert out[0].dim == 4


def test_grn_box():
    layer = GRN(dim=2)
    lb = np.zeros((4, 1))
    ub = np.ones((4, 1))
    out = grn_reach.grn_box(layer, [Box(lb, ub)])
    assert len(out) == 1
    assert out[0].dim == 4


# ---------------------------------------------------------------------------
# T0-4 (audit C2/C3/C4 + C-high + C7): the multi-token Star paths for
# RMSNorm/LayerNorm/GroupNorm, elu_reach for negative alpha, and
# linear_attention_box for mixed-sign V must FAIL LOUD until Commit 7 /
# Commit 8 land. These tests pin the raises so future PRs cannot regress
# back into silently-wrong reach.
# ---------------------------------------------------------------------------


def test_rmsnorm_star_raises_on_multi_token():
    layer = RMSNorm(2, eps=1e-6)  # normalized_shape=2, L=2 tokens -> 4 dims
    layer.eval()
    # 2 tokens x 2 D = 4 dims. Predicate-preserving path requires V.size > 0
    # so use a non-degenerate Star (V = identity from from_bounds).
    lb = np.array([[0.0], [0.0], [0.0], [0.0]])
    ub = np.array([[1.0], [1.0], [1.0], [1.0]])
    star = Star.from_bounds(lb, ub)
    with pytest.raises(NotImplementedError, match="multi-token"):
        rmsnorm_reach.rmsnorm_star_approx(layer, [star])


def test_layernorm_star_raises_on_multi_token():
    layer = nn.LayerNorm(2)
    layer.eval()
    lb = np.array([[0.0], [0.0], [0.0], [0.0]])
    ub = np.array([[1.0], [1.0], [1.0], [1.0]])
    star = Star.from_bounds(lb, ub)
    with pytest.raises(NotImplementedError, match="multi-token"):
        layernorm_reach.layernorm_star_approx(layer, [star])


def test_groupnorm_star_raises_on_multi_group():
    layer = nn.GroupNorm(num_groups=2, num_channels=4)
    layer.eval()
    lb = np.array([[-1.0], [-0.5], [0.0], [0.5]])
    ub = np.array([[0.0], [0.5], [1.0], [1.5]])
    star = Star.from_bounds(lb, ub)
    with pytest.raises(NotImplementedError, match="multi-group"):
        groupnorm_reach.groupnorm_star_approx(layer, [star])


# ---------------------------------------------------------------------------
# PR-1 audit N9: positive counterparts for the T0-4 raises. A regression
# that raised ``NotImplementedError`` for EVERY Star input -- not just the
# multi-token / multi-group case -- would silently pass the raise-tests
# above but make the layer unusable. These tests pin the SINGLE-token /
# SINGLE-group SUCCESS path so the regression case can't slip in unnoticed.
# ---------------------------------------------------------------------------


def test_rmsnorm_star_single_token_succeeds_audit_N9():
    """RMSNorm Star path must SUCCEED on a single-token input (the case
    that ``test_rmsnorm_star_raises_on_multi_token`` does NOT exercise)."""
    layer = RMSNorm(2, eps=1e-6)
    layer.eval()
    lb = np.array([[0.0], [0.0]])
    ub = np.array([[1.0], [1.0]])
    star = Star.from_bounds(lb, ub)
    out = rmsnorm_reach.rmsnorm_star_approx(layer, [star])
    assert len(out) == 1
    assert out[0].dim == 2


def test_layernorm_star_single_token_succeeds_audit_N9():
    """LayerNorm Star path must SUCCEED on a single-token input."""
    layer = nn.LayerNorm(2)
    layer.eval()
    lb = np.array([[0.0], [0.0]])
    ub = np.array([[1.0], [1.0]])
    star = Star.from_bounds(lb, ub)
    out = layernorm_reach.layernorm_star_approx(layer, [star])
    assert len(out) == 1
    assert out[0].dim == 2


def test_groupnorm_star_single_group_succeeds_audit_N9():
    """GroupNorm Star path must SUCCEED with num_groups=1 (single
    group, where the multi-group restriction does not apply)."""
    layer = nn.GroupNorm(num_groups=1, num_channels=4)
    layer.eval()
    lb = np.array([[-1.0], [-0.5], [0.0], [0.5]])
    ub = np.array([[0.0], [0.5], [1.0], [1.5]])
    star = Star.from_bounds(lb, ub)
    out = groupnorm_reach.groupnorm_star_approx(layer, [star])
    assert len(out) == 1
    assert out[0].dim == 4


def test_elu_raises_on_negative_alpha():
    """T0-4 (audit C-high): elu_reach assumes alpha >= 0 (monotone)."""
    box = Box(np.array([[-3.0]]), np.array([[2.0]]))
    with pytest.raises(NotImplementedError, match="alpha"):
        elu_reach.elu_box([box], alpha=-1.0)
    star = Star.from_bounds(np.array([[-3.0]]), np.array([[2.0]]))
    with pytest.raises(NotImplementedError, match="alpha"):
        elu_reach.elu_star_approx([star], alpha=-1.0)


def test_linear_attention_four_corner_matmul_handles_negative_v():
    """T1-9 (audit C7): linear_attention_box uses a SOUND four-corner
    interval matmul for phi(k).T @ v, valid for any sign of V.

    Counterexample from the external audit: phi_k in [1, 3], v in [-2, 1]
    -> true kv range is [-6, 3]. The previous two-corner formula returned
    [-2, 3] (missing -6); the four-corner formula returns [-6, 3].

    We do not check the *outer* phi(q) @ kv composition here since
    phi(q) is always non-negative (phi(x) = elu(x) + 1 + 1e... > 0); the
    bug was in the inner phi(k).T @ v product.
    """
    from n2v.nn.layer_ops import linear_attention_reach as lin_attn

    # L_k = 1, d_v = 1; phi_k ranges over [1, 3] via input k in [0, 2]
    # (phi(0) = 1, phi(2) = 3), v ranges over [-2, 1].
    q = Box(np.array([[0.0]]), np.array([[2.0]]))
    k = Box(np.array([[0.0]]), np.array([[2.0]]))
    v = Box(np.array([[-2.0]]), np.array([[1.0]]))

    # Manual ground-truth check of the inner kv product: the four-corner
    # interval product of [1, 3] x [-2, 1] is min over corners = -6,
    # max = 3. Then phi_q @ kv composes (phi_q in [1, 3]). With kv in
    # [-6, 3], the outer four-corner product of phi_q in [1, 3] and kv
    # in [-6, 3] is min(1*-6, 1*3, 3*-6, 3*3) = -18, max = 9.
    out = lin_attn.linear_attention_box([q], [k], [v], l_q=1, d_v=1)
    out_lb = float(out[0].lb.flatten()[0])
    out_ub = float(out[0].ub.flatten()[0])
    assert out_lb <= -18.0 + 1e-9, (
        f"linear_attention_box lower bound {out_lb} is above the true "
        f"minimum -18.0 -- worst corner dropped (unsound)."
    )
    assert out_ub >= 9.0 - 1e-9, (
        f"linear_attention_box upper bound {out_ub} is below the true "
        f"maximum 9.0."
    )

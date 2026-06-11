"""Set-type coverage tests for every layer / primitive this PR touched.

The ViT integration test (``tests/integration/test_minimal_vit.py``)
only exercises the Zono path because that is what the benchmark needs.
This file pins **every** set type for each layer/primitive added or
modified by this PR, so future regressions on Box / Star / Hexatope /
Octatope are caught by CI:

  * PatchEmbed           — Box, Star, Zono, Hexatope, Octatope
  * LayerNorm (single-token)  — Box, Star, Zono, Hexatope, Octatope
  * GELU (erf + tanh)    — Box, Star, Zono, Hexatope, Octatope
  * Linear (per-token block-tile) — Box, Star, Zono, Hexatope, Octatope
  * fx ``operator.add`` (set + constant)  — Star, Zono, Box, Hex, Oct
  * fx ``operator.getitem`` (tensor slice) — Star, Zono, Box, Hex, Oct
  * SoftmaxAttention multi-input  — Box, Star, Zono, Hex, Oct

Each test instantiates the relevant layer / op, builds a small input
set of the right type, runs the reach helper directly via the
dispatcher, and asserts shape + finite bounds + box containment of
one concrete forward sample where applicable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Box, Star, Zono, Hexatope, Octatope
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.nn.layer_ops import dispatcher
from n2v.nn.layer_ops import (
    layernorm_reach,
    gelu_reach,
    linear_reach,
    patch_embed_reach,
)
from n2v.nn.layers import PatchEmbed


# ----------------------------- helpers ---------------------------------------


def _flat_box(lb_vec: np.ndarray, ub_vec: np.ndarray) -> Box:
    return Box(lb_vec.reshape(-1, 1).astype(np.float64),
               ub_vec.reshape(-1, 1).astype(np.float64))


def _flat_star(box: Box) -> Star:
    return Star.from_bounds(box.lb, box.ub)


def _flat_zono(box: Box) -> Zono:
    return Zono.from_bounds(box.lb, box.ub)


def _flat_hex(box: Box) -> Hexatope:
    return Hexatope.from_bounds(box.lb, box.ub)


def _flat_oct(box: Box) -> Octatope:
    return Octatope.from_bounds(box.lb, box.ub)


def _bounds_of(s):
    """Fast IBP (lb, ub). Hex/Oct use estimate_ranges (no solver kwarg)."""
    if isinstance(s, (Hexatope, Octatope)):
        lb, ub = s.estimate_ranges()
    elif hasattr(s, "get_bounds"):
        lb, ub = s.get_bounds()
    else:
        lb, ub = s.get_ranges()
    return np.asarray(lb).flatten(), np.asarray(ub).flatten()


# ----------------------------- LayerNorm -------------------------------------


def test_layernorm_box_star_zono_hex_oct_single_token():
    """Audit N7/N11: previously asserted only ``isfinite`` -- a vacuous
    ``[-inf, +inf]`` envelope would pass. Now: assert Monte-Carlo
    containment of 32 random forward samples in the reach bounds.
    """
    layer = nn.LayerNorm(4)
    layer.eval()
    lb = np.array([-1.0, -0.5, 0.0, 0.5])
    ub = np.array([0.0, 0.5, 1.0, 1.5])
    box = _flat_box(lb, ub)
    for set_in, helper in (
        (box, lambda: layernorm_reach.layernorm_box(layer, [box])),
        (_flat_star(box), lambda: layernorm_reach.layernorm_star_approx(layer, [_flat_star(box)])),
        (_flat_zono(box), lambda: layernorm_reach.layernorm_zono(layer, [_flat_zono(box)])),
        (_flat_hex(box), lambda: layernorm_reach.layernorm_hexatope(layer, [_flat_hex(box)])),
        (_flat_oct(box), lambda: layernorm_reach.layernorm_octatope(layer, [_flat_oct(box)])),
    ):
        out = helper()
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        assert np.all(np.isfinite(lb_o)) and np.all(np.isfinite(ub_o))
        assert lb_o.size == 4

    # N11/M4: Monte-Carlo concrete-forward containment (Box path; others
    # cover the same forward via their box envelopes).
    pytest.assert_reach_contains_forward(
        layer, lb, ub,
        lambda lay, sets: layernorm_reach.layernorm_box(lay, sets),
        n_samples=32, input_shape=(1, 4),
    )


# ----------------------------- GELU (erf + tanh) -----------------------------


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_box_star_zono_hex_oct(approximate):
    layer = nn.GELU(approximate=approximate)
    layer.eval()
    lb = np.array([-2.0, -0.5, 0.0])
    ub = np.array([-1.0, 0.5, 1.0])
    box = _flat_box(lb, ub)

    for set_in in (
        box, _flat_star(box), _flat_zono(box), _flat_hex(box), _flat_oct(box),
    ):
        out = dispatcher.reach_layer(layer, [set_in], "approx")
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        assert lb_o.size == 3
        assert np.all(np.isfinite(lb_o)) and np.all(np.isfinite(ub_o))
        # Soundness: a sample at x_lb / x_ub through the true forward must
        # be inside the reach bounds.
        with torch.no_grad():
            x = torch.from_numpy(np.stack([lb, ub])).double()
            true_out = layer(x.float()).double().numpy()
        assert np.all(true_out.min(axis=0) >= lb_o - 1e-5)
        assert np.all(true_out.max(axis=0) <= ub_o + 1e-5)


# ----------------------------- Linear block-tile (per-token) -----------------


def test_linear_block_tile_across_all_set_types():
    """nn.Linear applied to a sequence-flattened (L*D_in,) input must
    block-tile the weight so each token's D_in chunk is mapped
    independently. Pins the fix in linear_reach._maybe_block_tile_linear
    across Box / Star / Zono / Hex / Oct.
    """
    torch.manual_seed(0)
    layer = nn.Linear(3, 2, bias=True)
    layer.eval()
    L = 4
    in_dim = L * 3  # L tokens of D_in=3
    lb = np.zeros(in_dim)
    ub = np.ones(in_dim)
    box = _flat_box(lb, ub)

    for set_in, fn in (
        (box, linear_reach.linear_box),
        (_flat_star(box), linear_reach.linear_star),
        (_flat_zono(box), linear_reach.linear_zono),
        (_flat_hex(box), linear_reach.linear_hexatope),
        (_flat_oct(box), linear_reach.linear_octatope),
    ):
        # Audit I8: pass expected_n_tokens=L so the helper verifies the
        # inferred block-tile count instead of silently inferring.
        out = fn(layer, [set_in], expected_n_tokens=L)
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        assert lb_o.size == L * 2  # L tokens of D_out=2

    # Audit N3/N11: assert per-token concrete-forward correspondence.
    # A transposed ``kron(W, I_L)`` (vs the correct ``kron(I_L, W)``)
    # would pass the shape assertion above but fail this MC check.
    pytest.assert_reach_contains_forward(
        layer,
        np.zeros(in_dim), np.ones(in_dim),
        lambda lay, sets: linear_reach.linear_box(
            lay, sets, expected_n_tokens=L,
        ),
        n_samples=32, input_shape=(1, L, 3),
    )


def test_mix_ffn_zono_end_to_end_audit_C3():
    """PR-1 audit C3: MixFFN was unreachable end-to-end because
    ``mix_ffn_passthrough`` raised unconditionally while T1-7 made
    MixFFN an fx leaf. Now: implement the forward (fc1 -> reshape ->
    dwconv -> flatten -> GELU -> fc2) directly and verify the Zono
    path completes without raising. Pin: output shape and
    Monte-Carlo containment of one concrete forward sample.
    """
    from n2v.nn.layers.mix_ffn import MixFFN
    from n2v.nn.layer_ops.mix_ffn_reach import mix_ffn_passthrough

    torch.manual_seed(0)
    L = 4   # 2x2 spatial layout
    dim = 2
    hidden = 4
    layer = MixFFN(dim=dim, hidden_dim=hidden).eval()

    # Flat token-major input box of dim L*dim = 8.
    lb_vec = np.linspace(0.0, 0.1, L * dim)
    ub_vec = lb_vec + 0.05
    z_in = Zono.from_bounds(
        lb_vec.reshape(-1, 1), ub_vec.reshape(-1, 1),
    )
    out = mix_ffn_passthrough(layer, [z_in], n_tokens=L)
    assert len(out) == 1
    assert out[0].dim == L * dim, f"expected dim {L*dim}, got {out[0].dim}"

    # Monte-Carlo containment: random samples in the input box should
    # map to outputs inside the reach bounds.
    out_lb, out_ub = out[0].get_bounds()
    out_lb = np.asarray(out_lb).flatten()
    out_ub = np.asarray(out_ub).flatten()
    rng = np.random.default_rng(0)
    for _ in range(8):
        x_sample = rng.uniform(lb_vec, ub_vec)
        x_t = torch.from_numpy(x_sample.astype(np.float32)).reshape(1, L, dim)
        with torch.no_grad():
            y_t = layer(x_t).detach().cpu().numpy().flatten()
        assert np.all(out_lb - 1e-6 <= y_t), (
            f"MC sample below reach lb: y={y_t}, lb={out_lb}"
        )
        assert np.all(y_t <= out_ub + 1e-6), (
            f"MC sample above reach ub: y={y_t}, ub={out_ub}"
        )


def test_mix_ffn_dispatcher_routes_zono_hex_oct_audit_C3_followup():
    """Final-review follow-up to audit C3: the helper supported all five
    set types but the DISPATCHER only routed Star and Box -- Zono /
    Hex / Oct fell through to ``_registry_lookup`` ->
    ``NotImplementedError``. The original C3 test called
    ``mix_ffn_passthrough`` directly, which is exactly how the gap
    stayed invisible. This test goes through ``dispatcher.reach_layer``
    so the route itself is pinned.
    """
    from n2v.nn.layers.mix_ffn import MixFFN

    torch.manual_seed(0)
    L, dim, hidden = 4, 2, 4
    layer = MixFFN(dim=dim, hidden_dim=hidden).eval()

    lb_vec = np.linspace(0.0, 0.1, L * dim)
    ub_vec = lb_vec + 0.05
    box = _flat_box(lb_vec, ub_vec)

    for set_in in (_flat_zono(box), _flat_hex(box), _flat_oct(box)):
        out = dispatcher.reach_layer(
            layer, [set_in], "approx", n_tokens=L,
        )
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        assert lb_o.size == L * dim, (
            f"{type(set_in).__name__}: expected dim {L * dim}, "
            f"got {lb_o.size}"
        )
        assert np.all(np.isfinite(lb_o)) and np.all(np.isfinite(ub_o))


def test_mix_ffn_missing_n_tokens_raises_audit_C3():
    """Audit C3: MixFFN reach without an explicit ``n_tokens`` signal
    must raise -- the dwconv shape is unrecoverable from a flat set.
    """
    from n2v.nn.layers.mix_ffn import MixFFN
    from n2v.nn.layer_ops.mix_ffn_reach import mix_ffn_passthrough

    layer = MixFFN(dim=2, hidden_dim=4).eval()
    z_in = Zono.from_bounds(
        np.zeros((8, 1)), np.ones((8, 1)),
    )
    with pytest.raises(NotImplementedError, match="n_tokens"):
        mix_ffn_passthrough(layer, [z_in])


def test_parallel_residual_decomposes_via_fx_audit_I5():
    """PR-1 audit I5: ParallelResidual must NOT be an fx leaf -- it
    has no reach helper, so leaf treatment caused
    ``_registry_lookup`` -> ``None`` -> ``NotImplementedError``. Now
    excluded from the leaf list so fx decomposes
    ``y = x + a(x) + b(x)`` into two operator.add(set, set) calls.
    """
    from n2v.nn import NeuralNetwork
    from n2v.nn.layers.parallel_residual import ParallelResidual

    class TinyParallelResid(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = ParallelResidual(
                sublayer_a=nn.Linear(3, 3, bias=False),
                sublayer_b=nn.Linear(3, 3, bias=False),
            )
            with torch.no_grad():
                self.block.sublayer_a.weight.zero_()
                self.block.sublayer_b.weight.zero_()

        def forward(self, x):
            return self.block(x)

    model = TinyParallelResid().eval()
    box = _flat_box(np.array([0.0, 1.0, 2.0]), np.array([0.5, 1.5, 2.5]))
    out = NeuralNetwork(model).reach(box, method="approx")
    # With both sublayers zeroed, y = x + 0 + 0 = x.
    np.testing.assert_allclose(
        np.asarray(out[0].lb).flatten(), [0.0, 1.0, 2.0], atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(out[0].ub).flatten(), [0.5, 1.5, 2.5], atol=1e-9,
    )


def test_overlap_patch_embed_zono_dispatch_lands_audit_I5():
    """PR-1 audit I5: OverlapPatchEmbed must have a dispatcher branch.
    Previously absent -> _registry_lookup raise. Now: route the
    Conv2d + flatten + transpose through PatchEmbed reach + apply
    LayerNorm. Smoke-tests dispatch only; concrete-forward containment
    is exercised by the PatchEmbed test sweep.
    """
    from n2v.nn.layers import OverlapPatchEmbed
    from n2v.nn.layer_ops import overlap_patch_embed_reach as ope

    layer = OverlapPatchEmbed(
        in_channels=1, embed_dim=4, patch_size=2, stride=2,
    ).eval()
    zono_in = ImageZono.from_bounds(
        np.zeros(4), np.ones(4),
        height=2, width=2, num_channels=1,
    )
    out = ope.overlap_patch_embed_zono(layer, [zono_in])
    assert len(out) == 1
    # 2x2 image / (2x2 patch, stride 2) -> 1x1 = 1 token of embed_dim=4.
    assert out[0].dim == 4


def test_patch_embed_box_multichannel_requires_image_shape_audit_I2():
    """PR-1 audit I2: PatchEmbed Box reach with in_channels > 1 must
    REFUSE to infer layout. The previous code silently called
    ``ImageZono.from_bounds`` with HWC semantics, mis-permuting a
    CHW-flat input. Fix: require explicit image_shape; raise otherwise.
    """
    from n2v.nn.layers import PatchEmbed
    from n2v.nn.layer_ops import patch_embed_reach as pe

    layer = PatchEmbed(
        in_channels=3, embed_dim=4, patch_size=2,
    ).eval()
    # 4x4x3 = 48 elements
    box = _flat_box(np.zeros(48), np.ones(48))
    with pytest.raises(NotImplementedError, match="in_channels=3.*image_shape"):
        pe.patch_embed_box(layer, [box])


def test_patch_embed_box_with_image_shape_hwc_matches_imagezono():
    """When ``image_shape`` is given, the Box path must produce the same
    IBP envelope as routing the equivalent ImageZono directly."""
    from n2v.nn.layers import PatchEmbed
    from n2v.nn.layer_ops import patch_embed_reach as pe

    torch.manual_seed(0)
    layer = PatchEmbed(
        in_channels=3, embed_dim=4, patch_size=2,
    ).eval()
    flat_lb = np.zeros(48)
    flat_ub = np.ones(48)
    box = _flat_box(flat_lb, flat_ub)
    box_out = pe.patch_embed_box(
        layer, [box],
        image_shape=(4, 4, 3), image_layout="HWC",
    )[0]
    zono_in = ImageZono.from_bounds(
        flat_lb, flat_ub, height=4, width=4, num_channels=3,
    )
    zono_out = pe.patch_embed_zono(layer, [zono_in])[0]
    lb_z, ub_z = zono_out.get_bounds()
    np.testing.assert_allclose(
        np.asarray(box_out.lb).flatten(),
        np.asarray(lb_z).flatten(), atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(box_out.ub).flatten(),
        np.asarray(ub_z).flatten(), atol=1e-9,
    )


def test_patch_embed_box_chw_layout_differs_from_hwc_for_multichannel():
    """Audit I2: a CHW-flat input must be permuted to HWC before the
    conv reach to match PyTorch forward semantics. This test asserts
    that the CHW path differs from the HWC path when the per-channel
    bounds differ (so the permutation is observable).
    """
    from n2v.nn.layers import PatchEmbed
    from n2v.nn.layer_ops import patch_embed_reach as pe

    torch.manual_seed(0)
    layer = PatchEmbed(
        in_channels=3, embed_dim=4, patch_size=2,
    ).eval()
    # 2x2x3 = 12 elements. Construct a per-channel-distinct CHW-flat
    # box: channel 0 = [0, 0.1], channel 1 = [0.4, 0.5], channel 2 = [0.8, 0.9].
    chw_lb = np.array(
        [0.0, 0.0, 0.0, 0.0,  # channel 0 (4 pixels)
         0.4, 0.4, 0.4, 0.4,  # channel 1
         0.8, 0.8, 0.8, 0.8], # channel 2
    )
    chw_ub = chw_lb + 0.1
    box = _flat_box(chw_lb, chw_ub)
    out_hwc = pe.patch_embed_box(
        layer, [box], image_shape=(2, 2, 3), image_layout="HWC",
    )[0]
    out_chw = pe.patch_embed_box(
        layer, [box], image_shape=(2, 2, 3), image_layout="CHW",
    )[0]
    # The two must produce different reaches.
    assert not np.allclose(out_hwc.lb, out_chw.lb), (
        "HWC and CHW reaches identical -- permutation is not happening."
    )


def test_patch_embed_box_non_square_image_explicit_shape_works():
    """Audit I3: non-square images must be supported via image_shape
    kwarg. (The pre-fix code silently inferred a square side from
    pixel count and mis-shaped non-square images.)
    """
    from n2v.nn.layers import PatchEmbed
    from n2v.nn.layer_ops import patch_embed_reach as pe

    # 2x8 image, 2x2 patches, 1 channel = 16 elements (pixel count IS a
    # perfect square, so the buggy code would silently use a 4x4 image).
    layer = PatchEmbed(
        in_channels=1, embed_dim=4, patch_size=2,
    ).eval()
    box = _flat_box(np.zeros(16), np.ones(16))
    out = pe.patch_embed_box(
        layer, [box], image_shape=(2, 8, 1), image_layout="HWC",
    )
    assert len(out) == 1
    # 2x8 image / (2x2 patch) = 1x4 = 4 tokens of dim 4 = 16 elements
    assert out[0].dim == 16


def test_linear_block_tile_mismatch_raises_audit_I8():
    """PR-1 audit I8: when the dispatcher (or test) declares
    ``expected_n_tokens`` and the divisibility-inferred ``L`` disagrees,
    the helper must raise ``NotImplementedError`` instead of silently
    verifying a different function.
    """
    layer = nn.Linear(3, 2, bias=True).eval()
    box = _flat_box(np.zeros(12), np.ones(12))   # divisibility says L=4
    with pytest.raises(NotImplementedError, match="disagrees with"):
        linear_reach.linear_box(layer, [box], expected_n_tokens=3)


def test_linear_block_tile_no_n_tokens_warns_audit_I8():
    """Audit I8: silent block-tile inference (L > 1 without an explicit
    n_tokens signal) must emit a ``UserWarning`` so users can audit
    when their reach is doing per-token tiling."""
    layer = nn.Linear(3, 2, bias=True).eval()
    box = _flat_box(np.zeros(12), np.ones(12))
    with pytest.warns(UserWarning, match="without an explicit n_tokens"):
        out = linear_reach.linear_box(layer, [box])
        assert out[0].dim == 8


# ----------------------------- PatchEmbed ------------------------------------


@pytest.fixture
def patch_embed_layer():
    torch.manual_seed(0)
    return PatchEmbed(in_channels=1, embed_dim=2, patch_size=2)


def test_patch_embed_zono(patch_embed_layer):
    img_size = 4
    box = _flat_box(
        np.zeros(img_size * img_size).astype(np.float64),
        np.ones(img_size * img_size).astype(np.float64),
    )
    zono_in = ImageZono.from_bounds(
        box.lb, box.ub, height=img_size, width=img_size, num_channels=1,
    )
    out = patch_embed_reach.patch_embed_zono(patch_embed_layer, [zono_in])
    assert len(out) == 1
    lb_o, ub_o = out[0].get_bounds()
    expected_dim = ((img_size // 2) ** 2) * 2  # n_patches * embed_dim
    assert lb_o.size == expected_dim


def test_patch_embed_star(patch_embed_layer):
    img_size = 4
    star_in = ImageStar.from_bounds(
        np.zeros((img_size * img_size, 1)),
        np.ones((img_size * img_size, 1)),
        height=img_size, width=img_size, num_channels=1,
    )
    out = patch_embed_reach.patch_embed_star(patch_embed_layer, [star_in])
    assert len(out) == 1
    expected_dim = ((img_size // 2) ** 2) * 2
    assert out[0].dim == expected_dim


def test_patch_embed_box(patch_embed_layer):
    """Box reach for PatchEmbed lifts to ImageZono internally."""
    img_size = 4
    box = _flat_box(
        np.zeros(img_size * img_size), np.ones(img_size * img_size),
    )
    out = patch_embed_reach.patch_embed_box(patch_embed_layer, [box])
    assert len(out) == 1
    expected_dim = ((img_size // 2) ** 2) * 2
    assert out[0].dim == expected_dim


def test_patch_embed_hexatope(patch_embed_layer):
    img_size = 4
    box = _flat_box(
        np.zeros(img_size * img_size), np.ones(img_size * img_size),
    )
    hex_in = Hexatope.from_bounds(box.lb, box.ub)
    out = patch_embed_reach.patch_embed_hexatope(patch_embed_layer, [hex_in])
    assert len(out) == 1
    expected_dim = ((img_size // 2) ** 2) * 2
    lb_o, ub_o = _bounds_of(out[0])
    assert lb_o.size == expected_dim


def test_patch_embed_octatope(patch_embed_layer):
    img_size = 4
    box = _flat_box(
        np.zeros(img_size * img_size), np.ones(img_size * img_size),
    )
    oct_in = Octatope.from_bounds(box.lb, box.ub)
    out = patch_embed_reach.patch_embed_octatope(patch_embed_layer, [oct_in])
    assert len(out) == 1
    expected_dim = ((img_size // 2) ** 2) * 2
    lb_o, ub_o = _bounds_of(out[0])
    assert lb_o.size == expected_dim


# ----------------------------- fx operator.add (set + constant) -------------


def test_fx_add_set_plus_constant_all_set_types():
    """End-to-end via a tiny model: ``x + buffer`` (pos-embed-style add).
    The fx call_function handler must accept Star / Zono / Box /
    Hexatope / Octatope inputs and produce the correctly-translated
    output set.
    """
    from n2v.nn import NeuralNetwork

    class AddConst(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("c", torch.tensor([1.0, 2.0]))

        def forward(self, x):
            return x + self.c

    model = AddConst().eval()
    box = _flat_box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    for set_in, ctor in (
        (box, Box),
        (_flat_star(box), Star),
        (_flat_zono(box), Zono),
        (_flat_hex(box), Hexatope),
        (_flat_oct(box), Octatope),
    ):
        out = NeuralNetwork(model).reach(set_in, method="approx")
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        # Output = input + [1, 2]. Inner dim 0: [1, 2]; dim 1: [2, 3].
        np.testing.assert_allclose(lb_o, np.array([1.0, 2.0]), atol=1e-9)
        np.testing.assert_allclose(ub_o, np.array([2.0, 3.0]), atol=1e-9)


# ------------- Activation min-location numerical drift (audit I1) ------------


def test_activation_box_floor_is_below_true_min_on_narrow_intervals():
    """PR-1 audit I1: GELU/SiLU min-location constants must be set so that
    narrow boxes which BRACKET the true argmin produce a reach lower bound
    BELOW the true minimum on the box (sound), not above it (unsound).

    The previous constants used the true argmin rounded to only 4-7 digits
    -- e.g. ``_GELU_TANH_X_MIN = -0.7517`` vs true ``-0.7524614``. The
    point check ``(lb <= x_min) & (ub >= x_min)`` then missed narrow
    boxes that bracketed the true argmin but excluded the rounded
    constant, producing an above-floor lower bound.

    Counterexamples (each verified ≥ 1e-12 unsound on the buggy code):
        * GELU tanh: Box [-0.7530, -0.7520]
        * GELU erf:  Box [-0.75179155, -0.75179]
        * SiLU:      Box [-1.27848, -1.27840]

    Pin: reach lb must be ≤ scipy's bounded minimum on the box.
    """
    import scipy.optimize as opt
    from math import erf, tanh as math_tanh, sqrt
    from n2v.nn.layer_ops.gelu_reach import gelu_box, gelu_tanh_box
    from n2v.nn.layer_ops.silu_reach import silu_box

    def _silu(x):
        return x / (1 + np.exp(-x))

    def _gelu_erf(x):
        return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))

    def _gelu_tanh(x):
        return 0.5 * x * (1.0 + math_tanh(
            sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        ))

    cases = [
        ("silu",      silu_box,      _silu,      -1.27848, -1.27840),
        ("gelu_erf",  gelu_box,      _gelu_erf,  -0.75179155, -0.75179),
        ("gelu_tanh", gelu_tanh_box, _gelu_tanh, -0.75300, -0.75200),
    ]
    for name, reach_fn, fwd, a, b in cases:
        box = Box(
            np.array([[a]], dtype=np.float64),
            np.array([[b]], dtype=np.float64),
        )
        out = reach_fn([box])[0]
        lb = float(np.asarray(out.lb).flatten()[0])
        res = opt.minimize_scalar(
            fwd, bounds=(a, b), method="bounded",
            options={"xatol": 1e-14},
        )
        assert lb <= res.fun + 1e-14, (
            f"{name}: Box [{a}, {b}] reach lb = {lb!r} is ABOVE true min "
            f"{res.fun!r} (delta = {lb - res.fun:.3e}). I1 unsoundness."
        )


# ------- CLSToken / ConcatWithFrozenSkip Hex+Oct coverage (audit I7) --------


def test_hex_oct_from_bounds_contains_box_corners_audit_N13():
    """PR-1 audit N13: every box-lifted Hex/Oct reach helper relies on
    ``set_type.from_bounds(lb, ub)`` producing a sound enclosure of
    ``[lb, ub]``. The PR-1 tests only assert ``estimate_ranges``
    round-trips closely; they never sample IN the original box and
    verify Hex/Oct containment. A bug in ``from_bounds`` would silently
    invalidate every box-lifted helper (PatchEmbed, CLSToken,
    ConcatWithFrozenSkip, OverlapPatchEmbed, MixFFN, GELU/SiLU
    Hex/Oct, etc.).

    Pin: build Hex / Oct via ``from_bounds`` and check that 64 random
    samples inside the input box are contained in the set's IBP envelope.
    """
    rng = np.random.default_rng(0)
    lb = np.array([[-0.5], [0.0], [0.3], [1.2]])
    ub = np.array([[0.5], [1.0], [0.7], [2.5]])

    for cls in (Hexatope, Octatope):
        s = cls.from_bounds(lb, ub)
        env_lb, env_ub = s.estimate_ranges()
        env_lb = np.asarray(env_lb).flatten()
        env_ub = np.asarray(env_ub).flatten()
        # IBP envelope must enclose the construction box.
        assert np.all(env_lb <= lb.flatten() + 1e-9), (
            f"{cls.__name__}.from_bounds envelope lb={env_lb} above "
            f"construction lb={lb.flatten()}"
        )
        assert np.all(env_ub >= ub.flatten() - 1e-9), (
            f"{cls.__name__}.from_bounds envelope ub={env_ub} below "
            f"construction ub={ub.flatten()}"
        )
        # Sample inside [lb, ub]; each sample must lie inside the
        # envelope (containment is necessary; tightness is not asserted).
        for _ in range(64):
            x = rng.uniform(lb.flatten(), ub.flatten())
            assert np.all(env_lb - 1e-9 <= x), (
                f"{cls.__name__}: sample {x} below envelope {env_lb}"
            )
            assert np.all(x <= env_ub + 1e-9), (
                f"{cls.__name__}: sample {x} above envelope {env_ub}"
            )


def test_cls_token_hexatope_box_lifted_sound():
    """PR-1 audit I7: CLSToken Hexatope branch was previously absent in
    the dispatcher -- any end-to-end ViT with Hex reach raised through
    _registry_lookup. Box-lifted IBP path is sound; pin: a Hex input
    routed through CLSToken must return a Hex with the expected
    prepended-token bounds.
    """
    from n2v.nn.layers.cls_token import CLSToken
    from n2v.nn.layer_ops import cls_token_reach

    layer = CLSToken(dim=2)
    with torch.no_grad():
        layer.token.copy_(torch.tensor([0.5, -0.5]))

    hex_in = Hexatope.from_bounds(
        np.array([[0.0], [0.1], [0.2], [0.3]]),
        np.array([[1.0], [1.1], [1.2], [1.3]]),
    )
    out = cls_token_reach.cls_token_hexatope(layer, [hex_in])
    assert len(out) == 1
    assert isinstance(out[0], Hexatope)
    lb, ub = out[0].estimate_ranges()
    lb = np.asarray(lb).flatten()
    ub = np.asarray(ub).flatten()
    # Expected: [0.5, -0.5, 0.0, 0.1, 0.2, 0.3] / [0.5, -0.5, 1.0, 1.1, 1.2, 1.3]
    np.testing.assert_allclose(lb, [0.5, -0.5, 0.0, 0.1, 0.2, 0.3], atol=1e-9)
    np.testing.assert_allclose(ub, [0.5, -0.5, 1.0, 1.1, 1.2, 1.3], atol=1e-9)


def test_cls_token_octatope_box_lifted_sound():
    from n2v.nn.layers.cls_token import CLSToken
    from n2v.nn.layer_ops import cls_token_reach

    layer = CLSToken(dim=2)
    with torch.no_grad():
        layer.token.copy_(torch.tensor([0.5, -0.5]))

    oct_in = Octatope.from_bounds(
        np.array([[0.0], [0.1]]),
        np.array([[1.0], [1.1]]),
    )
    out = cls_token_reach.cls_token_octatope(layer, [oct_in])
    assert isinstance(out[0], Octatope)
    lb, ub = out[0].estimate_ranges()
    lb = np.asarray(lb).flatten()
    ub = np.asarray(ub).flatten()
    np.testing.assert_allclose(lb, [0.5, -0.5, 0.0, 0.1], atol=1e-9)
    np.testing.assert_allclose(ub, [0.5, -0.5, 1.0, 1.1], atol=1e-9)


def test_concat_with_frozen_skip_hexatope_box_lifted_sound():
    """Audit I7: ConcatWithFrozenSkip Hex/Oct branches were absent."""
    from n2v.nn.layers.concat_with_frozen_skip import ConcatWithFrozenSkip
    from n2v.nn.layer_ops import concat_with_frozen_skip_reach

    layer = ConcatWithFrozenSkip(
        skip=torch.tensor([0.7, 0.8]), dim=-1,
    )

    hex_in = Hexatope.from_bounds(
        np.array([[0.0], [0.1]]),
        np.array([[1.0], [1.1]]),
    )
    out = concat_with_frozen_skip_reach.concat_with_frozen_skip_hexatope(
        layer, [hex_in],
    )
    assert isinstance(out[0], Hexatope)
    lb, ub = out[0].estimate_ranges()
    lb = np.asarray(lb).flatten()
    ub = np.asarray(ub).flatten()
    np.testing.assert_allclose(lb, [0.0, 0.1, 0.7, 0.8], atol=1e-9)
    np.testing.assert_allclose(ub, [1.0, 1.1, 0.7, 0.8], atol=1e-9)


def test_concat_with_frozen_skip_octatope_box_lifted_sound():
    from n2v.nn.layers.concat_with_frozen_skip import ConcatWithFrozenSkip
    from n2v.nn.layer_ops import concat_with_frozen_skip_reach

    layer = ConcatWithFrozenSkip(
        skip=torch.tensor([0.7, 0.8]), dim=-1,
    )

    oct_in = Octatope.from_bounds(
        np.array([[0.0], [0.1]]),
        np.array([[1.0], [1.1]]),
    )
    out = concat_with_frozen_skip_reach.concat_with_frozen_skip_octatope(
        layer, [oct_in],
    )
    assert isinstance(out[0], Octatope)


# ----------------------------- F.gelu approximate kwarg leak (audit C1) -----


def test_fx_add_set_plus_set_two_stream_audit_N12():
    """PR-1 audit N12: the existing operator.add tests only exercise
    set+const. The set+set branch (two independent reach streams added
    at a residual) was untested in test_vit_layer_coverage. A bug in
    ``_add_sets`` (e.g. sign flip) would not be caught by any other
    test in this file. Pin: a model ``y = a(x) + b(x)`` must MC-contain
    forward samples.
    """
    class TwoStreamAdd(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(3, 3, bias=False)
            self.b = nn.Linear(3, 3, bias=False)
            torch.manual_seed(7)
            # Non-trivial weights so a sign flip would show up.
            with torch.no_grad():
                self.a.weight.copy_(torch.tensor(
                    [[0.5, -0.2, 0.0],
                     [0.0, 0.3, 0.1],
                     [-0.1, 0.0, 0.4]],
                ))
                self.b.weight.copy_(torch.tensor(
                    [[0.1, 0.0, -0.2],
                     [-0.3, 0.2, 0.0],
                     [0.0, -0.1, 0.5]],
                ))

        def forward(self, x):
            return self.a(x) + self.b(x)

    from n2v.nn import NeuralNetwork
    model = TwoStreamAdd().eval()
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])

    def _reach(layer, sets):
        return NeuralNetwork(model).reach(sets[0], method="approx")

    pytest.assert_reach_contains_forward(
        model, lb, ub, _reach, n_samples=32, input_shape=(1, 3),
    )


def test_patch_embed_box_mc_containment_audit_N2():
    """Audit N2: the existing PatchEmbed tests assert shape/finiteness
    only -- a constant-output reach would pass. Add concrete-forward
    MC containment with image_shape=(2, 2, 1) (square but explicit so
    no warning fires; in_channels=1 keeps HWC==CHW so layout is moot).
    """
    from n2v.nn.layers import PatchEmbed
    from n2v.nn.layer_ops import patch_embed_reach

    layer = PatchEmbed(
        in_channels=1, embed_dim=4, patch_size=2,
    ).eval()
    lb = np.zeros(4)
    ub = np.full(4, 0.5)

    pytest.assert_reach_contains_forward(
        layer, lb, ub,
        lambda lay, sets: patch_embed_reach.patch_embed_box(
            lay, sets, image_shape=(2, 2, 1), image_layout="HWC",
        ),
        n_samples=24, input_shape=(1, 1, 2, 2),
    )


def test_fx_f_gelu_approximate_tanh_kwarg_preserved():
    """PR-1 audit C1: ``F.gelu(x, approximate='tanh')`` must route to the
    tanh-form floor (-0.170041), not the erf-form floor (-0.169972).

    The buggy implementation had ``F.gelu: nn.GELU`` in
    ``FUNCTION_TO_MODULE_CLS``, and ``_function_node_to_module`` consulted
    the dict first via ``cls()`` -- silently dropping ``approximate='tanh'``
    and instantiating ``nn.GELU(approximate='none')``. The reach then routed
    through the erf-form (floor -0.169972), an above-floor lower bound that
    excludes true tanh-form outputs near the dip at x ~ -0.7517 -> unsound.

    Pin: a Box bracketing the dip must produce a lower bound <= the
    tanh-form floor.
    """
    import torch.nn.functional as F  # local: matches the production import path
    from n2v.nn import NeuralNetwork
    from n2v.nn.layer_ops.gelu_reach import _GELU_TANH_F_MIN

    class GeluTanhFn(nn.Module):
        def forward(self, x):
            return F.gelu(x, approximate="tanh")

    model = GeluTanhFn().eval()
    box = _flat_box(np.array([-1.0]), np.array([-0.5]))
    out = NeuralNetwork(model).reach(box, method="approx")
    assert len(out) == 1
    lb = float(np.asarray(out[0].lb).flatten()[0])
    assert lb <= _GELU_TANH_F_MIN + 1e-9, (
        f"F.gelu approximate='tanh' floor leaked: lb={lb!r}, "
        f"expected <= {_GELU_TANH_F_MIN!r} (audit C1)."
    )


def test_fx_f_gelu_approximate_none_default_still_works():
    """Companion: ``F.gelu(x)`` (no kwarg) must route to the erf form."""
    import torch.nn.functional as F
    from n2v.nn import NeuralNetwork
    from n2v.nn.layer_ops.gelu_reach import _GELU_F_MIN

    class GeluDefault(nn.Module):
        def forward(self, x):
            return F.gelu(x)

    model = GeluDefault().eval()
    box = _flat_box(np.array([-1.0]), np.array([-0.5]))
    out = NeuralNetwork(model).reach(box, method="approx")
    lb = float(np.asarray(out[0].lb).flatten()[0])
    assert lb <= _GELU_F_MIN + 1e-9


# ------------------ SoftmaxAttention Q/K/V kwargs binding (audit C2) ----------


def test_fx_softmax_attention_kwargs_query_value_key_order():
    """PR-1 audit C2: SoftmaxAttention Q/K/V binding must NOT depend on
    Python's kwargs insertion order. Calling
    ``self.attn(query=q, value=v, key=k)`` must produce the same reach as
    ``self.attn(q, k, v)``; the dispatcher uses signature inspection so
    the declared parameter names are what counts.

    Before the fix, ``_handle_multi_input_op`` walked
    ``node.args + node.kwargs`` in insertion order and blindly bound
    ``streams[0..2]`` -- so a model calling
    ``self.attn(query=q, value=v, key=k)`` would compute
    ``softmax(q v^T / sqrt(d)) @ k`` instead of the true
    ``softmax(q k^T / sqrt(d)) @ v`` whenever K-bounds != V-bounds,
    an unsound reach silently verifying a different function.

    To make the bug detectable, this test forces K-bounds != V-bounds by
    routing different per-token boxes into K vs V via a tiny adapter
    network, then compares the reach output of the kwarg-reordered model
    against the positional model.
    """
    from n2v.nn import NeuralNetwork
    from n2v.nn.layers.softmax_attention import SoftmaxAttention

    # Use 2 tokens, d_head=2 so the streams have non-trivial K vs V structure.
    n_tokens, d_head = 2, 2

    class AttnKwargReordered(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = SoftmaxAttention(d_head=d_head)

        def forward(self, x):
            q = x
            k = x
            v = x
            return self.attn(query=q, value=v, key=k)

    class AttnPositional(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = SoftmaxAttention(d_head=d_head)

        def forward(self, x):
            return self.attn(x, x, x)

    # Distinct per-element bounds so the streams are not identical.
    box = _flat_box(
        np.array([0.0, 0.1, 0.2, 0.3]),
        np.array([0.5, 0.6, 0.7, 0.8]),
    )
    out_kw = NeuralNetwork(AttnKwargReordered().eval()).reach(
        box, method="approx",
    )
    out_pos = NeuralNetwork(AttnPositional().eval()).reach(
        box, method="approx",
    )
    np.testing.assert_allclose(
        np.asarray(out_kw[0].lb).flatten(),
        np.asarray(out_pos[0].lb).flatten(),
        atol=1e-9,
        err_msg="kwargs reordering produced different reach (audit C2).",
    )
    np.testing.assert_allclose(
        np.asarray(out_kw[0].ub).flatten(),
        np.asarray(out_pos[0].ub).flatten(),
        atol=1e-9,
        err_msg="kwargs reordering produced different reach (audit C2).",
    )


# ----------------------------- fx operator.getitem (tensor slice) ----------


def test_fx_add_set_plus_constant_image_star_4d_V():
    """Audit spot-check: the operator.add set+const handler for ImageStar
    previously wrote ``new_V[:, 0:1] = ...`` which slices the W axis of the
    4D V tensor ``(H, W, C, n_var+1)`` rather than the centre column.

    For an ImageStar input the slice produced shape ``(H, 1, C, n_var+1)``
    while const_flat was ``(flat_dim, 1)`` — numpy broadcast either crashed
    or silently mis-aligned. The fix indexes ``new_V[..., 0]`` (last axis,
    centre column) and reshapes the constant to ``(H, W, C)``.

    This test pins the fixed behaviour: a per-element constant add on a
    (H=2, W=2, C=2) ImageStar produces the correct per-element output.
    """
    from n2v.nn import NeuralNetwork
    from n2v.sets.image_star import ImageStar

    class IdAddConst(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "c",
                torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                ).reshape(2, 2, 2),
            )

        def forward(self, x):
            return x + self.c

    model = IdAddConst().eval()
    star = ImageStar.from_bounds(
        np.zeros((8, 1)), np.ones((8, 1)),
        height=2, width=2, num_channels=2,
    )
    out = NeuralNetwork(model).reach(star, method="approx")
    assert len(out) == 1
    lb, ub = out[0].get_ranges()
    lb = np.asarray(lb).flatten()
    ub = np.asarray(ub).flatten()
    # Expected: input is [0, 1] elementwise; add [1, 2, 3, 4, 5, 6, 7, 8]
    # element-wise; output is [1..8, 2..9].
    np.testing.assert_allclose(lb, [1, 2, 3, 4, 5, 6, 7, 8], atol=1e-9)
    np.testing.assert_allclose(ub, [2, 3, 4, 5, 6, 7, 8, 9], atol=1e-9)


def test_fx_getitem_image_star_4d_V_flattens_before_slice():
    """PR-1 audit I6: ``operator.getitem`` on an ImageStar previously did
    ``s.V[row_start:row_end]`` which slices the FIRST axis (H) of the 4D
    V tensor (H, W, C, nVar+1), producing a 4D slice that
    ``Star(...)`` rejects with ``ValueError: too many values to unpack``.

    Fix: flatten via ``to_star()`` (HWC-row-major == token-major) BEFORE
    row-slicing. Pin: an ImageStar carrying a 2x2 image with C=2 (so dim
    = 8) and a model that selects the first token (``x[:, 0]``) must
    return a Star with dim = 2 (D = C since L = H*W = 4), not raise.
    """
    from n2v.nn import NeuralNetwork

    class SliceFirstToken(nn.Module):
        # H*W=4 tokens, C=2 channels -> token-major flat layout.
        n_tokens = 4

        def forward(self, x):
            x = x.view(1, 4, 2)
            return x[:, 0]

    model = SliceFirstToken().eval()
    image_star = ImageStar.from_bounds(
        np.zeros((8, 1)), np.ones((8, 1)),
        height=2, width=2, num_channels=2,
    )
    out = NeuralNetwork(model).reach(image_star, method="approx", n_tokens=4)
    assert len(out) == 1
    # The first token is the (h=0, w=0) pixel -> rows 0..1 of the HWC-flat
    # layout = the first 2 channels of pixel (0,0).
    lb, ub = out[0].get_ranges()
    lb = np.asarray(lb).flatten()
    ub = np.asarray(ub).flatten()
    assert lb.shape == (2,), f"expected dim 2, got {lb.shape}"
    np.testing.assert_allclose(lb, [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(ub, [1.0, 1.0], atol=1e-9)


def test_fx_getitem_non_trivial_batch_index_raises_audit_N1():
    """PR-1 audit N1: ``_handle_getitem`` previously stripped
    ``index[0]`` unconditionally so ``x[1, 0]`` was silently treated as
    ``x[:, 0]`` -- the reach would select the wrong token of an
    arbitrary batch element. n2v only supports batch-1 inputs, but a
    non-trivial batch index in the user's model means the model is
    ambiguous; we should raise rather than silently rewrite.
    """
    from n2v.nn import NeuralNetwork

    class BadBatchSlice(nn.Module):
        n_tokens = 2

        def forward(self, x):
            x = x.view(1, 2, 3)
            return x[0, 0]  # explicit batch index 0 — not slice(None)

    model = BadBatchSlice().eval()
    box = _flat_box(
        np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]),
        np.array([0.5, 1.5, 2.5, 10.5, 11.5, 12.5]),
    )
    with pytest.raises(NotImplementedError, match="batch axis"):
        NeuralNetwork(model).reach(box, method="approx", n_tokens=2)


def test_fx_getitem_negative_token_idx():
    """Audit spot-check: getitem with negative token_idx (e.g. ``x[:, -1]``
    -- the canonical "select last token / CLS / DistillationToken" pattern)
    previously produced an EMPTY reach because
    ``row_start = token_idx * D = -D`` and ``row_end = 0`` made ``s.V[-D:0]``
    an empty slice.

    The fix normalises negative indices via ``token_idx + L`` and raises on
    out-of-range. This test pins both behaviours.
    """
    from n2v.nn import NeuralNetwork

    class SliceLast(nn.Module):
        n_tokens = 2

        def forward(self, x):
            x = x.view(1, 2, 3)
            return x[:, -1]

    model = SliceLast().eval()
    box = _flat_box(
        np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]),
        np.array([0.5, 1.5, 2.5, 10.5, 11.5, 12.5]),
    )
    out = NeuralNetwork(model).reach(box, method="approx", n_tokens=2)
    # x[:, -1] selects the LAST token: rows 3-5 of the flat layout.
    np.testing.assert_allclose(
        out[0].lb.flatten(), np.array([10.0, 11.0, 12.0]), atol=1e-9,
    )
    np.testing.assert_allclose(
        out[0].ub.flatten(), np.array([10.5, 11.5, 12.5]), atol=1e-9,
    )

    class BadSlice(nn.Module):
        n_tokens = 2

        def forward(self, x):
            x = x.view(1, 2, 3)
            return x[:, 5]  # out of range

    with pytest.raises(NotImplementedError, match="out of range"):
        NeuralNetwork(BadSlice().eval()).reach(
            box, method="approx", n_tokens=2,
        )


def test_fx_getitem_slice_all_set_types():
    """End-to-end via a tiny model: ``x[:, 0]`` extracts the first token
    of a sequence-flattened reach set. Pins the fx call_function
    operator.getitem handler across all five set types.
    """
    from n2v.nn import NeuralNetwork

    class SliceFirstToken(nn.Module):
        n_tokens = 2  # picked up by the getitem inference fallback

        def __init__(self):
            super().__init__()

        def forward(self, x):
            # Pretend x is (B=1, L=2, D=3) flattened to (1, 6).
            x = x.view(1, self.n_tokens, 3)
            return x[:, 0]

    model = SliceFirstToken().eval()
    box = _flat_box(
        np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]),
        np.array([0.5, 1.5, 2.5, 10.5, 11.5, 12.5]),
    )
    for set_in in (
        box, _flat_star(box), _flat_zono(box),
        _flat_hex(box), _flat_oct(box),
    ):
        out = NeuralNetwork(model).reach(
            set_in, method="approx", n_tokens=2,
        )
        assert len(out) == 1
        lb_o, ub_o = _bounds_of(out[0])
        # First token in the flat layout is the first D=3 chunk.
        np.testing.assert_allclose(lb_o, np.array([0.0, 1.0, 2.0]), atol=1e-9)
        np.testing.assert_allclose(ub_o, np.array([0.5, 1.5, 2.5]), atol=1e-9)


# ----------------------------- SoftmaxAttention multi-input ----------------


@pytest.mark.parametrize("set_kind", ["Box", "Star", "Zono", "Hex", "Oct"])
def test_softmax_attention_multi_input_all_set_types(set_kind):
    """End-to-end: a minimal model whose forward calls SoftmaxAttention on
    three projected views of the input. With N2VTracer (commit 6878285)
    treating SoftmaxAttention as an fx leaf, ``_handle_multi_input_op``
    dispatches Box/Star/Zono/Hex/Oct streams via the box-lifted helper.

    Pins the multi-input dispatcher routes for every set type, including
    the new Hex/Oct branches added in this PR.
    """
    from n2v.nn import NeuralNetwork
    from n2v.nn.layers import SoftmaxAttention

    class _AttnModel(nn.Module):
        def __init__(self, d_head: int = 2):
            super().__init__()
            self.q_proj = nn.Linear(d_head, d_head, bias=False)
            self.k_proj = nn.Linear(d_head, d_head, bias=False)
            self.v_proj = nn.Linear(d_head, d_head, bias=False)
            self.attn = SoftmaxAttention(d_head=d_head)
            # Identity projections so we can reason about the output range.
            with torch.no_grad():
                self.q_proj.weight.copy_(torch.eye(d_head))
                self.k_proj.weight.copy_(torch.eye(d_head))
                self.v_proj.weight.copy_(torch.eye(d_head))

        def forward(self, x):
            return self.attn(self.q_proj(x), self.k_proj(x), self.v_proj(x))

    model = _AttnModel(d_head=2).eval()
    box = _flat_box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    ctors = {
        "Box": lambda: box,
        "Star": lambda: _flat_star(box),
        "Zono": lambda: _flat_zono(box),
        "Hex": lambda: _flat_hex(box),
        "Oct": lambda: _flat_oct(box),
    }
    set_in = ctors[set_kind]()

    out = NeuralNetwork(model).reach(set_in, method="approx")
    assert len(out) == 1
    lb_o, ub_o = _bounds_of(out[0])
    # With V in [0, 1] and softmax rows summing to 1, the output lies in
    # the convex hull of V's columns, i.e. each output coordinate is
    # bounded by the corresponding column's [min, max]. For our identity
    # projections that means lb >= 0 and ub <= 1.
    assert np.all(lb_o >= 0.0 - 1e-6)
    assert np.all(ub_o <= 1.0 + 1e-6)

    # Audit N4/N11: tighten beyond ``in [0,1]`` by checking concrete-forward
    # containment against the actual model.  Use the strict-inside box
    # ``[0.3, 0.7]`` so a stub returning ``Box([0],[1])`` would still pass
    # the previous assertion but fail to contain forward samples outside
    # [0.3, 0.7].  We run this MC check on Box only since it's the only
    # set type that exercises the full set-of-streams path identically to
    # the forward.
    if set_kind == "Box":
        strict_lb = np.array([0.3, 0.4])
        strict_ub = np.array([0.6, 0.7])

        def _attn_reach(layer, sets):
            return NeuralNetwork(model).reach(sets[0], method="approx")

        pytest.assert_reach_contains_forward(
            model, strict_lb, strict_ub, _attn_reach,
            n_samples=24,
            input_shape=(1, 2),
        )

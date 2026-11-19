"""Soundness cells of nn.Upsample in nearest mode and of the ONNX Resize in
nearest mode with the rounding rule PyTorch exports (asymmetric, floor).

The other modes, the other rounding rules and the other ONNX spellings are in
``tests/soundness/test_soundness_resize.py``.
"""

import numpy as np
import pytest
import torch.nn as nn


# =========================================================================== #
# Soundness cells                                                               #
#                                                                               #
# Nearest-neighbour upsampling by integer factors (sh, sw) copies input pixel  #
# (c, i // sh, j // sw) to output pixel (c, i, j). Cells: nn.Upsample          #
# (nearest, equal and unequal integer factors) and ONNX Resize in nearest mode #
# with coordinate_transformation_mode 'asymmetric' and nearest_mode 'floor',   #
# given by scales and by sizes, with the roi input named (the form without it  #
# is marked in tests/contracts/test_reach_coverage.py). Other Resize modes and #
# rounding rules belong to tests/soundness/test_soundness_resize.py. Oracles:  #
# torch in float64, or onnxruntime for an ONNX Resize, on member points        #
# (corners included) with membership decided by one LP on the returned set's  #
# own rows (tests/_oracles/membership.py); and the exact image in rationals    #
# (tests/_oracles/rational.py) from the definition above, which shares no code #
# with n2v. Layout: a tensor is (C, H, W); an image set stores (H, W, C);      #
# every comparison converts explicitly (``out_chw``, ``hwc_index``). Every      #
# image has two channels and height different from width. The exact-image      #
# helpers live in test_soundness_conv2d.py.                                     #
# =========================================================================== #
import torch.nn.functional as F  # noqa: E402
from onnx import helper  # noqa: E402

from n2v.nn.layer_ops.dispatcher import reach_layer  # noqa: E402
from n2v.nn.layer_ops.upsample_reach import upsample_box  # noqa: E402
from n2v.utils.model_loader import load_onnx  # noqa: E402
from tests.cell_support import (  # noqa: E402
    assert_covered, constrained_image_star, flat_box, flat_set, image_set,
    pixel_box, reach_model, reach_one_layer, replay_onnx, replay_torch,
    samples, save_onnx)
from tests.known_bugs import known_bug  # noqa: E402
from tests.soundness.test_soundness_conv2d import (  # noqa: E402
    _assert_exact_image, _chain, _distinct, _few_var_inputs, _inputs,
    _out_chw, _piece, _two_straddlers)

# id: ((C, H, W) of the input, scale_factor of nn.Upsample).
_UPSAMPLE = {
    "factor-2": ((2, 3, 5), 2),
    "factors-2-3": ((2, 3, 5), (2, 3)),
    "factors-3-2": ((2, 4, 3), (3, 2)),
}


def _make_upsample(cfg):
    chw, factor = _UPSAMPLE[cfg]
    return nn.Upsample(scale_factor=factor, mode="nearest"), chw


def _factors(cfg):
    factor = _UPSAMPLE[cfg][1]
    return (factor, factor) if isinstance(factor, int) else factor


def _exact_upsample(chw, sh, sw):
    """``f(x) -> (y, (C, H sh, W sw))``: nearest upsampling by its definition
    on a flat (C, H, W) list of Fractions."""
    C, H, W = chw
    Ho, Wo = H * sh, W * sw

    def f(x):
        return ([x[(c * H + i // sh) * W + j // sw]
                 for c in range(C) for i in range(Ho) for j in range(Wo)],
                (C, Ho, Wo))
    return f


def _upsample_cells():
    out = []
    for cfg in _UPSAMPLE:
        variants = [("ImageStar", "exact", "split"),
                    ("ImageZono", "approx", "split"),
                    ("ImageStar", "approx", "row")]
        if cfg == "factors-2-3":
            variants += [("ImageStar", "exact", "wide"),
                         ("ImageStar", "approx", "wide"),
                         ("ImageZono", "approx", "wide")]
        for v in variants:
            out.append(pytest.param(cfg, *v, id=f"{cfg}-{v[0]}-{v[1]}-{v[2]}"))
    return out


@pytest.mark.parametrize("cfg,kind,method,variant", _upsample_cells())
def test_upsample_layer_holds_every_torch_output(cfg, kind, method, variant):
    """nn.Upsample (nearest) through ``reach_layer`` on three sets at once, on
    an ImageStar with a binding row, or (``wide``) on the pixel box times 8,
    whose entries and intervals reach far beyond the predicate box [-1, 1];
    against torch in float64, membership by LP on the rows. Catches a result
    that keeps only the first or only the last of several input sets, the two
    factors exchanged (for the centre, the generators or both), a tiled
    instead of a repeated copy, generators reshaped in another order than the
    centre, and a result that reads a predicate variable in place of the
    copied entry."""
    layer, chw = _make_upsample(cfg)
    if variant == "wide":
        lo, hi = (8 * v for v in pixel_box(*chw))
        sets, points = [image_set(kind, lo, hi)], samples(lo, hi)
    else:
        sets, points = _inputs(kind, variant, chw)
    what = f"{layer} on {kind} ({variant}, {method})"
    result = reach_one_layer(layer, sets, method, what)
    assert_covered(result, replay_torch(layer, points, (1,) + chw), what,
                   out_chw=_out_chw(layer, chw))


@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageStar", "approx"),
                                         ("ImageZono", "approx")])
@pytest.mark.parametrize("cfg", list(_UPSAMPLE))
def test_upsample_layer_is_the_exact_image_in_rationals(cfg, kind, method):
    """Two input sets at once on three predicate variables each (the
    ImageStar cut by a binding row), centre and generators dense over every
    pixel with 30 significant bits (not representable in float32), the
    second star on an asymmetric predicate box. The exact image is the
    definition of nearest upsampling in rationals; every vertex of it must
    be in one returned set and, for the exact method, every vertex of a
    returned set must have a preimage. Catches a generator copied to the
    wrong block even when sampled points are inside, a lost input set,
    predicate bounds not passed on as they are, and a copy made in
    float32."""
    layer, chw = _make_upsample(cfg)
    f = _exact_upsample(chw, *_factors(cfg))
    sets, pieces = [], []
    # 30 significant bits: a float32 round trip of the copied entries
    # moves them, so the exact check sees it
    for star, zono, c, G, star_rows, zono_rows in _few_var_inputs(
            chw, seed=4, bits=30):
        sets.append(star if kind == "ImageStar" else zono)
        piece, out_chw = _piece(f, c, G, star_rows if kind == "ImageStar"
                                else zono_rows)
        pieces.append(piece)
    what = f"{layer} on {kind} ({method}), exact image"
    result = reach_one_layer(layer, sets, method, what)
    _assert_exact_image(pieces, result, what, exact=(method == "exact"),
                        out_chw=out_chw)


@pytest.mark.parametrize("cfg,kind,method,variant", [
    ("factors-2-3", "ImageStar", "exact", "split"),
    ("factors-3-2", "ImageZono", "approx", "split"),
    ("factors-2-3", "ImageStar", "approx", "row"),
])
def test_upsample_inside_a_model_holds_every_torch_output(cfg, kind, method,
                                                          variant):
    """nn.Upsample between a channel-mixing 1x1 convolution and Flatten >
    Linear, through ``reach_pytorch_model``. Catches a replication whose rows
    or layout only the downstream affine map and the Flatten conversion
    expose."""
    op, chw = _make_upsample(cfg)
    model = _chain(op, chw)
    sets, points = _inputs(kind, variant, chw)
    what = f"conv > {op} > flatten > linear on {kind} ({method})"
    result = reach_model(model, sets, method, what)
    assert_covered(result, replay_torch(model, points, (1,) + chw), what)


def test_upsample_after_an_exact_relu_split_holds_every_torch_output():
    """ReLU > Upsample(2, 3) > Flatten > Linear with the exact method on a
    box with two pixels that straddle zero: the upsample receives the four
    sets of the exact split together. Catches an upsample that returns the
    image of only the first or only the last of several input sets."""
    op, chw = _make_upsample("factors-2-3")
    model = _chain(op, chw, head=nn.ReLU())
    lo, hi = _two_straddlers(chw)
    what = f"relu > {op} > flatten > linear, exact"
    result = reach_model(model, image_set("ImageStar", lo, hi), "exact", what)
    assert_covered(result, replay_torch(model, samples(lo, hi), (1,) + chw),
                   what)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_upsample_after_an_approx_relu_holds_every_torch_output(kind):
    """ReLU > Upsample(2, 3) > Flatten > Linear with the approx method on a
    box with two pixels that straddle zero: the relaxation of the two
    crossing neurons adds two predicate variables to the ImageStar (or two
    generators to the ImageZono), so the upsample receives a set with more
    variables than entries, the shape no box set has. Catches a kernel that
    takes the generator count from the entry count, or drops the variables
    a relaxation added."""
    op, chw = _make_upsample("factors-2-3")
    model = _chain(op, chw, head=nn.ReLU())
    lo, hi = _two_straddlers(chw)
    what = f"relu > {op} > flatten > linear on {kind}, approx"
    result = reach_model(model, image_set(kind, lo, hi), "approx", what)
    assert_covered(result, replay_torch(model, samples(lo, hi), (1,) + chw),
                   what)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("layer_id,build", [pytest.param(i, b, id=i) for i, b in (
    ("bilinear", lambda: nn.Upsample(scale_factor=2, mode="bilinear")),
    ("bicubic", lambda: nn.Upsample(scale_factor=(2, 3), mode="bicubic")),
    ("size", lambda: nn.Upsample(size=(6, 10), mode="nearest")),
    ("size-not-multiple", lambda: nn.Upsample(size=(7, 9), mode="nearest")),
    ("factor-1.5", lambda: nn.Upsample(scale_factor=1.5, mode="nearest")),
    ("factors-2-1.5", lambda: nn.Upsample(scale_factor=(2, 1.5),
                                          mode="nearest")),
)])
def test_upsample_that_is_not_an_integer_copy_refuses_or_is_sound(
        layer_id, build, kind):
    """nn.Upsample in an interpolating mode (bilinear by 2, bicubic by
    (2, 3)), given an output size, or with a non-integer factor: the reach
    must refuse (NotImplementedError) or answer with sets that hold every
    torch output. Catches a mode check that admits an interpolating mode as
    nearest (bilinear admitted, or only the modes named linear refused), a
    refusal of the mode check answered through the forward probe instead,
    a size or a non-integer factor truncated to an integer factor, and one of
    two factors not checked."""
    layer = build()
    chw = (2, 3, 5)
    lo, hi = pixel_box(*chw, seed=1)
    what = f"{layer} on {kind}"
    try:
        result = reach_one_layer(layer, [image_set(kind, lo, hi)], "approx",
                                 what)
    except NotImplementedError:
        return
    assert_covered(result, replay_torch(layer, samples(lo, hi), (1,) + chw),
                   what, out_chw=_out_chw(layer, chw))


# --------------------------------------------------------------------------- #
# ONNX Resize, nearest, asymmetric, floor: by scales and by sizes              #
# --------------------------------------------------------------------------- #
def _onnx_resize_chain(tmp_path, name, chw, scales=None, sizes=None,
                       opset=13):
    """X -> Conv 1x1 -> Resize -> Conv 1x1 -> Y with dyadic float32
    weights; the Resize names its roi input (empty) and gives ``scales`` or,
    with empty scales, ``sizes``. Returns ``(path, out_chw)``; the output
    shape is checked against the one onnxruntime computes."""
    rng = np.random.default_rng(10)
    C = chw[0]
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    inits = {"a_w": f32(_distinct(rng, (C, C, 1, 1))),
             "a_b": f32(_distinct(rng, (C,))),
             "z_w": f32(_distinct(rng, (2, C, 1, 1))),
             "z_b": f32(_distinct(rng, (2,))),
             "roi": f32([]), "scales": f32([] if scales is None else scales)}
    inputs = ["T0", "roi", "scales"]
    if sizes is not None:
        inits["sizes"] = np.asarray(sizes, dtype=np.int64)
        inputs.append("sizes")
    nodes = [helper.make_node("Conv", ["X", "a_w", "a_b"], ["T0"]),
             helper.make_node("Resize", inputs, ["T1"], mode="nearest",
                              coordinate_transformation_mode="asymmetric",
                              nearest_mode="floor"),
             helper.make_node("Conv", ["T1", "z_w", "z_b"], ["Y"])]
    if sizes is not None:
        out_hw = tuple(sizes[2:])
    else:
        out_hw = (int(chw[1] * scales[2]), int(chw[2] * scales[3]))
    path = save_onnx(tmp_path, nodes, [1, *chw], [1, 2, *out_hw], inits,
                     name=name, opset=opset)
    probe = replay_onnx(path, [np.zeros(chw)], [1, *chw])[0]
    if probe.size != 2 * out_hw[0] * out_hw[1]:
        pytest.fail(f"onnxruntime returned {probe.size} entries, expected "
                    f"(2, {out_hw[0]}, {out_hw[1]})")
    return path, (2,) + out_hw


# id: ((C, H, W), scales, sizes, opset). Every factor pair differs between
# height and width. On (2, 2, 4) the sizes 8 and 4 divide by both input
# extents, so reading them crosswise still gives integer factors.
_RESIZE = {
    "scales-2-3": ((2, 3, 5), [1, 1, 2, 3], None, 13),
    "scales-3-2-opset11": ((2, 4, 3), [1, 1, 3, 2], None, 11),
    "sizes-6-15": ((2, 3, 5), None, [1, 2, 6, 15], 13),
    "sizes-8-4": ((2, 2, 4), None, [1, 2, 8, 4], 13),
}


@pytest.mark.parametrize("kind,method", [("ImageStar", "exact"),
                                         ("ImageStar", "approx"),
                                         ("ImageZono", "approx")])
@pytest.mark.parametrize("cfg", list(_RESIZE))
def test_onnx_resize_nearest_holds_every_onnxruntime_output(tmp_path, cfg,
                                                           kind, method):
    """An ONNX Resize (nearest, asymmetric, floor) between two 1x1 Convs,
    by scales or by sizes with different height and width factors, loaded
    with ``load_onnx`` and replayed on onnxruntime. Catches the two scales
    read in the wrong order, the output height and width of ``sizes`` read
    in the wrong order or divided by the wrong input extents, and a
    replication of the generators along the wrong axes."""
    chw, scales, sizes, opset = _RESIZE[cfg]
    path, out_chw = _onnx_resize_chain(tmp_path, cfg, chw, scales, sizes,
                                       opset)
    lo, hi = pixel_box(*chw, seed=11)
    if kind == "ImageStar" and method == "approx":
        start, points = constrained_image_star(lo, hi, seed=11)
    else:
        start, points = image_set(kind, lo, hi), samples(lo, hi)
    what = f"ONNX Resize {cfg} on {kind} ({method})"
    result = reach_model(load_onnx(path), start, method, what)
    assert_covered(result, replay_onnx(path, points, [1, *chw]), what,
                   out_chw=out_chw)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("scales,sizes", [
    pytest.param([1, 1, 2.5, 2], None, id="scales-2.5"),
    pytest.param([1, 1, 2, 1.5], None, id="scales-1.5"),
    pytest.param(None, [1, 2, 7, 10], id="sizes-not-multiple"),
])
def test_onnx_resize_nearest_that_is_not_an_integer_copy_refuses_or_is_sound(
        tmp_path, scales, sizes, kind):
    """An ONNX Resize (nearest, asymmetric, floor) with a non-integer scale
    of the height (2.5, which rounds and truncates to 2) or of the width
    (1.5), or sizes whose height is not a multiple of the input: the reach
    must refuse (NotImplementedError) or answer with sets that hold every
    onnxruntime output. Catches an integer check that compares the rounded
    scale with the truncated one or reads one axis only, and a size reached
    as a smaller copy. (A scale of the channel axis is in
    ``test_resize_of_the_channel_or_batch_axis_is_refused_or_right``.)"""
    chw = (2, 3, 5)
    rng = np.random.default_rng(12)
    C2 = 2
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    inits = {"a_w": f32(_distinct(rng, (2, 2, 1, 1))),
             "a_b": f32(_distinct(rng, (2,))),
             "z_w": f32(_distinct(rng, (2, C2, 1, 1))),
             "z_b": f32(_distinct(rng, (2,))),
             "roi": f32([]), "scales": f32([] if scales is None else scales)}
    inputs = ["T0", "roi", "scales"]
    if sizes is not None:
        inits["sizes"] = np.asarray(sizes, dtype=np.int64)
        inputs.append("sizes")
    nodes = [helper.make_node("Conv", ["X", "a_w", "a_b"], ["T0"]),
             helper.make_node("Resize", inputs, ["T1"], mode="nearest",
                              coordinate_transformation_mode="asymmetric",
                              nearest_mode="floor"),
             helper.make_node("Conv", ["T1", "z_w", "z_b"], ["Y"])]
    # the ONNX rule: an output extent is floor(input extent * scale)
    if sizes is not None:
        out_hw = tuple(sizes[2:])
    else:
        out_hw = (int(np.floor(3 * scales[2])), int(np.floor(5 * scales[3])))
    path = save_onnx(tmp_path, nodes, [1, *chw], [1, 2, *out_hw], inits,
                     name="rsx")
    lo, hi = pixel_box(*chw, seed=13)
    outputs = replay_onnx(path, samples(lo, hi), [1, *chw])
    what = f"ONNX Resize scales {scales} sizes {sizes} on {kind}"
    try:
        result = reach_model(load_onnx(path), image_set(kind, lo, hi),
                             "approx", what)
    except NotImplementedError:
        return
    if outputs[0].size != 2 * out_hw[0] * out_hw[1]:
        pytest.fail(f"onnxruntime returned {outputs[0].size} entries, "
                    f"expected (2, {out_hw[0]}, {out_hw[1]})")
    assert_covered(result, outputs, what, out_chw=(2,) + out_hw)


# --------------------------------------------------------------------------- #
# Refusals                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["Star", "Zono", "Box", "Hexatope",
                                  "Octatope"])
def test_upsample_on_a_flat_set_is_refused(kind):
    """A flat set carries no height, width or channel count, so no rule
    upsamples it: ``reach_layer`` must refuse (NotImplementedError)."""
    lo, hi = flat_box(30)
    with pytest.raises(NotImplementedError):
        reach_layer(nn.Upsample(scale_factor=(2, 3), mode="nearest"),
                    [flat_set(kind, lo, hi)],
                    "exact" if kind == "Star" else "approx")


def test_upsample_box_kernel_is_refused():
    """``upsample_box`` (not routed by the dispatcher) has no image shape to
    replicate: a direct call must refuse."""
    from n2v.sets import Box
    lo, hi = flat_box(30)
    with pytest.raises(NotImplementedError):
        upsample_box(nn.Upsample(scale_factor=2, mode="nearest"),
                     [Box(lo, hi)])


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("mode", ["nearest"])
def test_onnx_resize_module_without_graph_scales_is_refused(mode, kind):
    """A bare OnnxResize handed to ``reach_layer`` without the scales or
    sizes of its graph cannot know its factors: the reach must refuse, never
    assume a factor. In nearest mode the refusal comes from the forward
    probe (``_detect_scale_factors_from_forward``); in linear or cubic mode
    the mode check of ``_get_scale_factors`` raises first, and the probe
    would refuse without it, so the mode check has no observable effect on
    a legal input (its refusal on the model route is held by the keyword
    cells of ``test_linear_and_cubic_resize_hold_every_onnxruntime_output``
    in test_soundness_resize.py); with graph scales the mode is not read at
    all (RESIZE-1). Catches a failed probe answered with an assumed
    factor."""
    from onnx2torch.node_converters.resize import OnnxResize
    lo, hi = pixel_box(2, 3, 5)
    with pytest.raises(NotImplementedError):
        reach_layer(OnnxResize(mode=mode), [image_set(kind, lo, hi)],
                    "approx")


@pytest.mark.parametrize("first,second", [
    pytest.param("ImageStar", "ImageZono", marks=known_bug(
        23, "a reach_layer list of an ImageStar and a set of another family "
            "crashes in upsample_star with TypeError where a refusal is due "
            "(REACH-6: the dispatcher picks the kernel by the first set and "
            "checks no other)", raises=TypeError)),
    ("ImageZono", "ImageStar")])
def test_upsample_on_a_list_of_mixed_set_types_refuses_or_is_sound(first,
                                                                   second):
    """An ImageStar and an ImageZono in one call: the kernel routed by the
    first set must refuse the other (NotImplementedError or ValueError) or
    answer with sets that hold the outputs of both. ``upsample_zono`` refuses
    an ImageStar today (the passing order); ``upsample_star`` raises
    ``TypeError`` for an ImageZono (marked, REACH-6). Catches a kernel that
    skips a set of the wrong type, or returns after the first set."""
    layer, chw = _make_upsample("factors-2-3")
    lo, hi = pixel_box(*chw)
    try:
        result = reach_layer(layer, [image_set(first, lo, hi),
                                     image_set(second, lo + 4, hi + 4)],
                             "approx")
    except (ValueError, NotImplementedError):
        return
    points = samples(lo, hi, 32) + samples(lo + 4, hi + 4, 32)
    assert_covered(result, replay_torch(layer, points, (1,) + chw),
                   f"{layer} on a mixed list", out_chw=_out_chw(layer, chw))


class _Interpolate(nn.Module):
    """A model that upsamples with the functional call by (2, 3), nearest
    unless another mode is given."""

    def __init__(self, mode="nearest"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=(2, 3), mode=self.mode)


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
def test_functional_interpolate_in_a_model_refuses_or_is_sound(kind):
    """``F.interpolate(x, scale_factor=(2, 3), mode="nearest")`` in a torch
    model's forward: the trace holds a function node, not an nn.Upsample, and
    the walker has no rule for it, so it refuses today
    (``NotImplementedError``). The promise: refuse, or answer with sets that
    hold every torch output. Catches a rule for the functional form that
    reads the factors in the wrong order or drops the mode."""
    chw = (2, 3, 5)
    lo, hi = pixel_box(*chw, seed=2)
    model = _Interpolate()
    what = f"F.interpolate nearest (2, 3) on {kind}"
    try:
        result = reach_model(model, image_set(kind, lo, hi), "approx", what)
    except NotImplementedError:
        return
    assert_covered(result, replay_torch(model, samples(lo, hi), (1,) + chw),
                   what, out_chw=(2, 6, 15))


@pytest.mark.parametrize("kind", ["ImageStar", "ImageZono"])
@pytest.mark.parametrize("layer_id", ["Upsample-bilinear", "interpolate-bilinear"])
def test_interpolating_upsample_in_a_torch_model_refuses_or_is_sound(layer_id,
                                                                     kind):
    """A torch model that upsamples by (2, 3) in bilinear mode, with the
    module ``nn.Upsample`` (a ``call_module`` node of the trace, reached
    through the dispatcher with no graph scales) or with the functional
    ``F.interpolate`` (a ``call_function`` node, which the walker has no rule
    for), through ``reach_pytorch_model``; the model is the operation alone,
    so every output it computes is checked (a head that mixes the outputs
    hides most of a wrong answer). Refused today with NotImplementedError;
    the promise: refuse, or answer with sets that hold every torch output.
    Catches a walker that hands an nn.Upsample its own factors as graph
    scales, which the reach rule uses before the mode check, and a rule for
    the functional form that drops the mode, so that bilinear is reached as
    nearest (nn.Upsample defaults to nearest)."""
    chw = (2, 3, 5)
    lo, hi = pixel_box(*chw, seed=1)
    if layer_id == "Upsample-bilinear":
        model = nn.Sequential(nn.Upsample(scale_factor=(2, 3), mode="bilinear"))
    else:
        model = _Interpolate("bilinear")
    what = f"{layer_id} (2, 3) in a torch model on {kind}"
    try:
        result = reach_model(model, image_set(kind, lo, hi), "approx", what)
    except NotImplementedError:
        return
    assert_covered(result, replay_torch(model, samples(lo, hi), (1,) + chw),
                   what, out_chw=(2, 6, 15))

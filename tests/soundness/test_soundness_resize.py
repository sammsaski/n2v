"""Soundness cells of the ONNX ``Resize`` operator and the old ``Upsample``.

n2v reaches every Resize it answers as block replication: each input pixel is
copied into a block of scale_h by scale_w output pixels
(``n2v/nn/layer_ops/upsample_reach.py``). That is the true map only for some
files. A cell here builds a small ONNX file, loads it with ``load_onnx``,
pushes an image set over a (C, H, W) = (2, 3, 4) box with its own interval per
pixel through it, and checks every true output of 128 member points (corners
included) against the set's own rows with the LP oracle. The promise of every
cell: the call raises ``NotImplementedError`` (at load or at reach), or the
result has the true entry count and holds every true output. The files the
library supports today and must keep answering (nearest mode with a rounding
rule whose formula is a block copy at every integer scale, the pairs DESIGN
section 17 keeps) do not accept the refusal.

The reference
-------------
Resize is linear in its input. Its true map is the formula of the ONNX
specification (the reference implementation ``onnx/reference/ops/op_resize.py``)
worked out here in exact fractions, one axis at a time (``_axis_rows``). Each
cell replays its member points on onnxruntime first and checks that
onnxruntime agrees with the formula: exactly in nearest mode, where every
output is a copy of a dyadic input, and to ``2**-16`` relative in linear and
cubic mode, where onnxruntime rounds its float32 weights. A wrong formula
therefore fails the cell's setup (``pytest.fail``, never absorbed by a
marker), not its verdict. The membership verdict is taken against the
formula's float64 outputs, with no slack beyond ``LP_TOL``. The cells whose
file the formula does not model (a resize of the batch or channel axis, the
opsets the converter refuses, non-integer scales other than half_pixel and
asymmetric) are nearest-mode copies or dyadic weights, where onnxruntime's
float32 output is exact, and use it directly. ``test_formula_*`` check the
formula itself against onnxruntime on unit and label images. Under the exact
method a block-copy result is also checked in the other direction, by linear
programs over its own rows: every point of it has a preimage in the input set
(``_assert_inside_image``).

Which cells are marked
----------------------
The markers are decided from the formula, not from what the library returns.
A file the library answers today (the roi input given by name, as an
initializer or a Constant node, which is the only spelling at opset 11, or the
old Upsample operator, which the loader upgrades to such a Resize) is marked
when its exact map differs from block replication in height or in width
(``_is_block``). ``test_the_marked_nearest_cells_are_the_x_cells_of_the_report``
checks that the rule gives exactly the X cells of the table in
``verify/resize.md`` section 7. A nearest defect is marked once per code path
of the reach rule (``scales`` or ``sizes``, initializer or Constant node,
ImageStar or ImageZono), not once per rounding rule or scale: n2v reads
neither attribute, so every rule takes the same path. RESIZE-1 likewise: the
mode, the coordinate rule and the cubic attributes are not read on the path
of graph scales, so one linear file per spelling and set type is marked,
and one cubic file per set type. Files whose roi input
is an empty name (legal from opset 13; what current exporters write) are
refused today and are not marked: they guard against a fix that starts
reading the keyword scales or sizes without checking the mode and the
rounding rule (DESIGN section 17, row "Nearest-upsample exports at opset 13
and later refused").

Registry rows (``test-audit/DEFECTS.md``): RESIZE-1 (linear and cubic, also
through the zonotope bound pre-pass), RESIZE-2 (the
opset 7 and 9 Upsample in linear mode), RESIZE-3 (nearest mode with the other
rounding rules), RESIZE-4 (``sizes`` that change the channel or the batch
entry), RESIZE-5 (``tf_crop_and_resize`` with a real crop), RESIZE-7 (the
converted module differs from onnxruntime and no load-time check exists).
"""

import math
from fractions import Fraction
from typing import NamedTuple, Optional

import numpy as np
import pytest
import torch
from onnx import helper, numpy_helper
from scipy.optimize import linprog

from n2v.nn.neural_network import NeuralNetwork
from n2v.sets import HalfSpace
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from n2v.utils.verify_specification import verify_specification
from tests._oracles.membership import LP_TOL, hwc, rows
from tests.cell_support import (IMAGE_KINDS, assert_covered,
                                constrained_image_star, flat_set, hwc_index,
                                image_set, pixel_box, reach_model,
                                replay_onnx, samples, save_onnx)
from tests.known_bugs import known_bug

# onnx2torch's Resize converter warns for every attribute value its torch
# module does not reproduce (third_party/onnx2torch/.../resize.py). The cells
# decide those files by membership, so the warnings are ignored by message.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Pytorch's nearest neighbor interpolat:UserWarning"),
    pytest.mark.filterwarnings(
        "ignore:For linear and cubic interpolation in:UserWarning"),
    pytest.mark.filterwarnings(
        "ignore:With a cubic coefficient value other than:UserWarning"),
    pytest.mark.filterwarnings(
        "ignore:With a exclude outside value other than:UserWarning"),
]

CHW = (2, 3, 4)
CTMS = ("half_pixel", "asymmetric", "align_corners", "pytorch_half_pixel")
NEAREST_MODES = ("round_prefer_floor", "round_prefer_ceil", "floor", "ceil")
# The (coordinate_transformation_mode, nearest_mode) pairs whose formula is a
# block copy at every integer scale; DESIGN section 17 keeps them supported
# (test_the_block_pairs_are_block_copies_at_every_integer_scale).
BLOCK_PAIRS = (("asymmetric", "floor"), ("half_pixel", "round_prefer_floor"),
               ("half_pixel", "round_prefer_ceil"),
               ("pytorch_half_pixel", "round_prefer_floor"),
               ("pytorch_half_pixel", "round_prefer_ceil"))
# onnxruntime against the formula in linear and cubic mode: its float32
# weights and coordinates (measured up to 3.8e-6 on the cells here).
ORT_AGREEMENT = 2.0 ** -16
HALF = Fraction(1, 2)

_DEFECT = {
    "RESIZE-1": "ONNX Resize in linear or cubic mode is reached as nearest "
                "neighbour block replication, because the graph's scales are "
                "used before the only mode check: a set of the right size "
                "that leaves out the true outputs (false UNSAT)",
    "RESIZE-2": "the opset 7 and 9 Upsample operator in linear mode is "
                "upgraded by the loader to a Resize with positional scales "
                "and reached as block replication: the true outputs escape",
    "RESIZE-3": "ONNX Resize in nearest mode with a rounding rule whose index "
                "map is not a block copy is reached as a block copy: a set of "
                "the right size that leaves out the true outputs (false UNSAT)",
    "RESIZE-4": "a nearest Resize by sizes that changes the channel or the "
                "batch entry is answered with the input's channel and batch "
                "count: a result with the wrong number of entries",
    "RESIZE-5": "tf_crop_and_resize is answered with the region of interest "
                "ignored: with a real crop the true outputs escape",
}


def _bug(defect):
    return known_bug(7, _DEFECT[defect])


# --------------------------------------------------------------------------- #
# The ONNX specification of Resize, in exact fractions                         #
# --------------------------------------------------------------------------- #
def _f32(value):
    """The float32 number the ONNX file stores, as an exact fraction."""
    return Fraction(float(np.float32(value)))


def _x_original(ctm, x, scale, n_in, roi):
    """The input coordinate of output index ``x`` along one axis (attribute
    ``coordinate_transformation_mode``), as the reference implementation
    computes it: ``length`` is the scale times the input length."""
    length = scale * n_in
    if ctm == "half_pixel":
        return (x + HALF) / scale - HALF
    if ctm == "pytorch_half_pixel":
        return (x + HALF) / scale - HALF if length > 1 else -HALF
    if ctm == "asymmetric":
        return Fraction(x) / scale
    if ctm == "align_corners":
        return Fraction(x * (n_in - 1)) / (length - 1) if length > 1 else Fraction(0)
    if ctm == "tf_crop_and_resize":
        start, end = roi
        if length > 1:
            value = x * (end - start) * (n_in - 1) / (length - 1)
        else:
            value = (end - start) * (n_in - 1) / 2
        return value + start * (n_in - 1)
    raise ValueError(f"coordinate_transformation_mode {ctm!r} is not modelled")


def _nearest_coeffs(ratio, nearest_mode):
    if ratio == 1:                       # an integer coordinate reads itself
        return [Fraction(0), Fraction(1)]
    if nearest_mode == "round_prefer_floor":
        return [Fraction(ratio <= HALF), Fraction(ratio > HALF)]
    if nearest_mode == "round_prefer_ceil":
        return [Fraction(ratio < HALF), Fraction(ratio >= HALF)]
    if nearest_mode == "floor":
        return [Fraction(1), Fraction(0)]
    if nearest_mode == "ceil":
        return [Fraction(0), Fraction(1)]
    raise ValueError(f"nearest_mode {nearest_mode!r}")


def _cubic_coeffs(r, a):
    return [((a * (r + 1) - 5 * a) * (r + 1) + 8 * a) * (r + 1) - 4 * a,
            ((a + 2) * r - (a + 3)) * r * r + 1,
            ((a + 2) * (1 - r) - (a + 3)) * (1 - r) * (1 - r) + 1,
            ((a * ((1 - r) + 1) - 5 * a) * ((1 - r) + 1) + 8 * a)
            * ((1 - r) + 1) - 4 * a]


def _axis_rows(mode, ctm, nearest_mode, scale, n_in, n_out,
               roi=(Fraction(0), Fraction(1)), cubic_a=Fraction(-3, 4),
               exclude_outside=False):
    """One dict ``{input index: weight}`` per output index of one axis, the
    weights exact fractions: the 1-D interpolation of the reference
    implementation (``_interpolate_1d_with_x``). The coefficients of the mode
    act on the ``n`` input positions nearest to the coordinate (ties to the
    smaller index) in the input padded by ``ceil(n / 2)`` copies of its edge,
    so a position outside the image reads the edge pixel; with
    ``exclude_outside`` the coefficients of such positions are dropped and the
    rest renormalised. A ``tf_crop_and_resize`` coordinate outside the image
    takes the extrapolation value, which this oracle does not model."""
    out = []
    for x in range(n_out):
        xo = _x_original(ctm, x, scale, n_in, roi)
        if ctm == "tf_crop_and_resize" and not 0 <= xo <= n_in - 1:
            raise ValueError("an output takes the extrapolation value")
        whole = math.floor(xo)
        ratio = Fraction(1) if xo == whole else xo - whole
        if mode == "nearest":
            coeffs = _nearest_coeffs(ratio, nearest_mode)
        elif mode == "linear":
            coeffs = [1 - ratio, ratio]
        elif mode == "cubic":
            coeffs = _cubic_coeffs(ratio, cubic_a)
        else:
            raise ValueError(f"mode {mode!r}")
        pad = (len(coeffs) + 1) // 2
        nearest = sorted(range(-pad, n_in + pad),
                         key=lambda i: (abs(xo - i), i))[:len(coeffs)]
        positions = sorted(nearest)
        if exclude_outside:
            coeffs = [c if 0 <= i < n_in else Fraction(0)
                      for c, i in zip(coeffs, positions)]
            total = sum(coeffs)
            coeffs = [c / total for c in coeffs]
        row = {}
        for c, i in zip(coeffs, positions):
            if c:
                j = min(max(i, 0), n_in - 1)
                row[j] = row.get(j, Fraction(0)) + c
        out.append({j: w for j, w in row.items() if w})
    return out


# --------------------------------------------------------------------------- #
# One ONNX file                                                                 #
# --------------------------------------------------------------------------- #
class Rz(NamedTuple):
    """One file: a ``Resize`` (or old ``Upsample``) node on X of shape
    (1, C, H, W). ``scales`` and ``sizes`` hold all four entries (batch,
    channel, height, width). ``keyword`` spells the roi input (and an unused
    scales input) as an empty name, which onnx2torch passes on as keywords;
    otherwise roi is an initializer named ``roi`` holding ``roi``, or with
    ``const`` a Constant node (and so are scales and sizes), the spelling of
    the YOLO benchmark files (``verify/resize.md`` section 10). ``wrap``
    adds nodes around the resize: ``"flatten"`` a Flatten after it,
    ``"chain"`` a 1x1 Conv before it and a 2x1 Conv after it, ``"relu"`` a
    Relu before it, ``"relu_conv"`` a Relu and a 1x1 Conv after it. ``box``
    multiplies the grid box ``pixel_box`` (a box of 8 reaches far beyond
    plus and minus 1, with intervals up to 10 wide)."""
    opset: int = 13
    mode: str = "nearest"
    ctm: Optional[str] = None
    nearest_mode: Optional[str] = None
    scales: Optional[tuple] = None
    sizes: Optional[tuple] = None
    keyword: bool = False
    const: bool = False
    roi: tuple = ()
    extra: tuple = ()
    chw: tuple = CHW
    op: str = "Resize"
    wrap: str = ""
    box: int = 1

    def label(self):
        parts = [f"{self.op.lower()}{self.opset}", self.mode]
        if self.ctm:
            parts.append(self.ctm)
        if self.nearest_mode:
            parts.append(self.nearest_mode)
        if self.scales is not None:
            parts.append("scales" + "x".join(f"{s:g}" for s in self.scales))
        if self.sizes is not None:
            parts.append("sizes" + "x".join(str(s) for s in self.sizes))
        parts += [f"{k}={v:g}" for k, v in self.extra]
        if self.roi:
            parts.append("roi" + "-".join(f"{r:g}" for r in self.roi[-2:]))
        if self.op == "Resize" and self.opset >= 11:
            parts.append("keyword" if self.keyword else
                         "constnodes" if self.const else "positional")
        if self.chw != CHW:
            parts.append("chw" + "x".join(str(v) for v in self.chw))
        if self.wrap:
            parts.append(self.wrap)
        if self.box != 1:
            parts.append(f"box{self.box}")
        return "-".join(parts)


def _resize_shape(spec):
    """The shape of the resize node's output: ``sizes``, else each input
    length times its scale, rounded down."""
    if spec.sizes is not None:
        return tuple(int(v) for v in spec.sizes)
    return tuple(math.floor(d * _f32(s))
                 for d, s in zip((1,) + tuple(spec.chw), spec.scales))


# The affine maps of the "chain" wrap: dyadic weights, so that float32 is
# exact on dyadic inputs. W1 mixes the two channels; W2 is a 2x1 kernel.
_W1 = np.array([[1.0, 0.5], [-0.25, 1.0]]).reshape(2, 2, 1, 1)
_B1 = np.array([0.25, -0.5])
_W2 = np.array([[[1.0, -0.5], [0.25, 0.0]],
                [[0.0, 1.0], [-1.0, 0.5]],
                [[0.5, 0.25], [0.75, -0.25]]]).reshape(3, 2, 2, 1)
_B2 = np.array([0.5, 0.0, -0.25])


def _model_shape(spec):
    shape = _resize_shape(spec)
    if spec.wrap == "flatten":
        return (shape[0], int(np.prod(shape[1:])))
    if spec.wrap == "chain":
        return (shape[0], 3, shape[2] - 1, shape[3])
    return shape


def _save(directory, spec, name):
    f32 = lambda v: np.array(v, dtype=np.float32)          # noqa: E731
    inits, nodes = {}, []
    source = "X"
    if spec.wrap == "relu":
        nodes.append(helper.make_node("Relu", ["X"], ["A"]))
        source = "A"
    if spec.wrap == "chain":
        inits.update(W1=f32(_W1), B1=f32(_B1), W2=f32(_W2), B2=f32(_B2))
        nodes.append(helper.make_node("Conv", ["X", "W1", "B1"], ["A"]))
        source = "A"
    if spec.wrap == "relu_conv":
        inits.update(W1=f32(_W1), B1=f32(_B1))
    target = "R" if spec.wrap in ("flatten", "chain", "relu_conv") else "Y"
    attrs = {"mode": spec.mode}
    if spec.op == "Upsample":
        if spec.opset >= 9:
            inits["scales"] = f32(spec.scales)
            node = helper.make_node("Upsample", [source, "scales"], [target],
                                    **attrs)
        else:
            node = helper.make_node("Upsample", [source], [target],
                                    scales=[float(s) for s in spec.scales],
                                    **attrs)
    elif spec.opset == 10:
        inits["scales"] = f32(spec.scales)
        node = helper.make_node("Resize", [source, "scales"], [target], **attrs)
    else:
        if spec.ctm is not None:
            attrs["coordinate_transformation_mode"] = spec.ctm
        if spec.nearest_mode is not None:
            attrs["nearest_mode"] = spec.nearest_mode
        attrs.update(dict(spec.extra))
        roi = "" if spec.keyword else "roi"
        if not spec.keyword:
            inits["roi"] = f32(spec.roi)
        if spec.scales is not None:
            inits["scales"] = f32(spec.scales)
            inputs = [source, roi, "scales"]
        else:
            if not spec.keyword:
                inits["scales"] = f32([])
            inits["sizes"] = np.array(spec.sizes, dtype=np.int64)
            inputs = [source, roi, "" if spec.keyword else "scales", "sizes"]
        node = helper.make_node("Resize", inputs, [target], **attrs)
    if spec.const:                     # the inputs as Constant nodes instead
        for name in ("roi", "scales", "sizes"):
            if name in inits:
                nodes.append(helper.make_node(
                    "Constant", [], [name], value=numpy_helper.from_array(
                        inits.pop(name), f"{name}_value")))
    nodes.append(node)
    if spec.wrap == "flatten":
        nodes.append(helper.make_node("Flatten", ["R"], ["Y"], axis=1))
    if spec.wrap == "chain":
        nodes.append(helper.make_node("Conv", ["R", "W2", "B2"], ["Y"]))
    if spec.wrap == "relu_conv":
        nodes.append(helper.make_node("Relu", ["R"], ["Q"]))
        nodes.append(helper.make_node("Conv", ["Q", "W1", "B1"], ["Y"]))
    return save_onnx(directory, nodes, (1,) + tuple(spec.chw),
                     _model_shape(spec), inits, name=name, opset=spec.opset)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """``build(spec)`` is the path of the spec's file, written once per module."""
    directory = tmp_path_factory.mktemp("resize")
    paths = {}

    def path_of(spec):
        if spec not in paths:
            paths[spec] = _save(directory, spec, f"m{len(paths)}")
        return paths[spec]
    return path_of


# --------------------------------------------------------------------------- #
# The true map of a file                                                        #
# --------------------------------------------------------------------------- #
def _semantics(spec):
    """(mode's coordinate rule, nearest rule) the file means. Opset 10 Resize
    and the old Upsample have no attributes: they mean asymmetric with floor
    (test_formula_matches_onnxruntime_on_unit_images)."""
    if spec.op == "Upsample" or spec.opset == 10:
        return "asymmetric", "floor"
    return spec.ctm or "half_pixel", spec.nearest_mode or "round_prefer_floor"


def _axes(spec):
    """The per-axis rows (height, width) of the resize node, or None where the
    formula here does not model the file."""
    channels, height, width = spec.chw
    shape = _resize_shape(spec)
    if shape[:2] != (1, channels):
        return None                                # a batch or channel resize
    h_out, w_out = shape[2:]
    if spec.sizes is not None:
        scale_h, scale_w = Fraction(h_out, height), Fraction(w_out, width)
    else:
        scale_h, scale_w = _f32(spec.scales[2]), _f32(spec.scales[3])
    ctm, nearest_mode = _semantics(spec)
    integer = (scale_h * height == h_out and scale_w * width == w_out)
    if ctm not in ("half_pixel", "asymmetric") and not integer:
        return None
    extra = dict(spec.extra)
    cubic_a = _f32(extra.get("cubic_coeff_a", -0.75))
    exclude = bool(extra.get("exclude_outside", 0))
    roi_h = roi_w = (Fraction(0), Fraction(1))
    if ctm == "tf_crop_and_resize":
        r = [_f32(v) for v in spec.roi]
        roi_h, roi_w = (r[2], r[6]), (r[3], r[7])
    return (_axis_rows(spec.mode, ctm, nearest_mode, scale_h, height, h_out,
                       roi_h, cubic_a, exclude),
            _axis_rows(spec.mode, ctm, nearest_mode, scale_w, width, w_out,
                       roi_w, cubic_a, exclude))


def _resize_matrix(spec):
    """The resize node's exact map as a list of rows ``{input: weight}``, both
    sides flat in (channel, height, width) order; None where not modelled."""
    axes = _axes(spec)
    if axes is None:
        return None
    rows_h, rows_w = axes
    channels, height, width = spec.chw
    out = []
    for c in range(channels):
        for row_h in rows_h:
            for row_w in rows_w:
                out.append({(c * height + p) * width + q: wp * wq
                            for p, wp in row_h.items()
                            for q, wq in row_w.items()})
    return out


def _is_block(spec):
    """Whether the resize node's exact map is block replication: every output
    position (i, j) reads exactly input (i // s_h, j // s_w), with s_h and s_w
    the integer ratios of the output to the input lengths."""
    axes = _axes(spec)
    if axes is None:
        return False
    for rows_axis, n_in in zip(axes, spec.chw[1:]):
        if len(rows_axis) % n_in:
            return False
        s = len(rows_axis) // n_in
        if any(row != {o // s: 1} for o, row in enumerate(rows_axis)):
            return False
    return True


def _dense(matrix, n_in):
    out = np.zeros((len(matrix), n_in))
    for k, row in enumerate(matrix):
        for i, w in row.items():
            out[k, i] = float(w)
    return out


def _conv(x, weight, bias):
    """A stride-1 convolution without padding, (C, H, W) -> (O, H', W')."""
    o, _, kh, kw = weight.shape
    h, w = x.shape[1] - kh + 1, x.shape[2] - kw + 1
    out = np.empty((o, h, w))
    for i in range(h):
        for j in range(w):
            out[:, i, j] = np.einsum("ocij,cij->o", weight,
                                     x[:, i:i + kh, j:j + kw]) + bias
    return out


def _formula_outputs(spec, points):
    """The model's true outputs on ``points`` ((C, H, W) arrays), each flat in
    the network's order, from the exact map in float64; and whether every
    output is a copy of one input (no arithmetic beyond the wraps' dyadic
    weights). None where the formula does not model the file."""
    matrix = _resize_matrix(spec)
    if matrix is None:
        return None
    channels, height, width = spec.chw
    dense = _dense(matrix, channels * height * width)
    copies = all(list(row.values()) == [1] for row in matrix)
    shape = _resize_shape(spec)
    outputs = []
    for x in points:
        x = np.asarray(x, dtype=np.float64).reshape(spec.chw)
        if spec.wrap == "relu":
            x = np.maximum(x, 0.0)
        if spec.wrap == "chain":
            x = _conv(x, _W1, _B1)
        y = (dense @ x.reshape(-1)).reshape(shape[1:])
        if spec.wrap == "chain":
            y = _conv(y, _W2, _B2)
        if spec.wrap == "relu_conv":
            y = _conv(np.maximum(y, 0.0), _W1, _B1)
        outputs.append(y.reshape(-1))
    return outputs, copies


def _truth(spec, path, points):
    """``(outputs, shape)``: the true outputs on ``points`` in the network's
    order and the model's output shape. onnxruntime replays every point
    first; where the formula models the file it must agree with onnxruntime
    (exactly for copies, to ``ORT_AGREEMENT`` otherwise) and its outputs are
    returned, else onnxruntime's (only used for copies of dyadic inputs)."""
    shape = _model_shape(spec)
    replayed = replay_onnx(path, points, (1,) + tuple(spec.chw))
    if any(y.size != int(np.prod(shape)) for y in replayed):
        pytest.fail(f"{spec.label()}: onnxruntime returned "
                    f"{replayed[0].size} entries, the file declares {shape}")
    formula = _formula_outputs(spec, points)
    if formula is None:
        return replayed, shape
    exact, copies = formula
    for k, (y_ort, y) in enumerate(zip(replayed, exact)):
        gap = np.abs(y_ort - y)
        bound = 0.0 if copies else ORT_AGREEMENT
        if np.any(gap > bound * np.maximum(1.0, np.abs(y))):
            pytest.fail(f"{spec.label()}: the formula disagrees with "
                        f"onnxruntime on point {k} by {gap.max():.3g}: the "
                        f"reference is wrong, not the library")
    return exact, shape


_TRUTH_CACHE = {}


def _grid_truth(spec, path):
    """``(lo, hi, points, outputs, shape)`` of the grid box ``pixel_box``
    (times ``spec.box``), computed once per file."""
    if spec not in _TRUTH_CACHE:
        lo, hi = (spec.box * v for v in pixel_box(*spec.chw))
        points = samples(lo, hi)
        _TRUTH_CACHE[spec] = (lo, hi, points) + _truth(spec, path, points)
    return _TRUTH_CACHE[spec]


# --------------------------------------------------------------------------- #
# The check of an answer                                                        #
# --------------------------------------------------------------------------- #
def _entries(s):
    return rows(s)[0].size


def _describe(s):
    if isinstance(s, (ImageStar, ImageZono)):
        return (f"{type(s).__name__} stored {s.height}x{s.width}x"
                f"{s.num_channels}")
    return f"{type(s).__name__} of {_entries(s)} entries"


def _outer_box(s):
    """Bounds of ``c + G a`` over the predicate box alone (the rows ``C a <=
    d`` dropped): a superset of the set, read from its arrays."""
    c, G, _, _, lb, ub = rows(s)
    with np.errstate(invalid="ignore"):
        low = c + np.where(G > 0, G * lb, 0.0).sum(1) + \
            np.where(G < 0, G * ub, 0.0).sum(1)
        high = c + np.where(G > 0, G * ub, 0.0).sum(1) + \
            np.where(G < 0, G * lb, 0.0).sum(1)
    return low, high


def _escapes_every_box(sets, outputs, out_chw, what):
    """A cheap first pass of the membership check: a true output outside the
    outer box of every returned set, by more than the oracle's slack, is in
    none of them. Skipped when a set's stored shape does not match (then
    ``assert_covered`` reports the shape)."""
    excess = np.full(len(outputs), np.inf)
    for s in sets:
        if isinstance(s, (ImageStar, ImageZono)):
            if out_chw is None or (s.height, s.width, s.num_channels) != (
                    out_chw[1], out_chw[2], out_chw[0]):
                return
            view = [hwc(y, *out_chw) for y in outputs]
        else:
            view = outputs
        low, high = _outer_box(s)
        for k, y in enumerate(view):
            slack = LP_TOL * max(1.0, float(np.max(np.abs(y))))
            excess[k] = min(excess[k], float(np.max(
                np.maximum(low - y, y - high))) - slack)
    missed = np.flatnonzero(excess > 0)
    if missed.size:
        raise AssertionError(
            f"{what}: {missed.size} of {len(outputs)} true outputs lie outside "
            f"the bounding box of every returned set, so in none of them "
            f"(largest excess {excess[missed].max():.3g}, first missed output "
            f"index {missed[0]}); result: "
            f"{'; '.join(_describe(s) for s in sets)}")


def _assert_holds(sets, outputs, shape, what):
    """The promise once the call answered: every returned set has the true
    entry count, and every true output is a member of one of them (one-sided,
    against the rows, slack ``LP_TOL``). An image output (batch 1, rank 4) is
    compared in each image set's stored order; a flat output, or one with a
    batch of more than one image, in the network's order."""
    total = int(np.prod(shape))
    for s in sets:
        if _entries(s) != total:
            raise AssertionError(
                f"{what}: a returned {_describe(s)} holds {_entries(s)} "
                f"entries, the true output of shape {tuple(shape)} has {total}")
    if len(shape) == 4 and shape[0] == 1:
        chw = tuple(shape[1:])
        _escapes_every_box(sets, outputs, chw, what)
        assert_covered(sets, outputs, what, out_chw=chw)
    else:
        _escapes_every_box(sets, outputs, None, what)
        assert_covered(sets, outputs, what, stored_order=True)


def _answer(path, input_set, method, entry, what, **options):
    """The library's answer for the file: the returned sets, or None for a
    refusal (``NotImplementedError`` at load or at reach). Any other
    exception is not a refusal and reaches the test."""
    try:
        model = load_onnx(path)
        if entry == "reach_pytorch_model":
            return reach_model(model, input_set, method, what, **options)
        result = NeuralNetwork(model).reach(input_set, method=method, **options)
    except NotImplementedError:
        return None
    if result is None or len(result) == 0:
        raise AssertionError(f"{what}: NeuralNetwork.reach returned no sets "
                             f"for a non-empty input set")
    return list(result)


def _grid_cell(build, spec, kind, method="approx",
               entry="reach_pytorch_model", must_answer=False):
    """The cell of one file on the grid box: refused (when allowed) or the
    true entry count with every true output a member. Returns the sets."""
    path = build(spec)
    lo, hi, points, outputs, shape = _grid_truth(spec, path)
    what = f"{spec.label()} on {kind} {method} via {entry}"
    sets = _answer(path, image_set(kind, lo, hi), method, entry, what)
    if sets is None:
        if must_answer:
            raise AssertionError(f"{what}: refused, but the file is a block "
                                 f"copy the library supports")
        return None
    _assert_holds(sets, outputs, shape, what)
    return sets


def _must_answer(spec):
    """Supported today and kept by DESIGN section 17: a positional nearest
    Resize whose pair is a block copy at every integer scale, at an integer
    scale, or the old Upsample in nearest mode."""
    if spec.mode != "nearest":
        return False
    if spec.op == "Upsample":
        return True
    if spec.keyword or not 11 <= spec.opset <= 17 or spec.roi:
        return False
    shape = _resize_shape(spec)
    if shape[:2] != (1, spec.chw[0]) or not _is_block(spec):
        return False
    return _semantics(spec) in BLOCK_PAIRS


def _answered_today(spec):
    """Whether the file reaches the block replication rule today: scales or
    sizes arrive positionally (roi named, or the old Upsample upgraded), at an
    integer scale with the batch and channel entries of ``scales`` at 1. The
    keyword spelling, opset 10, opsets above 17 and non-integer scales are
    refused (``verify/resize.md`` section 3)."""
    if spec.keyword or spec.opset == 10 or spec.opset > 17:
        return False
    if spec.scales is not None:
        if [float(s) for s in spec.scales[:2]] != [1.0, 1.0]:
            return False
        return all(_f32(s).denominator == 1 for s in spec.scales[2:])
    return all(o % i == 0 for o, i in zip(spec.sizes[2:], spec.chw[1:]))


def _marks(spec, defect):
    """The known-bug marker of a file answered today whose exact map is not
    block replication; none otherwise."""
    if _answered_today(spec) and not _is_block(spec):
        return (_bug(defect),)
    return ()


def _kinds(spec, defect):
    return [pytest.param(spec, kind, id=f"{spec.label()}-{kind}",
                         marks=_marks(spec, defect)) for kind in IMAGE_KINDS]


# --------------------------------------------------------------------------- #
# The formula, checked against onnxruntime                                      #
# --------------------------------------------------------------------------- #
def _scaled(factors):
    return {"scales": (1, 1) + tuple(factors)}


SCALE_FACTORS = ((2, 2), (3, 3), (2, 3))


def _index_map_cells():
    for opset in (11, 13):
        for ctm in CTMS:
            for nearest_mode in NEAREST_MODES:
                factors = [_scaled(f) for f in SCALE_FACTORS]
                if opset == 13:
                    factors.append({"sizes": (1, 2, 6, 12)})
                for how in factors:
                    spec = Rz(opset=opset, ctm=ctm, nearest_mode=nearest_mode,
                              **how)
                    yield pytest.param(spec, id=spec.label())


@pytest.mark.parametrize("spec", list(_index_map_cells()))
def test_formula_index_map_matches_onnxruntime(build, spec):
    """The oracle's own check, nearest mode: on a label image (pixel value =
    its flat index, exact in float32) onnxruntime's output at every position
    is the index of the input pixel it reads, which must equal the formula's.
    Catches a wrong coordinate rule or rounding rule in ``_axis_rows`` (for
    example halves rounded the wrong way), which would mark the wrong cells."""
    path = build(spec)
    n = int(np.prod(spec.chw))
    label = np.arange(n, dtype=np.float64).reshape(spec.chw)
    read = replay_onnx(path, [label], (1,) + tuple(spec.chw))[0]
    expected = [next(iter(row)) for row in _resize_matrix(spec)]
    assert all(list(row.values()) == [1] for row in _resize_matrix(spec))
    assert read.tolist() == [float(i) for i in expected], (
        f"{spec.label()}: onnxruntime reads {read.astype(int).tolist()}, the "
        f"formula {expected}")


def _weight_cells():
    for mode in ("linear", "cubic"):
        for ctm in CTMS:
            for how in [_scaled(f) for f in SCALE_FACTORS] + [
                    {"sizes": (1, 2, 6, 12)}]:
                yield Rz(mode=mode, ctm=ctm, **how)
        if mode == "cubic":
            for extra in ((("cubic_coeff_a", -0.5),), (("exclude_outside", 1),)):
                for ctm in CTMS:
                    yield Rz(mode=mode, ctm=ctm, extra=extra, **_scaled((2, 3)))
    quarter = (0, 0, 0, 0, 1, 1, 0.5, 0.5)
    for mode, nearest_mode in (("nearest", "round_prefer_floor"),
                               ("nearest", "ceil"), ("linear", None)):
        for how in (_scaled((2, 2)), {"sizes": (1, 2, 8, 8)}):
            yield Rz(mode=mode, ctm="tf_crop_and_resize", roi=quarter,
                     nearest_mode=nearest_mode, chw=(2, 4, 4), **how)
    for mode in ("nearest", "linear"):
        yield Rz(op="Upsample", opset=9, mode=mode, **_scaled((2, 3)))
        yield Rz(op="Upsample", opset=7, mode=mode, **_scaled((3, 2)))
        yield Rz(opset=10, mode=mode, **_scaled((2, 3)))
    for mode, ctm, nearest_mode in (("linear", "half_pixel", None),
                                    ("linear", "asymmetric", None),
                                    ("nearest", "asymmetric", "floor"),
                                    ("nearest", "half_pixel", "round_prefer_ceil")):
        for factors in ((1.5, 1.5), (0.5, 0.5), (2.25, 2.25)):
            yield Rz(mode=mode, ctm=ctm, nearest_mode=nearest_mode,
                     **_scaled(factors))


@pytest.mark.parametrize("spec", [pytest.param(s, id=s.label())
                                  for s in _weight_cells()])
def test_formula_matches_onnxruntime_on_unit_images(build, spec):
    """The oracle's own check, every mode: onnxruntime on the unit images gives
    the file's matrix column by column; it must equal the formula's matrix to
    ``ORT_AGREEMENT`` (exactly where every weight is 1). Covers linear and
    cubic in the four coordinate rules, ``cubic_coeff_a`` and
    ``exclude_outside``, a real crop with ``tf_crop_and_resize``, opset 10
    and the old Upsample (asymmetric with floor), and non-integer scales.
    Catches a wrong kernel, edge rule or crop in ``_axis_rows``."""
    path = build(spec)
    n = int(np.prod(spec.chw))
    replayed = np.stack(replay_onnx(path, list(np.eye(n)),
                                    (1,) + tuple(spec.chw)), axis=1)
    matrix = _resize_matrix(spec)
    assert matrix is not None, f"{spec.label()} is not modelled"
    dense = _dense(matrix, n)
    assert replayed.shape == dense.shape, (
        f"{spec.label()}: onnxruntime gives {replayed.shape}, the formula "
        f"{dense.shape}")
    copies = all(list(row.values()) == [1] for row in matrix)
    gap = np.abs(replayed - dense).max()
    assert gap <= (0.0 if copies else ORT_AGREEMENT), (
        f"{spec.label()}: onnxruntime and the formula differ by {gap:.3g}")


# verify/resize.md section 7: the X cells (block replication is wrong) of the
# positional table, per pair, at scales (2, 2), (3, 3), (2, 3) and sizes
# (6, 12) on the 3x4 image.
_REPORT_X = {
    ("half_pixel", "round_prefer_floor"): "....",
    ("half_pixel", "round_prefer_ceil"): "....",
    ("half_pixel", "floor"): "XXXX",
    ("half_pixel", "ceil"): "XXXX",
    ("asymmetric", "round_prefer_floor"): ".XXX",
    ("asymmetric", "round_prefer_ceil"): "XXXX",
    ("asymmetric", "floor"): "....",
    ("asymmetric", "ceil"): "XXXX",
    ("align_corners", "round_prefer_floor"): ".XXX",
    ("align_corners", "round_prefer_ceil"): ".XXX",
    ("align_corners", "floor"): "XXXX",
    ("align_corners", "ceil"): "XXXX",
    ("pytorch_half_pixel", "round_prefer_floor"): "....",
    ("pytorch_half_pixel", "round_prefer_ceil"): "....",
    ("pytorch_half_pixel", "floor"): "XXXX",
    ("pytorch_half_pixel", "ceil"): "XXXX",
}


def test_the_marked_nearest_cells_are_the_x_cells_of_the_report():
    """The rule that places the markers (``_is_block`` on the formula) gives
    exactly the X cells measured in ``verify/resize.md`` section 7, and
    linear and cubic are never a block copy at these scales. Catches a
    marker rule that would mark a sound cell or leave a wrong one unmarked."""
    hows = [_scaled(f) for f in SCALE_FACTORS] + [{"sizes": (1, 2, 6, 12)}]
    for pair, expected in _REPORT_X.items():
        got = "".join("." if _is_block(Rz(ctm=pair[0], nearest_mode=pair[1],
                                          **how)) else "X" for how in hows)
        assert got == expected, f"{pair}: the formula gives {got}, the report {expected}"
    for mode in ("linear", "cubic"):
        for ctm in CTMS:
            for how in hows:
                assert not _is_block(Rz(mode=mode, ctm=ctm, **how))


def test_the_block_pairs_are_block_copies_at_every_integer_scale():
    """``BLOCK_PAIRS`` (the cells that must keep answering) are exactly the
    pairs whose formula is a block copy at every integer scale 1 to 6 and
    input length 1 to 7; asymmetric with round_prefer_floor and align_corners
    with either round rule are block copies at scale 2 only. Catches a wrong
    list of supported cells."""
    def block(ctm, nearest_mode, s, n):
        got = _axis_rows("nearest", ctm, nearest_mode, Fraction(s), n, s * n)
        return all(row == {o // s: 1} for o, row in enumerate(got))
    always, at_two = set(), set()
    for ctm in CTMS:
        for nearest_mode in NEAREST_MODES:
            ok = {s: all(block(ctm, nearest_mode, s, n) for n in range(1, 8))
                  for s in range(1, 7)}
            if all(ok.values()):
                always.add((ctm, nearest_mode))
            elif ok[2]:
                at_two.add((ctm, nearest_mode))
    assert always == set(BLOCK_PAIRS)
    assert at_two == {("asymmetric", "round_prefer_floor"),
                      ("align_corners", "round_prefer_floor"),
                      ("align_corners", "round_prefer_ceil")}


# --------------------------------------------------------------------------- #
# RESIZE-3: nearest mode, the 16 rounding rules                                 #
# --------------------------------------------------------------------------- #
def _nearest_cells():
    """One cell per code path of the reach rule, plus each supported pair.

    n2v reads neither ``coordinate_transformation_mode`` nor
    ``nearest_mode``: every nearest file takes the same path, chosen by the
    spelling (``scales`` or ``sizes``, positional or keyword) and the set
    type (``upsample_star`` or ``upsample_zono``). Opsets 11 and 13 share one
    converter function, so opset 11 keeps one file; the formula is checked
    against onnxruntime for all 16 pairs at both opsets by
    ``test_formula_index_map_matches_onnxruntime``. Factors (2, 3) differ
    between height and width, so a swap of the two shows."""
    floor = dict(ctm="asymmetric", nearest_mode="floor")
    ceil = dict(ctm="asymmetric", nearest_mode="ceil")
    cells = []
    # Every pair of BLOCK_PAIRS must answer (DESIGN section 17); on the
    # ImageStar for every pair, on the ImageZono for the pair PyTorch exports.
    for ctm, nearest_mode in BLOCK_PAIRS:
        spec = Rz(ctm=ctm, nearest_mode=nearest_mode, **_scaled((2, 3)))
        cells += [(spec, kind) for kind in IMAGE_KINDS
                  if kind == "ImageStar" or spec.ctm == "asymmetric"]
    for spec in (Rz(sizes=(1, 2, 6, 12), **floor),
                 Rz(opset=11, **floor, **_scaled((2, 3))),
                 # RESIZE-3, one marked file per spelling: a rule that is a
                 # block copy at no scale (asymmetric with ceil) by scales, and
                 # one that is a block copy at scale 2 only (align_corners
                 # with round_prefer_floor; width factor 3) by sizes.
                 Rz(**ceil, **_scaled((2, 3))),
                 Rz(ctm="align_corners", nearest_mode="round_prefer_floor",
                    sizes=(1, 2, 6, 12)),
                 # the keyword spelling, refused today (RESIZE-6)
                 Rz(**ceil, keyword=True, **_scaled((2, 3))),
                 Rz(**ceil, keyword=True, sizes=(1, 2, 6, 12))):
        cells += [(spec, kind) for kind in IMAGE_KINDS]
    for spec, kind in cells:
        yield pytest.param(spec, kind, id=f"{spec.label()}-{kind}",
                           marks=_marks(spec, "RESIZE-3"))


@pytest.mark.parametrize("spec, kind", list(_nearest_cells()))
def test_nearest_resize_holds_every_onnxruntime_output(build, spec, kind):
    """Nearest Resize on (2, 3, 4) by scales (2, 3) or sizes (6, 12): the five
    pairs of ``BLOCK_PAIRS`` (must answer), the PyTorch export pair
    (asymmetric, floor) also by sizes, at opset 11 and on the ImageZono, two
    RESIZE-3 files (marked), and two files with the roi spelled as an empty
    name (keyword scales and keyword sizes, refused today). Catches a
    supported block-copy file refused or answered with the pixels in the
    wrong place (the factors or the sizes read for the wrong axis, a tiled
    copy, a reshape in the wrong order), the block copy applied to a rule
    that reads other pixels (RESIZE-3), and a fix of RESIZE-6 that reads the
    keyword scales or sizes without checking the rounding rule."""
    _grid_cell(build, spec, kind, must_answer=_must_answer(spec))


@pytest.mark.parametrize("kind, method", [("ImageStar", "exact"),
                                          ("ImageStar", "approx"),
                                          ("ImageZono", "approx")])
def test_nearest_resize_of_a_wide_box_holds_every_onnxruntime_output(
        build, kind, method):
    """The supported file (asymmetric, floor, scales (2, 3)) on the grid box
    times 8: entries from -16 to 24, intervals 2 to 10 wide, so the data
    leave the predicate box [-1, 1] of every variable. Catches a result that
    reads a predicate variable, or a unit generator, in place of the copied
    entry: inside [-1, 1] such a set still holds the outputs. Under the exact
    method every point of the result must also have a preimage."""
    spec = Rz(ctm="asymmetric", nearest_mode="floor", box=8, **_scaled((2, 3)))
    sets = _grid_cell(build, spec, kind, method, must_answer=True)
    if method == "exact":
        lo, hi = _grid_truth(spec, build(spec))[:2]
        _assert_inside_image(sets, spec, image_set(kind, lo, hi),
                             f"{spec.label()} exact")


# --------------------------------------------------------------------------- #
# RESIZE-1: linear and cubic                                                    #
# --------------------------------------------------------------------------- #
def _smooth_cells():
    """One cell per code path of the reach rule, as for nearest mode.

    n2v reads neither the mode nor ``coordinate_transformation_mode`` nor
    the cubic attributes on the path of graph scales or sizes
    (``_resolve_scale_factors``): every linear and cubic file by scales or
    by sizes takes the block copy of ``upsample_star`` or ``upsample_zono``,
    and opsets 11 and 13 share one converter function. So RESIZE-1 is marked
    once per spelling (scales, sizes) and set type, with half_pixel, the rule
    for which the converted module agrees with onnxruntime (a load-time
    comparison, PLAN 12b, would not catch these files), and once for cubic
    mode (a fix of the mode check that refuses only 'linear' leaves cubic
    marked). Factors (2, 3) differ between height and width."""
    marked = (Rz(mode="linear", ctm="half_pixel", **_scaled((2, 3))),
              Rz(mode="linear", ctm="half_pixel", sizes=(1, 2, 6, 12)),
              Rz(mode="cubic", ctm="half_pixel", **_scaled((2, 3))))
    # The roi input spelled as an empty name: refused today at the mode check
    # of _get_scale_factors (the walker reads no keyword inputs, RESIZE-6).
    keyword = (Rz(mode="linear", ctm="half_pixel", keyword=True,
                  **_scaled((2, 3))),
               Rz(mode="linear", ctm="half_pixel", keyword=True,
                  sizes=(1, 2, 6, 12)))
    # Passing neighbours, answered today: an axis of one pixel scaled up and
    # scale 1 on the other, where every mode copies (each output reads the
    # single pixel of its axis), in height and in width, by scales and by
    # sizes, in both modes.
    neighbours = (Rz(mode="linear", ctm="align_corners", chw=(2, 1, 4),
                     **_scaled((3, 1))),
                  Rz(mode="linear", ctm="asymmetric", chw=(2, 3, 1),
                     **_scaled((1, 2))),
                  Rz(mode="linear", ctm="half_pixel", chw=(2, 1, 4),
                     sizes=(1, 2, 3, 4)),
                  Rz(mode="cubic", ctm="align_corners", chw=(2, 1, 4),
                     **_scaled((3, 1))))
    for spec in marked + keyword + neighbours:
        yield from _kinds(spec, "RESIZE-1")


@pytest.mark.parametrize("spec, kind", list(_smooth_cells()))
def test_linear_and_cubic_resize_hold_every_onnxruntime_output(build, spec, kind):
    """Linear and cubic Resize on (2, 3, 4) by scales (2, 3) or sizes
    (6, 12): the block copy applied to an interpolating resize (RESIZE-1,
    marked once per spelling and set type, and once in cubic mode). The roi
    spelled as an empty name (keyword scales and keyword sizes, refused
    today) catches a fix of RESIZE-6 that reads the keyword inputs without
    checking the mode.
    The neighbours with an axis of one pixel are copies in every mode and
    pass today: they show the reference and the layout code right in linear
    and cubic mode, and catch such a file answered with the factors of the
    two axes exchanged, the generators reshaped in another order than the
    centre, or the sizes divided by the wrong input extent."""
    _grid_cell(build, spec, kind)


# --------------------------------------------------------------------------- #
# RESIZE-2: the opset 7 and 9 Upsample operator                                 #
# --------------------------------------------------------------------------- #
def _upsample_cells():
    for opset in (7, 9):
        for mode in ("nearest", "linear"):
            for factors in ((2, 2), (2, 3)):
                if mode == "nearest" and factors == (2, 2):
                    continue    # (2, 3) takes the same path and shows a swap
                if mode == "linear" and (opset, factors) != (9, (2, 3)):
                    # RESIZE-2 is one failing path (the upgrade to a Resize
                    # with positional scales, then the block copy): marked
                    # once per set type, at the opset of the cgan files.
                    continue
                yield from _kinds(Rz(op="Upsample", opset=opset, mode=mode,
                                     **_scaled(factors)), "RESIZE-2")


@pytest.mark.parametrize("spec, kind", list(_upsample_cells()))
def test_old_upsample_operator_holds_every_onnxruntime_output(build, spec, kind):
    """The opset 7 (scales attribute) and opset 9 (scales input) Upsample
    operator, which the loader upgrades to a Resize. Nearest mode is a block
    copy and must answer (the cgan benchmark files use it), by factors (2, 3)
    that differ between height and width; linear mode is reached as a block
    copy (RESIZE-2). Catches the loader's upgrade path misplacing pixels or
    exchanging the two factors in nearest mode, and linear answered as
    nearest."""
    _grid_cell(build, spec, kind, must_answer=_must_answer(spec))


# --------------------------------------------------------------------------- #
# The roi, scales and sizes inputs as Constant nodes (the benchmark spelling)    #
# --------------------------------------------------------------------------- #
def _const_cells():
    # Nearest: opset 11 only. The Constant converter of opsets 9 to 13 is one
    # function (onnx2torch node_converters/constant.py), so opsets 12 and 13
    # take the same path; the rounding rule is not read by n2v.
    for spec in (Rz(opset=11, const=True, ctm="asymmetric",
                    nearest_mode="floor", **_scaled((2, 3))),
                 Rz(opset=11, const=True, ctm="asymmetric",
                    nearest_mode="floor", sizes=(1, 2, 6, 12)),
                 Rz(opset=11, const=True, ctm="asymmetric",
                    nearest_mode="ceil", **_scaled((2, 3)))):
        yield from _kinds(spec, "RESIZE-3")
    # Linear: RESIZE-1 through the Constant nodes, marked once per set type
    # (opset 11; the Constant converter of opsets 9 to 13 is one function).
    yield from _kinds(Rz(opset=11, const=True, mode="linear",
                         ctm="half_pixel", **_scaled((2, 2))), "RESIZE-1")


@pytest.mark.parametrize("spec, kind", list(_const_cells()))
def test_resize_with_constant_node_inputs_holds_every_onnxruntime_output(
        build, spec, kind):
    """The roi (empty), scales and sizes inputs written as Constant nodes, as
    the cctsdb_yolo (opset 11) and collins_aerospace (opset 12) benchmark
    files write them, rather than as initializers: the graph value the reach
    driver reads is a constant module, not a stored tensor. The nearest
    block-copy files (by scales and by sizes, opset 11) must answer with
    every true output a member; asymmetric with ceil (RESIZE-3) and linear
    (RESIZE-1) are marked from the formula. Catches the scales or sizes of a
    Constant node not read (a supported benchmark file refused) or misread
    (pixels in the wrong place, a wrong scale on one axis)."""
    _grid_cell(build, spec, kind, must_answer=_must_answer(spec))


# --------------------------------------------------------------------------- #
# RESIZE-4: sizes that change the channel or the batch entry                    #
# --------------------------------------------------------------------------- #
def _sizes_cells():
    floor = dict(ctm="asymmetric", nearest_mode="floor")
    for opset in (13,):
        for sizes, marked in (((1, 2, 6, 8), False), ((1, 4, 6, 8), True),
                              ((2, 2, 6, 8), True)):
            spec = Rz(opset=opset, sizes=sizes, **floor)
            for kind in IMAGE_KINDS:
                yield pytest.param(spec, kind, id=f"{spec.label()}-{kind}",
                                   marks=(_bug("RESIZE-4"),) if marked else ())
    for scales in ((1, 2, 2, 2), (2, 1, 2, 2)):
        spec = Rz(scales=scales, **floor)
        for kind in IMAGE_KINDS:
            yield pytest.param(spec, kind, id=f"{spec.label()}-{kind}")


@pytest.mark.parametrize("spec, kind", list(_sizes_cells()))
def test_resize_of_the_channel_or_batch_axis_is_refused_or_right(build, spec, kind):
    """Nearest Resize (asymmetric, floor) by ``sizes`` (1, 2C, 2H, 2W) and
    (2, C, 2H, 2W): onnxruntime copies channels or images, so the true output
    has twice the entries; n2v answers with the input's channel and batch
    count (RESIZE-4, one marked file for the channel entry and one for the
    batch entry). The neighbour sizes (1, C, 2H, 2W) is a supported block copy
    and must answer; the same channel or batch request by ``scales`` is
    refused today. The truth is onnxruntime's output (copies of dyadic
    inputs, exact). Catches a result of the wrong size, and a check of the
    batch and channel entries of ``scales`` that is dropped or stops early."""
    path = build(spec)
    lo, hi, points, outputs, shape = _grid_truth(spec, path)
    what = f"{spec.label()} on {kind}"
    sets = _answer(path, image_set(kind, lo, hi), "approx",
                   "reach_pytorch_model", what)
    if sets is None:
        if spec.sizes == (1, 2, 6, 8):
            raise AssertionError(f"{what}: refused a supported block copy")
        return
    _assert_holds(sets, outputs, shape, what)


# --------------------------------------------------------------------------- #
# RESIZE-5: tf_crop_and_resize                                                  #
# --------------------------------------------------------------------------- #
_WHOLE = (0, 0, 0, 0, 1, 1, 1, 1)
# Height [0, 1/2], width [1/4, 1] of the image: inside [0, 1], so no output
# takes the extrapolation value, and a crop or a factor read for the other
# axis gives another map.
_CROP_HW = (0, 0, 0, 0.25, 1, 1, 0.5, 1)


def _crop_cells():
    for how in (_scaled((2, 3)), {"sizes": (1, 2, 6, 12)}):
        yield from _kinds(Rz(mode="nearest", ctm="tf_crop_and_resize",
                             roi=_CROP_HW, **how), "RESIZE-5")
    # Linear mode with tf_crop_and_resize takes the same failing path (n2v
    # reads neither the mode nor the roi): RESIZE-5 is marked by the nearest
    # cells above and RESIZE-1 by test_linear_and_cubic_resize_hold_every_*.
    yield from _kinds(Rz(mode="nearest", ctm="tf_crop_and_resize", roi=_WHOLE,
                         chw=(2, 4, 4), **_scaled((2, 2))), "RESIZE-1")


@pytest.mark.parametrize("spec, kind", list(_crop_cells()))
def test_tf_crop_and_resize_holds_every_onnxruntime_output(build, spec, kind):
    """``tf_crop_and_resize``, n2v ignores the region of interest (RESIZE-5).
    Nearest: on (2, 3, 4) with the region [0, 1/2] in height and [1/4, 1] in
    width, by scales (2, 3) and by sizes (6, 12), so that a fix that reads the
    crop or the factor of the other axis gives another map. With the whole
    image as the region, nearest at scale 2 passes by coincidence: output
    index x sits at 3x/7, which rounds (halves down) to x // 2 for every x in
    0..7, so the formula is a block copy. Linear mode takes the same path
    (the roi and the mode are not read) and is marked as RESIZE-1 in the
    linear grid. Catches the region of interest ignored, or read for the
    wrong axis."""
    _grid_cell(build, spec, kind)


# --------------------------------------------------------------------------- #
# Files refused today                                                           #
# --------------------------------------------------------------------------- #
def _refused_cells():
    floor = dict(ctm="asymmetric", nearest_mode="floor")
    # Opset 10 in linear mode guards a future rule for OnnxResizeV10 (which
    # keeps its mode on an inner module) that skips the mode check; the
    # non-integer scales and opset 18 take the same refusal whatever the mode
    # and are kept in nearest mode only.
    specs = [Rz(opset=10, **_scaled((2, 3))),
             Rz(opset=10, mode="linear", **_scaled((2, 2))),
             Rz(**floor, **_scaled((0.5, 0.5))),
             Rz(**floor, **_scaled((2.25, 2.25))),
             Rz(sizes=(1, 2, 4, 6), **floor),
             Rz(opset=18, **floor, **_scaled((2, 3))),
             Rz(opset=19, ctm="asymmetric", nearest_mode="ceil",
                **_scaled((2, 3)))]
    for spec in specs:
        for kind in IMAGE_KINDS:
            yield pytest.param(spec, kind, id=f"{spec.label()}-{kind}")


@pytest.mark.parametrize("spec, kind", list(_refused_cells()))
def test_unsupported_resize_files_are_refused_or_right(build, spec, kind):
    """Files refused today, which must stay refused or be answered right:
    opset 10 Resize in nearest and in linear mode, a non-integer scale (2.25,
    a scale whose rounding and truncation agree), a downscale (0.5), sizes
    that are not a multiple of the input, and opsets 18 and 19, which the
    converter refuses at load with ``NotImplementedError``. Catches a refusal
    dropped in favour of block replication (the integer check of the scales,
    the multiple check of the sizes, a rule for the opset 10 module that does
    not check its mode) and a load failure that is not a clear refusal (any
    other exception type fails the test)."""
    _grid_cell(build, spec, kind)


@pytest.mark.parametrize("kind", ["Star", "Zono", "Box", "Hexatope",
                                  "Octatope"])
@pytest.mark.parametrize("shaped", [False, True], ids=["unshaped", "input_shape"])
def test_flat_sets_on_a_resize_model_are_refused_or_right(build, kind, shaped):
    """A flat Star, Zono, Box, Hexatope or Octatope (entries in the tensor's
    channel, height, width order; dyadic bounds, so the Hexatope and the
    Octatope are exact) given to a nearest Resize model by (2, 3), with and
    without ``input_shape``. Refused today with ``NotImplementedError``
    (``upsample_star`` and ``upsample_zono`` for a flat Star and Zono, the
    dispatcher's Box, Hexatope and Octatope lanes, which have no Resize
    branch); any other exception fails. Catches a flat set answered with the
    input's entry count (a lane that takes Resize for a shape identity) or
    with the pixels copied in the stored image order."""
    spec = Rz(ctm="asymmetric", nearest_mode="floor", **_scaled((2, 3)))
    path = build(spec)
    lo, hi, points, outputs, shape = _grid_truth(spec, path)
    what = f"{spec.label()} on a flat {kind}"
    options = {"input_shape": spec.chw} if shaped else {}
    sets = _answer(path, flat_set(kind, lo, hi), "approx",
                   "reach_pytorch_model", what, **options)
    if sets is not None:
        # The true shape: a flat answer is compared in the tensor's order, an
        # image answer in its stored order (assert_covered with out_chw).
        _assert_holds(sets, outputs, shape, what)


@pytest.mark.parametrize("kind", ["ImageZono", "Zono", "Box", "Hexatope",
                                  "Octatope"])
def test_exact_method_on_a_set_with_no_exact_rule_is_refused(build, kind):
    """``method='exact'`` on an ImageZono, Zono, Box, Hexatope or Octatope
    through the model entry, on the supported nearest file (asymmetric,
    floor, scales (2, 3)): none of these families has an exact method, and
    the entry refuses before the walk (``ValueError`` today; whether that or
    ``NotImplementedError`` is the right type is decided in
    ``tests/contracts/test_reach_coverage.py``, so either passes here).
    Catches the method check of the model entry dropped or narrowed, so that
    ``exact`` on an ImageZono is answered by ``upsample_zono``, which never
    reads the method."""
    spec = Rz(ctm="asymmetric", nearest_mode="floor", **_scaled((2, 3)))
    path = build(spec)
    lo, hi = pixel_box(*spec.chw)
    start = (image_set(kind, lo, hi) if kind == "ImageZono"
             else flat_set(kind, lo, hi))
    with pytest.raises((ValueError, NotImplementedError)):
        reach_model(load_onnx(path), start, "exact",
                    f"{spec.label()} on {kind} exact")


# --------------------------------------------------------------------------- #
# Both entry points and the exact method, one file per defect                   #
# --------------------------------------------------------------------------- #
def _max_over(arrays, coef, const):
    """The largest value of ``coef . a + const`` over the predicate polytope
    ``C a <= d, lb <= a <= ub`` of a returned set (``arrays`` as
    ``membership.rows`` reads them), by scipy's HiGHS called directly; inf
    when unbounded. A zero ``coef`` gives ``const`` exactly, with no solve."""
    _, _, C, d, lb, ub = arrays
    if not np.any(coef):
        return float(const)
    bounds = [(float(lo_) if np.isfinite(lo_) else None,
               float(hi_) if np.isfinite(hi_) else None)
              for lo_, hi_ in zip(lb.reshape(-1), ub.reshape(-1))]
    rows_ = C.size > 0
    res = linprog(-np.asarray(coef, dtype=np.float64),
                  A_ub=C if rows_ else None,
                  b_ub=d.reshape(-1) if rows_ else None,
                  bounds=bounds, method="highs")
    if res.status == 3:
        return math.inf
    if res.status != 0:
        raise RuntimeError(f"the LP over a returned set stopped: {res.message}")
    return float(-res.fun) + float(const)


def _assert_inside_image(sets, spec, input_star, what):
    """Exactness in the direction "no point of the result without a
    preimage", proved by linear programs, for a file whose resize is a block
    copy (``_is_block``) and an input ``ImageStar`` built by this test (one
    generator per pixel of positive width). A block copy is injective, so a
    returned set R lies in the true image of the input set S exactly when,
    over R's own predicate polytope: every output equals the other copies of
    its input pixel; a pixel of zero width keeps its value; and the input
    pixels read off the output satisfy every row and bound of S. Each is the
    largest value of a linear function over R (``_max_over``), compared with
    its bound with the solver's slack ``LP_TOL``. Catches an exact result that
    drops or loosens the input set's rows, or mixes pixels."""
    matrix = _resize_matrix(spec)
    source = [next(iter(row)) for row in matrix]
    shape = _resize_shape(spec)
    image_of = np.empty(len(source), dtype=int)     # image sets: stored order
    image_of[hwc_index(*shape[1:])] = np.arange(len(source))
    first = {}
    for o, i in enumerate(source):
        first.setdefault(i, o)
    c_in, G_in, C_in, d_in, lb_in, ub_in = rows(input_star)
    d_in, lb_in, ub_in = (v.reshape(-1) for v in (d_in, lb_in, ub_in))
    nonzero = G_in != 0
    if np.any(nonzero.sum(0) != 1) or np.any(nonzero.sum(1) > 1):
        pytest.fail(f"{what}: the input set is not one generator per pixel")
    live = np.argmax(nonzero, axis=0)
    gain = G_in[live, np.arange(G_in.shape[1])]
    fixed = np.flatnonzero(nonzero.sum(1) == 0)
    for s in sets:
        # entry of the set holding output o: stored (height, width, channel)
        # order for an image set, the tensor's own order for a flat one
        stored_of = (image_of if isinstance(s, (ImageStar, ImageZono))
                     else np.arange(len(source)))
        # the entry that carries input pixel k (k in the input's stored order)
        carrier = stored_of[[first[i] for i in hwc_index(*spec.chw)]]
        arrays = rows(s)
        c, G = arrays[0], arrays[1]
        checks = []                                 # (coef, const, bound, words)
        for o, i in enumerate(source):
            p, q = stored_of[o], stored_of[first[i]]
            if p != q:
                checks += [(G[p] - G[q], c[p] - c[q], 0.0,
                            f"output {o} differs from its copy {first[i]}"),
                           (G[q] - G[p], c[q] - c[p], 0.0,
                            f"output {o} differs from its copy {first[i]}")]
        x_c, x_G = c[carrier], G[carrier]           # input pixels, stored order
        for k in fixed:
            checks += [(x_G[k], x_c[k] - c_in[k], 0.0,
                        f"zero-width input entry {k} moves"),
                       (-x_G[k], c_in[k] - x_c[k], 0.0,
                        f"zero-width input entry {k} moves")]
        a_G = x_G[live] / gain[:, None]             # input predicate values
        a_c = (x_c[live] - c_in[live]) / gain
        for k in range(C_in.shape[0]):
            checks.append((C_in[k] @ a_G, C_in[k] @ a_c, float(d_in[k]),
                           f"input row {k} broken"))
        for j in range(a_G.shape[0]):
            if np.isfinite(ub_in[j]):
                checks.append((a_G[j], a_c[j], float(ub_in[j]),
                               f"input variable {j} above its bound"))
            if np.isfinite(lb_in[j]):
                checks.append((-a_G[j], -a_c[j], -float(lb_in[j]),
                               f"input variable {j} below its bound"))
        for coef, const, bound, words in checks:
            top = _max_over(arrays, coef, const)
            slack = LP_TOL * max(1.0, abs(bound), float(np.abs(coef).sum()))
            if top > bound + slack:
                raise AssertionError(
                    f"{what}: a point of a returned {_describe(s)} has no "
                    f"preimage in the input set: {words} (reaches {top:.6g}, "
                    f"bound {bound:.6g})")


_FLOOR = dict(ctm="asymmetric", nearest_mode="floor")
_ALL_ENTRIES = (("ImageStar", "reach_pytorch_model", "exact"),
                ("ImageStar", "NeuralNetwork.reach", "approx"),
                ("ImageStar", "NeuralNetwork.reach", "exact"),
                ("ImageZono", "NeuralNetwork.reach", "approx"))
_EXACT_ONLY = (("ImageStar", "reach_pytorch_model", "exact"),)
# Nearest: the supported files only, under the exact method through the
# model entry (scales, sizes, the upgraded opset 9 Upsample) and through
# NeuralNetwork.reach with each answering set type (one file). The marked
# nearest defects take the same kernel under every entry and method; they
# are marked once in the grid, sizes and crop cells. RESIZE-1 likewise
# (NeuralNetwork.reach calls reach_pytorch_model, and the kernel does not
# read the method): one linear file under the exact method, the evidence of
# the exact ImageStar row; the approx grid holds the other marks.
_ENTRY_SPECS = [
    (Rz(ctm="half_pixel", nearest_mode="round_prefer_floor", **_scaled((2, 3))),
     None, _ALL_ENTRIES[:2] + _ALL_ENTRIES[3:]),
    (Rz(**_FLOOR, sizes=(1, 2, 6, 8)), None, _EXACT_ONLY),
    (Rz(op="Upsample", opset=9, **_scaled((2, 3))), None, _EXACT_ONLY),
    (Rz(mode="linear", ctm="half_pixel", **_scaled((2, 2))), "RESIZE-1",
     _EXACT_ONLY),
]


def _entry_cells():
    for spec, defect, entries in _ENTRY_SPECS:
        for kind, entry, method in entries:
            yield pytest.param(spec, kind, entry, method,
                               id=f"{spec.label()}-{kind}-{entry}-{method}",
                               marks=(_bug(defect),) if defect else ())


@pytest.mark.parametrize("spec, kind, entry, method", list(_entry_cells()))
def test_resize_through_both_entry_points_and_the_exact_method(build, spec,
                                                               kind, entry,
                                                               method):
    """The supported nearest files (by scales, by sizes, and the upgraded
    opset 9 Upsample) as an ImageStar through ``reach_pytorch_model`` with
    the exact method, and one of them through ``NeuralNetwork.reach`` as an
    ImageStar (approx) and an ImageZono; one linear file (RESIZE-1) under
    the exact method (the grid cells use ``reach_pytorch_model`` with the
    approx method). A supported file must answer, and under the exact
    method every point of the result must have a preimage in the input set
    (``_assert_inside_image``). Catches an exact result that holds points
    outside the true image (rows dropped or loosened, pixels mixed), and a
    defect that only one entry point or one method shows."""
    sets = _grid_cell(build, spec, kind, method, entry,
                      must_answer=_must_answer(spec))
    if sets is not None and method == "exact" and _is_block(spec):
        lo, hi = _grid_truth(spec, build(spec))[:2]
        _assert_inside_image(sets, spec, image_set(kind, lo, hi),
                             f"{spec.label()} via {entry} exact")


# --------------------------------------------------------------------------- #
# Composition, several input sets, a binding row, a zero-width pixel            #
# --------------------------------------------------------------------------- #
def _chain_cells():
    # RESIZE-1 in a chain takes the failing path of the linear grid's marked
    # cells; only the supported nearest files are chained.
    for spec, defect in (
            (Rz(ctm="half_pixel", nearest_mode="round_prefer_floor",
                wrap="chain", **_scaled((2, 3))), None),
            (Rz(**_FLOOR, wrap="chain", sizes=(1, 2, 6, 12)), None)):
        for kind in IMAGE_KINDS:
            yield pytest.param(spec, kind, id=f"{spec.label()}-{kind}",
                               marks=(_bug(defect),) if defect else ())


@pytest.mark.parametrize("spec, kind", list(_chain_cells()))
def test_resize_between_two_convolutions_holds_every_output(build, spec, kind):
    """Conv 1x1 (mixing the two channels), then the Resize, then a Conv with a
    2x1 kernel and three output channels, so the set the Resize emits is
    consumed downstream in its stored layout. The truth is the formula
    composed with the two convolutions in float64 (dyadic weights; onnxruntime
    must agree). The supported nearest files by scales and by sizes.
    Catches a resize result whose rows or pixel order are wrong in a way the
    next layer reads."""
    _grid_cell(build, spec, kind, must_answer=_must_answer(spec))


@pytest.mark.parametrize("precompute", [False, "ibp", "zono"])
def test_nearest_resize_then_relu_with_precomputed_bounds_holds_every_output(
        build, precompute):
    """A supported nearest Resize (asymmetric, floor, scales (2, 3)), then a
    Relu, then a 1x1 Conv, on an ImageStar, approx, with ``precompute_bounds``
    False (the Relu computes its own bounds), ``"ibp"`` (the interval
    pre-pass has no Resize rule and stops trusting bounds after it) and
    ``"zono"`` (the zonotope pre-pass runs ``upsample_zono`` on the ImageZono
    of the box, and its sets give the bounds the Relu of the ImageStar reach
    intersects with its own). Every pixel interval contains [-1/4, 1/4] and
    has ends of its own: the Relu drops pre-pass bounds that miss its own
    enclosure anywhere, so only bounds that overlap everywhere can do harm,
    and here bounds of other pixels always overlap and clip the triangle of
    a crossing neuron. Catches a wrong ``upsample_zono`` (pixels misplaced)
    that only the bounds of the next layer read under the ImageStar row, and
    a bound rule for Resize that hands the Relu bounds of other pixels."""
    spec = Rz(ctm="asymmetric", nearest_mode="floor", wrap="relu_conv",
              **_scaled((2, 3)))
    path = build(spec)
    index = np.arange(int(np.prod(spec.chw)), dtype=np.float64).reshape(spec.chw)
    lo = -0.25 - (index % 7) / 8.0
    hi = 0.25 + (index % 5) / 8.0 + (index % 3) / 16.0
    points = samples(lo, hi)
    outputs, shape = _truth(spec, path, points)
    what = f"{spec.label()} on ImageStar approx, precompute_bounds={precompute}"
    sets = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                       "approx", what, precompute_bounds=precompute)
    _assert_holds(sets, outputs, shape, what)


@_bug("RESIZE-1")
def test_linear_resize_then_relu_with_zonotope_bounds_holds_every_output(build):
    """A linear Resize (half_pixel, scales (2, 3)), then a Relu, then a 1x1
    Conv, on an ImageStar, approx, with ``precompute_bounds="zono"``: the
    zonotope pre-pass runs ``upsample_zono`` on the ImageZono of the box and
    hands its bounds to the Relu of the ImageStar reach. The box is the one
    of the nearest cell above (every pixel interval holds [-1/4, 1/4] and has
    ends of its own, so pre-pass bounds of other pixels overlap the Relu's
    own and clip its triangles). The promise: refused, or every true output a
    member. Marked RESIZE-1 today (the ImageStar kernel is itself the block
    copy). Catches, once RESIZE-1 is fixed in ``upsample_star`` alone, the
    block copy left in ``upsample_zono`` whose bounds the pre-pass brings
    into the ImageStar row: with ``upsample_star`` replaced by the exact
    linear map of this file, the cell holds every output under
    ``precompute_bounds=False`` and ``"ibp"`` and misses most of them under
    ``"zono"``."""
    spec = Rz(mode="linear", ctm="half_pixel", wrap="relu_conv",
              **_scaled((2, 3)))
    path = build(spec)
    index = np.arange(int(np.prod(spec.chw)), dtype=np.float64).reshape(spec.chw)
    lo = -0.25 - (index % 7) / 8.0
    hi = 0.25 + (index % 5) / 8.0 + (index % 3) / 16.0
    points = samples(lo, hi)
    outputs, shape = _truth(spec, path, points)
    what = f"{spec.label()} on ImageStar approx, precompute_bounds=zono"
    sets = _answer(path, image_set("ImageStar", lo, hi), "approx",
                   "reach_pytorch_model", what, precompute_bounds="zono")
    if sets is not None:
        _assert_holds(sets, outputs, shape, what)


def _split_box():
    """A (2, 3, 4) box, positive except two pixels that straddle zero (so an
    exact Relu splits it into four sets) and one pixel of zero width."""
    index = np.arange(24, dtype=np.float64).reshape(CHW)
    lo = 0.25 + (index % 7) / 8.0
    hi = lo + 0.25 + (index % 3) / 8.0
    lo[0, 0, 1], lo[1, 2, 3] = -0.5, -0.75
    hi[1, 1, 1] = lo[1, 1, 1]
    return lo, hi


@pytest.mark.parametrize("spec", [
    pytest.param(Rz(wrap="relu", **_FLOOR, **_scaled((2, 3))),
                 id="asymmetric-floor-2x3"),
    pytest.param(Rz(wrap="relu", ctm="pytorch_half_pixel",
                    nearest_mode="round_prefer_ceil", sizes=(1, 2, 9, 4)),
                 id="pytorch_half_pixel-round_prefer_ceil-sizes9x4"),
])
def test_resize_after_an_exact_relu_split_holds_every_output(build, spec):
    """Relu then a supported nearest Resize, an ImageStar under the exact
    method: the Relu splits the box (two pixels straddle zero) into four sets,
    which reach the Resize together; one pixel has zero width (no predicate
    variable). Every returned set must also lie in the image of the Relu's
    exact output, the box of the clipped bounds (``_assert_inside_image``).
    Catches a resize that returns the result of one input set only, or
    mishandles a pixel with no generator, or drops the rows a split adds."""
    path = build(spec)
    lo, hi = _split_box()
    points = samples(lo, hi)
    outputs, shape = _truth(spec, path, points)
    what = f"{spec.label()} after an exact Relu split"
    sets = _answer(path, image_set("ImageStar", lo, hi), "exact",
                   "reach_pytorch_model", what)
    assert sets is not None, f"{what}: refused a supported block copy"
    assert len(sets) == 4, f"{what}: {len(sets)} sets for four Relu patterns"
    _assert_holds(sets, outputs, shape, what)
    _assert_inside_image(sets, spec, image_set("ImageStar", np.maximum(lo, 0.0),
                                               np.maximum(hi, 0.0)), what)


@pytest.mark.parametrize("method", ["approx", "exact"])
@pytest.mark.parametrize("spec", [
    pytest.param(Rz(**_FLOOR, **_scaled((2, 3))), id="asymmetric-floor-2x3"),
])
def test_resize_of_a_star_with_a_binding_row_holds_its_points(build, spec, method):
    """A supported nearest Resize on an ImageStar whose predicate is cut by a
    row that is not a box row (one corner of the box removed, the bounding box
    unchanged). The member points include points on the cut face, where the
    row binds. Under the exact method every returned set must lie in the
    image of the input set, proved by linear programs over the result's own
    rows (``_assert_inside_image``): the cut-off corner, and any point that
    breaks the row by any amount, has no preimage. Catches a resize that
    rewrites the rows (members escape) and, under the exact method, one that
    drops or loosens them (the box image or more is returned)."""
    path = build(spec)
    lo, hi = pixel_box(*spec.chw)
    star, points = constrained_image_star(lo, hi)
    outputs, shape = _truth(spec, path, points)
    what = f"{spec.label()} on a constrained ImageStar {method}"
    sets = _answer(path, star, method, "reach_pytorch_model", what)
    assert sets is not None, f"{what}: refused a supported block copy"
    _assert_holds(sets, outputs, shape, what)
    if method == "exact":
        _assert_inside_image(sets, spec, star, what)


# --------------------------------------------------------------------------- #
# A false UNSAT end to end                                                      #
# --------------------------------------------------------------------------- #
def _separated_box(chw):
    """Pixel i (flat, channel-height-width order) in [2i, 2i + 1/4]: any
    output that reads another pixel, or mixes in one with a weight of 1/4 or
    more, leaves the interval of the pixel a block copy gives it."""
    lo = 2.0 * np.arange(int(np.prod(chw)), dtype=np.float64).reshape(chw)
    return lo, lo + 0.25


def _end_to_end_cells():
    # The false UNSAT of RESIZE-1 (verify/resize.md section 6) comes from the
    # failing path the linear grid marks; only the block copy is checked here.
    for spec, defect, kinds in (
            (Rz(**_FLOOR, wrap="flatten", **_scaled((2, 3))), None,
             (("ImageStar", "approx"), ("ImageZono", "approx"),
              ("ImageStar", "exact"))),):
        for kind, method in kinds:
            yield pytest.param(spec, kind, method,
                               id=f"{spec.label()}-{kind}-{method}",
                               marks=(_bug(defect),) if defect else ())


@pytest.mark.parametrize("spec, kind, method", list(_end_to_end_cells()))
def test_no_false_unsat_after_a_resize(build, spec, kind, method):
    """Resize then Flatten, reached, then ``verify_specification`` against the
    unsafe regions y_k >= m_k and y_k <= M_k for every output k, where
    [m_k, M_k] is the exact range of y_k over the input box (the formula in
    fractions; the data are dyadic, so the ranges are exact in float64 and
    no pad is needed). Each region holds every true output, so no verdict
    may be UNSAT. A refusal keeps the promise, except for the supported block
    copy. Catches the false UNSAT that RESIZE-1 produces (``verify/resize.md``
    section 6), and on the block copy a result whose pixels are misplaced, or
    that lies wholly beyond one end of an output's exact range by any amount
    (a region is then disjoint from it; a pad would forgive a miss smaller
    than the pad)."""
    path = build(spec)
    lo, hi = _separated_box(spec.chw)
    points = samples(lo, hi)
    outputs, shape = _truth(spec, path, points)
    lo_f = [Fraction(v) for v in lo.reshape(-1)]
    hi_f = [Fraction(v) for v in hi.reshape(-1)]
    ranges = []
    for row in _resize_matrix(spec):
        ranges.append((sum(min(w * lo_f[i], w * hi_f[i]) for i, w in row.items()),
                       sum(max(w * lo_f[i], w * hi_f[i]) for i, w in row.items())))
    for y in outputs:                    # the regions hold the sampled outputs
        if not all(float(m) <= v <= float(M) for v, (m, M) in zip(y, ranges)):
            pytest.fail(f"{spec.label()}: a true output lies outside its exact "
                        f"range: the reference is wrong")
    what = f"{spec.label()} on {kind} {method}"
    sets = _answer(path, image_set(kind, lo, hi), method,
                   "reach_pytorch_model", what)
    if sets is None:
        assert not _must_answer(spec), f"{what}: refused a supported block copy"
        return
    n = len(ranges)
    for k, (m, M) in enumerate(ranges):
        unit = np.zeros((1, n))
        unit[0, k] = 1.0
        for G, g, words in ((-unit, -float(m), f"y[{k}] >= {float(m)}"),
                            (unit, float(M), f"y[{k}] <= {float(M)}")):
            verdict = verify_specification(sets, HalfSpace(G, np.array([[g]]))).verdict
            assert verdict != "UNSAT", (
                f"{what}: UNSAT on the unsafe region {words}, which holds "
                f"every true output (exact range of y[{k}] is "
                f"[{float(m)}, {float(M)}])")


# --------------------------------------------------------------------------- #
# RESIZE-7: the converted module against onnxruntime                            #
# --------------------------------------------------------------------------- #
def _conversion_gap(path, spec):
    """Largest |module(x) - onnxruntime(x)| / max(1, |y|) over the grid
    points, for the torch module ``load_onnx`` returns (float32, as loaded)."""
    lo, hi = pixel_box(*spec.chw)
    points = samples(lo, hi)
    replayed = replay_onnx(path, points, (1,) + tuple(spec.chw))
    module = load_onnx(path)
    gap = 0.0
    with torch.no_grad():
        for x, y in zip(points, replayed):
            out = module(torch.tensor(x.reshape((1,) + tuple(spec.chw)),
                                      dtype=torch.float32))
            out = out.detach().numpy().astype(np.float64).reshape(-1)
            if out.size != y.size:
                return math.inf
            gap = max(gap, float(np.max(np.abs(out - y)
                                        / np.maximum(1.0, np.abs(y)))))
    return gap


_AGREEING = [Rz(**_FLOOR, **_scaled((2, 3))),
             Rz(ctm="half_pixel", nearest_mode="round_prefer_floor",
                **_scaled((3, 3))),
             Rz(mode="linear", ctm="half_pixel", **_scaled((2, 2))),
             Rz(mode="linear", ctm="align_corners", **_scaled((2, 3))),
             Rz(mode="cubic", ctm="half_pixel", **_scaled((2, 2)))]
# RESIZE-7 fails at one place today (from_onnx does not exist), so one
# differing file is marked; the linear and cubic conversions that differ
# (asymmetric, cubic_coeff_a -0.5, exclude_outside 1, the upgraded opset 9
# Upsample) are listed in the registry row.
_DIFFERING = [Rz(ctm="asymmetric", nearest_mode="ceil", **_scaled((2, 3)))]


@pytest.mark.parametrize("spec", [pytest.param(s, id=s.label()) for s in _AGREEING])
def test_the_converted_module_of_these_resize_files_agrees_with_onnxruntime(build, spec):
    """The module ``load_onnx`` returns for these files computes what
    onnxruntime computes, to float32 rounding (``ORT_AGREEMENT``). It is the
    passing neighbour of the RESIZE-7 cells, with the same comparison. The
    linear and cubic files (half_pixel, align_corners) are the premise of
    RESIZE-1's own fix: their conversion is right, so a load-time comparison
    (PLAN 12b) cannot replace the mode check of the reach rule. Catches a
    converter that maps an attribute to the wrong torch mode."""
    gap = _conversion_gap(build(spec), spec)
    assert gap <= ORT_AGREEMENT, (
        f"{spec.label()}: the converted module differs from onnxruntime by "
        f"{gap:.3g}")


@pytest.mark.parametrize("spec, differs", [
    pytest.param(s, True, id=f"{s.label()}-differs") for s in _DIFFERING] + [
    pytest.param(s, False, id=f"{s.label()}-agrees") for s in _AGREEING[:1]])
# Step 12 without 12b (from_onnx without the comparison) turns the differing
# items into unabsorbed DID NOT RAISE failures and the agreeing one into an
# XPASS; the marker then changes to raises=DID_NOT_RAISE on the differing ones.
@known_bug("12b", "NeuralNetwork.from_onnx with the load-time onnxruntime "
                  "check (DESIGN section 9) does not exist; load_onnx returns "
                  "a converted Resize that differs from onnxruntime without "
                  "any check (RESIZE-7)", raises=AttributeError)
def test_from_onnx_refuses_a_resize_conversion_that_differs_from_onnxruntime(
        build, spec, differs):
    """DESIGN section 9: ``from_onnx(..., check_onnxruntime=True)`` compares
    the conversion with onnxruntime and raises ``ModelError`` (a
    ``ValueError``) beyond float32 rounding. The differing file is a RESIZE-3
    rule (asymmetric with ceil); an agreeing file must load. The setup check
    measures the gap first."""
    path = build(spec)
    gap = _conversion_gap(path, spec)
    if (gap > ORT_AGREEMENT) != differs:
        pytest.fail(f"{spec.label()}: the conversion gap is {gap:.3g}, the "
                    f"cell expects it to {'differ' if differs else 'agree'}")
    if differs:
        with pytest.raises(ValueError):
            NeuralNetwork.from_onnx(path, check_onnxruntime=True)
    else:
        assert isinstance(NeuralNetwork.from_onnx(path, check_onnxruntime=True),
                          NeuralNetwork)

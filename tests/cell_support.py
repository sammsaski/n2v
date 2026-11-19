"""Shared support for soundness cells: input sets of every type, sample
points, reference models and replays, reach wrappers and the coverage check.

A cell is one (operation, set type, method, option); its test pushes an input
set through n2v and checks that every true output is in some returned set
(PROTOCOL.md section 4, ``tests/README.md`` "A soundness cell"). This module
holds the parts every cell needs, so that each is written, documented and
self-tested once (``tests/test_cell_support.py``). The builders and reach
wrappers call n2v constructors and entry points; nothing that produces a
reference value does: true outputs come from onnxruntime or from torch in
float64, and membership from ``tests/_oracles``.

Layout, the one convention every helper here keeps:

- a network tensor of an image is (channels, height, width), row-major;
- an image set (``ImageStar``, ``ImageZono``) stores its entries in
  (height, width, channels) order and carries ``height``, ``width`` and
  ``num_channels``;
- a flat set that stands for a tensor is in the tensor's own row-major order
  (channel, height, width for an image).

Bounds, sample points and outputs are always passed in the network's order;
the helpers convert explicitly where a set stores another order
(:func:`image_set`, :func:`assert_covered` with ``out_chw``,
:func:`hwc_index`). A layout bug is exactly a confusion of the two orders.

A complete cell::

    def test_relu_on_an_image_star_holds_every_model_output(tmp_path):
        \"\"\"Exact ReLU on a (2, 3, 5) ImageStar: catches a dropped piece and a
        result whose pixels are stored in the wrong order.\"\"\"
        shape = [1, 2, 3, 5]
        path = save_onnx(tmp_path, [helper.make_node("Relu", ["X"], ["Y"])],
                         shape, shape)
        lo, hi = pixel_box(2, 3, 5)
        sets = reach_model(load_onnx(path), image_set("ImageStar", lo, hi),
                           "exact", "Relu on ImageStar")
        assert_covered(sets, replay_onnx(path, samples(lo, hi), shape),
                       "Relu on ImageStar", out_chw=(2, 3, 5))
"""

import copy
import hashlib
import os

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.reach import reach_pytorch_model
from n2v.sets import Box, Hexatope, Octatope, Star, Zono
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from tests._oracles.membership import LP_TOL, hwc, uncovered

FLAT_KINDS = ("Box", "Star", "Zono", "Hexatope", "Octatope")
IMAGE_KINDS = ("ImageStar", "ImageZono")

N_SAMPLES = 128

__all__ = [
    "FLAT_KINDS", "IMAGE_KINDS", "N_SAMPLES", "flat_set", "image_set",
    "pixel_box", "flat_box", "samples", "constrained_star",
    "constrained_image_star", "coupled_star", "split_cover",
    "image_split_cover", "save_onnx",
    "replay_onnx", "replay_torch", "reach_model", "reach_one_layer",
    "assert_covered", "hwc_index",
]


# --------------------------------------------------------------------------- #
# Set builders                                                                  #
# --------------------------------------------------------------------------- #
def _bounds(lo, hi):
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError(f"lo has shape {lo.shape}, hi has {hi.shape}")
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("the bounds must be finite")
    if np.any(lo > hi):
        raise ValueError("lo is above hi somewhere")
    return lo, hi


def flat_set(kind, lo, hi):
    """The box [lo, hi] as a flat set of ``kind`` (one of :data:`FLAT_KINDS`).

    ``lo`` and ``hi`` may have any shape; the set's entries are their
    row-major flattening, the network's own order. Each kind is built by its
    own constructor: ``Box(lo, hi)``, ``Star.from_bounds`` and
    ``Zono.from_bounds`` on column vectors (both round the half-widths
    outward), ``Hexatope.from_bounds`` and ``Octatope.from_bounds`` on flat
    vectors (which round to nearest: exact for dyadic bounds, so use dyadic
    bounds). A zero-width coordinate is legal for every kind (Box and Star
    drop its generator, the others keep a zero one)."""
    lo, hi = _bounds(lo, hi)
    lo, hi = lo.reshape(-1), hi.reshape(-1)
    if kind == "Box":
        return Box(lo, hi)
    if kind in ("Star", "Zono"):
        cls = Star if kind == "Star" else Zono
        return cls.from_bounds(lo.reshape(-1, 1), hi.reshape(-1, 1))
    if kind in ("Hexatope", "Octatope"):
        cls = Hexatope if kind == "Hexatope" else Octatope
        return cls.from_bounds(lo, hi)
    raise ValueError(f"unknown flat kind {kind!r}; one of {FLAT_KINDS}")


def image_set(kind, lo_chw, hi_chw):
    """The box [lo_chw, hi_chw] of a (channels, height, width) image as an
    ``ImageStar`` or ``ImageZono``. The bounds are transposed to the set's
    stored (height, width, channels) order before ``from_bounds`` and the set
    carries ``height``, ``width`` and ``num_channels`` of the image. A
    zero-width pixel is legal (``ImageStar`` gives it no predicate variable,
    ``ImageZono`` a zero generator)."""
    lo_chw, hi_chw = _bounds(lo_chw, hi_chw)
    if lo_chw.ndim != 3:
        raise ValueError(f"image bounds are (C, H, W); got shape {lo_chw.shape}")
    channels, height, width = lo_chw.shape
    cls = {"ImageStar": ImageStar, "ImageZono": ImageZono}.get(kind)
    if cls is None:
        raise ValueError(f"unknown image kind {kind!r}; one of {IMAGE_KINDS}")
    return cls.from_bounds(lo_chw.transpose(1, 2, 0), hi_chw.transpose(1, 2, 0),
                           height, width, channels)


# --------------------------------------------------------------------------- #
# Boxes and sample points                                                       #
# --------------------------------------------------------------------------- #
def pixel_box(channels, height, width, seed=0):
    """A different dyadic interval for every pixel, as (C, H, W) arrays
    ``(lo, hi)``: ``lo`` in [-2, 1.75] in steps of 1/4, width in [1/4, 5/4],
    so a result that reads the pixels in the wrong order misses the outputs."""
    rng = np.random.default_rng(seed)
    lo = rng.integers(-8, 8, size=(channels, height, width)) / 4.0
    return lo, lo + rng.integers(1, 6, size=(channels, height, width)) / 4.0


def flat_box(n, seed=0, zero_width=False):
    """Dyadic bounds ``(lo, hi)`` of a flat vector of ``n`` entries, in
    multiples of 1/8. With ``zero_width`` the coordinate ``n // 2`` has
    ``lo == hi`` (a nonzero value). The other coordinates, in order, cycle
    through three regimes: straddling zero (``lo < 0 < hi``), strictly
    positive, strictly negative; so ``n >= 3`` (``n >= 4`` with
    ``zero_width``) has all three."""
    if n < (2 if zero_width else 1):
        raise ValueError("too few coordinates")
    rng = np.random.default_rng(seed)
    lo, hi = np.zeros(n), np.zeros(n)
    flat = int(n // 2) if zero_width else None
    others = [i for i in range(n) if i != flat]
    for rank, i in enumerate(others):
        width = rng.integers(1, 9) / 8.0
        if rank % 3 == 0:                                   # straddles zero
            lo[i], hi[i] = -rng.integers(1, 9) / 8.0, rng.integers(1, 9) / 8.0
        elif rank % 3 == 1:                                 # positive
            lo[i] = rng.integers(1, 9) / 8.0
            hi[i] = lo[i] + width
        else:                                               # negative
            hi[i] = -rng.integers(1, 9) / 8.0
            lo[i] = hi[i] - width
    if flat is not None:
        lo[flat] = hi[flat] = rng.choice([-1, 1]) * rng.integers(1, 9) / 8.0
    return lo, hi


def samples(lo, hi, n=N_SAMPLES, seed=0):
    """``n`` points of the box [lo, hi], each an array of ``lo``'s shape: the
    two extreme corners ``lo`` and ``hi``, then alternately a point of the
    quarter grid (each coordinate at lo + (hi - lo) k / 4, k in 0..4) and a
    random corner. All dyadic when ``lo`` and ``hi`` are."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    rng = np.random.default_rng(seed)
    points = [lo, hi]
    for k in range(n - 2):
        steps = (4 * rng.integers(0, 2, size=lo.shape) if k % 2
                 else rng.integers(0, 5, size=lo.shape))
        points.append(lo + (hi - lo) * steps / 4.0)
    return points


# --------------------------------------------------------------------------- #
# A Star and an ImageStar with a binding row that is not a box row              #
# --------------------------------------------------------------------------- #
def _constrained(lo, hi, n, seed):
    """Basis, rows and member points of a box star cut by one extra row, all
    in the flat order of ``lo``: ``(V, C, d, n_var, points)``."""
    lo, hi = lo.reshape(-1), hi.reshape(-1)
    live = np.flatnonzero(hi > lo)
    k = live.size
    if k < 2:
        raise ValueError("a constrained star needs two coordinates of "
                         "positive width")
    rng = np.random.default_rng(seed)
    centre = np.where(hi > lo, (lo + hi) / 2, lo)
    G = np.zeros((lo.size, k))
    G[live, np.arange(k)] = (hi - lo)[live] / 2
    chosen = np.sort(rng.choice(k, size=min(3, k), replace=False))
    weights = np.array([1.0] + list(rng.choice([1.0, 0.5, 0.25],
                                               size=chosen.size - 1)))
    weights *= rng.choice([-1.0, 1.0], size=chosen.size)
    row = np.zeros(k)
    row[chosen] = weights
    # The corner a = sign(row) is cut off; each coordinate still reaches both
    # ends of its interval, so the bounding box is unchanged.
    rhs = float(np.sum(np.abs(weights)) - np.min(np.abs(weights)))
    C = np.vstack([np.eye(k), -np.eye(k), row[None, :]])
    d = np.concatenate([np.ones(2 * k), [rhs]]).reshape(-1, 1)
    V = np.hstack([centre[:, None], G])

    lead = chosen[0]                     # weight +-1: the face point is dyadic
    points = []
    for attempt in range(200 * n):
        if len(points) >= n:
            break
        kind = attempt % 3
        if kind == 0:                                       # a corner
            a = rng.choice([-1.0, 1.0], size=k)
        else:                                               # the half grid
            a = rng.integers(-2, 3, size=k) / 2.0
        if kind == 2:                                       # on the cut face
            a[lead] = 0.0
            a[lead] = (rhs - row @ a) / row[lead]
            if abs(a[lead]) > 1:
                continue
        if row @ a <= rhs:
            points.append(centre + G @ a)
    if len(points) < n:
        raise ValueError("could not draw enough member points")
    return V, C, d, k, points


def constrained_star(lo, hi, n=N_SAMPLES, seed=0):
    """``(star, points)``: a ``Star`` whose predicate box [-1, 1]^k (one
    variable per coordinate of positive width, generator half the width)
    gives exactly [lo, hi], cut by one extra row of ``C a <= d`` over two or
    three predicate variables with dyadic coefficients (1, 1/2 or 1/4, random
    signs). The row cuts off one corner of the box: the set is strictly
    smaller than its bounding box, which is still [lo, hi], and the row binds.
    ``C`` holds the box rows ``+-I`` too, as ``Star.from_bounds`` does.

    ``points`` are at least ``n`` member points in ``lo``'s shape and order,
    computed without n2v: box corners and half-grid predicate points that keep
    the row, and points exactly on the cut face, mapped through the basis.
    Exact (dyadic) when ``lo`` and ``hi`` are."""
    lo, hi = _bounds(lo, hi)
    V, C, d, k, points = _constrained(lo, hi, n, seed)
    star = Star(V, C, d, -np.ones((k, 1)), np.ones((k, 1)))
    return star, [p.reshape(lo.shape) for p in points]


def constrained_image_star(lo_chw, hi_chw, n=N_SAMPLES, seed=0):
    """``(image_star, points)``: :func:`constrained_star` for an image. The
    ``ImageStar`` stores (height, width, channels); its predicate variables
    are the pixels of positive width in that stored order. ``points`` are
    member points as (C, H, W) arrays, the network's order, ready for a
    replay; convert with ``membership.hwc`` to compare with the set."""
    lo_chw, hi_chw = _bounds(lo_chw, hi_chw)
    if lo_chw.ndim != 3:
        raise ValueError(f"image bounds are (C, H, W); got shape {lo_chw.shape}")
    channels, height, width = lo_chw.shape
    V, C, d, k, points = _constrained(
        hwc(lo_chw, channels, height, width), hwc(hi_chw, channels, height,
                                                  width), n, seed)
    image = ImageStar(V, C, d, -np.ones((k, 1)), np.ones((k, 1)),
                      height=height, width=width, num_channels=channels)
    order = hwc_index(channels, height, width)
    tensors = []
    for p in points:
        chw = np.empty(p.size)
        chw[order] = p
        tensors.append(chw.reshape(channels, height, width))
    return image, tensors


def coupled_star(kind, lo, hi, kept, n=N_SAMPLES, seed=0):
    """``(star, points)``: the box [lo, hi] of a tensor as a ``Star`` (the
    tensor's row-major order) or an ``ImageStar`` (a (C, H, W) tensor, stored
    height, width, channels), one predicate variable a in [-1, 1] per entry
    of positive width, cut by two rows that tie entries together. It is the
    input for an operation that keeps some entries and drops or moves the
    others (a slice, a gather, one output of a split, one input of a concat):
    ``kept`` lists the row-major indices of the entries the operation keeps.

    - ``a_i + a_j <= 0`` ties the first kept entry i to the first entry j
      that is not kept. Entry i still reaches its whole interval, but its
      upper end only where a_j = -1: a result that keeps the row and loses
      the variable of entry j (or fixes it) misses those outputs. Left out
      when every entry of positive width is kept.
    - ``a_k - a_i <= 1/2`` ties the last kept entry k to the first. It cuts a
      corner off the projection on the kept entries, so it binds in the
      result: a result that is the bounding box of the kept entries is sound
      and not exact, one that puts the row on other entries misses outputs.

    The bounding box is still [lo, hi]. ``points`` are at least ``n`` member
    points in ``lo``'s shape and order (the network's), computed without n2v:
    corners and half-grid predicate points that keep both rows, points with
    a_i = 1 and a_j = -1, and points on the face of each row. Exact (dyadic)
    when ``lo`` and ``hi`` are. :func:`constrained_star` cuts one corner off
    any box; this builder is for the cells where which entries a row ties
    together decides whether a bug shows."""
    lo, hi = _bounds(lo, hi)
    if kind not in ("Star", "ImageStar"):
        raise ValueError(f"unknown kind {kind!r}; 'Star' or 'ImageStar'")
    if kind == "ImageStar" and lo.ndim != 3:
        raise ValueError(f"image bounds are (C, H, W); got shape {lo.shape}")
    shape = lo.shape
    flat_lo, flat_hi = lo.reshape(-1), hi.reshape(-1)
    kept = [int(p) for p in np.asarray(kept).reshape(-1)]
    if any(p < 0 or p >= flat_lo.size for p in kept):
        raise ValueError(f"kept holds an index outside 0..{flat_lo.size - 1}")
    order = (hwc_index(*shape) if kind == "ImageStar"
             else np.arange(flat_lo.size))
    live = [int(p) for p in order if flat_hi[p] > flat_lo[p]]
    var = {p: v for v, p in enumerate(live)}
    k = len(live)
    kept_live = [p for p in kept if p in var]
    if len(set(kept_live)) < 2:
        raise ValueError("a coupled star needs two kept entries of positive "
                         "width")
    removed_live = [p for p in live if p not in set(kept)]
    i, kk = kept_live[0], kept_live[-1]
    rows, rhs = [], []
    if removed_live:
        row = np.zeros(k)
        row[var[i]], row[var[removed_live[0]]] = 1.0, 1.0
        rows.append(row)
        rhs.append(0.0)
    row = np.zeros(k)
    row[var[kk]], row[var[i]] = 1.0, -1.0
    rows.append(row)
    rhs.append(0.5)
    C = np.vstack([np.eye(k), -np.eye(k)] + [r[None, :] for r in rows])
    d = np.concatenate([np.ones(2 * k), rhs]).reshape(-1, 1)
    centre = (flat_lo + flat_hi) / 2.0
    half = (flat_hi - flat_lo) / 2.0
    V = np.zeros((flat_lo.size, k + 1))
    V[:, 0] = centre[order]
    for v, p in enumerate(live):
        V[int(np.flatnonzero(order == p)[0]), v + 1] = half[p]
    rng = np.random.default_rng(seed)
    points = []
    for attempt in range(200 * n):
        if len(points) >= n:
            break
        if attempt % 4 == 0:
            a = rng.choice([-1.0, 1.0], size=k)
        else:
            a = rng.integers(-2, 3, size=k) / 2.0
        if attempt % 4 == 1 and removed_live:             # the escape witness
            a[var[i]], a[var[removed_live[0]]] = 1.0, -1.0
        elif attempt % 4 == 2 and removed_live:           # face of row 1
            a[var[removed_live[0]]] = -a[var[i]]
        elif attempt % 4 == 3:                            # face of row 2
            a[var[kk]] = min(1.0, a[var[i]] + 0.5)
        if np.all(C @ a <= d.ravel()):
            x = centre.copy()
            x[live] = centre[live] + half[live] * a
            points.append(x.reshape(shape))
    if len(points) < n:
        raise ValueError("could not draw enough member points")
    pred = (-np.ones((k, 1)), np.ones((k, 1)))
    if kind == "ImageStar":
        star = ImageStar(V, C, d, *pred, height=shape[1], width=shape[2],
                         num_channels=shape[0])
    else:
        star = Star(V, C, d, *pred)
    return star, points


# --------------------------------------------------------------------------- #
# Several input sets at once                                                    #
# --------------------------------------------------------------------------- #
def _part_sizes(parts):
    """Distinct positive integers, the smallest first, summing to a power of
    two: the parts' shares of the cut interval in dyadic steps."""
    sizes = list(range(1, parts))
    total = 2
    while total - sum(sizes) <= max(sizes, default=0):
        total *= 2
    return sizes + [total - sum(sizes)], total


def _cuts(lo, hi, parts):
    """The parts of the box [lo, hi] cut along its widest coordinate (the
    first one in row-major order), as ``(lo, hi)`` pairs of ``lo``'s shape."""
    if parts < 1:
        raise ValueError("parts must be at least 1")
    flat_lo, flat_hi = lo.reshape(-1), hi.reshape(-1)
    j = int(np.argmax(flat_hi - flat_lo))
    if flat_hi[j] == flat_lo[j]:
        raise ValueError("the box has no coordinate of positive width")
    if parts == 1:
        return [(lo.copy(), hi.copy())]
    sizes, total = _part_sizes(parts)
    ends = np.concatenate([[0], np.cumsum(sizes)]) / total
    points = flat_lo[j] + (flat_hi[j] - flat_lo[j]) * ends
    points[0], points[-1] = flat_lo[j], flat_hi[j]
    out = []
    for a, b in zip(points[:-1], points[1:]):
        part_lo, part_hi = flat_lo.copy(), flat_hi.copy()
        part_lo[j], part_hi[j] = a, b
        out.append((part_lo.reshape(lo.shape), part_hi.reshape(lo.shape)))
    return out


def split_cover(kind, lo, hi, parts=3):
    """``parts`` sets of the flat ``kind`` whose union is the box [lo, hi]: the
    box cut along its widest coordinate at dyadic points (exact when ``lo``
    and ``hi`` are dyadic), into pieces of distinct sizes, the smallest first
    (shares 1, 2, 5 of 8 for three parts). A layer that processes only the
    first set, or reuses its bounds for the others, misses most samples of
    the box."""
    lo, hi = _bounds(lo, hi)
    return [flat_set(kind, a, b) for a, b in _cuts(lo, hi, parts)]


def image_split_cover(kind, lo_chw, hi_chw, parts=3):
    """:func:`split_cover` for an image: the (C, H, W) box cut along its widest
    pixel, each part an ``ImageStar`` or ``ImageZono`` built by
    :func:`image_set`."""
    lo_chw, hi_chw = _bounds(lo_chw, hi_chw)
    return [image_set(kind, a, b) for a, b in _cuts(lo_chw, hi_chw, parts)]


# --------------------------------------------------------------------------- #
# Reference models                                                              #
# --------------------------------------------------------------------------- #
def _tensor_type(value):
    return getattr(TensorProto, value) if isinstance(value, str) else int(value)


def save_onnx(tmp_path, nodes, in_shape, out_shape, inits=None, name="m",
              opset=13, in_type=TensorProto.FLOAT, out_type=TensorProto.FLOAT,
              ir_version=None):
    """Save a graph ``X -> Y`` of ``nodes`` and return the file's path.

    ``in_shape``/``out_shape`` are the tensor shapes in the network's order
    (``out_shape=None`` leaves the output shape open); ``inits`` maps names to
    arrays, stored with their own dtype (pass float32 arrays for a float
    graph). ``opset`` is the default-domain opset written to the model (the
    cells use 7 to 18); ``in_type``/``out_type`` are ``TensorProto`` element
    types, as numbers or names (``"DOUBLE"``). The IR version is
    ``max(10, the least the opset needs)`` unless given: onnxruntime 1.26
    loads every opset from 7 to 21 at IR 10, and IR 3 (the least for opsets
    7 and 8) would need every initializer listed as a graph input. The model
    passes ``onnx.checker`` before it is written."""
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("X", _tensor_type(in_type),
                                       list(in_shape))],
        [helper.make_tensor_value_info(
            "Y", _tensor_type(out_type),
            None if out_shape is None else list(out_shape))],
        initializer=[numpy_helper.from_array(np.asarray(a), k)
                     for k, a in (inits or {}).items()])
    opsets = [helper.make_opsetid("", int(opset))]
    model = helper.make_model(graph, opset_imports=opsets)
    model.ir_version = (int(ir_version) if ir_version is not None
                        else max(10, helper.find_min_ir_version_for(opsets)))
    onnx.checker.check_model(model)
    path = os.path.join(str(tmp_path), f"{name}.onnx")
    onnx.save(model, path)
    return path


_ORT_DTYPES = {"tensor(float)": np.float32, "tensor(double)": np.float64,
               "tensor(float16)": np.float16}


def replay_onnx(path, points, in_shape):
    """The model's outputs on ``points`` by onnxruntime on CPU (one thread),
    each a flat float64 vector in the model's own output order (row-major
    over the first graph output). Each point, a flat vector or a tensor in
    the network's order, is reshaped to ``in_shape`` and cast to the model's
    input type: float32 for a float graph, exact for dyadic points."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    feed = session.get_inputs()[0]
    dtype = _ORT_DTYPES.get(feed.type)
    if dtype is None:
        raise ValueError(f"no replay for an input of type {feed.type}")
    return [np.asarray(session.run(None, {feed.name: np.asarray(
                x, dtype=np.float64).reshape(in_shape).astype(dtype)})[0],
                dtype=np.float64).reshape(-1) for x in points]


def replay_torch(module, points, in_shape):
    """The outputs of a torch module on ``points``, evaluated in float64 on a
    deep copy in eval mode with no gradient (the caller's module, its dtype
    and its training flag are not touched), each a flat float64 vector in the
    module's own output order (row-major; the first element when the module
    returns a tuple). Points are flat vectors or tensors in the network's
    order, reshaped to ``in_shape``."""
    model = copy.deepcopy(module).double().eval()
    outputs = []
    with torch.no_grad():
        for x in points:
            y = model(torch.as_tensor(
                np.asarray(x, dtype=np.float64).reshape(in_shape)))
            if isinstance(y, (tuple, list)):
                y = y[0]
            outputs.append(y.detach().cpu().numpy().astype(np.float64)
                           .reshape(-1))
    return outputs


# --------------------------------------------------------------------------- #
# Reach wrappers                                                                #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# A reach leaves its input sets as they were                                    #
# --------------------------------------------------------------------------- #
# The ranges a set caches about itself; a reach may fill them in.
_RANGE_CACHES = ("state_lb", "state_ub")


def _walk_contents(value, path, out, depth):
    if isinstance(value, np.ndarray) and value.dtype != object:
        out[path] = (value.shape, value.dtype.str,
                     hashlib.blake2b(value.tobytes(), digest_size=16).digest())
    elif isinstance(value, (list, tuple)):
        out[path] = (type(value).__name__, len(value))
        for k, item in enumerate(value):
            _walk_contents(item, f"{path}[{k}]", out, depth + 1)
    elif isinstance(value, dict):
        out[path] = ("dict", len(value))
        for key, item in value.items():
            _walk_contents(item, f"{path}[{key!r}]", out, depth + 1)
    elif hasattr(value, "__dict__") and not isinstance(value, type) \
            and depth < 5:
        for name, item in vars(value).items():
            if depth == 0 and name in _RANGE_CACHES:
                continue
            _walk_contents(item, f"{path}.{name}", out, depth + 1)
    else:
        out[path] = repr(value)


def _set_contents(s):
    """Everything ``s`` holds, by attribute path: arrays by shape, type and a
    digest of their bytes, lists, dicts and nested objects (a Hexatope's
    constraint system, a Star's outer zonotope) entry by entry. The cached
    ranges ``state_lb`` and ``state_ub`` are left out."""
    out = {}
    _walk_contents(s, type(s).__name__, out, 0)
    return out


def _assert_unchanged(before, sets, what):
    for k, (was, s) in enumerate(zip(before, sets)):
        now = _set_contents(s)
        changed = sorted(path for path in set(was) | set(now)
                         if was.get(path) != now.get(path))
        if changed:
            # pytest.fail and not an AssertionError: a known-bug marker of the
            # calling test names another failure and must not absorb this one.
            pytest.fail(
                f"{what}: the reach changed input set {k} of {len(sets)} "
                f"({type(s).__name__}): {', '.join(changed[:6])}. A set that "
                f"feeds two nodes of a graph (a residual connection) is read "
                f"again after the first reach, and the caller may reach the "
                f"same set again")


def _as_list(input_sets):
    if isinstance(input_sets, (list, tuple)):
        sets = list(input_sets)
    else:
        sets = [input_sets]
    if not sets:
        raise ValueError("no input set given")
    return sets


def reach_model(model, input_sets, method, what, **options):
    """Reach sets of a model through ``n2v.nn.reach.reach_pytorch_model``.

    ``input_sets`` is one set or a list. The entry point takes one input set,
    so each set of a list is reached on its own and the results are joined in
    order. ``options`` go to the entry point unchanged, for example
    ``precompute_bounds="ibp"`` or ``"zono"`` (the bound pre-pass),
    ``input_shape=(2, 2)`` (the tensor shape of a flat set), ``lp_solver``.
    Every input set the builders here make is non-empty, so an empty result
    for any one of them, which the specification check would read as a proof,
    fails with an ``AssertionError`` naming ``what``. No exception is caught:
    a refusal (``NotImplementedError``) reaches the caller. The entry point is
    looked up in this module (``tests.cell_support.reach_pytorch_model``).

    The input set must come back as it was given, apart from the ranges a set
    caches about itself (``state_lb``, ``state_ub``): a kernel that works on
    the arrays of its input set changes the set for every later reader, which
    in a graph is the second consumer of a fan-out. A changed input set fails
    the test through ``pytest.fail``, naming the attributes; that is no
    ``AssertionError``, so a known-bug marker of the calling test does not
    absorb it."""
    out = []
    sets = _as_list(input_sets)
    for k, input_set in enumerate(sets):
        before = [_set_contents(input_set)]
        result = reach_pytorch_model(model, input_set, method=method,
                                     **options)
        _assert_unchanged(before, [input_set], f"{what} (input set {k})")
        if result is None or len(result) == 0:
            raise AssertionError(
                f"{what}: reach returned no sets for input set {k} of "
                f"{len(sets)} ({type(input_set).__name__}), which is not "
                f"empty")
        out.extend(result)
    return out


def reach_one_layer(layer, input_sets, method, what, **options):
    """Reach sets of one layer through
    ``n2v.nn.layer_ops.dispatcher.reach_layer``, which takes the whole list of
    input sets at once (one set is wrapped in a list). An empty result fails
    with an ``AssertionError`` naming ``what``; no exception is caught. The
    dispatcher is looked up in this module
    (``tests.cell_support.reach_layer``). The input sets must come back as
    they were given, as in :func:`reach_model`."""
    sets = _as_list(input_sets)
    before = [_set_contents(input_set) for input_set in sets]
    result = reach_layer(layer, sets, method, **options)
    _assert_unchanged(before, sets, what)
    if result is None or len(result) == 0:
        raise AssertionError(f"{what}: reach_layer returned no sets for "
                             f"{len(sets)} non-empty input set(s)")
    return list(result)


# --------------------------------------------------------------------------- #
# The coverage assertion                                                        #
# --------------------------------------------------------------------------- #
def _is_image(s):
    return isinstance(s, (ImageStar, ImageZono))


def _describe(s):
    if _is_image(s):
        return (f"{type(s).__name__} stored {s.height}x{s.width}x"
                f"{s.num_channels} (height x width x channels)")
    return f"{type(s).__name__} of {s.dim} entries"


def _summary(sets):
    names = [_describe(s) for s in sets]
    shown = sorted(set(names))
    return "; ".join(f"{names.count(n)} x {n}" for n in shown)


def assert_covered(sets, outputs, what, out_chw=None, tol=LP_TOL,
                   stored_order=False):
    """Every true output is a member of at least one of ``sets``, by
    ``membership.uncovered`` against each set's own rows (slack ``tol``
    relative to the size of the output, the oracle's rule).

    ``outputs`` are in the network's own order (flat vectors or tensors).

    - ``out_chw=(C, H, W)``: the output is an image tensor. Every image set
      among the results must store exactly height ``H``, width ``W`` and
      ``C`` channels (a set with height and width swapped fails even when its
      entry count agrees), and the outputs are converted to its stored order
      with ``membership.hwc``; a flat set among the results is compared in the
      tensor's own order.
    - neither ``out_chw`` nor ``stored_order``: the output is flat; an image
      set among the results fails unless it is 1x1 (its stored order is then
      the channel order), because nothing says how its entries line up.
    - ``stored_order=True``: the caller has already put the outputs in the
      order every returned set stores (for example ``hwc`` applied to a
      reference computed in tensor order); they are compared as given.

    Fails with an ``AssertionError`` (an empty list of sets, an image set of
    the wrong shape, an entry count that differs from the output, outputs in
    no set) whose message says how many outputs were missed, the largest
    distance, and the type and stored shape of the result. Wrong arguments
    (no outputs, both ``out_chw`` and ``stored_order``) are a ``ValueError``."""
    sets = list(sets)
    outputs = [np.asarray(y, dtype=np.float64).reshape(-1) for y in outputs]
    if not outputs:
        raise ValueError(f"{what}: no outputs to check")
    if out_chw is not None and stored_order:
        raise ValueError("give out_chw or stored_order, not both")
    if not sets:
        raise AssertionError(f"{what}: the result is an empty list of sets, "
                             f"so none of the {len(outputs)} outputs is in it")
    size = outputs[0].size
    views = []
    for s in sets:
        if out_chw is not None and _is_image(s):
            channels, height, width = (int(v) for v in out_chw)
            stored = (s.height, s.width, s.num_channels)
            if stored != (height, width, channels):
                raise AssertionError(
                    f"{what}: the result holds a {_describe(s)}, the output "
                    f"is a (C, H, W) = {tuple(out_chw)} tensor, stored as "
                    f"{height}x{width}x{channels}")
            views.append([hwc(y, channels, height, width) for y in outputs])
            continue
        if (out_chw is None and not stored_order and _is_image(s)
                and s.height * s.width > 1):
            raise AssertionError(
                f"{what}: the result holds a {_describe(s)} and the outputs "
                f"are flat: pass out_chw=(C, H, W) for an image output, or "
                f"stored_order=True with outputs already in the stored order")
        if s.dim != size:
            raise AssertionError(f"{what}: the result holds a {_describe(s)}, "
                                 f"the output has {size} entries")
        views.append(outputs)
    best = {i: float("inf") for i in range(len(outputs))}
    for view, s in zip(views, sets):
        remaining = sorted(best)
        missed = dict(uncovered([s], [view[i] for i in remaining], tol))
        for position, i in enumerate(remaining):
            if position in missed:
                best[i] = min(best[i], missed[position])
            else:
                del best[i]
        if not best:
            return
    largest = max(best.values())
    raise AssertionError(
        f"{what}: {len(best)} of {len(outputs)} true outputs are in none of "
        f"the {len(sets)} returned sets (largest distance {largest:.3g}; "
        f"first missed output index {min(best)}); result: {_summary(sets)}")


def hwc_index(channels, height, width):
    """The index array ``p`` with ``y_hwc = y_chw[p]``: entry ``p[k]`` of a
    (C, H, W) tensor flattened row-major is entry ``k`` of the vector an image
    set stores. Build an exact reference in tensor order and permute it once
    (``y_chw[p]``); ``x[p] = v`` puts a stored vector ``v`` back in tensor
    order."""
    return np.arange(channels * height * width).reshape(
        channels, height, width).transpose(1, 2, 0).reshape(-1)

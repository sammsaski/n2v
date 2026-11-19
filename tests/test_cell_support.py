"""Self-test of the shared cell support (``tests/cell_support.py``).

Every soundness cell rests on these helpers: a builder that builds the wrong
set, a replay in the wrong layout, a reach wrapper that lets an empty result
through or a coverage check that accepts everything would spoil all of them
at once. Each test plants such an error and watches the helper, or the oracle
reading the helper's output, reject it. Membership is decided by
``tests/_oracles`` only: the LP oracle on the set's own rows and, for dyadic
data, the exact rational oracle.
"""

from itertools import product

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
from onnx import TensorProto, helper

import n2v.sets
from n2v.sets import Star
from n2v.sets.image_star import ImageStar
from n2v.sets.image_zono import ImageZono
from n2v.utils.model_loader import load_onnx
from tests import cell_support as CS
from tests._oracles import membership as M
from tests._oracles import rational as R

C3, H3, W3 = 2, 3, 5                  # an asymmetric image: 30 entries


def _stored(points, chw=(C3, H3, W3)):
    return [M.hwc(p, *chw) for p in points]


def _faces_outside(lo, hi, step=2.0 ** -12):
    """The middle of every face of the box, pushed out by ``step``."""
    lo, hi = np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1)
    middle, out = (lo + hi) / 2, []
    for i in range(lo.size):
        for end in (lo[i] - step, hi[i] + step):
            y = middle.copy()
            y[i] = end
            out.append(y)
    return out


# --------------------------------------------------------------------------- #
# Set builders                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", CS.FLAT_KINDS)
def test_a_flat_set_holds_its_samples_and_nothing_just_outside(kind):
    """The box of five coordinates, one of zero width (index 2): every sample
    is a member, the two extreme corners exactly; the middle of every face
    pushed out by 2^-12 is not. A builder that shrinks a coordinate by a
    quarter misses samples."""
    lo, hi = CS.flat_box(5, zero_width=True)
    s = CS.flat_set(kind, lo, hi)
    assert type(s) is getattr(n2v.sets, kind) and s.dim == 5
    points = CS.samples(lo, hi, n=48)
    assert M.uncovered([s], points) == []
    member = R.Membership(R.star(s))
    assert member(lo) and member(hi)
    outside = _faces_outside(lo, hi)
    assert [i for i, _ in M.uncovered([s], outside)] == list(range(10))
    assert not any(member(y) for y in outside)
    shrunk = hi.copy()
    shrunk[0] -= (hi[0] - lo[0]) / 4
    assert M.uncovered([CS.flat_set(kind, lo, shrunk)], points)


@pytest.mark.parametrize("kind", CS.IMAGE_KINDS)
def test_an_image_set_holds_its_samples_in_its_stored_order(kind):
    """A (2, 3, 5) box with one degenerate pixel. The set stores height 3,
    width 5, 2 channels; the samples converted to that order are members, the
    faces pushed out are not. Planted layout bugs: the samples flattened in
    tensor order, and a set built from tensor-order bounds labelled as stored
    order, both miss."""
    lo, hi = CS.pixel_box(C3, H3, W3)
    lo[1, 2, 4] = hi[1, 2, 4] = 0.75
    s = CS.image_set(kind, lo, hi)
    assert type(s).__name__ == kind
    assert (s.height, s.width, s.num_channels) == (H3, W3, C3)
    points = CS.samples(lo, hi, n=48)
    assert M.uncovered([s], _stored(points)) == []
    member = R.Membership(R.star(s))
    assert member(M.hwc(lo, C3, H3, W3)) and member(M.hwc(hi, C3, H3, W3))
    outside = [M.hwc(y.reshape(C3, H3, W3), C3, H3, W3)
               for y in _faces_outside(lo, hi)]
    assert len(M.uncovered([s], outside)) == len(outside)
    as_tensor = M.uncovered([s], [p.reshape(-1) for p in points])
    assert len(as_tensor) >= 0.9 * len(points)
    cls = {"ImageStar": ImageStar, "ImageZono": ImageZono}[kind]
    mislabelled = cls.from_bounds(lo.reshape(-1), hi.reshape(-1), H3, W3, C3)
    assert len(M.uncovered([mislabelled], _stored(points))) >= 0.9 * len(points)


@pytest.mark.parametrize("build, kind", [
    (CS.flat_set, "ImageStar"), (CS.flat_set, "star"),
    (CS.image_set, "Star"), (CS.image_set, "imagezono")])
def test_an_unknown_kind_is_an_error(build, kind):
    lo, hi = CS.pixel_box(C3, H3, W3)
    with pytest.raises(ValueError, match="unknown"):
        build(kind, lo, hi)


def test_bounds_that_cross_or_do_not_match_are_an_error():
    with pytest.raises(ValueError):
        CS.flat_set("Star", [0.0, 1.0], [1.0, 0.5])
    with pytest.raises(ValueError):
        CS.flat_set("Box", [0.0, 1.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        CS.image_set("ImageStar", np.zeros((3, 5)), np.ones((3, 5)))


# --------------------------------------------------------------------------- #
# Boxes and sample points                                                       #
# --------------------------------------------------------------------------- #
def test_pixel_box_gives_every_pixel_its_own_dyadic_interval():
    lo, hi = CS.pixel_box(C3, H3, W3)
    assert lo.shape == hi.shape == (C3, H3, W3)
    assert np.all(hi > lo)
    assert np.all(lo * 4 == np.round(lo * 4)) and np.all(hi * 4 == np.round(hi * 4))
    assert len(set(zip(lo.ravel(), hi.ravel()))) >= 20
    # Height and width swapped is another box: a reader that swaps them is seen.
    assert not np.array_equal(lo.transpose(0, 2, 1).reshape(C3, H3, W3), lo)
    assert np.array_equal(CS.pixel_box(C3, H3, W3)[0], lo)
    assert not np.array_equal(CS.pixel_box(C3, H3, W3, seed=1)[0], lo)


@pytest.mark.parametrize("n, zero_width", [(3, False), (7, False), (4, True),
                                           (9, True)])
def test_flat_box_has_every_regime_and_the_degenerate_coordinate(n, zero_width):
    lo, hi = CS.flat_box(n, zero_width=zero_width)
    assert lo.shape == hi.shape == (n,)
    assert np.all(lo * 8 == np.round(lo * 8)) and np.all(hi * 8 == np.round(hi * 8))
    flat = lo == hi
    assert flat.sum() == int(zero_width)
    if zero_width:
        assert flat[n // 2] and lo[n // 2] != 0
    live = ~flat
    assert np.all(lo[live] < hi[live])
    assert np.any((lo < 0) & (hi > 0)) and np.any(lo > 0) and np.any(hi < 0)


def test_samples_are_the_corners_and_the_quarter_grid_of_the_box():
    lo, hi = CS.flat_box(4, zero_width=True)
    points = CS.samples(lo, hi, n=40)
    assert len(points) == 40 and all(p.shape == lo.shape for p in points)
    assert np.array_equal(points[0], lo) and np.array_equal(points[1], hi)
    live = hi > lo
    steps = [((p - lo)[live] / (hi - lo)[live] * 4) for p in points]
    assert all(np.array_equal(s, np.round(s)) and s.min() >= 0 and s.max() <= 4
               for s in steps)
    assert all(np.all(p[~live] == lo[~live]) for p in points)
    corners = sum(bool(np.all((s == 0) | (s == 4))) for s in steps)
    inner = sum(bool(np.any((s > 0) & (s < 4))) for s in steps)
    assert corners >= 19 and inner >= 15
    image = CS.samples(*CS.pixel_box(C3, H3, W3), n=8)
    assert all(p.shape == (C3, H3, W3) for p in image)


# --------------------------------------------------------------------------- #
# A Star and an ImageStar with a binding row                                    #
# --------------------------------------------------------------------------- #
LO3, HI3 = np.array([-0.5, 0.25, -2.0]), np.array([1.0, 0.75, -1.5])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_constrained_star_has_a_binding_row_and_member_points(seed):
    """The points are exact members; the extra row cuts off a corner of the
    box (the exact vertex list is not the box's corners, and that corner is
    reported outside), the bounding box is still the box, and some points lie
    on the cut face. The same star without its row holds the corner: the cut
    is what the oracle sees."""
    star, points = CS.constrained_star(LO3, HI3, n=64, seed=seed)
    assert isinstance(star, Star) and len(points) >= 64
    exact = R.star(star)
    member = R.Membership(exact)
    assert all(member(p) for p in points)
    assert M.uncovered([star], points) == []
    corners = [np.array(c) for c in product(*zip(LO3, HI3))]
    vertices = R.set_vertices(exact)
    assert vertices != sorted(tuple(R.vector(c)) for c in corners)
    cut = [c for c in corners if not member(c)]
    assert len(cut) == 1
    assert len(M.uncovered([star], cut)) == 1
    for i in range(3):
        assert min(v[i] for v in vertices) == LO3[i]
        assert max(v[i] for v in vertices) == HI3[i]
    c, G, C, d, lb, ub = M.rows(star)
    row, rhs = C[-1], d[-1]
    assert np.count_nonzero(row) >= 2
    on_face = sum(row @ np.linalg.solve(G, p - c) == rhs for p in points)
    assert on_face >= 5
    loose = Star(star.V, star.C[:-1], star.d[:-1], star.predicate_lb,
                 star.predicate_ub)
    assert M.uncovered([loose], cut) == []


def test_the_constrained_star_is_accepted_by_both_entry_points():
    """Through ``reach_layer`` (exact ReLU, reference ``max(x, 0)``) and
    through ``reach_pytorch_model`` (approx, reference torch float64), with a
    degenerate coordinate in the box."""
    lo, hi = CS.flat_box(4, zero_width=True)
    star, points = CS.constrained_star(lo, hi)
    sets = CS.reach_one_layer(nn.ReLU(), [star], "exact", "ReLU")
    CS.assert_covered(sets, [np.maximum(p, 0) for p in points], "ReLU")
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    with torch.no_grad():
        for layer in (model[0], model[2]):
            layer.weight.copy_(torch.randint(-4, 5, layer.weight.shape) / 4)
            layer.bias.copy_(torch.randint(-4, 5, layer.bias.shape) / 8)
    sets = CS.reach_model(model, star, "approx", "Linear-ReLU-Linear")
    CS.assert_covered(sets, CS.replay_torch(model, points, (1, 4)),
                      "Linear-ReLU-Linear")


def test_the_constrained_image_star_stores_height_by_width_by_channel():
    """Its points are (C, H, W) tensors whose stored order is a member; in
    tensor order they are not. The predicate corner the row cuts off is a
    point of the box and not of the set. Through a strided convolution the
    model's outputs are in the reach set."""
    lo, hi = CS.pixel_box(C3, H3, W3)
    image, points = CS.constrained_image_star(lo, hi, n=64)
    assert isinstance(image, ImageStar)
    assert (image.height, image.width, image.num_channels) == (H3, W3, C3)
    assert len(points) >= 64 and all(p.shape == (C3, H3, W3) for p in points)
    assert M.uncovered([image], _stored(points)) == []
    flat = M.uncovered([image], [p.reshape(-1) for p in points])
    assert len(flat) >= 0.9 * len(points)
    c, G, C, d, lb, ub = M.rows(image)
    corner = c + G @ np.where(C[-1] >= 0, 1.0, -1.0)
    assert np.all(M.hwc(lo, C3, H3, W3) <= corner)
    assert np.all(corner <= M.hwc(hi, C3, H3, W3))
    assert M.distance(image, corner) > 0
    assert not R.member(R.star(image), corner)
    model = nn.Sequential(nn.Conv2d(C3, 3, (2, 3), stride=(1, 2)), nn.ReLU(),
                          nn.Flatten())
    with torch.no_grad():
        model[0].weight.copy_(torch.randint(-4, 5, model[0].weight.shape) / 4)
        model[0].bias.copy_(torch.randint(-4, 5, model[0].bias.shape) / 8)
    sets = CS.reach_model(model, image, "approx", "Conv-ReLU-Flatten")
    CS.assert_covered(sets, CS.replay_torch(model, points, (1, C3, H3, W3)),
                      "Conv-ReLU-Flatten")


def test_a_constrained_star_needs_two_live_coordinates():
    with pytest.raises(ValueError, match="two coordinates"):
        CS.constrained_star([0.0, 1.0], [0.0, 2.0])


# --------------------------------------------------------------------------- #
# A Star and an ImageStar whose rows tie a kept entry to a dropped one          #
# --------------------------------------------------------------------------- #
LO5 = np.array([-0.5, 0.25, -2.0, 1.0, -1.0])
HI5 = np.array([1.0, 0.75, -1.5, 3.0, 0.5])
KEPT5 = [1, 3, 4]                     # entries 0 and 2 are dropped


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_coupled_star_ties_a_kept_entry_to_a_dropped_one(seed):
    """Row 1 ties kept entry 1 to dropped entry 0: both at their upper ends is
    outside, entry 1 at its upper end with entry 0 at its lower end is a
    member and is among the points, and no member has entry 1 at its upper
    end otherwise. Row 2 ties kept entries 4 and 1 and cuts a corner of their
    projection. The bounding box is still the box, points lie on the face of
    each row, and the same star without the two rows holds both cut points:
    the rows are what the oracle sees."""
    star, points = CS.coupled_star("Star", LO5, HI5, KEPT5, n=64, seed=seed)
    assert isinstance(star, Star) and len(points) >= 64
    exact = R.star(star)
    member = R.Membership(exact)
    assert all(member(p) for p in points)
    assert M.uncovered([star], points) == []
    vertices = R.set_vertices(exact)
    for i in range(5):
        assert min(v[i] for v in vertices) == LO5[i]
        assert max(v[i] for v in vertices) == HI5[i]
    middle = (LO5 + HI5) / 2
    both_up, witness, corner = middle.copy(), middle.copy(), middle.copy()
    both_up[[0, 1]] = HI5[[0, 1]]
    witness[1], witness[0] = HI5[1], LO5[0]
    corner[4], corner[1] = HI5[4], LO5[1]
    assert member(witness)
    assert not member(both_up) and not member(corner)
    assert len(M.uncovered([star], [both_up, corner])) == 2
    at_upper_end = [p for p in points if p[1] == HI5[1]]
    assert len(at_upper_end) >= 5
    assert all(p[0] == LO5[0] for p in at_upper_end)
    c, G, C, d, lb, ub = M.rows(star)
    assert np.count_nonzero(C[-2]) == 2 and np.count_nonzero(C[-1]) == 2
    predicate = [np.linalg.solve(G, p - c) for p in points]
    assert sum(C[-2] @ a == d[-2] for a in predicate) >= 5
    assert sum(C[-1] @ a == d[-1] for a in predicate) >= 5
    loose = Star(star.V, star.C[:-2], star.d[:-2], star.predicate_lb,
                 star.predicate_ub)
    assert M.uncovered([loose], [both_up, corner]) == []


def test_the_coupled_image_star_stores_height_by_width_by_channel():
    """Its points are (C, H, W) tensors whose stored order is a member; in
    tensor order they are not. Read back from the stored rows, row 1 ties one
    kept pixel to one dropped pixel and row 2 two kept pixels; the point with
    the two pixels of row 1 at their upper ends is in the box and not in the
    set. Through a strided convolution the model's outputs are in the reach
    set."""
    lo, hi = CS.pixel_box(C3, H3, W3)
    kept = np.arange(C3 * H3 * W3).reshape(C3, H3, W3)[:, 1:, ::2].ravel()
    image, points = CS.coupled_star("ImageStar", lo, hi, kept, n=64)
    assert isinstance(image, ImageStar)
    assert (image.height, image.width, image.num_channels) == (H3, W3, C3)
    assert len(points) >= 64 and all(p.shape == (C3, H3, W3) for p in points)
    assert M.uncovered([image], _stored(points)) == []
    flat = M.uncovered([image], [p.reshape(-1) for p in points])
    assert len(flat) >= 0.9 * len(points)
    c, G, C, d, lb, ub = M.rows(image)
    tensor_index = CS.hwc_index(C3, H3, W3)

    def pixels(row):
        """Tensor (row-major) indices of the pixels a row ties together."""
        stored = [int(np.flatnonzero(G[:, v])[0]) for v in np.flatnonzero(row)]
        return {int(tensor_index[s]) for s in stored}

    kept_set = {int(p) for p in kept}
    assert len(pixels(C[-2]) & kept_set) == 1
    assert len(pixels(C[-2]) - kept_set) == 1
    assert len(pixels(C[-1])) == 2 and pixels(C[-1]) <= kept_set
    corner = c + G @ np.where(C[-2] != 0, 1.0, 0.0)
    assert np.all(M.hwc(lo, C3, H3, W3) <= corner)
    assert np.all(corner <= M.hwc(hi, C3, H3, W3))
    assert M.distance(image, corner) > 0
    assert not R.member(R.star(image), corner)
    model = nn.Sequential(nn.Conv2d(C3, 3, (2, 3), stride=(1, 2)), nn.ReLU(),
                          nn.Flatten())
    with torch.no_grad():
        model[0].weight.copy_(torch.randint(-4, 5, model[0].weight.shape) / 4)
        model[0].bias.copy_(torch.randint(-4, 5, model[0].bias.shape) / 8)
    sets = CS.reach_model(model, image, "approx", "Conv-ReLU-Flatten")
    CS.assert_covered(sets, CS.replay_torch(model, points, (1, C3, H3, W3)),
                      "Conv-ReLU-Flatten")


def test_a_coupled_star_with_every_entry_kept_has_the_second_row_only():
    """Nothing is dropped, so no row ties a kept entry to a dropped one; the
    row between two kept entries is still there and still binds. A
    zero-width entry gets no predicate variable."""
    lo, hi = LO5.copy(), HI5.copy()
    hi[2] = lo[2]
    star, points = CS.coupled_star("Star", lo, hi, [0, 1, 2, 3, 4], n=32)
    c, G, C, d, lb, ub = M.rows(star)
    assert G.shape == (5, 4) and C.shape[0] == 2 * 4 + 1
    assert all(p[2] == lo[2] for p in points)
    assert M.uncovered([star], points) == []
    corner = (lo + hi) / 2
    corner[4], corner[0] = hi[4], lo[0]
    assert len(M.uncovered([star], [corner])) == 1


@pytest.mark.parametrize("kind, lo, hi, kept, message", [
    ("Zono", LO5, HI5, KEPT5, "unknown kind"),
    ("ImageStar", LO5, HI5, KEPT5, r"\(C, H, W\)"),
    ("Star", LO5, HI5, [1, 5], "outside"),
    ("Star", LO5, HI5, [3], "two kept entries"),
    ("Star", LO5, HI5, [3, 3], "two kept entries"),
])
def test_a_coupled_star_refuses_what_it_cannot_build(kind, lo, hi, kept,
                                                     message):
    with pytest.raises(ValueError, match=message):
        CS.coupled_star(kind, lo, hi, kept)


# --------------------------------------------------------------------------- #
# Several input sets at once                                                    #
# --------------------------------------------------------------------------- #
def _hold_counts(parts, lo, hi, to_set=lambda y: y):
    """How many of 17 points along the widest coordinate each part holds."""
    flat_lo, flat_hi = np.asarray(lo).reshape(-1), np.asarray(hi).reshape(-1)
    j = int(np.argmax(flat_hi - flat_lo))
    line = []
    for t in range(17):
        y = (flat_lo + flat_hi) / 2
        y[j] = flat_lo[j] + (flat_hi[j] - flat_lo[j]) * t / 16
        line.append(to_set(y))
    return [17 - len(M.uncovered([p], line)) for p in parts]


@pytest.mark.parametrize("kind", CS.FLAT_KINDS)
def test_a_split_cover_is_the_box_in_parts_of_different_sizes(kind):
    """The union holds every sample; each part alone misses some; the parts
    have different sizes, the first the smallest. Planted bugs: a layer that
    reached only the first part, or reused it for the others, or dropped the
    last."""
    lo, hi = CS.flat_box(4, zero_width=True)
    parts = CS.split_cover(kind, lo, hi, parts=3)
    assert len(parts) == 3 and all(type(p).__name__ == kind for p in parts)
    points = CS.samples(lo, hi, n=24)
    assert M.uncovered(parts, points) == []
    # Each part alone (the first reached three times over) misses samples.
    assert all(M.uncovered([p], points) for p in parts)
    counts = _hold_counts(parts, lo, hi)
    assert counts == [3, 5, 11]                   # shares 1, 2, 5 of 8, ends shared
    assert M.uncovered(parts[:2], points)
    assert [len(CS.split_cover(kind, lo, hi, parts=k)) for k in (1, 2, 4)] \
        == [1, 2, 4]
    assert _hold_counts(CS.split_cover(kind, lo, hi, parts=2), lo, hi) == [5, 13]


@pytest.mark.parametrize("kind", CS.IMAGE_KINDS)
def test_an_image_split_cover_is_the_image_box_in_parts(kind):
    lo, hi = CS.pixel_box(C3, H3, W3)
    parts = CS.image_split_cover(kind, lo, hi, parts=3)
    assert all(type(p).__name__ == kind
               and (p.height, p.width, p.num_channels) == (H3, W3, C3)
               for p in parts)
    stored = _stored(CS.samples(lo, hi, n=24))
    assert M.uncovered(parts, stored) == []
    assert all(M.uncovered([p], stored) for p in parts)
    counts = _hold_counts(parts, lo, hi,
                          lambda y: M.hwc(y.reshape(C3, H3, W3), C3, H3, W3))
    assert counts == [3, 5, 11]


# --------------------------------------------------------------------------- #
# Reference models                                                              #
# --------------------------------------------------------------------------- #
def _conv_parts():
    """A strided, padded convolution with dyadic weights, then ReLU and
    Flatten: (2, 3, 5) -> (3, 4, 2) -> 24 entries."""
    rng = np.random.default_rng(3)
    weight = (rng.integers(-4, 5, (3, C3, 2, 3)) / 4).astype(np.float32)
    bias = (rng.integers(-4, 5, 3) / 8).astype(np.float32)
    conv = nn.Conv2d(C3, 3, (2, 3), stride=(1, 2), padding=(1, 0))
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(weight))
        conv.bias.copy_(torch.from_numpy(bias))
    return weight, bias, nn.Sequential(conv, nn.ReLU(), nn.Flatten())


def _conv_onnx(tmp_path, weight, bias, opset=13):
    nodes = [helper.make_node("Conv", ["X", "W", "B"], ["A"],
                              kernel_shape=[2, 3], strides=[1, 2],
                              pads=[1, 0, 1, 0]),
             helper.make_node("Relu", ["A"], ["R"]),
             helper.make_node("Flatten", ["R"], ["Y"])]
    return CS.save_onnx(tmp_path, nodes, [1, C3, H3, W3], [1, 24],
                        {"W": weight, "B": bias}, name=f"conv{opset}",
                        opset=opset)


def test_onnx_and_torch_replays_agree_exactly_and_see_the_layout(tmp_path):
    """The same network as an ONNX graph (onnxruntime, float32) and as a torch
    module (float64) gives the same outputs, bit for bit, on dyadic points.
    The same points read with height and width swapped give other outputs on
    both, so a replay that mixes the layout up is seen."""
    weight, bias, module = _conv_parts()
    path = _conv_onnx(tmp_path, weight, bias)
    points = CS.samples(*CS.pixel_box(C3, H3, W3), n=32)
    by_onnx = CS.replay_onnx(path, points, [1, C3, H3, W3])
    by_torch = CS.replay_torch(module, points, (1, C3, H3, W3))
    assert all(y.dtype == np.float64 and y.shape == (24,)
               for y in by_onnx + by_torch)
    assert all(np.array_equal(a, b) for a, b in zip(by_onnx, by_torch))
    flat = CS.replay_onnx(path, [p.reshape(-1) for p in points], [1, C3, H3, W3])
    assert all(np.array_equal(a, b) for a, b in zip(flat, by_onnx))
    swapped = [p.transpose(0, 2, 1).reshape(C3, H3, W3) for p in points]
    for replay in (CS.replay_onnx(path, swapped, [1, C3, H3, W3]),
                   CS.replay_torch(module, swapped, (1, C3, H3, W3))):
        differ = sum(not np.array_equal(a, b) for a, b in zip(replay, by_onnx))
        assert differ >= 0.9 * len(points)


def test_replay_torch_leaves_the_callers_module_alone():
    _, _, module = _conv_parts()
    module.train()
    before = [p.detach().clone() for p in module.parameters()]
    CS.replay_torch(module, CS.samples(*CS.pixel_box(C3, H3, W3), n=4),
                    (1, C3, H3, W3))
    assert module.training
    assert all(p.dtype == torch.float32 for p in module.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, module.parameters()))


def test_replay_torch_evaluates_in_float64_and_in_eval_mode():
    """0.1 is not a float32 number: a float32 evaluation would round it.
    Dropout in training mode would zero entries at random."""
    module = nn.Sequential(nn.Dropout(0.5), nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        module[1].weight.fill_(3.0)
    (y,) = CS.replay_torch(module, [np.array([0.1])], (1, 1))
    assert y.tolist() == [3.0 * 0.1]


@pytest.mark.parametrize("opset", [7, 9, 11, 13, 16, 18])
def test_save_onnx_writes_the_opset_it_is_asked_for(tmp_path, opset):
    """The old private helper wrote opset 13 whatever the cell needed. The
    model records the opset asked for, an IR version onnxruntime loads, and
    the replay runs it."""
    k = np.array([[0.5, -1.0, 2.0]], dtype=np.float32)
    path = CS.save_onnx(tmp_path, [helper.make_node("Add", ["X", "k"], ["Y"])],
                        [1, 3], [1, 3], {"k": k}, name=f"add{opset}",
                        opset=opset)
    model = onnx.load(path)
    assert [(o.domain, o.version) for o in model.opset_import] == [("", opset)]
    least = helper.find_min_ir_version_for([helper.make_opsetid("", opset)])
    assert model.ir_version == max(10, least)
    (y,) = CS.replay_onnx(path, [np.array([0.25, 0.5, -0.75])], [1, 3])
    assert y.tolist() == [0.75, -0.5, 1.25]
    old = CS.save_onnx(tmp_path, [helper.make_node("Add", ["X", "k"], ["Y"])],
                       [1, 3], [1, 3], {"k": k}, name=f"ir{opset}",
                       opset=opset, ir_version=11)
    assert onnx.load(old).ir_version == 11


def test_save_onnx_takes_element_types_and_the_replay_follows_them(tmp_path):
    """A double graph is fed float64: 0.1 comes back as 0.1. A float graph is
    fed float32: 0.1 comes back rounded, so a cell that needs float64 inputs
    sees the difference."""
    double = CS.save_onnx(tmp_path, [helper.make_node("Identity", ["X"], ["Y"])],
                          [1, 2], [1, 2], name="double", in_type="DOUBLE",
                          out_type=TensorProto.DOUBLE)
    (y,) = CS.replay_onnx(double, [np.array([0.1, -0.3])], [1, 2])
    assert y.tolist() == [0.1, -0.3]
    single = CS.save_onnx(tmp_path, [helper.make_node("Identity", ["X"], ["Y"])],
                          [1, 2], [1, 2], name="single")
    (y,) = CS.replay_onnx(single, [np.array([0.1, -0.3])], [1, 2])
    assert y.tolist() == [float(np.float32(0.1)), float(np.float32(-0.3))]
    cast = CS.save_onnx(tmp_path, [helper.make_node("Cast", ["X"], ["Y"],
                                                    to=TensorProto.INT32)],
                        [1, 2], [1, 2], name="cast", out_type="INT32")
    (y,) = CS.replay_onnx(cast, [np.array([1.75, -1.75])], [1, 2])
    assert y.tolist() == [1.0, -1.0]


# --------------------------------------------------------------------------- #
# The coverage assertion                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", CS.IMAGE_KINDS)
def test_assert_covered_passes_on_a_correct_image_result(kind):
    lo, hi = CS.pixel_box(C3, H3, W3)
    CS.assert_covered([CS.image_set(kind, lo, hi)], CS.samples(lo, hi, n=48),
                      "identity", out_chw=(C3, H3, W3))


@pytest.mark.parametrize("kind", CS.IMAGE_KINDS)
@pytest.mark.parametrize("plant", ["shrunk", "stored-chw", "swapped-labels",
                                   "swapped-image", "empty"])
def test_assert_covered_rejects_a_wrong_image_result(kind, plant):
    """(a) the set shrunk inward by a quarter on one side of one pixel, (b)
    the correct entries stored in (C, H, W) order, (c) the stored vector right
    but height and width swapped in the labels, (d) the image itself
    transposed, (e) no set at all."""
    lo, hi = CS.pixel_box(C3, H3, W3)
    cls = {"ImageStar": ImageStar, "ImageZono": ImageZono}[kind]
    if plant == "shrunk":
        low = hi.copy()
        low[1, 2, 3] -= (hi - lo)[1, 2, 3] / 4
        sets = [CS.image_set(kind, lo, low)]
    elif plant == "stored-chw":
        sets = [cls.from_bounds(lo.reshape(-1), hi.reshape(-1), H3, W3, C3)]
    elif plant == "swapped-labels":
        sets = [cls.from_bounds(M.hwc(lo, C3, H3, W3), M.hwc(hi, C3, H3, W3),
                                W3, H3, C3)]
    elif plant == "swapped-image":
        sets = [cls.from_bounds(lo.transpose(2, 1, 0), hi.transpose(2, 1, 0),
                                W3, H3, C3)]
    else:
        sets = []
    with pytest.raises(AssertionError):
        CS.assert_covered(sets, CS.samples(lo, hi, n=32), "planted",
                          out_chw=(C3, H3, W3))


def test_assert_covered_reports_what_was_missed():
    lo, hi = CS.pixel_box(C3, H3, W3)
    low = hi.copy()
    low[1, 2, 3] -= (hi - lo)[1, 2, 3] / 4
    with pytest.raises(AssertionError) as caught:
        CS.assert_covered([CS.image_set("ImageStar", lo, low)] * 2,
                          CS.samples(lo, hi, n=32), "the cell",
                          out_chw=(C3, H3, W3))
    message = str(caught.value)
    assert message.startswith("the cell: ")
    assert "of 32 true outputs are in none of the 2 returned sets" in message
    assert "largest distance" in message
    assert "2 x ImageStar stored 3x5x2 (height x width x channels)" in message
    with pytest.raises(AssertionError, match="stored 5x3x2"):
        CS.assert_covered([ImageStar.from_bounds(
            M.hwc(lo, C3, H3, W3), M.hwc(hi, C3, H3, W3), W3, H3, C3)],
            CS.samples(lo, hi, n=8), "labels", out_chw=(C3, H3, W3))


def test_assert_covered_on_flat_results():
    """A flat result is compared in the tensor's own order: the right set
    passes, a set shrunk by a quarter or of the wrong size fails, two parts
    pass together and one alone fails."""
    lo, hi = CS.flat_box(5, zero_width=True)
    points = CS.samples(lo, hi, n=32)
    for kind in CS.FLAT_KINDS:
        CS.assert_covered([CS.flat_set(kind, lo, hi)], points, kind)
    shrunk = lo.copy()
    shrunk[3] += (hi[3] - lo[3]) / 4
    with pytest.raises(AssertionError, match="true outputs are in none"):
        CS.assert_covered([CS.flat_set("Star", shrunk, hi)], points, "shrunk")
    with pytest.raises(AssertionError, match="6 entries"):
        CS.assert_covered([CS.flat_set("Zono", np.append(lo, 0),
                                       np.append(hi, 1))], points, "size")
    parts = CS.split_cover("Star", lo, hi, parts=2)
    CS.assert_covered(parts, points, "two parts")
    with pytest.raises(AssertionError, match="1 returned sets"):
        CS.assert_covered(parts[:1], points, "one part")


def test_an_image_result_needs_its_layout_named():
    """Without ``out_chw`` an image result is a failure, since nothing says
    how its entries line up with flat outputs; ``stored_order=True`` with
    outputs already converted passes; a 1x1 image is its channel order. A flat
    set beside an image one is compared in tensor order."""
    lo, hi = CS.pixel_box(C3, H3, W3)
    image = CS.image_set("ImageStar", lo, hi)
    points = CS.samples(lo, hi, n=32)
    with pytest.raises(AssertionError, match="out_chw"):
        CS.assert_covered([image], _stored(points), "no layout")
    CS.assert_covered([image], _stored(points), "stored", stored_order=True)
    with pytest.raises(AssertionError):
        CS.assert_covered([image], [p.reshape(-1) for p in points], "stored",
                          stored_order=True)
    pixel = CS.image_set("ImageZono", lo[:, :1, :1], hi[:, :1, :1])
    CS.assert_covered([pixel], [p[:, :1, :1] for p in points], "one pixel")
    flat = CS.flat_set("Star", lo, hi)
    CS.assert_covered([flat, image], points, "mixed", out_chw=(C3, H3, W3))
    with pytest.raises(ValueError):
        CS.assert_covered([image], points, "both", out_chw=(C3, H3, W3),
                          stored_order=True)
    with pytest.raises(ValueError):
        CS.assert_covered([image], [], "no outputs", out_chw=(C3, H3, W3))


def test_the_example_cell_of_the_module_docstring_runs(tmp_path):
    """The example: exact ReLU on a (2, 3, 5) ImageStar against onnxruntime.
    The same outputs without the layout, or without one returned piece,
    fail."""
    shape = [1, C3, H3, W3]
    path = CS.save_onnx(tmp_path, [helper.make_node("Relu", ["X"], ["Y"])],
                        shape, shape)
    lo, hi = CS.pixel_box(C3, H3, W3)
    sets = CS.reach_model(load_onnx(path), CS.image_set("ImageStar", lo, hi),
                          "exact", "Relu on ImageStar")
    outputs = CS.replay_onnx(path, CS.samples(lo, hi), shape)
    CS.assert_covered(sets, outputs, "Relu on ImageStar", out_chw=(C3, H3, W3))
    with pytest.raises(AssertionError):
        CS.assert_covered(sets, outputs[:32], "no layout", stored_order=True)
    # A piece where a crossing neuron is inactive is covered by the active
    # pieces at 0; the piece the oracle shows to be needed is dropped.
    few = _stored(outputs[:32])
    held = [set(range(32)) - {i for i, _ in M.uncovered([s], few)}
            for s in sets]
    needed = [k for k in range(len(sets))
              if held[k] - set().union(*(held[j] for j in range(len(sets))
                                         if j != k))]
    if not needed:
        pytest.fail("no piece of the exact ReLU is needed for the samples")
    with pytest.raises(AssertionError):
        CS.assert_covered(sets[:needed[0]] + sets[needed[0] + 1:],
                          outputs[:32], "a piece missing", out_chw=(C3, H3, W3))


def test_hwc_index_is_the_stored_order_of_a_tensor():
    t = np.arange(30.0).reshape(C3, H3, W3) * 7 + 1
    p = CS.hwc_index(C3, H3, W3)
    assert p.tolist() == [c * H3 * W3 + h * W3 + w for h in range(H3)
                          for w in range(W3) for c in range(C3)]
    assert np.array_equal(t.reshape(-1)[p], M.hwc(t, C3, H3, W3))
    back = np.empty(30)
    back[p] = M.hwc(t, C3, H3, W3)
    assert np.array_equal(back, t.reshape(-1))
    # The flat order sees only the channel count and the number of pixels:
    # height and width swapped give the same index, which is why
    # assert_covered checks the stored height and width themselves.
    assert np.array_equal(CS.hwc_index(C3, W3, H3), p)
    for wrong in ((H3, C3, W3), (W3, H3, C3)):
        assert not np.array_equal(t.reshape(-1)[CS.hwc_index(*wrong)],
                                  M.hwc(t, C3, H3, W3))


# --------------------------------------------------------------------------- #
# Reach wrappers                                                                #
# --------------------------------------------------------------------------- #
def test_reach_model_fails_on_an_empty_result_and_lets_a_refusal_through(
        monkeypatch):
    """An empty list would read as a proof. Each set of a list is reached on
    its own with the caller's options, results joined in order; an empty
    result for any one fails; a refusal is not caught."""
    parts = CS.split_cover("Star", [-1.0, -1.0], [1.0, 1.0], parts=3)
    calls = []

    def echo(model, input_set, method, **options):
        calls.append((input_set, method, options))
        return [] if input_set is EMPTY[0] else [input_set]

    EMPTY = [object()]
    monkeypatch.setattr(CS, "reach_pytorch_model", echo)
    out = CS.reach_model("net", parts, "exact", "echo",
                         precompute_bounds="ibp", input_shape=(2,))
    assert len(out) == 3 and all(a is b for a, b in zip(out, parts))
    assert calls == [(p, "exact", {"precompute_bounds": "ibp",
                                   "input_shape": (2,)}) for p in parts]
    assert CS.reach_model("net", parts[0], "approx", "one")[0] is parts[0]
    EMPTY[0] = parts[1]
    with pytest.raises(AssertionError, match="echo: reach returned no sets "
                                             "for input set 1 of 3"):
        CS.reach_model("net", parts, "exact", "echo")
    with pytest.raises(AssertionError, match="no sets"):
        CS.reach_model("net", parts[1], "exact", "single")

    def refuse(*args, **kwargs):
        raise NotImplementedError("no rule for this cell")

    monkeypatch.setattr(CS, "reach_pytorch_model", refuse)
    with pytest.raises(NotImplementedError, match="no rule"):
        CS.reach_model("net", parts, "exact", "refused")


def test_reach_one_layer_fails_on_an_empty_result_and_lets_a_refusal_through(
        monkeypatch):
    """The dispatcher gets the whole list in one call."""
    parts = CS.split_cover("Zono", [-1.0, -1.0], [1.0, 1.0], parts=3)
    calls = []

    def stub(layer, input_sets, method, **options):
        calls.append((layer, list(input_sets), method, options))
        return list(input_sets) if layer == "keep" else []

    monkeypatch.setattr(CS, "reach_layer", stub)
    assert CS.reach_one_layer("keep", parts, "approx", "keep",
                              lp_solver="highs") == parts
    assert calls == [("keep", parts, "approx", {"lp_solver": "highs"})]
    assert CS.reach_one_layer("keep", parts[0], "approx", "one") == [parts[0]]
    with pytest.raises(AssertionError, match="drop: reach_layer returned no "
                                             "sets for 3"):
        CS.reach_one_layer("drop", parts, "approx", "drop")

    def refuse(*args, **kwargs):
        raise NotImplementedError("no rule for this cell")

    monkeypatch.setattr(CS, "reach_layer", refuse)
    with pytest.raises(NotImplementedError, match="no rule"):
        CS.reach_one_layer("keep", parts, "approx", "refused")
    with pytest.raises(ValueError):
        CS.reach_one_layer("keep", [], "approx", "nothing")


def _change_first_array(s):
    """Double the first defining array of ``s`` in place."""
    for name in ("V", "generators", "lb"):
        if isinstance(getattr(s, name, None), np.ndarray):
            getattr(s, name)[...] *= 2.0
            return
    raise AssertionError(f"no array to change on {type(s).__name__}")


@pytest.mark.parametrize("kind", CS.FLAT_KINDS + CS.IMAGE_KINDS)
@pytest.mark.parametrize("wrapper", ["reach_one_layer", "reach_model"])
def test_a_reach_that_changes_its_input_set_fails(monkeypatch, kind, wrapper):
    """A kernel that scales the arrays of its input set where they are, in
    place of a copy, returns a right answer and leaves a wrong set behind for
    the second reader of a fan-out. Both wrappers compare each input set with
    what it held before the reach. A reach that only fills in the cached
    ranges of a set passes."""
    if kind in CS.IMAGE_KINDS:
        lo, hi = CS.pixel_box(2, 2, 3)
        build = lambda: CS.image_set(kind, lo, hi)
    else:
        lo, hi = np.array([-1.0, 0.5, 2.0]), np.array([1.0, 3.0, 2.5])
        build = lambda: CS.flat_set(kind, lo, hi)

    def in_place(target, sets, method=None, **options):
        sets = sets if isinstance(sets, list) else [sets]
        _change_first_array(sets[-1])
        return list(sets)

    def caches_only(target, sets, method=None, **options):
        sets = sets if isinstance(sets, list) else [sets]
        for s in sets:
            s.state_lb = np.zeros((3, 1))
            s.state_ub = np.ones((3, 1))
        return list(sets)

    entry = "reach_layer" if wrapper == "reach_one_layer" \
        else "reach_pytorch_model"
    call = getattr(CS, wrapper)
    monkeypatch.setattr(CS, entry, caches_only)
    assert len(call("op", [build(), build()], "approx", "caches")) == 2
    monkeypatch.setattr(CS, entry, in_place)
    with pytest.raises(pytest.fail.Exception,
                       match="the reach changed input set"):
        call("op", [build(), build()], "approx", "in place")


def test_a_change_inside_a_nested_part_of_a_set_is_seen(monkeypatch):
    """The comparison goes into the objects a set holds: the constraint
    system of a Hexatope is not an array of the set itself."""
    lo, hi = np.array([-1.0, 0.5, 2.0]), np.array([1.0, 3.0, 2.5])
    hexa = CS.flat_set("Hexatope", lo, hi)
    before = CS._set_contents(hexa)
    nested = [path for path in before if ".dcs." in path]
    assert nested, sorted(before)

    def in_place(layer, sets, method, **options):
        system = sets[0].dcs
        for name, value in vars(system).items():
            if isinstance(value, np.ndarray) and value.size:
                value[...] += 1.0
                return list(sets)
            if isinstance(value, list) and value:
                value.pop()
                return list(sets)
        raise AssertionError(f"nothing to change in {vars(system)}")

    monkeypatch.setattr(CS, "reach_layer", in_place)
    with pytest.raises(pytest.fail.Exception, match=r"Hexatope\.dcs"):
        CS.reach_one_layer("op", hexa, "approx", "nested")

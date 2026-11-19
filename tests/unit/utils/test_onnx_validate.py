"""Unit tests for the onnxruntime counterexample re-validation gate
(``n2v.utils.onnx_validate``), used by the VNN-COMP runner to confirm a
falsification witness still violates on the RAW ONNX before emitting ``sat``.

The gate is on the −150 verdict path, so these tests pin its mechanics
deterministically: it must ACCEPT a genuine counterexample and REJECT a safe
point. A silent over-rejection bug would turn every falsifier ``sat`` into
``unknown`` (a breadth catastrophe), which a torch-only test would miss — so we
export a real model to ONNX and round-trip through onnxruntime.
"""

import os
import tempfile

import numpy as np
import onnx
import onnxruntime  # noqa: F401 - a hard dependency; a broken install must fail
import pytest
import torch
import torch.nn as nn
from onnx import TensorProto, helper
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

from n2v.sets import HalfSpace
from n2v.utils.onnx_validate import (in_unsafe_region, make_onnx_forward,
                                     onnx_forward)
from tests.known_bugs import known_bug


@pytest.fixture(scope="module")
def identity_onnx():
    """Export y = x (2->2 identity Linear) to a temp ONNX file."""
    model = nn.Linear(2, 2, bias=False)
    model.weight.data = torch.eye(2)
    model.eval()
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    # dynamo=False uses the legacy TorchScript exporter (no onnxscript dep).
    torch.onnx.export(model, torch.zeros(1, 2), path,
                      input_names=["x"], output_names=["y"], opset_version=13,
                      dynamo=False)
    yield path
    os.remove(path)


class TestOnnxForward:
    def test_roundtrip_matches(self, identity_onnx):
        y = onnx_forward(identity_onnx, np.array([0.3, 0.7], dtype=np.float32))
        np.testing.assert_allclose(y, [0.3, 0.7], atol=1e-6)

    def test_accepts_flat_or_shaped_input(self, identity_onnx):
        # Flat input in ONNX-variable order is reshaped to the declared shape.
        y = onnx_forward(identity_onnx, [0.1, 0.9])
        assert y.shape == (2,)


class TestInUnsafeRegion:
    def test_single_halfspace(self):
        # Unsafe: y0 >= 0.5  (-y0 <= -0.5)
        groups = [[HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))]]
        assert in_unsafe_region([1.0, 0.0], groups)
        assert not in_unsafe_region([0.0, 0.0], groups)

    def test_and_across_groups(self):
        # (y0 >= 0.5) AND (y1 >= 0.5)
        groups = [
            [HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))],
            [HalfSpace(np.array([[0.0, -1.0]]), np.array([[-0.5]]))],
        ]
        assert in_unsafe_region([1.0, 1.0], groups)
        assert not in_unsafe_region([1.0, 0.0], groups)   # only one group hit
        assert not in_unsafe_region([0.0, 1.0], groups)

    def test_or_within_group(self):
        # One group, OR of two halfspaces: y0 >= 0.8  OR  y0 <= 0.2
        groups = [[
            HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.8]])),
            HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.2]])),
        ]]
        assert in_unsafe_region([0.9, 0.0], groups)
        assert in_unsafe_region([0.1, 0.0], groups)
        assert not in_unsafe_region([0.5, 0.0], groups)

    def test_tolerance_boundary(self):
        # y0 >= 0.5; a point 1e-5 short still counts unsafe at the default
        # tol of 1e-4 (a permissive check; the VNN-COMP 2026 grader and the
        # runner's sat gate use tol 0), but a point 1e-3 short does not.
        groups = [[HalfSpace(np.array([[-1.0]]), np.array([[-0.5]]))]]
        assert in_unsafe_region([0.5 - 1e-5], groups)        # within 1e-4
        assert not in_unsafe_region([0.5 - 1e-3], groups, tol=1e-4)


class TestGateEndToEnd:
    def test_accepts_real_ce_rejects_safe(self, identity_onnx):
        # Unsafe region: y0 >= 0.5 (since y = x, x0 >= 0.5 is a CE).
        groups = [[HalfSpace(np.array([[-1.0, 0.0]]), np.array([[-0.5]]))]]
        # genuine counterexample
        y_ce = onnx_forward(identity_onnx, np.array([0.9, 0.1], dtype=np.float32))
        assert in_unsafe_region(y_ce, groups)
        # safe point -> gate rejects (would downgrade sat->unknown)
        y_safe = onnx_forward(identity_onnx, np.array([0.1, 0.1], dtype=np.float32))
        assert not in_unsafe_region(y_safe, groups)


# ---------------------------------------------------------------------------
# Layout, batch dimension, several inputs and outputs, against numpy
# ---------------------------------------------------------------------------

# Dyadic values: exact in float32, so onnxruntime's float32 run of a
# transpose or a negation is exactly the numpy value.
_X6 = [0.5, 1.25, -2.0, 3.75, 8.0, -0.125]
_ROWS = np.array([_X6, [-v for v in _X6], [4.0 * v for v in _X6]])


def _save(tmp_path, name, nodes, inputs, outputs, inits=(),
          elem=TensorProto.FLOAT):
    """An ONNX file of ``nodes``; ``inputs``/``outputs`` are (name, shape)
    pairs, a shape entry being an int or a symbolic name; every tensor has
    the element type ``elem``."""
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info(n, elem, s) for n, s in inputs],
        [helper.make_tensor_value_info(n, elem, s) for n, s in outputs],
        list(inits))
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = str(tmp_path / f"{name}.onnx")
    onnx.save(model, path)
    return path


def _transpose_net(tmp_path, batch, negated_output=False):
    """X [batch, 2, 3] -> Y = X with its last two axes swapped, [batch, 3, 2];
    with ``negated_output`` a second output Z = -X [batch, 2, 3]."""
    nodes = [helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1])]
    outputs = [("Y", [batch, 3, 2])]
    if negated_output:
        nodes.append(helper.make_node("Neg", ["X"], ["Z"]))
        outputs.append(("Z", [batch, 2, 3]))
    return _save(tmp_path, f"t{batch}{int(negated_output)}", nodes,
                 [("X", [batch, 2, 3])], outputs)


def _transposed(row):
    """The row-major flat output of the transpose for one flat input row."""
    return np.asarray(row, dtype=np.float64).reshape(2, 3).T.reshape(-1)


class TestOnnxForwardLayout:
    @pytest.mark.parametrize("batch", [1, "N"],
                             ids=["fixed batch", "dynamic batch"])
    def test_input_and_output_are_read_row_major(self, tmp_path, batch):
        """X [batch, 2, 3] -> its transpose [batch, 3, 2]: the flat witness
        fills the input tensor row-major and the output tensor is flattened
        row-major. Catches a column-major reshape of either (the witness check
        then reads permuted outputs) and a dynamic batch dim not read as 1."""
        y = onnx_forward(_transpose_net(tmp_path, batch), _X6)
        assert y.tolist() == _transposed(_X6).tolist()


class TestMakeOnnxForward:
    @pytest.mark.parametrize("batch", [1, "N"],
                             ids=["fixed batch", "dynamic batch"])
    def test_each_row_is_the_output_of_that_row(self, tmp_path, batch):
        """Three different rows through X [batch, 2, 3] -> its transpose; a
        dynamic batch dim runs them in one call, a fixed one row by row. Each
        returned row must be the transpose of its own input row. Catches a
        batched reshape in column-major order (the rows' entries are mixed)
        and the batched call used on a fixed batch dim of 1."""
        out = make_onnx_forward(_transpose_net(tmp_path, batch))(_ROWS)
        assert out.shape == (3, 6)
        for row, got in zip(_ROWS, out):
            assert got.tolist() == _transposed(row).tolist()

    def test_a_flat_vector_is_one_row(self, tmp_path):
        """A 1-D input of the model's size is one row, not six rows of one
        entry each."""
        out = make_onnx_forward(_transpose_net(tmp_path, "N"))(np.array(_X6))
        assert out.shape == (1, 6)
        assert out[0].tolist() == _transposed(_X6).tolist()

    def test_each_input_tensor_is_fed_its_own_slice(self, tmp_path):
        """Two inputs A [1, 2] and B [1, 3], Y = Concat(A, -B) [1, 5]; a row is
        A's entries then B's. Catches a feed that gives B the slice of A (the
        offset not advanced), which replays an input that was never given."""
        path = _save(tmp_path, "two_in",
                     [helper.make_node("Neg", ["B"], ["nb"]),
                      helper.make_node("Concat", ["A", "nb"], ["Y"], axis=1)],
                     [("A", [1, 2]), ("B", [1, 3])], [("Y", [1, 5])])
        rows = np.array([[0.5, 1.25, -2.0, 3.75, 8.0],
                         [-0.125, 0.25, 6.0, -1.5, 2.5]])
        out = make_onnx_forward(path)(rows)
        expected = [list(r[:2]) + [-v for v in r[2:]] for r in rows]
        assert out.tolist() == expected

    @pytest.mark.parametrize("batch", [1, "N"],
                             ids=["fixed batch", "dynamic batch"])
    def test_every_output_tensor_is_returned(self, tmp_path, batch):
        """Two outputs, the transpose Y [batch, 3, 2] and Z = -X [batch, 2, 3]:
        each row is Y's six entries then Z's six, as the VNN-LIB resolver
        indexes the outputs jointly. Catches a forward that keeps only the
        first output tensor, in the row loop and in the batched call."""
        fwd = make_onnx_forward(_transpose_net(tmp_path, batch,
                                               negated_output=True))
        out = fwd(_ROWS[:2])
        assert out.shape == (2, 12)
        for row, got in zip(_ROWS[:2], out):
            assert got.tolist() == (_transposed(row).tolist()
                                    + [-v for v in row])


class TestInUnsafeRegionRows:
    def test_every_row_of_a_half_space_must_hold(self):
        """One half-space of two rows, y0 >= 0.5 and y1 <= 0.25, is their
        conjunction. Catches 'some row holds' read for 'every row holds':
        y = (1, 0.5) meets the first row only and is not in the region."""
        groups = [[HalfSpace(np.array([[-1.0, 0.0], [0.0, 1.0]]),
                             np.array([[-0.5], [0.25]]))]]
        assert in_unsafe_region([1.0, 0.0], groups, tol=0.0)
        assert not in_unsafe_region([1.0, 0.5], groups, tol=0.0)
        assert not in_unsafe_region([0.0, 0.0], groups, tol=0.0)

    def test_each_row_is_compared_with_its_own_bound(self):
        """Rows y0 <= 0.25 and y1 <= 0.75. y = (0.125, 0.5) holds both; its
        y1 lies between the two bounds, so comparing every row with every
        bound (a right-hand side broadcast as a row) rejects it. y = (0.5,
        0.125) breaks the first row (bounds swapped would accept it)."""
        groups = [[HalfSpace(np.array([[1.0, 0.0], [0.0, 1.0]]),
                             np.array([[0.25], [0.75]]))]]
        assert in_unsafe_region([0.125, 0.5], groups, tol=0.0)
        assert not in_unsafe_region([0.5, 0.125], groups, tol=0.0)

    def test_zero_tolerance_adds_no_slack(self):
        """At tol=0 the region is exactly the closed G y <= g, in float64. An
        output on the boundary is in it; one 2^-15 outside (less than the
        default 1e-4) is not; nor is 0.7000000001 or the next float64 after
        0.7 against y <= 0.7, which an output rounded to float32 first
        (0.69999999) would let in. The runner's sat gate relies on this."""
        ge = [[HalfSpace(np.array([[-1.0]]), np.array([[-0.5]]))]]   # y >= 0.5
        assert in_unsafe_region([0.5], ge, tol=0.0)
        assert not in_unsafe_region([0.5 - 2.0 ** -15], ge, tol=0.0)
        le = [[HalfSpace(np.array([[1.0]]), np.array([[0.7]]))]]     # y <= 0.7
        assert in_unsafe_region([0.7], le, tol=0.0)
        assert not in_unsafe_region([0.7000000001], le, tol=0.0)
        assert not in_unsafe_region([np.nextafter(0.7, 1.0)], le, tol=0.0)


# ---------------------------------------------------------------------------
# The replay runs in the model's own precision
# ---------------------------------------------------------------------------

_FLOAT64_REPLAY = known_bug(
    11, "onnx_forward and make_onnx_forward feed the witness to onnxruntime "
        "as float32 whatever the model's input type, so a float64 (double) "
        "ONNX model refuses it with InvalidArgument where its replay in "
        "float64 is due; the runner loses the counterexample",
    raises=InvalidArgument)


@pytest.mark.parametrize("forward,elem", [
    pytest.param("onnx_forward", TensorProto.FLOAT, id="onnx_forward-float32"),
    pytest.param("onnx_forward", TensorProto.DOUBLE, id="onnx_forward-float64",
                 marks=_FLOAT64_REPLAY),
    pytest.param("make_onnx_forward", TensorProto.FLOAT,
                 id="make_onnx_forward-float32"),
    pytest.param("make_onnx_forward", TensorProto.DOUBLE,
                 id="make_onnx_forward-float64", marks=_FLOAT64_REPLAY),
])
def test_the_replay_runs_in_the_models_own_precision(tmp_path, forward, elem):
    """Y = -X on a float32 and on a float64 (double) model, at x = (0.1,
    0.30000000000000004): the checker replays the file in its own precision,
    so the forward must return -float32(x) for the one and exactly -x for the
    other. The float64 cases fail today with onnxruntime's InvalidArgument
    (the witness is always fed as float32); the float32 cases are their
    passing neighbours."""
    path = _save(tmp_path, f"neg{elem}", [helper.make_node("Neg", ["X"], ["Y"])],
                 [("X", [1, 2])], [("Y", [1, 2])], elem=elem)
    x = [0.1, 0.30000000000000004]
    own = np.float32 if elem == TensorProto.FLOAT else np.float64
    expected = [-float(own(v)) for v in x]
    if forward == "onnx_forward":
        got = onnx_forward(path, x)
    else:
        got = make_onnx_forward(path)(np.array([x]))[0]
    assert got.tolist() == expected

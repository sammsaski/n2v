"""
Cross-tool soundness diff: n2v GINE reach vs. MATLAB GNNV reference.

The reference is produced by
    tests/fixtures/gnn/matlab_refs/generate_ieee24_reference.m
run on the gine_pretrain (HuGINEConv) checkpoint.  GINE carries internal message
and MLP ReLUs, so the bounds are looser than the linear layers, but n2v's box
must still contain MATLAB's at every (instance, eps) cell.

The reference is bundled; MATLAB is only needed to regenerate it.
"""

import numpy as np
import pytest
from scipy.io import loadmat

from n2v.sets import GraphStar
from n2v.nn.layer_ops.gine_reach import gine_stack_reach
from n2v.utils import load_gnn_mat
from tests.fixtures.gnn import checkpoint, matlab_reference


REFERENCE_PATH = matlab_reference("gine_pretrain")

SOUNDNESS_SLACK = 1e-7


def _decode_str(arr) -> str:
    return str(np.asarray(arr).flatten()[0])


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE_PATH.exists():
        pytest.skip(f"reference file not found: {REFERENCE_PATH}")
    raw = loadmat(str(REFERENCE_PATH))
    return {
        "checkpoint": _decode_str(raw["checkpoint"]),
        "instances": np.asarray(raw["instances"]).flatten().astype(int),
        "eps_values": np.asarray(raw["eps_values"]).flatten().astype(float),
        "lb": raw["lb"],
        "ub": raw["ub"],
    }


@pytest.fixture(scope="module")
def model(reference):
    return load_gnn_mat(checkpoint("gine_pretrain"))


def pytest_generate_tests(metafunc):
    if "cross_tool_case" not in metafunc.fixturenames:
        return
    if not REFERENCE_PATH.exists():
        metafunc.parametrize("cross_tool_case", [], ids=[])
        return
    raw = loadmat(str(REFERENCE_PATH))
    instances = np.asarray(raw["instances"]).flatten().astype(int)
    eps_values = np.asarray(raw["eps_values"]).flatten().astype(float)
    cases, ids = [], []
    for ii, inst in enumerate(instances):
        for jj, eps in enumerate(eps_values):
            cases.append((ii, jj, int(inst), float(eps)))
            ids.append(f"inst{int(inst)}_eps{float(eps):.0e}")
    metafunc.parametrize("cross_tool_case", cases, ids=ids)


def test_n2v_contains_matlab_box(cross_tool_case, reference, model):
    ii, jj, inst_idx, eps = cross_tool_case
    X = model.X_test[inst_idx - 1]                       # MATLAB indices are 1-based
    gs_in = GraphStar.from_bounds(X - eps, X + eps)
    outs = gine_stack_reach(gs_in, model.E, model.edge_index, model.layers,
                            variant=model.gine_variant, edge_weights=model.edge_weights)
    assert len(outs) == 1
    lb_n2v, ub_n2v = outs[0].get_ranges()

    lb_mat = np.asarray(reference["lb"][ii, jj], dtype=float)
    ub_mat = np.asarray(reference["ub"][ii, jj], dtype=float)
    assert lb_mat.shape == lb_n2v.shape

    slack_lb = lb_mat - lb_n2v
    slack_ub = ub_n2v - ub_mat
    worst = float(min(slack_lb.min(), slack_ub.min()))
    assert worst >= -SOUNDNESS_SLACK, (
        f"n2v box does not contain MATLAB box for instance {inst_idx} eps {eps}. "
        f"Worst slack = {worst:+.3e} (allowed -{SOUNDNESS_SLACK:.0e})."
    )

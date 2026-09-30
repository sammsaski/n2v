"""
Cross-tool soundness diff: n2v GraphStar reach vs. MATLAB GNNV reference.

The reference is a static .mat produced by
    tests/fixtures/gnn/matlab_refs/generate_ieee24_reference.m

Since both tools produce sound over-approximations of the same true
reachable set, n2v's box must *contain* MATLAB's box.  The test asserts
that hard property at every (instance, eps) cell up to LP solver
tolerance.

Width inflation is reported in test output but not asserted: n2v's
approx-star ReLU classifies neurons via the predicate-box estimate
instead of LP, so its bounds widen at large eps.  This is a known
tightness gap, not a soundness regression.

The reference is bundled under tests/fixtures/gnn, so MATLAB is only needed
to regenerate it.
"""

import numpy as np
import pytest
from scipy.io import loadmat

from n2v.sets import GraphStar
from n2v.nn.layer_ops.gcn_reach import gcn_stack_reach
from n2v.utils import load_gnn_mat
from tests.fixtures.gnn import checkpoint, matlab_reference


REFERENCE_PATH = matlab_reference("gcn")


SOUNDNESS_SLACK = 1e-7   # LP solver tolerance floor; same-shape across solvers
WIDTH_FLOOR = 1e-6


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
    return load_gnn_mat(checkpoint("gcn"))


def pytest_generate_tests(metafunc):
    """Discover reference cells and parametrize the soundness test cleanly."""
    if "cross_tool_case" not in metafunc.fixturenames:
        return
    if not REFERENCE_PATH.exists():
        metafunc.parametrize("cross_tool_case", [], ids=[])
        return
    raw = loadmat(str(REFERENCE_PATH))
    instances = np.asarray(raw["instances"]).flatten().astype(int)
    eps_values = np.asarray(raw["eps_values"]).flatten().astype(float)
    cases = []
    ids = []
    for ii, inst in enumerate(instances):
        for jj, eps in enumerate(eps_values):
            cases.append((ii, jj, int(inst), float(eps)))
            ids.append(f"inst{int(inst)}_eps{float(eps):.0e}")
    metafunc.parametrize("cross_tool_case", cases, ids=ids)


def test_n2v_contains_matlab_box(cross_tool_case, reference, model):
    """n2v bounds must over-approximate MATLAB bounds at every output."""
    ii, jj, inst_idx, eps = cross_tool_case
    X = model.X_test[inst_idx - 1]      # MATLAB indices are 1-based.
    gs_in = GraphStar.from_bounds(X - eps, X + eps, adjacency=model.A_norm)
    outs = gcn_stack_reach(gs_in, model.gcn_layers, has_relu=model.has_relu)
    assert len(outs) == 1
    lb_n2v, ub_n2v = outs[0].get_ranges()

    lb_mat = np.asarray(reference["lb"][ii, jj], dtype=float)
    ub_mat = np.asarray(reference["ub"][ii, jj], dtype=float)
    assert lb_mat.shape == lb_n2v.shape

    # Soundness: n2v's box must contain MATLAB's box, modulo LP tolerance.
    slack_lb = lb_mat - lb_n2v
    slack_ub = ub_n2v - ub_mat
    worst = float(min(slack_lb.min(), slack_ub.min()))
    assert worst >= -SOUNDNESS_SLACK, (
        f"n2v box does not contain MATLAB box for instance {inst_idx} eps {eps}. "
        f"Worst slack = {worst:+.3e} (allowed -{SOUNDNESS_SLACK:.0e}). "
        f"min(matlab_lb - n2v_lb) = {slack_lb.min():+.3e}, "
        f"min(n2v_ub - matlab_ub) = {slack_ub.min():+.3e}."
    )


def test_small_eps_tightness_matches_matlab(reference, model):
    """At eps <= 1e-2, n2v's mean bound width should be within 25% of MATLAB.

    Larger eps trigger the known LP-vs-estimate ReLU gap; that regime is
    informational only (not asserted).
    """
    instances = reference["instances"]
    eps_values = reference["eps_values"]
    inflation = []
    for ii, inst in enumerate(instances):
        X = model.X_test[int(inst) - 1]
        for jj, eps in enumerate(eps_values):
            if float(eps) > 1e-2:
                continue
            gs = GraphStar.from_bounds(X - float(eps), X + float(eps),
                                       adjacency=model.A_norm)
            outs = gcn_stack_reach(gs, model.gcn_layers, has_relu=model.has_relu)
            lb_n2v, ub_n2v = outs[0].get_ranges()
            w_n2v = (ub_n2v - lb_n2v)
            w_mat = (np.asarray(reference["ub"][ii, jj], dtype=float)
                     - np.asarray(reference["lb"][ii, jj], dtype=float))
            mask = w_mat > WIDTH_FLOOR
            if mask.any():
                inflation.append(float((w_n2v[mask] / w_mat[mask]).mean()))

    if not inflation:
        pytest.skip("no eps <= 1e-2 cells in reference")
    worst_mean = max(inflation)
    assert worst_mean < 1.25, (
        f"at eps <= 1e-2 n2v mean bound width inflated to {worst_mean:.3f}x MATLAB; "
        f"expected < 1.25x"
    )

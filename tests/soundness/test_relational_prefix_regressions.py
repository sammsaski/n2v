"""Regression: the relational join must not read an output star's predicates
as the joint input predicates unless the network's reach keeps them.

A softmax kernel re-encodes its output with ``Star.from_bounds``, so the first
predicates of its output star are fresh output-box variables. Identifying them
with the joint input predicates invents a correlation between the input and
``Y_f`` and gave a false UNSAT.
"""

import numpy as np
import torch
import torch.nn as nn

from n2v.sets.halfspace import HalfSpace
from n2v.nn.relational import relational_reach, verify_relational
from tests._oracles.membership import uncovered


def _linear(W, b=None):
    W = np.asarray(W, dtype=np.float64)
    lin = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        lin.weight[:] = torch.tensor(W)
        lin.bias[:] = torch.tensor(np.zeros(W.shape[0]) if b is None else b)
    return lin


def _spec(n_out_f, n_out_g, prop_G, prop_g):
    """Joint input [x_f; x_g] in [0,1]^2 coupled by x_f == x_g, with one
    unsafe HalfSpace over the joint output [Y_f; Y_g]."""
    return {
        "format": "relational",
        "networks": [
            {"name": "f", "relation": None, "input_offset": 0,
             "input_size": 1, "output_offset": 0, "output_size": n_out_f},
            {"name": "g", "relation": None, "input_offset": 1,
             "input_size": 1, "output_offset": n_out_f,
             "output_size": n_out_g},
        ],
        "lb": np.zeros(2), "ub": np.ones(2),
        "input_coupling": HalfSpace(np.array([[1.0, -1.0], [-1.0, 1.0]]),
                                    np.zeros((2, 1))),
        "prop": [{"Hg": HalfSpace(np.asarray(prop_G, dtype=float),
                                  np.asarray(prop_g, dtype=float))}],
    }


def _softmax_case():
    # y_f0 = sigmoid(2x), y_g = x. Unsafe: y_f0 <= 0.55 and |y_g| <= 0.05,
    # which x = 0 reaches (y = [0.5, 0.5, 0]).
    f = nn.Sequential(_linear([[1.0], [-1.0]]), nn.Softmax(dim=1)).double()
    g = nn.Sequential(_linear([[1.0]])).double()
    G = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    gv = [[0.55], [0.05], [0.05]]
    return f.eval(), g.eval(), _spec(2, 1, G, gv), np.array(G), np.array(gv)


def _joint_output(f, g, x):
    xt = torch.tensor([[x]], dtype=torch.float64)
    with torch.no_grad():
        return np.concatenate([f(xt).numpy().ravel(), g(xt).numpy().ravel()])


def test_softmax_network_is_not_falsely_unsat():
    f, g, spec, G, gv = _softmax_case()
    y = _joint_output(f, g, 0.0).reshape(-1, 1)
    assert np.all(G @ y <= gv)          # the unsafe region is reachable
    assert verify_relational(f, g, spec, method="approx").verdict != "UNSAT"


def test_softmax_join_contains_true_joint_outputs():
    """Every joint output of 129 allowed pairs (x_f = x_g = x on a grid of
    [0, 1], ends included), replayed in float64, is a member of the joint
    star by an LP on its own rows (``membership.uncovered``); the first 11
    are also checked with ``Star.contains``. Catches the softmax network's
    first predicates dropped or read as the input, swapped right-hand sides
    or own bounds in the join, and g reached with model_f."""
    f, g, spec, _, _ = _softmax_case()
    joint = relational_reach(f, g, spec, method="approx")
    for x in np.linspace(0.0, 1.0, 11):
        assert joint.contains(_joint_output(f, g, x)), f"x={x}"
    xs = np.linspace(0.0, 1.0, 129)
    missed = uncovered([joint], [_joint_output(f, g, x) for x in xs])
    assert not missed, (f"{len(missed)} of 129 joint outputs outside the "
                        f"joint star, e.g. x = {xs[missed[0][0]]}")


def test_linear_relu_networks_keep_the_shared_input():
    # Every ReLU is active on [0, 1], so f(x_f) == f(x_g) exactly when
    # x_f == x_g. Y_f - Y_g >= 0.1 is then provably unreachable, but only
    # through the shared input predicates.
    f = nn.Sequential(_linear([[1.0], [2.0]], [0.5, 0.5]), nn.ReLU(),
                      _linear([[1.0, -1.0]])).double().eval()
    spec = _spec(1, 1, [[-1.0, 1.0]], [[-0.1]])
    assert verify_relational(f, f, spec, method="approx").verdict == "UNSAT"


def test_relu_networks_with_own_predicates_keep_the_shared_input():
    """f = g = relu(x) on x in [-0.25, 4] with x_f == x_g: the ReLU is
    unstable, so each output star appends one own predicate after the two
    shared input predicates. The two relaxations differ by at most the
    triangle's height 4 * 0.25 / 4.25 < 0.5, so |Y_f - Y_g| >= 0.5 is
    unreachable and provable only through the shared input predicates.
    Catches a prefix test that rejects every output star with own
    predicates (the join then loses the coupling: UNKNOWN)."""
    f = nn.Sequential(_linear([[1.0]]), nn.ReLU(),
                      _linear([[1.0]])).double().eval()
    for G in ([[-1.0, 1.0]], [[1.0, -1.0]]):       # B - D, D - B >= 0.5
        spec = _spec(1, 1, G, [[-0.5]])
        spec["lb"] = np.array([-0.25, -0.25])
        spec["ub"] = np.array([4.0, 4.0])
        verdict = verify_relational(f, f, spec, method="approx").verdict
        assert verdict == "UNSAT", (G, verdict)

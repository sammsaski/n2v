"""Regression tests for :func:`n2v.utils.verify_nonlinear.verify_nonlinear_reach`
reading output predicates as input coordinates.

The verifier used to share the input predicates with every output star whose
first predicate bounds matched the input's. A star re-encoded with
``Star.from_bounds`` (the softmax, softmax-attention and bilinear MatMul
kernels) also has ``[-1, 1]`` predicates, so ``x`` was read from predicates
that describe an unrelated output box. That invented an ``x``/``y``
correlation and proved reachable unsafe regions empty (false UNSAT).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from n2v.nn import NeuralNetwork
from n2v.sets import Star
from n2v.utils.vnnlib2 import evaluate_nonlinear, load_vnnlib_v2
from n2v.utils.verify_nonlinear import (
    network_keeps_input_predicates, verify_nonlinear_reach,
)


def _spec(n_out, output_assert):
    """x in [0, 1], a trivially true product term (forces the nonlinear
    format), plus one output assertion."""
    return load_vnnlib_v2("mem.vnnlib", text=(
        "(vnnlib-version <2.0>)\n"
        "(declare-network N (declare-input X float32 [1])"
        f" (declare-output Y float32 [{n_out}]))\n"
        "(assert (and (>= X[0] 0.0) (<= X[0] 1.0)))\n"
        "(assert (<= (* X[0] X[0]) 1.0))\n"
        f"(assert {output_assert})\n"
    ))


def _linear_softmax():
    """y = softmax([x, -x]), so y0 = sigmoid(2x) in [0.5, 0.881] on [0, 1]."""
    lin = nn.Linear(1, 2)
    with torch.no_grad():
        lin.weight[:] = torch.tensor([[1.0], [-1.0]])
        lin.bias.zero_()
    return nn.Sequential(lin, nn.Softmax(dim=1)).eval()


def test_from_bounds_output_is_not_read_as_input():
    """y = 1 - x on [0, 1], over-approximated by the box star y in [0, 1]
    that a ``Star.from_bounds`` kernel emits. The unsafe region
    ``y - x >= 0.5`` is reachable at x = 0 (y = 1); reading the output's
    fresh predicate as x made y - x == 0 and returned UNSAT."""
    p = _spec(1, "(>= (- Y[0] X[0]) 0.5)")
    assert p["format"] == "nonlinear"
    assert evaluate_nonlinear(p, np.array([0.0]), np.array([1.0]))
    iset = Star.from_bounds(np.array([0.0]), np.array([1.0]))
    out = Star.from_bounds(np.array([0.0]), np.array([1.0]))
    assert np.array_equal(out.predicate_lb, iset.predicate_lb)  # bounds match
    assert verify_nonlinear_reach([out], iset, p) == "UNKNOWN"


def test_linear_softmax_reach_is_not_falsely_unsat():
    """Real reach through Linear+Softmax (the softmax kernel re-encodes its
    output with ``Star.from_bounds``). ``y0 - 0.3808 x >= 0.52`` holds at
    x = 0.5 (0.5407), so the verdict must not be UNSAT."""
    p = _spec(2, "(>= (- Y[0] (* 0.3808 X[0])) 0.52)")
    model = _linear_softmax()
    with torch.no_grad():
        y = model(torch.tensor([[0.5]])).numpy().ravel().astype(np.float64)
    assert evaluate_nonlinear(p, np.array([0.5]), y)

    net = NeuralNetwork(model)
    iset = Star.from_bounds(np.array([0.0]), np.array([1.0]))
    reach = net.reach(iset, method="approx", input_shape=(1,))
    assert network_keeps_input_predicates(net) is False
    assert verify_nonlinear_reach(reach, iset, p) == "UNKNOWN"
    assert verify_nonlinear_reach(
        reach, iset, p,
        input_predicates_shared=network_keeps_input_predicates(net),
    ) == "UNKNOWN"


def test_network_keeps_input_predicates_allowlist():
    """Only graphs made of Linear/ReLU/Identity modules share predicates;
    a softmax module or any traced function call does not."""
    assert network_keeps_input_predicates(
        NeuralNetwork(nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))))
    assert network_keeps_input_predicates(nn.Sequential(nn.Identity()))
    assert not network_keeps_input_predicates(NeuralNetwork(_linear_softmax()))

    class FunctionalReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

        def forward(self, x):
            return F.relu(self.fc(x))

    assert not network_keeps_input_predicates(FunctionalReLU())
    assert not network_keeps_input_predicates(object())

    class ModelAttrSoftmax(nn.Module):
        # A plain module whose submodule is named ``model``: the helper must
        # vet the whole forward, not just ``self.model``.
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(1, 2))
            self.head = nn.Softmax(dim=1)

        def forward(self, x):
            return self.head(self.model(x))

    assert not network_keeps_input_predicates(ModelAttrSoftmax())
    assert not network_keeps_input_predicates(NeuralNetwork(ModelAttrSoftmax()))


def test_shared_predicates_keep_joint_model_on_linear_relu():
    """y = ReLU(x) on x in [0.5, 1] is always active, so y == x and
    ``y - x >= 0.25`` is unreachable. The append-only Linear/ReLU reach keeps
    the input predicates, so the shared model proves it; the default
    disjoint model cannot (y - x in [-0.5, 0.5])."""
    p = load_vnnlib_v2("mem.vnnlib", text=(
        "(vnnlib-version <2.0>)\n"
        "(declare-network N (declare-input X float32 [1])"
        " (declare-output Y float32 [1]))\n"
        "(assert (and (>= X[0] 0.5) (<= X[0] 1.0)))\n"
        "(assert (<= (* X[0] X[0]) 1.0))\n"
        "(assert (>= (- Y[0] X[0]) 0.25))\n"
    ))
    lin = nn.Linear(1, 1)
    with torch.no_grad():
        lin.weight.fill_(1.0)
        lin.bias.zero_()
    net = NeuralNetwork(nn.Sequential(lin, nn.ReLU()).eval())
    iset = Star.from_bounds(np.array([0.5]), np.array([1.0]))
    reach = net.reach(iset, method="approx", input_shape=(1,))
    shared = network_keeps_input_predicates(net)
    assert shared is True
    assert verify_nonlinear_reach(
        reach, iset, p, input_predicates_shared=shared) == "UNSAT"
    assert verify_nonlinear_reach(reach, iset, p) == "UNKNOWN"


def test_prefix_bound_mismatch_falls_back_to_disjoint_model():
    """With sharing asserted but prefix bounds that differ from the input's,
    x gets its own predicates instead of giving up: the output bound alone
    (y in [1, 2]) still refutes ``y >= 10``."""
    p = _spec(1, "(>= Y[0] 10.0)")
    iset = Star.from_bounds(np.array([0.0]), np.array([1.0]))
    out = Star(V=np.array([[1.0, 1.0]]), C=None, d=None,  # y in [1, 2]
               pred_lb=np.array([0.0]), pred_ub=np.array([1.0]))
    assert verify_nonlinear_reach(
        [out], iset, p, input_predicates_shared=True) == "UNSAT"

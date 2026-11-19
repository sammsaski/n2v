"""Add and Sub of two computed sets outside the ONNX graph walk: a PyTorch
model whose forward adds or subtracts two branches, and the ONNX layer
handed to ``reach_layer`` on its own.

The sum or difference of two computed sets is answered only in the graph
walk of an ONNX model (case 1 of ``_handle_onnx_binary_op`` in
``n2v/nn/reach.py``; its cells are in ``test_soundness_set_arith_cells.py``).
The other two entry routes of the same rows are checked here:

- a torch model traced by ``torch.fx`` (``reach_pytorch_model`` and
  ``NeuralNetwork.reach``), where ``a + b``, ``a - b``, ``torch.add`` and
  ``torch.sub`` become ``call_function`` nodes. Today they have no reach rule
  and raise ``NotImplementedError``; whatever answers them later must hold
  every output of the model, checked by torch float64 replay and one LP per
  point on the rows of the returned sets;
- ``reach_layer`` with the ``OnnxBinaryMathOperation`` module: the call
  carries one list of sets, so the second operand is not there to be added,
  and every set type must refuse with ``NotImplementedError``.

The older sampling tests of this file (``_add_sets`` called directly,
bounding-box checks with float32 slack) were removed: each
duplicated a cell of ``test_soundness_set_arith_cells.py`` that reaches the
same code through the model entry with an exact reference.
"""

import operator

import numpy as np
import pytest
import torch
import torch.nn as nn
from onnx2torch.node_converters.binary_math_operations import \
    OnnxBinaryMathOperation

from n2v.nn.layer_ops.dispatcher import reach_layer
from n2v.nn.neural_network import NeuralNetwork
from tests.cell_support import (FLAT_KINDS, IMAGE_KINDS, assert_covered,
                                flat_box, flat_set, image_set, pixel_box,
                                reach_model, replay_torch, samples)

# Dyadic weights: fc1 is affine, fc2 feeds a ReLU whose first neuron crosses
# zero on the box below, the second is positive and the third negative.
_W1 = np.array([[1.0, -0.5], [0.25, 0.75], [-1.0, 0.5]])
_B1 = np.array([0.5, -0.25, 1.0])
_W2 = np.array([[0.5, 1.0], [1.0, 0.25], [-0.5, -0.75]])
_B2 = np.array([0.125, 4.0, -4.0])
_LO, _HI = np.array([-2.0, -1.5]), np.array([2.0, 1.0])


class _TwoBranchJoin(nn.Module):
    """``join(fc1(x), relu(fc2(x)))``, the shape of a residual block. The
    parameters are frozen, as a user hands a trained model over."""

    def __init__(self, join):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(2, 3)
        with torch.no_grad():
            for layer, (W, b) in ((self.fc1, (_W1, _B1)),
                                  (self.fc2, (_W2, _B2))):
                layer.weight.copy_(torch.tensor(W, dtype=torch.float32))
                layer.bias.copy_(torch.tensor(b, dtype=torch.float32))
        self.requires_grad_(False)
        self.join = join

    def forward(self, x):
        return self.join(self.fc1(x), torch.relu(self.fc2(x)))


@pytest.mark.parametrize("join", [operator.add, operator.sub, torch.add,
                                  torch.sub],
                         ids=["a+b", "a-b", "torch.add", "torch.sub"])
def test_torch_add_or_sub_of_two_branches_refuses_or_holds_every_output(join):
    """``fc1(x) +- relu(fc2(x))`` as a torch model, reached on a Star, a Zono
    and a Box through ``reach_pytorch_model`` and on a Star through
    ``NeuralNetwork.reach``. A refusal (``NotImplementedError``, today's
    answer) passes; an answer must hold the torch float64 output of 128
    member points, decided by one LP per point on the returned rows. Catches
    a ``call_function`` add or sub that is answered by passing one operand
    through, or that is given the other operation's sign."""
    model = _TwoBranchJoin(join).eval()
    outputs = replay_torch(model, samples(_LO, _HI), [1, 2])
    routes = [("reach_pytorch_model", kind) for kind in ("Star", "Zono",
                                                         "Box")]
    routes.append(("NeuralNetwork.reach", "Star"))
    for route, kind in routes:
        what = f"{route} of a torch model joining two branches on a {kind}"
        input_set = flat_set(kind, _LO, _HI)
        try:
            if route == "NeuralNetwork.reach":
                sets = NeuralNetwork(model).reach(input_set, method="approx")
            else:
                sets = reach_model(model, input_set, "approx", what)
        except NotImplementedError:
            continue
        if not sets:
            pytest.fail(f"{what}: no set returned for a non-empty input")
        assert_covered(sets, outputs, what)


@pytest.mark.parametrize("op", ["Add", "Sub"])
def test_the_onnx_add_or_sub_layer_handed_one_operand_is_refused(op):
    """``reach_layer(OnnxBinaryMathOperation(op), [s], method)`` for each of
    the seven set types and both methods: the layer joins two computed
    values and the call holds only one, so it must raise
    ``NotImplementedError``. Catches a dispatcher that passes the operand
    through (or treats the layer as any other one-input layer), which
    returns one branch where the sum of two is due."""
    layer = OnnxBinaryMathOperation(op)
    lo, hi = flat_box(3)
    plo, phi = pixel_box(2, 3, 5)
    inputs = [(kind, flat_set(kind, lo, hi)) for kind in FLAT_KINDS]
    inputs += [(kind, image_set(kind, plo, phi)) for kind in IMAGE_KINDS]
    for kind, input_set in inputs:
        for method in ("exact", "approx"):
            with pytest.raises(NotImplementedError):
                reach_layer(layer, [input_set], method)

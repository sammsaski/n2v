"""Mul and Div of two computed sets outside the ONNX graph walk: a PyTorch
model whose forward multiplies or divides two branches, and the ONNX layer
handed to ``reach_layer`` on its own.

The product or quotient of two computed sets is answered only in the graph
walk of an ONNX model (case 1 of ``_handle_onnx_binary_op`` in
``n2v/nn/reach.py``, with ``_mul_sets`` and ``_div_sets``; its cells are in
``test_soundness_set_arith_cells.py``). The other two entry routes of the
same rows are checked here:

- a torch model traced by ``torch.fx`` (``reach_pytorch_model`` and
  ``NeuralNetwork.reach``), where ``a * b``, ``a / b``, ``torch.mul`` and
  ``torch.div`` become ``call_function`` nodes. Today they have no reach
  rule and raise ``NotImplementedError``; whatever answers them later must
  hold every output of the model, checked by torch float64 replay and one LP
  per point on the rows of the returned sets;
- ``reach_layer`` with the ``OnnxBinaryMathOperation`` module: the call
  carries one list of sets, so the second operand is not there, and every
  set type must refuse with ``NotImplementedError``.

The four older cases of this file (Mul and Div by a constant, with
``_mul_sets_by_constant`` called directly and a bounding-box check with
float32 slack) were removed: each duplicated a cell of
``test_soundness_const_arith_cells.py``
(``test_constant_on_a_flat_set_holds_the_exact_image[Star-Mul-full]``,
``[Star-Div-full]``, ``[Zono-Mul-full]``, ``[Box-Mul-full]``), which reach the
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

# Dyadic weights. fc1: three entries that cross zero, stay positive and stay
# negative on the box below. fc2 feeds the second operand: for the product
# through a ReLU whose first neuron crosses zero; for the quotient directly,
# its biases keep every entry away from zero (in [2, 13/2], [-43/8, -11/4]
# and [29/16, 49/8] on the box).
_W1 = np.array([[1.0, -0.5], [0.25, 0.75], [-1.0, 0.5]])
_B1 = np.array([0.5, 2.0, -3.0])
_W2 = np.array([[0.5, 1.0], [-0.5, 0.25], [1.0, 0.125]])
_B2 = {"product": np.array([0.125, 4.0, -4.0]),
       "quotient": np.array([4.5, -4.0, 4.0])}
_LO, _HI = np.array([-2.0, -1.5]), np.array([2.0, 1.0])


class _TwoBranchJoin(nn.Module):
    """``join(fc1(x), g(fc2(x)))`` with g a ReLU for the product and the
    identity for the quotient. The parameters are frozen, as a user hands a
    trained model over."""

    def __init__(self, join, kind):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(2, 3)
        with torch.no_grad():
            for layer, (W, b) in ((self.fc1, (_W1, _B1)),
                                  (self.fc2, (_W2, _B2[kind]))):
                layer.weight.copy_(torch.tensor(W, dtype=torch.float32))
                layer.bias.copy_(torch.tensor(b, dtype=torch.float32))
        self.requires_grad_(False)
        self.join = join
        self.relu = kind == "product"

    def forward(self, x):
        second = self.fc2(x)
        if self.relu:
            second = torch.relu(second)
        return self.join(self.fc1(x), second)


@pytest.mark.parametrize("join, kind", [
    (operator.mul, "product"), (operator.truediv, "quotient"),
    (torch.mul, "product"), (torch.div, "quotient")],
    ids=["a*b", "a/b", "torch.mul", "torch.div"])
def test_torch_product_or_quotient_of_two_branches_refuses_or_holds_every_output(
        join, kind):
    """``fc1(x) * relu(fc2(x))`` and ``fc1(x) / fc2(x)`` as a torch model,
    reached on a Star, a Zono and a Box through ``reach_pytorch_model`` and on
    a Star through ``NeuralNetwork.reach``. A refusal
    (``NotImplementedError``, today's answer) passes; an answer must hold the
    torch float64 output of 128 member points, decided by one LP per point on
    the returned rows. Catches a ``call_function`` product or quotient that
    is answered by passing one operand through (or as another operation)."""
    model = _TwoBranchJoin(join, kind).eval()
    outputs = replay_torch(model, samples(_LO, _HI), [1, 2])
    routes = [("reach_pytorch_model", k) for k in ("Star", "Zono", "Box")]
    routes.append(("NeuralNetwork.reach", "Star"))
    for route, set_kind in routes:
        what = f"{route} of a torch model, {kind} of two branches, {set_kind}"
        input_set = flat_set(set_kind, _LO, _HI)
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


@pytest.mark.parametrize("op", ["Mul", "Div"])
def test_the_onnx_mul_or_div_layer_handed_one_operand_is_refused(op):
    """``reach_layer(OnnxBinaryMathOperation(op), [s], method)`` for each of
    the seven set types and both methods: the layer multiplies or divides
    two computed values and the call holds only one, so it must raise
    ``NotImplementedError``. Catches a dispatcher that passes the operand
    through (or treats the layer as any other one-input layer), which
    returns one branch where the product or quotient of two is due."""
    layer = OnnxBinaryMathOperation(op)
    lo, hi = flat_box(3)
    plo, phi = pixel_box(2, 3, 5)
    inputs = [(kind, flat_set(kind, lo, hi)) for kind in FLAT_KINDS]
    inputs += [(kind, image_set(kind, plo, phi)) for kind in IMAGE_KINDS]
    for kind, input_set in inputs:
        for method in ("exact", "approx"):
            with pytest.raises(NotImplementedError):
                reach_layer(layer, [input_set], method)

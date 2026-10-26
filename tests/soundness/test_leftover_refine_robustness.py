"""
Regression tests for the refine drivers' input finitization and AvgPool2d
handling.

- ``_finite_input_star`` let a ``ValueError`` from
  ``ensure_finite_predicate_box`` escape, so both drivers (``verify_refine``
  and ``verify_refine_input_crown``) crashed on an input star with no
  trustworthy finite predicate box (an unbounded predicate variable, an
  'infeasible' report that emptiness confirmation does not back, bound LPs
  that contradict each other). They now return UNKNOWN. ``verify_refine``
  still raises on its own argument errors (a malformed spec, an unknown
  solver name) for such a star: it checks them before the finitizer.
- ``ConvLayer`` applied ``nn.AvgPool2d(ceil_mode=True,
  count_include_pad=True)`` with torch's border divisor. When the last
  window overhangs the padded input, ONNX AveragePool (onnxruntime) divides
  it by the full kernel area instead, and the core reach refuses such a
  layer. The refine path proved ``y <= 0.5`` unreachable for a model whose
  ONNX origin outputs 0.25 there (a false UNSAT for the ONNX model); it now
  refuses the same layers as the core reach.

Convention: SAT = unsafe (the unsafe region is ``G y <= g``), UNSAT = safe.
"""

import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.layer_ops import avgpool2d_reach
from n2v.refine import (FaithfulSelector, LinearSpec, Status, verify_refine,
                        verify_refine_input_crown)
from n2v.refine.reach_relaxed import ConvLayer, extract_layers
from n2v.sets.star import Star


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 4), nn.ReLU(),
                         nn.Linear(4, 2)).double()


def _run(driver, S, net, spec):
    if driver == "verify_refine":
        return verify_refine(S, net, spec, FaithfulSelector(),
                             bound_mode="box", prune_mode="exact",
                             witness_mode="exact", frontier_width=1,
                             node_budget=100)
    return verify_refine_input_crown(S, net, spec, node_budget=100)


def _inf_box_star(C, d):
    V = np.hstack([np.zeros((2, 1)), np.eye(2)])
    inf = np.full((2, 1), np.inf)
    return Star(V, C, d, -inf, inf.copy())


_DRIVERS = ["verify_refine", "verify_refine_input_crown"]


# --------------------------------------------------------------------------- #
# Input finitization failures give UNKNOWN                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("driver", _DRIVERS)
def test_unbounded_input_star_is_unknown(driver, caplog):
    """No C rows and an infinite box: the bound LPs report the predicate
    variables unbounded and the finitizer raises. Both drivers crashed."""
    S = _inf_box_star(np.zeros((0, 2)), np.zeros((0, 1)))
    spec = LinearSpec([[1.0, 0.0]], [1e9])
    with caplog.at_level(logging.WARNING, logger="n2v.refine.bab"):
        res = _run(driver, S, _net(), spec)
    assert res.status is Status.UNKNOWN and res.nodes == 0
    assert "could not be finitized" in caplog.text


@pytest.mark.parametrize("driver", _DRIVERS)
def test_unconfirmed_empty_input_star_is_not_a_crash(driver):
    """``1e-3 alpha_0 <= -1e-8`` and ``1e-3 alpha_0 >= 1e-8``: empty, but at
    a scale where the emptiness confirmation does not certify it, so the
    finitizer raises. The set is genuinely empty, so UNSAT would be correct
    too; SAT or an exception is not."""
    C = np.array([[1e-3, 0.0], [-1e-3, 0.0], [0.0, 1.0], [0.0, -1.0]])
    d = np.array([[-1e-8], [-1e-8], [1.0], [0.0]])
    res = _run(driver, _inf_box_star(C, d), _net(),
               LinearSpec([[1.0, 0.0]], [1e9]))
    assert res.status in (Status.UNKNOWN, Status.UNSAT)


@pytest.mark.parametrize("driver", _DRIVERS)
def test_contradicting_bound_lps_give_unknown(driver, monkeypatch):
    """Whatever the finitizer's reason for raising, the run is UNKNOWN."""
    def _raise(star, lp_solver=None):
        raise ValueError("the bound LPs contradict each other")

    monkeypatch.setattr("n2v.refine.bab.ensure_finite_predicate_box", _raise)
    res = _run(driver, Star.from_bounds(np.array([-1.0, -1.0]),
                                        np.array([1.0, 1.0])),
               _net(), LinearSpec([[1.0, 0.0]], [1e9]))
    assert res.status is Status.UNKNOWN and res.nodes == 0


_BAD_ARGS = {
    "empty spec list": (TypeError, dict(spec=[])),
    "non-spec entry": (TypeError, dict(spec=["not a spec"])),
    "multi-disjunct with contract_inputs": (
        TypeError, dict(spec=[LinearSpec([[1.0, 0.0]], [1e9])] * 2,
                        contract_inputs=True)),
    "unknown solver name": (ValueError, dict(lp_solver="bogus_solver")),
}


@pytest.mark.parametrize("case", list(_BAD_ARGS))
def test_argument_errors_raise_even_when_the_box_cannot_be_finitized(case):
    """``verify_refine`` checks its arguments before the finitizer, whose
    ValueErrors become UNKNOWN: on an unbounded input star a malformed spec
    or an unknown solver name came back as UNKNOWN instead of an error."""
    exc, kw = _BAD_ARGS[case]
    kw = dict(kw)
    spec = kw.pop("spec", LinearSpec([[1.0, 0.0]], [1e9]))
    S = _inf_box_star(np.zeros((0, 2)), np.zeros((0, 1)))
    with pytest.raises(exc):
        verify_refine(S, _net(), spec, FaithfulSelector(), bound_mode="box",
                      prune_mode="exact", witness_mode="exact",
                      frontier_width=1, **kw)


# --------------------------------------------------------------------------- #
# AvgPool2d: the refine path refuses what the core reach refuses              #
# --------------------------------------------------------------------------- #
def _corner_pool_net():
    """y = the last (overhanging) window of a k2 s2 ceil_mode pool on 5x5.
    torch divides that window by 1, onnxruntime by 4."""
    pool = nn.AvgPool2d(2, stride=2, ceil_mode=True, count_include_pad=True)
    net = nn.Sequential(nn.Unflatten(1, (1, 5, 5)), pool, nn.Flatten(),
                        nn.Linear(9, 1)).double()
    with torch.no_grad():
        net[3].weight.zero_()
        net[3].weight[0, 8] = 1.0
        net[3].bias.zero_()
    return net


def test_overhanging_count_include_pad_pool_is_refused():
    """For x in [0.9, 1.1]^25 torch gives y in [0.9, 1.1] and the ONNX model
    y in [0.225, 0.275]; the refine path returned UNSAT for ``y <= 0.5``."""
    net = _corner_pool_net()
    with pytest.raises(NotImplementedError):
        extract_layers(net)
    S = Star.from_bounds(np.full(25, 0.9), np.full(25, 1.1))
    with pytest.raises(NotImplementedError):
        verify_refine(S, net, LinearSpec([[1.0]], [0.5]), FaithfulSelector(),
                      bound_mode="box", prune_mode="exact",
                      witness_mode="exact", frontier_width=1)


def test_hand_built_conv_layer_is_checked_on_the_module_output():
    """A ConvLayer built by hand is checked too, on the module's own output
    size rather than the declared ``out_shape``."""
    pool = nn.AvgPool2d(2, stride=2, ceil_mode=True, count_include_pad=True)
    with pytest.raises(NotImplementedError):
        ConvLayer(pool, (1, 5, 5), (1, 2, 2))


_POOL_CASES = [
    (dict(kernel_size=2, stride=2, ceil_mode=True), (1, 5, 5)),
    (dict(kernel_size=3, stride=2, padding=1, ceil_mode=True), (2, 6, 6)),
    (dict(kernel_size=2, stride=2, padding=1, ceil_mode=True), (2, 5, 6)),
    (dict(kernel_size=2, stride=2, ceil_mode=True), (1, 4, 4)),
    (dict(kernel_size=2, stride=2, ceil_mode=True,
          count_include_pad=False), (1, 5, 5)),
    (dict(kernel_size=2, stride=2, ceil_mode=True, divisor_override=3),
     (1, 5, 5)),
    (dict(kernel_size=3, stride=1, padding=1), (2, 5, 6)),
]


@pytest.mark.parametrize("cfg,shape", _POOL_CASES)
def test_refine_refuses_exactly_what_the_core_reach_refuses(cfg, shape):
    pool = nn.AvgPool2d(**cfg)
    try:
        avgpool2d_reach._avg_pool(pool, torch.zeros(1, *shape,
                                                    dtype=torch.float64))
        core_refuses = False
    except NotImplementedError:
        core_refuses = True
    net = nn.Sequential(nn.Unflatten(1, shape), pool, nn.Flatten()).double()
    try:
        extract_layers(net)
        refine_refuses = False
    except NotImplementedError:
        refine_refuses = True
    assert refine_refuses == core_refuses

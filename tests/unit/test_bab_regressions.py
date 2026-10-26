"""Regressions for the ReLU neuron-split BaB (``verify_bab_relu``).

- Forced neurons must be pinned to their sign even though the LP ranges are
  padded outward, so a fully split leaf is exact (no triangle predicates).
- The unstable list must be exactly the neurons the triangle ReLU relaxes: a
  dead neuron (zero row) is never branched on.
- ``relax_factor=0`` (exact ReLU splitting, several stars per layer) must not
  reach the single-star bounding step, which used to keep only the first star
  and return a false VERIFIED.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.sets import Star
from n2v.sets.halfspace import HalfSpace
from n2v.nn.bab import _seq_reach_relu, verify_bab_relu


def _two_neuron_net(w_in, w_out):
    """Linear(1,2) -> ReLU -> Linear(2,1), no biases, float64."""
    m = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 1)).double()
    with torch.no_grad():
        m[0].weight[:] = torch.tensor(w_in, dtype=torch.float64)
        m[0].bias.zero_()
        m[2].weight[:] = torch.tensor(w_out, dtype=torch.float64)
        m[2].bias.zero_()
    m.eval()
    return m


def _unsafe_geq(thresh):  # unsafe {y >= thresh} == {-y <= -thresh}
    return HalfSpace(np.array([[-1.0]]), np.array([[-thresh]]))


def _dead_neuron_net():
    torch.manual_seed(1)
    m = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1))
    with torch.no_grad():
        m[0].weight[4:] = 0.0                  # neurons 4..9 are dead
        m[0].bias[4:] = 0.0
    m.eval()
    return m


@pytest.mark.parametrize("decisions, lo, hi", [
    ({0: 1, 1: -1}, 0.0, 1.0),                  # x >= 0: y = x
    ({0: -1, 1: 1}, -1.0, 0.0),                 # x <= 0: y = x
])
def test_forced_neurons_are_pinned_exactly(decisions, lo, hi):
    # y = relu(x) - relu(-x) = x on [-1, 1]. With both neurons forced the leaf
    # is exact: no triangle predicate, and the range is not widened by the
    # 1e-7 LP padding.
    m = _two_neuron_net([[1.0], [-1.0]], [[1.0, -1.0]])
    s = Star.from_bounds(np.array([-1.0]), np.array([1.0]))
    out, unstable, empty = _seq_reach_relu(list(m), s, {0: decisions}, 0.5, "default")
    assert not empty
    assert unstable == []
    assert out.nVar == s.nVar
    olb, oub = out.get_ranges()
    assert float(np.ravel(olb)[0]) == pytest.approx(lo, abs=1e-12)
    assert float(np.ravel(oub)[0]) == pytest.approx(hi, abs=1e-12)


def test_fully_split_leaf_proves_tight_margin():
    # y = 1000 * x on x in [-1e-3, 1e-3] (so y in [-1, 1]); unsafe y >= 1 + 1e-5
    # is truly safe. Before the fix the forced leaves stayed relaxed and the
    # 1e-7 padding, amplified by the 1000x output weights, left the bound
    # about 1e-4 loose, so the search ended UNKNOWN with nothing to split.
    m = _two_neuron_net([[1.0], [-1.0]], [[1000.0, -1000.0]])
    res = verify_bab_relu(m, np.array([-1e-3]), np.array([1e-3]),
                          _unsafe_geq(1.0 + 1e-5), falsify_method=None,
                          max_nodes=50)
    assert res.verdict == "VERIFIED", res


def test_unstable_list_matches_relaxed_neurons():
    # Each relaxed neuron adds one predicate; every one of them must be
    # offered for branching, and nothing else (dead neurons stay [0, 0]).
    m = _dead_neuron_net()
    s = Star.from_bounds(-np.ones(2), np.ones(2))
    out, unstable, empty = _seq_reach_relu(list(m), s, {}, 0.5, "default")
    assert not empty
    assert all(n < 4 for _, n, _, _ in unstable), unstable
    assert out.nVar - s.nVar == len(unstable)


def test_dead_neurons_are_not_branched():
    # Unprovable spec with the falsifier off: the search must end UNKNOWN
    # after splitting only live neurons (it used to split all 6 dead ones too).
    m = _dead_neuron_net()
    X = np.random.default_rng(0).uniform(-1, 1, (20000, 2)).astype(np.float32)
    with torch.no_grad():
        ymax = float(m(torch.from_numpy(X)).max())
    res = verify_bab_relu(m, -np.ones(2), np.ones(2), _unsafe_geq(ymax - 0.01),
                          falsify_method=None, max_nodes=2000)
    assert res.verdict == "UNKNOWN", res
    assert res.max_depth <= 4, res              # at most the 4 live neurons


def test_relax_factor_zero_is_rejected():
    # y = |x| on [-1, 1] reaches 1 >= 0.9, so the property is violated.
    # relax_factor=0 used to keep the first of relu_star_exact's stars (range
    # [0, 0]) and return VERIFIED.
    m = _two_neuron_net([[1.0], [-1.0]], [[1.0, 1.0]])
    with torch.no_grad():
        assert float(m(torch.tensor([[1.0]], dtype=torch.float64))) >= 0.9
    spec = _unsafe_geq(0.9)
    with pytest.raises(ValueError, match="relax_factor"):
        verify_bab_relu(m, np.array([-1.0]), np.array([1.0]), spec,
                        falsify_method=None, relax_factor=0.0)
    res = verify_bab_relu(m, np.array([-1.0]), np.array([1.0]), spec,
                          falsify_method=None, relax_factor=1.0)
    assert res.verdict == "UNKNOWN", res


def test_multi_star_relu_output_fails_loudly(monkeypatch):
    # The bounding step handles one star per node; a split ReLU output must
    # raise instead of silently dropping all but the first star.
    import n2v.nn.layer_ops.relu_reach as relu_reach

    real = relu_reach.relu_star_approx

    def two_stars(*args, **kwargs):
        out = real(*args, **kwargs)
        return out + out

    monkeypatch.setattr(relu_reach, "relu_star_approx", two_stars)
    m = _two_neuron_net([[1.0], [-1.0]], [[1.0, 1.0]])
    with pytest.raises(RuntimeError, match="expected 1"):
        verify_bab_relu(m, np.array([-1.0]), np.array([1.0]), _unsafe_geq(0.9),
                        falsify_method=None)

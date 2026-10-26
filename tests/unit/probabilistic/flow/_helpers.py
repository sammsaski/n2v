"""Shared helpers for unit tests in tests/unit/probabilistic/flow/.

Consolidates duplicate copies of ``_train_small_2d_flow`` that previously
lived inline in several test modules (test_certify_halfspace_disjoint,
test_certify_adaptive_n, test_scenario_sampling_qmc, test_amls). All
copies trained the same toy flow on N(0, I_2) with sinkhorn coupling,
EMA, and uniform time sampling; the only divergence was ``n_epochs``
(100 vs 200), so we adopt the 200-epoch canonical version used by the
majority of call sites.
"""
from __future__ import annotations

import copy
import functools

import numpy as np
import torch


def _train_small_2d_flow(seed: int = 0):
    """Tiny 2D flow trained on N(0, I_2), reused across flow unit tests.

    Training is deterministic in ``seed`` and costs about 3 s, so the flow is
    trained once per seed per process and every caller gets its own deep
    copy: a test that mutates its copy (``train()`` mode, a device move)
    cannot affect the next test.
    """
    return copy.deepcopy(_train_small_2d_flow_cached(seed))


@functools.lru_cache(maxsize=None)
def _train_small_2d_flow_cached(seed: int = 0):
    """The trained flow for ``seed`` (see ``_train_small_2d_flow``).

    The output distribution is Gaussian by construction (toy target).
    Uses sinkhorn OT coupling with auto regularization, EMA weights, and
    uniform time sampling for stable training in ~200 epochs on 2000
    samples.
    """
    from n2v.probabilistic.flow.model import VelocityField
    from n2v.probabilistic.flow.ode import FlowODE
    from n2v.probabilistic.flow.train import train_flow

    torch.manual_seed(seed)
    vf = VelocityField(
        dim=2, hidden=64, n_layers=2,
        activation='silu', time_embed='concat',
    )
    rng = np.random.default_rng(seed)
    y_train = torch.from_numpy(
        rng.standard_normal((2000, 2)).astype(np.float32)
    )
    vf, _ = train_flow(
        vf, y_train, n_epochs=200, batch_size=512, lr=1e-3,
        coupling='sinkhorn', sinkhorn_reg='auto', sinkhorn_iters=5,
        use_ema=True, standardize_outputs=False, time_sampling='uniform',
    )
    vf.eval()
    return FlowODE(vf)

"""Shared helpers for the refine unit tests."""

import numpy as np
import torch
import torch.nn as nn

from n2v.refine.types import LinearSpec, Status


def seq_net(*dims, seed=0):
    """Build a random FC ReLU net (Linear/ReLU/.../Linear) in float64.

    Seeded inside a forked RNG scope so the global torch generator is left as
    the caller found it."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        return nn.Sequential(*layers).double()


def _assert_honest_ce(res, net, specs, lb, ub):
    """On a SAT result, the counterexample must lie in the raw float64 input
    box ``[lb, ub]`` and the float64 forward pass must land in the closed
    unsafe region of at least one spec in ``specs`` (a LinearSpec or a list of
    disjuncts), both at zero tolerance. Any other status is not checked."""
    if res.status != Status.SAT:
        return
    if isinstance(specs, LinearSpec):
        specs = [specs]
    assert res.counterexample_x is not None, "SAT without a counterexample"
    x = np.asarray(res.counterexample_x, dtype=np.float64).flatten()
    assert np.all(x >= np.asarray(lb, dtype=np.float64).flatten()), \
        "counterexample below the input box"
    assert np.all(x <= np.asarray(ub, dtype=np.float64).flatten()), \
        "counterexample above the input box"
    with torch.no_grad():
        y = net(torch.as_tensor(x, dtype=torch.float64).unsqueeze(0))
    y = y.numpy().flatten()
    assert any(np.all(np.asarray(sp.G) @ y - np.asarray(sp.g).flatten() <= 0.0)
               for sp in specs), "counterexample is not in any unsafe region"

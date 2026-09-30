"""
Counterexample search (falsification) for GNN output-box specifications.

Sound reachability returns "verified" or "unknown".  Falsification attacks the
"unknown" gap from the other side: it searches the input set for a concrete
node-feature matrix whose forward output escapes the spec box.  A hit upgrades
"unknown" to "falsified" with a witness; a miss leaves "unknown" (search is
not exhaustive).

The GNN forward map is piecewise-linear (affine layers + ReLU), so the
violation margin is piecewise-linear over the input box and its maximum sits at
a vertex.  The search therefore combines uniform random multi-start with
coordinate ascent that pushes each input coordinate to a box extreme — far more
effective at reaching vertices than random sampling alone.  No autograd needed.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class FalsifyResult:
    """Outcome of a counterexample search."""
    found: bool
    counterexample: Optional[np.ndarray] = None   # (N, F_in) input
    output: Optional[np.ndarray] = None           # (N, F_out) violating output
    margin: float = 0.0                            # max box-escape (>0 => violation)


def _violation_margin(Y, spec_lb, spec_ub, node_mask):
    """Max amount any selected output entry escapes [spec_lb, spec_ub] (>0 = violation)."""
    v = np.maximum(Y - spec_ub, spec_lb - Y)       # (N, F)
    if node_mask is not None:
        v = v[node_mask]
    return float(v.max()) if v.size else float("-inf")


def falsify_node_bounds(
    evaluate: Callable[[np.ndarray], np.ndarray],
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    spec_lb: np.ndarray,
    spec_ub: np.ndarray,
    target_nodes: Optional[Sequence[int]] = None,
    n_random: int = 2000,
    n_restarts: int = 8,
    tol: float = 1e-9,
    seed: int = 0,
) -> FalsifyResult:
    """Search the input box for a point whose output leaves the spec box.

    Args:
        evaluate: forward map X (N, F_in) -> Y (N, F_out) (e.g. net.evaluate).
        input_lb, input_ub: (N, F_in) input perturbation box.
        spec_lb, spec_ub: (N, F_out) output box the property claims to hold.
        target_nodes: restrict the spec to these node indices (default: all).
        n_random: uniform random samples in the first phase.
        n_restarts: coordinate-ascent restarts (from best random + random corners).
        tol: a margin above this counts as a genuine counterexample.

    Returns:
        FalsifyResult; ``found`` is True iff a counterexample was located.
    """
    input_lb = np.asarray(input_lb, dtype=np.float64)
    input_ub = np.asarray(input_ub, dtype=np.float64)
    spec_lb = np.asarray(spec_lb, dtype=np.float64)
    spec_ub = np.asarray(spec_ub, dtype=np.float64)
    rng = np.random.default_rng(seed)
    N, F_in = input_lb.shape

    node_mask = None
    if target_nodes is not None:
        node_mask = np.zeros(N, dtype=bool)
        node_mask[list(target_nodes)] = True

    best_X, best_margin, best_Y = None, float("-inf"), None

    def consider(X):
        nonlocal best_X, best_margin, best_Y
        Y = evaluate(X)
        m = _violation_margin(Y, spec_lb, spec_ub, node_mask)
        if m > best_margin:
            best_X, best_margin, best_Y = X.copy(), m, Y
        return m

    # ---- phase 1: uniform random multi-start ------------------------------
    consider(0.5 * (input_lb + input_ub))                      # center first
    for _ in range(n_random):
        X = input_lb + rng.uniform(size=(N, F_in)) * (input_ub - input_lb)
        if consider(X) > tol:
            return FalsifyResult(True, best_X, best_Y, best_margin)

    # ---- phase 2: coordinate ascent to box vertices -----------------------
    starts = [best_X.copy()]
    for _ in range(n_restarts - 1):
        # random vertex: each coordinate independently at lb or ub
        pick = rng.integers(0, 2, size=(N, F_in)).astype(bool)
        starts.append(np.where(pick, input_ub, input_lb))

    for X0 in starts:
        X = X0.copy()
        improved = True
        while improved:
            improved = False
            for i in range(N):
                for j in range(F_in):
                    cur = _violation_margin(evaluate(X), spec_lb, spec_ub, node_mask)
                    for val in (input_lb[i, j], input_ub[i, j]):
                        if val == X[i, j]:
                            continue
                        old = X[i, j]
                        X[i, j] = val
                        m = _violation_margin(evaluate(X), spec_lb, spec_ub, node_mask)
                        if m > cur + 1e-12:
                            cur, improved = m, True
                            if m > best_margin:
                                best_X, best_margin, best_Y = X.copy(), m, evaluate(X)
                            if m > tol:
                                return FalsifyResult(True, best_X, best_Y, best_margin)
                        else:
                            X[i, j] = old

    return FalsifyResult(False, None, None, best_margin)

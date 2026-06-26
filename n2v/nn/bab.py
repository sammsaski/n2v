"""Branch-and-bound verification via input-domain splitting.

General and model-agnostic: BaB recursively bisects the input box, bounds each
subdomain with the sound reach engine, tries to falsify it, and recurses on the
ones that remain UNKNOWN. It only ever touches the *input box* and then reuses
the layer-general reach / verify_specification / falsify machinery, so it works
for any n2v model and any set of layers — it is not specific to any architecture.

Soundness (by construction):
  * a subdomain is pruned only when the sound bounder proves it safe (UNSAT);
  * FALSIFIED is returned only with a concrete counterexample (a real input
    whose output violates the property);
  * VERIFIED is returned only when every leaf of a *covering* bisection is
    proven safe (the two children of a split exactly tile the parent box);
  * UNKNOWN is returned when the node/time budget is exhausted — never a
    robustness claim.

This is the complete (search) layer on top of the incomplete (bounding) reach
engine; it is implemented from scratch (no external verifier).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from n2v.utils.verify_specification import verify_specification


@dataclass
class BaBResult:
    verdict: str                       # 'VERIFIED' | 'FALSIFIED' | 'UNKNOWN'
    counterexample: Optional[np.ndarray] = None
    nodes: int = 0                     # subdomains bounded
    splits: int = 0                    # branch operations
    max_depth: int = 0
    time_s: float = 0.0
    reason: str = ""


def _default_bound_fn(reach_sets, spec) -> bool:
    """Sound safety decision: True iff the reach set is provably disjoint from
    the unsafe region (verify_specification UNSAT)."""
    return verify_specification(reach_sets, spec).verdict == "UNSAT"


def star_input_sensitivity(reach_sets, spec, n_input: int) -> Optional[np.ndarray]:
    """Per-input-dimension sensitivity of the binding margin, for branching.

    For a Star reach output whose leading ``n_input`` predicates are the input
    box dimensions (the case when the input was built by ``from_bounds`` and the
    network only applied affine maps + relaxation-appended predicates), the
    generator coefficient of output row k on input predicate j is the (linear
    part of the) sensitivity d Y_k / d alpha_j. We score each input dim by the
    largest such coefficient magnitude across the spec's halfspace directions —
    i.e. the dim whose perturbation most moves the margin we must bound. Returns
    None when the leading-input-predicate assumption can't be applied.
    """
    from n2v.sets import Star

    try:
        groups = _spec_halfspaces(spec)
    except Exception:  # noqa: BLE001
        return None
    score = np.zeros(n_input)
    for s in reach_sets:
        if not isinstance(s, Star) or s.nVar < n_input:
            return None
        G = s.V[:, 1:1 + n_input]          # (dim, n_input) input-pred generators
        for H in groups:                    # H.G: (rows, dim)
            # sensitivity of each halfspace row's value to each input dim
            contrib = np.abs(np.asarray(H.G, dtype=np.float64) @ G)  # (rows, n_input)
            score = np.maximum(score, contrib.max(axis=0))
    return score


def _spec_halfspaces(spec) -> List:
    """Flatten the spec to a list of HalfSpace objects (for sensitivity only)."""
    from n2v.sets.halfspace import HalfSpace

    out: List = []

    def _walk(x):
        if isinstance(x, HalfSpace):
            out.append(x)
        elif isinstance(x, dict) and "Hg" in x:
            _walk(x["Hg"])
        elif isinstance(x, (list, tuple)):
            for e in x:
                _walk(e)
    _walk(spec)
    return out


def _pick_split_dim(lb, ub, reach_sets, spec, branch, sens_fn) -> int:
    width = ub - lb
    active = width > 1e-12
    if not np.any(active):
        return int(np.argmax(width))
    if branch == "widest":
        return int(np.argmax(width))
    # sensitivity * width: the dim that most reduces the bound when halved
    sens = None
    if sens_fn is not None:
        sens = sens_fn(reach_sets, spec, lb.size)
    if sens is None:
        return int(np.argmax(width))
    score = sens * width
    score[~active] = -np.inf
    return int(np.argmax(score))


def verify_bab(
    reach_fn: Callable[[np.ndarray, np.ndarray], list],
    lb: np.ndarray,
    ub: np.ndarray,
    spec,
    *,
    falsify_fn: Optional[Callable[[np.ndarray, np.ndarray], Optional[np.ndarray]]] = None,
    bound_fn: Optional[Callable[[list, object], bool]] = None,
    sensitivity_fn: Optional[Callable] = star_input_sensitivity,
    branch: str = "sensitivity",
    max_nodes: int = 1000,
    timeout_s: Optional[float] = None,
    verbose: bool = False,
    _clock: Optional[Callable[[], float]] = None,
) -> BaBResult:
    """Branch-and-bound verify the property ``spec`` over the input box [lb, ub].

    Args:
        reach_fn(lb, ub): returns a list of sound reach sets (Star/Box) for the
            output over the given input sub-box.
        lb, ub: input box (any shape; flattened internally for splitting).
        spec: unsafe region (HalfSpace / list / dict), as verify_specification
            and falsify accept.
        falsify_fn(lb, ub): returns a concrete counterexample input (array) if
            it finds the sub-box unsafe, else None. Optional but recommended —
            it ends the search early on truly-unsafe instances.
        bound_fn(reach_sets, spec): sound safety test; default
            verify_specification UNSAT.
        sensitivity_fn / branch: branching heuristic (see module docstring).
        max_nodes / timeout_s: budget; exhausting it yields UNKNOWN.
    """
    bound_fn = bound_fn or _default_bound_fn
    clock = _clock or time.monotonic
    t0 = clock()
    shape = np.asarray(lb).shape
    lb0 = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub0 = np.asarray(ub, dtype=np.float64).reshape(-1)

    # DFS stack of (lb, ub, depth); LIFO keeps memory ~ depth.
    stack: List = [(lb0, ub0, 0)]
    nodes = splits = max_depth = 0
    while stack:
        if nodes >= max_nodes:
            return BaBResult("UNKNOWN", None, nodes, splits, max_depth,
                             clock() - t0, f"node budget {max_nodes} exhausted")
        if timeout_s is not None and clock() - t0 > timeout_s:
            return BaBResult("UNKNOWN", None, nodes, splits, max_depth,
                             clock() - t0, f"timeout {timeout_s}s")
        slb, sub, depth = stack.pop()
        max_depth = max(max_depth, depth)
        nodes += 1
        reach_sets = reach_fn(slb.reshape(shape), sub.reshape(shape))
        if bound_fn(reach_sets, spec):
            if verbose:
                print(f"  node {nodes} d{depth}: SAFE (pruned)", flush=True)
            continue                                   # subdomain proven safe
        if falsify_fn is not None:
            cex = falsify_fn(slb.reshape(shape), sub.reshape(shape))
            if cex is not None:
                return BaBResult("FALSIFIED", np.asarray(cex), nodes, splits,
                                 max_depth, clock() - t0,
                                 "counterexample found")
        dim = _pick_split_dim(slb, sub, reach_sets, spec, branch, sensitivity_fn)
        mid = 0.5 * (slb[dim] + sub[dim])
        if not (slb[dim] < mid < sub[dim]):            # cannot split further
            return BaBResult("UNKNOWN", None, nodes, splits, max_depth,
                             clock() - t0,
                             f"degenerate split at dim {dim} (depth {depth})")
        left_ub = sub.copy(); left_ub[dim] = mid
        right_lb = slb.copy(); right_lb[dim] = mid
        stack.append((slb, left_ub, depth + 1))
        stack.append((right_lb, sub, depth + 1))
        splits += 1
        if verbose:
            print(f"  node {nodes} d{depth}: split dim {dim} "
                  f"[{slb[dim]:.4g},{sub[dim]:.4g}]@{mid:.4g}", flush=True)
    return BaBResult("VERIFIED", None, nodes, splits, max_depth,
                     clock() - t0, "all subdomains proven safe")


def _seq_reach_relu(layers, input_star, splits, relax_factor, lp_solver):
    """Reach an nn.Sequential of Linear/Flatten/ReLU with forced ReLU signs.

    ``splits`` maps relu-layer index -> {neuron: +1 (active) | -1 (inactive)}.
    Forced neurons are pinned by intersecting the pre-activation star with the
    sign halfspace; LP ranges (which respect those constraints) are passed to the
    triangle ReLU so a forced neuron is treated exactly (stable). Returns
    (out_star, undecided_unstable_list, empty)."""
    from n2v.nn.layer_ops.linear_reach import linear_star
    from n2v.nn.layer_ops.relu_reach import relu_star_approx

    s = input_star
    relu_id = 0
    unstable = []
    for layer in layers:
        name = type(layer).__name__
        if name == "Linear":
            s = linear_star(layer, [s])[0]
        elif name == "Flatten":
            pass
        elif name == "ReLU":
            dec = splits.get(relu_id, {})
            if dec:
                rows, gs = [], []
                for n, sign in dec.items():
                    r = np.zeros(s.dim)
                    r[n] = -1.0 if sign > 0 else 1.0   # s_n>=0  or  s_n<=0
                    rows.append(r); gs.append(0.0)
                s = s.intersect_half_space(np.array(rows), np.array(gs))
                if s.is_empty_set(lp_solver):
                    return None, [], True              # infeasible -> vacuously safe
            lb, ub = s.get_ranges(lp_solver=lp_solver)
            lbf, ubf = lb.reshape(-1), ub.reshape(-1)
            for n in range(s.dim):
                if n not in dec and lbf[n] < -1e-9 and ubf[n] > 1e-9:
                    unstable.append((relu_id, n, float(lbf[n]), float(ubf[n])))
            s = relu_star_approx([s], relax_factor=relax_factor,
                                 lp_solver=lp_solver, precomputed_bounds=(lb, ub))[0]
            relu_id += 1
        else:
            raise NotImplementedError(f"verify_bab_relu: unsupported layer {name}")
    return s, unstable, False


def verify_bab_relu(
    model,
    lb: np.ndarray,
    ub: np.ndarray,
    spec,
    *,
    falsify_method: Optional[str] = "random+pgd",
    falsify_kwargs: Optional[dict] = None,
    relax_factor: float = 0.5,
    lp_solver: str = "default",
    max_nodes: int = 1000,
    timeout_s: Optional[float] = None,
    verbose: bool = False,
    _clock: Optional[Callable[[], float]] = None,
) -> BaBResult:
    """ReLU **neuron-split** branch-and-bound for an ``nn.Sequential`` of
    Linear / Flatten / ReLU (the canonical complete-verification BaB, the right
    split space for ReLU classifiers — unlike input splitting).

    The input region is fixed; the search branches on unstable ReLU neurons
    (forcing active/inactive), which exactly tile the parent, so it is sound and
    complete-in-the-limit. Prunes a subdomain when the bound proves it safe or
    the forced signs are infeasible; falsifies the (fixed) input box once.
    """
    from n2v.sets import Star
    from n2v.utils.falsify import falsify as _falsify

    clock = _clock or time.monotonic
    t0 = clock()
    layers = list(model)
    lb0 = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub0 = np.asarray(ub, dtype=np.float64).reshape(-1)
    input_star = Star.from_bounds(lb0, ub0)

    # The input region is identical at every node, so falsify once up front.
    if falsify_method is not None:
        res, cex = _falsify(model, lb0, ub0, spec, method=falsify_method,
                            **(falsify_kwargs or {}))
        if res == 0 and cex is not None:
            return BaBResult("FALSIFIED", np.asarray(cex[0]), 0, 0, 0,
                             clock() - t0, "counterexample found")

    stack = [{}]
    nodes = splits = max_depth = 0
    while stack:
        if nodes >= max_nodes:
            return BaBResult("UNKNOWN", None, nodes, splits, max_depth,
                             clock() - t0, f"node budget {max_nodes} exhausted")
        if timeout_s is not None and clock() - t0 > timeout_s:
            return BaBResult("UNKNOWN", None, nodes, splits, max_depth,
                             clock() - t0, f"timeout {timeout_s}s")
        decisions = stack.pop()
        depth = sum(len(v) for v in decisions.values())
        max_depth = max(max_depth, depth)
        nodes += 1
        out, unstable, empty = _seq_reach_relu(
            layers, input_star, decisions, relax_factor, lp_solver)
        if empty:
            continue                                   # infeasible split -> safe
        if verify_specification([out], spec).verdict == "UNSAT":
            continue                                   # subdomain proven safe
        if not unstable:
            return BaBResult("UNKNOWN", None, nodes, splits, max_depth,
                             clock() - t0,
                             "bound loose with no unstable ReLU left to split")
        # branch the most balanced unstable neuron (largest min(|lb|, ub))
        relu_id, n, nlb, nub = max(unstable, key=lambda t: min(-t[2], t[3]))
        for sign in (1, -1):
            child = {k: dict(v) for k, v in decisions.items()}
            child.setdefault(relu_id, {})[n] = sign
            stack.append(child)
        splits += 1
        if verbose:
            print(f"  node {nodes} d{depth}: split relu{relu_id} neuron {n} "
                  f"[{nlb:.3g},{nub:.3g}]", flush=True)
    return BaBResult("VERIFIED", None, nodes, splits, max_depth,
                     clock() - t0, "all subdomains proven safe")


def verify_bab_model(
    model,
    lb: np.ndarray,
    ub: np.ndarray,
    spec,
    *,
    set_type=None,
    reach_method: str = "approx",
    falsify_method: Optional[str] = "random+pgd",
    falsify_kwargs: Optional[dict] = None,
    **bab_kwargs,
) -> BaBResult:
    """Branch-and-bound verify a property for a torch ``nn.Module`` using n2v's
    standard reach (bounding) + falsify (search) — the general toolbox entry.

    Builds a ``reach_fn`` from ``NeuralNetwork(model).reach`` (default Star) and
    a ``falsify_fn`` from ``n2v.utils.falsify.falsify``, then calls
    :func:`verify_bab`. Layer-agnostic: works for any model n2v's reach handles.
    """
    from n2v.nn.neural_network import NeuralNetwork
    from n2v.sets import Star
    from n2v.utils.falsify import falsify as _falsify

    set_type = set_type or Star
    net = NeuralNetwork(model)
    fk = falsify_kwargs or {}

    def reach_fn(l, u):
        s = set_type.from_bounds(np.asarray(l).reshape(-1), np.asarray(u).reshape(-1))
        return net.reach(s, method=reach_method)

    falsify_fn = None
    if falsify_method is not None:
        def falsify_fn(l, u):  # noqa: E306
            res, cex = _falsify(model, np.asarray(l), np.asarray(u), spec,
                                method=falsify_method, **fk)
            if res == 0 and cex is not None:
                return np.asarray(cex[0])
            return None

    return verify_bab(reach_fn, lb, ub, spec, falsify_fn=falsify_fn, **bab_kwargs)

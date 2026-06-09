"""
LP-based SNN verification core.

Contains the triangle relaxation LP construction functions for F2F SNN verification.
Modified from external_snnv/snn_comparison.py to accept per-dimension input bounds
directly (bypassing the image_flat + epsilon interface), enabling integration with
n2v's Star and Box set types.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import product as iproduct

import numpy as np
import torch

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    class _tqdm:  # no-op stand-in for tqdm when not installed
        def __init__(self, it=None, **kw):
            self._it = it
        def __iter__(self):
            return iter(self._it)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            pass

from n2v.snn.model import F2FMLP
from n2v.snn.encoding import latency_from_values, spike_train_from_latencies


# ---------------------------------------------------------------------------
# LP helpers
# ---------------------------------------------------------------------------

def feasible_latencies(lb: float, ub: float, num_steps: int) -> list[int]:
    """Return the set of timesteps t that are consistent with input value in [lb, ub].

    A timestep t is feasible if the value interval [x_low(t), x_high(t)]
    corresponding to latency t overlaps with the perturbation interval [lb, ub].

    The value that maps to latency t satisfies:
        x_low(t)  = 1 - (t + 0.5) / (T-1)   (lower edge of the latency-t bin)
        x_high(t) = 1 - (t - 0.5) / (T-1)   (upper edge)

    The silent latency T is feasible whenever lb <= 0 (the input might become
    zero or below-zero after perturbation).
    """
    vals = []

    # Silent (T) is feasible if the lower bound of the input value is <= 0,
    # meaning the input could become background after perturbation.
    if lb <= 0.0:
        vals.append(num_steps)  # sentinel T = silent

    lo = max(lb, np.nextafter(0.0, 1.0))  # only non-zero values can fire
    hi = ub
    if hi <= 0.0:
        return vals  # all perturbations force silence, no finite latency is feasible

    for t in range(num_steps):
        # Compute the value bin edges for latency t.
        x_low = 1.0 - (t + 0.5) / (num_steps - 1)
        x_high = 1.0 - (t - 0.5) / (num_steps - 1)
        x_low = max(0.0, x_low)
        x_high = min(1.0, x_high)
        # Feasible if the bins overlap: max(lo, x_low) <= min(hi, x_high).
        if max(lo, x_low) <= min(hi, x_high):
            vals.append(t)
    return sorted(set(vals))


def make_bounds(image_flat: np.ndarray, indices: np.ndarray, epsilon: float):
    """Build per-pixel lower and upper bound arrays for the perturbation set.

    For selected pixels: lb[i] = clip(x[i] - ε, 0, 1), ub[i] = clip(x[i] + ε, 0, 1).
    For unselected pixels: lb[i] = ub[i] = x[i] (no perturbation).
    """
    lb = image_flat.copy()
    ub = image_flat.copy()
    for idx in indices:
        lb[idx] = np.clip(image_flat[idx] - epsilon, 0.0, 1.0)
        ub[idx] = np.clip(image_flat[idx] + epsilon, 0.0, 1.0)
    return lb, ub


ALL_SPLIT_STRATEGIES = ["random", "selected", "influence", "choice", "choice-influence"]

# Mutable context dict populated before each LP solve so that failure messages
# can report which image/pixels triggered the infeasibility.  Updated by
# SNNVerifier.verify (image_idx) and build_symbolic_relaxation_lp (pixel info).
_LP_CONTEXT: dict = {}

# When False, equality constraints (a@x = b) are converted to pairs of
# inequalities (a@x <= b, -a@x <= -b) before passing to the solver.
# Set by SNNVerifier.verify; controlled by --equality-constraints CLI flag.
_USE_EQ_CONSTRAINTS: bool = True

# When True, print LP context (image, pixels, dimensions) for every LP build,
# regardless of success or failure.  Controlled by --debug-lp CLI flag.
_DEBUG_LP: bool = False

# When True, add explicit at-most-once LP constraints (Σ_t spk[i,t] <= 1) for
# every hidden and output neuron.  Off by default because these constraints can
# become contradictory with the lower-triangle bounds.  Controlled by --amo.
_USE_AMO_CONSTRAINTS: bool = False


def order_split_indices(model, indices: np.ndarray, lb_x: np.ndarray, ub_x: np.ndarray,
                        num_steps: int, strategy: str = "selected") -> np.ndarray:
    """Return the pixel indices in the order they should be split on first.

    When the symbolic split descends to depth d, it branches on the first d
    pixels in this ordering. Splitting on the 'most important' pixels first
    tends to prune the LP branch tree more aggressively.

    Strategies:
      selected:         keep original index order (no reordering)
      influence:        sum of |W1[:, pix]| — pixels with large first-layer
                        weight impact are split first
      choice:           number of feasible latency values — ambiguous pixels
                        (widest perturbation interval) are split first
      choice-influence: influence * max(choices - 1, 0) — splits pixels that
                        are both high-influence AND have multiple feasible
                        latencies; pixels with only one feasible latency need
                        not be split at all (they contribute 0 to this score)
      random:           shuffled with a fixed seed (for ablations)
    """
    if strategy == "selected":
        return indices  # no reordering

    # First-layer weight matrix: shape (hidden_size, input_size).
    W1 = model.fcs[0].weight.detach().cpu().numpy()

    # Sum of absolute weights for each selected pixel across all hidden neurons.
    influence = np.sum(np.abs(W1[:, indices]), axis=0)  # shape (k,)

    # Number of feasible latency values for each selected pixel.
    choice_counts = np.array([
        len(feasible_latencies(float(lb_x[p]), float(ub_x[p]), num_steps))
        for p in indices
    ], dtype=np.float64)  # shape (k,)

    if strategy == "influence":
        scores = influence
    elif strategy == "choice":
        scores = choice_counts
    elif strategy == "choice-influence":
        # Pixels with only one feasible latency are already determined; no
        # benefit to splitting on them, so subtract 1 before multiplying.
        scores = influence * np.maximum(choice_counts - 1.0, 0.0)
    elif strategy == "random":
        shuffled = indices.copy()
        np.random.default_rng(42).shuffle(shuffled)
        return shuffled
    elif strategy.startswith("split_"):
        # Manual override: "split_0" puts index 0 first, "split_1_2" puts
        # indices 1 and 2 first. Used for debugging specific pixel orderings.
        parts = strategy.split("_")[1:]
        if len(parts) == 1:
            pos = int(parts[0])
            if pos >= len(indices):
                raise ValueError(f"{strategy}: position {pos} out of range for k={len(indices)}")
            return np.concatenate([[indices[pos]], np.delete(indices, pos)])
        elif len(parts) == 2:
            pos0, pos1 = int(parts[0]), int(parts[1])
            if pos0 >= len(indices) or pos1 >= len(indices):
                raise ValueError(f"{strategy}: positions out of range for k={len(indices)}")
            rest = [i for i in range(len(indices)) if i != pos0 and i != pos1]
            return np.array([indices[pos0], indices[pos1]] + [indices[i] for i in rest])
        else:
            raise ValueError(f"Unsupported split strategy: {strategy}")
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    # Sort descending: highest-scoring pixels are split first.
    return indices[np.argsort(scores)[::-1]]


# ---------------------------------------------------------------------------
# LP solver internals
# ---------------------------------------------------------------------------

def _linear_interval(center: float, coeffs: np.ndarray,
                     bounds: list[tuple[float, float]]) -> tuple[float, float]:
    """Compute a fast interval bound for a linear expression: center + coeffs @ vars.

    Uses the current variable bounds (not an LP) to get a quick lower/upper
    estimate. This is used for the at-most-one membrane potential bounds
    (u_lb_amo, u_ub_amo) when tight_bounds=False.
    """
    lb = center
    ub = center
    nz = np.nonzero(coeffs)[0]
    for j in nz:
        lo, hi = bounds[j]
        val = coeffs[j]
        if val >= 0.0:
            lb += val * lo   # positive coeff: lb uses variable's lb
            ub += val * hi   # positive coeff: ub uses variable's ub
        else:
            lb += val * hi   # negative coeff: lb uses variable's ub
            ub += val * lo   # negative coeff: ub uses variable's lb
    return float(lb), float(ub)


def _solve_bound(obj: np.ndarray, A_ub: list[np.ndarray], b_ub: list[float],
                 A_eq: list[np.ndarray], b_eq: list[float],
                 bounds: list[tuple[float, float]], maximize: bool = False) -> float:
    """Solve a single LP to get the tightest bound on the linear objective 'obj'.

    Used only when tight_bounds=True to recompute u_lb and u_ub via LP rather
    than the cheaper interval arithmetic. This is much slower but produces
    tighter neuron-firing bounds.
    """
    lp_matrices = _prepare_lp_matrices(A_ub, b_ub, A_eq, b_eq)
    return _solve_bound_prepared(obj, lp_matrices, bounds, maximize=maximize)


def _prepare_lp_matrices(A_ub: list[np.ndarray], b_ub: list[float],
                         A_eq: list[np.ndarray], b_eq: list[float]):
    """Convert lists of constraint rows into sparse matrices for the LP solver.

    If _USE_EQ_CONSTRAINTS is False, each equality a@x = b is replaced by the
    pair a@x <= b and -a@x <= -b, so A_eq_mat/b_eq_vec are passed as None.
    This can help when equality constraints cause spurious infeasibility.

    If SNNV_LP_BACKEND=highspy is set, returns a _HighsPreparedLP object that
    caches the HiGHS model and only swaps out the objective vector between
    solves — faster than re-building the LP from scratch each time.

    Otherwise returns (A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec) for scipy.linprog.
    """
    from scipy import sparse

    if not _USE_EQ_CONSTRAINTS:
        # Drop equality constraints entirely — the LP feasible set is strictly
        # larger, so certification is less tight but the solver cannot become
        # infeasible due to an over-constrained equality.
        A_eq = []
        b_eq = []

    A_ub_mat = sparse.csr_matrix(np.vstack(A_ub)) if A_ub else None
    b_ub_vec = np.asarray(b_ub, dtype=np.float64) if A_ub else None
    A_eq_mat = sparse.csr_matrix(np.vstack(A_eq)) if A_eq else None
    b_eq_vec = np.asarray(b_eq, dtype=np.float64) if A_eq else None
    lp_matrices = A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec
    if os.environ.get("SNNV_LP_BACKEND", "").lower() == "highspy":
        return _HighsPreparedLP(lp_matrices, bounds=None)
    return lp_matrices


class _HighsPreparedLP:
    """Optional HiGHS-backed LP wrapper that reuses the model across objective changes.

    Building a HiGHS model is expensive. This class builds it once on the first
    call to solve(), then reuses it for subsequent solves by changing only the
    cost vector. This is important when solving many LPs with the same constraint
    matrix but different objectives (as happens in build_symbolic_relaxation_lp).
    """
    def __init__(self, lp_matrices, bounds):
        self.lp_matrices = lp_matrices
        self.bounds = bounds
        self.highs = None       # lazily initialized on first solve
        self.highspy = None
        self.indices = None

    def _init_model(self, bounds: list[tuple[float, float]]):
        """Build the HiGHS model from the constraint matrices and variable bounds."""
        from scipy import sparse
        import highspy

        A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec = self.lp_matrices

        # Combine inequality and equality constraints into a single row-ranged
        # system: row_lower[i] <= A[i,:] @ x <= row_upper[i].
        mats = []
        lower_parts = []
        upper_parts = []
        if A_ub_mat is not None:
            mats.append(A_ub_mat)
            lower_parts.append(np.full(A_ub_mat.shape[0], -highspy.kHighsInf))
            upper_parts.append(b_ub_vec)
        if A_eq_mat is not None:
            mats.append(A_eq_mat)
            lower_parts.append(b_eq_vec)
            upper_parts.append(b_eq_vec)
        if mats:
            A = sparse.vstack(mats, format="csr")
            row_lower = np.concatenate(lower_parts).astype(np.float64)
            row_upper = np.concatenate(upper_parts).astype(np.float64)
        else:
            A = sparse.csr_matrix((0, len(bounds)), dtype=np.float64)
            row_lower = np.empty(0, dtype=np.float64)
            row_upper = np.empty(0, dtype=np.float64)

        lower = np.asarray([lo for lo, _ in bounds], dtype=np.float64)
        upper = np.asarray([hi for _, hi in bounds], dtype=np.float64)
        lower[np.isneginf(lower)] = -highspy.kHighsInf
        upper[np.isposinf(upper)] = highspy.kHighsInf

        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)    # suppress HiGHS console output
        highs.setOptionValue("threads", int(os.environ.get("SNNV_HIGHS_THREADS", "1")))
        highs.setOptionValue("solver", "simplex")
        highs.addVars(len(bounds), lower, upper)
        highs.addRows(
            A.shape[0], row_lower, row_upper,
            A.nnz, A.indptr.astype(np.int32),
            A.indices.astype(np.int32), A.data.astype(np.float64),
        )
        self.highs = highs
        self.highspy = highspy
        self.indices = np.arange(len(bounds), dtype=np.int32)
        self.bounds = bounds

    def solve(self, obj: np.ndarray, bounds: list[tuple[float, float]],
              maximize: bool = False) -> float:
        """Solve the LP by swapping only the cost vector and re-running HiGHS."""
        from scipy.optimize import linprog

        if self.highs is None:
            self._init_model(bounds)
        c = np.asarray(-obj if maximize else obj, dtype=np.float64)
        self.highs.changeColsCost(len(c), self.indices, c)
        self.highs.run()
        status = self.highs.getModelStatus()
        if status != self.highspy.HighsModelStatus.kOptimal:
            # Fall back to scipy if HiGHS fails (e.g., infeasible sub-problem).
            A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec = self.lp_matrices
            res = linprog(c, A_ub=A_ub_mat, b_ub=b_ub_vec,
                          A_eq=A_eq_mat, b_eq=b_eq_vec, bounds=bounds, method="highs")
            if not res.success:
                ctx = _LP_CONTEXT
                print(f"  [LP-fail/highs] status={res.status} msg={res.message!r} "
                      f"maximize={maximize}")
                print(f"    image_idx={ctx.get('image_idx', '?')}  "
                      f"label={ctx.get('label', '?')}  "
                      f"epsilon={ctx.get('epsilon', '?')}")
                pidx = ctx.get("pixel_indices", [])
                pval = ctx.get("pixel_values", [])
                print(f"    perturbed pixels (in selection order): {pidx}")
                print(f"    pixel values at those positions:       "
                      f"{[round(v, 4) for v in pval]}")
                fixed = ctx.get("fixed_latencies") or {}
                if fixed:
                    print(f"    fixed latencies (split branch):        {fixed}")
                _diagnose_infeasible(A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec, bounds)
                return float("inf") if maximize else float("-inf")
            return float(-res.fun if maximize else res.fun)
        value = float(self.highs.getInfo().objective_function_value)
        return -value if maximize else value


def _diagnose_infeasible(A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec,
                         bounds: list[tuple[float, float]],
                         max_report: int = 10) -> None:
    """Print which individual constraints are infeasible given the variable bounds.

    Uses bound propagation: for each constraint row a, the achievable range of
    a @ x given variable bounds [lb, ub] is:
        min = sum(a[j]*lb[j] for a[j]>0) + sum(a[j]*ub[j] for a[j]<0)
        max = sum(a[j]*ub[j] for a[j]>0) + sum(a[j]*lb[j] for a[j]<0)

    A constraint is individually infeasible if the rhs falls outside [min, max]
    (equality) or if min > rhs (inequality a@x <= rhs).
    """
    lb_arr = np.array([b[0] for b in bounds], dtype=np.float64)
    ub_arr = np.array([b[1] for b in bounds], dtype=np.float64)

    # Report variables whose own bounds are contradictory.
    bad = np.where(lb_arr > ub_arr + 1e-9)[0]
    if len(bad):
        print(f"    Bad variable bounds (lb>ub): {bad[:max_report].tolist()}"
              + (" ..." if len(bad) > max_report else ""))

    def _check(mat, rhs_vec, kind: str):
        if mat is None:
            return
        from scipy import sparse
        A = mat.tocsr()
        n_flagged = 0
        for i in range(A.shape[0]):
            row = A.getrow(i).toarray().ravel()
            pos = row > 0
            neg = row < 0
            row_min = float(row[pos] @ lb_arr[pos]) + float(row[neg] @ ub_arr[neg])
            row_max = float(row[pos] @ ub_arr[pos]) + float(row[neg] @ lb_arr[neg])
            rhs = float(rhs_vec[i])
            infeasible = False
            if kind == "eq":
                infeasible = rhs < row_min - 1e-9 or rhs > row_max + 1e-9
            else:  # ub: a@x <= rhs
                infeasible = row_min > rhs + 1e-9
            if infeasible:
                nnz = int((row != 0).sum())
                print(f"    Infeasible {kind} constraint {i}: "
                      f"achievable=[{row_min:.6g}, {row_max:.6g}]  rhs={rhs:.6g}  nnz={nnz}")
                n_flagged += 1
                if n_flagged >= max_report:
                    remaining = A.shape[0] - i - 1
                    if remaining > 0:
                        print(f"    ... ({remaining} more {kind} constraints not shown)")
                    break
        if n_flagged == 0:
            print(f"    No individually infeasible {kind} constraints found "
                  f"(infeasibility may be joint).")

    _check(A_eq_mat, b_eq_vec, "eq")
    _check(A_ub_mat, b_ub_vec, "ub")


def _solve_bound_prepared(obj: np.ndarray, lp_matrices,
                          bounds: list[tuple[float, float]],
                          maximize: bool = False) -> float:
    """Solve a single LP bound using the prepared (possibly cached) matrices.

    Dispatches to HiGHS or scipy depending on whether lp_matrices is a
    _HighsPreparedLP or a plain (A_ub, b_ub, A_eq, b_eq) tuple.
    """
    from scipy.optimize import linprog

    if isinstance(lp_matrices, _HighsPreparedLP):
        return lp_matrices.solve(obj, bounds, maximize=maximize)

    A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec = lp_matrices
    c = -obj if maximize else obj

    # HELP EASE BOUNDS: nudge RHS slightly positive so numerical borderline
    # feasible points don't get rejected.  This matches line 723-724 in the
    # original snn_comparison.py.  b_ub_vec is guaranteed non-None here
    # because tight_bounds=False always produces triangle-relaxation ub
    # constraints for the hidden neurons (assumes >= 1 hidden layer).
    b_ub_vec = b_ub_vec + 1e-5

    res = linprog(c,
                A_ub=A_ub_mat,
                b_ub=b_ub_vec,
                A_eq=A_eq_mat,
                b_eq=b_eq_vec,
                bounds=bounds,
                method="highs-ipm")

    if not res.success:
        n_ub = A_ub_mat.shape[0] if A_ub_mat is not None else 0
        n_eq = A_eq_mat.shape[0] if A_eq_mat is not None else 0
        n_v = len(bounds)
        ctx = _LP_CONTEXT
        print(f"  [LP-fail] status={res.status} msg={res.message!r} "
              f"maximize={maximize} n_vars={n_v} n_ub={n_ub} n_eq={n_eq}")
        print(f"    image_idx={ctx.get('image_idx', '?')}  "
              f"label={ctx.get('label', '?')}  "
              f"epsilon={ctx.get('epsilon', '?')}")
        pidx = ctx.get("pixel_indices", [])
        pval = ctx.get("pixel_values", [])
        print(f"    perturbed pixels (in selection order): {pidx}")
        print(f"    pixel values at those positions:       "
              f"{[round(v, 4) for v in pval]}")
        fixed = ctx.get("fixed_latencies") or {}
        if fixed:
            print(f"    fixed latencies (split branch):        {fixed}")
        _diagnose_infeasible(A_ub_mat, b_ub_vec, A_eq_mat, b_eq_vec, bounds)
        return float("inf") if maximize else float("-inf")
    return float(-res.fun if maximize else res.fun)


# ---------------------------------------------------------------------------
# Process-pool worker (used by build_symbolic_relaxation_lp_split)
# ---------------------------------------------------------------------------

# Module-level dict used to share state with worker processes/threads.
# Populated by _init_symbolic_split_worker before any _solve_symbolic_split_worker
# calls; process-safe because each worker process has its own copy.
_SPLIT_WORKER = {}


def _init_symbolic_split_worker(config, state_dict, image_flat, epsilon, k,
                                num_steps, keys, indices, base_lat,
                                tight_bounds, label, cert_only,
                                singleton_bounds=False):
    """Initializer for each worker process in the ProcessPoolExecutor.

    Builds and stores the model (from config + state_dict) in the module-level
    _SPLIT_WORKER dict so _solve_symbolic_split_worker can access it without
    re-loading the model for each combo. torch.set_num_threads(1) prevents
    inter-process thread contention.
    """
    torch.set_num_threads(1)
    model = F2FMLP(**config)
    model.load_state_dict(state_dict)
    model.eval()
    _SPLIT_WORKER.clear()
    _SPLIT_WORKER.update({
        "model": model,
        "image_flat": image_flat,
        "epsilon": epsilon,
        "k": k,
        "num_steps": num_steps,
        "keys": keys,
        "indices": set(int(i) for i in indices),
        "pixel_indices": np.asarray(indices, dtype=int),
        "base_lat": base_lat,
        "tight_bounds": tight_bounds,
        "label": label,
        "cert_only": cert_only,
        "singleton_bounds": singleton_bounds,
    })


def _solve_symbolic_split_worker(combo):
    """Worker function: solve one LP sub-problem for a fixed latency combination.

    combo is a tuple of latency assignments for the split pixels (those whose
    latency is fixed in this branch). The remaining perturbed pixels are still
    treated symbolically inside build_symbolic_relaxation_lp.

    If all perturbed pixels are fixed (depth == k), skip the LP and simulate
    the model exactly for this specific spike-timing assignment.
    """
    cfg = _SPLIT_WORKER
    keys = cfg["keys"]
    # Map each split pixel to its fixed latency for this branch.
    fixed = {keys[j]: combo[j] for j in range(len(keys))} if keys else {}
    label = cfg["label"]

    if len(fixed) == len(cfg["indices"]):
        # All k perturbed pixels are pinned — no LP needed, just simulate exactly.
        lats = cfg["base_lat"].copy()
        for pix, lat in fixed.items():
            lats[pix] = lat
        st = spike_train_from_latencies(lats, cfg["num_steps"])
        score, _, _ = cfg["model"].simulate_with_patterns(st)
        score_arr = np.array(score, dtype=np.float64)
        gap_joint = None
        if label is not None:
            # Compute exact margins for this spike pattern.
            gap_joint = np.full(len(score_arr), np.inf, dtype=np.float64)
            for c in range(len(score_arr)):
                if c != label:
                    gap_joint[c] = float(score_arr[label] - score_arr[c])
        return {
            "lb": score_arr, "ub": score_arr, "gap_joint": gap_joint,
            "n_lp_variables": 0, "n_lp_constraints": 0, "invalid_subproblem": False,
        }
    # Some pixels are still free — solve a partial LP with the fixed latencies.
    return build_symbolic_relaxation_lp(
        cfg["model"], cfg["image_flat"], cfg["epsilon"], cfg["k"], cfg["num_steps"],
        tight_bounds=cfg["tight_bounds"], fixed_latencies=fixed, label=label,
        cert_only=cfg["cert_only"], pixel_indices=cfg["pixel_indices"],
        singleton_bounds=cfg.get("singleton_bounds", False),
    )


# ---------------------------------------------------------------------------
# Core LP verification functions
# ---------------------------------------------------------------------------

def build_symbolic_relaxation_lp(model, image_flat: np.ndarray, epsilon: float,
                                 k: int, num_steps: int,
                                 tight_bounds: bool = False,
                                 fixed_latencies: dict[int, int] | None = None,
                                 label: int | None = None,
                                 cert_only: bool = False,
                                 pixel_indices: np.ndarray | None = None,
                                 singleton_bounds: bool = False,
                                 parallel_workers: int = 1,
                                 parallel_backend: str = "thread",
                                 input_bounds: tuple[np.ndarray, np.ndarray] | None = None):
    """Build and solve the triangle LP relaxation for the multi-layer F2F MLP.

    This is the central verification function. It constructs a linear program
    whose feasible set is a convex outer approximation of all spike-timing
    patterns consistent with the perturbation interval [lb_x, ub_x].

    === input_bounds ===

    When input_bounds=(lb_arr, ub_arr) is provided, the per-dimension bounds are
    used directly instead of calling make_bounds(image_flat, indices, epsilon).
    This is the bridge from n2v's Star/Box sets to the LP engine.
    image_flat is still used to compute base_lat (nominal latencies for the
    unperturbed input midpoint). epsilon is ignored when input_bounds is provided.

    === Variables ===

    1. Input spike variables  p[(pix, t)]  in [0, 1]
       For each perturbed pixel 'pix' and each feasible latency t.
    2. Hidden spike variables  h_var(layer, i, t)  in [0, 1]
    3. Output spike variables  o_var(c, t)  in [0, 1]

    === Constraints ===

    At-most-once (AMO): Σ_t h_var(l, i, t) <= 1 for every hidden/output neuron.
    Input pixel sum: Σ_t p[(pix, t)] = 1 if silence is not feasible, else <= 1.
    First-fire relaxation: triangle constraints for threshold crossing.

    === Objective ===

    For each competitor class c' != label, solve:
        minimize  score[label] - score[c']
    where score[c] = Σ_t (T - t) * o_var(c, t).

    pixel_indices must be provided by the caller; k is kept for legacy compatibility.
    """

    # Extract weight matrices and biases from all layers.
    Ws = [fc.weight.detach().cpu().numpy() for fc in model.fcs]   # list of (out, in) arrays
    bs = [fc.bias.detach().cpu().numpy() for fc in model.fcs]     # list of (out,) arrays
    n_hidden_layers = len(model.hidden_sizes)
    hidden_sizes_list = model.hidden_sizes
    num_classes = Ws[-1].shape[0]

    # Retrieve beta (leak) and threshold from the first LIF neuron.
    beta = float(model.lifs[0].beta.item() if hasattr(model.lifs[0].beta, "item")
                 else model.lifs[0].beta)
    theta = float(model.lifs[0].threshold.item() if hasattr(model.lifs[0].threshold, "item")
                  else model.lifs[0].threshold)

    # Selected pixels whose latency is uncertain (the perturbation set).
    indices = np.asarray(pixel_indices, dtype=int) if pixel_indices is not None else np.empty(0, dtype=int)

    # Per-dimension bounds: use input_bounds if provided, otherwise derive from image+epsilon.
    if input_bounds is not None:
        lb_x, ub_x = input_bounds
    else:
        lb_x, ub_x = make_bounds(image_flat, indices, epsilon)   # per-pixel bounds

    # Nominal latency for each pixel under the clean (unperturbed) image.
    base_lat = latency_from_values(torch.from_numpy(image_flat).float(), num_steps).numpy()

    # -------------------------------------------------------------------------
    # Variable allocation
    # -------------------------------------------------------------------------
    # p_vars[(pix, t)] = index of the LP variable for pixel pix firing at time t.
    p_vars: dict[tuple[int, int], int] = {}

    # fixed_input[t, pix] = 1 if pixel pix definitely fires at time t (no LP variable needed).
    # Non-selected pixels with a finite latency are placed here directly.
    fixed_input = np.zeros((num_steps, image_flat.shape[0]), dtype=np.float64)

    # LP constraint accumulators.
    A_ub: list[np.ndarray] = []   # inequality constraint rows (A_ub @ x <= b_ub)
    b_ub: list[float] = []
    A_eq: list[np.ndarray] = []   # equality constraint rows (A_eq @ x = b_eq)
    b_eq: list[float] = []
    bounds: list[tuple[float, float]] = []  # per-variable [lb, ub]

    selected = set(int(i) for i in indices)

    # Assign fixed spike times for unperturbed pixels: they fire at their nominal latency.
    for pix, latency in enumerate(base_lat):
        if pix not in selected and latency < num_steps:
            # Non-selected pixel with a finite latency: fixed spike at time 'latency'.
            fixed_input[int(latency), pix] = 1.0

    if fixed_latencies is None:
        fixed_latencies = {}  # no latencies are pinned for this sub-problem

    multi_choices = False
    # Allocate LP variables for perturbed pixels.
    for pix in indices:
        if int(pix) in fixed_latencies:
            # This pixel's latency is pinned in this branch of the split tree.
            t_fix = fixed_latencies[int(pix)]
            if t_fix < num_steps:
                fixed_input[t_fix, pix] = 1.0  # treat as a fixed spike
            continue

        # Compute which timesteps are feasible given the pixel's bounds.
        choices = feasible_latencies(float(lb_x[pix]), float(ub_x[pix]), num_steps)
        if len(choices) > 1:
            multi_choices = True
        finite_choices = [t for t in choices if t < num_steps]  # exclude the silent sentinel
        silent_feasible = num_steps in choices                   # can this pixel be silent?

        if len(finite_choices) == 1 and not silent_feasible:
            # Only one latency is feasible and silence is not possible: fix it.
            fixed_input[finite_choices[0], pix] = 1.0
            continue

        # Create one LP variable per feasible firing time.
        pix_var_ids = []
        for t in finite_choices:
            p_vars[(int(pix), int(t))] = len(bounds)
            pix_var_ids.append(len(bounds))
            bounds.append((0.0, 1.0))   # spike probability in [0, 1]

        # Add sum constraint for this pixel's spike variables.
        if pix_var_ids:
            row = np.zeros(len(bounds), dtype=np.float64)
            row[pix_var_ids] = 1.0
            if silent_feasible:
                # Sum <= 1: allows the pixel to be silent (no spike) as well.
                A_ub.append(row.copy())
                b_ub.append(1.0)
            else:
                # Sum = 1: pixel must fire at exactly one of its feasible times.
                A_eq.append(row.copy())
                b_eq.append(1.0)

    n_input_vars = len(bounds)   # number of input spike LP variables allocated so far

    # -------------------------------------------------------------------------
    # Allocate hidden and output spike variables.
    # Each (layer l, neuron i) gets one variable per timestep: h_var(l, i, t).
    # -------------------------------------------------------------------------
    hidden_offsets: list[int] = []   # base index for each hidden layer's variables
    cur_offset = n_input_vars
    for hs in hidden_sizes_list:
        hidden_offsets.append(cur_offset)
        cur_offset += hs * num_steps   # hs neurons × T timesteps
    output_offset = cur_offset         # base index for output spike variables
    n_vars = output_offset + num_classes * num_steps

    # All hidden and output spike variables are bounded to [0, 1].
    bounds.extend([(0.0, 1.0)] * (sum(hidden_sizes_list) * num_steps + num_classes * num_steps))

    def pad_rows(rows: list[np.ndarray]) -> list[np.ndarray]:
        """Extend constraint rows to the current total variable count n_vars."""
        padded = []
        for row in rows:
            if row.shape[0] == n_vars:
                padded.append(row)
            else:
                full = np.zeros(n_vars, dtype=np.float64)
                full[:row.shape[0]] = row
                padded.append(full)
        return padded

    # Existing rows were built before hidden/output variables were added; extend them.
    A_ub = pad_rows(A_ub)
    A_eq = pad_rows(A_eq)

    # Helper accessors for variable indices.
    def h_var(layer: int, i: int, t: int) -> int:
        """Index of the spike variable for hidden neuron i in 'layer' at time t."""
        return hidden_offsets[layer] + i * num_steps + t

    def o_var(c: int, t: int) -> int:
        """Index of the spike variable for output class c at time t."""
        return output_offset + c * num_steps + t

    def add_amo(var_fn, size: int):
        """Add at-most-once (AMO) constraints: Σ_t var_fn(i, t) <= 1 for each i."""
        for i in range(size):
            row = np.zeros(n_vars, dtype=np.float64)
            for t in range(num_steps):
                row[var_fn(i, t)] = 1.0
            A_ub.append(row)
            b_ub.append(1.0)

    # AMO constraints for all hidden layers and the output layer (optional).
    if _USE_AMO_CONSTRAINTS:
        for layer in range(n_hidden_layers):
            add_amo(lambda i, t, layer=layer: h_var(layer, i, t), hidden_sizes_list[layer])
        add_amo(o_var, num_classes)

    # -------------------------------------------------------------------------
    # First-fire relaxation (triangle constraints)
    # -------------------------------------------------------------------------
    def add_first_fire_relaxation(var_idx: int, u_center: float, u_row: np.ndarray,
                                  previous_vars: list[int],
                                  u_lb_amo: float | None = None,
                                  u_ub_amo: float | None = None):
        """Add triangle constraints relating a spike variable to its membrane potential.

        The membrane potential at timestep t for neuron i is:
            u(t) = u_center + u_row @ lp_vars
        where u_center is the contribution from fixed (non-LP) inputs, and
        u_row carries the contributions from LP variables.

        The relaxation encodes the following:
          - If u_ub < θ (membrane can never reach threshold): spk[i, t] = 0.
          - Otherwise, two linear inequalities are added:
              Upper bound: u >= θ implies spk <= 1 (triangle upper edge)
              Lower bound: if u >= θ AND no earlier spike, spk >= linear function of u
            These form a triangle in the (u, spk) plane.

        previous_vars: indices of the same neuron's spike variables at t' < t.
        The lower-bound constraint subtracts previous spikes because a neuron
        that already fired at t' < t cannot fire again (at-most-once).
        """
        if tight_bounds:
            # Solve two LPs to get tight bounds on u (expensive, used rarely).
            u_lb = u_center + _solve_bound(u_row, A_ub, b_ub, A_eq, b_eq, bounds)
            u_ub = u_center + _solve_bound(u_row, A_ub, b_ub, A_eq, b_eq, bounds, maximize=True)
        elif u_lb_amo is not None:
            # Use the pre-computed at-most-once interval bounds.
            u_lb, u_ub = u_lb_amo, u_ub_amo
        else:
            # Cheap interval arithmetic over the variable bounds.
            u_lb, u_ub = _linear_interval(u_center, u_row, bounds)

        if u_ub < theta:
            # Membrane potential can never reach threshold: neuron cannot fire at t.
            # Force spk[i, t] = 0 with the constraint spk[i, t] <= 0.
            row = np.zeros(n_vars, dtype=np.float64)
            row[var_idx] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)
            return

        # Upper-triangle constraint: spk[i,t] <= (u - u_lb) / (θ - u_lb)
        # Rearranged: -u_row @ x + (θ - u_lb) * spk <= u_center - u_lb
        upper_denom = theta - u_lb
        if upper_denom > 1e-10:
            row = -u_row.copy()
            row[var_idx] += upper_denom
            A_ub.append(row)
            b_ub.append(u_center - u_lb)

        # Lower-triangle constraint: spk[i,t] >= (u - θ) / (u_ub - θ) - Σ_{t'<t} spk[i,t']
        # Rearranged: u_row @ x - (u_ub - θ) * spk - (u_ub - θ) * Σprev <= θ - u_center
        lower_denom = u_ub - theta
        if lower_denom > 1e-10:
            row = u_row.copy()
            row[var_idx] -= lower_denom
            for prev in previous_vars:
                row[prev] -= lower_denom   # subtract earlier spike contributions
            A_ub.append(row)
            b_ub.append(theta - u_center)

    # -------------------------------------------------------------------------
    # Build first-layer membrane potential expressions.
    # -------------------------------------------------------------------------
    W0 = Ws[0]     # (hidden_size_0, input_size)
    b0 = bs[0]     # (hidden_size_0,)
    hs0 = hidden_sizes_list[0]

    # layer0_current_c[i, t] = bias contribution + fixed-input contribution
    # to neuron i's membrane at time t (from non-LP inputs).
    layer0_current_c = W0 @ fixed_input.T + b0[:, None]   # shape (hs0, T)

    # layer0_current_rows[i, t, :] = LP-variable coefficients for neuron i's
    # membrane potential at time t (from perturbed-pixel LP variables).
    layer0_current_rows = np.zeros((hs0, num_steps, n_vars), dtype=np.float64)
    for (pix, t), var_idx in p_vars.items():
        # W0[:, pix] is the weight column connecting pixel pix to all hidden neurons.
        layer0_current_rows[:, t, var_idx] = W0[:, pix]

    # Pre-index p_vars by pixel for fast lookup in the AMO-bounds loop.
    pix_to_choices: dict[int, list[tuple[int, int]]] = {}
    for (pix, tau), var_idx in p_vars.items():
        pix_to_choices.setdefault(int(pix), []).append((int(tau), var_idx))

    # -------------------------------------------------------------------------
    # Layer-by-layer triangle relaxation construction.
    # -------------------------------------------------------------------------
    all_t_min: list[np.ndarray] = []   # t_min[layer][i] = earliest possible fire time
    all_t_max: list[np.ndarray] = []   # t_max[layer][i] = latest possible fire time
    all_t_def: list[np.ndarray] = []   # t_def[layer][i] = timestep at which firing is guaranteed

    for layer in range(n_hidden_layers):
        hs_L = hidden_sizes_list[layer]
        W_L = Ws[layer]
        b_L = bs[layer]

        # Initialize firing-time range arrays for this layer.
        t_min_L = np.full(hs_L, num_steps, dtype=int)   # T means "never fires"
        t_max_L = np.full(hs_L, -1, dtype=int)          # -1 means "no upper bound found"
        t_def_L = np.full(hs_L, num_steps, dtype=int)   # T means "not guaranteed to fire"

        # Loose AMO bounds on membrane potential (used when tight_bounds=False).
        ub_NR_L = np.full((hs_L, num_steps), -np.inf)
        lb_NR_L = np.full((hs_L, num_steps), np.inf)

        if layer == 0:
            t_min_prev = t_max_prev = t_def_prev = None
        else:
            t_min_prev = all_t_min[layer - 1]
            t_max_prev = all_t_max[layer - 1]
            t_def_prev = all_t_def[layer - 1]
            hs_prev = hidden_sizes_list[layer - 1]

        for i in range(hs_L):
            for t in range(num_steps):
                u_center = 0.0
                u_row = np.zeros(n_vars, dtype=np.float64)

                if layer == 0:
                    # First hidden layer: input spikes come from the pixel LP variables.
                    for tau in range(t + 1):
                        scale = beta ** (t - tau)   # exponential decay from timestep tau to t
                        u_center += scale * layer0_current_c[i, tau]
                        u_row += scale * layer0_current_rows[i, tau]

                    # Compute AMO-based interval bounds on u (without solving an LP).
                    u_lb_amo = u_center
                    u_ub_amo = u_center
                    for pix, choices_pix in pix_to_choices.items():
                        at_or_before = [(tau, vid) for (tau, vid) in choices_pix if tau <= t]
                        if not at_or_before:
                            continue
                        max_scale = max(beta ** (t - tau) for (tau, _) in at_or_before)
                        w = W_L[i, pix]
                        if w >= 0.0:
                            u_ub_amo += w * max_scale
                        else:
                            u_lb_amo += w * max_scale
                else:
                    # Deeper hidden layer: input spikes come from the previous layer's
                    # hidden spike LP variables h_var(layer-1, kk, tau).
                    for tau in range(t + 1):
                        scale = beta ** (t - tau)
                        u_center += scale * b_L[i]   # bias accumulated over tau steps
                    for tau in range(t + 1):
                        scale = beta ** (t - tau)
                        for kk in range(hs_prev):
                            u_row[h_var(layer - 1, kk, tau)] += scale * W_L[i, kk]

                    # AMO bounds for deeper layers.
                    u_lb_amo = u_center
                    u_ub_amo = u_center
                    for kk in range(hs_prev):
                        if t_min_prev[kk] > t or t_max_prev[kk] < 0:
                            continue
                        if (singleton_bounds
                                and t_def_prev[kk] <= t
                                and t_min_prev[kk] == t_max_prev[kk] == t_def_prev[kk]):
                            t_fire = t_def_prev[kk]
                            exact_scale = beta ** (t - t_fire)
                            u_lb_amo += W_L[i, kk] * exact_scale
                            u_ub_amo += W_L[i, kk] * exact_scale
                            continue
                        latest = min(t, t_max_prev[kk])
                        max_scale = beta ** (t - latest)
                        w = W_L[i, kk]
                        if w >= 0.0:
                            u_ub_amo += w * max_scale
                        else:
                            u_lb_amo += w * max_scale

                # Store the AMO bounds for this neuron-timestep.
                ub_NR_L[i, t] = u_ub_amo
                lb_NR_L[i, t] = u_lb_amo

                # Add the triangle relaxation constraints for h_var(layer, i, t).
                amo_sound = (layer == 0) or _USE_AMO_CONSTRAINTS
                add_first_fire_relaxation(
                    h_var(layer, i, t), u_center, u_row,
                    [h_var(layer, i, tau) for tau in range(t)],
                    u_lb_amo=u_lb_amo if amo_sound else None,
                    u_ub_amo=u_ub_amo if amo_sound else None,
                )

        # -------------------------------------------------------------------------
        # Compute firing-time bounds for this layer.
        # -------------------------------------------------------------------------
        for i in range(hs_L):
            for t in range(num_steps):
                if ub_NR_L[i, t] >= theta:
                    t_min_L[i] = t
                    break
            for t in range(num_steps):
                if lb_NR_L[i, t] >= theta:
                    t_def_L[i] = t
                    break
            t_def = t_def_L[i]
            if t_def < num_steps:
                t_max_L[i] = t_def
            elif t_min_L[i] < num_steps:
                for t in range(num_steps - 1, -1, -1):
                    if ub_NR_L[i, t] >= theta:
                        t_max_L[i] = t
                        break

        # -------------------------------------------------------------------------
        # Enforce the firing window on the LP variables.
        # -------------------------------------------------------------------------
        for i in range(hs_L):
            if t_max_L[i] < 0:
                continue
            for tau in range(t_max_L[i] + 1, num_steps):
                bounds[h_var(layer, i, tau)] = (0.0, 0.0)

        # Optional tighter bounds for neurons that fire at exactly one timestep.
        if singleton_bounds:
            for i in range(hs_L):
                if (t_def_L[i] < num_steps
                        and t_min_L[i] == t_max_L[i] == t_def_L[i]):
                    t_fire = t_def_L[i]
                    row = np.zeros(n_vars, dtype=np.float64)
                    row[h_var(layer, i, t_fire)] = 1.0
                    A_eq.append(row)
                    b_eq.append(1.0)

        all_t_min.append(t_min_L)
        all_t_max.append(t_max_L)
        all_t_def.append(t_def_L)

    # -------------------------------------------------------------------------
    # Output layer triangle relaxation.
    # -------------------------------------------------------------------------
    t_min_last = all_t_min[-1]
    t_max_last = all_t_max[-1]
    t_def_last = all_t_def[-1]
    hs_last = hidden_sizes_list[-1]
    W_out = Ws[-1]     # (num_classes, hs_last)
    b_out = bs[-1]     # (num_classes,)
    last_layer_idx = n_hidden_layers - 1

    for c in range(num_classes):
        for t in range(num_steps):
            u_center = 0.0
            u_row = np.zeros(n_vars, dtype=np.float64)
            for tau in range(t + 1):
                scale = beta ** (t - tau)
                u_center += scale * b_out[c]
                for i in range(hs_last):
                    u_row[h_var(last_layer_idx, i, tau)] += scale * W_out[c, i]

            # AMO bounds for the output layer.
            u_lb_amo = u_center
            u_ub_amo = u_center
            for i in range(hs_last):
                if t_min_last[i] > t or t_max_last[i] < 0:
                    continue
                if (singleton_bounds
                        and t_def_last[i] <= t
                        and t_min_last[i] == t_max_last[i] == t_def_last[i]):
                    t_fire = t_def_last[i]
                    exact_scale = beta ** (t - t_fire)
                    u_lb_amo += W_out[c, i] * exact_scale
                    u_ub_amo += W_out[c, i] * exact_scale
                    continue
                latest = min(t, t_max_last[i])
                max_scale = beta ** (t - latest)
                w = W_out[c, i]
                if w >= 0.0:
                    u_ub_amo += w * max_scale
                else:
                    u_lb_amo += w * max_scale

            add_first_fire_relaxation(
                o_var(c, t), u_center, u_row, [o_var(c, tau) for tau in range(t)],
                u_lb_amo=u_lb_amo if _USE_AMO_CONSTRAINTS else None,
                u_ub_amo=u_ub_amo if _USE_AMO_CONSTRAINTS else None,
            )

    # -------------------------------------------------------------------------
    # Objective rows: score[c] = Σ_t (T - t) * o_var(c, t)
    # -------------------------------------------------------------------------
    score_rows = np.zeros((num_classes, n_vars), dtype=np.float64)
    for c in range(num_classes):
        for t in range(num_steps):
            score_rows[c, o_var(c, t)] = float(num_steps - t)

    # Build the final sparse LP matrices from the accumulated constraint lists.
    lp_matrices = _prepare_lp_matrices(A_ub, b_ub, A_eq, b_eq)

    # Populate the failure-diagnostic context.
    _LP_CONTEXT.update({
        "pixel_indices": [int(i) for i in indices],
        "pixel_values": image_flat[indices].tolist() if len(indices) > 0 else [],
        "label": label,
        "epsilon": epsilon,
        "fixed_latencies": fixed_latencies if fixed_latencies else {},
    })

    # -------------------------------------------------------------------------
    # Optional self-check: verify the NOMINAL (unperturbed) trajectory is LP-feasible.
    # -------------------------------------------------------------------------
    if os.environ.get("SNNV_CHECK_NOMINAL"):
        inv_pvars = {vid: (pix, t) for (pix, t), vid in p_vars.items()}

        def _decode_var(j: int) -> str:
            if j in inv_pvars:
                pix, t = inv_pvars[j]
                return f"in(pix={pix},t={t})"
            if j >= output_offset:
                rel = j - output_offset
                return f"o(c={rel // num_steps},t={rel % num_steps})"
            for L in range(n_hidden_layers):
                base = hidden_offsets[L]
                size = hidden_sizes_list[L] * num_steps
                if base <= j < base + size:
                    rel = j - base
                    ii, tt = rel // num_steps, rel % num_steps
                    return (f"h(L={L},i={ii},t={tt}) "
                            f"[t_min={all_t_min[L][ii]},t_max={all_t_max[L][ii]},"
                            f"t_def={all_t_def[L][ii]}]")
            return f"var#{j}"

        nom_lat = base_lat.astype(int).copy()
        for pix, tt in (fixed_latencies or {}).items():
            nom_lat[int(pix)] = int(tt)
        xc = np.zeros(n_vars, dtype=np.float64)
        for (pix, t), vid in p_vars.items():
            if int(nom_lat[pix]) == t:
                xc[vid] = 1.0
        st = spike_train_from_latencies(nom_lat, num_steps)
        _, hid, outp = model.simulate_with_patterns(st)
        for L in range(n_hidden_layers):
            col0 = int(sum(hidden_sizes_list[:L]))
            for ii in range(hidden_sizes_list[L]):
                for tt in range(num_steps):
                    xc[h_var(L, ii, tt)] = float(hid[tt, col0 + ii])
        for c in range(num_classes):
            for tt in range(num_steps):
                xc[o_var(c, tt)] = float(outp[tt, c])

        tol = 1e-6
        ctx = _LP_CONTEXT
        hdr = (f"image_idx={ctx.get('image_idx', '?')} label={label} "
               f"eps={epsilon} k={len(indices)} pix={[int(i) for i in indices]}")
        viols: list[str] = []
        for j, (lo, hi) in enumerate(bounds):
            if xc[j] < lo - tol or xc[j] > hi + tol:
                viols.append(f"    BOUND  {_decode_var(j)}  value={xc[j]:.4g} "
                             f"not in [{lo:.4g},{hi:.4g}]")
        A_ub_arr = np.asarray(A_ub) if A_ub else np.zeros((0, n_vars))
        A_eq_arr = np.asarray(A_eq) if A_eq else np.zeros((0, n_vars))
        if len(A_ub_arr):
            ub_val = A_ub_arr @ xc
            for r in np.where(ub_val > np.asarray(b_ub) + tol)[0]:
                nz = np.nonzero(A_ub_arr[r])[0]
                terms = ", ".join(f"{A_ub_arr[r, j]:+.3g}*{_decode_var(j)}" for j in nz[:6])
                viols.append(f"    UB#{r}  lhs={ub_val[r]:.4g} > rhs={b_ub[r]:.4g}  "
                             f"[{terms}{' ...' if len(nz) > 6 else ''}]")
        if len(A_eq_arr):
            eq_val = A_eq_arr @ xc
            for r in np.where(np.abs(eq_val - np.asarray(b_eq)) > tol)[0]:
                nz = np.nonzero(A_eq_arr[r])[0]
                terms = ", ".join(f"{A_eq_arr[r, j]:+.3g}*{_decode_var(j)}" for j in nz[:6])
                viols.append(f"    EQ#{r}  lhs={eq_val[r]:.4g} != rhs={b_eq[r]:.4g}  "
                             f"[{terms}{' ...' if len(nz) > 6 else ''}]")
        if viols:
            print(f"[NOMINAL-INFEASIBLE] {hdr}  ({len(viols)} violations)")
            for line in viols[:20]:
                print(line)
            if len(viols) > 20:
                print(f"    ... ({len(viols) - 20} more)")

    if _DEBUG_LP:
        mats = lp_matrices if not isinstance(lp_matrices, _HighsPreparedLP) \
               else lp_matrices.lp_matrices
        A_ub_m, _, A_eq_m, _ = mats
        n_ub = A_ub_m.shape[0] if A_ub_m is not None else 0
        n_eq = A_eq_m.shape[0] if A_eq_m is not None else 0
        ctx = _LP_CONTEXT
        pidx = ctx.get("pixel_indices", [])
        pval = ctx.get("pixel_values", [])
        fixed = ctx.get("fixed_latencies") or {}
        print(f"  [LP-debug] image_idx={ctx.get('image_idx', '?')}  "
              f"label={ctx.get('label', '?')}  epsilon={ctx.get('epsilon', '?')}  "
              f"n_vars={n_vars}  n_ub={n_ub}  n_eq={n_eq}")
        print(f"    perturbed pixels (in selection order): {pidx}")
        print(f"    pixel values at those positions:       "
              f"{[round(v, 4) for v in pval]}")
        if fixed:
            print(f"    fixed latencies (split branch):        {fixed}")

    t0 = time.perf_counter()

    # -------------------------------------------------------------------------
    # Solve for output score bounds (lb_y, ub_y) if requested.
    # -------------------------------------------------------------------------
    lb_y = None
    ub_y = None
    if not (cert_only and label is not None):
        lb_y = np.zeros(num_classes, dtype=np.float64)
        ub_y = np.zeros(num_classes, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Solve for the certification gap: min_c' score[label] - score[c'].
    # -------------------------------------------------------------------------
    gap_joint = None
    if label is not None:
        gap_joint = np.full(num_classes, np.inf)

    # Build the full list of LP tasks.
    tasks = []
    if lb_y is not None:
        for c in range(num_classes):
            tasks.append((lb_y, c, score_rows[c],                          False))
            tasks.append((ub_y, c, score_rows[c],                          True))
    if gap_joint is not None:
        for c_prime in range(num_classes):
            if c_prime != label:
                tasks.append((gap_joint, c_prime, score_rows[label] - score_rows[c_prime], False))

    if parallel_workers > 1 and len(tasks) > 1 and parallel_backend != "process":
        with ThreadPoolExecutor(max_workers=int(parallel_workers)) as executor:
            futures = {
                executor.submit(_solve_bound_prepared, obj, lp_matrices, bounds, maximize): (arr, idx)
                for arr, idx, obj, maximize in tasks
            }
            for future in futures:
                arr, idx = futures[future]
                arr[idx] = future.result()
    else:
        for arr, idx, obj, maximize in tasks:
            arr[idx] = _solve_bound_prepared(obj, lp_matrices, bounds, maximize)

    runtime = time.perf_counter() - t0

    total_hidden_spike_vars = sum(hs * num_steps for hs in hidden_sizes_list)
    return {
        "lb": lb_y, "ub": ub_y, "gap_joint": gap_joint,
        "runtime_s": runtime,
        "n_input_vars": n_input_vars,
        "n_hidden_spike_vars": total_hidden_spike_vars,
        "n_output_spike_vars": num_classes * num_steps,
        "n_lp_variables": n_vars,
        "n_lp_constraints": len(A_ub) + len(A_eq),
        "t_min_h": all_t_min[-1].tolist(),
        "t_max_h": all_t_max[-1].tolist(),
        "invalid_subproblem": False,
    }


def build_symbolic_relaxation_lp_split(
    model, image_flat: np.ndarray, epsilon: float,
    k: int, num_steps: int,
    split_depth: int = 0,
    label: int | None = None,
    tight_bounds: bool = False,
    parallel_workers: int = 1,
    cert_only: bool = False,
    parallel_backend: str = "thread",
    split_strategy: str = "selected",
    pixel_indices: np.ndarray | None = None,
    singleton_bounds: bool = False,
    input_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Symbolic split verification: branch on the first split_depth pixels' latencies.

    This function enumerates all combinations of feasible latency assignments for
    the top split_depth pixels (ordered by split_strategy) and solves one LP per
    branch. The final bounds are the element-wise min/max over all branch results.

    If all k pixels are fixed in a branch (split_depth == k), the model is
    simulated exactly for that latency assignment (no LP needed).

    When input_bounds=(lb_arr, ub_arr) is provided, per-dimension bounds are used
    directly instead of calling make_bounds(image_flat, indices, epsilon).

    Note: input_bounds is not compatible with parallel_backend='process' because
    the process pool worker initializer does not serialize input_bounds.
    """
    if input_bounds is not None and parallel_backend == "process":
        raise ValueError(
            "input_bounds is not supported with parallel_backend='process'. "
            "Use parallel_backend='thread' instead."
        )

    indices = np.asarray(pixel_indices, dtype=int) if pixel_indices is not None else np.empty(0, dtype=int)

    # Per-dimension bounds for split ordering.
    if input_bounds is not None:
        lb_x, ub_x = input_bounds
    else:
        lb_x, ub_x = make_bounds(image_flat, indices, epsilon)

    num_classes = model.fcs[-1].out_features

    # Determine the split ordering for the first split_depth pixels.
    ordered_indices = order_split_indices(model, indices, lb_x, ub_x, num_steps, split_strategy)
    split_indices = ordered_indices[:split_depth]   # pixels whose latency will be branched

    # For each split pixel, enumerate all feasible latency values.
    pix_choices_split = {}
    for pix in split_indices:
        choices = feasible_latencies(float(lb_x[pix]), float(ub_x[pix]), num_steps)
        pix_choices_split[int(pix)] = choices

    # Generate all Cartesian-product combinations of fixed latency assignments.
    keys = sorted(pix_choices_split)
    combos = list(iproduct(*[pix_choices_split[p] for p in keys])) if keys else [()]

    # Accumulators for merging results across all branches.
    lb_merged: np.ndarray | None = None
    ub_merged: np.ndarray | None = None
    gap_joint_merged: np.ndarray | None = None
    if label is not None:
        gap_joint_merged = np.full(num_classes, np.inf, dtype=np.float64)
    n_lp_vars_total = 0
    n_lp_constr_total = 0
    invalid_subproblem = False

    base_lat = latency_from_values(torch.from_numpy(image_flat).float(), num_steps).numpy()
    simulate_lock = threading.Lock()  # protect model.simulate_with_patterns in threaded mode

    def solve_combo(combo):
        """Solve one branch: fix latencies for split pixels, free the rest."""
        fixed = {keys[j]: combo[j] for j in range(len(keys))} if keys else {}
        if len(fixed) == len(indices):
            # All pixels are pinned — exact simulation, no LP.
            lats = base_lat.copy()
            for pix, lat in fixed.items():
                lats[pix] = lat
            st = spike_train_from_latencies(lats, num_steps)
            with simulate_lock:
                score, _, _ = model.simulate_with_patterns(st)
            score_arr = np.array(score, dtype=np.float64)
            gap_joint = None
            if label is not None and gap_joint_merged is not None:
                gap_joint = np.full(num_classes, np.inf, dtype=np.float64)
                for c in range(len(score_arr)):
                    if c != label:
                        gap = float(score_arr[label] - score_arr[c])
                        gap_joint[c] = min(gap_joint[c], gap)
            return {
                "lb": score_arr, "ub": score_arr, "gap_joint": gap_joint,
                "n_lp_variables": 0, "n_lp_constraints": 0, "invalid_subproblem": False,
            }
        # Partially fixed: solve the relaxation LP with the fixed latencies.
        return build_symbolic_relaxation_lp(
            model, image_flat, epsilon, k, num_steps,
            tight_bounds=tight_bounds, fixed_latencies=fixed, label=label,
            cert_only=cert_only, pixel_indices=indices, singleton_bounds=singleton_bounds,
            parallel_workers=parallel_workers, parallel_backend=parallel_backend,
            input_bounds=input_bounds,
        )

    n_combos = len(combos)
    if parallel_workers > 1 and n_combos > 1 and parallel_backend == "process":
        # Process pool: serialize the model and state for the worker initializer.
        config = {
            "input_size": model.fcs[0].in_features,
            "hidden_sizes": [fc.out_features for fc in model.fcs[:-1]],
            "num_classes": model.fcs[-1].out_features,
            "beta": float(model.lifs[0].beta.item() if hasattr(model.lifs[0].beta, "item")
                          else model.lifs[0].beta),
            "threshold": float(model.lifs[0].threshold.item()
                               if hasattr(model.lifs[0].threshold, "item")
                               else model.lifs[0].threshold),
            "num_steps": num_steps,
        }
        state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        mp_context = multiprocessing.get_context("fork" if os.name == "posix" else "spawn")
        with ProcessPoolExecutor(
            max_workers=int(parallel_workers), mp_context=mp_context,
            initializer=_init_symbolic_split_worker,
            initargs=(config, state_dict, image_flat, epsilon, k, num_steps, keys, indices,
                      base_lat, tight_bounds, label, cert_only, singleton_bounds),
        ) as executor:
            combo_results = list(_tqdm(
                executor.map(_solve_symbolic_split_worker, combos),
                total=n_combos, desc="      combos", leave=False,
            ))
    elif parallel_workers > 1 and n_combos > 1:
        # Thread pool: simpler than process pool, avoids serialization cost.
        from concurrent.futures import as_completed as _as_completed
        combo_results = [None] * n_combos
        with ThreadPoolExecutor(max_workers=int(parallel_workers)) as executor:
            futures = {executor.submit(solve_combo, combo): i for i, combo in enumerate(combos)}
            with _tqdm(total=n_combos, desc="      combos", leave=False) as cbar:
                for future in _as_completed(futures):
                    combo_results[futures[future]] = future.result()
                    cbar.update(1)
    else:
        # Sequential: iterate over all branches without parallelism.
        combo_results = []
        for combo in _tqdm(combos, desc="      combos", leave=False):
            combo_results.append(solve_combo(combo))

    # -------------------------------------------------------------------------
    # Merge branch results: element-wise min for lower bounds, max for upper bounds.
    # -------------------------------------------------------------------------
    for r in combo_results:
        if r["lb"] is not None and r["ub"] is not None and (
            np.any(np.isinf(r["lb"])) or np.any(np.isinf(r["ub"]))
        ):
            invalid_subproblem = True
            break
        if r["lb"] is not None and r["ub"] is not None:
            if lb_merged is None:
                lb_merged = r["lb"].copy()
                ub_merged = r["ub"].copy()
            else:
                lb_merged = np.minimum(lb_merged, r["lb"])
                ub_merged = np.maximum(ub_merged, r["ub"])
        if label is not None and gap_joint_merged is not None and r.get("gap_joint") is not None:
            gap_joint_merged = np.minimum(gap_joint_merged, r["gap_joint"])
        n_lp_vars_total += r["n_lp_variables"]
        n_lp_constr_total += r["n_lp_constraints"]

    return {
        "lb": None if invalid_subproblem else lb_merged,
        "ub": None if invalid_subproblem else ub_merged,
        "gap_joint": gap_joint_merged,
        "n_cases": len(combos),
        "invalid_subproblem": invalid_subproblem,
        "n_lp_variables": n_lp_vars_total,
        "n_lp_constraints": n_lp_constr_total,
        "n_input_vars": 0,
        "n_hidden_spike_vars": 0,
        "n_output_spike_vars": 0,
        "runtime_s": 0.0,
        "split_pixel_order": split_indices.tolist(),
    }


def _gap_from_result(result, label):
    """Extract the certification gap from an LP result dict.

    Prefers gap_joint (directly certified margin) when available.
    Falls back to lb_y[label] - max(ub_y[not label]) if gap_joint is missing.

    Returns +inf if certified with large margin, negative if not certified.
    """
    gap_joint = result.get("gap_joint")
    if gap_joint is not None:
        competitors = [c for c in range(len(gap_joint)) if c != label]
        if competitors and np.all(np.isfinite(gap_joint[competitors])):
            return float(np.min(gap_joint[competitors]))
    # Fallback: use per-class score bounds.
    lb_y = result["lb"]
    ub_y = result["ub"]
    if lb_y is None or ub_y is None:
        return float("-inf")
    return float(lb_y[label] - np.max(np.delete(ub_y, label)))


def verify_symbolic_sample(model, image_flat: np.ndarray, label: int, epsilon: float,
                           k: int, num_steps: int,
                           tight_bounds: bool = False,
                           split_depth: int = 0,
                           max_depth_cap: int | None = None,
                           parallel_workers: int = 1,
                           cert_only: bool = False,
                           parallel_backend: str = "thread",
                           split_strategy: str = "selected",
                           pixel_indices: np.ndarray | None = None,
                           track_depth: bool = False,
                           singleton_bounds: bool = False,
                           input_bounds: tuple[np.ndarray, np.ndarray] | None = None):
    """Attempt to certify a sample by iterating over split depths until certified.

    Starts at depth 0 (single LP over all k pixels). If the gap is not positive,
    tries depth 1 (branch on 1 pixel), depth 2, ..., up to max_depth.

    For split_depth > 0, each branch fixes one pixel's latency and solves a
    smaller LP for the remaining pixels. Higher depth = more branches = slower
    but tighter. Stops as soon as gap > 0 (certified).

    When called from SNNVerifier.verify with split_depth=0 and max_depth_cap=0,
    this is the pure depth-0 LP (Stage 1 of the two-stage strategy).

    input_bounds: when provided, bypasses make_bounds(image_flat, indices, epsilon)
    and uses per-dimension bounds directly (from Star.get_ranges() or Box.lb/ub).
    """
    t0 = time.perf_counter()
    max_depth = split_depth if split_depth > 0 else min(
        len(pixel_indices) if pixel_indices is not None else k,
        max_depth_cap if max_depth_cap is not None else (
            len(pixel_indices) if pixel_indices is not None else k
        )
    )
    depth_reached = 0
    for depth in range(max_depth + 1):
        depth_reached = depth
        if depth == 0:
            # Single LP covering all k perturbed pixels simultaneously.
            result = build_symbolic_relaxation_lp(
                model, image_flat, epsilon, k, num_steps, tight_bounds=tight_bounds,
                label=label, cert_only=cert_only, pixel_indices=pixel_indices,
                singleton_bounds=singleton_bounds,
                parallel_workers=parallel_workers, parallel_backend=parallel_backend,
                input_bounds=input_bounds,
            )
        else:
            # Branch on the first 'depth' pixels; solve one LP per branch.
            result = build_symbolic_relaxation_lp_split(
                model, image_flat, epsilon, k, num_steps,
                split_depth=depth, label=label, tight_bounds=tight_bounds,
                parallel_workers=parallel_workers, cert_only=cert_only,
                parallel_backend=parallel_backend, split_strategy=split_strategy,
                pixel_indices=pixel_indices, singleton_bounds=singleton_bounds,
                input_bounds=input_bounds,
            )
        gap = _gap_from_result(result, label)
        if track_depth:
            elapsed = time.perf_counter() - t0
            status = "certified" if gap > 0.0 else "not certified"
            print(f"    depth {depth}/{max_depth}  gap={gap:.3f}  {status}  ({elapsed:.1f}s)")
        if gap > 0.0:
            break   # certified — no need to go deeper
    runtime = time.perf_counter() - t0
    lb_y = result["lb"]
    ub_y = result["ub"]
    if lb_y is None or ub_y is None:
        bound_width = float("nan")
        lb_list = None
        ub_list = None
    else:
        bound_width = float(np.mean(ub_y - lb_y))
        lb_list = lb_y.tolist()
        ub_list = ub_y.tolist()
    return {
        "epsilon": float(epsilon),
        "k": int(k),
        "label": int(label),
        "mode": "symbolic",
        "tight_bounds": bool(tight_bounds),
        "gap": gap,
        "certified": gap > 0.0,
        "bound_width": bound_width,
        "runtime_s": runtime,
        "lp_runtime_s": result["runtime_s"],
        "depth_reached": int(depth_reached),
        "max_depth": int(max_depth),
        "pixel_order": result.get("split_pixel_order", []),
        "n_cases": int(result.get("n_cases", 1)),
        "n_lambda": result["n_lp_variables"],
        "n_patterns": result["n_lp_constraints"],
        "n_input_vars": result["n_input_vars"],
        "n_hidden_spike_vars": result["n_hidden_spike_vars"],
        "n_output_spike_vars": result["n_output_spike_vars"],
        "n_lp_variables": result["n_lp_variables"],
        "n_lp_constraints": result["n_lp_constraints"],
        "lb": lb_list,
        "ub": ub_list,
    }

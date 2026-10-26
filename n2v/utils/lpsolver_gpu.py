"""NS-certified batched LP bounds for the refine engine (``n2v.refine``).

Bounds are solved by projected-Adam descent on the Neumaier-Shcherbina dual
(best iterate kept) in float32 and certified in float64 by
:mod:`n2v.utils.ns_certificate`. The single value ever returned is the
certified bound, sound at any solver accuracy. Also a PDHG primal heuristic
(``solve_lp_primal_population_gpu``) for witness candidates, never used for a
verification decision. torch is imported lazily; computations run on CUDA when
a device is usable, else on CPU torch.
"""

from typing import List, Optional, Sequence

import numpy as np

from n2v.profiling import count, region, set_meta
from n2v.utils.ns_certificate import ns_bound, ns_bounds_population

# Lazy torch handle: imported on first GPU use so importing this module (and the
# whole CPU LP stack) never requires torch.
_torch = None


def _torch_mod():
    global _torch
    if _torch is None:
        import torch  # noqa: PLC0415  (lazy by design)
        _torch = torch
    return _torch


def gpu_available() -> bool:
    """True iff a CUDA device is usable (the GPU LP paths otherwise run on CPU
    torch)."""
    try:
        return _torch_mod().cuda.is_available()
    except Exception:
        return False


def _resolve_device(device: Optional[str]) -> str:
    """``None`` -> "cuda" when a CUDA device is usable, else "cpu"."""
    if device is not None:
        return device
    return "cuda" if gpu_available() else "cpu"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def solve_lp_batch_gpu(
    objectives: Sequence[np.ndarray],
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    minimize_flags: Optional[Sequence[bool]] = None,
    max_iters: int = 300,
    warm_duals: Optional[Sequence[Optional[np.ndarray]]] = None,
    return_duals: bool = False,
    device: Optional[str] = None,
):
    """Solve a batch of LPs sharing ``A,b,lb,ub`` and return *certified* bounds.

    Same contract as :func:`n2v.utils.lpsolver.solve_lp_batch`: one bound per
    objective (lower bound on the min when ``minimize_flags[i]`` else upper bound
    on the max), in input order. Unlike the CPU path, every value is the
    NS-certified bound (sound for any solver accuracy), not the raw optimum.

    Args:
        objectives: ``k`` objective vectors, each length ``n``.
        A, b: Shared inequality constraints ``A x <= b`` (``None`` => box-only).
        lb, ub: Shared finite box bounds, length ``n``.
        minimize_flags: ``k`` booleans (True => minimize). Default all-minimize,
            matching ``solve_lp_batch``.
        max_iters: iteration cap of the dual descent.
        warm_duals: optional per-objective dual warm starts --
            length-``k`` sequence of arrays covering a PREFIX of the constraint
            rows (zero-padded to ``m``; ``None`` entries start cold). Sound for
            any values (clamped ``>= 0``); affects iterations-to-tightness only.
        return_duals: if True return ``(bounds, duals)`` where ``duals`` is the
            per-objective row-dual list actually certified — for caching and
            warm-starting descendant solves.
        device: torch device for the descent (``None``: CUDA when usable,
            else CPU).

    Returns:
        ``list[float]`` of certified bounds.
    """
    if not objectives:
        return []

    objs = [np.asarray(o, dtype=np.float64).reshape(-1) for o in objectives]
    n = objs[0].shape[0]
    if not all(o.shape[0] == n for o in objs):
        raise ValueError("All objectives must have the same length")

    k = len(objs)
    if minimize_flags is None:
        minimize_flags = [True] * k
    elif len(minimize_flags) != k:
        raise ValueError(
            f"minimize_flags length ({len(minimize_flags)}) must match "
            f"objectives length ({k})"
        )

    A_np = np.asarray(A, dtype=np.float64) if A is not None else None
    b_np = np.asarray(b, dtype=np.float64).reshape(-1) if b is not None else None
    lb_np = (np.full(n, -np.inf) if lb is None
             else np.asarray(lb, dtype=np.float64).reshape(-1))
    ub_np = (np.full(n, np.inf) if ub is None
             else np.asarray(ub, dtype=np.float64).reshape(-1))

    has_constraints = A_np is not None and b_np is not None and b_np.size > 0

    # Duals (one per objective) for the A x <= b rows. Box-only LPs need none.
    if not has_constraints:
        m = 0
        duals = [np.zeros(0) for _ in range(k)]
    else:
        m = b_np.shape[0]
        Y0 = None
        if warm_duals is not None:
            Y0 = np.zeros((m, k))
            for i, w in enumerate(warm_duals):
                if w is not None and w.size > 0:
                    r = min(w.size, m)
                    # A non-finite warm dual makes the NS certificate
                    # NaN; zero is always a valid multiplier.
                    Y0[:r, i] = np.nan_to_num(
                        np.clip(w[:r], 0.0, None), nan=0.0, posinf=0.0)
        duals = _grad_duals_gpu(
            objs, A_np, b_np, lb_np, ub_np, minimize_flags,
            max_iters=max_iters, Y0=Y0, device=device,
        )

    # Certify with the audited float64 NS implementation -- BATCHED (B=1
    # population, k objective columns). The scalar per-bound loop
    # re-materialized |A| per call, which dominated conv-scale reaches
    # (profiled 2026-07-19: 279 s of a 287 s OVAL-Base root reach was
    # ns_bound; ns_bounds_population is the same certified arithmetic, and
    # the Higham-style error constants bound ANY summation order, so batching
    # preserves the certificate's validity). Box-only LPs (no A) keep the
    # scalar path -- nothing to amortize there.
    count("n_gpu_lp_solves", k)
    if A_np is None or b_np is None or len(b_np) == 0:
        out: List[float] = [
            float(ns_bound(objs[i], duals[i], A=A_np, b=b_np, lb=lb_np,
                           ub=ub_np, minimize=minimize_flags[i]))
            for i in range(k)
        ]
    else:
        C_arr = np.stack([np.asarray(o, dtype=np.float64).flatten()
                          for o in objs], axis=1)              # (n, k)
        Y_arr = np.stack([np.asarray(d, dtype=np.float64).flatten()
                          for d in duals], axis=1)             # (m, k)
        bounds_arr = ns_bounds_population(
            C_arr[None], Y_arr[None],
            np.asarray(A_np, dtype=np.float64)[None],
            np.asarray(b_np, dtype=np.float64).flatten()[None],
            np.asarray(lb_np, dtype=np.float64).flatten()[None],
            np.asarray(ub_np, dtype=np.float64).flatten()[None],
            list(minimize_flags),
        )[0]
        out = [float(v) for v in bounds_arr]
    if return_duals:
        return out, duals
    return out


# --------------------------------------------------------------------------- #
# Witness heuristic: batched PDHG primal points (never a verification decision).
# --------------------------------------------------------------------------- #
def solve_lp_primal_population_gpu(
    Ap: np.ndarray,
    bp: np.ndarray,
    lbp: np.ndarray,
    ubp: np.ndarray,
    fp: np.ndarray,
    max_iters: int = 400,
    device: Optional[str] = None,
) -> np.ndarray:
    """Approximate PRIMAL minimizers for a POPULATION of single-objective LPs:
    per system ``B``: ``min f_B^T x  s.t.  A_B x <= b_B, lb_B <= x <= ub_B``.

    One 3D-batched PDHG run for all systems (two ``bmm`` per iteration), so a
    frontier wave's witness LPs cost one GPU call instead of one per branch --
    the sequential-iteration launch cost is paid once for the whole batch.
    Heterogeneous shapes are zero-padded upstream: an all-zero constraint row
    with ``b=0`` has zero residual (inert), and a padded variable with
    ``lb=ub=0`` is clamped to 0 every iteration (inert).

    The points are box-feasible BY CONSTRUCTION (clamped every iteration) but
    satisfy ``A x <= b`` only up to the primal residual -- heuristic use only
    (counterexample candidates re-validated by an exact forward pass; split
    scores tolerate approximate points). Never used for a verification
    decision.

    Args:
        Ap: ``(B, m, n)`` per-system constraints (zero-padded rows/cols).
        bp: ``(B, m)`` per-system RHS.
        lbp, ubp: ``(B, n)`` per-system finite box bounds.
        fp: ``(B, n)`` per-system minimization objectives.
        device: torch device (``None``: CUDA when usable, else CPU).

    Returns:
        ``(B, n)`` float64 array of primal iterates.
    """
    torch = _torch_mod()
    dev = torch.device(_resolve_device(device))
    f32 = torch.float32

    Ap = np.asarray(Ap, dtype=np.float64)
    B, m, n = Ap.shape

    with region("pdhg_population_primal", n_systems=B, n_constraints=m,
                n_vars=n):
        At = torch.tensor(Ap, dtype=f32, device=dev)             # (B, m, n)
        bt = torch.tensor(np.asarray(bp, dtype=np.float64),
                          dtype=f32, device=dev).reshape(B, m, 1)
        lbt = torch.tensor(np.asarray(lbp, dtype=np.float64),
                           dtype=f32, device=dev).reshape(B, n, 1)
        ubt = torch.tensor(np.asarray(ubp, dtype=np.float64),
                           dtype=f32, device=dev).reshape(B, n, 1)
        Gt = torch.tensor(np.asarray(fp, dtype=np.float64),
                          dtype=f32, device=dev).reshape(B, n, 1)

        # Per-system spectral norm via batched power iteration (padded rows
        # contribute nothing, so padded systems get small L -- harmless).
        v = torch.ones((B, n, 1), dtype=f32, device=dev)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        for _ in range(20):
            w = torch.bmm(At, v)                                 # (B, m, 1)
            v = torch.bmm(At.transpose(1, 2), w)                 # (B, n, 1)
            nv = v.norm(dim=1, keepdim=True)
            v = v / (nv + 1e-12)
        L = torch.sqrt(nv.clamp_min(1e-12)).reshape(B, 1, 1)     # (B, 1, 1)
        step = 0.9 / L.clamp_min(1e-12)
        tau = sigma = step

        X = torch.clamp(torch.zeros((B, n, 1), dtype=f32, device=dev),
                        lbt, ubt)
        Y = torch.zeros((B, m, 1), dtype=f32, device=dev)

        for _ in range(max_iters):
            AtY = torch.bmm(At.transpose(1, 2), Y)               # (B, n, 1)
            X_new = torch.clamp(X - tau * (Gt + AtY), lbt, ubt)
            X_bar = 2.0 * X_new - X
            Y = torch.clamp_min(Y + sigma * (torch.bmm(At, X_bar) - bt), 0.0)
            X = X_new

        return X.double().cpu().numpy().reshape(B, n)


# --------------------------------------------------------------------------- #
# Bounds: projected-Adam descent on the NS dual (best-iterate). float32 solve.
# --------------------------------------------------------------------------- #
def _grad_duals_gpu(
    objs, A, b, lb, ub, minimize_flags, max_iters: int,
    lrs=(1.0, 0.1, 0.01), lr_decay: float = 0.997,
    Y0: Optional[np.ndarray] = None,
    device: Optional[str] = None,
):
    """Batched projected-Adam descent on the NS dual -> best-iterate row duals.

    The NS certificate value *is* the Lagrangian dual function of the
    box-constrained LP: ``g(y) = b^T y + sum_j max(r_j lb_j, r_j ub_j)`` with
    ``r = g_max - A^T y`` -- convex piecewise-linear in ``y >= 0`` with the
    closed-form subgradient ``b - A x*(y)`` (``x*`` the box argmax of ``r``).
    Descending ``g`` therefore optimizes the certified bound *directly*, and
    because ``g`` is evaluated as a byproduct of the subgradient computation,
    the per-column **best iterate** is tracked and returned -- a monotone
    anytime bound, immune to the last-iterate oscillation that makes truncated
    PDHG duals unreliable (measured: the PDHG last iterate recovered as little
    as 0.7% of the box->exact gap where this method recovered >= 93%).

    A small learning-rate grid is folded into the batch axis (columns tiled
    per lr); the best (lr, iterate) pair is selected per objective at the end.

    ``Y0`` (optional, ``(m, k)``, clamped to ``>= 0``) warm-starts the descent —
    e.g. a parent BaB node's duals on the shared constraint-row prefix. Any
    ``Y0`` is sound (weak duality holds for every nonnegative dual); a good one
    starts the certified bound near the parent's and saves iterations.

    Thin wrapper over :func:`_grad_population_duals_gpu` with ``B = 1`` — ONE
    optimizer implementation, so the shared-``A`` and population paths cannot
    drift numerically.
    """
    C = np.stack([np.asarray(c, dtype=np.float64) for c in objs], axis=1)
    Y = _grad_population_duals_gpu(
        np.asarray(A, dtype=np.float64)[None],
        np.asarray(b, dtype=np.float64).reshape(1, -1),
        np.asarray(lb, dtype=np.float64).reshape(1, -1),
        np.asarray(ub, dtype=np.float64).reshape(1, -1),
        C[None], minimize_flags, max_iters=max_iters, lrs=lrs,
        lr_decay=lr_decay,
        Y0=None if Y0 is None else np.asarray(Y0, dtype=np.float64)[None],
        device=device,
    )[0]
    return [Y[:, i] for i in range(C.shape[1])]


# --------------------------------------------------------------------------- #
# Population batch: one 3D batched dual descent over many stars at once.
# --------------------------------------------------------------------------- #
def solve_lp_population_gpu(
    A: np.ndarray,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    C: np.ndarray,
    minimize_flags: Sequence[bool],
    max_iters: int = 300,
    Y0: Optional[np.ndarray] = None,
    return_duals: bool = False,
    device: Optional[str] = None,
):
    """Certified bounds for a *population* of LPs, each with its own constraints.

    Solves ``B`` stars' bound LPs in a single 3D batched projected-Adam descent
    on the NS dual (one ``bmm`` per step over the star axis, best iterate
    kept), then certifies every (star, objective) with the vectorized NS
    certificate. The shared objective count ``k`` and var count ``n`` let the
    whole population ride one GPU kernel; heterogeneous constraint counts are
    handled by zero-row padding upstream (inert under NS).

    Args:
        A: ``(B, m, n)`` per-star constraints (zero-padded rows allowed).
        b: ``(B, m)`` per-star RHS (0 on padded rows).
        lb, ub: ``(B, n)`` per-star finite box bounds.
        C: ``(B, n, k)`` per-star objective coefficients.
        minimize_flags: length-``k`` booleans, shared across stars.
        max_iters: iteration cap of the dual descent.
        Y0: optional ``(B, m, k)`` dual warm start (clamped >= 0).
        return_duals: if True return ``(bounds, Y)`` for caching/warm-starting.
        device: torch device for the descent (``None``: CUDA when usable,
            else CPU).

    Returns:
        ``(B, k)`` array of NS-certified bounds (or ``(bounds, Y)``).
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    B, m, n = A.shape
    k = C.shape[2]

    Y = _grad_population_duals_gpu(
        A, b, lb, ub, C, minimize_flags, max_iters=max_iters, Y0=Y0,
        device=device,
    )
    count("n_gpu_lp_solves", B * k)
    bounds = ns_bounds_population(C, Y, A, b, lb, ub, minimize_flags)
    if return_duals:
        return bounds, Y
    return bounds


def _grad_population_duals_gpu(
    A, b, lb, ub, C, minimize_flags, max_iters: int,
    lrs=(1.0, 0.1, 0.01), lr_decay: float = 0.997,
    Y0: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    """Population form of :func:`_grad_duals_gpu`: projected-Adam descent on the
    per-(star, objective) NS dual, best iterate kept, lr grid folded into the
    objective axis. Padded rows (zero ``A`` row, zero ``b``) and padded
    objective columns (zero ``C`` column) are inert. Returns ``(B, m, k)``.
    """
    torch = _torch_mod()
    dev = torch.device(_resolve_device(device))
    f32 = torch.float32
    B, m, n = A.shape
    k = C.shape[2]
    L = len(lrs)

    sign = np.where(np.asarray(minimize_flags, dtype=bool), -1.0, 1.0)
    G = (C * sign[None, None, :]).astype(np.float32)             # (B, n, k) max-sense

    with region("grad_population_solve", n_stars=B, n_objectives=k,
                n_constraints=m, n_vars=n):
        At = torch.tensor(A, dtype=f32, device=dev)              # (B, m, n)
        AtT = At.transpose(1, 2).contiguous()                    # (B, n, m)
        bt = torch.tensor(b, dtype=f32, device=dev)              # (B, m)
        lbt = torch.tensor(lb, dtype=f32, device=dev).unsqueeze(2)  # (B, n, 1)
        ubt = torch.tensor(ub, dtype=f32, device=dev).unsqueeze(2)
        # upload once, expand the lr grid ON DEVICE (no host-side tile / extra
        # PCIe traffic for byte-identical slots)
        Gt = torch.tensor(G, device=dev).repeat(1, 1, L)         # (B, n, kL)
        lr = torch.tensor(np.repeat(np.asarray(lrs, np.float64), k),
                          dtype=f32, device=dev)                 # (kL,)

        if Y0 is None:
            Y = torch.zeros((B, m, k * L), dtype=f32, device=dev)
        else:
            Y = (torch.tensor(np.asarray(Y0, dtype=np.float32), device=dev)
                 .clamp_min(0.0).repeat(1, 1, L))
        mom = torch.zeros_like(Y)
        vel = torch.zeros_like(Y)
        best_f = torch.full((B, k * L), float("inf"), dtype=f32, device=dev)
        best_Y = Y.clone()
        b1, b2, eps = 0.9, 0.999, 1e-8

        def value_and_xstar(Y):
            R = Gt - torch.bmm(AtT, Y)                           # (B, n, kL)
            Xs = torch.where(R > 0, ubt, lbt)
            f = torch.einsum("bm,bmk->bk", bt, Y) + (R * Xs).sum(1)
            return f, Xs

        for t in range(max_iters):
            f, Xs = value_and_xstar(Y)
            imp = f < best_f                                     # (B, kL)
            best_f = torch.where(imp, f, best_f)
            mask = imp.unsqueeze(1)                              # (B, 1, kL)
            best_Y = torch.where(mask, Y, best_Y)
            grad = bt.unsqueeze(2) - torch.bmm(At, Xs)           # (B, m, kL)
            mom = b1 * mom + (1 - b1) * grad
            vel = b2 * vel + (1 - b2) * grad * grad
            step = (mom / (1 - b1 ** (t + 1))) / (
                (vel / (1 - b2 ** (t + 1))).sqrt() + eps)
            Y = (Y - lr * (lr_decay ** t) * step).clamp_min(0.0)

        f, _ = value_and_xstar(Y)
        imp = f < best_f
        best_f = torch.where(imp, f, best_f)
        best_Y = torch.where(imp.unsqueeze(1), Y, best_Y)

        pick = best_f.view(B, L, k).argmin(1)                    # (B, k)
        idx = pick * k + torch.arange(k, device=dev).unsqueeze(0)  # (B, k)
        Yb = torch.gather(best_Y, 2, idx.unsqueeze(1).expand(B, m, k))
        set_meta(grad_iters=max_iters, lr_grid=L)
        # transfer float32 (half the D2H bytes); the NS certificate casts to
        # float64 on the host itself, so precision of the BOUND is unaffected
        # (any dual is valid -- dtype only perturbs which dual we certify)
        Y_host = Yb.cpu().numpy()                                # (B, m, k) f32
    return Y_host


__all__ = ["solve_lp_batch_gpu", "solve_lp_population_gpu",
           "solve_lp_primal_population_gpu", "gpu_available"]

"""Batched first-order (GPU) sound LP backend for Star bound computation (PT-2).

n2v's per-star bound computation issues *many small, structurally-identical*
LPs: per star, ``2*dim`` axis objectives share one constraint set
(``A x <= b, lb <= x <= ub``); across a layer's star population there are
hundreds-to-thousands more. This is exactly the regime batched first-order GPU
LP targets. This module solves a whole batch at once with a batched PDHG
(primal-dual hybrid gradient -- the PDLP inner loop) and returns
**Neumaier-Shcherbina certified** bounds, so the result is *sound* regardless of
the first-order solver's (low) accuracy: PDHG is only a heuristic for producing
a dual; the certificate (:mod:`n2v.utils.ns_certificate`) turns any dual into a
guaranteed bound by weak duality.

Design choices (see the PT-2 plan):
  * **Mixed precision** -- PDHG runs in float32 (the A30's fast path) to produce
    an approximate dual; the NS certificate runs in float64 (audited CPU code)
    for the sound bound. Solver precision affects only tightness, never soundness.
  * **Shared-constraint batch** -- all objectives in a call share ``A,b,lb,ub``,
    so every PDHG step is a pair of plain (batched) mat-muls ``A^T Y`` / ``A X``;
    the spectral-norm step size is estimated once for the shared ``A``.
  * **Lazy torch import** -- the module imports cleanly without torch/CUDA; the
    ``cpu_ns`` backend (HiGHS + NS) needs no GPU and serves as reference/fallback.

The single value ever returned is the certified bound. Never the raw PDHG value.
"""

from typing import List, Optional, Sequence, Union

import numpy as np

from contextlib import contextmanager

from n2v.profiling import count, is_enabled, region, set_meta
from n2v.utils.ns_certificate import ns_bound, ns_bounds_population

try:
    from n2v.profiling import add_gpu_time
except ImportError:  # PT-2 GPU-timing profiler hook absent on this branch
    def add_gpu_time(_seconds):  # no-op fallback (profiling-only)
        return None

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
    """True iff a CUDA device is usable for the ``pdhg`` backend."""
    try:
        return _torch_mod().cuda.is_available()
    except Exception:
        return False


@contextmanager
def _cuda_timed():
    """Measure enclosed CUDA-kernel time and report it to the profiler.

    Uses ``torch.cuda.Event`` + ``synchronize()`` (a small, documented
    perturbation -- the design's prescribed GPU-timing mechanism) and forwards
    seconds to :func:`n2v.profiling.add_gpu_time`. A true no-op when profiling is
    disabled, so the hot path pays nothing. CUDA-event timing keeps the profiler
    core torch-free: the device math lives here, beside the kernels it times.
    """
    if not is_enabled():
        yield
        return
    torch = _torch_mod()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        torch.cuda.synchronize()
        add_gpu_time(start.elapsed_time(end) / 1000.0)  # ms -> s


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
    backend: str = "pdhg",
    max_iters: int = 2000,
    tol: float = 1e-6,
) -> List[float]:
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
        backend: ``"pdhg"`` (GPU first-order) or ``"cpu_ns"`` (HiGHS + NS, no GPU).
        max_iters: PDHG iteration cap (``pdhg`` only).
        tol: PDHG relative KKT-residual early-stop tolerance (``pdhg`` only).

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
        if backend == "pdhg":
            duals = _pdhg_duals_gpu(
                objs, A_np, b_np, lb_np, ub_np, minimize_flags,
                max_iters=max_iters, tol=tol,
            )
        elif backend == "cpu_ns":
            duals = _highs_duals_cpu(
                objs, A_np, b_np, lb_np, ub_np, minimize_flags,
            )
        else:
            raise ValueError(f"unknown backend {backend!r}; expected 'pdhg'|'cpu_ns'")

    # Certify on CPU with the single audited float64 NS implementation. The dual
    # is small (k x m); the transfer/loop is negligible vs the PDHG iterations.
    count("n_gpu_lp_solves", k)
    out: List[float] = []
    for i in range(k):
        bd = ns_bound(objs[i], duals[i], A=A_np, b=b_np, lb=lb_np, ub=ub_np,
                      minimize=minimize_flags[i])
        out.append(float(bd))
    return out


# --------------------------------------------------------------------------- #
# Backend: CPU reference (HiGHS solve -> dual -> NS). No GPU required.
# --------------------------------------------------------------------------- #
def _highs_duals_cpu(
    objs, A, b, lb, ub, minimize_flags,
) -> List[np.ndarray]:
    """Per-objective row duals from HiGHS for ``min/max c^T x s.t. Ax<=b, box``.

    Used by the ``cpu_ns`` backend (the GPU-free reference that proves the
    approximate-solve + NS-certificate pipeline) and as the certified fallback.
    """
    import highspy

    m = b.shape[0]
    n = objs[0].shape[0]
    inf = highspy.kHighsInf

    h = highspy.Highs()
    h.silent()
    h.addVars(n, lb, ub)
    for i in range(m):
        row = A[i, :]
        nz = np.nonzero(row)[0]
        h.addRow(-inf, float(b[i]), len(nz), nz.astype(np.int32), row[nz])
    col_idx = np.arange(n, dtype=np.int32)

    duals: List[np.ndarray] = []
    for c, do_min in zip(objs, minimize_flags):
        h.changeObjectiveSense(
            highspy.ObjSense.kMinimize if do_min else highspy.ObjSense.kMaximize
        )
        h.changeColsCost(n, col_idx, c)
        h.run()
        if h.getModelStatus() == highspy.HighsModelStatus.kOptimal:
            yd = np.array(h.getSolution().row_dual, dtype=np.float64)
            # NS clamps to >=0 internally; orient toward the active sign so the
            # certificate is tight (HiGHS reports active-constraint duals with a
            # sign that depends on sense -- |.| recovers the magnitude, clamping
            # discards the inactive sign. Soundness holds for either choice.)
            duals.append(np.abs(yd))
        else:
            duals.append(np.zeros(m))  # box bound -- still sound
        h.clearSolver()
    return duals


# --------------------------------------------------------------------------- #
# Backend: batched PDHG on GPU (the PDLP inner loop). float32 solve.
# --------------------------------------------------------------------------- #
def _pdhg_duals_gpu(
    objs, A, b, lb, ub, minimize_flags, max_iters: int, tol: float,
) -> List[np.ndarray]:
    """Batched primal-dual hybrid gradient -> approximate row duals (one per obj).

    Solves, for every objective at once, the minimization saddle point of
    ``min_x g^T x  s.t.  A x <= b, lb <= x <= ub`` where ``g = c`` if that
    objective minimizes else ``g = -c`` (so the returned dual is the multiplier
    of the constraints for the certificate's sense, handled in :func:`ns_bound`).

    All objectives share ``A`` -> each iteration is two batched mat-muls. Runs in
    float32 on CUDA; the dual is returned to host as float64 for the certificate.
    """
    torch = _torch_mod()
    dev = torch.device("cuda")
    f32 = torch.float32

    m, n = A.shape
    k = len(objs)

    with region("pdhg_solve", n_objectives=k, n_constraints=m, n_vars=n), _cuda_timed():
        At = torch.tensor(A, dtype=f32, device=dev)            # (m, n)
        bt = torch.tensor(b, dtype=f32, device=dev).reshape(m, 1)
        # Finite-box clamp tensors; +-inf is preserved by torch.clamp.
        lbt = torch.tensor(lb, dtype=f32, device=dev).reshape(n, 1)
        ubt = torch.tensor(ub, dtype=f32, device=dev).reshape(n, 1)

        # Stack the (minimization) objective of every batch member: G is (n, k).
        G = np.stack([
            (c if do_min else -c) for c, do_min in zip(objs, minimize_flags)
        ], axis=1).astype(np.float32)
        Gt = torch.tensor(G, device=dev)                        # (n, k)

        # Spectral norm of the shared A via power iteration (once for the batch).
        L = _spectral_norm(At, torch)
        step = 0.9 / max(L, 1e-12)
        tau = sigma = step

        X = torch.zeros((n, k), dtype=f32, device=dev)
        Y = torch.zeros((m, k), dtype=f32, device=dev)
        X = torch.clamp(X, lbt, ubt)

        check_every = 50
        for it in range(max_iters):
            AtY = At.t() @ Y                                    # (n, k)
            X_new = torch.clamp(X - tau * (Gt + AtY), lbt, ubt)
            X_bar = 2.0 * X_new - X                             # extrapolation
            Y = torch.clamp_min(Y + sigma * (At @ X_bar - bt), 0.0)
            X = X_new

            if tol > 0 and (it + 1) % check_every == 0:
                # KKT residual: primal infeasibility max(Ax - b, 0) and dual
                # stationarity are bounded; cheap proxy = primal residual norm.
                primal_res = torch.clamp_min(At @ X - bt, 0.0)
                r = float(primal_res.abs().max().item())
                scale = float(bt.abs().max().item()) + 1.0
                if r / scale < tol:
                    break

        set_meta(pdhg_iters=it + 1, spectral_norm=float(L))
        Y_host = Y.double().cpu().numpy()                       # (m, k) float64

    return [Y_host[:, i] for i in range(k)]


# --------------------------------------------------------------------------- #
# Cross-population batch (4a-ii): one 3D batched PDHG over many stars at once.
# --------------------------------------------------------------------------- #
def solve_lp_population_gpu(
    A: np.ndarray,
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    C: np.ndarray,
    minimize_flags: Sequence[bool],
    max_iters: int = 2000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Certified bounds for a *population* of LPs, each with its own constraints.

    Solves ``B`` stars' bound LPs in a single 3D batched PDHG (one ``bmm`` per
    step over the star axis), then certifies every (star, objective) with the
    vectorized NS certificate. The shared objective count ``k`` and var count
    ``n`` let the whole population ride one GPU kernel; heterogeneous constraint
    counts are handled by zero-row padding upstream (inert under NS).

    Args:
        A: ``(B, m, n)`` per-star constraints (zero-padded rows allowed).
        b: ``(B, m)`` per-star RHS (0 on padded rows).
        lb, ub: ``(B, n)`` per-star finite box bounds.
        C: ``(B, n, k)`` per-star objective coefficients.
        minimize_flags: length-``k`` booleans, shared across stars.

    Returns:
        ``(B, k)`` array of NS-certified bounds.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    B, m, n = A.shape
    k = C.shape[2]

    Y = _pdhg_population_duals_gpu(
        A, b, lb, ub, C, minimize_flags, max_iters=max_iters, tol=tol,
    )
    count("n_gpu_lp_solves", B * k)
    return ns_bounds_population(C, Y, A, b, lb, ub, minimize_flags)


def _pdhg_population_duals_gpu(
    A, b, lb, ub, C, minimize_flags, max_iters: int, tol: float,
) -> np.ndarray:
    """Batched (over stars) PDHG -> per-(star, objective) approximate duals.

    ``A`` is ``(B, m, n)`` with a *distinct* constraint matrix per star, so each
    step is a batched matrix product (``torch.bmm``) over the star axis. The
    spectral-norm step size is estimated per star by batched power iteration.
    """
    torch = _torch_mod()
    dev = torch.device("cuda")
    f32 = torch.float32
    B, m, n = A.shape
    k = C.shape[2]

    sign = np.where(np.asarray(minimize_flags, dtype=bool), -1.0, 1.0).astype(np.float32)
    # Minimization objective per (star, column): g = c if min else -c. (sign=-1
    # for minimize, so g = c * (-sign)? careful) -- PDHG minimizes g^T x; for a
    # max objective we minimize -c. So g = -c for max, +c for min => g = -sign*c.
    G = (-sign)[None, None, :] * C.astype(np.float32)            # (B, n, k)

    with region("pdhg_population_solve", n_stars=B, n_objectives=k,
                n_constraints=m, n_vars=n), _cuda_timed():
        At = torch.tensor(A, dtype=f32, device=dev)             # (B, m, n)
        AtT = At.transpose(1, 2).contiguous()                   # (B, n, m)
        bt = torch.tensor(b, dtype=f32, device=dev).unsqueeze(2)  # (B, m, 1)
        lbt = torch.tensor(lb, dtype=f32, device=dev).unsqueeze(2)  # (B, n, 1)
        ubt = torch.tensor(ub, dtype=f32, device=dev).unsqueeze(2)
        Gt = torch.tensor(G, device=dev)                        # (B, n, k)

        L = _spectral_norm_batched(At, AtT, torch)              # (B, 1, 1)
        step = 0.9 / torch.clamp(L, min=1e-12)
        tau = sigma = step                                      # (B, 1, 1)

        X = torch.clamp(torch.zeros((B, n, k), dtype=f32, device=dev), lbt, ubt)
        Y = torch.zeros((B, m, k), dtype=f32, device=dev)

        check_every = 50
        it = 0
        for it in range(max_iters):
            AtY = torch.bmm(AtT, Y)                             # (B, n, k)
            X_new = torch.clamp(X - tau * (Gt + AtY), lbt, ubt)
            X_bar = 2.0 * X_new - X
            Y = torch.clamp_min(Y + sigma * (torch.bmm(At, X_bar) - bt), 0.0)
            X = X_new
            if tol > 0 and (it + 1) % check_every == 0:
                primal_res = torch.clamp_min(torch.bmm(At, X) - bt, 0.0)
                r = float(primal_res.abs().max().item())
                scale = float(bt.abs().max().item()) + 1.0
                if r / scale < tol:
                    break

        set_meta(pdhg_iters=it + 1)
        Y_host = Y.double().cpu().numpy()                       # (B, m, k)
    return Y_host


def _spectral_norm_batched(At, AtT, torch, iters: int = 30):
    """Per-star ``||A_s||_2`` via batched power iteration. Returns ``(B, 1, 1)``."""
    B, m, n = At.shape
    v = torch.randn((B, n, 1), dtype=At.dtype, device=At.device)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-30)
    s = torch.zeros((B, 1, 1), dtype=At.dtype, device=At.device)
    for _ in range(iters):
        Av = torch.bmm(At, v)                                   # (B, m, 1)
        w = torch.bmm(AtT, Av)                                  # (B, n, 1)
        s = w.norm(dim=1, keepdim=True)                         # (B, 1, 1)
        v = w / (s + 1e-30)
    return torch.sqrt(s)


def _spectral_norm(At, torch, iters: int = 30) -> float:
    """Estimate ||A||_2 by power iteration on A^T A (A shared across the batch)."""
    m, n = At.shape
    v = torch.randn((n, 1), dtype=At.dtype, device=At.device)
    v = v / (v.norm() + 1e-30)
    s = torch.tensor(0.0, device=At.device, dtype=At.dtype)
    for _ in range(iters):
        Av = At @ v
        w = At.t() @ Av
        s = w.norm()
        v = w / (s + 1e-30)
    # s ~ ||A^T A v|| ~ sigma_max^2 after convergence; ||A||_2 = sqrt(sigma_max^2).
    return float(torch.sqrt(s).item())


__all__ = ["solve_lp_batch_gpu", "solve_lp_population_gpu", "gpu_available"]

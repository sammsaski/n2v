"""
Finite predicate boxes for star sets.

Interval bounds over a star (``Star.estimate_ranges``) are undefined on
infinite predicate intervals (``0 * inf = NaN``), and a NaN bound reads as
"stably active" in the ReLU classifiers. ``ensure_finite_predicate_box``
replaces non-finite box entries with exact-LP bounds over the full predicate
polytope, so the star is unchanged as a set.
"""

import numpy as np

# The exact-LP box bounds in ``ensure_finite_predicate_box`` are pushed
# outward by the shared LP pad (``n2v.utils.lpsolver._lp_pad``, 1e-7 relative),
# which guards the float LP optimum against shaving a boundary sliver off the
# true range: HiGHS's default primal/dual feasibility tolerances are 1e-7, so a
# smaller pad can sit below the solver's own optimum error on ill-conditioned
# polytopes (measured: up to ~5e-9 undershoot on thin-wedge systems).


def ensure_finite_predicate_box(star, lp_solver=None):
    """Return ``(star2, empty)``: the same set with a fully FINITE predicate box.

    Interval bound machinery is undefined on infinite intervals
    (``0 * inf = NaN`` poisons bounds silently -- a demonstrated false-UNSAT
    path), so reach and verification entry points call this first.

    A fully finite box returns ``(star, False)`` unchanged, no LPs. Otherwise
    every non-finite bound entry is recomputed as the LP optimum of that
    predicate variable over the FULL polytope ``{C alpha <= d, box}`` and
    pushed outward by a relative epsilon -- the new box still contains every
    feasible point, so the star is unchanged as a set and every downstream
    use stays sound. The bounds are LP optima, not certificates: an NS
    certificate would need a finite box, which is what is being computed.
    Each bound is therefore solved twice, with presolve off
    (``_solve_lp_careful``) and with the caller's solver, and the more
    outward optimum is kept: on ill-conditioned polytopes either solve can
    land inside the true range by more than the pad (measured up to ~4e-5
    relative on column-scaled polytopes), but both on the same side only
    about once in 3000 bound LPs. The box is then widened to contain every
    LP solution point, and bound LPs that contradict each other (a lower
    bound above an upper bound) raise instead of producing an empty box,
    which would read as a false UNSAT. ``empty=True`` means every bound LP
    that decided reported the polytope infeasible AND ``confirm_infeasible``
    agrees: the star covers no points (vacuously safe for verification).

    Args:
        star: flat Star; ``predicate_lb``/``predicate_ub`` may be ``None``
            (treated as all-infinite) or contain ``+/-inf`` entries. ``NaN``
            entries anywhere in the box or the ``C``/``d`` rows raise: NaN is
            numerical corruption (the very artifact of the ``0 * inf`` bug
            class this helper defends against), not a bound with interval
            semantics -- and it makes LP solvers fail in ways that must never
            be mistaken for an emptiness certificate.
        lp_solver: solver for the second solve of each bound (``None`` or
            ``'default'`` resolves via the global config); the first always
            takes the presolve-off HiGHS path.

    Raises:
        ValueError: NaN in the box or ``C``/``d``; a predicate variable
            reported unbounded over the polytope; no bound LP deciding; an
            'infeasible' report that the emptiness check does not confirm
            (the polytope is empty or within solver tolerance of empty); or
            bound LPs that contradict each other -- in every case no
            trustworthy finite box exists and the caller's engine cannot
            proceed soundly.
    """
    n = star.nVar
    lb = (np.full((n, 1), -np.inf) if star.predicate_lb is None
          else np.asarray(star.predicate_lb, dtype=np.float64).reshape(-1, 1))
    ub = (np.full((n, 1), np.inf) if star.predicate_ub is None
          else np.asarray(star.predicate_ub, dtype=np.float64).reshape(-1, 1))
    if np.any(np.isnan(lb)) or np.any(np.isnan(ub)):
        raise ValueError(
            "the star's predicate box contains NaN -- corrupted bounds, not "
            "an interval; refusing to finitize (a NaN-poisoned LP failure "
            "must never pass for an emptiness certificate)"
        )
    if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
        return star, False

    from n2v.sets.star import Star
    from n2v.utils.lpsolver import (_lp_pad, _solve_lp_careful,
                                    confirm_infeasible, solve_lp)

    A = star.C if star.C is not None and star.C.size > 0 else None
    b = np.asarray(star.d).flatten() if A is not None else None
    if A is not None and not (np.all(np.isfinite(A)) and
                              np.all(np.isfinite(b))):
        raise ValueError(
            "the star's C/d rows contain non-finite entries -- refusing to "
            "finitize the predicate box over a corrupted polytope"
        )
    lb_f, ub_f = lb.flatten(), ub.flatten()
    new_lb, new_ub = lb.copy(), ub.copy()
    points = []
    for dim in range(n):
        for minimize, bnd in ((True, new_lb), (False, new_ub)):
            if np.isfinite(bnd[dim, 0]):
                continue
            f = np.zeros(n)
            f[dim] = 1.0
            solves = [
                _solve_lp_careful(f, A, b, lb_f, ub_f, minimize=minimize),
                solve_lp(f, A, b, lb=lb_f, ub=ub_f, lp_solver=lp_solver,
                         minimize=minimize),
            ]
            statuses = [status for _, _, status, _ in solves]
            vals = [val for _, val, status, _ in solves
                    if status in ('optimal', 'optimal_inaccurate')
                    and val is not None and np.isfinite(val)]
            if 'unbounded' in statuses:
                raise ValueError(
                    f"predicate variable {dim} is reported unbounded over "
                    f"the star's predicate polytope (statuses {statuses!r}): "
                    f"no trustworthy finite predicate box exists, and the "
                    f"certified bound engines require one"
                )
            if not vals:
                if 'infeasible' in statuses:
                    # SOUNDNESS: an LP status is not an emptiness certificate
                    # on tiny-scale systems; empty=True makes reach return
                    # nothing and the verdict a vacuous UNSAT, so it must be
                    # confirmed.
                    if confirm_infeasible(A, b, lb_f, ub_f):
                        return star, True
                    raise ValueError(
                        "the bound LP reported the star's predicate polytope "
                        "infeasible, but emptiness could not be confirmed (it "
                        "is empty or within solver tolerance of empty): no "
                        "trustworthy finite predicate box exists"
                    )
                raise ValueError(
                    f"the bound LPs for predicate variable {dim} failed "
                    f"(statuses {statuses!r}): no trustworthy finite "
                    f"predicate box exists, and the certified bound engines "
                    f"require one"
                )
            # the more outward optimum; an 'infeasible' from the other solve
            # is outvoted by this feasible optimum (emptiness is never
            # claimed without confirm_infeasible)
            val = min(vals) if minimize else max(vals)
            pad = _lp_pad(val)
            bnd[dim, 0] = val - pad if minimize else val + pad
            points += [np.asarray(x, dtype=np.float64).flatten()
                       for x, _, status, _ in solves
                       if status in ('optimal', 'optimal_inaccurate')
                       and x is not None]
    # SOUNDNESS: a lower bound above an upper bound is an empty box, i.e. a
    # vacuous UNSAT for every spec; on a polytope the LPs did not call
    # infeasible it can only mean they contradict each other.
    bad = np.flatnonzero(new_lb[:, 0] > new_ub[:, 0])
    if bad.size:
        raise ValueError(
            f"the bound LPs contradict each other on predicate variables "
            f"{bad.tolist()} (lower bound above upper bound): no trustworthy "
            f"finite predicate box exists"
        )
    # Every LP solution point is feasible up to solver tolerance, so the box
    # must contain it (with the same pad); one that lies outside means another
    # bound LP stopped inside the true range. Widening only enlarges the box.
    pts = [p for p in points if p.size == n and np.all(np.isfinite(p))]
    if pts:
        P = np.stack(pts, axis=1)
        p_lo, p_hi = P.min(axis=1), P.max(axis=1)
        p_lo = p_lo - _lp_pad(p_lo)
        p_hi = p_hi + _lp_pad(p_hi)
        free_lb, free_ub = ~np.isfinite(lb[:, 0]), ~np.isfinite(ub[:, 0])
        new_lb[free_lb, 0] = np.minimum(new_lb[free_lb, 0], p_lo[free_lb])
        new_ub[free_ub, 0] = np.maximum(new_ub[free_ub, 0], p_hi[free_ub])
    star2 = Star(star.V, star.C, star.d, new_lb, new_ub,
                 state_lb=star.state_lb, state_ub=star.state_ub,
                 outer_zono=star.Z, relax_meta=star.relax_meta)
    return star2, False

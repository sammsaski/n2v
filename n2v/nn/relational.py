"""
Relational (multi-network) reachability and verification.

VNN-COMP relational benchmarks (isomorphic_acasxu, monotonic_acasxu)
declare TWO networks f and g over a JOINT, coupled input space and assert
a property over the JOINT output [Y_f; Y_g]. The parser
(`_load_relational`) lowers a spec to:

    spec['networks']       : [{name, input_offset, input_size,
                               output_offset, output_size, relation}, ...]
    spec['lb'], spec['ub'] : joint input box over [X_f; X_g]
    spec['input_coupling'] : HalfSpace  G_in . x_in <= h_in   (equalities
                             appear as +/- row pairs; inequalities single)
    spec['prop']           : list[{'Hg': HalfSpace|list}] unsafe regions
                             over the joint output [Y_f; Y_g]

Engine — self-composition (product construction) with a prefix-aligned
predicate join:

  1. Build ONE joint input Star over the coupled input (x = alpha; the
     box gives the predicate bounds, LP-tightened against the coupling so
     equality/inequality-coupled dims become finite; the coupling is
     carried as the star's C/d).
  2. Slice the network sub-inputs x_f, x_g as sub-stars that SHARE the
     joint predicate vector and coupling.
  3. Reach f on x_f and g on x_g independently. When a network's reach is
     known to keep the input predicates as an aligned prefix (every node is
     a Linear, ReLU or Identity module; see
     `n2v.utils.verify_nonlinear.network_keeps_input_predicates`), its
     output star lives over [shared input predicates | its own appended
     ReLU predicates]. Otherwise every predicate of its output star is its
     own: a kernel such as softmax re-encodes its output with
     Star.fresh_from_bounds, and reading those fresh predicates as the input
     would invent an input-output correlation (a false UNSAT).
  4. Join into one joint output star [Y_f; Y_g] over
     [input | f's own | g's own] predicates. The input predicates take
     their bounds and coupling C/d from the joint input star; a network
     with a shared prefix keeps it IDENTIFIED with them, which preserves
     the input coupling exactly — the property of relational verification
     that a plain block-diagonal join would destroy.
  5. Check the joint output star against the unsafe `prop` with the same
     `verify_specification` used for single-network specs.

Soundness of the join: for any concrete coupled input (alpha_in) there
exist own assignments alpha_f, alpha_g realizing f and g's relaxations
(for a shared network over [alpha_in; alpha_f], otherwise over alpha_f
alone), so [f(x_f); g(x_g)] is realized at [alpha_in; alpha_f; alpha_g],
which satisfies the joint constraints.
"""

import logging

import numpy as np
from scipy.optimize import linprog

from n2v.sets import Star
from n2v.nn import NeuralNetwork
from n2v.utils.verify_nonlinear import network_keeps_input_predicates
from n2v.utils.verify_specification import verify_specification

logger = logging.getLogger(__name__)


def _tighten_predicate_box(lb, ub, G, h):
    """Finite box enclosing {x : lb<=x<=ub (finite parts) and G x <= h}.

    Coupled/unbounded dims (lb/ub = +/-inf) become finite through the
    coupling. Raises if a dim stays unbounded (neither boxed nor coupled).
    """
    lb = np.asarray(lb, dtype=np.float64).flatten()
    ub = np.asarray(ub, dtype=np.float64).flatten()
    n = lb.size
    plb, pub = lb.copy(), ub.copy()
    bounds = [(lo if np.isfinite(lo) else None,
               hi if np.isfinite(hi) else None) for lo, hi in zip(lb, ub)]
    G = np.asarray(G, dtype=np.float64).reshape(-1, n) if np.asarray(G).size \
        else np.zeros((0, n))
    h = np.asarray(h, dtype=np.float64).flatten() if np.asarray(h).size \
        else np.zeros(0)
    A_ub = G if G.shape[0] else None
    b_ub = h if G.shape[0] else None
    for i in range(n):
        if not np.isfinite(plb[i]):
            c = np.zeros(n); c[i] = 1.0
            r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if r.success and np.isfinite(r.fun):
                plb[i] = r.fun
        if not np.isfinite(pub[i]):
            c = np.zeros(n); c[i] = -1.0
            r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if r.success and np.isfinite(-r.fun):
                pub[i] = -r.fun
    if not (np.all(np.isfinite(plb)) and np.all(np.isfinite(pub))):
        bad = np.where(~(np.isfinite(plb) & np.isfinite(pub)))[0].tolist()
        raise ValueError(
            f"joint input dims {bad} are neither bounded nor coupled to a "
            f"bounded dim")
    return plb, pub


def build_joint_input_star(spec):
    """Joint input Star over [X_f; X_g] (x = alpha): box -> predicate
    bounds (LP-tightened by the coupling), coupling -> C/d."""
    lb = np.asarray(spec['lb'], dtype=np.float64).flatten()
    ub = np.asarray(spec['ub'], dtype=np.float64).flatten()
    n = lb.size
    coup = spec.get('input_coupling')
    G = coup.G if coup is not None else np.zeros((0, n))
    h = coup.g if coup is not None else np.zeros((0, 1))
    plb, pub = _tighten_predicate_box(lb, ub, G, h)
    V = np.hstack([np.zeros((n, 1)), np.eye(n)])            # x = alpha
    C = np.asarray(G, dtype=np.float64).reshape(-1, n) if np.asarray(G).size \
        else np.zeros((0, n))
    d = np.asarray(h, dtype=np.float64).reshape(-1, 1) if np.asarray(h).size \
        else np.zeros((0, 1))
    return Star(V, C, d, plb.reshape(-1, 1), pub.reshape(-1, 1))


def _slice_star(star, r0, r1):
    """Sub-star over rows [r0:r1) of the basis — same predicate system."""
    return Star(star.V[r0:r1, :].copy(), star.C, star.d,
                star.predicate_lb, star.predicate_ub)


def _prefix_matches(Y, S):
    """True iff output star ``Y`` has at least ``S.nVar`` predicates and its
    first ``S.nVar`` predicate bounds equal the joint input star's. This is
    necessary for a shared input prefix but proves nothing on its own: a
    star rebuilt from bounds (``Star.from_bounds`` gives every box the same
    ``[-1, 1]`` predicates) can have the same bounds without sharing the
    input's variables."""
    n_in = S.nVar
    if Y.nVar < n_in or Y.predicate_lb is None or Y.predicate_ub is None:
        return False
    ylb = np.asarray(Y.predicate_lb, dtype=np.float64).ravel()[:n_in]
    yub = np.asarray(Y.predicate_ub, dtype=np.float64).ravel()[:n_in]
    return bool(np.allclose(ylb, S.predicate_lb.ravel())
                and np.allclose(yub, S.predicate_ub.ravel()))


def _own_parts(Y, n_in, shared):
    """Split output star ``Y`` into its input part and its own part.

    With ``shared`` the first ``n_in`` predicates of ``Y`` are the joint
    input predicates; otherwise every predicate of ``Y`` is its own and the
    input columns are zero. Returns ``(center, V_in, V_own, C_in, C_own, d,
    lb_own, ub_own)``."""
    k = n_in if shared else 0
    C = (np.asarray(Y.C, dtype=np.float64).reshape(-1, Y.nVar)
         if np.asarray(Y.C).size else np.zeros((0, Y.nVar)))
    d = (np.asarray(Y.d, dtype=np.float64).reshape(-1, 1)
         if np.asarray(Y.d).size else np.zeros((0, 1)))
    V = np.asarray(Y.V, dtype=np.float64)
    if shared:
        V_in, C_in = V[:, 1:1 + k], C[:, :k]
    else:
        V_in, C_in = np.zeros((Y.dim, n_in)), np.zeros((C.shape[0], n_in))
    lb = np.asarray(Y.predicate_lb, dtype=np.float64).reshape(-1, 1)
    ub = np.asarray(Y.predicate_ub, dtype=np.float64).reshape(-1, 1)
    return V[:, :1], V_in, V[:, 1 + k:], C_in, C[:, k:], d, lb[k:], ub[k:]


def _relational_join(Yf, Yg, S, shared_f, shared_g):
    """Join Y_f and Y_g into one star [Y_f; Y_g] over
    [in(n_in) | f_own | g_own].

    ``in`` are the joint input predicates, with the bounds and coupling C/d
    of the joint input star ``S``. A network flagged ``shared_*`` has its
    first ``n_in`` predicates identified with ``in``; the caller must know
    its reach kept them as an aligned prefix. Every other predicate of an
    output star is appended as that network's own."""
    n_in = S.nVar
    for Y, shared in ((Yf, shared_f), (Yg, shared_g)):
        if shared and Y.nVar < n_in:
            raise ValueError("output star has fewer predicates than the input")
    cf, Vf_in, Vf_own, Cf_in, Cf_own, df_d, flb, fub = \
        _own_parts(Yf, n_in, shared_f)
    cg, Vg_in, Vg_own, Cg_in, Cg_own, dg_d, glb, gub = \
        _own_parts(Yg, n_in, shared_g)
    a, b = Vf_own.shape[1], Vg_own.shape[1]
    df, dg = Yf.dim, Yg.dim

    # V: Y_f rows get [center | in | f_own | 0(b)]; Y_g rows get
    # [center | in | 0(a) | g_own]
    V = np.vstack([np.hstack([cf, Vf_in, Vf_own, np.zeros((df, b))]),
                   np.hstack([cg, Vg_in, np.zeros((dg, a)), Vg_own])])

    # C: the input coupling, then f's rows, then g's rows. A shared network
    # already carries the coupling rows; repeating them is harmless.
    CS = (np.asarray(S.C, dtype=np.float64).reshape(-1, n_in)
          if np.asarray(S.C).size else np.zeros((0, n_in)))
    dS = (np.asarray(S.d, dtype=np.float64).reshape(-1, 1)
          if np.asarray(S.d).size else np.zeros((0, 1)))
    C = np.vstack([
        np.hstack([CS, np.zeros((CS.shape[0], a + b))]),
        np.hstack([Cf_in, Cf_own, np.zeros((Cf_own.shape[0], b))]),
        np.hstack([Cg_in, np.zeros((Cg_own.shape[0], a)), Cg_own])])
    d = np.vstack([dS, df_d, dg_d])

    # predicate bounds: input (from S), then f_own, then g_own
    plb = np.vstack([S.predicate_lb.reshape(-1, 1), flb, glb])
    pub = np.vstack([S.predicate_ub.reshape(-1, 1), fub, gub])
    return Star(V, C, d, plb, pub)


def _as_single_star(reach_out, who):
    if isinstance(reach_out, list):
        if len(reach_out) != 1:
            raise NotImplementedError(
                f"relational reach expects one output star for {who}, got "
                f"{len(reach_out)} (exact-mode splitting not yet handled)")
        return reach_out[0]
    return reach_out


def relational_reach(model_f, model_g, spec, method='approx', **kwargs):
    """Joint output Star [Y_f; Y_g] over the coupled input."""
    S = build_joint_input_star(spec)
    nets = spec['networks']
    nf, ng = nets[0], nets[1]
    xf = _slice_star(S, nf['input_offset'],
                     nf['input_offset'] + nf['input_size'])
    xg = _slice_star(S, ng['input_offset'],
                     ng['input_offset'] + ng['input_size'])
    Yf = _as_single_star(
        NeuralNetwork(model_f).reach(
            xf, method=method, input_shape=(nf['input_size'],), **kwargs), 'f')
    Yg = _as_single_star(
        NeuralNetwork(model_g).reach(
            xg, method=method, input_shape=(ng['input_size'],), **kwargs), 'g')
    # Identify an output star's prefix with the input predicates only when
    # the network's kernels are known to keep it; equal bounds alone would
    # accept a star rebuilt from bounds.
    shared_f = network_keeps_input_predicates(model_f) \
        and _prefix_matches(Yf, S)
    shared_g = network_keeps_input_predicates(model_g) \
        and _prefix_matches(Yg, S)
    if not (shared_f and shared_g):
        logger.debug("relational reach: input predicates shared by f=%s, "
                     "g=%s", shared_f, shared_g)
    return _relational_join(Yf, Yg, S, shared_f, shared_g)


def verify_relational(model_f, model_g, spec, method='approx', **kwargs):
    """Sound relational verification: UNSAT (safe) / UNKNOWN. SAT comes
    from the falsifier lane."""
    joint = relational_reach(model_f, model_g, spec, method=method, **kwargs)
    return verify_specification([joint], spec['prop'])


# ---------------------------------------------------------------------------
# Falsification over the coupled input (counterexample search -> SAT)
# ---------------------------------------------------------------------------

def _is_unsafe(y, prop, margin=0.0):
    """Joint output y is in the unsafe region iff every property group is
    hit (AND across groups); a group is hit iff any of its HalfSpaces
    holds (OR within a group) — matching verify_specification. ``margin``
    requires the point to be strictly inside (G y <= g - margin), so a
    reported counterexample is a genuine violation rather than a boundary
    artifact (the parser lowers strict ``<`` to closed ``<=``)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    for grp in prop:
        Hg = grp['Hg']
        hss = Hg if isinstance(Hg, list) else [Hg]
        if not any(bool(np.all(hs.G @ y <= hs.g - margin)) for hs in hss):
            return False
    return True


def _sample_coupled(spec, plb, pub, rng):
    """Draw a joint input satisfying box + coupling. X_f is sampled from
    its box; each later coord is tightened by every coupling row that
    references it given the already-set coords (handles the
    equality/inequality couplings in the acasxu relational specs)."""
    n = plb.size
    nf = spec['networks'][0]['input_size']
    coup = spec.get('input_coupling')
    x = np.zeros(n)
    x[:nf] = rng.uniform(plb[:nf], pub[:nf])
    G = (np.asarray(coup.G, dtype=np.float64).reshape(-1, n)
         if coup is not None and np.asarray(coup.G).size else np.zeros((0, n)))
    hvec = (np.asarray(coup.g, dtype=np.float64).flatten()
            if coup is not None and np.asarray(coup.g).size else np.zeros(0))
    for i in range(nf, n):
        lo, hi = plb[i], pub[i]
        for row, rhs in zip(G, hvec):
            if row[i] == 0.0:
                continue
            others = row.copy(); others[i] = 0.0
            bound = (rhs - others @ x) / row[i]
            if row[i] > 0.0:
                hi = min(hi, bound)
            else:
                lo = max(lo, bound)
        x[i] = rng.uniform(lo, hi) if hi > lo else lo
    return x


def falsify_relational(model_f, model_g, spec, n_samples=500, seed=42,
                       margin=1e-6):
    """Search for a coupled input whose joint output is strictly unsafe
    (by ``margin``). Returns (x_joint, y_joint) on success, else None."""
    import torch
    S = build_joint_input_star(spec)
    plb = S.predicate_lb.flatten()
    pub = S.predicate_ub.flatten()
    nf = spec['networks'][0]['input_size']
    ng = spec['networks'][1]['input_size']
    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        x = _sample_coupled(spec, plb, pub, rng)
        with torch.no_grad():
            yf = model_f(torch.tensor(x[:nf], dtype=torch.float32)
                         .reshape(1, nf)).numpy().flatten()
            yg = model_g(torch.tensor(x[nf:nf + ng], dtype=torch.float32)
                         .reshape(1, ng)).numpy().flatten()
        y = np.concatenate([yf, yg])
        if _is_unsafe(y, spec['prop'], margin=margin):
            return x, y
    return None


def solve_relational(model_f, model_g, spec, method='approx',
                     n_rand=500, seed=42, **kwargs):
    """Full relational verdict: falsify (SAT) then sound reach
    (UNSAT/UNKNOWN). Returns (verdict, counterexample_or_None)."""
    cex = falsify_relational(model_f, model_g, spec,
                             n_samples=n_rand, seed=seed)
    if cex is not None:
        return 'sat', cex
    res = verify_relational(model_f, model_g, spec, method=method, **kwargs)
    return ('unsat' if res.verdict == 'UNSAT' else 'unknown'), None

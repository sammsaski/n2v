"""Sound verification of ``format='nonlinear'`` VNNLIB 2.0 specifications.

VNN-COMP 2026 introduced specs outside the linear half-space fragment
(e.g. ``adaptive_cruise_control_non_linear_2026``), whose output property
contains products of variables (``Pow``/``Mul``). :func:`load_vnnlib` parses
those to ``format='nonlinear'`` carrying the resolved assertion ASTs (see
:func:`n2v.utils.vnnlib2._load_nonlinear`) instead of the linear ``'pairs'``
structure the reach->verify pipeline normally consumes. This module supplies
the missing verify path:

  * :func:`verify_nonlinear_reach` — sound UNSAT/UNKNOWN over a computed
    reach set. We build an affine model ``[x; y] = c + V @ alpha`` and
    evaluate the assertion conjunction with three-valued interval/affine
    arithmetic. If the conjunction (the *unsafe* region, per the SAT/UNSAT
    convention) is provably FALSE over the predicate box, no reachable point
    violates the spec -> ``UNSAT`` (safe). Otherwise ``UNKNOWN``. By default
    ``x`` and ``y`` get disjoint predicate variables (``x`` over the input
    box, ``y`` over the output star), which is sound for any reach set. The
    caller may assert that the input predicates are an aligned prefix of every
    output star (true for the append-only kernels, see
    :func:`network_keeps_input_predicates`); ``x`` and ``y`` then share those
    predicates, which keeps their correlation.

  * :func:`falsify_nonlinear` — concrete counterexample search: sample inputs
    in the box, forward through a caller-supplied backend (the runner passes
    onnxruntime, the backend VNN-COMP 2026 grades against), and test
    membership with :func:`n2v.utils.vnnlib2.evaluate_nonlinear`.

Soundness rests on two over-approximations, both in the safe direction:
dropping the predicate constraints ``C alpha <= d`` (enlarges the region)
and interval arithmetic on products (encloses the true range). A three-valued
FALSE therefore means "provably false for every alpha in the predicate box",
which is exactly what certifies the unsafe region empty. Sharing predicates
between ``x`` and ``y`` adds a third premise: every true pair ``(x, f(x))``
must be reachable with the input's own predicate values in the shared prefix.
Equal predicate bounds do not establish that (a star rebuilt from bounds,
e.g. by ``Star.from_bounds``, has ``[-1, 1]`` predicates whatever it came
from), so the shared model is used only when the caller asserts it.
"""

import logging

import numpy as np

from n2v.sets import Star
# ``_COMPARE`` shared from the parser so a new VNNLIB 2.0 relational operator
# only has to be added in one place.
from n2v.utils.vnnlib2 import _COMPARE, evaluate_nonlinear

logger = logging.getLogger(__name__)

# Outward soundness margin for the three-valued comparisons. ``_interval``
# accumulates in round-to-nearest with no directed rounding, so the computed
# ``[lo, hi]`` can sit a few ulp INSIDE the true range. We widen the
# zero-thresholds in ``_eval_compare`` by ``_ROUND_EPS * (|lo| + |hi| + 1)`` so
# a rounding error can never flip a conjunct to a wrong provable True/False (a
# wrong *False* would yield an unsound UNSAT). The value is far above the real
# accumulation error (~nVar * 2e-16 * magnitude) yet far below realistic spec
# decision margins, so it costs no completeness on the benchmarks of interest.
_ROUND_EPS = 1e-9


class _Aff:
    """An expression value tracked as an affine form over predicate
    variables ``alpha`` with a sound interval enclosure.

    ``lin`` is the linear coefficient vector (length ``nVar``) when the
    value is affine in ``alpha``; ``None`` once a nonlinear op (a
    variable*variable product) collapses it to an interval only. ``lo``/
    ``hi`` always bound the value over the predicate box.
    """

    __slots__ = ("c", "lin", "lo", "hi", "plb", "pub")

    def __init__(self, c, lin, lo, hi, plb, pub):
        self.c = c
        self.lin = lin
        self.lo = lo
        self.hi = hi
        self.plb = plb
        self.pub = pub

    @property
    def is_const(self):
        return self.lin is not None and not np.any(self.lin)

    @staticmethod
    def _interval(c, lin, plb, pub):
        """Tight per-coordinate interval of ``c + lin . alpha`` over the
        predicate box. Zero-coefficient coordinates never touch their
        (possibly infinite) bound, so an unbounded-but-unused predicate
        does not poison the enclosure."""
        lo = hi = float(c)
        for i in np.nonzero(lin)[0]:
            t1, t2 = lin[i] * plb[i], lin[i] * pub[i]
            lo += min(t1, t2)
            hi += max(t1, t2)
        return lo, hi

    @classmethod
    def const(cls, value, plb, pub):
        v = float(value)
        return cls(v, np.zeros(len(plb)), v, v, plb, pub)

    @classmethod
    def affine(cls, c, lin, plb, pub):
        lin = np.asarray(lin, dtype=np.float64)
        lo, hi = cls._interval(c, lin, plb, pub)
        return cls(float(c), lin, lo, hi, plb, pub)

    @classmethod
    def interval(cls, lo, hi, plb, pub):
        """A value known only by its interval (post-nonlinear-op)."""
        return cls(0.0, None, lo, hi, plb, pub)


def _ival_mul(a, b):
    ps = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi)
    if any(np.isnan(p) for p in ps):
        # 0 * inf -> NaN in IEEE; treat any NaN product as sound-unbounded.
        return -np.inf, np.inf
    return min(ps), max(ps)


def _add(a, b):
    # When both are affine, recompute the interval from the *combined*
    # coefficient vector over the predicate box. Summing the two endpoint
    # intervals would double-count shared predicates and lose the x<->y
    # correlation that makes the joint model worthwhile.
    if a.lin is not None and b.lin is not None:
        return _Aff.affine(a.c + b.c, a.lin + b.lin, a.plb, a.pub)
    return _Aff.interval(a.lo + b.lo, a.hi + b.hi, a.plb, a.pub)


def _neg(a):
    lin = None if a.lin is None else -a.lin
    return _Aff(-a.c, lin, -a.hi, -a.lo, a.plb, a.pub)


def _mul(a, b):
    # Scaling by a constant preserves the affine form (and its tight box).
    if a.is_const or b.is_const:
        k, v = (a.c, b) if a.is_const else (b.c, a)
        if k == 0.0:
            # Exact algebraic zero, regardless of v -- avoids 0 * inf = NaN
            # when v is an unbounded interval (e.g. from a div-by-zero).
            return _Aff.const(0.0, v.plb, v.pub)
        if v.lin is not None:
            return _Aff.affine(k * v.c, k * v.lin, v.plb, v.pub)
        lo, hi = (k * v.lo, k * v.hi) if k >= 0 else (k * v.hi, k * v.lo)
        return _Aff.interval(lo, hi, v.plb, v.pub)
    lo, hi = _ival_mul(a, b)
    return _Aff.interval(lo, hi, a.plb, a.pub)


def _div(a, b):
    if b.is_const:
        if b.c == 0.0:
            # Division by a constant zero: degrade to an unbounded interval
            # (-> the enclosing comparison becomes UNKNOWN) rather than raising,
            # which would abort the whole reach verdict and concede a spec that
            # other conjuncts might prove UNSAT.
            return _Aff.interval(-np.inf, np.inf, a.plb, a.pub)
        return _mul(a, _Aff.const(1.0 / b.c, a.plb, a.pub))
    if b.lo <= 0.0 <= b.hi:
        # Denominator straddles zero: unbounded. Sound but useless interval.
        return _Aff.interval(-np.inf, np.inf, a.plb, a.pub)
    invs = (1.0 / b.lo, 1.0 / b.hi)
    return _mul(a, _Aff.interval(min(invs), max(invs), a.plb, a.pub))


def _eval_arith(node, var_aff, plb, pub):
    op = node[0]
    if op == "const":
        return _Aff.const(node[1], plb, pub)
    if op == "var":
        return var_aff(node[1], node[2])
    args = [_eval_arith(c, var_aff, plb, pub) for c in node[1:]]
    if op == "+":
        out = args[0]
        for a in args[1:]:
            out = _add(out, a)
        return out
    if op == "-":
        if len(args) == 1:
            return _neg(args[0])
        out = args[0]
        for a in args[1:]:
            out = _add(out, _neg(a))
        return out
    if op == "*":
        out = args[0]
        for a in args[1:]:
            out = _mul(out, a)
        return out
    if op == "/":
        return _div(args[0], args[1])
    raise ValueError(f"unsupported arithmetic op in nonlinear spec: {op!r}")


def _eval_compare(node, var_aff, plb, pub):
    """Three-valued evaluation of a comparison: True/False if the relation
    holds/fails for every ``alpha`` in the predicate box, else ``None``.

    A magnitude-scaled outward margin (``tol``) guards every zero-threshold so
    floating-point rounding in the interval accumulation cannot flip a conjunct
    to a wrong provable ``False`` (which would yield an unsound UNSAT). The
    ``False`` branches all require the interval to clear ``tol``; ties go to
    ``None``. ``==``/``!=`` never return ``False`` for "provably equal" -- that
    needs an *exact* zero-width interval, which floating point cannot certify."""
    op = node[0]
    d = _add(_eval_arith(node[1], var_aff, plb, pub),
             _neg(_eval_arith(node[2], var_aff, plb, pub)))
    lo, hi = d.lo, d.hi
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None  # unbounded / NaN term -> cannot decide (sound)
    tol = _ROUND_EPS * (abs(lo) + abs(hi) + 1.0)
    if op == "<=":
        return True if hi <= -tol else (False if lo > tol else None)
    if op == "<":
        return True if hi <= -tol else (False if lo >= tol else None)
    if op == ">=":
        return True if lo >= tol else (False if hi < -tol else None)
    if op == ">":
        return True if lo >= tol else (False if hi <= -tol else None)
    if op == "==":
        # Provably false iff the interval excludes 0 by the margin; never
        # provably true (would require certifying an exact zero).
        return False if (lo > tol or hi < -tol) else None
    if op == "!=":
        # Provably true iff the interval excludes 0; never provably false.
        return True if (lo > tol or hi < -tol) else None
    raise ValueError(f"unsupported comparison in nonlinear spec: {op!r}")


def _eval_bool(node, var_aff, plb, pub):
    op = node[0]
    if op == "and":
        out = True
        for c in node[1:]:
            v = _eval_bool(c, var_aff, plb, pub)
            if v is False:
                return False
            if v is None:
                out = None
        return out
    if op == "or":
        out = False
        for c in node[1:]:
            v = _eval_bool(c, var_aff, plb, pub)
            if v is True:
                return True
            if v is None:
                out = None
        return out
    if op in _COMPARE:
        return _eval_compare(node, var_aff, plb, pub)
    raise ValueError(f"unsupported boolean node in nonlinear spec: {op!r}")


def _input_affine(input_set):
    """Recover ``x = center + basis @ alpha_in`` from a flat box-derived input
    ``Star``. Returns ``(center, basis, n_in, n_pred)`` -- ``center`` is the
    per-dim midpoint (length ``n_in``), ``basis`` is ``(n_in, n_pred)``, and
    ``n_pred`` is the number of input predicates -- or ``None`` if the set is
    not a flat per-coordinate box (e.g. an ImageStar, or a rotated/coupled
    star).

    ``n_pred`` may be SMALLER than ``n_in`` when some input dims are pinned
    (``lb == ub``): ``Box.to_zono`` drops the zero-width generator, so a pinned
    dim contributes an all-zero basis row (``x_i == center_i``) and no
    predicate column. The canonical-box guard (every generator column scales
    exactly one coordinate) rejects any input whose predicates are not the bare
    input coordinates, so reading the output star predicate-wise stays sound."""
    if not isinstance(input_set, Star) or input_set.V.size == 0:
        return None
    V = np.asarray(input_set.V, dtype=np.float64)
    n_in = V.shape[0]
    n_pred = V.shape[1] - 1
    if n_pred > n_in:
        return None  # more predicates than dims -> not a plain coordinate box
    basis = V[:, 1:]
    if basis.size and not np.all((basis != 0).sum(axis=0) == 1):
        return None  # a generator couples >1 coordinate -> not a box
    return V[:, 0], basis, n_in, n_pred


def network_keeps_input_predicates(network):
    """True iff a flat-``Star`` ``'approx'``/``'exact'`` reach of ``network``
    keeps the input predicates as an aligned prefix of every output star, so
    that :func:`verify_nonlinear_reach` may share them between ``x`` and ``y``
    (``input_predicates_shared=True``).

    Decided from the traced graph: every node must be the input, the output,
    or a call to an ``nn.Linear``, ``nn.ReLU`` or ``nn.Identity`` module. Their
    star kernels only rewrite the basis ``V`` (affine map), add rows to
    ``C``/``d`` (exact ReLU split), append one predicate per unstable neuron
    (ReLU relaxation) or return the set unchanged. Any other node (a function
    call, a constant, another module type) returns False: a kernel such as
    softmax re-encodes its output with ``Star.fresh_from_bounds`` and mints
    fresh predicates in the prefix. ``network`` is an ``n2v.nn.NeuralNetwork`` or a
    ``torch.nn.Module``; an untraceable model returns False."""
    import torch.fx as fx
    import torch.nn as nn
    from n2v.nn import NeuralNetwork

    # Unwrap only a real NeuralNetwork. A plain torch module may have its own
    # submodule called ``model``; vetting just that submodule would skip the
    # rest of its forward (e.g. a softmax head after it).
    model = network.model if isinstance(network, NeuralNetwork) else network
    if not isinstance(model, nn.Module):
        return False
    if not isinstance(model, fx.GraphModule):
        try:
            model = fx.symbolic_trace(model)
        except Exception:  # noqa: BLE001
            return False
    modules = dict(model.named_modules())
    for node in model.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op != "call_module" or not isinstance(
                modules.get(node.target), (nn.Linear, nn.ReLU, nn.Identity)):
            return False
    return True


def verify_nonlinear_reach(reach_sets, input_set, result, *,
                           input_predicates_shared=False):
    """Sound verdict for a ``format='nonlinear'`` spec over a reach set.

    Returns ``'UNSAT'`` (provably safe) when every output ``Star`` is
    provably free of any point satisfying the assertion conjunction (the
    unsafe region), else ``'UNKNOWN'``. Never returns ``'SAT'`` — concrete
    violations come from :func:`falsify_nonlinear`.

    By default the input ``x`` and the output ``y`` are modelled over disjoint
    predicate variables: ``x`` over the input set's predicate box and ``y``
    over the output star's. Every true pair ``(x, f(x))`` lies in that product,
    so the verdict is sound for any reach set, but a conjunct that needs the
    ``x``/``y`` correlation (e.g. ``y - x >= 0.5`` with ``y == x``) is only
    decided through the output star's own bounds.

    ``input_predicates_shared=True`` is the caller's guarantee that the input
    set's predicates are the first ``n_pred`` predicates of every output star
    (the reach sets came from an ``'approx'``/``'exact'`` reach of this input
    set through append-only kernels; see
    :func:`network_keeps_input_predicates`). ``x`` is then read from that
    shared prefix. An output star whose prefix bounds differ from the input's
    falls back to disjoint variables.
    """
    assertions = result["assertions"]
    aff = _input_affine(input_set)
    if aff is None:
        logger.debug("nonlinear verify: input set is not a flat box Star -> unknown")
        return "UNKNOWN"
    if input_set.predicate_lb is None or input_set.predicate_ub is None:
        logger.debug("nonlinear verify: input set has no predicate box -> unknown")
        return "UNKNOWN"
    in_center, in_basis, n_in, n_pred = aff
    in_plb = np.asarray(input_set.predicate_lb, dtype=np.float64).ravel()
    in_pub = np.asarray(input_set.predicate_ub, dtype=np.float64).ravel()

    if not reach_sets:
        # An empty reach list is the sound kernels' representation of an empty
        # (infeasible) reachable region -> nothing can violate -> UNSAT. Sound
        # because n2v sound kernels return [] only for a provably-empty set; a
        # failure raises (and is caught by the caller -> UNKNOWN), never [].
        return "UNSAT"

    for out in reach_sets:
        if not isinstance(out, Star) or out.predicate_lb is None \
                or out.predicate_ub is None:
            return "UNKNOWN"
        nVar = out.nVar
        out_plb = np.asarray(out.predicate_lb, dtype=np.float64).ravel()
        out_pub = np.asarray(out.predicate_ub, dtype=np.float64).ravel()
        out_V = out.V

        # Sharing the input predicates with this star needs the caller's
        # guarantee: equal bounds alone prove nothing, since a star rebuilt
        # from bounds (e.g. by Star.from_bounds) has [-1, 1] predicates
        # whatever it came from, and reading them as the input would invent an
        # x<->y correlation (a false UNSAT). Equal prefix bounds stay a
        # necessary condition on top of the guarantee.
        shared = input_predicates_shared and nVar >= n_pred and (
            np.allclose(out_plb[:n_pred], in_plb)
            and np.allclose(out_pub[:n_pred], in_pub))
        if shared:
            plb, pub = out_plb, out_pub
            in_off = 0
        else:
            if input_predicates_shared:
                logger.debug("nonlinear verify: input-prefix mismatch -> "
                             "disjoint input predicates")
            # x over its own predicate columns appended after the star's.
            plb = np.concatenate([out_plb, in_plb])
            pub = np.concatenate([out_pub, in_pub])
            in_off = nVar

        # Coefficient rows over the combined predicate vector are built only
        # for the variables the spec reads, once per star (a spec reads each
        # input bound twice), so no padded (dim, nVar + n_pred) basis is
        # materialised. Sharing the cached value is safe because no operation
        # mutates an _Aff in place. Bind every per-star value as a default arg
        # so the closure cannot pick up a later iteration's state (the closure
        # is only called synchronously today, but the defensive binding keeps
        # it correct if that changes).
        star_cache = {}

        def var_aff(kind, idx, _V=out_V, _nv=nVar, _off=in_off,
                    _plb=plb, _pub=pub, _cache=star_cache):
            aff_v = _cache.get((kind, idx))
            if aff_v is None:
                lin = np.zeros(len(_plb), dtype=np.float64)
                if kind == "in":
                    lin[_off:_off + n_pred] = in_basis[idx]
                    aff_v = _Aff.affine(in_center[idx], lin, _plb, _pub)
                else:
                    lin[:_nv] = _V[idx, 1:]
                    aff_v = _Aff.affine(float(_V[idx, 0]), lin, _plb, _pub)
                _cache[(kind, idx)] = aff_v
            return aff_v

        # The unsafe region is the conjunction of all assertions. Provably
        # false over the predicate box => this star cannot witness it.
        verdict = True
        for a in assertions:
            v = _eval_bool(a, var_aff, plb, pub)
            if v is False:
                verdict = False
                break
            if v is None:
                verdict = None
        if verdict is not False:
            return "UNKNOWN"  # this star may contain a violation

    return "UNSAT"


def falsify_nonlinear(model, lb, ub, result, input_shape, *,
                      n_samples=2000, seed=42, forward=None):
    """Sample inputs in ``[lb, ub]`` and return the first ``x`` (flat
    float64 vector) whose forward witnesses the nonlinear spec (per
    :func:`evaluate_nonlinear`), or ``None``.

    ``forward`` is a BATCHED callable ``(N, d) -> (N, out)`` used as the
    membership oracle. Pass the SAME backend the verdict is graded against
    (VNN-COMP 2026 replays the witness through onnxruntime), so the search does
    not miss a witness the grader would accept on a spec boundary where the
    onnx2torch float32 forward and onnxruntime disagree. When ``forward`` is
    ``None`` it defaults to the torch ``model`` in float32 -- fast, but only a
    proposer that the caller must re-confirm on the grading backend. A
    non-finite (NaN/Inf) forward output is never accepted as a witness."""
    lb = np.asarray(lb, dtype=np.float64).ravel()
    ub = np.asarray(ub, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)

    if forward is None:
        import torch

        def forward(X, _m=model, _s=input_shape):
            X = np.asarray(X, np.float32)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            xt = torch.from_numpy(X.reshape((X.shape[0],) + tuple(_s)))
            with torch.no_grad():
                return _m(xt).detach().cpu().numpy().reshape(X.shape[0], -1)

    # Box corners first (cheap, often decisive), then uniform interior draws.
    corners = []
    if lb.size <= 12:
        from itertools import product
        for bits in product((0, 1), repeat=lb.size):
            corners.append(np.where(np.asarray(bits), ub, lb))
    interior = lb + (ub - lb) * rng.random((n_samples, lb.size))
    candidates = np.vstack(corners + [interior]) if corners else interior

    # One batched forward over all candidates (the backend batches in a single
    # session call when the model's batch dim is dynamic).
    Y = np.asarray(forward(candidates), dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(candidates.shape[0], -1)
    for i in range(candidates.shape[0]):
        y = Y[i]
        if np.all(np.isfinite(y)) and evaluate_nonlinear(result, candidates[i], y):
            return candidates[i]
    return None

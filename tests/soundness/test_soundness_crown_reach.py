"""Soundness of the LP-free CROWN bound engine over a flat operation list
(``n2v.nn.crown_reach``), the n2v port of NNV's ``ViTCrown`` attention
verifier, and of the verdict route that reaches it
(``n2v.nn.vit_crown.verify_halfspace_group_safe``).

The promise: for every input of the box, the true value of every operation of
the list lies inside the bounds the engine returns. It holds for every
operation kind the list can hold (affine, relu, add, concat, bmatmul in both
modes, softmax, exp, reciprocal, eprod), for both passes (``forward_ibp``,
``backward_crown``), for every value of the relu slopes and attention plane
weights in [0, 1] (``optimize_alpha`` may change how tight a bound is, never
whether it holds), with and without ``refine_bounds`` and ``refine_scores``,
and from every ``start_op``. A verdict 'safe' is given only when no input of
the box reaches the unsafe region ``G y <= g``.

The oracles never use ``eval_ops``; ``eval_ops`` itself is pinned against them.

- Affine, relu, add and concat lists (the ``pl-*`` cases): the exact range of
  every coordinate of every op and the exact minimum of every linear function
  of an op, in rationals. The box is cut into the linear regions of the
  network by the sign of every relu input, and the value of every op is
  evaluated exactly at every vertex of every region. Data are dyadic, so
  interval arithmetic is exact in float64 and its bounds are compared with no
  slack.
- Products of two affine maps of disjoint input coordinates (eprod, bmatmul):
  every op is multilinear in the input, so its extremes are at the corners of
  the box, which are all evaluated.
- Attention blocks and softmax: an independent float64 forward pass written
  here in matrix form (``_mat``/``_vec`` state the column-major layout of the
  module docstring), on every corner, the centre and seeded samples of the box.
  Softmax is increasing in its own logit and decreasing in the others, so the
  per-entry extremes over a logit box are at corners.

Slack: none against exact references; ``_pad`` (1e-12 times one plus the
magnitude) against float64 values. The rounding to nearest of the engine is
the registered defect VERDICT-1; its second code path, the interval pass, has
its own marked cases here with no slack.
"""
import functools
from fractions import Fraction
from itertools import product

import mpmath
import numpy as np
import pytest

from n2v.nn import crown_reach as cr
from n2v.nn import vit_crown
from tests._oracles.rational import dot, frac, matrix, vector, vertices
from tests.known_bugs import known_bug


# --------------------------------------------------------------------------- #
#  shared helpers                                                               #
# --------------------------------------------------------------------------- #
def _pad(v):
    """Slack against a float64 reference value (PROTOCOL 3.4)."""
    return 1e-12 * (1.0 + abs(float(v)))


def _corners(lb, ub):
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    bits = np.array(list(product((0.0, 1.0), repeat=lb.size)))
    return lb + bits * (ub - lb)


def _points(lb, ub, seed, n_samples):
    """Every corner, the centre and seeded samples of the box."""
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    rng = np.random.default_rng(seed)
    return np.vstack([_corners(lb, ub), 0.5 * (lb + ub)[None, :],
                      lb + (ub - lb) * rng.random((n_samples, lb.size))])


def _vec(M):
    """Column-major flatten of a batch of matrices: (P, r, c) -> (P, r*c)."""
    return M.transpose(0, 2, 1).reshape(M.shape[0], -1)


def _mat(v, r, c):
    """The inverse of ``_vec``: (P, r*c) -> (P, r, c)."""
    return v.reshape(v.shape[0], c, r).transpose(0, 2, 1)


def _softmax_rows(S):
    E = np.exp(S - S.max(axis=-1, keepdims=True))
    return E / E.sum(axis=-1, keepdims=True)


def _affine(src, W, b, mat=None):
    W = np.asarray(W, float)
    return dict(type="affine", **{"in": src}, W=W,
                b=np.asarray(b, float).reshape(-1), dim=W.shape[0], mat=mat)


def _token_affine(src, W, b, N):
    """The per-token map T -> T W^T + b on the column-major token matrix."""
    W = np.asarray(W, float)
    return _affine(src, np.kron(W, np.eye(N)), np.repeat(b, N), (N, W.shape[0]))


def _head_slice(src, h, D, N, width):
    W = np.zeros((D * N, N * width))
    W[np.arange(D * N), h * D * N + np.arange(D * N)] = 1.0
    return _affine(src, W, np.zeros(D * N), (N, D))


def _violations(claimed, true, side, slack):
    """Entries where a claimed bound does not hold: 'lower' needs claimed <=
    true, 'upper' needs claimed >= true. ``true`` holds Fractions (exact) or
    floats; NaN never holds."""
    claimed = np.asarray(claimed, dtype=float).reshape(-1)
    if claimed.size != len(true):
        return [("size", claimed.size, len(true))]
    bad = []
    for j, (c, t) in enumerate(zip(claimed, true)):
        s = _pad(t) if slack else 0.0
        if np.isnan(c):
            ok = False
        elif np.isinf(c):
            ok = (c < 0) if side == "lower" else (c > 0)
        elif isinstance(t, Fraction):
            ok = (Fraction(c) <= t + Fraction(s)) if side == "lower" else \
                 (Fraction(c) >= t - Fraction(s))
        else:
            ok = (c <= t + s) if side == "lower" else (c >= t - s)
        if not ok:
            bad.append((j, float(c), float(t)))
    return bad


# --------------------------------------------------------------------------- #
#  cases: an op list, a box and the true values on it                           #
# --------------------------------------------------------------------------- #
class _Case:
    """``ops`` with their ``names``, the box, and the truth: ``exact`` holds
    one list of exact op values per vertex of every linear region (rational
    cases), ``values`` the float64 values of every op at ``points``."""

    def __init__(self, ops, names, lb, ub, exact=None, values=None):
        self.ops, self.names = ops, names
        self.lb = np.asarray(lb, float)
        self.ub = np.asarray(ub, float)
        self.exact, self.values = exact, values

    @functools.cache
    def lo(self, k):
        if self.exact is not None:
            return [min(col) for col in zip(*(pt[k] for pt in self.exact))]
        return list(self.values[k].min(axis=0))

    @functools.cache
    def hi(self, k):
        if self.exact is not None:
            return [max(col) for col in zip(*(pt[k] for pt in self.exact))]
        return list(self.values[k].max(axis=0))

    def min_of(self, k, C):
        """True minimum over the box of every row of ``C @ value[k]``."""
        if self.exact is not None:
            rows = matrix(C)
            return [min(dot(r, pt[k]) for pt in self.exact) for r in rows]
        return list((self.values[k] @ np.asarray(C, float).T).min(axis=0))

    @property
    def out(self):
        return len(self.ops) - 1

    def assert_box(self, k, lo, hi, what, slack=True):
        bad = (_violations(lo, self.lo(k), "lower", slack)
               + _violations(hi, self.hi(k), "upper", slack))
        assert not bad, (f"{what}: the box of op {k} ({self.names[k]}) misses "
                         f"true values (entry, claimed, true): {bad[:4]}")

    def assert_lower(self, k, C, lbnd, what, slack=True):
        bad = _violations(lbnd, self.min_of(k, C), "lower", slack)
        assert not bad, (f"{what}: the lower bound of C @ op {k} "
                         f"({self.names[k]}) exceeds a true value "
                         f"(row, claimed, true min): {bad[:4]}")


# ---- affine, relu, add and concat: the exact image in rationals ------------ #
_PL = dict(
    W1=[[1, -0.5], [-0.75, 1.25], [0.5, 0.5], [-1.5, -0.25]],
    b1=[0.25, -0.5, -0.125, 0.75],
    P=[[0.5, -1], [1.5, 0.25]], c=[-0.25, 0.5],
    W2=[[0.5, -1, 0.25, 0.75, -0.5, 1], [-0.75, 0.5, 1, -0.25, 1.25, -0.5],
        [1, 0.25, -0.5, 0.5, 0.25, -1.25]],
    b2=[0.125, -0.25, 0.5],
    S=[[1, 0.5], [-0.5, 0.75], [0.25, -1]], s=[0, 0.25, -0.5],
    W3=[[1.5, -0.75, 0.5], [-1, 1.25, 0.75]], b3=[-0.125, 0.375])

# x -> h1 -> relu -> concat with an input projection p -> h2, plus a skip
# projection of x (add) -> relu -> y. Every weight is a multiple of 1/8.
_PL_SPEC = [
    ("x", "input", None, None, None),
    ("h1", "affine", 0, "W1", "b1"),
    ("r1", "relu", 1, None, None),
    ("p", "affine", 0, "P", "c"),
    ("cat", "concat", [2, 3], None, None),
    ("h2", "affine", 4, "W2", "b2"),
    ("sk", "affine", 0, "S", "s"),
    ("a", "add", [5, 6], None, None),
    ("r2", "relu", 7, None, None),
    ("y", "affine", 8, "W3", "b3"),
]


def _pl_ops():
    ops, names = [], []
    for name, kind, src, W, b in _PL_SPEC:
        names.append(name)
        if kind == "input":
            ops.append(dict(type="input", **{"in": []}, dim=2, mat=None))
        elif kind == "affine":
            ops.append(_affine(src, _PL[W], _PL[b]))
        elif kind == "relu":
            ops.append(dict(type="relu", **{"in": src}, dim=ops[src]["dim"], mat=None))
        elif kind == "add":
            ops.append(dict(type="add", **{"in": src}, dim=ops[src[0]]["dim"], mat=None))
        else:
            ops.append(dict(type="concat", **{"in": src},
                            dim=sum(ops[j]["dim"] for j in src), mat=None))
    return ops, names


def _pl_forward(X):
    """The float64 value of every op of ``_PL_SPEC`` at each row of X."""
    v = [np.atleast_2d(np.asarray(X, float))]
    for name, kind, src, W, b in _PL_SPEC[1:]:
        if kind == "affine":
            v.append(v[src] @ np.asarray(_PL[W], float).T + np.asarray(_PL[b], float))
        elif kind == "relu":
            v.append(np.maximum(v[src], 0.0))
        elif kind == "add":
            v.append(v[src[0]] + v[src[1]])
        else:
            v.append(np.concatenate([v[j] for j in src], axis=1))
    return v


def _pl_region_vertices(lb, ub):
    """The exact value of every op at every vertex of every linear region of
    the box. Each op's value is an affine form ``M x + c`` on a region; a relu
    neuron splits the region by the sign of its input."""
    n = len(lb)
    eye = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    A0 = eye + [[-v for v in row] for row in eye]
    b0 = vector(ub) + [-v for v in vector(lb)]
    found = []

    def walk(k, forms, A, b):
        if k == len(_PL_SPEC):
            for x in vertices(A, b):
                found.append([[dot(row, x) + ci for row, ci in zip(M, c)]
                              for M, c in forms])
            return
        name, kind, src, W, bias = _PL_SPEC[k]
        if kind == "input":
            walk(k + 1, forms + [(eye, [Fraction(0)] * n)], A, b)
        elif kind == "affine":
            M, c = forms[src]
            Wf, bf = matrix(_PL[W]), vector(_PL[bias])
            cols = list(zip(*M))
            walk(k + 1, forms + [([[dot(w, col) for col in cols] for w in Wf],
                                  [dot(w, c) + bi for w, bi in zip(Wf, bf)])], A, b)
        elif kind == "add":
            (M0, c0), (M1, c1) = forms[src[0]], forms[src[1]]
            walk(k + 1, forms + [([[p + q for p, q in zip(r0, r1)]
                                   for r0, r1 in zip(M0, M1)],
                                  [p + q for p, q in zip(c0, c1)])], A, b)
        elif kind == "concat":
            walk(k + 1, forms + [([r for j in src for r in forms[j][0]],
                                  [v for j in src for v in forms[j][1]])], A, b)
        else:
            M, c = forms[src]

            def branch(i, rows, consts, A, b):
                if i == len(M):
                    walk(k + 1, forms + [(rows, consts)], A, b)
                    return
                row, off = M[i], c[i]
                A_on, b_on = A + [[-v for v in row]], b + [off]     # input >= 0
                if vertices(A_on, b_on):
                    branch(i + 1, rows + [row], consts + [off], A_on, b_on)
                A_off, b_off = A + [list(row)], b + [-off]          # input <= 0
                if vertices(A_off, b_off):
                    branch(i + 1, rows + [[Fraction(0)] * n], consts + [Fraction(0)],
                           A_off, b_off)
            branch(0, [], [], A, b)

    walk(0, [], A0, b0)
    return found


def _pl_case(lb, ub, stable=False):
    ops, names = _pl_ops()
    exact = _pl_region_vertices(lb, ub)
    case = _Case(ops, names, lb, ub, exact=exact)
    if stable:      # setup check: every relu input keeps one sign on the box
        for k in (1, 7):
            if any(lo < 0 < hi for lo, hi in zip(case.lo(k), case.hi(k))):
                raise RuntimeError(f"pl-stable: op {k} changes sign on the box")
    return case


# ---- attention blocks: an independent float64 forward pass ----------------- #
def _attention(seed, softmax_op, box_seed, width, N=3, E=4, H=2, D=2, F=4,
               n_out=3):
    """One pre-norm-free transformer block on N tokens of width E: Q, K, V per
    token, H heads of width D (scores Q K^T scaled, softmax as one op or as
    exp -> row sum -> reciprocal -> broadcast -> eprod, then A V), the heads
    concatenated, an output projection with a residual add, a relu feed-forward
    with a second residual add, and an affine readout."""
    rng = np.random.default_rng(seed)
    HD = H * D
    Wq, bq = rng.standard_normal((HD, E)), 0.3 * rng.standard_normal(HD)
    Wk, bk = rng.standard_normal((HD, E)), 0.3 * rng.standard_normal(HD)
    Wv, bv = rng.standard_normal((HD, E)), 0.3 * rng.standard_normal(HD)
    Wo, bo = 0.5 * rng.standard_normal((E, HD)), 0.3 * rng.standard_normal(E)
    W1, b1 = rng.standard_normal((F, E)), 0.3 * rng.standard_normal(F)
    W2, b2 = 0.5 * rng.standard_normal((E, F)), 0.3 * rng.standard_normal(E)
    Wout, bout = rng.standard_normal((n_out, N * E)), 0.3 * rng.standard_normal(n_out)
    scale = 1.0 / np.sqrt(D)
    Wsum = np.zeros((N, N * N))
    Wbc = np.zeros((N * N, N))
    for i in range(N):
        Wsum[i, np.arange(N) * N + i] = 1.0
        Wbc[np.arange(N) * N + i, i] = 1.0

    ops, names = [], []

    def push(name, op):
        ops.append(op)
        names.append(name)
        return len(ops) - 1

    x = push("x", dict(type="input", **{"in": []}, dim=N * E, mat=(N, E)))
    q = push("q", _token_affine(x, Wq, bq, N))
    k = push("k", _token_affine(x, Wk, bk, N))
    v = push("v", _token_affine(x, Wv, bv, N))
    heads = []
    for h in range(H):
        qh = push(f"q{h}", _head_slice(q, h, D, N, HD))
        kh = push(f"k{h}", _head_slice(k, h, D, N, HD))
        vh = push(f"v{h}", _head_slice(v, h, D, N, HD))
        s = push(f"s{h}", dict(type="bmatmul", **{"in": [qh, kh]}, mode="abt",
                               ra=N, ca=D, rb=N, cb=D, dim=N * N, mat=(N, N)))
        ss = push(f"ss{h}", _affine(s, scale * np.eye(N * N), np.zeros(N * N), (N, N)))
        if softmax_op:
            a = push(f"a{h}", dict(type="softmax", **{"in": ss}, dim=N * N, mat=(N, N)))
        else:
            e = push(f"e{h}", dict(type="exp", **{"in": ss}, dim=N * N, mat=(N, N)))
            t = push(f"t{h}", _affine(e, Wsum, np.zeros(N), (N, 1)))
            r = push(f"r{h}", dict(type="reciprocal", **{"in": t}, dim=N, mat=(N, 1)))
            rb = push(f"rb{h}", _affine(r, Wbc, np.zeros(N * N), (N, N)))
            a = push(f"a{h}", dict(type="eprod", **{"in": [e, rb]}, dim=N * N, mat=(N, N)))
        heads.append(push(f"o{h}", dict(type="bmatmul", **{"in": [a, vh]}, mode="ab",
                                        ra=N, ca=N, rb=N, cb=D, dim=N * D, mat=(N, D))))
    cat = push("cat", dict(type="concat", **{"in": heads}, dim=N * HD, mat=(N, HD)))
    ao = push("ao", _token_affine(cat, Wo, bo, N))
    res = push("res", dict(type="add", **{"in": [x, ao]}, dim=N * E, mat=(N, E)))
    f1 = push("f1", _token_affine(res, W1, b1, N))
    rl = push("relu", dict(type="relu", **{"in": f1}, dim=N * F, mat=(N, F)))
    f2 = push("f2", _token_affine(rl, W2, b2, N))
    res2 = push("res2", dict(type="add", **{"in": [res, f2]}, dim=N * E, mat=(N, E)))
    push("out", _affine(res2, Wout, bout))

    def forward(X):
        P = X.shape[0]
        T = _mat(X, N, E)
        Q, K, V = T @ Wq.T + bq, T @ Wk.T + bk, T @ Wv.T + bv
        val = {"x": X, "q": _vec(Q), "k": _vec(K), "v": _vec(V)}
        outs = []
        for h in range(H):
            cols = slice(h * D, (h + 1) * D)
            Qh, Kh, Vh = Q[:, :, cols], K[:, :, cols], V[:, :, cols]
            S = Qh @ Kh.transpose(0, 2, 1)
            A = _softmax_rows(scale * S)
            val.update({f"q{h}": _vec(Qh), f"k{h}": _vec(Kh), f"v{h}": _vec(Vh),
                        f"s{h}": _vec(S), f"ss{h}": _vec(scale * S), f"a{h}": _vec(A)})
            if not softmax_op:
                Ex = np.exp(scale * S)
                R = 1.0 / Ex.sum(axis=2)
                val.update({f"e{h}": _vec(Ex), f"t{h}": 1.0 / R, f"r{h}": R,
                            f"rb{h}": _vec(np.broadcast_to(R[:, :, None], (P, N, N)))})
            O = A @ Vh
            val[f"o{h}"] = _vec(O)
            outs.append(O)
        Ocat = np.concatenate(outs, axis=2)
        AO = Ocat @ Wo.T + bo
        RES = T + AO
        F1 = RES @ W1.T + b1
        RL = np.maximum(F1, 0.0)
        RES2 = RES + RL @ W2.T + b2
        val.update(cat=_vec(Ocat), ao=_vec(AO), res=_vec(RES), f1=_vec(F1),
                   relu=_vec(RL), f2=_vec(RL @ W2.T + b2), res2=_vec(RES2),
                   out=_vec(RES2) @ Wout.T + bout)
        return [val[nm] for nm in names]

    centre = np.random.default_rng(box_seed).uniform(-1.5, 1.5, N * E)
    lb, ub = centre - width / 2, centre + width / 2
    pts = centre[None, :] if width == 0 else _points(lb, ub, box_seed, 400)
    return _Case(ops, names, lb, ub, values=forward(pts)), forward


def _eprod_signed():
    """p = affine of x[:5], q = affine of x[5:], their entry-wise product and a
    mixed-sign readout. Operand intervals straddle zero (entries 0 to 3) or
    lie below it (q of entry 4); every op is multilinear in x, so the corners
    give the exact extremes. The data are not dyadic on purpose: the
    McCormick plane picked at the box midpoint is an exact tie decided by
    rounding, and with these intervals the second upper plane (entries 1 and
    4) and the second lower plane (entry 3) are picked."""
    Pd = np.array([1.1, -0.7, 1.3, 1 / 3, 0.3]); pc = np.array([0.3, -0.6, 0.1, 0.3, 0.1])
    Qd = np.array([-1.3, 0.7, 0.9, -1.3, 0.7]); qc = np.array([0.6, 0.2, -0.7, 0.1, -2.1])
    Wo = np.array([[1.0, -2.0, 0.5, 1.5, -1.0], [-1.5, 0.75, 1.25, -1.0, 2.0]])
    bo = np.array([0.25, -0.5])
    Sp = np.zeros((5, 10)); Sp[:, :5] = np.diag(Pd)
    Sq = np.zeros((5, 10)); Sq[:, 5:] = np.diag(Qd)
    ops = [dict(type="input", **{"in": []}, dim=10, mat=None),
           _affine(0, Sp, pc), _affine(0, Sq, qc),
           dict(type="eprod", **{"in": [1, 2]}, dim=5, mat=None),
           _affine(3, Wo, bo)]
    names = ["x", "p", "q", "pq", "out"]
    lb = np.array([-1.5, -2.0, -1.0, -1.5, -1.5, -1.25, -2.5, -0.5, -2.0, -1.25])
    ub = np.array([2.0, 1.25, 1.5, 2.0, 2.0, 2.25, 1.0, 2.5, 1.25, 1.0])

    def forward(X):
        p, q = X[:, :5] * Pd + pc, X[:, 5:] * Qd + qc
        return [X, p, q, p * q, (p * q) @ Wo.T + bo]

    return _Case(ops, names, lb, ub, values=forward(_points(lb, ub, 5, 200))), forward


def _softmax_dominant():
    """A 2 x 3 logit matrix (positive diagonal map of x plus offsets, one entry
    of row 0 far above the others) through a softmax op, a mixed-sign readout
    and a residual projection of x (so the output also depends on x outside
    the softmax). Softmax entry extremes over the box are at corners."""
    N, M = 2, 3
    d = np.array([1.0, 0.5, 2.0, 1.5, 0.75, 1.25])
    o = np.array([6.0, 0.5, 0.0, 0.2, -1.0, -0.3])
    Wf = np.array([[1.0, -2.0, 0.5, 1.5, -0.5, 0.25], [-1.0, 0.5, 2.0, -0.75, 1.0, -1.5],
                   [0.5, 0.5, -1.0, 1.0, -2.0, 0.75]])
    R = np.array([[0.5, -0.25, 0.0, 1.0, 0.0, -0.5], [0.0, 1.0, -0.5, 0.0, 0.25, 0.0],
                  [-0.75, 0.0, 0.5, 0.0, 0.0, 1.0]])
    ops = [dict(type="input", **{"in": []}, dim=6, mat=None),
           _affine(0, np.diag(d), o, (N, M)),
           dict(type="softmax", **{"in": 1}, dim=6, mat=(N, M)),
           _affine(2, Wf, np.zeros(3)),
           _affine(0, R, np.zeros(3)),
           dict(type="add", **{"in": [3, 4]}, dim=3, mat=None)]
    names = ["x", "z", "sm", "f", "rx", "out"]
    lb = np.array([-0.5, 0.25, -1.0, 0.5, -0.25, -1.5])
    ub = lb + 0.75

    def forward(X):
        Z = X * d + o
        A = _vec(_softmax_rows(_mat(Z, N, M)))
        return [X, Z, A, A @ Wf.T, X @ R.T, A @ Wf.T + X @ R.T]

    return _Case(ops, names, lb, ub, values=forward(_points(lb, ub, 6, 200))), forward


_CASES = {
    # affine/relu/add/concat, exact rational references (dyadic data)
    "pl-wide": lambda: (_pl_case([-2.5, -1.25], [1.75, 3.0]), _pl_forward),
    "pl-slope": lambda: (_pl_case([0.75, -1.75], [2.75, 0.25]), _pl_forward),
    "pl-stable": lambda: (_pl_case([0.96875, -0.28125], [1.03125, -0.21875],
                                   stable=True), _pl_forward),
    "pl-degenerate": lambda: (_pl_case([-1.5, 0.75], [2.25, 0.75]), _pl_forward),
    # attention, independent float64 forward pass
    "attn-narrow": lambda: _attention(11, False, 21, 0.05),
    "attn-wide": lambda: _attention(12, False, 22, 0.5),
    "attn-point": lambda: _attention(13, False, 23, 0.0),
    "attn-softmax-op": lambda: _attention(14, True, 24, 0.1),
    # products of signed operands; softmax op with a dominating entry
    "eprod-signed": _eprod_signed,
    "softmax-dominant": _softmax_dominant,
}
_ALL = list(_CASES)


@functools.cache
def _built(name):
    return _CASES[name]()


def _case(name):
    return _built(name)[0]


def _forward_of(name):
    return _built(name)[1]


def _through_softmax_only(ops, k):
    """Does op k depend on the input only through softmax ops? Then its
    backward pass reaches no input coefficient (registered as new here)."""
    @functools.cache
    def reaches(j):
        op = ops[j]
        if op["type"] == "input":
            return True
        if op["type"] == "softmax":
            return False
        srcs = op["in"] if isinstance(op["in"], (list, tuple)) else [op["in"]]
        return any(reaches(s) for s in srcs)
    return not reaches(k)


def _both_sides(dim):
    return np.vstack([np.eye(dim), -np.eye(dim)])


def _mixed_rows(dim, seed, n=3):
    """Mixed-sign functionals with dyadic entries."""
    return np.round(np.random.default_rng(seed).uniform(-2, 2, (n, dim)) * 8) / 8


# --------------------------------------------------------------------------- #
#  the concrete evaluator, pinned against the independent forward pass          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["pl-wide", "attn-narrow", "attn-softmax-op"])
def test_eval_ops_matches_an_independent_forward_pass(name):
    """``eval_ops`` gives the value of every op at a point. Catches a wrong
    layout of a bmatmul result (row-major instead of column-major), a softmax
    shifted per column instead of per row, a concat or add that reads the
    wrong input: every other test of the lane that uses ``eval_ops`` as its
    reference would agree with such a bug."""
    case = _case(name)
    X = _points(case.lb, case.ub, 7, 5)[-12:]
    ref = _forward_of(name)(X)
    for i, x in enumerate(X):
        _, vals = cr.eval_ops(case.ops, x)
        for k, v in enumerate(vals):
            np.testing.assert_allclose(
                v, ref[k][i], rtol=1e-12, atol=1e-12,
                err_msg=f"eval_ops: op {k} ({case.names[k]}) at point {i}")


@pytest.mark.parametrize("entry", ["eval_ops", "forward_ibp", "backward_crown"])
def test_an_unknown_op_type_is_refused(entry):
    """An op type the engine does not know must raise ValueError, never be
    skipped (a skipped op would leave its value, interval or coefficient
    unset and the bound of everything after it wrong)."""
    known = [dict(type="input", **{"in": []}, dim=2, mat=None),
             dict(type="relu", **{"in": 0}, dim=2, mat=None)]
    unknown = [known[0], dict(type="gelu", **{"in": 0}, dim=2, mat=None)]
    lb, ub = np.array([-1.0, 0.5]), np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="unknown op"):
        if entry == "eval_ops":
            cr.eval_ops(unknown, lb)
        elif entry == "forward_ibp":
            cr.forward_ibp(unknown, lb, ub)
        else:
            cl, cu = cr.forward_ibp(known, lb, ub)
            cr.backward_crown(unknown, lb, ub, cl, cu, np.eye(2))


# --------------------------------------------------------------------------- #
#  the two passes on every case                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _ALL)
def test_forward_ibp_holds_every_op_value(name):
    """Interval bounds of every op hold every true value. Rational cases are
    exact in float64 (dyadic data), so they are checked with no slack. Catches
    a wrong interval rule for any op kind: the upper bound of an affine op
    with a negative weight, of an add (second input), of a stably inactive
    relu, a reciprocal taken as increasing, a product of signed operands taken
    as a product of positive ones, a dropped radius term of the interval
    matrix product, a softmax entry bound evaluated away from its maximiser
    or minimiser, a concat upper bound built from lower bounds, a bmatmul box
    written in the wrong layout."""
    case = _case(name)
    cl, cu = cr.forward_ibp(case.ops, case.lb, case.ub)
    for k in range(len(case.ops)):
        case.assert_box(k, cl[k], cu[k], "forward_ibp", slack=case.exact is None)


@pytest.mark.parametrize("name", _ALL)
def test_backward_crown_bounds_the_output_from_both_sides(name):
    """The backward bound of ``C @ output`` is at most its true minimum, for
    C = +I, -I and mixed-sign rows, so every relaxation is used with both
    signs of its coefficient (relu upper chord and its intercept, exp and
    reciprocal tangent and chord, the four McCormick planes, the softmax
    box). Catches a wrong relaxation, a wrong coefficient routing (add,
    concat offset, fan-out to Q, K, V) and the concretisation at the box
    maximum; the zero-width and stable cases catch an additive error of any
    size, because there the bound equals the value."""
    case = _case(name)
    cl, cu = cr.forward_ibp(case.ops, case.lb, case.ub)
    dim = case.ops[-1]["dim"]
    C = np.vstack([_both_sides(dim), _mixed_rows(dim, 31)])
    lbnd, Lin, bL = cr.backward_crown(case.ops, case.lb, case.ub, cl, cu, C)
    case.assert_lower(case.out, C, lbnd, "backward_crown")


@pytest.mark.parametrize("name", _ALL)
def test_crown_bounds_hold_every_intermediate_value(name):
    """``crown_bounds`` of every op (a backward pass from ``start_op`` with +I
    for the lower and -I for the upper side) holds every true value of that
    op. Catches an upper pass run with +I, a start op that is ignored, and
    every relaxation error of the ops before it. Ops that reach the input
    only through a softmax op are left to the marked case below."""
    case = _case(name)
    cl, cu = cr.forward_ibp(case.ops, case.lb, case.ub)
    mmaps = cr.precompute_maps(case.ops, cl, cu)
    for k in range(1, len(case.ops)):
        if _through_softmax_only(case.ops, k):
            continue
        lo, hi = cr.crown_bounds(case.ops, case.lb, case.ub, cl, cu, k, mmaps)
        case.assert_box(k, lo, hi, "crown_bounds")


@pytest.mark.parametrize("name, iters, max_dim", [
    ("pl-wide", 2, np.inf), ("pl-slope", 1, np.inf), ("attn-narrow", 2, np.inf),
    ("attn-wide", 1, np.inf), ("attn-wide", 2, 4), ("eprod-signed", 1, np.inf)])
def test_refine_bounds_boxes_hold_every_op_value(name, iters, max_dim):
    """The refined boxes of every op hold every true value, and so does the
    backward bound computed with the refined boxes and the McCormick maps
    ``refine_bounds`` returns. Catches an override whose upper side is the
    CROWN lower bound (or whose lower side is the upper bound), and every
    relaxation error that reaches a refined box through ``crown_bounds``.
    ``max_dim=4`` refines only the narrow ops."""
    case = _case(name)
    cl, cu, mmaps = cr.refine_bounds(case.ops, case.lb, case.ub, iters=iters,
                                     max_dim=max_dim)
    for k in range(len(case.ops)):
        case.assert_box(k, cl[k], cu[k], "refine_bounds")
    C = _both_sides(case.ops[-1]["dim"])
    lbnd, _, _ = cr.backward_crown(case.ops, case.lb, case.ub, cl, cu, C, mmaps=mmaps)
    case.assert_lower(case.out, C, lbnd, "backward_crown after refine_bounds")


@pytest.mark.parametrize("name", ["attn-narrow", "attn-wide"])
def test_refine_scores_boxes_hold_every_op_value(name):
    """``refine_scores`` tightens the inputs of the exp ops (the scores) with
    ``crown_bounds`` of those inputs. Catches bounds computed for the exp op
    itself and intersected with the score box, which clips the scores to
    the range of their exponentials."""
    case = _case(name)
    cl, cu, mmaps = cr.refine_scores(case.ops, case.lb, case.ub)
    for k in range(len(case.ops)):
        case.assert_box(k, cl[k], cu[k], "refine_scores")
    C = _both_sides(case.ops[-1]["dim"])
    lbnd, _, _ = cr.backward_crown(case.ops, case.lb, case.ub, cl, cu, C, mmaps=mmaps)
    case.assert_lower(case.out, C, lbnd, "backward_crown after refine_scores")


@pytest.mark.parametrize("name", ["pl-wide", "pl-stable", "attn-narrow", "attn-softmax-op"])
def test_backward_crown_with_any_slopes_in_the_unit_interval_is_sound(name):
    """Any relu lower slope and any attention plane weight in [0, 1] gives a
    sound bound (the module's promise that optimisation changes tightness
    only). Three settings: every slope 1 with lower weight 1 and upper weight
    0, every slope 0 with the weights the other way round, and random values
    with both ends present, for the output and for every bmatmul op as the
    start op. Catches a given slope added to a stably active neuron
    ('pl-stable', slope 1) and an effective plane that mixes the lower and
    the upper weight (bmatmul ops, weights at opposite ends)."""
    case = _case(name)
    cl, cu = cr.forward_ibp(case.ops, case.lb, case.ub)
    C = np.vstack([_both_sides(case.ops[-1]["dim"]), _mixed_rows(case.ops[-1]["dim"], 32)])
    rng = np.random.default_rng(41)
    for setting in ("ones", "zeros", "random"):
        alpha = [None] * len(case.ops)
        battn = [None] * len(case.ops)
        for k, op in enumerate(case.ops):
            if op["type"] == "relu":
                alpha[k] = {"ones": np.ones(op["dim"]), "zeros": np.zeros(op["dim"]),
                            "random": rng.random(op["dim"])}[setting]
            elif op["type"] == "bmatmul":
                d = op["dim"]
                if setting == "random":
                    aL, aU = rng.random(d), rng.random(d)
                    aL[0], aU[0], aL[1], aU[1] = 0.0, 1.0, 1.0, 0.0
                else:
                    aL = np.full(d, 1.0 if setting == "ones" else 0.0)
                    aU = 1.0 - aL
                battn[k] = {"aL": aL, "aU": aU}
        lbnd, _, _ = cr.backward_crown(case.ops, case.lb, case.ub, cl, cu, C,
                                       alpha=alpha, battn=battn)
        case.assert_lower(case.out, C, lbnd, f"backward_crown with slopes '{setting}'")
        for k, op in enumerate(case.ops):          # each bmatmul bounded from itself
            if op["type"] == "bmatmul":
                Ck = _both_sides(op["dim"])
                lbnd, _, _ = cr.backward_crown(case.ops, case.lb, case.ub, cl, cu, Ck,
                                               alpha=alpha, battn=battn, start_op=k)
                case.assert_lower(k, Ck, lbnd, f"backward_crown from op {k}, '{setting}'")


@pytest.mark.parametrize("name", ["pl-wide", "attn-narrow", "attn-wide",
                                  "attn-softmax-op", "eprod-signed"])
def test_optimize_alpha_margin_is_a_lower_bound(name):
    """The re-evaluated margins of ``optimize_alpha`` are at most the true
    minimum, and every slope and plane weight it returns lies in [0, 1] (the
    returned solution is reused by branch and bound). Catches a projection
    to [-1, 1], which lets an attention weight go negative. 'eprod-signed'
    has nothing to optimise and takes the plain-bound route."""
    case = _case(name)
    cl, cu = cr.forward_ibp(case.ops, case.lb, case.ub)
    C = _both_sides(case.ops[-1]["dim"])
    margins, sol = cr.optimize_alpha(case.ops, case.lb, case.ub, cl, cu, C,
                                     n_iter=20, lr=0.1)
    case.assert_lower(case.out, C, margins, "optimize_alpha")
    for k in range(len(case.ops)):
        values = [sol["alpha"][k]] if sol["alpha"][k] is not None else []
        if sol["battn"][k] is not None:
            values += [sol["battn"][k]["aL"], sol["battn"][k]["aU"]]
        for v in values:
            v = np.asarray(v, float)
            assert np.all((v >= 0.0) & (v <= 1.0)), \
                f"optimize_alpha returned a weight outside [0, 1] at op {k}: {v}"


# --------------------------------------------------------------------------- #
#  options                                                                      #
# --------------------------------------------------------------------------- #
def _interval_boxes(ops, lb, ub, ov=None):
    """Interval arithmetic in rationals for affine, relu, add and concat ops,
    with the override ``ov[k] = (lo, hi)`` intersected after op k."""
    lo, hi = [], []
    for k, op in enumerate(ops):
        t = op["type"]
        if t == "input":
            l, u = vector(lb), vector(ub)
        elif t == "affine":
            W, b = matrix(op["W"]), vector(op["b"])
            pl, pu = lo[op["in"]], hi[op["in"]]
            l = [sum((w * (a if w >= 0 else c) for w, a, c in zip(row, pl, pu)), bi)
                 for row, bi in zip(W, b)]
            u = [sum((w * (c if w >= 0 else a) for w, a, c in zip(row, pl, pu)), bi)
                 for row, bi in zip(W, b)]
        elif t == "relu":
            l = [max(v, 0) for v in lo[op["in"]]]
            u = [max(v, 0) for v in hi[op["in"]]]
        elif t == "add":
            i, j = op["in"]
            l = [a + c for a, c in zip(lo[i], lo[j])]
            u = [a + c for a, c in zip(hi[i], hi[j])]
        else:
            l = [v for j in op["in"] for v in lo[j]]
            u = [v for j in op["in"] for v in hi[j]]
        if ov is not None and ov[k] is not None:
            l = [max(a, frac(o)) for a, o in zip(l, ov[k][:, 0])]
            u = [min(a, frac(o)) for a, o in zip(u, ov[k][:, 1])]
        lo.append(l)
        hi.append(u)
    return lo, hi


def _assert_boxes_equal(cl, cu, lo, hi, what):
    for k, (a, b, c, d) in enumerate(zip(cl, cu, lo, hi)):
        assert [Fraction(v) for v in a] == c and [Fraction(v) for v in b] == d, \
            f"{what}: op {k}: returned [{a}, {b}], interval arithmetic gives " \
            f"[{[float(v) for v in c]}, {[float(v) for v in d]}]"


def test_forward_ibp_intersects_an_override_with_the_interval_box():
    """``ov[k]`` is intersected with the interval box of op k, and the ops
    after it are propagated from the intersection. The override of 'a' is
    tighter on both sides for entry 0, wider on both sides for entry 1, and
    tighter below and wider above for entry 2; the other ops have none.
    Catches an override merged by maximum on the upper side (or minimum on
    the lower side), which ignores it, and an override not propagated to
    the relu after it."""
    case = _case("pl-wide")
    lo0, hi0 = _interval_boxes(case.ops, case.lb, case.ub)
    a = case.names.index("a")
    l, u = [float(v) for v in lo0[a]], [float(v) for v in hi0[a]]
    ov = [None] * len(case.ops)
    ov[a] = np.array([[l[0] + 2.0, u[0] - 3.0], [l[1] - 1.0, u[1] + 1.0],
                      [l[2] + 4.5, u[2] + 0.5]])
    cl, cu = cr.forward_ibp(case.ops, case.lb, case.ub, ov)
    lo, hi = _interval_boxes(case.ops, case.lb, case.ub, ov)
    _assert_boxes_equal(cl, cu, lo, hi, "forward_ibp with an override")


@pytest.mark.parametrize("route", ["max_dim=0", "iters=0", "refine_scores-without-exp"])
def test_refinement_without_targets_returns_the_interval_boxes(route):
    """With no op to refine, the refinement returns the interval boxes.
    ``max_dim=0`` excludes every target, ``iters=0`` runs no pass, and
    ``refine_scores`` on a list without exp has no score to tighten.
    Catches a ``max_dim`` filter that is dropped (the add before the second
    relu would be tightened)."""
    case = _case("pl-wide")
    if route == "max_dim=0":
        cl, cu, _ = cr.refine_bounds(case.ops, case.lb, case.ub, iters=2, max_dim=0)
    elif route == "iters=0":
        cl, cu, _ = cr.refine_bounds(case.ops, case.lb, case.ub, iters=0)
    else:
        cl, cu, _ = cr.refine_scores(case.ops, case.lb, case.ub)
    lo, hi = _interval_boxes(case.ops, case.lb, case.ub)
    _assert_boxes_equal(cl, cu, lo, hi, route)


@pytest.mark.parametrize("do_relu, do_attn", [(False, True), (True, False), (False, False)])
def test_optimize_alpha_optimises_only_the_requested_variables(do_relu, do_attn):
    """``do_relu`` and ``do_attn`` choose which variables are optimised; the
    returned solution has slopes exactly for the relu ops when ``do_relu``
    and plane weights exactly for the bmatmul ops when ``do_attn``; with
    neither, the plain bound is returned. The margins hold in every case.
    Catches the two flags swapped or read from each other."""
    case = _case("attn-wide")
    cl, cu = cr.forward_ibp(case.ops, case.lb, case.ub)
    C = _both_sides(case.ops[-1]["dim"])
    margins, sol = cr.optimize_alpha(case.ops, case.lb, case.ub, cl, cu, C,
                                     n_iter=5, do_relu=do_relu, do_attn=do_attn)
    for k, op in enumerate(case.ops):
        if op["type"] == "relu":
            assert (sol["alpha"][k] is not None) == do_relu, f"relu op {k}"
        if op["type"] == "bmatmul":
            assert (sol["battn"][k] is not None) == do_attn, f"bmatmul op {k}"
    case.assert_lower(case.out, C, margins, "optimize_alpha")


def test_optimize_alpha_with_a_zero_learning_rate_keeps_the_initial_weights():
    """With ``lr=0`` nothing moves: every relu slope stays at the area
    default (1 where u > -l, else 0, from the boxes passed in) and every
    plane weight at its 0/1 default. Catches ``lr`` ignored (the optimizer
    built with its default rate)."""
    case = _case("attn-wide")
    cl, cu = cr.forward_ibp(case.ops, case.lb, case.ub)
    C = _both_sides(case.ops[-1]["dim"])
    _, sol = cr.optimize_alpha(case.ops, case.lb, case.ub, cl, cu, C, n_iter=10, lr=0.0)
    for k, op in enumerate(case.ops):
        if op["type"] == "relu":
            src = op["in"]
            expected = (cu[src] > -cl[src]).astype(float)
            assert np.array_equal(sol["alpha"][k], expected), \
                f"relu op {k}: {sol['alpha'][k]} instead of {expected}"
        if op["type"] == "bmatmul":
            for side in ("aL", "aU"):
                v = sol["battn"][k][side]
                assert np.all((v == 0.0) | (v == 1.0)), f"bmatmul op {k} {side}: {v}"


# --------------------------------------------------------------------------- #
#  the public route: the verdict of verify_halfspace_group_safe                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name, options", [
    ("pl-stable", {}), ("pl-stable", {"refine": False}),
    ("pl-stable", {"alpha": True, "alpha_iter": 10}),
    ("attn-point", {}), ("attn-point", {"refine": False}),
    ("attn-point", {"alpha": True, "alpha_iter": 10}),
    ("attn-point", {"refine_max_dim": 0}), ("attn-narrow", {}),
], ids=lambda v: v if isinstance(v, str) else ("default" if not v else
                                               ",".join(f"{a}={b}" for a, b in v.items())))
def test_the_verdict_is_not_safe_when_a_true_output_is_unsafe(name, options):
    """For every row r of +I and -I on the output, the unsafe region
    ``G_r y <= g`` with g just above the true minimum of ``G_r y`` holds a
    true output, so ``verify_halfspace_group_safe`` (the competition
    runner's entry) must not answer safe. On a zero-width or stable box the
    bound equals the value, so any error that raises it is a false UNSAT
    here; the options reach refine_bounds, forward_ibp only, optimize_alpha
    with an offset, and the max_dim filter."""
    case = _case(name)
    dim = case.ops[-1]["dim"]
    G = _both_sides(dim)
    tmin = case.min_of(case.out, G)
    for r in range(G.shape[0]):
        g = float(tmin[r]) + 1e-9 * (1.0 + abs(float(tmin[r])))
        safe, margin = vit_crown.verify_halfspace_group_safe(
            case.ops, case.lb, case.ub, G[r:r + 1], [g], **options)
        assert not safe, (f"row {r}: reported safe on a margin of "
                          f"{float(margin[0]):.3e}, but a true output has "
                          f"G y = {float(tmin[r]):.12g} <= g = {g:.12g}")


# --------------------------------------------------------------------------- #
#  the earlier cases: a dense random attention + feed-forward list              #
# --------------------------------------------------------------------------- #
def _build_attn_dag(N, D, rng, with_ff=True):
    """input = token state (dim N*D, e-major) -> Q,K,V dense affine maps ->
    S=scale*QK^T -> softmax(decomposed) -> O=A*V -> [FF relu] -> output.
    Returns the ops, the input dimension, three random functionals and an
    independent forward pass that gives every op's value."""
    E = D                                   # single head
    nin = N * E
    ops = []

    def push(op):
        ops.append(op); return len(ops) - 1

    def aff(in_idx, W, b, mat):
        return dict(type="affine", **{"in": in_idx},
                    W=np.asarray(W, float), b=np.asarray(b, float).reshape(-1),
                    dim=np.asarray(W).shape[0], mat=mat)

    push(dict(type="input", **{"in": []}, dim=nin, mat=None))
    Wq = 0.5 * rng.standard_normal((nin, nin)); bq = 0.1 * rng.standard_normal(nin)
    Wk = 0.5 * rng.standard_normal((nin, nin)); bk = 0.1 * rng.standard_normal(nin)
    Wv = 0.5 * rng.standard_normal((nin, nin)); bv = 0.1 * rng.standard_normal(nin)
    q = push(aff(0, Wq, bq, (N, E)))
    k = push(aff(0, Wk, bk, (N, E)))
    v = push(aff(0, Wv, bv, (N, E)))
    scale = 1.0 / np.sqrt(D)
    sc = push(dict(type="bmatmul", **{"in": [q, k]}, mode="abt",
                   ra=N, ca=D, rb=N, cb=D, dim=N * N, mat=(N, N)))
    scs = push(aff(sc, scale * np.eye(N * N), np.zeros(N * N), (N, N)))
    expi = push(dict(type="exp", **{"in": scs}, dim=N * N, mat=(N, N)))
    Wsum = np.zeros((N, N * N))
    for i in range(N):
        Wsum[i, np.arange(N) * N + i] = 1.0
    sumi = push(aff(expi, Wsum, np.zeros(N), (N, 1)))
    reci = push(dict(type="reciprocal", **{"in": sumi}, dim=N, mat=(N, 1)))
    Wexp = np.zeros((N * N, N))
    for kk in range(N):
        for i in range(N):
            Wexp[kk * N + i, i] = 1.0
    rexp = push(aff(reci, Wexp, np.zeros(N * N), (N, N)))
    ai = push(dict(type="eprod", **{"in": [expi, rexp]}, dim=N * N, mat=(N, N)))
    o = push(dict(type="bmatmul", **{"in": [ai, v]}, mode="ab",
                  ra=N, ca=N, rb=N, cb=D, dim=N * D, mat=(N, D)))
    if with_ff:
        W1 = 0.6 * rng.standard_normal((N * D, N * D)); b1 = 0.1 * rng.standard_normal(N * D)
        f1 = push(aff(o, W1, b1, (N, D)))
        r = push(dict(type="relu", **{"in": f1}, dim=N * D, mat=(N, D)))
        W2 = 0.6 * rng.standard_normal((N * D, N * D)); b2 = 0.1 * rng.standard_normal(N * D)
        push(aff(r, W2, b2, (N, D)))
    # DAG ends at the feature output (dim N*D); the 3 'margin' functionals C are
    # applied by the verifier as the backward spec (not baked into the DAG).
    Cm = rng.standard_normal((3, N * D))

    def forward(X):
        P = X.shape[0]
        Q = _mat(X @ Wq.T + bq, N, E)
        K = _mat(X @ Wk.T + bk, N, E)
        V = _mat(X @ Wv.T + bv, N, E)
        S = Q @ K.transpose(0, 2, 1)
        Ex = np.exp(scale * S)
        R = 1.0 / Ex.sum(axis=2)
        O = _softmax_rows(scale * S) @ V
        vals = [X, _vec(Q), _vec(K), _vec(V), _vec(S), _vec(scale * S), _vec(Ex),
                1.0 / R, R, _vec(np.broadcast_to(R[:, :, None], (P, N, N))),
                _vec(_softmax_rows(scale * S)), _vec(O)]
        if with_ff:
            F1 = _vec(O) @ W1.T + b1
            vals += [F1, np.maximum(F1, 0.0), np.maximum(F1, 0.0) @ W2.T + b2]
        return vals

    return ops, nin, Cm, forward


def _box(nin, rng, width=0.4):
    c = 0.5 * rng.standard_normal(nin)
    return c - width / 2, c + width / 2


def _dag_truth(ops, forward, lb, ub, seed, n_samples):
    pts = _points(lb, ub, seed, n_samples)
    return _Case(ops, [str(i) for i in range(len(ops))], lb, ub, values=forward(pts))


@pytest.mark.parametrize("seed", [0, 1])
def test_forward_ibp_encloses_all_ops(seed):
    """Interval bounds of every op of a dense random attention list hold the
    independent forward pass at every corner, the centre and 400 samples.
    (Seeds 2 and 3 were removed: their catches are a subset of those of
    seeds 0 and 1.)"""
    rng = np.random.default_rng(seed)
    ops, nin, _, forward = _build_attn_dag(3, 2, rng)
    lb, ub = _box(nin, rng)
    case = _dag_truth(ops, forward, lb, ub, 1000 + seed, 400)
    cl, cu = cr.forward_ibp(ops, lb, ub)
    for kk in range(len(ops)):
        case.assert_box(kk, cl[kk], cu[kk], "forward_ibp")


@pytest.mark.parametrize("seed", [1])
def test_backward_crown_is_true_lower_bound(seed):
    """The backward bound of three random functionals of the output is at
    most their true minimum over the corners, the centre and 2000 samples.
    (Seeds 0, 2, 3 and 4 were removed; the wide random box
    leaves a large gap, and the tight checks are in
    test_backward_crown_bounds_the_output_from_both_sides.)"""
    rng = np.random.default_rng(100 + seed)
    ops, nin, C, forward = _build_attn_dag(3, 2, rng)
    lb, ub = _box(nin, rng)
    case = _dag_truth(ops, forward, lb, ub, 1100 + seed, 2000)
    cl, cu = cr.forward_ibp(ops, lb, ub)
    mb, _, _ = cr.backward_crown(ops, lb, ub, cl, cu, C)
    case.assert_lower(case.out, C, mb, "backward_crown")


@pytest.mark.parametrize("seed", [0, 2])
def test_refine_is_sound_and_not_looser(seed):
    """The boxes of two refinement passes hold every op's true values, and
    so does the backward bound computed from them. (The name is kept for the
    test id; the tightness checks it once had were removed: tightness is not
    a test-suite concern. Seed 1 was removed: its catches are a subset of
    those of seeds 0 and 2.)"""
    rng = np.random.default_rng(200 + seed)
    ops, nin, C, forward = _build_attn_dag(3, 2, rng)
    lb, ub = _box(nin, rng)
    case = _dag_truth(ops, forward, lb, ub, 1200 + seed, 400)
    clr, cur, mmaps = cr.refine_bounds(ops, lb, ub, iters=2)
    for kk in range(len(ops)):
        case.assert_box(kk, clr[kk], cur[kk], "refine_bounds")
    mbr, _, _ = cr.backward_crown(ops, lb, ub, clr, cur, C, mmaps=mmaps)
    case.assert_lower(case.out, C, mbr, "backward_crown after refine_bounds")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_alpha_optimized_margin_still_sound(seed):
    """The re-evaluated margins of ``optimize_alpha`` after refinement are at
    most the true minimum (any slope in [0, 1] is sound)."""
    rng = np.random.default_rng(300 + seed)
    ops, nin, C, forward = _build_attn_dag(3, 2, rng)
    lb, ub = _box(nin, rng)
    case = _dag_truth(ops, forward, lb, ub, 1300 + seed, 2000)
    cl, cu, _ = cr.refine_bounds(ops, lb, ub, iters=1)
    margins, sol = cr.optimize_alpha(ops, lb, ub, cl, cu, C, n_iter=20, lr=0.1)
    case.assert_lower(case.out, C, margins, "optimize_alpha")


# --------------------------------------------------------------------------- #
#  the McCormick planes of one product                                          #
# --------------------------------------------------------------------------- #
def test_every_mccormick_plane_bounds_the_product_on_the_box():
    """Each of the two lower planes of ``mc_corners`` lies below p*q and each
    of the two upper planes above it on [pl, pu] x [ql, qu]. A plane minus
    p*q is bilinear, so it is checked exactly at the four corners. The
    intervals have both signs, one is entirely negative and one has zero
    width. The plane picked at the midpoint is a rounding tie (all four have
    the same gap there), so a wrong second plane is used by the backward pass
    only where rounding picks it; this check sees every plane."""
    pl = np.array([-1.5, 0.5, -3.0, 1.25, -0.75])
    pu = np.array([2.25, 3.0, -1.0, 1.25, 0.5])
    ql = np.array([-0.75, -2.0, 0.25, -1.0, -2.5])
    qu = np.array([1.5, -0.25, 0.5, 2.0, -1.5])
    PL, PU, _ = cr.mc_corners(pl, pu, ql, qu)
    for j in range(pl.size):
        for p, q in product((pl[j], pu[j]), (ql[j], qu[j])):
            true = Fraction(p) * Fraction(q)
            for planes, side in ((PL, "lower"), (PU, "upper")):
                for i in (1, 2):
                    value = (Fraction(planes[f"ap{i}"][j]) * Fraction(p)
                             + Fraction(planes[f"aq{i}"][j]) * Fraction(q)
                             + Fraction(planes[f"a0{i}"][j]))
                    ok = value <= true if side == "lower" else value >= true
                    assert ok, (f"{side} plane {i} of entry {j} is {float(value)} at "
                                f"p={p}, q={q}, where p*q = {float(true)}")


# --------------------------------------------------------------------------- #
#  bmatmul alone: the McCormick relaxation against the exact extremes           #
# --------------------------------------------------------------------------- #
def _bmatmul_case(mode, ra, ca, rb, cb, seed):
    """A and B are disjoint coordinates of the input, so every entry of the
    product (and any linear function of it) is multilinear and has its
    extremes at the corners of the box."""
    nA, nB = ra * ca, rb * cb
    nin = nA + nB
    Sa = np.zeros((nA, nin)); Sa[:, :nA] = np.eye(nA)
    Sb = np.zeros((nB, nin)); Sb[:, nA:] = np.eye(nB)
    dimY = ra * rb if mode == "abt" else ra * cb
    ops = [dict(type="input", **{"in": []}, dim=nin, mat=None),
           _affine(0, Sa, np.zeros(nA), (ra, ca)), _affine(0, Sb, np.zeros(nB), (rb, cb)),
           dict(type="bmatmul", **{"in": [1, 2]}, mode=mode, ra=ra, ca=ca, rb=rb, cb=cb,
                dim=dimY, mat=None)]
    rng = np.random.default_rng(seed)
    centre = rng.uniform(-1.0, 1.0, nin)
    lb, ub = centre - 0.3, centre + 0.3

    def forward(X):
        A, B = _mat(X[:, :nA], ra, ca), _mat(X[:, nA:], rb, cb)
        Y = A @ B.transpose(0, 2, 1) if mode == "abt" else A @ B
        return [X, X[:, :nA], X[:, nA:], _vec(Y)]

    case = _Case(ops, ["x", "A", "B", "Y"], lb, ub, values=forward(_corners(lb, ub)))
    C = np.vstack([_both_sides(dimY), rng.standard_normal((4, dimY))])
    return case, C


def _assert_bmatmul_bounds(case, C):
    cl, cu = cr.forward_ibp(case.ops, case.lb, case.ub)
    case.assert_box(3, cl[3], cu[3], "forward_ibp")
    mb, _, _ = cr.backward_crown(case.ops, case.lb, case.ub, cl, cu, C)
    case.assert_lower(3, C, mb, "backward_crown")


def test_bmatmul_both_modes_mccormick_sound():
    """Both bilinear modes, Q K^T ('abt') and A V ('ab' with V not square):
    the interval box and the backward bound of +I, -I and random functionals
    hold the exact extremes (all corners). Catches a wrong McCormick plane,
    a plane stored at the transposed output entry, a wrong operand stride and
    a dropped radius term of the interval product."""
    for mode, shape in [("abt", (3, 2, 3, 2)), ("ab", (3, 3, 3, 2))]:
        case, C = _bmatmul_case(mode, *shape, seed=7)
        _assert_bmatmul_bounds(case, C)


_ROWS = ("bmm_maps addresses the second operand with the row count of the "
         "first (N = ra), so a bmatmul whose operands have different row counts "
         "gets McCormick planes on the wrong entries: the backward bound misses "
         "true products, or bmm_maps raises IndexError for the other orientation")


@pytest.mark.parametrize("mode, shape", [
    pytest.param("abt", (2, 2, 3, 2), marks=known_bug(19, _ROWS), id="abt-more-key-rows"),
    pytest.param("ab", (2, 3, 3, 2), marks=known_bug(19, _ROWS), id="ab-wide-first-operand"),
])
def test_bmatmul_bounds_hold_when_the_operands_have_different_row_counts(mode, shape):
    """Cross-attention shapes: Q with 2 rows against K with 3 rows ('abt'),
    and a 2 x 3 weight matrix times a 3 x 2 value matrix ('ab'). The neighbour
    with equal row counts is test_bmatmul_both_modes_mccormick_sound."""
    case, C = _bmatmul_case(mode, *shape, seed=8)
    _assert_bmatmul_bounds(case, C)


# --------------------------------------------------------------------------- #
#  defects of today's library on legal op lists                                 #
# --------------------------------------------------------------------------- #
_SOFTMAX_ONLY = ("backward_crown leaves the coefficient of the input unset when "
                 "the bounded value reaches the input only through softmax ops, "
                 "and the concretisation fails with TypeError on None; "
                 "refine_bounds (the default of crown_margins) reaches it for a "
                 "softmax op that feeds a bmatmul")


@pytest.mark.parametrize("route", [
    pytest.param("backward_crown", marks=known_bug(19, _SOFTMAX_ONLY, raises=TypeError)),
    pytest.param("crown_margins", marks=known_bug(19, _SOFTMAX_ONLY, raises=TypeError)),
])
def test_a_value_reached_only_through_softmax_is_bounded(route):
    """'backward_crown': the bounds of a softmax op of an affine map of the
    input (both sides). 'crown_margins': a softmax op that feeds a bmatmul
    with a value matrix, through the public entry with its default
    refinement; the margins of +I and -I must be at most the true minima.
    The passing neighbour is the case 'attn-softmax-op' of the tests above."""
    if route == "backward_crown":
        case = _case("softmax-dominant")
        ops = case.ops[:3]
        cl, cu = cr.forward_ibp(ops, case.lb, case.ub)
        lbnd, _, _ = cr.backward_crown(ops, case.lb, case.ub, cl, cu, _both_sides(6))
        case.assert_lower(2, _both_sides(6), lbnd, "backward_crown at the softmax op")
    else:
        N, D = 2, 3
        nin = N * N + N * D
        Ss = np.zeros((N * N, nin)); Ss[:, :N * N] = np.diag([1.0, 2.0, -1.5, 0.5])
        Sv = np.zeros((N * D, nin)); Sv[:, N * N:] = np.eye(N * D)
        ops = [dict(type="input", **{"in": []}, dim=nin, mat=None),
               _affine(0, Ss, [0.5, -0.25, 0.0, 1.0], (N, N)),
               dict(type="softmax", **{"in": 1}, dim=N * N, mat=(N, N)),
               _affine(0, Sv, np.zeros(N * D), (N, D)),
               dict(type="bmatmul", **{"in": [2, 3]}, mode="ab", ra=N, ca=N, rb=N,
                    cb=D, dim=N * D, mat=(N, D))]
        lb = np.linspace(-1.0, 1.0, nin)
        ub = lb + 0.25

        def forward(X):
            Z = X @ Ss.T + np.array([0.5, -0.25, 0.0, 1.0])
            A = _softmax_rows(_mat(Z, N, N))
            V = _mat(X @ Sv.T, N, D)
            return [X, Z, _vec(A), _vec(V), _vec(A @ V)]

        case = _Case(ops, ["x", "z", "a", "v", "o"], lb, ub,
                     values=forward(_points(lb, ub, 9, 200)))
        C = _both_sides(N * D)
        margins = vit_crown.crown_margins(ops, lb, ub, C)
        case.assert_lower(case.out, C, margins, "crown_margins")


def _softmax_row_ops(n):
    """exp -> row sum -> reciprocal -> broadcast -> eprod on one row of n logits."""
    return [dict(type="input", **{"in": []}, dim=n, mat=None),
            dict(type="exp", **{"in": 0}, dim=n, mat=(1, n)),
            _affine(1, np.ones((1, n)), np.zeros(1), (1, 1)),
            dict(type="reciprocal", **{"in": 2}, dim=1, mat=(1, 1)),
            _affine(3, np.ones((n, 1)), np.zeros(n), (1, n)),
            dict(type="eprod", **{"in": [1, 4]}, dim=n, mat=(1, n))]


@known_bug(19, "the reciprocal op floors its input range at 1e-12: when the sum "
               "of exponentials of an attention row is below 1e-12 (every score "
               "below about -28) its interval is inverted and its relaxation is "
               "built on the wrong range, so the attention weights' bounds miss "
               "their true values and a reachable unsafe region is reported safe")
def test_attention_weights_are_bounded_when_every_score_is_very_negative():
    """Two logits in [-30, -29.5] and [-31, -30.5]: the true sum of their
    exponentials lies in [1.3e-13, 2.2e-13]. The verdict that the first
    weight stays below 0.75 must not be 'safe' (it reaches 0.82), and the
    interval and backward bounds of both weights must hold. The passing
    neighbour is every attention case above (scores of moderate size)."""
    ops = _softmax_row_ops(2)
    lb, ub = np.array([-30.0, -31.0]), np.array([-29.5, -30.5])
    pts = _points(lb, ub, 10, 300)
    A = _softmax_rows(pts)
    if not A[:, 0].max() > 0.8:
        pytest.fail("setup: the unsafe region is not reached")
    safe, margin = vit_crown.verify_halfspace_group_safe(
        ops, lb, ub, [[-1.0, 0.0]], [-0.75])
    assert not safe, f"reported safe on a margin of {margin[0]:.3f}"
    cl, cu = cr.forward_ibp(ops, lb, ub)
    lbnd, _, _ = cr.backward_crown(ops, lb, ub, cl, cu, _both_sides(2))
    assert np.all(cl[5] <= A.min(0)) and np.all(cu[5] >= A.max(0)), (cl[5], cu[5])
    assert np.all(lbnd[:2] <= A.min(0)) and np.all(-lbnd[2:] >= A.max(0)), lbnd


@known_bug(19, "the reciprocal op assumes a positive input (it floors the range "
               "at 1e-12): over a negative range it answers with bounds that do "
               "not hold instead of bounding 1/t or refusing")
def test_a_reciprocal_of_a_negative_range_is_bounded_or_refused():
    """1/t over t in [-2, -1] has the exact range [-1, -0.5]. A refusal
    (NotImplementedError or ValueError) keeps the promise; bounds must hold."""
    ops = [dict(type="input", **{"in": []}, dim=1, mat=None),
           dict(type="reciprocal", **{"in": 0}, dim=1, mat=None)]
    lb, ub = np.array([-2.0]), np.array([-1.0])
    try:
        cl, cu = cr.forward_ibp(ops, lb, ub)
        lbnd, _, _ = cr.backward_crown(ops, lb, ub, cl, cu, np.array([[1.0], [-1.0]]))
    except (NotImplementedError, ValueError):
        return
    assert cl[1][0] <= -1.0 and cu[1][0] >= -0.5, ("interval", cl[1], cu[1])
    assert lbnd[0] <= -1.0 and -lbnd[1] >= -0.5, ("backward", lbnd)


# --------------------------------------------------------------------------- #
#  rounding to nearest (registry VERDICT-1), the interval pass                  #
# --------------------------------------------------------------------------- #
_ROUNDING = ("VERDICT-1: the CROWN engine computes bounds with rounding to "
             "nearest and no outward term; here the interval pass, whose box of a "
             "zero-width input can exclude the exact value")


@pytest.mark.parametrize("kind, x", [
    pytest.param("affine", (1.0, 2.0 ** -53), marks=known_bug(7, _ROUNDING),
                 id="affine-rounded"),
    pytest.param("affine", (1.0, 2.0 ** -4), id="affine-exact"),
    pytest.param("exp", (0.5,), marks=known_bug(7, _ROUNDING), id="exp-rounded"),
    pytest.param("exp", (0.0,), id="exp-exact"),
])
def test_forward_ibp_box_of_a_point_holds_its_exact_value(kind, x):
    """On a zero-width box the interval of an op must still hold its exact
    value: the sum 1 + 2^-53 rounds to 1, and exp(0.5) is irrational, so a
    box [v, v] with a float v cannot hold it. The exact value is computed in
    rationals or with 200-bit mpmath. 'affine-exact' and 'exp-exact' are the
    neighbours where float64 is exact."""
    x = np.array(x)
    if kind == "affine":
        ops = [dict(type="input", **{"in": []}, dim=2, mat=None),
               _affine(0, [[1.0, 1.0]], [0.0])]
        cl, cu = cr.forward_ibp(ops, x, x)
        exact = Fraction(x[0]) + Fraction(x[1])
        assert Fraction(cl[1][0]) <= exact <= Fraction(cu[1][0]), (cl[1], cu[1], exact)
    else:
        ops = [dict(type="input", **{"in": []}, dim=1, mat=None),
               dict(type="exp", **{"in": 0}, dim=1, mat=None)]
        cl, cu = cr.forward_ibp(ops, x, x)
        with mpmath.workprec(200):
            exact = mpmath.exp(mpmath.mpf(float(x[0])))
            assert mpmath.mpf(float(cl[1][0])) <= exact <= mpmath.mpf(float(cu[1][0])), \
                (cl[1], cu[1], exact)


@pytest.mark.parametrize("tiny", [
    pytest.param(2.0 ** -53, marks=known_bug(7, _ROUNDING + "; a relu whose input "
                 "is exactly 2^-53 is taken as stably inactive and a reachable "
                 "unsafe region is reported safe"), id="rounded"),
    pytest.param(2.0 ** -4, id="exact"),
])
def test_relu_stability_is_not_decided_on_a_rounded_interval(tiny):
    """z = (x1 + x2) - 1 at the point (1, tiny) is exactly tiny, so relu(z)
    = tiny and the unsafe region -relu(z) <= -tiny/2 is reached. With
    tiny = 2^-53 the interval of x1 + x2 rounds to [1, 1], z to [0, 0], the
    relu is taken as inactive and its bound is 0: the verdict must still
    not be 'safe'. 'exact' is the neighbour where float64 is exact."""
    ops = [dict(type="input", **{"in": []}, dim=2, mat=None),
           _affine(0, [[1.0, 1.0]], [0.0]), _affine(1, [[1.0]], [-1.0]),
           dict(type="relu", **{"in": 2}, dim=1, mat=None)]
    x = np.array([1.0, tiny])
    g = -tiny / 2
    z = Fraction(1) + Fraction(tiny) - 1
    if not -max(z, 0) <= Fraction(g):
        pytest.fail("setup: the point is not in the unsafe region")
    safe, margin = vit_crown.verify_halfspace_group_safe(ops, x, x, [[-1.0]], [g],
                                                         refine=False)
    assert not safe, f"reported safe on a margin of {margin[0]:.3e}"

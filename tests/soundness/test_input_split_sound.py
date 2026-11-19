"""
Soundness of the batched-CROWN input-split driver
(``n2v.refine.verify_refine_input_crown``) and of its bound engine
(``n2v.refine.crown_bounds``: ``prepare_layers``, ``CrownBatch`` and its
helpers).

Promises checked:

* ``CrownBatch.margin_report`` returns, for every box of a batch, a lower
  bound of each spec-row margin that is not above the exact minimum over the
  box, rounding included; ``CrownBatch.pre_bounds`` hold the exact range of
  every ReLU input. Reference: the exact image in fractions
  (``tests/_oracles/rational.py``) on dyadic networks, and on non-dyadic
  rounding-scale instances (cancelling biases, products of one sign, a
  crossing neuron behind a cancelling affine map, a deep chain whose backward
  products all round the same way). The ReLU relaxation and the magnitude
  chain behind the rounding slack are checked on their own promises.
* The driving row, its candidate vertex and the split scores of
  ``margin_report`` follow ``row_rule``, checked against exact linear forms.
* The driver answers UNSAT only when the boxes it proved tile the input box
  (recorded through a wrapper around ``CrownBatch``), SAT only with a witness
  in the user's box (zero tolerance) whose exact and float64 outputs lie in
  the closed unsafe region, and UNKNOWN when a budget runs out, a box is a
  sliver or the input set cannot be finitized; ``node_budget`` is exact,
  ``time_budget`` is checked per wave, the counters describe the bisection
  tree. Unsupported models and options are refused.
* The older cases below (``TestInputCrown``) compare verdicts with exact star
  reach (an n2v path) on random nets and keep the regression cases of earlier
  fixes (touching specs, the batched-hit re-check, finitization).

Every case runs on the CPU; the CUDA path (``device='cuda'``) is the same
tensor code on another device and is not exercised here.
"""

import sys
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.nn as nn

from n2v.nn.reach import reach_pytorch_model
from n2v.sets.star import Star
from n2v.utils.lpsolver import check_feasibility
from n2v.refine import verify_refine_input_crown
from n2v.refine.crown_bounds import (CrownBatch, _backward, _relu_relax,
                                     prepare_layers)
from n2v.refine.reach_relaxed import LinearLayer, ReluLayer, extract_layers
from n2v.refine.types import LinearSpec, Status
from tests._oracles import rational as Q
from tests.known_bugs import DID_NOT_RAISE, known_bug
from tests.unit.refine.conftest import seq_net


def _star_intersects_unsafe(S: Star, spec: LinearSpec) -> bool:
    c_out = S.V[:, 0]; V_out = S.V[:, 1:]
    A_rows = [spec.G @ V_out]; b_rows = [spec.g - spec.G @ c_out]
    if S.C.size > 0:
        A_rows.append(S.C); b_rows.append(S.d.flatten())
    return check_feasibility(A=np.vstack(A_rows), b=np.concatenate(b_rows),
                             lb=S.predicate_lb.flatten(), ub=S.predicate_ub.flatten())


def _oracle(model, S_in, spec):
    for S in reach_pytorch_model(model, S_in, method="exact"):
        if _star_intersects_unsafe(S, spec):
            return Status.SAT
    return Status.UNSAT


class TestInputCrown:
    """Soundness of the batched-CROWN input-split driver."""

    @staticmethod
    def _setup(seed, dims=(2, 6, 6, 2)):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        mods = []
        for i in range(len(dims) - 2):
            mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        mods += [nn.Linear(dims[-2], dims[-1])]
        return nn.Sequential(*mods).double()

    def test_crown_bounds_enclose_exact_outputs(self):
        """``output_forms`` (no rounding slack) hold the float64 outputs of
        200 sampled points per box, for 16 boxes, with a pad of the 1e-12
        class (at the exact vertices of the pieces the forms are off by at
        most 6e-18 on this net). Catches a lower or upper form that takes the
        wrong side of a ReLU relaxation, a bias through the wrong
        coefficient, and pre-activation bounds that are too narrow."""
        import torch
        from n2v.refine.crown_bounds import CrownBatch, prepare_layers
        from n2v.refine.reach_relaxed import extract_layers
        rng = np.random.default_rng(0)
        net = self._setup(3)
        layers = extract_layers(net)
        LB = rng.uniform(-1, -0.1, size=(16, 2))
        UB = LB + rng.uniform(0.2, 1.0, size=(16, 2))
        cb = CrownBatch(prepare_layers(layers), LB, UB)
        Al, bl, Au, bu = cb.output_forms()
        for b in range(16):
            X = rng.uniform(LB[b], UB[b], size=(200, 2))
            with torch.no_grad():
                Y = net(torch.as_tensor(X)).numpy()
            lo = (Al[b].numpy() @ X.T).T + bl[b].numpy()
            hi = (Au[b].numpy() @ X.T).T + bu[b].numpy()
            assert np.all(Y >= lo - 1e-12), "CROWN lower form violated"
            assert np.all(Y <= hi + 1e-12), "CROWN upper form violated"

    # Random nets, all of them SAT by the oracle; seeds 0, 5 and 7 catch every
    # injected bug that the eight seeds 0 to 7 caught.
    @pytest.mark.parametrize("seed", [0, 5, 7])
    def test_verdict_matches_oracle(self, seed):
        from n2v.refine import verify_refine_input_crown
        rng = np.random.default_rng(100 + seed)
        net = self._setup(seed)
        S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        out0 = reach_pytorch_model(net, S_in, method="exact")
        mn = min(float(S.get_range(0)[0]) for S in out0)
        mx = max(float(S.get_range(0)[1]) for S in out0)
        tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
        spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))
        oracle = _oracle(net, S_in, spec)
        # node-budget bounded (deterministic), no wall-clock coupling: tiny
        # nets decide within a few hundred boxes
        res = verify_refine_input_crown(S_in, net, spec, node_budget=200_000)
        assert res.status != Status.UNKNOWN
        assert res.status == oracle, f"crown verdict {res.status} != {oracle}"

    @pytest.mark.parametrize("row_rule,seed", [
        ("argmax", 0), ("argmin", 0), ("argmax", 1), ("argmin", 2)])
    def test_verdict_matches_oracle_multirow(self, seed, row_rule):
        """Multi-row unsafe regions exercise the driving-row selection (the
        single-row tests cannot: argmin == argmax there). Seed 0 is UNSAT by
        the oracle under both rules (a box ruled out by one row of the
        conjunction while the other is satisfiable), seeds 1 and 2 SAT; the
        four cases catch every injected bug that the eight cases of both
        rules on seeds 0 to 3 caught."""
        from n2v.refine import verify_refine_input_crown
        rng = np.random.default_rng(700 + seed)
        net = self._setup(seed)
        S_in = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        out0 = reach_pytorch_model(net, S_in, method="exact")
        mn0 = min(float(x.get_range(0)[0]) for x in out0)
        mx0 = max(float(x.get_range(0)[1]) for x in out0)
        mn1 = min(float(x.get_range(1)[0]) for x in out0)
        mx1 = max(float(x.get_range(1)[1]) for x in out0)
        t0 = mn0 + (mx0 - mn0) * (0.3 + 0.4 * rng.random())
        t1 = mn1 + (mx1 - mn1) * (0.3 + 0.4 * rng.random())
        # unsafe = {y0 <= t0 AND y1 <= t1}: G = -I rows (y_i >= -t_i form)
        spec = LinearSpec(G=np.array([[1.0, 0.0], [0.0, 1.0]]),
                          g=np.array([t0, t1]))
        oracle = _oracle(net, S_in, spec)
        res = verify_refine_input_crown(S_in, net, spec, node_budget=200_000,
                                        row_rule=row_rule)
        assert res.status != Status.UNKNOWN
        assert res.status == oracle, f"multirow {row_rule}: {res.status} != {oracle}"

    @pytest.mark.parametrize("seed", [0, 3])
    def test_constrained_input_star_stays_sound(self, seed):
        """On a genuinely constrained input star (extra C/d halfspace) the
        driver searches the BOUNDING box: it must never report a SAT whose
        counterexample lies outside the true input set, and must never report
        UNSAT when the exact oracle (which respects C/d) finds the set unsafe.
        UNKNOWN is acceptable (documented incompleteness for polytope inputs)."""
        from n2v.refine import verify_refine_input_crown
        rng = np.random.default_rng(500 + seed)
        net = self._setup(seed)
        S = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        S2 = Star(S.V, np.vstack([S.C, [[1.0, 1.0]]]),
                  np.vstack([S.d, [[0.5]]]), S.predicate_lb, S.predicate_ub)
        out0 = reach_pytorch_model(net, S2, method="exact")
        mn = min(float(x.get_range(0)[0]) for x in out0)
        mx = max(float(x.get_range(0)[1]) for x in out0)
        tau = mn + (mx - mn) * (0.2 + 0.6 * rng.random())
        spec = LinearSpec(G=np.array([[1.0, 0.0]]), g=np.array([tau]))
        oracle = _oracle(net, S2, spec)
        res = verify_refine_input_crown(S2, net, spec, time_budget=20.0)
        if res.status == Status.SAT:
            assert oracle == Status.SAT
            assert np.all(S2.contains(
                np.asarray(res.counterexample_x).reshape(1, -1), method="lp"))
        elif res.status == Status.UNSAT:
            assert oracle == Status.UNSAT

    def test_out_of_set_unsafe_vertex_is_never_sat(self):
        """Deterministic non-vacuous membership check: identity net, unsafe
        region = exactly the box corner that a C/d halfspace CUTS OFF. The
        CROWN driver searches the bounding box, so its candidate vertex IS
        that corner and its exact forward IS unsafe -- the zero-tolerance
        membership filter is the only thing standing between this and a false
        SAT. The box also can't be pruned (the unsafe corner is in the
        bounding box), so a SAT return here is a soundness bug, full stop."""
        import torch.nn as nn
        import torch as th
        from n2v.refine import verify_refine_input_crown
        net = nn.Sequential(nn.Linear(2, 2))
        with th.no_grad():
            net[0].weight.copy_(th.eye(2))
            net[0].bias.zero_()
        net = net.double()
        S = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        # cut off the (1,1) corner: alpha_0 + alpha_1 <= 0.5
        S2 = Star(S.V, np.vstack([S.C, [[1.0, 1.0]]]),
                  np.vstack([S.d, [[0.5]]]), S.predicate_lb, S.predicate_ub)
        # unsafe = {y_0 >= 0.9 AND y_1 >= 0.9} -- reachable only from the cut
        # corner: G = -I, g = -0.9
        spec = LinearSpec(G=-np.eye(2), g=np.array([-0.9, -0.9]))
        res = verify_refine_input_crown(S2, net, spec, node_budget=20_000)
        assert res.status != Status.SAT, \
            "SAT with an out-of-set counterexample (membership filter broken)"

    @pytest.mark.parametrize("seed", [15, 33, 36, 106])
    def test_touching_spec_is_never_unsat(self, seed):
        """The unsafe region touches the image of the input box at one vertex
        (exact margin 0 there). The CROWN margins must be outward-rigorous in
        floating point, so the touching vertex is never rounded into a
        prune; these seeds gave a false UNSAT without the rounding slack.
        The four kept seeds have g = y < -1 (15), > 1 (33), in (0, 1) (36)
        and in (-1, 0) (106); at seeds 15 and 33 the rebuilt corner leaves
        the user's box by an ulp, so the witness must be snapped back."""
        import torch
        import torch.nn as nn
        from n2v.refine import verify_refine_input_crown
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 1)).double()
        with torch.no_grad():
            net[0].bias.add_(10.0)            # hidden neurons stably active
        lb = rng.uniform(-1, 0, size=5)
        ub = lb + rng.uniform(0.1, 1, size=5)
        S = Star.from_bounds(lb, ub)
        # the vertex minimizing the (affine) output, as a star member
        W = (net[2].weight @ net[0].weight).detach().numpy().flatten()
        x = S.V[:, 0] + np.diag(S.V[:, 1:]) * np.where(W >= 0, -1.0, 1.0)
        with torch.no_grad():
            y = net(torch.as_tensor(x).unsqueeze(0)).numpy().flatten()[0]
        spec = LinearSpec(G=np.array([[1.0]]), g=np.array([y]))
        res = verify_refine_input_crown(S, net, spec, node_budget=3000)
        assert res.status != Status.UNSAT, "false UNSAT on a touching spec"
        if res.status == Status.SAT:
            xc = np.asarray(res.counterexample_x, dtype=np.float64)
            assert np.all(xc >= lb) and np.all(xc <= ub)
            with torch.no_grad():
                yc = net(torch.as_tensor(xc).unsqueeze(0)).numpy().flatten()
            assert np.all(spec.G @ yc - spec.g <= 0.0)

    def test_box_corner_counterexample_is_found(self):
        """y0 + y1 >= 2 holds only at the corner (1, 1). The outward search
        box puts that vertex a few ulps outside the input set; projected back
        into the predicate set it is a valid counterexample at once."""
        import torch
        import torch.nn as nn
        from n2v.refine import verify_refine_input_crown
        net = nn.Sequential(nn.Linear(2, 2)).double()
        with torch.no_grad():
            net[0].weight.copy_(torch.eye(2, dtype=torch.float64))
            net[0].bias.zero_()
        S = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        spec = LinearSpec(G=-np.ones((1, 2)), g=np.array([-2.0]))
        res = verify_refine_input_crown(S, net, spec, node_budget=20_000)
        assert res.status == Status.SAT and res.nodes <= 10, \
            f"{res.status} after {res.nodes} nodes"
        x = np.asarray(res.counterexample_x)
        assert np.all(x >= -1.0) and np.all(x <= 1.0)

    @staticmethod
    def _inf_box_instance():
        """y = relu(x0) + relu(x1), with x0 bounded only by C rows (the
        predicate box is +/-inf on dim 0)."""
        import torch
        import torch.nn as nn
        net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1)).double()
        with torch.no_grad():
            net[0].weight.copy_(torch.eye(2, dtype=torch.float64))
            net[0].bias.zero_()
            net[2].weight.copy_(torch.tensor([[1.0, 1.0]], dtype=torch.float64))
            net[2].bias.zero_()
        V = np.hstack([np.zeros((2, 1)), np.eye(2)])
        C = np.array([[1.0, 0.0], [-1.0, 0.0]])       # -1 <= alpha_0 <= 1
        d = np.array([[1.0], [1.0]])
        S = Star(V, C, d, np.array([[-np.inf], [-1.0]]),
                 np.array([[np.inf], [1.0]]))
        return net, S

    def test_infinite_predicate_box_is_finitized(self):
        """The CROWN search box comes from interval bounds, which are NaN on
        an infinite predicate box (such margins never prune): the driver
        finitizes the box first and decides the far-UNSAT spec."""
        from n2v.refine import verify_refine_input_crown
        net, S = self._inf_box_instance()
        res = verify_refine_input_crown(S, net, LinearSpec([[-1.0]], [-99.0]),
                                        node_budget=5000)
        assert res.status == Status.UNSAT
        res = verify_refine_input_crown(S, net, LinearSpec([[-1.0]], [-1.5]),
                                        node_budget=5000)
        assert res.status != Status.UNSAT
        if res.status == Status.SAT:
            x = np.asarray(res.counterexample_x, dtype=np.float64)
            assert np.all(S.C @ x <= S.d.flatten()), "counterexample outside C"
            assert -1.0 <= x[1] <= 1.0

    def test_batched_forward_hit_is_revalidated_per_point(self):
        """The batched exact forward pass only filters candidates: a model
        that reports "unsafe" for every batch of more than one input must not
        yield SAT, because each hit is re-checked by a single-input pass."""
        import torch
        import torch.nn as nn
        from n2v.refine import verify_refine_input_crown
        from n2v.refine.reach_relaxed import extract_layers

        net = seq_net(2, 8, 8, 1, seed=1)
        S = Star.from_bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        out0 = reach_pytorch_model(net, S, method="exact")
        mn = min(float(T.get_range(0)[0]) for T in out0)
        mx = max(float(T.get_range(0)[1]) for T in out0)
        spec = LinearSpec(np.array([[1.0]]), np.array([mn - 0.05 * (mx - mn)]))

        class _BatchLiar(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.lies = 0

            def forward(self, x):
                if x.shape[0] > 1:
                    self.lies += 1
                    return torch.full((x.shape[0], 1), -1e9,
                                      dtype=torch.float64)
                return self.inner(x)

        # eval mode up front: no eval-mode copy is made, so the lie counter is
        # read from this object
        wrapper = _BatchLiar(net).eval()
        res = verify_refine_input_crown(S, wrapper, spec,
                                        layers=extract_layers(net),
                                        node_budget=5000)
        assert res.status != Status.SAT
        assert wrapper.lies > 0, "the batched check was never exercised"


# =========================================================================== #
# Exact references for the bound engine and the driver                        #
# =========================================================================== #
# A dyadic 2-3-3-2 ReLU network: weights and biases are multiples of 1/8 with
# both signs and magnitudes above and below one; the middle layer is square and
# not symmetric, so a transposed weight matrix describes another network.
# Float64 arithmetic on it and on dyadic boxes is exact, so the rational oracle
# is the reference with zero slack.
_DYADIC = (
    ([[1.5, -0.75], [-1.25, 2.0], [0.5, 1.0]], [0.25, -0.5, -1.0], True),
    ([[1.0, -2.0, 0.75], [-0.5, 1.25, 1.5], [2.0, 0.5, -1.0]],
     [0.5, -0.25, 0.125], True),
    ([[1.0, -1.5, 0.5], [-0.75, 0.25, 2.0]], [-0.5, 1.0], False),
)
# Input box of the driver cases: bounds beyond +-1, wider than 2, a different
# interval per coordinate (every predicate variable of a box star ranges over
# [-1, 1], so a box like this one shows a variable read in place of a point).
_ROOT_LB, _ROOT_UB = [-3.0, -1.5], [2.5, 4.0]
# Margin row of the driver cases: its exact minimum over the root box is
# -1743/128, attained only at (7/9, 4), a point that is not a float.
_ROW = [[1.0, 1.5]]


def _module(layers, relu_last=False):
    """The float64 ``nn.Sequential`` of ``layers`` = [(W, b, relu_after)]."""
    mods = []
    for W, b, relu_after in layers:
        lin = nn.Linear(len(W[0]), len(W)).double()
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(W, dtype=torch.float64))
            lin.bias.copy_(torch.tensor(b, dtype=torch.float64))
        mods.append(lin)
        if relu_after:
            mods.append(nn.ReLU())
    if relu_last:
        mods.append(nn.ReLU())
    return nn.Sequential(*mods)


def _engine_layers(layers):
    """The same network as the engine's own layer list (no torch module)."""
    out = []
    for W, b, relu_after in layers:
        out.append(LinearLayer(np.array(W, dtype=np.float64),
                               np.array(b, dtype=np.float64)))
        if relu_after:
            out.append(ReluLayer())
    return out


def _exact(layers, lb, ub, relu_last=False):
    """The exact pieces of the network on the box ``[lb, ub]`` (rational
    oracle) and, per ReLU layer, the exact range ``(min, max)`` of each of its
    inputs over the box."""
    pieces = Q.box(lb, ub)
    ranges = []
    for k, (W, b, relu_after) in enumerate(layers):
        pieces = Q.affine(pieces, W, b)
        if relu_after or (relu_last and k == len(layers) - 1):
            image = [v for p in pieces for v in p.image()]
            ranges.append([(min(v[j] for v in image), max(v[j] for v in image))
                           for j in range(len(W))])
            pieces = Q.relu(pieces)
    return pieces, ranges


def _row_minima(pieces, G, g):
    """The exact minimum of every margin row ``G_r y - g_r`` over the image:
    a linear function on a piece's image is smallest at one of its vertices."""
    image = [v for p in pieces for v in p.image()]
    return [min(Q.dot(Q.vector(row), v) for v in image) - Q.frac(gr)
            for row, gr in zip(G, g)]


def _assert_bounds_hold(cb, b, margin_lo, pieces, ranges, G, g):
    """Box ``b`` of the batch ``cb``: every margin lower bound is at or below
    the exact minimum and every pre-activation interval holds the exact range,
    in fractions with zero slack."""
    exact = _row_minima(pieces, G, g)
    for r, e in enumerate(exact):
        assert Fraction(float(margin_lo[b, r])) <= e, \
            f"box {b} row {r}: margin_lo {margin_lo[b, r]!r} above the " \
            f"exact minimum {float(e)!r}"
    assert len(cb.pre_bounds) == len(ranges), "one interval pair per ReLU layer"
    for k, ((lo, hi), layer) in enumerate(zip(cb.pre_bounds, ranges)):
        for j, (emin, emax) in enumerate(layer):
            assert Fraction(float(lo[b, j])) <= emin, \
                f"box {b} layer {k} neuron {j}: lower bound {float(lo[b, j])!r} " \
                f"above the exact minimum {float(emin)!r}"
            assert Fraction(float(hi[b, j])) >= emax, \
                f"box {b} layer {k} neuron {j}: upper bound {float(hi[b, j])!r} " \
                f"below the exact maximum {float(emax)!r}"


def _deep_chain(seed, depth=40):
    """Weights of a chain of ``depth`` one-wide Linear layers, chosen so that
    every product of the backward pass (seeded with G = -1) rounds toward
    zero: the computed form drifts above the exact one by about half an ulp
    per layer, which only a slack that counts every layer covers."""
    rng = np.random.default_rng(seed)
    ws, acc = [], -1.0
    for _ in range(depth):
        best = None
        for _ in range(400):
            w = rng.uniform(0.9, 1.1)
            p = acc * w
            drift = (Fraction(p) - Fraction(acc) * Fraction(w)) / abs(Fraction(p))
            if best is None or drift > best[0]:
                best = (drift, w, p)
        ws.append(best[1])
        acc = best[2]
    return ws[::-1]        # layer order: the backward pass starts at the last


def _rounding_instance(name):
    """``(layers, lb, ub, G, g)`` of a rounding-scale instance (float64 data
    that is not dyadic; the rational oracle reads every float exactly)."""
    if name == "one-sign-products":
        # point box in 5-D, no bias, every product w_j x_j < 0, g the float
        # nearest the exact output: the slack term of the offset is |g|-sized
        rng = np.random.default_rng(1)
        x = rng.uniform(0.3, 3.0, 5) * rng.choice([-1, 1], 5)
        w = -np.sign(x) * rng.uniform(0.3, 3.0, 5)
        y = sum(Fraction(wi) * Fraction(xi) for wi, xi in zip(w, x))
        return ([([list(w)], [0.0], False)], list(x), list(x),
                [[1.0]], [float(y)])
    if name == "cancelling-biases":
        # y = w2 relu(w1 x + b1) + b2 with b2 almost cancelling w2 b1, |x|
        # small: the rounding comes from the bias chain
        x = 0.0009436088293560713
        return ([([[1.232747859625954]], [1.256021723602395], True),
                 ([[1.4304086125753435]], [-1.7966242910225934], False)],
                [x], [x], [[1.0]], [0.001663896734666402])
    if name == "crossing-behind-cancellation":
        # z = w (x - x0) crosses zero with u < -l (lower slope 0) after a
        # cancelling affine map; y = -c relu(z)
        rng = np.random.default_rng(0)
        w, x0 = rng.uniform(500, 2000), rng.uniform(0.5, 2.0)
        u = rng.uniform(0.01, 0.1)
        l = -rng.uniform(1.2, 3.0) * u
        c = rng.uniform(0.5, 2.0)
        return ([([[w]], [-w * x0], True), ([[-c]], [0.0], False)],
                [x0 + l / w], [x0 + u / w], [[1.0]], [0.0])
    if name == "deep-chain":
        ws = _deep_chain(0)
        y = Fraction(1)
        for wi in ws:
            y *= Fraction(wi)
        return ([([[wi]], [0.0], False) for wi in ws], [1.0], [1.0],
                [[-1.0]], [-float(y)])
    raise ValueError(name)


# =========================================================================== #
# The bound engine: CrownBatch and its helpers                                #
# =========================================================================== #
_G3 = [[1.0, -1.0], [-2.0, 0.5], [0.75, 1.25]]
_g3 = [1.5, -2.0, 0.25]          # both signs; g > 0 shows an offset added
_BATCHES = {
    "one-box": ([[-3.0, -1.5]], [[2.5, 4.0]]),
    # boxes of different widths and positions, one a square
    "many-boxes": ([[-3.0, -1.5], [-0.5, 0.25], [1.0, -1.0], [0.125, 0.5],
                    [-1.0, -1.0]],
                   [[2.5, 4.0], [0.75, 0.5], [1.25, 3.5], [0.25, 0.625],
                    [1.0, 1.0]]),
    # two points and a box with one zero-width coordinate
    "zero-width": ([[0.5, -0.25], [-1.0, 0.75], [1.25, 1.25]],
                   [[0.5, -0.25], [-1.0, 2.0], [1.25, 1.25]]),
    # every ReLU stable on both boxes, active and inactive ones among them
    "all-stable": ([[-2.0, 2.0], [0.5, -0.25]], [[-1.5, 2.25], [0.5, -0.25]]),
}


class TestCrownBatchBounds:
    """``CrownBatch`` against exact minima and ranges in fractions."""

    @pytest.mark.parametrize("model,batch", [
        ("affine-last", "one-box"), ("affine-last", "many-boxes"),
        ("affine-last", "zero-width"), ("affine-last", "all-stable"),
        ("relu-last", "many-boxes")])
    def test_margins_and_pre_bounds_hold_the_exact_extremes(self, model, batch):
        """Every ``margin_lo[b, r]`` is at or below the exact minimum of row r
        over box b, and every ``pre_bounds`` interval holds the exact range of
        its neuron over box b, for each box of a batch. Catches a box
        concretized at its maximum, a spec offset added, a chord intercept of
        the wrong sign, a lower or upper form taking the wrong side of a
        relaxation (the upper forms feed the pre-activation bounds of the
        second ReLU layer), an upper pre-activation bound concretized at the
        minimum, a transposed square weight matrix, and a stable neuron
        relaxed with the wrong slope (the all-stable batch, where the bound
        is the exact minimum minus the rounding slack)."""
        relu_last = model == "relu-last"
        LB, UB = (np.array(a, dtype=np.float64) for a in _BATCHES[batch])
        cb = CrownBatch(prepare_layers(extract_layers(
            _module(_DYADIC, relu_last))), LB, UB)
        margin_lo, _, _ = cb.margin_report(np.array(_G3), np.array(_g3))
        assert margin_lo.shape == (LB.shape[0], len(_G3))
        for b in range(LB.shape[0]):
            pieces, ranges = _exact(_DYADIC, LB[b], UB[b], relu_last)
            if batch == "all-stable" and any(
                    lo < 0 < hi for layer in ranges for lo, hi in layer):
                pytest.fail(f"setup: a ReLU crosses zero on box {b}")
            _assert_bounds_hold(cb, b, margin_lo, pieces, ranges, _G3, _g3)

    @pytest.mark.parametrize("instance", [
        "one-sign-products", "cancelling-biases",
        "crossing-behind-cancellation", "deep-chain"])
    def test_margins_and_pre_bounds_hold_at_rounding_scale(self, instance):
        """The same promise on instances where the float64 backward pass
        rounds by more than the exact gap: the rounding slack must pay for it
        (``_rounding_slack`` with the magnitude chain, the bias chain mb, the
        |g| term of the offset, the operation count of ``_n_ops``; the
        pre-activation bounds widened by their own slack). Catches the slack
        left out of the margins or of the pre-activation bounds, the bias
        chain left out, the offset term taken with its sign, and an operation
        count that leaves out the layers (the deep chain)."""
        layers, lb, ub, G, g = _rounding_instance(instance)
        cb = CrownBatch(prepare_layers(_engine_layers(layers)),
                        np.array([lb]), np.array([ub]))
        margin_lo, _, _ = cb.margin_report(np.array(G), np.array(g))
        pieces, ranges = _exact(layers, lb, ub)
        _assert_bounds_hold(cb, 0, margin_lo, pieces, ranges, G, g)

    def test_relu_relaxation_contains_relu_in_exact_arithmetic(self):
        """``_relu_relax`` keeps ``sl z <= relu(z) <= su z + bu`` on [l, u]
        for the stored (rounded) coefficients, checked in fractions at l, 0
        and u (both sides are linear between them). Crossing neurons with
        u >= -l and u < -l, stable active and inactive ones, l = u, l = 0 and
        u = 0, non-dyadic bounds of magnitude up to 30. Catches the outward
        pad of the chord intercept left out (the rounded chord dips an ulp
        below relu at an end), the intercept sign and a stable slope taken
        from the chord formula; through the driver the backward slack hides
        a missing pad, so this is the case that sees it."""
        rng = np.random.default_rng(0)
        l = np.concatenate([-rng.uniform(0.01, 30.0, 40),
                            [0.0, -1.0, 0.3, -0.7, -2.5, 0.0, -0.1]])
        u = np.concatenate([rng.uniform(0.01, 30.0, 40),
                            [0.7, 0.0, 0.9, -0.2, -2.5, 0.0, 3.3]])
        sl, su, bu = (t.numpy() for t in _relu_relax(torch.tensor(l),
                                                     torch.tensor(u)))
        for i in range(l.size):
            for z in {l[i], u[i], 0.0}:
                if not l[i] <= z <= u[i]:
                    continue
                zf = Fraction(z)
                relu = max(zf, Fraction(0))
                assert Fraction(sl[i]) * zf <= relu, (l[i], u[i], z, "lower")
                assert relu <= Fraction(su[i]) * zf + Fraction(bu[i]), \
                    (l[i], u[i], z, "upper")

    def test_magnitude_chain_dominates_the_forms(self):
        """``_backward`` returns with the forms the magnitude chain (the same
        pass on absolute values), which the rounding slack multiplies by
        gamma; it must dominate every form it bounds: ``|Al|, |Au| <= M`` and
        ``|bl|, |bu| <= mb`` entry by entry (the same operations on absolute
        values, rounded to nearest, never come out smaller). Random nets with
        crossing neurons of both kinds, five boxes each. Catches a magnitude
        chain scaled by the lower slope only, which zeroes the magnitude of
        every path through a crossing neuron with lower slope 0; through the
        driver the padded pre-activation bounds of that neuron hide it."""
        for seed in range(10):
            rng = np.random.default_rng(seed)
            dims = [2, 4, 4, 3]
            layers = []
            for i in range(3):
                layers.append(LinearLayer(rng.normal(size=(dims[i + 1], dims[i])),
                                          rng.normal(size=dims[i + 1])))
                if i < 2:
                    layers.append(ReluLayer())
            prep = prepare_layers(layers)
            LB = rng.uniform(-2.0, 0.0, size=(5, 2))
            UB = LB + rng.uniform(0.1, 2.0, size=(5, 2))
            cb = CrownBatch(prep, LB, UB)
            C = torch.tensor(rng.normal(size=(2, 3))).expand(5, 2, 3)
            Al, bl, Au, bu, M, mb = _backward(prep, len(prep), cb._relax, C)
            assert torch.all(Al.abs() <= M) and torch.all(Au.abs() <= M), seed
            assert torch.all(bl.abs() <= mb) and torch.all(bu.abs() <= mb), seed

    @pytest.mark.parametrize("row_rule", ["argmax", "argmin"])
    def test_vertices_and_split_scores_follow_the_driving_row(self, row_rule):
        """On an affine network the lower forms are exact (``A_r = G_r W``),
        so the driving row (the row with the largest exact minimum for
        'argmax', the smallest for 'argmin'), its minimizing vertex (``lb_d``
        where ``A_d >= 0``, else ``ub_d``) and the split scores
        ``|A_d| (ub_d - lb_d)`` are known in fractions. Two boxes whose
        coordinates have different widths, rows whose vertices and scores
        differ. Catches the vertex taken at the maximizing corner, the score
        without the width, and the two row rules swapped."""
        W, b = [[1.5, -0.5], [0.25, 2.0]], [0.5, -1.0]
        G, g = [[1.0, 0.0], [0.0, -1.0]], [-2.0, 1.0]
        LB = np.array([[-3.0, -0.5], [0.5, -2.0]])
        UB = np.array([[1.0, 0.25], [0.625, 3.0]])
        net = _module([(W, b, False)])
        cb = CrownBatch(prepare_layers(extract_layers(net)), LB, UB)
        margin_lo, vertices, smear = cb.margin_report(np.array(G), np.array(g),
                                                      row_rule=row_rule)
        A = [[Q.dot(Q.vector(Gr), col) for col in zip(*Q.matrix(W))] for Gr in G]
        for k in range(2):
            pieces, _ = _exact([(W, b, False)], LB[k], UB[k])
            exact = _row_minima(pieces, G, g)
            if abs(exact[0] - exact[1]) < 1:
                pytest.fail("setup: the two rows' minima must be well apart")
            for r in range(2):
                assert Fraction(float(margin_lo[k, r])) <= exact[r]
            pick = max if row_rule == "argmax" else min
            r = pick(range(2), key=lambda i: exact[i])
            want_v = [LB[k, d] if A[r][d] >= 0 else UB[k, d] for d in range(2)]
            want_s = [abs(A[r][d]) * (Fraction(UB[k, d]) - Fraction(LB[k, d]))
                      for d in range(2)]
            assert list(vertices[k]) == want_v, (k, row_rule, vertices[k])
            assert [Fraction(float(s)) for s in smear[k]] == want_s, \
                (k, row_rule, smear[k])


# =========================================================================== #
# The driver: verify_refine_input_crown                                       #
# =========================================================================== #
_ROOT_MIN = []          # exact minimum of _ROW over the root box (computed once)


def _root_minimum():
    if not _ROOT_MIN:
        pieces, _ = _exact(_DYADIC, _ROOT_LB, _ROOT_UB)
        _ROOT_MIN.append(_row_minima(pieces, _ROW, [0.0])[0])
    return _ROOT_MIN[0]


def _root_star():
    return Star.from_bounds(np.array(_ROOT_LB), np.array(_ROOT_UB))


def _record_batches(monkeypatch, clock=None):
    """Replace ``CrownBatch`` (read by the driver at call time) by a subclass
    that records every wave: its boxes and its ``margin_report`` output. With
    ``clock``, one second of the driver's clock passes per wave."""
    import n2v.refine.crown_bounds as crown_bounds
    real = crown_bounds.CrownBatch
    waves = []

    class _Recording(real):
        def __init__(self, prepared, lb, ub, device="cpu"):
            super().__init__(prepared, lb, ub, device=device)
            self._wave = {"lb": np.array(lb, dtype=np.float64),
                          "ub": np.array(ub, dtype=np.float64)}
            waves.append(self._wave)
            if clock is not None:
                clock.now += 1.0

        def margin_report(self, G, g, row_rule="argmax"):
            out = super().margin_report(G, g, row_rule=row_rule)
            self._wave["margin_lo"] = out[0].copy()
            return out

    monkeypatch.setattr(crown_bounds, "CrownBatch", _Recording)
    return waves


def _depth(lo, hi, root_lo, root_hi):
    """Bisections between the root box and ``[lo, hi]``: every coordinate's
    width is the root's divided by a power of two."""
    depth = 0
    for a, b, ra, rb in zip(lo, hi, root_lo, root_hi):
        ratio = (Fraction(rb) - Fraction(ra)) / (Fraction(b) - Fraction(a))
        assert ratio.denominator == 1 and ratio.numerator & (ratio.numerator - 1) == 0, \
            f"box [{lo}, {hi}] is not a bisection of the root"
        depth += ratio.numerator.bit_length() - 1
    return depth


def _assert_real_witness(res, layers, G, g, lb, ub, model):
    """A SAT witness lies in the user's box at zero tolerance, and both its
    exact output (rational oracle) and its float64 output lie in the closed
    unsafe region ``G y <= g``."""
    assert res.counterexample_x is not None, "SAT without a counterexample"
    x = np.asarray(res.counterexample_x, dtype=np.float64).ravel()
    assert np.all(x >= np.asarray(lb)) and np.all(x <= np.asarray(ub)), \
        f"witness {x!r} outside the input box"
    pieces, _ = _exact(layers, lb, ub)
    y = Q.evaluate(pieces, x)
    for row, gr in zip(G, g):
        assert Q.dot(Q.vector(row), y) <= Q.frac(gr), \
            f"exact output of {x!r} is not in the unsafe region"
    with torch.no_grad():
        yf = model(torch.as_tensor(x).unsqueeze(0)).numpy().ravel()
    assert np.all(np.asarray(G) @ yf - np.asarray(g) <= 0.0), \
        f"float64 output of {x!r} is not in the unsafe region"


def _identity(n):
    net = nn.Sequential(nn.Linear(n, n)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.eye(n, dtype=torch.float64))
        net[0].bias.zero_()
    return net


class _WaveClock:
    """``perf_counter`` for the driver's clock; ``_record_batches`` advances
    it by one second per wave."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now


class TestInputSplitEngine:
    """``verify_refine_input_crown`` against exact references."""

    def test_unsat_is_backed_by_proved_boxes_that_tile_the_input_box(
            self, monkeypatch):
        """Safe by a small margin: the exact minimum of y0 + 1.5 y1 over the
        box is m*, the unsafe region is y0 + 1.5 y1 <= m* - 1/32. UNSAT, and
        the record of every wave shows why: the boxes with some
        ``margin_lo > 0`` lie in the input box, have disjoint interiors and
        their exact volumes add up to the input box's (no hole), and each of
        their pruning rows is a true bound (at or below the exact minimum over
        that box). ``nodes`` counts the boxes bounded and ``max_depth`` is the
        depth of the deepest one. Catches a pending counter that ends the
        search with boxes unexplored, children that do not cover their
        parent, a budget exit read as UNSAT, and wrong counters."""
        waves = _record_batches(monkeypatch)
        net = _module(_DYADIC)
        g = _root_minimum() - Fraction(1, 32)
        res = verify_refine_input_crown(_root_star(), net,
                                        LinearSpec(_ROW, [float(g)]),
                                        node_budget=20_000)
        assert res.status == Status.UNSAT, res.status
        boxes = [(lo, hi, m) for w in waves
                 for lo, hi, m in zip(w["lb"], w["ub"], w["margin_lo"])]
        assert res.nodes == len(boxes)
        assert res.max_depth == max(_depth(lo, hi, _ROOT_LB, _ROOT_UB)
                                    for lo, hi, _ in boxes)
        proved = [(lo, hi, m) for lo, hi, m in boxes if (m > 0).any()]
        L = np.array([p[0] for p in proved])
        U = np.array([p[1] for p in proved])
        assert np.all(L >= _ROOT_LB) and np.all(U <= _ROOT_UB)
        overlap = np.all(np.minimum(U[:, None], U[None]) >
                         np.maximum(L[:, None], L[None]), axis=2)
        np.fill_diagonal(overlap, False)
        assert not overlap.any(), "two proved boxes overlap"
        volume = sum((Fraction(u0) - Fraction(l0)) * (Fraction(u1) - Fraction(l1))
                     for (l0, l1), (u0, u1) in zip(L, U))
        assert volume == ((Fraction(_ROOT_UB[0]) - Fraction(_ROOT_LB[0]))
                          * (Fraction(_ROOT_UB[1]) - Fraction(_ROOT_LB[1]))), \
            "the proved boxes do not tile the input box"
        for lo, hi, m in proved:
            pieces, _ = _exact(_DYADIC, lo, hi)
            exact = _row_minima(pieces, _ROW, [float(g)])[0]
            assert Fraction(float(m[0])) <= exact, (lo, hi, m)

    def test_sat_near_the_minimizer_has_a_real_witness(self):
        """Unsafe in a small region: y0 + 1.5 y1 <= m* + 1/32 holds only
        near (7/9, 4). SAT, with a witness in the box whose exact and float64
        outputs are unsafe. Catches every bound that is too high (a false
        UNSAT here) and witnesses that are not counterexamples."""
        net = _module(_DYADIC)
        g = float(_root_minimum() + Fraction(1, 32))
        res = verify_refine_input_crown(_root_star(), net, LinearSpec(_ROW, [g]),
                                        node_budget=20_000)
        assert res.status == Status.SAT, res.status
        _assert_real_witness(res, _DYADIC, _ROW, [g], _ROOT_LB, _ROOT_UB, net)

    @pytest.mark.parametrize("stop", ["node-budget", "slivers"])
    def test_single_non_float_counterexample_is_never_unsat(self, stop):
        """Unsafe in exactly one point: y0 + 1.5 y1 <= m* holds only at
        (7/9, 4), which no float reaches, so the search can neither prove the
        box nor find a witness: UNKNOWN, by the node budget (40 boxes,
        exactly) or by boxes narrower than ``min_width`` (2^-6). Catches a
        budget exit answered UNSAT and a wave of slivers dropped without
        marking the run inconclusive."""
        net = _module(_DYADIC)
        g = float(_root_minimum())
        kw = (dict(node_budget=40) if stop == "node-budget"
              else dict(min_width=2.0 ** -6))
        res = verify_refine_input_crown(_root_star(), net, LinearSpec(_ROW, [g]),
                                        **kw)
        assert res.status == Status.UNKNOWN, res.status
        if stop == "node-budget":
            assert res.nodes == 40

    def test_tie_at_a_single_dyadic_point_is_sat(self):
        """Unsafe in exactly one point that bisection reaches:
        y = -relu(x - 1/4) - relu(1/4 - x) on [-1, 1], unsafe y >= 0, only at
        x = 1/4 (margin exactly 0 there). SAT with x = 1/4. Catches a candidate
        filter or a confirmation that reads the closed unsafe region as open."""
        layers = [([[1.0], [-1.0]], [-0.25, 0.25], True), ([[-1.0, -1.0]], [0.0], False)]
        net = _module(layers)
        res = verify_refine_input_crown(Star.from_bounds(np.array([-1.0]),
                                                         np.array([1.0])),
                                        net, LinearSpec([[-1.0]], [0.0]),
                                        node_budget=1000)
        assert res.status == Status.SAT, res.status
        _assert_real_witness(res, layers, [[-1.0]], [0.0], [-1.0], [1.0], net)
        assert list(res.counterexample_x) == [0.25]

    def test_constant_output_on_the_unsafe_boundary_is_sat(self):
        """Every hidden neuron is dead on the box and the output bias is 0, so
        y = (0, 0) everywhere and the margin bound is exactly 0 (no rounding
        slack either). The unsafe region y0 - y1 <= 0 is closed, so every
        input is a counterexample: SAT at the root box. Catches a prune on
        ``margin_lo >= 0``."""
        layers = [([[1.0, 0.0], [0.0, 1.0]], [-5.0, -5.0], True),
                  ([[1.0, 2.0], [3.0, 4.0]], [0.0, 0.0], False)]
        net = _module(layers)
        res = verify_refine_input_crown(Star.from_bounds(np.zeros(2), np.ones(2)),
                                        net, LinearSpec([[1.0, -1.0]], [0.0]),
                                        node_budget=100)
        assert res.status == Status.SAT and res.nodes == 1, (res.status, res.nodes)
        _assert_real_witness(res, layers, [[1.0, -1.0]], [0.0], [0.0, 0.0],
                             [1.0, 1.0], net)

    @pytest.mark.parametrize("instance", [
        "cancelling-biases", "deep-chain",
        pytest.param("subnormal-products", marks=known_bug(
            19, "the CROWN rounding slack has no absolute term for products "
                "that underflow into the subnormal range: margin_lo = 5e-324 "
                "> 0 where the exact margin is -2^-1077, a false UNSAT")),
        "normal-range-products"])
    def test_counterexample_at_rounding_scale_is_never_unsat(self, instance):
        """A point input whose exact output lies in the unsafe region by less
        than the float64 rounding of the bound computation: the driver must
        not prove it safe (UNKNOWN: a point box has no candidate vertex).
        Cancelling biases: the bias chain of the slack. Deep chain (40
        one-wide layers whose backward products all round the same way): the
        operation count of the slack. Subnormal products: three products
        w_j x_j of about 2^-1074 that round to nearest in the subnormal range
        (exact output -2^-1077); the same instance scaled into the normal
        range passes."""
        if instance == "cancelling-biases":
            layers, lb, _, G, g = _rounding_instance("cancelling-biases")
        elif instance == "deep-chain":
            ws = _deep_chain(3)
            y = Fraction(1)
            for wi in ws:
                y *= Fraction(wi)
            layers = [([[wi]], [0.0], False) for wi in ws]
            lb, G, g = [1.0], [[-1.0]], [-float(y)]
        else:
            scale = 2.0 ** -537 if instance == "subnormal-products" else 2.0 ** 60
            layers = [([[1.625 * scale, 1.625 * scale, -3.375 * scale]], [0.0], False)]
            lb, G, g = [scale] * 3, [[1.0]], [0.0]
        pieces, _ = _exact(layers, lb, lb)
        if not _row_minima(pieces, G, g)[0] <= 0:
            pytest.fail("setup: the point must be a counterexample")
        res = verify_refine_input_crown(Star.from_bounds(np.array(lb), np.array(lb)),
                                        _module(layers), LinearSpec(G, g),
                                        node_budget=50)
        assert res.status != Status.UNSAT

    def test_corner_counterexample_of_a_non_dyadic_box_is_found_at_the_root(self):
        """Box [0.1, 0.3] x [-2.5, 1.7] (half-widths not 1, bounds that do not
        survive the centre and radius round trip), identity network, unsafe
        y0 >= 0.3: only the corner x0 = 0.3 is a counterexample. The root's
        candidate vertex, mapped to predicate coordinates (divided by the
        generator), clipped into the predicate box and snapped into the
        user's box, is that corner: SAT at the first box with x0 = 0.3 exactly.
        Catches the vertex multiplied by the generator, the clip into the
        predicate box left out (the box rows then reject the corner) and a
        witness left one ulp outside the user's box (0.30000000000000004)."""
        lb, ub = [0.1, -2.5], [0.3, 1.7]
        res = verify_refine_input_crown(Star.from_bounds(np.array(lb), np.array(ub)),
                                        _identity(2), LinearSpec([[-1.0, 0.0]], [-0.3]),
                                        node_budget=1000)
        assert res.status == Status.SAT and res.nodes == 1, (res.status, res.nodes)
        _assert_real_witness(res, [([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], False)],
                             [[-1.0, 0.0]], [-0.3], lb, ub, _identity(2))
        assert res.counterexample_x[0] == 0.3

    def test_rebuilt_corner_outside_the_user_box_is_never_a_witness(self):
        """Same box, unsafe y0 >= 0.30000000000000004: no point of the user's
        box is unsafe, but the corner rebuilt from predicate coordinates is
        0.30000000000000004 and passes the batched check. The re-check must
        run on the point that is returned (snapped into the box, where it is
        safe): never SAT. Catches the re-check run on the unsnapped point
        while the snapped one is returned, and the unsnapped point returned."""
        lb, ub = [0.1, -2.5], [0.3, 1.7]
        g = -0.30000000000000004
        res = verify_refine_input_crown(Star.from_bounds(np.array(lb), np.array(ub)),
                                        _identity(2), LinearSpec([[-1.0, 0.0]], [g]),
                                        node_budget=300)
        assert res.status != Status.SAT, res.counterexample_x

    def test_corner_cut_off_by_a_tiny_margin_is_never_a_witness(self):
        """The box [-1, 1]^2 cut by alpha0 + alpha1 <= 2 - 2^-40: the corner
        (1, 1), the only point with y0 + y1 >= 2, is outside the input set by
        2^-40. The candidate's constraint rows are checked at zero tolerance:
        never SAT (truly safe). Catches a tolerance on the row check and the
        row check left out."""
        S = Star.from_bounds(-np.ones(2), np.ones(2))
        S = Star(S.V, np.vstack([S.C, [[1.0, 1.0]]]),
                 np.vstack([S.d, [[2.0 - 2.0 ** -40]]]), S.predicate_lb,
                 S.predicate_ub)
        res = verify_refine_input_crown(S, _identity(2),
                                        LinearSpec([[-1.0, -1.0]], [-2.0]),
                                        node_budget=300)
        assert res.status != Status.SAT, res.counterexample_x

    def test_sheared_input_star_never_yields_an_outside_witness(self):
        """A box sheared by x = (a0 + a1, a1): the input map is square but not
        diagonal, so vertices cannot be mapped back by a division and no
        candidate is tried. Identity network, unsafe y0 - y1 >= 1.5; on the
        set y0 - y1 = a0 <= 1, so it is truly safe while corners of the
        bounding box such as (2, -1) are unsafe. Never SAT. Catches the
        diagonal check left out (the corner is rebuilt with the diagonal alone
        and returned)."""
        S = Star.from_bounds(-np.ones(2), np.ones(2)).affine_map(
            np.array([[1.0, 1.0], [0.0, 1.0]]))
        res = verify_refine_input_crown(S, _identity(2),
                                        LinearSpec([[-1.0, 1.0]], [-1.5]),
                                        node_budget=300)
        assert res.status != Status.SAT, res.counterexample_x

    def test_training_mode_model_is_checked_in_eval_mode(self):
        """A model handed over in training mode with Dropout(0.1): the eval
        function y = relu(x0 + x1) + relu(x0 - x1) has maximum 2 on [-1, 1]^2,
        so y >= 2.05 is truly safe, while a training-mode forward pass scales
        each kept unit by 10/9 (2.22 at the corner (1, -1) whenever the unit
        is kept, nine times in ten). Never SAT, and the caller's model stays
        in training mode. Catches the counterexample checks run on the
        training-mode model."""
        m = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Dropout(0.1),
                          nn.Linear(2, 1)).double()
        with torch.no_grad():
            m[0].weight.copy_(torch.tensor([[1.0, 1.0], [1.0, -1.0]],
                                           dtype=torch.float64))
            m[0].bias.zero_()
            m[3].weight.copy_(torch.tensor([[1.0, 1.0]], dtype=torch.float64))
            m[3].bias.zero_()
        m.train()
        res = verify_refine_input_crown(Star.from_bounds(-np.ones(2), np.ones(2)),
                                        m, LinearSpec([[-1.0]], [-2.05]),
                                        node_budget=5000)
        assert res.status != Status.SAT, res.counterexample_x
        assert m.training

    @pytest.mark.parametrize("input_set", ["unbounded", "empty"])
    def test_input_set_that_decides_the_run(self, input_set):
        """Before any search: an unbounded input set (x = alpha, alpha free,
        no rows; x = 10 gives y = 10 >= 5) cannot be finitized and is UNKNOWN,
        never UNSAT; an input set whose rows contradict each other
        (alpha <= -1/2 and alpha >= 1/2, infinite predicate box) is confirmed
        empty and is UNSAT. Both with no box bounded. Catches every early
        decision of the finitizer answered UNSAT."""
        V = np.array([[0.0, 1.0]])
        inf = np.array([[np.inf]])
        if input_set == "unbounded":
            S = Star(V, np.zeros((0, 1)), np.zeros((0, 1)), -inf, inf)
            want = Status.UNKNOWN
        else:
            S = Star(V, np.array([[1.0], [-1.0]]), np.array([[-0.5], [-0.5]]),
                     -inf, inf)
            want = Status.UNSAT
        res = verify_refine_input_crown(S, _identity(1),
                                        LinearSpec([[-1.0]], [-5.0]))
        assert (res.status, res.nodes) == (want, 0)

    @pytest.mark.parametrize("row_rule", ["argmax", "argmin"])
    def test_row_rule_picks_the_candidate_at_the_root(self, row_rule):
        """Identity network on [-1, 1]^2 and a two-row unsafe region whose
        only counterexample at the root's corners is the minimizing vertex of
        the row the rule picks ('argmax': the row with the larger margin
        bound; 'argmin': the smaller), while the other row's vertex is not a
        counterexample. SAT at the first box with that vertex. Catches
        ``row_rule`` not passed on to ``margin_report`` and the rules swapped."""
        if row_rule == "argmax":
            # x0 - x1 <= -2 (bound 0, vertex (-1, 1)); x0 + x1 <= 0 (bound -2)
            G, g, want = [[1.0, -1.0], [1.0, 1.0]], [-2.0, 0.0], [-1.0, 1.0]
        else:
            # -x0 <= -1/2 (bound -1/2, vertex (1, -1)); x0/8 + x1/4 <= 0
            # (bound -3/8, vertex (-1, -1), where the first row fails)
            G, g, want = [[-1.0, 0.0], [0.125, 0.25]], [-0.5, 0.0], [1.0, -1.0]
        res = verify_refine_input_crown(Star.from_bounds(-np.ones(2), np.ones(2)),
                                        _identity(2), LinearSpec(G, g),
                                        node_budget=1000, row_rule=row_rule)
        assert res.status == Status.SAT and res.nodes == 1, (res.status, res.nodes)
        assert list(res.counterexample_x) == want

    def test_first_split_halves_the_dimension_of_largest_score(self, monkeypatch):
        """Identity network on [-4, 4] x [-1/2, 1/2], unsafe -x0 <= -1/2 and
        x0/8 + x1/4 <= 0; the root is undecided (the driving row is the second,
        whose vertex (-4, -1/2) fails the first). Its split scores are
        |A_d| times the width: (1/8 * 8, 1/4 * 1) = (1, 1/4), so the root is
        halved in x0 at 0. Catches a score without the width (which picks
        x1)."""
        waves = _record_batches(monkeypatch)
        lb, ub = [-4.0, -0.5], [4.0, 0.5]
        verify_refine_input_crown(Star.from_bounds(np.array(lb), np.array(ub)),
                                  _identity(2),
                                  LinearSpec([[-1.0, 0.0], [0.125, 0.25]],
                                             [-0.5, 0.0]),
                                  node_budget=3)
        assert len(waves) >= 2 and waves[1]["lb"].shape[0] == 2, \
            "setup: the root must be split"
        children = sorted((tuple(lo), tuple(hi)) for lo, hi in
                          zip(waves[1]["lb"], waves[1]["ub"]))
        assert children == [((-4.0, -0.5), (0.0, 0.5)), ((0.0, -0.5), (4.0, 0.5))]

    @pytest.mark.parametrize("case", ["budget-5", "wave-3-budget-30"])
    def test_node_budget_is_exact(self, case):
        """``node_budget`` is exact: waves shrink to the remaining budget, also
        when a wave gathers boxes from two frontier chunks. Two searches that
        cannot finish: the box [0, 1]^2 cut by x0 + x1 <= 1 with unsafe
        y0 + y1 >= 1.5 (bounded with the default wave and budget 5: waves of
        1, 2 and 2), and 1 - 10 |x0 - 0.33| - 10 |x1 - 0.7| >= 1.01 on
        [0, 1]^2 with wave 3 and budget 30. UNKNOWN after exactly the budget.
        Catches a wave not capped by the remaining budget and a partial take
        from an older chunk that keeps back too few boxes."""
        if case == "budget-5":
            S = Star.from_bounds(np.zeros(2), np.ones(2)).intersect_half_space(
                np.array([[1.0, 1.0]]), np.array([[1.0]]))
            net, spec, kw = _identity(2), LinearSpec([[-1.0, -1.0]], [-1.5]), {}
            budget = 5
        else:
            p0, p1 = 0.33, 0.7
            net = _module([([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                            [-p0, p0, -p1, p1], True),
                           ([[-10.0, -10.0, -10.0, -10.0]], [1.0], False)])
            S = Star.from_bounds(np.zeros(2), np.ones(2))
            spec, kw, budget = LinearSpec([[-1.0]], [-1.01]), dict(wave=3), 30
        res = verify_refine_input_crown(S, net, spec, node_budget=budget, **kw)
        assert (res.status, res.nodes) == (Status.UNKNOWN, budget)

    def test_time_budget_stops_after_the_wave_that_exhausts_it(self, monkeypatch):
        """``time_budget`` is checked per wave and the clock runs from the
        start of the call. On a driver clock that advances one second per
        wave, a 2.5 s budget stops the search (UNKNOWN) after the third wave,
        on a search (the single non-float counterexample) that needs many
        more. Catches the start time taken anew at every wave."""
        net = _module(_DYADIC)
        spec = LinearSpec(_ROW, [float(_root_minimum())])
        waves = _record_batches(monkeypatch)
        verify_refine_input_crown(_root_star(), net, spec, node_budget=400)
        if len(waves) <= 3:
            pytest.fail("setup: the search must need more than three waves")
        clock = _WaveClock()
        monkeypatch.setattr(sys.modules["n2v.refine.input_split"], "time", clock)
        waves = _record_batches(monkeypatch, clock)
        res = verify_refine_input_crown(_root_star(), net, spec,
                                        node_budget=400, time_budget=2.5)
        assert res.status == Status.UNKNOWN and len(waves) == 3, \
            (res.status, len(waves))

    @pytest.mark.parametrize("case", ["conv-net", "relu-first", "relu-after-relu",
                                      "float32-model"])
    def test_unsupported_models_and_options_are_refused(self, case):
        """Refusals, by type: a conv layer (``TypeError`` from
        ``prepare_layers``: the bound engine takes Linear and ReLU only), a
        ReLU that does not follow a Linear (``ValueError``: the engine reads
        the preceding Linear's width) and a float32 model (``TypeError``).
        Catches the structural check of ``prepare_layers`` left out or an
        unknown layer skipped (the model is then bounded as another
        function), and the float64 check left out."""
        S = Star.from_bounds(-np.ones(2), np.ones(2))
        spec = LinearSpec([[1.0]], [-1e9])
        if case == "conv-net":
            net = nn.Sequential(nn.Unflatten(1, (1, 1, 2)), nn.Conv2d(1, 1, (1, 1)),
                                nn.Flatten(), nn.Linear(2, 1)).double()
            exc = TypeError
        elif case == "relu-first":
            net, exc = nn.Sequential(nn.ReLU(), nn.Linear(2, 1)).double(), ValueError
        elif case == "relu-after-relu":
            net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.ReLU(),
                                nn.Linear(2, 1)).double()
            exc = ValueError
        else:
            net, exc = seq_net(2, 3, 1, seed=0).float(), TypeError
        with pytest.raises(exc):
            verify_refine_input_crown(S, net, spec)

    @known_bug(19, "an unknown row_rule string (a typo such as 'ARGMAX') is "
                   "accepted and silently runs the 'argmin' rule",
               raises=DID_NOT_RAISE)
    def test_unknown_row_rule_is_refused(self):
        """``row_rule`` has two values; any other string must raise
        ``ValueError``, not select one of them silently."""
        with pytest.raises(ValueError):
            verify_refine_input_crown(Star.from_bounds(-np.ones(2), np.ones(2)),
                                      seq_net(2, 3, 1, seed=0),
                                      LinearSpec([[1.0]], [-1e9]),
                                      row_rule="ARGMAX")

    @known_bug(23, "an unknown device string reaches torch, which raises "
                   "RuntimeError instead of a ValueError refusal",
               raises=RuntimeError)
    def test_unknown_device_is_refused(self):
        with pytest.raises(ValueError):
            verify_refine_input_crown(Star.from_bounds(-np.ones(2), np.ones(2)),
                                      seq_net(2, 3, 1, seed=0),
                                      LinearSpec([[1.0]], [-1e9]),
                                      device="no-such-device")

    @pytest.mark.parametrize("model", [
        "identity-with-parameters",
        pytest.param("identity-without-parameters", marks=known_bug(
            19, "the driver reads the model device from next(model.parameters()) "
                "and raises StopIteration on a parameterless chain (and "
                "AttributeError on a plain callable with layers=), where "
                "verify_refine answers", raises=StopIteration))])
    def test_identity_chain_is_answered(self, model):
        """An identity network, as a Linear layer with identity weights or as
        a chain with no parameter at all (``nn.Flatten`` alone, which
        ``extract_layers`` accepts): y0 <= 1/2 is reached on [-1, 1]^2 at the
        corner (-1, -1), so the answer is SAT with a real witness."""
        net = (_identity(2) if model == "identity-with-parameters"
               else nn.Sequential(nn.Flatten()).double())
        res = verify_refine_input_crown(Star.from_bounds(-np.ones(2), np.ones(2)),
                                        net, LinearSpec([[1.0, 0.0]], [0.5]),
                                        node_budget=100)
        assert res.status == Status.SAT
        _assert_real_witness(res, [([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], False)],
                             [[1.0, 0.0]], [0.5], [-1.0, -1.0], [1.0, 1.0], net)

    @pytest.mark.parametrize("centre", [
        "dyadic",
        pytest.param("not-dyadic", marks=known_bug(
            19, "on an input star without raw bounds the SAT witness is the "
                "corner rebuilt as fl(c + r) and not checked against the set: "
                "with c = 0.2, r = 0.1 it is 0.30000000000000004, one ulp "
                "outside the set, and a truly safe problem is answered SAT"))])
    def test_witness_lies_in_a_directly_built_input_star(self, centre):
        """A star built directly as c + r alpha, alpha in [-1, 1]^2 (no raw
        box to snap into), identity network, unsafe y0 >= t with
        t = fl(c + r), the top of the set rounded to nearest. A SAT witness
        must be a member of the set in exact arithmetic (rational membership
        on the star's own rows). With c = 1/4, r = 1/2 the top 3/4 is a
        float, the corner is a real counterexample and is returned; with
        c = 0.2, r = 0.1 the float fl(c + r) = 0.30000000000000004 lies above
        the exact top c + r, so the problem is truly safe."""
        c, r = (0.25, 0.5) if centre == "dyadic" else (0.2, 0.1)
        S = Star(np.array([[c, r, 0.0], [c, 0.0, r]]), np.zeros((0, 2)),
                 np.zeros((0, 1)), -np.ones((2, 1)), np.ones((2, 1)))
        res = verify_refine_input_crown(S, _identity(2),
                                        LinearSpec([[-1.0, 0.0]], [-(c + r)]),
                                        node_budget=200)
        if res.status == Status.SAT:
            x = np.asarray(res.counterexample_x, dtype=np.float64)
            assert Q.member(Q.star(S), x), f"witness {x!r} outside the input set"
            assert -x[0] <= -(c + r)
        if centre == "dyadic":
            assert res.status == Status.SAT, res.status

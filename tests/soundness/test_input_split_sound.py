"""
Soundness of the batched-CROWN input-split driver.

Same contract as the activation-split engine: the verdict must equal a trusted
oracle (exact star reach), the branching axis (input bisection) only affects the
node count, and every SAT carries a concrete input the exact model maps into the
unsafe region. No false UNSAT.
"""

import numpy as np
import pytest

from n2v.nn.reach import reach_pytorch_model
from n2v.sets.star import Star
from n2v.utils.lpsolver import check_feasibility
from n2v.refine.types import LinearSpec, Status
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
            assert np.all(Y >= lo - 1e-9), "CROWN lower form violated"
            assert np.all(Y <= hi + 1e-9), "CROWN upper form violated"

    @pytest.mark.parametrize("seed", range(8))
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

    @pytest.mark.parametrize("seed", range(4))
    @pytest.mark.parametrize("row_rule", ["argmax", "argmin"])
    def test_verdict_matches_oracle_multirow(self, seed, row_rule):
        """Multi-row unsafe regions exercise the driving-row selection (the
        single-row tests cannot: argmin == argmax there)."""
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

    @pytest.mark.parametrize("seed", range(4))
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

    @pytest.mark.parametrize("seed", [4, 15, 19, 23, 33, 36, 38, 45, 49, 51,
                                      56, 59, 70, 74, 76, 106])
    def test_touching_spec_is_never_unsat(self, seed):
        """The unsafe region touches the image of the input box at one vertex
        (exact margin 0 there). The CROWN margins must be outward-rigorous in
        floating point, so the touching vertex is never rounded into a
        prune; these seeds gave a false UNSAT without the rounding slack."""
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

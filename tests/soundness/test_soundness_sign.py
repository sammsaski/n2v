"""Soundness tests for Sign activation reachability."""

import numpy as np
from n2v.sets import Star, Zono, Box
from n2v.nn.layer_ops.sign_reach import sign_star, sign_zono, sign_box


class TestSignStarSoundness:
    def test_random_samples_approx(self):
        np.random.seed(42)
        center = np.random.randn(5, 1)
        star = Star.from_bounds(center - 1.0, center + 1.0)
        result = sign_star(None, [star], method='approx')
        lb, ub = result[0].get_ranges()
        for _ in range(500):
            x = np.random.uniform((center - 1.0).flatten(), (center + 1.0).flatten())
            y = np.sign(x).reshape(-1, 1)
            assert np.all(y >= lb - 1e-6)
            assert np.all(y <= ub + 1e-6)

    def test_approx_contains_exact(self):
        star = Star.from_bounds(
            np.array([[-0.5], [0.2], [-1.0]]),
            np.array([[0.3], [0.8], [-0.1]])
        )
        approx_result = sign_star(None, [star], method='approx')
        exact_result = sign_star(None, [star], method='exact')
        approx_lb, approx_ub = approx_result[0].get_ranges()
        for exact_star in exact_result:
            exact_lb, exact_ub = exact_star.get_ranges()
            assert np.all(exact_lb >= approx_lb - 1e-6)
            assert np.all(exact_ub <= approx_ub + 1e-6)


class TestSignZonoSoundness:
    def test_random_samples(self):
        np.random.seed(42)
        center = np.random.randn(5, 1)
        zono = Zono.from_bounds(center - 1.0, center + 1.0)
        result = sign_zono([zono])
        lb, ub = result[0].get_bounds()
        for _ in range(500):
            x = np.random.uniform((center - 1.0).flatten(), (center + 1.0).flatten())
            y = np.sign(x).reshape(-1, 1)
            assert np.all(y >= lb - 1e-6)
            assert np.all(y <= ub + 1e-6)


class TestSignBoxSoundness:
    def test_random_samples(self):
        np.random.seed(42)
        center = np.random.randn(5, 1)
        box = Box(center - 1.0, center + 1.0)
        result = sign_box([box])
        for _ in range(500):
            x = np.random.uniform((center - 1.0).flatten(), (center + 1.0).flatten())
            y = np.sign(x).reshape(-1, 1)
            assert np.all(y >= result[0].lb - 1e-6)
            assert np.all(y <= result[0].ub + 1e-6)


class TestSignStarPolytopeMembership:
    """Regression for the C2 soundness bug (red-team PR#25): a crossing
    Sign neuron must NOT be relaxed with a secant coupling — that
    excludes the true output -1 for x in (l,0) (and +1 for x in (0,u)),
    shrinking the reach set (unsound) and potentially yielding a wrong
    `unsat`. The sound relaxation is the free box y in [-1,1].

    This checks POLYTOPE membership (LP feasibility of the true
    (x, sign(x)) against C v <= d), not just the per-coordinate box — the
    box check is blind to this class of bug (red-team H1)."""

    def _feasible(self, star, assignment):
        from scipy.optimize import linprog
        C = np.asarray(star.C, dtype=float)
        d = np.asarray(star.d, dtype=float).flatten()
        plb = np.asarray(star.predicate_lb, dtype=float).flatten()
        pub = np.asarray(star.predicate_ub, dtype=float).flatten()
        bounds = [(plb[k], pub[k]) for k in range(star.nVar)]
        for var, val in assignment.items():
            bounds[var] = (val, val)
        r = linprog(np.zeros(star.nVar),
                    A_ub=(C if C.size else None),
                    b_ub=(d if C.size else None),
                    bounds=bounds, method="highs")
        return r.success

    def test_crossing_neuron_keeps_true_outputs(self):
        # x = alpha in [-1, 1]; predicates after reach are [alpha, y]
        I = Star.from_bounds(np.array([-1.0]), np.array([1.0]))
        out = sign_star(None, [I], method="approx", lp_solver="linprog")[0]
        assert out.nVar == 2  # original alpha + fresh y
        for a in np.linspace(-0.99, 0.99, 41):
            y_true = float(np.sign(a))
            assert self._feasible(out, {0: a, 1: y_true}), (
                f"true (x={a}, sign={y_true}) excluded from the Sign star "
                f"polytope — unsound secant relaxation (C2 regressed)")
        # the exact red-team counterexample
        assert self._feasible(out, {0: -0.5, 1: -1.0})

"""Soundness tests for the batched-GPU sound LP path (PT-2).

The contract: the GPU first-order solver + Neumaier-Shcherbina certificate must
return bounds that *enclose* the true reachable range of every Star coordinate.
We pin this two ways:

  * **Differential vs exact CPU HiGHS** -- the GPU+NS bound is never *tighter*
    than the exact LP optimum (``gpu_ub >= cpu_ub``, ``gpu_lb <= cpu_lb``). A GPU
    bound inside the exact one is a soundness bug; fail hard.
  * **Containment sampling** -- thousands of points drawn from each Star fall
    within the returned ``[lb, ub]`` (the ultimate ground truth, solver-agnostic).

Both the within-star path (``get_ranges`` with ``use_gpu_lp``) and the
cross-population path (``get_ranges_population``) are covered. A no-GPU-safety
test pins that the CPU path is untouched when no device is present.

Skipped without CUDA, except the no-GPU-safety test.
"""

import numpy as np
import pytest

import n2v
from n2v.sets import Star
from n2v.utils.lpsolver_gpu import gpu_available

_HAS_CUDA = gpu_available()


def _random_star(rng, dim, nvar, nconstr):
    """A random bounded Star: V (dim, nvar+1), C alpha <= d, finite box."""
    V = rng.standard_normal((dim, nvar + 1))
    C = rng.standard_normal((nconstr, nvar))
    # RHS generous vs the box so the polytope is non-empty.
    d = (np.abs(C).sum(1) + rng.uniform(1.0, 2.5, nconstr)).reshape(-1, 1)
    plb = rng.uniform(-1.5, 0.0, (nvar, 1))
    pub = plb + rng.uniform(0.5, 2.5, (nvar, 1))
    return Star(V, C, d, plb, pub)


def _sample_points(star, n, rng):
    """Ground-truth points: sample alpha in the box, keep feasible, map to x."""
    plb = star.predicate_lb.flatten()
    pub = star.predicate_ub.flatten()
    alpha = rng.uniform(plb, pub, size=(n, star.nVar)).T          # (nvar, n)
    if star.C.size > 0:
        feasible = np.all(star.C @ alpha <= star.d + 1e-9, axis=0)
        alpha = alpha[:, feasible]
    x = star.V[:, 0:1] + star.V[:, 1:] @ alpha                    # (dim, k)
    return x


# --------------------------------------------------------------------------- #
# Differential: GPU+NS encloses the exact CPU optimum
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")
class TestDifferentialEnclosesExact:
    def test_get_ranges_within_star(self):
        rng = np.random.default_rng(101)
        worst_inward = 0.0
        n2v.set_gpu_lp(False)
        try:
            for _ in range(60):
                dim = int(rng.integers(2, 20))
                nvar = int(rng.integers(2, 8))
                star = _random_star(rng, dim, nvar, int(rng.integers(1, 6)))
                lb_c, ub_c = star.get_ranges()
                n2v.set_gpu_lp(True)
                lb_g, ub_g = star.get_ranges()
                n2v.set_gpu_lp(False)
                # gpu must enclose cpu: lb_g <= lb_c, ub_g >= ub_c.
                worst_inward = max(worst_inward,
                                   float(np.max(lb_g - lb_c)),
                                   float(np.max(ub_c - ub_g)))
                assert np.all(lb_g <= lb_c + 1e-6), "within-star lb not sound"
                assert np.all(ub_g >= ub_c - 1e-6), "within-star ub not sound"
        finally:
            n2v.set_gpu_lp(False)
        # Report the tightest the GPU bound ever got relative to exact (should be
        # ~0 or negative = strictly outward).
        assert worst_inward < 1e-6

    def test_get_ranges_population(self):
        rng = np.random.default_rng(202)
        dim, nvar = 8, 5
        stars = [_random_star(rng, dim, nvar, int(rng.integers(1, 6)))
                 for _ in range(12)]
        pop = Star.get_ranges_population(stars)
        for star, (lb_g, ub_g) in zip(stars, pop):
            lb_c, ub_c = star.get_ranges()
            assert np.all(lb_g <= lb_c + 1e-6), "population lb not sound"
            assert np.all(ub_g >= ub_c - 1e-6), "population ub not sound"


# --------------------------------------------------------------------------- #
# Containment sampling: the returned box contains every sampled point
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")
class TestContainmentSampling:
    def test_samples_inside_gpu_bounds(self):
        rng = np.random.default_rng(303)
        n2v.set_gpu_lp(True)
        try:
            for _ in range(20):
                dim = int(rng.integers(2, 12))
                nvar = int(rng.integers(2, 7))
                star = _random_star(rng, dim, nvar, int(rng.integers(1, 5)))
                lb_g, ub_g = star.get_ranges()
                x = _sample_points(star, 4000, rng)
                if x.shape[1] == 0:
                    continue
                assert np.all(x >= lb_g - 1e-6), "sample below GPU lb (unsound)"
                assert np.all(x <= ub_g + 1e-6), "sample above GPU ub (unsound)"
        finally:
            n2v.set_gpu_lp(False)


# --------------------------------------------------------------------------- #
# No-GPU safety: CPU path untouched when no device / GPU disabled
# --------------------------------------------------------------------------- #
class TestNoGpuSafety:
    def test_disabled_flag_uses_cpu_unchanged(self):
        rng = np.random.default_rng(404)
        star = _random_star(rng, 6, 4, 3)
        n2v.set_gpu_lp(False)
        lb1, ub1 = star.get_ranges()
        lb2, ub2 = star.get_ranges()  # deterministic CPU result
        assert np.allclose(lb1, lb2) and np.allclose(ub1, ub2)

    def test_falls_back_to_cpu_when_no_device(self, monkeypatch):
        """use_gpu_lp on, but gpu_available()->False: must use exact CPU path."""
        import n2v.utils.lpsolver_gpu as gpu_mod
        monkeypatch.setattr(gpu_mod, "gpu_available", lambda: False)
        rng = np.random.default_rng(505)
        star = _random_star(rng, 6, 4, 3)
        lb_cpu, ub_cpu = star.get_ranges()
        n2v.set_gpu_lp(True)
        try:
            lb, ub = star.get_ranges()
        finally:
            n2v.set_gpu_lp(False)
        assert np.allclose(lb, lb_cpu) and np.allclose(ub, ub_cpu)

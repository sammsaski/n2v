"""Tests for the batched first-order (GPU) sound LP backend.

``solve_lp_batch_gpu`` shares ``solve_lp_batch``'s contract -- same args, returns
one bound per objective -- but solves all objectives at once with a first-order
primal-dual method and returns the *Neumaier-Shcherbina certified* bound, never
the raw solver optimum. Two backends:
  * ``cpu_ns``  -- HiGHS solves + NS certificate (no GPU; the reference/fallback
                   that proves the approximate-solve+certificate pipeline).
  * ``pdhg``    -- batched PyTorch PDHG on GPU + NS (the speed path; skipped if
                   no CUDA device is present).

The cross-cutting invariant for *both* backends: every returned bound must
*enclose* the exact HiGHS optimum (>= for max, <= for min). Tightness is a
quality metric, soundness is mandatory.
"""

import numpy as np
import pytest

from n2v.utils.lpsolver import solve_lp, solve_lp_batch
from n2v.utils.lpsolver_gpu import (
    gpu_available,
    solve_lp_batch_gpu,
    solve_lp_population_gpu,
)

_HAS_CUDA = gpu_available()
_BACKENDS = ["cpu_ns"] + (["pdhg"] if _HAS_CUDA else [])


def _random_lp(rng, n, m):
    c = rng.standard_normal(n)
    A = rng.standard_normal((m, n))
    b = rng.uniform(1.0, 3.0, size=m) + np.abs(A).sum(axis=1)
    lb = rng.uniform(-2.0, 0.0, size=n)
    ub = lb + rng.uniform(0.5, 3.0, size=n)
    return c, A, b, lb, ub


def _bound_objectives(n):
    """The 2n min/max axis objectives a Star issues, plus their flags."""
    objs, flags = [], []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        objs.extend([e.copy(), e.copy()])
        flags.extend([True, False])
    return objs, flags


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
class TestContract:
    def test_empty_objectives(self):
        assert solve_lp_batch_gpu([], A=None, b=None, lb=None, ub=None,
                                  minimize_flags=[], backend="cpu_ns") == []

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_returns_one_float_per_objective(self, backend):
        rng = np.random.default_rng(0)
        c, A, b, lb, ub = _random_lp(rng, 4, 3)
        objs = [c, -c, rng.standard_normal(4)]
        flags = [False, True, False]
        out = solve_lp_batch_gpu(objs, A=A, b=b, lb=lb, ub=ub,
                                 minimize_flags=flags, backend=backend)
        assert len(out) == len(objs)
        assert all(isinstance(v, float) and np.isfinite(v) for v in out)


# --------------------------------------------------------------------------- #
# Soundness: certified bounds enclose the exact HiGHS optimum
# --------------------------------------------------------------------------- #
class TestSoundness:
    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_encloses_exact_on_random_lps(self, backend):
        rng = np.random.default_rng(123)
        for _ in range(40):
            n, m = int(rng.integers(2, 6)), int(rng.integers(1, 6))
            c, A, b, lb, ub = _random_lp(rng, n, m)
            objs, flags = _bound_objectives(n)
            exact = solve_lp_batch(objs, A=A, b=b, lb=lb, ub=ub,
                                   minimize_flags=flags)
            cert = solve_lp_batch_gpu(objs, A=A, b=b, lb=lb, ub=ub,
                                      minimize_flags=flags, backend=backend)
            for i, do_min in enumerate(flags):
                if exact[i] is None:
                    continue
                if do_min:
                    assert cert[i] <= exact[i] + 1e-7, \
                        f"{backend}: min bound {cert[i]} above exact {exact[i]}"
                else:
                    assert cert[i] >= exact[i] - 1e-7, \
                        f"{backend}: max bound {cert[i]} below exact {exact[i]}"

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_box_only_lp(self, backend):
        """No general constraints -> exact box bound from both backends."""
        lb = np.array([-1.0, 0.0, 2.0])
        ub = np.array([1.0, 3.0, 5.0])
        objs, flags = _bound_objectives(3)
        cert = solve_lp_batch_gpu(objs, A=None, b=None, lb=lb, ub=ub,
                                  minimize_flags=flags, backend=backend)
        # min/max of each coordinate == its box endpoint.
        expected = [lb[0], ub[0], lb[1], ub[1], lb[2], ub[2]]
        for got, exp in zip(cert, expected):
            assert got == pytest.approx(exp, abs=1e-6)


# --------------------------------------------------------------------------- #
# Tightness (quality, not soundness) -- only meaningful for the GPU solver
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")
class TestPdhgTightness:
    def test_pdhg_bounds_reasonably_tight(self):
        """On well-conditioned LPs the PDHG dual should certify bounds close to
        the exact optimum (loose-but-sound is allowed, but not wildly loose)."""
        rng = np.random.default_rng(7)
        gaps = []
        for _ in range(20):
            n, m = 5, 4
            c, A, b, lb, ub = _random_lp(rng, n, m)
            objs, flags = _bound_objectives(n)
            exact = solve_lp_batch(objs, A=A, b=b, lb=lb, ub=ub, minimize_flags=flags)
            cert = solve_lp_batch_gpu(objs, A=A, b=b, lb=lb, ub=ub,
                                      minimize_flags=flags, backend="pdhg")
            span = max(ub) - min(lb)
            for i, do_min in enumerate(flags):
                if exact[i] is None:
                    continue
                gaps.append(abs(cert[i] - exact[i]) / (abs(span) + 1e-9))
        # Median relative gap should be small if PDHG converges adequately.
        assert np.median(gaps) < 0.1, f"median rel gap {np.median(gaps):.3f} too loose"


# --------------------------------------------------------------------------- #
# Cross-population batch (4a-ii)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")
class TestPopulation:
    def test_population_encloses_per_star_exact(self):
        """Population bounds enclose each star's exact HiGHS optimum, including a
        ragged population padded to a common constraint count."""
        rng = np.random.default_rng(31)
        B, n, k = 6, 5, 8
        flags = [j % 2 == 0 for j in range(k)]  # [min,max]*... arbitrary mix
        ms = [int(rng.integers(1, 5)) for _ in range(B)]
        m_max = max(ms)
        A = np.zeros((B, m_max, n))
        b = np.zeros((B, m_max))
        lb = rng.uniform(-2.0, 0.0, size=(B, n))
        ub = lb + rng.uniform(0.5, 3.0, size=(B, n))
        C = rng.standard_normal((B, n, k))
        for s in range(B):
            As = rng.standard_normal((ms[s], n))
            A[s, :ms[s], :] = As
            b[s, :ms[s]] = rng.uniform(1.0, 3.0, size=ms[s]) + np.abs(As).sum(1)

        cert = solve_lp_population_gpu(A, b, lb, ub, C, flags)
        assert cert.shape == (B, k)
        for s in range(B):
            Ar = A[s, :ms[s], :]
            br = b[s, :ms[s]]
            for j in range(k):
                _, exact, status, _ = solve_lp(
                    f=C[s, :, j], A=Ar, b=br, lb=lb[s], ub=ub[s],
                    minimize=flags[j],
                )
                if status != "optimal":
                    continue
                if flags[j]:
                    assert cert[s, j] <= exact + 1e-6
                else:
                    assert cert[s, j] >= exact - 1e-6

    def test_get_ranges_population_matches_per_star(self):
        """Star.get_ranges_population encloses per-star get_ranges (CPU exact)."""
        from n2v.sets import Star

        rng = np.random.default_rng(7)
        stars = []
        dim, nv = 6, 4
        for _ in range(5):
            V = rng.standard_normal((dim, nv + 1))
            m = int(rng.integers(2, 5))
            Cm = rng.standard_normal((m, nv))
            d = (np.abs(Cm).sum(1) + rng.uniform(1, 2, m)).reshape(-1, 1)
            plb = rng.uniform(-1, 0, (nv, 1))
            pub = plb + rng.uniform(0.5, 2, (nv, 1))
            stars.append(Star(V, Cm, d, plb, pub))

        pop = Star.get_ranges_population(stars)
        assert len(pop) == len(stars)
        for s, (lb_g, ub_g) in zip(stars, pop):
            lb_c, ub_c = s.get_ranges()
            assert np.all(lb_g <= lb_c + 1e-6), "population lb not sound"
            assert np.all(ub_g >= ub_c - 1e-6), "population ub not sound"

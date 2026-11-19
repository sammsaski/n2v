"""Tests for PCATransform (whiten-then-project transform for flow+PCA reach)."""

import numpy as np
import pytest

from n2v.sets import HalfSpace
from n2v.probabilistic.flow.sets import PCATransform


def _make_transform(D=20, k=6, seed=0, residual_bound=0.0):
    rng = np.random.default_rng(seed)
    mean = rng.standard_normal(D)
    std = np.abs(rng.standard_normal(D)) + 0.5
    # orthonormal rows (k, D)
    A = rng.standard_normal((D, k))
    Q, _ = np.linalg.qr(A)              # (D, k) orthonormal columns
    components = Q.T                     # (k, D) orthonormal rows
    pca_mean = rng.standard_normal(D)
    return PCATransform(mean=mean, std=std, components=components,
                        pca_mean=pca_mean, residual_bound=residual_bound), rng


class TestPCATransform:

    def test_dims(self):
        t, _ = _make_transform(D=20, k=6)
        assert t.dim == 6 and t.full_dim == 20

    def test_whiten_unwhiten_roundtrip_in_subspace(self):
        """A point built from a latent z reconstructs to that z exactly."""
        t, rng = _make_transform(D=20, k=6, seed=1)
        z = rng.standard_normal((5, 6))
        y = t.unwhiten(z)
        z_rec = t.whiten(y)
        np.testing.assert_allclose(z_rec, z, atol=1e-8)

    def test_halfspace_exact_margin_no_residual(self):
        """residual_bound=0: reduced margin == full margin for in-subspace points."""
        t, rng = _make_transform(D=20, k=6, seed=2, residual_bound=0.0)
        z0 = rng.standard_normal((1, 6))
        y0 = t.unwhiten(z0)                       # (1, 20), exactly in subspace
        n_rows = 3
        G = rng.standard_normal((n_rows, 20))
        g = rng.standard_normal((n_rows, 1))
        hs = HalfSpace(G, g)
        hs_red = t.transform_halfspace(hs)
        # full-space margins  G y0 - g   vs  reduced margins  G_red z0 - g_red
        full = (G @ y0.reshape(-1, 1) - g).flatten()
        red = (hs_red.G @ z0.reshape(-1, 1) - hs_red.g).flatten()
        np.testing.assert_allclose(full, red, atol=1e-8)

    def test_residual_inflation_enlarges_unsafe(self):
        """residual_bound>0 raises g_red (unsafe region grows) -> conservative."""
        t0, rng = _make_transform(D=20, k=6, seed=3, residual_bound=0.0)
        t1 = PCATransform(mean=t0.mean, std=t0.std, components=t0.components,
                          pca_mean=t0.pca_mean, residual_bound=0.25)
        G = rng.standard_normal((4, 20))
        g = rng.standard_normal((4, 1))
        hs = HalfSpace(G, g)
        g0 = t0.transform_halfspace(hs).g.flatten()
        g1 = t1.transform_halfspace(hs).g.flatten()
        # inflated bound is larger by residual_bound * ||G⊙std||_1 (Hölder ℓ1–ℓ∞)
        assert np.all(g1 >= g0 - 1e-12)
        assert np.all(g1 > g0)  # strictly, since rows are nonzero
        expected = 0.25 * np.abs(G * t0.std[None, :]).sum(axis=1)
        np.testing.assert_allclose(g1 - g0, expected, atol=1e-8)

    def test_residual_inflation_single_cell(self):
        """For a single-cell halfspace, inflation = residual_bound * std_cell
        (ℓ1 row-norm of a one-hot row is just std_i)."""
        t, _ = _make_transform(D=12, k=4, seed=5, residual_bound=0.3)
        t0 = PCATransform(mean=t.mean, std=t.std, components=t.components,
                          pca_mean=t.pca_mean, residual_bound=0.0)
        i = 3
        G = np.zeros((1, 12)); G[0, i] = 1.0
        hs = HalfSpace(G, np.array([[0.5]]))
        infl = (t.transform_halfspace(hs).g.flatten()[0]
                - t0.transform_halfspace(hs).g.flatten()[0])
        np.testing.assert_allclose(infl, 0.3 * t.std[i], atol=1e-8)

    def test_rejects_bad_shapes(self):
        with pytest.raises(ValueError):
            PCATransform(mean=np.zeros(5), std=np.ones(4),
                         components=np.zeros((2, 5)), pca_mean=np.zeros(5))
        with pytest.raises(ValueError):
            PCATransform(mean=np.zeros(5), std=np.ones(5),
                         components=np.zeros((2, 5)), pca_mean=np.zeros(5),
                         residual_bound=-1.0)
        with pytest.raises(ValueError):  # non-positive std
            PCATransform(mean=np.zeros(3), std=np.array([1.0, 0.0, 1.0]),
                         components=np.zeros((2, 3)), pca_mean=np.zeros(3))

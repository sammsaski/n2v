"""Tests for RandomizedSVDPCA (matches DeflationPCA's interface conventions)."""

import numpy as np
import pytest

from n2v.probabilistic.dimensionality import RandomizedSVDPCA


def _low_rank_data(t=300, n=400, rank=8, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((t, rank))
    V = rng.standard_normal((rank, n))
    mean = rng.standard_normal(n) * 5.0
    return U @ V + mean, mean, rank


class TestRandomizedSVDPCA:

    def test_shapes_and_orthonormal_rows(self):
        X, _, _ = _low_rank_data()
        pca = RandomizedSVDPCA(n_components=10, random_state=0).fit(X)
        assert pca.components_.shape == (10, X.shape[1])
        assert pca.mean_.shape == (X.shape[1],)
        # rows orthonormal
        gram = pca.components_ @ pca.components_.T
        np.testing.assert_allclose(gram, np.eye(10), atol=1e-8)

    def test_recovers_exact_low_rank(self):
        """k = true rank -> near-perfect reconstruction."""
        X, _, rank = _low_rank_data(rank=8)
        pca = RandomizedSVDPCA(n_components=8, random_state=0).fit(X)
        Xrec = pca.inverse_transform(pca.transform(X))
        rel = np.linalg.norm(X - Xrec) / np.linalg.norm(X)
        assert rel < 1e-6, f"reconstruction rel error {rel}"

    def test_transform_inverse_shapes(self):
        X, _, _ = _low_rank_data()
        pca = RandomizedSVDPCA(n_components=5, random_state=0).fit(X)
        Z = pca.transform(X)
        assert Z.shape == (X.shape[0], 5)
        assert pca.inverse_transform(Z).shape == X.shape

    def test_matches_exact_svd_subspace(self):
        """Captured subspace agrees with exact SVD (projection equivalence)."""
        X, _, _ = _low_rank_data(t=200, n=300, rank=12)
        k = 12
        rp = RandomizedSVDPCA(n_components=k, random_state=1).fit(X)
        # exact PCA directions
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        exact = Vt[:k]
        # Same subspace: projecting Xc with either basis gives the same energy.
        e_rand = np.linalg.norm(Xc @ rp.components_.T)
        e_exact = np.linalg.norm(Xc @ exact.T)
        assert abs(e_rand - e_exact) / e_exact < 1e-6

    def test_reduces_to_k(self):
        X, _, _ = _low_rank_data(t=100, n=5000, rank=20)
        pca = RandomizedSVDPCA(n_components=15, random_state=0)
        Z = pca.fit_transform(X)
        assert Z.shape == (100, 15)
        assert pca.n_components_fitted_ == 15
        # explained variance is descending
        ev = pca.explained_variance_
        assert np.all(np.diff(ev) <= 1e-9)

    def test_caps_components_at_min_dim(self):
        X, _, _ = _low_rank_data(t=20, n=50, rank=5)
        pca = RandomizedSVDPCA(n_components=100, random_state=0).fit(X)
        assert pca.n_components_fitted_ == min(20, 50)

    def test_requires_fit(self):
        pca = RandomizedSVDPCA(n_components=3)
        with pytest.raises(RuntimeError):
            pca.transform(np.zeros((2, 4)))

    def test_reproducible(self):
        X, _, _ = _low_rank_data()
        a = RandomizedSVDPCA(n_components=6, random_state=7).fit_transform(X)
        b = RandomizedSVDPCA(n_components=6, random_state=7).fit_transform(X)
        np.testing.assert_allclose(a, b)

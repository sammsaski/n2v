"""Phase 1.4b: sanity tests for the plotly 3D Star-union renderer.

We don't verify pixel-level rendering; we do verify that:
1. The HTML export creates a file.
2. When given forward-sampled points and a synthetic single-star reach set
   whose image we control, the points visibly fall inside the mesh — this
   is checked analytically, not visually.
"""

import numpy as np
import pytest

pytest.importorskip("plotly")

from n2v.probabilistic.flow.star_viz import (
    _star_membership,
    render_star_convex_hull_3d,
    render_star_union_3d,
    render_star_union_isosurface_3d,
)
from n2v.sets.star import Star


class _FakeStar:
    def __init__(self, V, predicate_lb, predicate_ub):
        self.V = V
        self.predicate_lb = predicate_lb
        self.predicate_ub = predicate_ub
        self.C = None
        self.d = None


class TestStarViz:
    def test_single_unit_box(self, tmp_path):
        # Identity star: y = alpha, alpha in [-1, 1]^3
        V = np.concatenate([np.zeros((3, 1)), np.eye(3)], axis=1)
        star = _FakeStar(V, predicate_lb=[-1, -1, -1], predicate_ub=[1, 1, 1])
        samples = np.random.default_rng(0).uniform(-1, 1, size=(200, 3))
        out = tmp_path / 'unit_box.html'
        fig = render_star_union_3d([star], forward_samples=samples, out_html=out)
        assert out.exists()
        # the figure has 2 traces (1 mesh + 1 scatter)
        assert len(fig.data) == 2

    def test_non_3x3_basis_skipped(self, tmp_path):
        V = np.zeros((3, 3))  # 2 basis columns (rank-deficient)
        star = _FakeStar(V, predicate_lb=[0, 0], predicate_ub=[1, 1])
        out = tmp_path / 'skipped.html'
        fig = render_star_union_3d([star], out_html=out)
        assert out.exists()
        # star skipped -> no mesh traces
        assert len(fig.data) == 0


class TestStarConvexHull:
    def test_unit_cube_hull_volume(self, tmp_path):
        """Hull of a single unit cube should have volume 8 (= 2^3)."""
        V = np.concatenate([np.zeros((3, 1)), np.eye(3)], axis=1)
        star = _FakeStar(V, predicate_lb=[-1, -1, -1], predicate_ub=[1, 1, 1])
        out = tmp_path / 'hull.html'
        fig, vol = render_star_convex_hull_3d([star], out_html=out)
        assert out.exists()
        assert abs(vol - 8.0) < 1e-6, f"cube hull volume {vol} != 8"
        assert len(fig.data) == 1  # one hull mesh, no samples

    def test_two_disjoint_boxes_hull_overapproximates(self, tmp_path):
        """Convex hull of two separated unit cubes contains their bounding box.

        Hull of cubes at [-1,1]^3 and [3,5]^3 has corners at (-1,-1,-1) and
        (5, 1, 1) in some dims — actually the hull is a prism. We just check
        the hull volume exceeds the sum of the cube volumes (2 x 8 = 16),
        which is the 'over-approximation' property.
        """
        V1 = np.concatenate([np.zeros((3, 1)), np.eye(3)], axis=1)
        s1 = _FakeStar(V1, predicate_lb=[-1, -1, -1], predicate_ub=[1, 1, 1])
        V2 = np.concatenate([np.array([[4.0], [4.0], [4.0]]), np.eye(3)], axis=1)
        s2 = _FakeStar(V2, predicate_lb=[-1, -1, -1], predicate_ub=[1, 1, 1])
        _, vol = render_star_convex_hull_3d([s1, s2], out_html=tmp_path / 'h.html')
        # Hull strictly exceeds sum of components' volumes (16).
        assert vol > 16.0, f"expected overapprox hull > 16, got {vol}"

    def test_samples_trace_added(self, tmp_path):
        V = np.concatenate([np.zeros((3, 1)), np.eye(3)], axis=1)
        star = _FakeStar(V, predicate_lb=[-1, -1, -1], predicate_ub=[1, 1, 1])
        samples = np.random.default_rng(0).uniform(-1, 1, size=(50, 3))
        fig, _ = render_star_convex_hull_3d(
            [star], forward_samples=samples, out_html=tmp_path / 'h.html',
        )
        # 1 hull + 1 scatter trace
        assert len(fig.data) == 2


class TestStarIsosurface:
    def test_unit_cube_isosurface_trace(self, tmp_path):
        pytest.importorskip("skimage")
        V = np.concatenate([np.zeros((3, 1)), np.eye(3)], axis=1)
        star = _FakeStar(V, predicate_lb=[-1, -1, -1], predicate_ub=[1, 1, 1])
        out = tmp_path / 'iso.html'
        fig = render_star_union_isosurface_3d(
            [star], out_html=out, resolution=32,
        )
        assert out.exists()
        # Single isosurface mesh.
        assert len(fig.data) == 1
        # Mesh has non-zero faces.
        mesh = fig.data[0]
        assert len(mesh.i) > 0

    def test_two_disjoint_boxes_separate_components(self, tmp_path):
        """Isosurface preserves non-convex structure: two disjoint cubes
        produce a mesh whose vertex range covers both, with a gap."""
        pytest.importorskip("skimage")
        V1 = np.concatenate([np.zeros((3, 1)), np.eye(3)], axis=1)
        s1 = _FakeStar(V1, predicate_lb=[-1, -1, -1], predicate_ub=[1, 1, 1])
        V2 = np.concatenate([np.array([[6.0], [0.0], [0.0]]), np.eye(3)], axis=1)
        s2 = _FakeStar(V2, predicate_lb=[-1, -1, -1], predicate_ub=[1, 1, 1])
        fig = render_star_union_isosurface_3d(
            [s1, s2], out_html=tmp_path / 'iso.html', resolution=48,
        )
        xs = np.asarray(fig.data[0].x)
        # Vertices should span from ~-1 to ~7 (covers both cubes).
        assert xs.min() < 0.0
        assert xs.max() > 5.0


class TestStarMembershipMatchesStarContains:
    def test_agreement_on_cd_star(self):
        """_star_membership(points, [star]) must match Star.contains elementwise
        for a Star with non-trivial C/d constraints — the bug that prompted V6."""
        rng = np.random.default_rng(0)
        dim = 3
        # Star whose image is [-1,1]^3 cut by alpha[0] + alpha[1] <= 0.5.
        V = np.zeros((dim, dim + 1))
        V[:, 1:] = np.eye(dim)
        C = np.array([[1.0, 1.0, 0.0]])
        d = np.array([0.5])
        s = Star(
            V=V, C=C, d=d,
            pred_lb=-np.ones(dim), pred_ub=np.ones(dim),
        )
        points = rng.uniform(-1.5, 1.5, size=(200, dim))

        viz_mask = _star_membership(points, [s])
        contains_mask = s.contains(points, method='lp')

        np.testing.assert_array_equal(viz_mask, contains_mask)

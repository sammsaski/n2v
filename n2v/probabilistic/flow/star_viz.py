"""3D Plotly rendering for unions of n2v Stars.

Three rendering modes:

* :func:`render_star_union_3d` — draws every Star's image as a separate
  translucent parallelepiped. Faithful to the individual Stars but noisy
  when the union has hundreds of overlapping pieces.
* :func:`render_star_convex_hull_3d` — draws a single convex-hull mesh
  over all Star vertices. Clean single surface; exact when the union is
  convex and an over-approximation otherwise. Useful to detect whether
  the reach set is approximately convex.
* :func:`render_star_union_isosurface_3d` — samples Star-union membership
  on a 3D grid and extracts the boundary via marching cubes. A single
  clean mesh that preserves non-convex structure. Recommended for reach
  sets that are substantially non-convex (the convex-hull-over-approx
  case).

All three modes accept forward-sampled network outputs that should lie
inside the union; these are drawn as a point cloud for sanity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


def _parallelepiped_vertices(offset: np.ndarray, basis: np.ndarray,
                             plb: np.ndarray, pub: np.ndarray) -> np.ndarray:
    """Return the 8 corners of the affine image of the box [plb, pub] under
    y = offset + basis @ alpha. Assumes 3x3 basis.
    """
    corners = np.stack(np.meshgrid(
        [plb[0], pub[0]], [plb[1], pub[1]], [plb[2], pub[2]], indexing='ij'
    ), axis=-1).reshape(-1, 3)
    return offset[None, :] + corners @ basis.T


def _box_faces():
    """Triangle indices for the 12 faces of a hexahedron (8 vertices
    ordered as the np.meshgrid above). i, j, k arrays."""
    # vertex indices per (x,y,z) in {0,1}:
    # 0=(0,0,0) 1=(0,0,1) 2=(0,1,0) 3=(0,1,1)
    # 4=(1,0,0) 5=(1,0,1) 6=(1,1,0) 7=(1,1,1)
    tris = [
        (0, 1, 3), (0, 3, 2),  # x=0 face
        (4, 6, 7), (4, 7, 5),  # x=1 face
        (0, 4, 5), (0, 5, 1),  # y=0 face
        (2, 3, 7), (2, 7, 6),  # y=1 face
        (0, 2, 6), (0, 6, 4),  # z=0 face
        (1, 5, 7), (1, 7, 3),  # z=1 face
    ]
    i = np.array([t[0] for t in tris])
    j = np.array([t[1] for t in tris])
    k = np.array([t[2] for t in tris])
    return i, j, k


def render_star_union_3d(
    stars: Iterable,
    forward_samples: Optional[np.ndarray] = None,
    title: str = 'Star union reach set',
    out_html: Optional[Path] = None,
    max_stars: int = 500,
    mesh_opacity: float = 0.25,
    include_plotlyjs: str = 'directory',
) -> 'plotly.graph_objects.Figure':
    """Render a list of n2v Stars as a 3D Plotly figure.

    Args:
        stars: iterable of n2v Stars (with V, predicate_lb, predicate_ub).
        forward_samples: optional (N, 3) array of network-output samples
            expected to lie inside the union.
        out_html: if given, write the figure to this path.
        max_stars: cap number of stars drawn (for interactivity).
        mesh_opacity: per-star opacity (low to avoid solid-color blobs).

    Returns the Plotly Figure; caller may further customize or display.
    """
    import plotly.graph_objects as go

    stars = list(stars)[:max_stars]
    i_tri, j_tri, k_tri = _box_faces()
    traces = []
    n_bad_shape = 0
    for star in stars:
        V = np.asarray(star.V, dtype=np.float64)
        offset = V[:, 0]
        basis = V[:, 1:]
        if basis.shape != (3, 3):
            n_bad_shape += 1
            continue
        plb = np.asarray(star.predicate_lb, dtype=np.float64).flatten()
        pub = np.asarray(star.predicate_ub, dtype=np.float64).flatten()
        verts = _parallelepiped_vertices(offset, basis, plb, pub)
        traces.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=i_tri, j=j_tri, k=k_tri,
            opacity=mesh_opacity, color='steelblue',
            flatshading=True, showscale=False, hoverinfo='skip',
        ))

    if n_bad_shape > 0:
        logger.info("render_star_union_3d skipped %d stars with non-3x3 bases", n_bad_shape)

    if forward_samples is not None:
        traces.append(go.Scatter3d(
            x=forward_samples[:, 0], y=forward_samples[:, 1], z=forward_samples[:, 2],
            mode='markers',
            marker=dict(size=2, color='orangered', opacity=0.8),
            name='forward-sampled outputs',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if out_html is not None:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_html), include_plotlyjs=include_plotlyjs)
    return fig


def _collect_star_vertices(stars: Iterable) -> Tuple[np.ndarray, int]:
    """Collect 8-corner vertex sets from every Star with a 3x3 basis.

    Returns ``(verts, n_skipped)`` where ``verts`` has shape ``(8 * K, 3)``
    and ``n_skipped`` counts stars whose basis was not 3x3 (rank-deficient
    or differently-shaped image — measure-zero contribution in 3D).
    """
    chunks = []
    n_skipped = 0
    for star in stars:
        V = np.asarray(star.V, dtype=np.float64)
        basis = V[:, 1:]
        if basis.shape != (3, 3):
            n_skipped += 1
            continue
        offset = V[:, 0]
        plb = np.asarray(star.predicate_lb, dtype=np.float64).flatten()
        pub = np.asarray(star.predicate_ub, dtype=np.float64).flatten()
        chunks.append(_parallelepiped_vertices(offset, basis, plb, pub))
    if not chunks:
        return np.zeros((0, 3)), n_skipped
    return np.concatenate(chunks, axis=0), n_skipped


def render_star_convex_hull_3d(
    stars: Iterable,
    forward_samples: Optional[np.ndarray] = None,
    title: str = 'Star union (convex hull over-approximation)',
    out_html: Optional[Path] = None,
    mesh_opacity: float = 0.35,
    include_plotlyjs: str = 'directory',
) -> Tuple['plotly.graph_objects.Figure', float]:
    """Render the convex hull of all Star vertices as a single mesh.

    Gives a single clean surface instead of a fuzzy pile of overlapping
    parallelepipeds. The hull is an over-approximation of the true Star
    union (exact iff the union is already convex).

    Args:
        stars: iterable of n2v Stars (V, predicate_lb, predicate_ub).
        forward_samples: optional (N, 3) array; drawn as a scatter.
        title: figure title.
        out_html: if given, write the figure to this path.
        mesh_opacity: hull mesh opacity.

    Returns:
        Tuple ``(fig, hull_volume)``. ``hull_volume`` is the 3D volume of
        the convex hull — compare against an exact MC volume to gauge how
        tight the over-approximation is.

    Raises:
        ValueError: if fewer than 4 valid vertices are produced (need at
            least 4 non-coplanar points for a 3D hull).
    """
    import plotly.graph_objects as go
    from scipy.spatial import ConvexHull

    verts, n_skipped = _collect_star_vertices(stars)
    if n_skipped > 0:
        logger.info(
            "render_star_convex_hull_3d skipped %d stars with non-3x3 bases",
            n_skipped,
        )
    if verts.shape[0] < 4:
        raise ValueError(
            f"need at least 4 vertices to form a 3D convex hull, got {verts.shape[0]}"
        )

    hull = ConvexHull(verts)
    simplices = hull.simplices  # (n_faces, 3) vertex indices

    traces = [go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=simplices[:, 0], j=simplices[:, 1], k=simplices[:, 2],
        opacity=mesh_opacity, color='steelblue',
        flatshading=True, showscale=False, name='convex hull',
    )]

    if forward_samples is not None:
        traces.append(go.Scatter3d(
            x=forward_samples[:, 0],
            y=forward_samples[:, 1],
            z=forward_samples[:, 2],
            mode='markers',
            marker=dict(size=2, color='orangered', opacity=0.8),
            name='forward-sampled outputs',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if out_html is not None:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_html), include_plotlyjs=include_plotlyjs)

    return fig, float(hull.volume)


def _star_membership(
    points: np.ndarray, stars: Iterable, eps: float = 1e-9,
) -> np.ndarray:
    """Vectorized membership in the union of stars with full-rank 3x3 bases.

    For each Star, a point ``y`` is inside iff the unique
    ``alpha = basis^{-1} (y - offset)`` satisfies both the predicate box
    ``[plb, pub]`` **and** (if present) the affine predicate constraint
    ``C @ alpha <= d``. The previous version of this helper only checked
    the predicate box, which is an over-approximation when Stars carry
    non-trivial C/d.

    Processes Stars sequentially (vectorized over the ``points`` batch per
    Star) with a short-circuit OR across Stars. For Stars with non-3x3 or
    rank-deficient bases, contribution is skipped (measure-zero in 3D).

    Args:
        points: ``(N, 3)`` array of candidate points.
        stars: iterable of Stars.
        eps: tolerance for ``C @ alpha <= d`` check.

    Returns:
        ``(N,)`` boolean array; ``True`` means the point is in the union.
    """
    N = points.shape[0]
    in_union = np.zeros(N, dtype=bool)
    for star in stars:
        V = np.asarray(star.V, dtype=np.float64)
        basis = V[:, 1:]
        if basis.shape != (3, 3):
            continue
        try:
            binv = np.linalg.inv(basis)
        except np.linalg.LinAlgError:
            continue
        offset = V[:, 0]
        plb = np.asarray(star.predicate_lb, dtype=np.float64).flatten()
        pub = np.asarray(star.predicate_ub, dtype=np.float64).flatten()
        # Only evaluate points that haven't already been accepted.
        todo = ~in_union
        if not todo.any():
            break
        alpha = (points[todo] - offset) @ binv.T  # (M, 3)
        inside = ((alpha >= plb) & (alpha <= pub)).all(axis=1)
        if star.C is not None and np.asarray(star.C).size > 0:
            C = np.asarray(star.C, dtype=np.float64)
            d = np.asarray(star.d, dtype=np.float64).flatten()
            cmask = ((alpha @ C.T) <= d + eps).all(axis=1)
            inside &= cmask
        # Scatter accepted points back.
        idxs = np.flatnonzero(todo)
        in_union[idxs[inside]] = True
    return in_union


def render_star_union_isosurface_3d(
    stars: Iterable,
    forward_samples: Optional[np.ndarray] = None,
    title: str = 'Star union (marching-cubes isosurface)',
    out_html: Optional[Path] = None,
    resolution: int = 96,
    padding_frac: float = 0.05,
    mesh_opacity: float = 0.4,
    include_plotlyjs: str = 'directory',
) -> 'plotly.graph_objects.Figure':
    """Render the Star union as a marching-cubes isosurface.

    Samples ``resolution^3`` points on a regular grid over the bounding
    box of the Stars' image, evaluates union membership, and extracts the
    boundary via ``skimage.measure.marching_cubes``. Produces a single
    mesh that preserves non-convex structure.

    Args:
        stars: iterable of n2v Stars.
        forward_samples: optional (N, 3) array; drawn as a scatter.
        resolution: grid side length (total voxels = resolution^3).
        padding_frac: expand the Stars' bbox by this fraction per side
            so the isosurface doesn't get clipped.
        mesh_opacity: isosurface opacity.

    Returns the Plotly Figure.

    Raises:
        ValueError: if no Stars contribute valid vertices.
    """
    import plotly.graph_objects as go
    from skimage import measure

    verts_all, n_skipped = _collect_star_vertices(stars)
    if n_skipped > 0:
        logger.info(
            "render_star_union_isosurface_3d skipped %d stars with non-3x3 bases",
            n_skipped,
        )
    if verts_all.shape[0] == 0:
        raise ValueError("no Stars with 3x3 basis to render")

    lo = verts_all.min(axis=0)
    hi = verts_all.max(axis=0)
    pad = padding_frac * (hi - lo)
    lo, hi = lo - pad, hi + pad

    xs = np.linspace(lo[0], hi[0], resolution)
    ys = np.linspace(lo[1], hi[1], resolution)
    zs = np.linspace(lo[2], hi[2], resolution)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    inside = _star_membership(grid, stars).reshape(xx.shape)

    # marching_cubes needs a non-trivial level; use 0.5 on the boolean
    # field cast to float to extract the boundary.
    field = inside.astype(np.float32)
    if field.max() == 0:
        raise ValueError("no grid points landed inside any Star")

    dx = (hi - lo) / (resolution - 1)
    mc_verts, faces, _, _ = measure.marching_cubes(field, level=0.5, spacing=dx)
    # Shift from grid-local coords to absolute coords.
    mc_verts = mc_verts + lo[None, :]

    traces = [go.Mesh3d(
        x=mc_verts[:, 0], y=mc_verts[:, 1], z=mc_verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=mesh_opacity, color='steelblue',
        flatshading=True, showscale=False, name='Star union',
    )]

    if forward_samples is not None:
        traces.append(go.Scatter3d(
            x=forward_samples[:, 0],
            y=forward_samples[:, 1],
            z=forward_samples[:, 2],
            mode='markers',
            marker=dict(size=2, color='orangered', opacity=0.9),
            name='forward-sampled outputs',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if out_html is not None:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_html), include_plotlyjs=include_plotlyjs)
    return fig


def render_probabilistic_set_isosurface_3d(
    prob_set,
    bounding_box: Tuple[torch.Tensor, torch.Tensor],
    forward_samples: Optional[np.ndarray] = None,
    star_meshes: Optional[Iterable] = None,
    title: str = 'Probabilistic reachset + Star-union ground truth',
    out_html: Optional[Path] = None,
    resolution: int = 64,
    mesh_opacity: float = 0.45,
    star_opacity: float = 0.2,
    include_plotlyjs: str = 'directory',
    batch_size: int = 65536,
) -> 'plotly.graph_objects.Figure':
    """Render the conformal probabilistic reachset as a marching-cubes
    isosurface of its score function.

    Evaluates ``prob_set.score_fn`` on a ``resolution^3`` regular grid
    inside ``bounding_box``, then extracts the isosurface at
    ``prob_set.threshold``. Overlays forward-sampled network outputs and
    optionally a Star-union ground-truth mesh (same grid, clean visual
    comparison).

    Args:
        prob_set: :class:`ProbabilisticSet` whose sublevel set to render.
        bounding_box: ``(lo, hi)`` tensors / arrays of shape ``(3,)``.
        forward_samples: optional ``(N, 3)`` array drawn as scatter.
        star_meshes: optional iterable of n2v Stars; if given, their
            isosurface is overlaid in a distinct color.
        resolution: grid side length; score_fn is called resolution**3
            times (chunked via ``batch_size``).
        batch_size: chunk size for score_fn calls (GPU-memory cap).

    Raises ``ValueError`` when no grid cell satisfies ``score <= threshold``
    (bbox too small / threshold too tight).
    """
    import plotly.graph_objects as go
    from skimage import measure

    lo, hi = bounding_box
    lo_np = np.asarray(lo.cpu().numpy() if isinstance(lo, torch.Tensor) else lo,
                       dtype=np.float64).flatten()
    hi_np = np.asarray(hi.cpu().numpy() if isinstance(hi, torch.Tensor) else hi,
                       dtype=np.float64).flatten()
    if lo_np.shape != (3,) or hi_np.shape != (3,):
        raise ValueError("bounding_box must describe a 3-D box")

    xs = np.linspace(lo_np[0], hi_np[0], resolution)
    ys = np.linspace(lo_np[1], hi_np[1], resolution)
    zs = np.linspace(lo_np[2], hi_np[2], resolution)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    grid_t = torch.from_numpy(grid).to(torch.float32)

    scores = np.empty(grid.shape[0], dtype=np.float32)
    with torch.no_grad():
        for i in range(0, grid_t.shape[0], batch_size):
            chunk = grid_t[i:i + batch_size]
            scores[i:i + batch_size] = (
                prob_set.score_fn(chunk).detach().cpu().numpy()
            )

    score_vol = scores.reshape(xx.shape)
    level = float(prob_set.threshold)
    if score_vol.min() > level:
        raise ValueError(
            f"no grid cells satisfy score <= threshold ({level}); "
            f"score range [{score_vol.min():.3f}, {score_vol.max():.3f}]"
        )

    dx = tuple(((hi_np - lo_np) / (resolution - 1)).tolist())
    mc_verts, faces, _, _ = measure.marching_cubes(score_vol, level=level, spacing=dx)
    mc_verts = mc_verts + lo_np[None, :]

    traces = [go.Mesh3d(
        x=mc_verts[:, 0], y=mc_verts[:, 1], z=mc_verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=mesh_opacity, color='crimson',
        flatshading=True, showscale=False, name='flow reachset',
    )]

    if star_meshes is not None:
        # Overlay Star-union ground truth on the SAME grid so the two
        # surfaces visually align (no sampling-artefact offsets).
        inside = _star_membership(grid, star_meshes).reshape(xx.shape)
        field = inside.astype(np.float32)
        if field.max() > 0:
            sv, sf, _, _ = measure.marching_cubes(field, level=0.5, spacing=dx)
            sv = sv + lo_np[None, :]
            traces.append(go.Mesh3d(
                x=sv[:, 0], y=sv[:, 1], z=sv[:, 2],
                i=sf[:, 0], j=sf[:, 1], k=sf[:, 2],
                opacity=star_opacity, color='steelblue',
                flatshading=True, showscale=False,
                name='Star-union ground truth',
            ))

    if forward_samples is not None:
        traces.append(go.Scatter3d(
            x=forward_samples[:, 0],
            y=forward_samples[:, 1],
            z=forward_samples[:, 2],
            mode='markers',
            marker=dict(size=2, color='black', opacity=0.8),
            name='forward-sampled outputs',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if out_html is not None:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_html), include_plotlyjs=include_plotlyjs)
    return fig

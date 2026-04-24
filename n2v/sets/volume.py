"""Volume computation for reach sets represented as Stars or lists of Stars.

Three families of methods are provided:

* Monte-Carlo volume estimation with Hoeffding confidence intervals
  (:func:`star_volume`, :func:`star_union_volume_mc`).
* Exact (deterministic) single-Star volume via halfspace-intersection +
  simplex triangulation (:func:`star_volume` with ``method='exact'``).
* Sound (deterministic) lower bound on a Star union
  (:func:`star_union_volume_sound_lower`).

Sound upper bounds are deferred to a future iteration.

See ``docs/plans/2026-04-22-volume-toolbox.md`` for the design discussion
and motivation (resolved a membership-correctness issue in the flow-conformal
3D visualization that conflated the box envelope of Stars with their true
C/d-constrained images).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from n2v.sets.star import Star


logger = logging.getLogger(__name__)


@dataclass
class VolumeEstimate:
    """Container for a volume estimate.

    Attributes:
        mean: Point estimate of the volume.
        se: 1-sigma standard error for Monte-Carlo estimates. ``None`` for
            deterministic methods (exact and sound bounds).
        ci_low: Lower end of the confidence interval. Equal to ``mean`` for
            deterministic methods.
        ci_high: Upper end of the confidence interval. Equal to ``mean`` for
            deterministic methods.
        method: String tag identifying which method produced this estimate
            (e.g. ``'mc'``, ``'exact'``, ``'mc_union'``, ``'sound_lower/max_star'``).
        n_samples: Number of Monte-Carlo samples used. ``None`` for deterministic.
        meta: Open bag of method-specific metadata (bbox, confidence level,
            seed, notes about degenerate inputs, ...).
    """

    mean: float
    se: Optional[float]
    ci_low: float
    ci_high: float
    method: str
    n_samples: Optional[int]
    meta: dict = field(default_factory=dict)


def _hoeffding_half_width(
    scale: float, n_samples: int, confidence: float,
) -> float:
    """Hoeffding half-width for a bounded [0, scale] MC estimate.

    P(|mean_hat - mean| >= eps) <= 2 exp(-2 n eps^2 / scale^2). Setting the
    RHS equal to ``1 - confidence`` yields
    ``eps = scale * sqrt(ln(2 / (1 - c)) / (2 n))``.
    """
    if n_samples <= 0:
        return float('inf')
    return scale * math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2.0 * n_samples))


def _predicate_bounds(star: Star) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(plb, pub)`` from the Star, defaulting to -1/+1 arrays if unset."""
    plb = star.predicate_lb
    pub = star.predicate_ub
    if plb is None:
        plb = -np.ones(star.nVar)
    if pub is None:
        pub = np.ones(star.nVar)
    plb = np.asarray(plb, dtype=np.float64).flatten()
    pub = np.asarray(pub, dtype=np.float64).flatten()
    return plb, pub


def _predicate_box_is_feasible(plb: np.ndarray, pub: np.ndarray) -> bool:
    """Return False if ``plb > pub`` anywhere, i.e. the box is empty."""
    return bool(np.all(plb <= pub + 1e-12))


def _sample_predicate_polytope_mc(
    plb: np.ndarray, pub: np.ndarray,
    C: Optional[np.ndarray], d: Optional[np.ndarray],
    n_samples: int, seed: int,
) -> Tuple[int, int]:
    """MC count of points inside the predicate polytope.

    Draws ``n_samples`` uniform points from the box ``[plb, pub]``, then
    counts how many also satisfy ``C @ alpha <= d`` (if any C).

    Returns:
        ``(inside, n_samples)`` tuple of ints.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(plb, pub, size=(n_samples, plb.shape[0]))
    if C is None or C.size == 0:
        inside = n_samples
    else:
        C_arr = np.asarray(C, dtype=np.float64)
        d_arr = np.asarray(d, dtype=np.float64).flatten()
        mask = ((pts @ C_arr.T) <= d_arr[None, :] + 1e-12).all(axis=1)
        inside = int(mask.sum())
    return inside, n_samples


def _predicate_polytope_vertices(
    plb: np.ndarray, pub: np.ndarray,
    C: Optional[np.ndarray], d: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Vertex enumeration for the predicate polytope
    ``{alpha : plb <= alpha <= pub, C @ alpha <= d}``.

    Uses :class:`scipy.spatial.HalfspaceIntersection` once we have located
    an interior feasible point via LP. Returns ``None`` if the polytope
    is empty or lower-dimensional (no 3D interior point exists).
    """
    from scipy.optimize import linprog

    dim = plb.shape[0]
    # Assemble halfspaces in the ``A x + b <= 0`` form that HalfspaceIntersection
    # expects: each row is ``[a_1, ..., a_d, b]``.
    rows = []
    # Lower bound: -alpha_k <= -plb_k.
    rows.append(np.concatenate([-np.eye(dim), plb.reshape(-1, 1)], axis=1))
    # Upper bound: alpha_k <= pub_k.
    rows.append(np.concatenate([np.eye(dim), -pub.reshape(-1, 1)], axis=1))
    if C is not None and np.asarray(C).size > 0:
        C_arr = np.asarray(C, dtype=np.float64)
        d_arr = np.asarray(d, dtype=np.float64).flatten()
        rows.append(np.concatenate([C_arr, -d_arr.reshape(-1, 1)], axis=1))
    halfspaces = np.concatenate(rows, axis=0)

    # Interior point via Chebyshev center LP: maximise r s.t. A_i x + ||a_i|| r <= -b_i.
    norm_A = np.linalg.norm(halfspaces[:, :-1], axis=1, keepdims=True)
    A_ub = np.concatenate([halfspaces[:, :-1], norm_A], axis=1)
    b_ub = -halfspaces[:, -1]
    c_obj = np.concatenate([np.zeros(dim), [-1.0]])  # maximise r -> minimise -r
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * dim + [(0, None)])
    if not res.success or res.x[-1] <= 0:
        # No strictly-interior point -> polytope is empty or lower-dim.
        return None
    interior = res.x[:dim]

    try:
        from scipy.spatial import HalfspaceIntersection
        hs = HalfspaceIntersection(halfspaces, interior)
        return np.asarray(hs.intersections)
    except Exception:
        return None


def _simplex_volume(simplex: np.ndarray) -> float:
    """Euclidean volume of a ``d+1``-vertex simplex in R^d."""
    d = simplex.shape[1]
    mat = simplex[1:] - simplex[0:1]
    return abs(np.linalg.det(mat)) / math.factorial(d)


def star_volume(
    star: Star,
    method: str = 'mc',
    n_samples: int = 100_000,
    seed: int = 0,
    confidence: float = 0.99,
) -> VolumeEstimate:
    """Volume of a single :class:`Star`.

    Two methods are supported:

    * ``method='mc'`` (default): Monte-Carlo over the predicate box with a
      C/d rejection step. The MC count is multiplied by the predicate-box
      volume and ``|det(V[:, 1:])|`` (the Jacobian of the affine map from
      predicate to output space). Returns a Hoeffding-bounded
      :class:`VolumeEstimate`.
    * ``method='exact'``: vertex-enumerates the predicate polytope via
      :class:`scipy.spatial.HalfspaceIntersection`, triangulates the
      vertex hull with Delaunay, sums simplex volumes, then multiplies by
      ``|det(V[:, 1:])|``. Deterministic; supports arbitrary dim but
      scales poorly above roughly d=5.

    Degenerate cases:

    * If ``V[:, 1:]`` is rank-deficient, ``|det|`` = 0 — the Star's image is
      a measure-zero submanifold and volume is returned as exactly ``0.0``
      (with ``meta['notes']`` explaining).
    * If the predicate box ``[plb, pub]`` is infeasible (``plb > pub`` any
      component) or the ``C/d`` constraints make the feasible set empty,
      volume is ``0.0``.

    Args:
        star: The Star.
        method: ``'mc'`` or ``'exact'``.
        n_samples: Number of MC samples. Ignored by ``'exact'``.
        seed: RNG seed for MC. Ignored by ``'exact'``.
        confidence: Hoeffding confidence level (default 99%). Ignored by
            ``'exact'``.

    Returns:
        :class:`VolumeEstimate`.

    Raises:
        ValueError: If ``method`` is not one of the supported values.
    """
    if method not in ('mc', 'exact'):
        raise ValueError(f"unknown method {method!r}; expected 'mc' or 'exact'")
    if star.V is None or star.V.size == 0:
        return VolumeEstimate(
            mean=0.0, se=0.0, ci_low=0.0, ci_high=0.0,
            method=method, n_samples=(None if method == 'exact' else n_samples),
            meta={'notes': 'empty Star'},
        )

    basis = np.asarray(star.V[:, 1:], dtype=np.float64)
    plb, pub = _predicate_bounds(star)

    # Degenerate: empty predicate box.
    if not _predicate_box_is_feasible(plb, pub):
        return VolumeEstimate(
            mean=0.0, se=0.0, ci_low=0.0, ci_high=0.0,
            method=method, n_samples=(None if method == 'exact' else n_samples),
            meta={'notes': 'infeasible predicate box'},
        )

    # Basis Jacobian. For square basis use det directly; for non-square use
    # sqrt(det(B^T B)), which is the Lebesgue factor of the affine map onto
    # the image subspace.
    n_out, n_var = basis.shape
    if n_out == n_var:
        det_scale = abs(float(np.linalg.det(basis)))
    elif n_out > n_var:
        gram = basis.T @ basis
        det_scale = math.sqrt(max(0.0, float(np.linalg.det(gram))))
    else:
        # Wide basis: image fills R^{n_out} but alpha has extra dims.
        # The relevant integration measure reduces to a marginal; for the
        # purpose of the output-space volume, use the row-space factor.
        det_scale = math.sqrt(max(0.0, float(np.linalg.det(basis @ basis.T))))

    if det_scale == 0.0:
        return VolumeEstimate(
            mean=0.0, se=0.0, ci_low=0.0, ci_high=0.0,
            method=method, n_samples=(None if method == 'exact' else n_samples),
            meta={'notes': 'rank-deficient basis (measure-zero image)'},
        )

    C = star.C if (star.C is not None and np.asarray(star.C).size > 0) else None
    d = star.d if (star.d is not None and np.asarray(star.d).size > 0) else None

    box_vol = float(np.prod(pub - plb))

    if method == 'mc':
        inside, n_total = _sample_predicate_polytope_mc(
            plb, pub, C, d, n_samples=n_samples, seed=seed,
        )
        frac = inside / n_total
        mean = frac * box_vol * det_scale
        se = math.sqrt(frac * (1 - frac) / n_total) * box_vol * det_scale
        half = _hoeffding_half_width(box_vol * det_scale, n_total, confidence)
        return VolumeEstimate(
            mean=mean, se=se,
            ci_low=max(0.0, mean - half), ci_high=mean + half,
            method='mc', n_samples=n_total,
            meta={
                'confidence': confidence,
                'box_vol': box_vol,
                'det_scale': det_scale,
            },
        )

    # Exact.
    verts = _predicate_polytope_vertices(plb, pub, C, d)
    if verts is None or verts.shape[0] < plb.shape[0] + 1:
        return VolumeEstimate(
            mean=0.0, se=None, ci_low=0.0, ci_high=0.0,
            method='exact', n_samples=None,
            meta={'notes': 'predicate polytope empty or lower-dimensional'},
        )
    from scipy.spatial import Delaunay
    try:
        tri = Delaunay(verts)
    except Exception as exc:  # numerically-degenerate vertex set
        return VolumeEstimate(
            mean=0.0, se=None, ci_low=0.0, ci_high=0.0,
            method='exact', n_samples=None,
            meta={'notes': f'Delaunay failure: {exc}'},
        )
    total = 0.0
    for simplex_idx in tri.simplices:
        total += _simplex_volume(verts[simplex_idx])
    mean = total * det_scale
    return VolumeEstimate(
        mean=mean, se=None, ci_low=mean, ci_high=mean,
        method='exact', n_samples=None,
        meta={'box_vol': box_vol, 'det_scale': det_scale, 'n_vertices': int(verts.shape[0])},
    )


def star_union_volume_sound_lower(
    stars: List[Star],
    method: str = 'max_star',
) -> VolumeEstimate:
    """Sound (deterministic) lower bound on a Star union's volume.

    Currently supports:

    * ``method='max_star'``: the largest individual Star's exact volume.
      Trivially sound — the union is at least as big as any member — and
      often loose. Useful as a sanity rail below any MC estimate.

    Args:
        stars: Non-empty list of Stars.
        method: only ``'max_star'`` is implemented for now.

    Returns:
        :class:`VolumeEstimate` with ``method='sound_lower/max_star'``.

    Raises:
        ValueError: on unknown ``method`` or empty ``stars``.
    """
    if not stars:
        raise ValueError("star_union_volume_sound_lower requires a non-empty list")
    if method != 'max_star':
        raise ValueError(f"unknown method {method!r}; expected 'max_star'")

    best_vol = 0.0
    best_idx = -1
    for i, star in enumerate(stars):
        v = star_volume(star, method='exact').mean
        if v > best_vol:
            best_vol = v
            best_idx = i
    return VolumeEstimate(
        mean=best_vol, se=None, ci_low=best_vol, ci_high=best_vol,
        method='sound_lower/max_star', n_samples=None,
        meta={'argmax_star': best_idx, 'n_stars': len(stars)},
    )


def _per_star_bboxes(stars: List[Star]) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bboxes for every Star's image. Returns ``(lo, hi)`` both
    shaped ``(K, dim)``. Used by the bbox prefilter to skip expensive LPs
    for points that can't possibly lie in a given Star.
    """
    lo_list, hi_list = [], []
    for star in stars:
        box = star.get_box()
        lo_list.append(np.asarray(box.lb).flatten())
        hi_list.append(np.asarray(box.ub).flatten())
    return np.stack(lo_list), np.stack(hi_list)


def _mc_worker_chunk(args):
    """Worker: count points inside the union for a chunk of samples.

    Defined at module level so ``multiprocessing`` / ``loky`` can pickle it.
    Takes a dict of already-serialized star components (V, plb, pub, C, d)
    so scipy-HiGHS / pure-numpy containment can run in a fresh interpreter
    without re-importing heavy Star machinery.
    """
    pts = args['pts']
    star_blobs = args['star_blobs']
    star_lo = args['star_lo']
    star_hi = args['star_hi']
    contains_method = args['contains_method']
    lp_solver = args['lp_solver']

    from n2v.sets.star import Star as _Star

    # Reconstruct Stars once per worker (cheap compared to the LPs).
    local_stars = []
    for blob in star_blobs:
        s = _Star(
            V=blob['V'], C=blob['C'], d=blob['d'],
            pred_lb=blob['plb'], pred_ub=blob['pub'],
        )
        local_stars.append(s)

    n_this = pts.shape[0]
    in_bbox = (
        (pts[:, None, :] >= star_lo[None, :, :]).all(axis=2)
        & (pts[:, None, :] <= star_hi[None, :, :]).all(axis=2)
    )
    inside = 0
    n_lps = 0
    for i in range(n_this):
        candidates = np.nonzero(in_bbox[i])[0]
        for k in candidates:
            n_lps += 1
            if local_stars[k].contains(
                pts[i], method=contains_method, lp_solver=lp_solver,
            ):
                inside += 1
                break
    return inside, n_this, n_lps


def star_union_volume_mc(
    stars: List[Star],
    bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    n_samples: int = 1_000_000,
    batch_size: int = 50_000,
    seed: int = 0,
    confidence: float = 0.99,
    lp_solver: str = 'default',
    contains_method: str = 'lp',
    n_workers: int = 1,
) -> VolumeEstimate:
    """Monte-Carlo volume estimate of the union of a list of Stars.

    Procedure:

    1. If ``bbox`` is None, compute the union bbox via :func:`star_union_bbox`.
    2. Precompute each Star's axis-aligned bbox for a cheap pre-filter.
    3. Draw ``n_samples`` uniformly in ``bbox``. For each sample:
       iterate the Stars with bbox pre-filter + short-circuit. If the sample
       is inside any Star's bbox, run :meth:`Star.contains` on that Star;
       on the first positive hit, mark the point inside and stop.
    4. ``vol = (inside / n) * bbox_vol`` with Hoeffding confidence interval.

    Args:
        stars: list of Stars whose union is of interest.
        bbox: optional ``(lo, hi)`` to use; otherwise computed tightly
            from the union of per-Star bboxes.
        n_samples: total number of MC samples.
        batch_size: draw this many points at a time (memory / batching).
        seed: RNG seed.
        confidence: Hoeffding confidence level (default 99%).
        lp_solver: passed through to :meth:`Star.contains`.
        contains_method: ``'lp'`` (default) or ``'algebraic'``.
        n_workers: parallel worker count. ``1`` (default) keeps the original
            single-process loop; ``>1`` splits samples across processes via
            ``multiprocessing`` so LP-based containment scales out to all
            cores. ``-1`` means ``os.cpu_count()``. Each worker reconstructs
            Stars locally from a picklable blob, so the per-start overhead
            is ``n_stars * dim * 8B``-ish — negligible for a few thousand
            stars.

    Returns:
        :class:`VolumeEstimate` with ``method='mc_union'``.

    Raises:
        ValueError: If ``stars`` is empty.
    """
    if not stars:
        raise ValueError("star_union_volume_mc requires a non-empty list of Stars")

    if bbox is None:
        lo, hi = star_union_bbox(stars)
    else:
        lo = np.asarray(bbox[0], dtype=np.float64).flatten()
        hi = np.asarray(bbox[1], dtype=np.float64).flatten()

    bbox_vol = float(np.prod(hi - lo))
    if bbox_vol <= 0.0:
        return VolumeEstimate(
            mean=0.0, se=0.0, ci_low=0.0, ci_high=0.0,
            method='mc_union', n_samples=n_samples,
            meta={'notes': 'degenerate bbox (zero volume)'},
        )

    star_lo, star_hi = _per_star_bboxes(stars)
    rng = np.random.default_rng(seed)
    inside = 0
    total = 0
    n_lps = 0
    dim = lo.shape[0]

    # Resolve worker count.
    import os
    if n_workers == -1:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, int(n_workers))

    if n_workers == 1:
        # Original single-process path — byte-identical to pre-change code.
        for start in range(0, n_samples, batch_size):
            n_this = min(batch_size, n_samples - start)
            pts = rng.uniform(lo, hi, size=(n_this, dim))
            in_bbox = (
                (pts[:, None, :] >= star_lo[None, :, :]).all(axis=2)
                & (pts[:, None, :] <= star_hi[None, :, :]).all(axis=2)
            )
            for i in range(n_this):
                candidate_idxs = np.nonzero(in_bbox[i])[0]
                for k in candidate_idxs:
                    n_lps += 1
                    if stars[k].contains(
                        pts[i], method=contains_method, lp_solver=lp_solver,
                    ):
                        inside += 1
                        break
            total += n_this
    else:
        # Multiprocess path: draw all samples up-front on the master (RNG
        # remains deterministic for a given seed), chunk into worker-sized
        # tasks, fan out via multiprocessing.Pool. Each LP holds the GIL
        # release inside HiGHS, so multiprocessing is the safe primitive;
        # threading would also work for LP but falls back to serial for
        # pure-numpy algebraic containment.
        from multiprocessing import get_context
        # Serialize stars once into plain numpy blobs so workers avoid
        # pickling the full Star object (which pulls in circular imports).
        star_blobs = []
        for s in stars:
            star_blobs.append({
                'V': np.asarray(s.V, dtype=np.float64),
                'C': (None if s.C is None or np.asarray(s.C).size == 0
                      else np.asarray(s.C, dtype=np.float64)),
                'd': (None if s.d is None or np.asarray(s.d).size == 0
                      else np.asarray(s.d, dtype=np.float64).flatten()),
                'plb': (None if s.predicate_lb is None
                        else np.asarray(s.predicate_lb, dtype=np.float64).flatten()),
                'pub': (None if s.predicate_ub is None
                        else np.asarray(s.predicate_ub, dtype=np.float64).flatten()),
            })

        # Pre-generate ALL sample points here (deterministic w.r.t. seed),
        # then split into worker chunks.
        pts_all = rng.uniform(lo, hi, size=(n_samples, dim))
        chunks = []
        for start in range(0, n_samples, batch_size):
            chunk_pts = pts_all[start:start + batch_size]
            chunks.append({
                'pts': chunk_pts,
                'star_blobs': star_blobs,
                'star_lo': star_lo,
                'star_hi': star_hi,
                'contains_method': contains_method,
                'lp_solver': lp_solver,
            })

        ctx = get_context('spawn')  # safer across CUDA-loaded parents
        with ctx.Pool(processes=min(n_workers, len(chunks))) as pool:
            for got_inside, got_total, got_lps in pool.imap_unordered(
                _mc_worker_chunk, chunks,
            ):
                inside += got_inside
                total += got_total
                n_lps += got_lps

    frac = inside / total
    mean = frac * bbox_vol
    se = math.sqrt(frac * (1 - frac) / total) * bbox_vol
    half = _hoeffding_half_width(bbox_vol, total, confidence)
    return VolumeEstimate(
        mean=mean, se=se,
        ci_low=max(0.0, mean - half), ci_high=mean + half,
        method='mc_union', n_samples=total,
        meta={
            'confidence': confidence,
            'bbox': (lo.tolist(), hi.tolist()),
            'bbox_vol': bbox_vol,
            'n_lp_calls': n_lps,
            'contains_method': contains_method,
            'n_workers': n_workers,
        },
    )


def star_union_bbox(stars: List[Star]) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box fully containing every Star's image.

    The returned ``(lo, hi)`` arrays have shape ``(dim,)`` and satisfy
    ``lo[k] <= y[k] <= hi[k]`` for every ``y`` in the union. The bbox is
    computed via each Star's :meth:`Star.get_box` (tight per-Star axis ranges
    via LP) and then merged elementwise.

    Args:
        stars: Non-empty list of Stars. All must share the same output
            dimension.

    Returns:
        ``(lo, hi)`` tuple of 1-D numpy arrays.

    Raises:
        ValueError: If ``stars`` is empty or Stars have inconsistent
            output dimensions.
    """
    if not stars:
        raise ValueError("star_union_bbox requires a non-empty list of Stars")

    dim = stars[0].dim
    lo = np.full(dim, np.inf)
    hi = np.full(dim, -np.inf)
    for idx, star in enumerate(stars):
        if star.dim != dim:
            raise ValueError(
                f"star {idx} has dim {star.dim}, expected {dim}"
            )
        box = star.get_box()
        lo = np.minimum(lo, np.asarray(box.lb).flatten())
        hi = np.maximum(hi, np.asarray(box.ub).flatten())
    return lo, hi


# -----------------------------------------------------------------------------
# Utilities promoted from experiments/hashemi_comparison/v2_adaptive_reg/
# (previously `v2.compute_mc_bbox`, `v2.exact_volume_2d`). Used throughout
# examples/FlowConformal/ for bounding-box construction on forward samples and
# 2D Star-union rasterization.
# -----------------------------------------------------------------------------


def compute_mc_bbox(
    network,
    x_center_np: np.ndarray,
    radius: float,
    output_dim: int,
    n_samples: int = 5000,
    pad: float = 1.0,
) -> Tuple['torch.Tensor', 'torch.Tensor']:
    """Bounding box covering the reach set, constructed from forward samples.

    Draws ``n_samples`` uniform points from the L-infinity input ball around
    ``x_center_np``, pushes them through ``network``, then returns the
    axis-aligned bounding box of the outputs padded by ``pad`` on each side.
    Intended as the sampling region for MC volume estimation.

    Args:
        network: A callable that maps an ``(n, input_dim)`` torch float tensor
            to an ``(n, output_dim)`` tensor (e.g. ``torch.nn.Module``).
        x_center_np: ``(input_dim,)`` numpy array giving the input-ball center.
        radius: Half-width of the L-infinity input ball.
        output_dim: Dimensionality of the network output.
        n_samples: How many forward samples to push through the network.
        pad: Symmetric padding added to each axis of the empirical bbox.

    Returns:
        ``(lo, hi)`` — a pair of float32 torch tensors, both of shape
        ``(output_dim,)``.
    """
    import torch
    rng = torch.Generator().manual_seed(12345)
    x = torch.tensor(x_center_np, dtype=torch.float32) + (
        torch.rand(n_samples, x_center_np.shape[0], generator=rng) * 2 - 1
    ) * radius
    with torch.no_grad():
        y = network(x).numpy()
    lo = y.min(axis=0) - pad
    hi = y.max(axis=0) + pad
    return (
        torch.tensor(lo[:output_dim], dtype=torch.float32),
        torch.tensor(hi[:output_dim], dtype=torch.float32),
    )


def _star_polygon_2d(
    star: Star,
    rng: np.random.Generator,
    n_samples: int = 600,
) -> Optional[np.ndarray]:
    """For a 2D output star, sample the predicate polytope and return the
    convex hull of the image points as a ``(k, 2)`` array of vertices.
    Returns ``None`` for degenerate stars (rank-deficient basis, collinear
    samples, etc.).
    """
    from scipy.spatial import ConvexHull

    V = star.V
    offset = V[:, 0]
    basis = V[:, 1:]
    nVar = star.nVar
    if nVar == 0:
        return None
    plb = np.asarray(star.predicate_lb, dtype=np.float64).flatten()
    pub = np.asarray(star.predicate_ub, dtype=np.float64).flatten()
    accepted = []
    for _ in range(40):
        alpha = rng.uniform(plb, pub, size=(n_samples * 4, nVar))
        if star.C is not None and np.asarray(star.C).size > 0:
            mask = (star.C @ alpha.T <= star.d + 1e-9).all(axis=0)
            alpha = alpha[mask]
        if len(alpha) > 0:
            accepted.append(alpha)
            if sum(len(a) for a in accepted) >= n_samples:
                break
    if not accepted:
        return None
    alpha_all = np.vstack(accepted)
    y_samples = offset[None, :] + alpha_all @ basis.T
    if len(y_samples) < 3:
        return None
    if np.ptp(y_samples, axis=0).min() < 1e-6:
        return None
    try:
        hull = ConvexHull(y_samples)
    except Exception:
        return None
    return y_samples[hull.vertices]


def exact_volume_2d(
    stars: List[Star],
    bbox: Tuple[np.ndarray, np.ndarray],
    resolution: int = 500,
) -> float:
    """2D area of a Star union via occupancy rasterization.

    For 2D Star unions (e.g. the banana benchmark's reach set), exact volume
    via polytope triangulation is overkill — a fine raster of the bounding
    box combined with ``matplotlib.path.Path.contains_points`` for each
    Star's 2D convex hull is fast and accurate enough for verification.

    Steps:
        1. Per Star, sample its predicate polytope, push through the basis,
           and take the convex hull of the image points — gives a 2D polygon.
        2. Rasterize the union: test every pixel of the grid for membership
           in any polygon.
        3. Clean up with binary closing + hole filling to handle sampling
           artefacts on the polygon boundaries.
        4. Return (filled-pixel count) × cell area.

    Args:
        stars: List of Stars with 2D output.
        bbox: ``((xmin, ymin), (xmax, ymax))`` bounding box for rasterization.
            Each element can be a numpy array / torch tensor / tuple.
        resolution: Grid side length. Runtime scales as ``resolution^2``.

    Returns:
        Approximate area of the union as a scalar float. Returns ``0.0`` if
        no Star produced a valid polygon (all rank-deficient or collinear).
    """
    import matplotlib.path as mpath
    from scipy.ndimage import binary_closing, binary_fill_holes

    rng = np.random.default_rng(0)
    polygons = []
    for s in stars:
        poly = _star_polygon_2d(s, rng)
        if poly is not None:
            polygons.append(poly)
    if not polygons:
        return 0.0
    xmin, ymin = float(bbox[0][0]), float(bbox[0][1])
    xmax, ymax = float(bbox[1][0]), float(bbox[1][1])
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    occ = np.zeros(pts.shape[0], dtype=bool)
    for poly in polygons:
        occ |= mpath.Path(poly).contains_points(pts)
    mask = occ.reshape(resolution, resolution)
    mask = binary_closing(mask, iterations=2)
    mask = binary_fill_holes(mask)
    cell_area = (xmax - xmin) * (ymax - ymin) / (resolution * resolution)
    return float(mask.sum() * cell_area)

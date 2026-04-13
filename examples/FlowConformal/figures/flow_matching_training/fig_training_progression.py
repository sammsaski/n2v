"""
Training Progression Figure (for paper).

Produces overlay.png: a single-axis figure with flow-conformal reach
sets from four training snapshots (0, 2, 5, 1500 epochs) drawn as
nested translucent blue polygons, with the exact reach set (n2v Star
propagation) overlaid as a solid black boundary. Numbered circles on
each ring's left edge map to the epoch count.

All snapshots come from a single flow training run (fixed seed,
checkpointed at each snapshot epoch) so the only thing changing across
rings is the amount of training the flow has received.
"""

import os
import sys
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'examples'))

from n2v.probabilistic.flow import (
    VelocityField, FlowODE, FlowScore, calibrate, ProbabilisticSet,
)
from n2v.probabilistic.flow.train import sinkhorn_coupling
from n2v.sets.box import Box
from n2v.nn import NeuralNetwork
from FlowConformal.networks import RotatedBananaNet


FIGURE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---- Configuration ----
# Log-spaced snapshots plus one "fully trained" endpoint.
SNAPSHOT_EPOCHS = [0, 2, 5, 25, 100, 400, 1500]
N_TRAIN = 2000
N_CALIB = 8000
FLOW_HIDDEN = 64
FLOW_N_LAYERS = 4
BATCH_SIZE = 256
LR = 1e-3
EPSILON_1 = 0.001
SEED = 42

# Samples per exact star for polygon hull
SAMPLES_PER_STAR = 500


def compute_exact_reach_polygons(net, lb, ub):
    """Run n2v exact reachability on the banana net and return a list
    of 2D polygons (one per output Star), each as an (V, 2) ndarray of
    convex-hull vertices in CCW order."""
    box = Box(
        np.asarray(lb, dtype=float).reshape(-1, 1),
        np.asarray(ub, dtype=float).reshape(-1, 1),
    )
    wrapper = NeuralNetwork(net.net)
    stars = wrapper.reach(box.to_star(), method='exact')
    print(f"  exact reach produced {len(stars)} output stars")

    polygons = []
    rng = np.random.default_rng(0)
    for star in stars:
        V = star.V  # shape (2, nVar+1)
        offset = V[:, 0]
        basis = V[:, 1:]  # (2, nVar)
        nVar = star.nVar
        plb = star.predicate_lb.flatten()
        pub = star.predicate_ub.flatten()

        # Sample alpha uniformly from the predicate box, reject those
        # violating C*alpha <= d.
        n_accept = 0
        accepted = []
        n_try = 0
        max_try = 50
        while n_accept < SAMPLES_PER_STAR and n_try < max_try:
            n_try += 1
            alpha = rng.uniform(plb, pub, size=(SAMPLES_PER_STAR * 4, nVar))
            if star.C is not None and star.C.size > 0:
                mask = (star.C @ alpha.T <= star.d + 1e-9).all(axis=0)
                alpha = alpha[mask]
            if len(alpha) == 0:
                continue
            accepted.append(alpha)
            n_accept += len(alpha)
        if n_accept == 0:
            continue
        alpha_all = np.vstack(accepted)
        # y = offset + basis @ alpha^T
        y_samples = offset[None, :] + alpha_all @ basis.T

        if len(y_samples) < 3:
            continue

        # Degenerate (zero-area) stars show up when ReLU collapses a
        # dimension. Skip them for plotting — they contribute a line or
        # point to the reach set which is visually invisible anyway.
        if np.ptp(y_samples, axis=0).min() < 1e-6:
            continue

        try:
            hull = ConvexHull(y_samples)
        except Exception:
            continue
        verts = y_samples[hull.vertices]
        polygons.append(verts)

    return polygons


def polygons_bbox(polygons):
    """Return (lo, hi) bounding box of all polygon vertices."""
    all_pts = np.vstack(polygons)
    return all_pts.min(axis=0), all_pts.max(axis=0)


def train_with_checkpoints(vf, training_outputs, n_epochs, checkpoint_epochs,
                            batch_size=256, lr=1e-3):
    """Train an OT-CFM velocity field while saving state_dict snapshots
    at the requested epoch indices. Returns a dict mapping
    epoch -> state_dict (including epoch=0 for the fresh, untrained flow).
    """
    checkpoints = {}
    checkpoint_set = set(checkpoint_epochs)
    if 0 in checkpoint_set:
        checkpoints[0] = copy.deepcopy(vf.state_dict())

    optimizer = torch.optim.Adam(vf.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    dataset = torch.utils.data.TensorDataset(training_outputs)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in range(1, n_epochs + 1):
        for (x1_batch,) in loader:
            x0_batch = torch.randn_like(x1_batch)
            x0_batch, x1_batch = sinkhorn_coupling(x0_batch, x1_batch)
            t = torch.rand(x1_batch.shape[0], device=x1_batch.device)
            x_t = (1 - t.unsqueeze(1)) * x0_batch + t.unsqueeze(1) * x1_batch
            target_v = x1_batch - x0_batch
            pred_v = vf(t, x_t)
            loss = F.mse_loss(pred_v, target_v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch in checkpoint_set:
            checkpoints[epoch] = copy.deepcopy(vf.state_dict())
    return checkpoints


class CenteredFlowScore:
    def __init__(self, flow_score, center):
        self.flow_score = flow_score
        self.center = center

    def __call__(self, y):
        return self.flow_score(y - self.center)


def build_pset_for_checkpoint(vf, state_dict, y_calib, center):
    """Load a state_dict into vf, calibrate on y_calib, and return a
    ProbabilisticSet for membership / boundary queries."""
    vf.load_state_dict(state_dict)
    vf.eval()
    flow_ode = FlowODE(vf)
    flow_score = CenteredFlowScore(FlowScore(flow_ode, t=1.0), center)
    with torch.no_grad():
        calib_scores = flow_score(y_calib)
    ell = N_CALIB - 1
    threshold = calibrate(calib_scores, ell).item()
    pset = ProbabilisticSet(
        score_fn=flow_score,
        threshold=threshold,
        m=N_CALIB, ell=ell, epsilon=EPSILON_1,
        dim=2,
    )
    return pset, threshold


def run():
    print("=" * 70)
    print("Training Progression Figure")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\nBuilding RotatedBananaNet...")
    net = RotatedBananaNet()

    # Compute exact reach polygons once.
    print("\nComputing exact reach set via n2v star propagation...")
    t0 = time.time()
    polygons = compute_exact_reach_polygons(
        net, lb=[0.0, 0.0], ub=[1.0, 1.0]
    )
    print(f"  got {len(polygons)} polygons in {time.time() - t0:.2f}s")

    # Bounding box for boundary evaluation: tight around the exact set
    # plus a pad. The pad needs to cover the untrained flow reach set
    # too, which is a wide L2 ball.
    poly_lo, poly_hi = polygons_bbox(polygons)
    pad = 1.0
    bbox = (
        torch.tensor([poly_lo[0] - pad, poly_lo[1] - pad], dtype=torch.float32),
        torch.tensor([poly_hi[0] + pad, poly_hi[1] + pad], dtype=torch.float32),
    )
    print(f"  polygon bbox: {poly_lo} to {poly_hi}")

    # Sample training data for the flow.
    torch.manual_seed(SEED)
    x_train = torch.rand(N_TRAIN, 2)
    x_calib = torch.rand(N_CALIB, 2)
    with torch.no_grad():
        y_train = net(x_train)
        y_calib = net(x_calib)
    center = y_train.mean(dim=0)
    y_train_c = y_train - center

    # Fresh velocity field, train with checkpoints.
    # Cache checkpoints to disk so plot-style iteration doesn't retrain.
    cache_path = os.path.join(
        FIGURE_DIR, 'fig_training_progression_checkpoints.pt'
    )
    torch.manual_seed(SEED)
    vf = VelocityField(dim=2, hidden=FLOW_HIDDEN, n_layers=FLOW_N_LAYERS)

    if os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
        if cache.get('snapshot_epochs') == SNAPSHOT_EPOCHS:
            print(f"\nLoading cached checkpoints from {cache_path}")
            checkpoints = cache['checkpoints']
        else:
            checkpoints = None
    else:
        checkpoints = None

    if checkpoints is None:
        max_epochs = max(SNAPSHOT_EPOCHS)
        print(f"\nTraining flow for {max_epochs} epochs with checkpoints "
              f"at {SNAPSHOT_EPOCHS}...")
        t0 = time.time()
        checkpoints = train_with_checkpoints(
            vf, y_train_c,
            n_epochs=max_epochs,
            checkpoint_epochs=SNAPSHOT_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
        )
        print(f"  training done in {time.time() - t0:.1f}s, "
              f"{len(checkpoints)} checkpoints saved")
        torch.save(
            {'snapshot_epochs': SNAPSHOT_EPOCHS, 'checkpoints': checkpoints},
            cache_path,
        )
        print(f"  cached to {cache_path}")

    # Build ProbabilisticSets + contours per snapshot.
    snapshot_data = []
    for epoch in SNAPSHOT_EPOCHS:
        pset, threshold = build_pset_for_checkpoint(
            vf, checkpoints[epoch], y_calib, center
        )
        t0 = time.time()
        volume, vol_se = pset.estimate_volume(
            n_samples=200_000, bounding_box=bbox
        )
        contours = pset.boundary_2d(resolution=400, bounds=bbox)
        print(f"  epoch {epoch}: threshold={threshold:.4f} "
              f"volume={volume:.4f} ({time.time() - t0:.1f}s)")
        snapshot_data.append({
            'epoch': epoch,
            'threshold': threshold,
            'volume': volume,
            'contours': contours,
        })

    # Produce the single-axis overlay figure.
    _plot_overlay(snapshot_data, polygons, bbox)


def _exact_union_boundary(polygons, bbox, resolution=600):
    """Compute the boundary of the union of exact Star polygons by
    rasterizing occupancy on a grid and extracting a 0.5 contour.
    Applies morphological closing to eliminate sub-pixel gaps between
    adjacent polygons that would otherwise show as spurious interior
    contour lines."""
    import matplotlib.path as mpath
    from scipy.ndimage import binary_closing, binary_fill_holes

    xmin, ymin = bbox[0][0].item(), bbox[0][1].item()
    xmax, ymax = bbox[1][0].item(), bbox[1][1].item()
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    occupancy = np.zeros(pts.shape[0], dtype=bool)
    for poly in polygons:
        path = mpath.Path(poly)
        occupancy |= path.contains_points(pts)
    mask = occupancy.reshape(resolution, resolution)
    # Close sub-pixel gaps between tiles, then fill any interior holes
    # so stray pixels don't produce phantom contour lines.
    mask = binary_closing(mask, iterations=3)
    mask = binary_fill_holes(mask)
    return xs, ys, mask.astype(float)


def _plot_overlay(snapshot_data, polygons, bbox):
    """Single axis with all flow reach sets drawn as nested filled
    blue polygons (darker = later epoch), exact reach set as a solid
    black boundary on top. Polygons are numbered 1..N from outer (least
    trained) to inner (most trained)."""
    print("\nPlotting single-axis overlay layout...")

    # Subset of epochs for the overlay.
    overlay_epochs = {0, 2, 5, 1500}
    subset = [d for d in snapshot_data if d['epoch'] in overlay_epochs]

    # Recompute a tight bbox from the flow contours we actually draw
    # plus the exact polygons, with a small margin so the outer ring
    # doesn't touch the frame.
    all_pts = []
    for data in subset:
        for seg in data['contours']:
            all_pts.append(np.asarray(seg))
    for poly in polygons:
        all_pts.append(np.asarray(poly))
    all_pts = np.vstack(all_pts)
    tight_lo = all_pts.min(axis=0)
    tight_hi = all_pts.max(axis=0)
    pad_frac = 0.08
    span = tight_hi - tight_lo
    tight_lo = tight_lo - pad_frac * span
    tight_hi = tight_hi + pad_frac * span
    tight_bbox = (
        torch.tensor([tight_lo[0], tight_lo[1]], dtype=torch.float32),
        torch.tensor([tight_hi[0], tight_hi[1]], dtype=torch.float32),
    )
    xs, ys, occ = _exact_union_boundary(polygons, tight_bbox)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Light (outer/early) -> dark (inner/late) blue shades. Moderate
    # contrast and translucency so the exact reach set's black border
    # stays prominent on top.
    cmap = plt.get_cmap('Blues')
    n = len(subset)
    shade_levels = np.linspace(0.22, 0.70, n)
    fill_alpha = 0.75

    # Draw outer (largest, earliest) first so inner epochs sit on top.
    # Each set gets a thin solid stroke at a darker shade of its fill.
    for i, data in enumerate(subset):
        fill_color = cmap(shade_levels[i])
        edge_color = cmap(min(shade_levels[i] + 0.15, 1.0))
        for seg in data['contours']:
            poly = MplPolygon(
                seg, closed=True,
                facecolor=fill_color,
                edgecolor=edge_color,
                linewidth=1.2,
                alpha=fill_alpha,
            )
            ax.add_patch(poly)

    # Exact union boundary as solid black line drawn on top.
    ax.contour(
        xs, ys, occ, levels=[0.5],
        colors='black', linestyles='-', linewidths=1.4,
    )

    # Numbered circle labels: each label sits exactly on its epoch's
    # boundary, all along a single horizontal ray passing through the
    # innermost reach set's centroid. As the rings get tighter, the
    # rightward boundary crossings march from far right (outermost
    # epoch) toward the center, producing a visual "line" of numbers.
    def _largest_seg(segs):
        return max(segs, key=len)

    def _polygon_centroid(points):
        x, y = points[:, 0], points[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        cross = x * yn - xn * y
        area = 0.5 * cross.sum()
        if abs(area) < 1e-12:
            return points.mean(axis=0)
        cx = ((x + xn) * cross).sum() / (6 * area)
        cy = ((y + yn) * cross).sum() / (6 * area)
        return np.array([cx, cy])

    inner_centroid = _polygon_centroid(_largest_seg(subset[-1]['contours']))
    y_anchor = inner_centroid[1]

    def _leftmost_crossing(seg, y):
        """Leftmost x where the closed polyline `seg` crosses y."""
        xs_cross = []
        for j in range(len(seg)):
            p0 = seg[j]
            p1 = seg[(j + 1) % len(seg)]
            if (p0[1] - y) * (p1[1] - y) < 0:
                t = (y - p0[1]) / (p1[1] - p0[1])
                xs_cross.append(p0[0] + t * (p1[0] - p0[0]))
        return min(xs_cross) if xs_cross else None

    # Place each label at the leftmost boundary crossing of its epoch
    # so the labels read left-to-right: 1 (outermost) sits furthest
    # left, 4 (innermost) sits nearest the center. Then nudge any
    # overlapping pairs apart.
    label_positions = []
    for data in subset:
        seg = _largest_seg(data['contours'])
        x_edge = _leftmost_crossing(seg, y_anchor)
        label_positions.append(x_edge)

    xmin_frame = tight_bbox[0][0].item()
    xmax_frame = tight_bbox[1][0].item()
    min_gap = 0.04 * (xmax_frame - xmin_frame)
    # Sweep from outermost (leftmost) rightward, pushing each label
    # right if it's closer than min_gap to the previous one.
    valid_idx = [i for i, x in enumerate(label_positions) if x is not None]
    order = sorted(valid_idx, key=lambda i: label_positions[i])
    prev_x = None
    for idx in order:
        x = label_positions[idx]
        if prev_x is not None and (x - prev_x) < min_gap:
            x = prev_x + min_gap
            label_positions[idx] = x
        prev_x = x

    for i, data in enumerate(subset):
        x_edge = label_positions[i]
        if x_edge is None:
            continue
        ax.scatter(
            [x_edge], [y_anchor],
            s=130, color='white', edgecolor='black', linewidths=1.1,
            zorder=6,
        )
        ax.text(
            x_edge, y_anchor, str(i + 1),
            ha='center', va='center',
            fontsize=8, color='black',
            zorder=7,
        )

    # External legend: each epoch entry is a white circle containing
    # its number (matching the labels placed on the plot), rendered
    # via a custom legend handler.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    from matplotlib.text import Text
    from matplotlib.legend_handler import HandlerBase

    class _NumberedCircleHandler(HandlerBase):
        def __init__(self, number):
            self.number = number
            super().__init__()

        def create_artists(self, legend, orig_handle,
                            xdescent, ydescent, width, height,
                            fontsize, trans):
            cx = width / 2 - xdescent
            cy = height / 2 - ydescent
            radius = min(width, height) / 2.4
            circle = Circle(
                (cx, cy), radius,
                facecolor='white', edgecolor='black', linewidth=1.0,
                transform=trans,
            )
            text = Text(
                cx, cy, str(self.number),
                ha='center', va='center',
                fontsize=7, color='black',
                transform=trans,
            )
            return [circle, text]

    handles = []
    handler_map = {}
    for i, d in enumerate(subset):
        proxy = Line2D(
            [0], [0], linestyle='', marker='',
            label=f"  {d['epoch']} epochs",
        )
        handles.append(proxy)
        handler_map[proxy] = _NumberedCircleHandler(i + 1)

    handles.append(
        Line2D([0], [0], color='black', linewidth=1.0,
               label='exact reach set')
    )

    ax.legend(
        handles=handles,
        handler_map=handler_map,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
        handlelength=1.6,
        handleheight=1.6,
        handletextpad=0.6,
        labelspacing=1.0,
    )

    ax.set_xlim(tight_bbox[0][0].item(), tight_bbox[1][0].item())
    ax.set_ylim(tight_bbox[0][1].item(), tight_bbox[1][1].item())
    ax.set_aspect('equal')
    ax.set_xlabel(r'$y_1$', fontsize=13)
    ax.set_ylabel(r'$y_2$', fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'overlay.png')
    plt.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == '__main__':
    run()

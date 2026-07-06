"""Bridge between the external ``depth-verification`` repo and n2v.

This module is the single adapter layer for the depth-estimation robustness
experiment. It:

  * bootstraps ``sys.path`` so we can import the Track-A helpers verbatim
    (``depth_models.load``, ``run_scenario_depth.median_absrel/perturb/DELTAS``,
    ``risk.epsilon_upper/epsilon_two_sided/two_level``) — reused *unmodified* so
    the head-to-head with Track A is faithful;
  * loads the cached ``.npz`` NYU/KITTI frames offline (no network / datasets);
  * exposes the two "networks" the n2v probabilistic pipeline verifies:
      1. a **scalar** median-relative-depth-change functional  g(x)  (Deliverable
         1 — certified with ``conformal_reach``, no flow);
      2. a **dense deviation map**  Δ(x) = f(x) − d₀  (Deliverable 2 — certified
         with ``flow_reach`` + PCA against spatial per-region ℓ∞ specs).

Both networks are returned as numpy callables ``y = fn(X)`` taking a batch of
*flattened* perturbed images ``X`` of shape ``(N, 3*res*res)`` and running the
(GPU) forward pass with internal sub-batching, exactly the contract
``flow_reach`` / ``conformal_reach`` expect.

NOTE: importing this module imports ``transformers`` (via ``depth_models``).
Those deps are experiment-only — see ``requirements-depth-experiment.txt``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, List, Tuple

# Default to offline HuggingFace: cached models load in seconds, and we never
# stall on network metadata calls (this environment has no outbound network).
# To (re)download models/corpora, run with HF_HUB_OFFLINE=0 in a networked env.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

from n2v.sets import Box

# --------------------------------------------------------------------------
# Paths into the external depth-verification repository.
# --------------------------------------------------------------------------
DEPTH_REPO = Path(
    os.environ.get(
        "DEPTH_VERIFICATION_REPO",
        str(Path.home() / "v" / "other" / "depth-verification"),
    )
).resolve()

_DEPTH_DIR = DEPTH_REPO / "scenario-approach" / "Stage-2-verification" / "depth"
_SCENARIO_DIR = DEPTH_REPO / "scenario-approach" / "Stage-2-verification" / "scenario"
_NYU_NPZ = (DEPTH_REPO / "sound-tier-theory" / "reports" / "bench"
            / "sota_depth" / "data" / "nyu_sample")
_KITTI_NPZ = (DEPTH_REPO / "sound-tier-theory" / "research"
              / "W12-lean-formal-proofs" / "validation" / "data" / "kitti_sample")

# Per-model working resolution (square), matching Track A's RES dict.
RES = {"depthanything": 252, "dpt_midas": 384, "dpt_swin": 256}
MODEL_KEYS = tuple(RES.keys())
DATASETS = ("nyu", "kitti")

_CACHE_NPZ = {"nyu": _NYU_NPZ, "kitti": _KITTI_NPZ}


def _bootstrap_paths() -> None:
    """Put the Track-A dirs on ``sys.path`` (idempotent)."""
    for p in (str(_DEPTH_DIR), str(_SCENARIO_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _import_track_a():
    """Import the reused Track-A modules. Requires ``transformers`` installed."""
    _bootstrap_paths()
    import depth_models as DM  # noqa: E402  (path-bootstrapped)
    import risk  # noqa: E402
    import run_scenario_depth as RSD  # noqa: E402
    return DM, risk, RSD


# --------------------------------------------------------------------------
# Model loading.
# --------------------------------------------------------------------------
def load_depth_model(name: str):
    """Load a Track-A depth transformer.

    Returns ``(forward_fn, res, device)`` where ``forward_fn`` maps a
    ``(B, 3, res, res)`` float tensor in ``[0, 1]`` on ``device`` to a
    ``(B, h, w)`` predicted-depth tensor (the HF ImageProcessor mean/std
    normalisation is folded into ``forward_fn`` by ``depth_models.load``).
    """
    if name not in RES:
        raise KeyError(f"unknown model {name!r}; choose from {MODEL_KEYS}")
    DM, _risk, _RSD = _import_track_a()
    forward_fn, _H, _W = DM.load(name)
    return forward_fn, RES[name], DM.DEV


# --------------------------------------------------------------------------
# Cached frame loading (offline; no network / datasets).
# --------------------------------------------------------------------------
def load_cached_frames(dataset: str, n: int, res: int, device) -> List[Tuple[int, torch.Tensor]]:
    """Load up to ``n`` cached ``.npz`` frames, resized to ``(res, res)``.

    Returns a list of ``(frame_idx, x01)`` where ``x01`` is a ``(3, res, res)``
    float tensor in ``[0, 1]`` on ``device`` (bilinear resize, matching the
    ``Image.BILINEAR`` convention Track A uses for its ``.jpg`` corpus).
    """
    if dataset not in _CACHE_NPZ:
        raise KeyError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    npz_dir = _CACHE_NPZ[dataset]
    files = sorted(npz_dir.glob("*.npz"))[:n]
    out: List[Tuple[int, torch.Tensor]] = []
    for f in files:
        d = np.load(f)
        rgb = d["rgb"].astype(np.float32) / 255.0            # (3, H, W) in [0,1]
        t = torch.from_numpy(rgb).unsqueeze(0)               # (1, 3, H, W)
        t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)
        out.append((int(f.stem), t.squeeze(0).to(device)))
    return out


# --------------------------------------------------------------------------
# Clean prediction + scale floor (Track A convention).
# --------------------------------------------------------------------------
@torch.no_grad()
def clean_depth(forward_fn: Callable, x01: torch.Tensor):
    """Return ``(d0, scale_floor)`` for a clean image ``x01`` of shape (3,res,res).

    ``d0`` is the ``(h, w)`` clean depth; ``scale_floor`` is Track A's
    ``1e-3 * median(|d0|)`` clamped to ``1e-6`` (scalar tensor).
    """
    d0 = forward_fn(x01.unsqueeze(0))[0].float()             # (h, w)
    scale_floor = (1e-3 * d0.abs().median()).clamp_min(1e-6)
    return d0, scale_floor


# --------------------------------------------------------------------------
# The two "networks" verified by the n2v probabilistic pipeline.
# --------------------------------------------------------------------------
def make_scalar_absrel_callable(
    forward_fn: Callable,
    d0: torch.Tensor,
    scale_floor: torch.Tensor,
    res: int,
    device,
    batch: int = 48,
) -> Callable[[np.ndarray], np.ndarray]:
    """Deliverable 1: scalar median-relative-depth-change functional.

    Returns a numpy callable ``g`` mapping flattened perturbed images
    ``X`` of shape ``(N, 3*res*res)`` to ``(N, 1)`` where

        g(x) = median_pixels |f(x) - d0| / max(|d0|, scale_floor).

    This is exactly the per-sample quantity Track A's ``median_absrel``
    thresholds at each δ; certifying ``Pr[g <= δ] >= 1-ε`` is the
    apples-to-apples probabilistic robustness statement.
    """
    d0f = d0.reshape(-1)                                      # (h*w,)
    denom = torch.clamp(d0.abs(), min=float(scale_floor)).reshape(-1)

    @torch.no_grad()
    def g(X: np.ndarray) -> np.ndarray:
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
        n = Xt.shape[0]
        meds = torch.empty(n, dtype=torch.float32, device=device)
        for i in range(0, n, batch):
            xb = Xt[i:i + batch].reshape(-1, 3, res, res)
            dp = forward_fn(xb).float().reshape(xb.shape[0], -1)   # (b, h*w)
            absrel = (dp - d0f).abs() / denom
            meds[i:i + xb.shape[0]] = absrel.median(dim=1).values
        return meds.reshape(-1, 1).detach().cpu().numpy()

    return g


def make_deviation_callable(
    forward_fn: Callable,
    d0: torch.Tensor,
    res: int,
    device,
    batch: int = 48,
    coarse: Optional[int] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Deliverable 2: deviation-map network Δ(x) = f(x) - d0.

    Returns a numpy callable mapping flattened perturbed images ``X`` of shape
    ``(N, 3*res*res)`` to the flattened deviation map. Centring on ``d0`` keeps
    the flow's target distribution near the origin and makes spatial ℓ∞ specs
    read directly as ``|Δ_region| > δ``.

    Args:
        coarse: If given, average-pool the deviation map to a ``coarse × coarse``
            grid before flattening, so the output is ``(N, coarse*coarse)``. Depth
            maps are smooth, so this loses little, keeps PCA/flow tractable, and
            makes each coarse cell a *region* for the spatial per-region ℓ∞
            property. ``None`` returns the full ``(N, h*w)`` map.
    """
    h, w = int(d0.shape[0]), int(d0.shape[1])

    @torch.no_grad()
    def dev(X: np.ndarray) -> np.ndarray:
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
        n = Xt.shape[0]
        outs = []
        for i in range(0, n, batch):
            xb = Xt[i:i + batch].reshape(-1, 3, res, res)
            dp = forward_fn(xb).float()                       # (b, h, w)
            delta = dp - d0                                   # (b, h, w)
            if coarse is not None:
                delta = F.adaptive_avg_pool2d(
                    delta.unsqueeze(1), (coarse, coarse)).squeeze(1)  # (b,c,c)
            outs.append(delta.reshape(delta.shape[0], -1).detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    return dev


def make_input_box(x01: torch.Tensor, eps: float) -> Box:
    """ℓ∞ ball of radius ``eps`` around ``x01``, clipped to ``[0, 1]``, flattened.

    Matches Track A's perturbation support (uniform ℓ∞ sensor noise, clipped
    to valid pixel range) as an axis-aligned :class:`n2v.sets.Box`.
    """
    x = x01.detach().cpu().numpy().reshape(-1).astype(np.float32)
    lb = np.clip(x - eps, 0.0, 1.0)
    ub = np.clip(x + eps, 0.0, 1.0)
    return Box(lb, ub)


# --------------------------------------------------------------------------
# Track-A reference (verbatim reuse) for the head-to-head numbers.
# --------------------------------------------------------------------------
def track_a_counts(forward_fn: Callable, x01: torch.Tensor, eps: float,
                   N: int, seed: int, device):
    """Track A's per-image violation counts, reusing an already-loaded model.

    Returns ``(k, deltas)`` where ``k[j]`` is the violation count at
    ``deltas[j]`` over ``N`` uniform-ℓ∞ perturbations, computed by Track A's
    ``median_absrel`` reused unmodified (same metric, sampling, and clip).
    """
    _DM, _risk, RSD = _import_track_a()
    d0 = forward_fn(x01.unsqueeze(0))[0].float()
    scale_floor = (1e-3 * d0.abs().median()).clamp_min(1e-6)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    k = RSD.median_absrel(forward_fn, x01, d0, scale_floor, eps, N, gen)
    return [int(v) for v in k], list(RSD.DELTAS)


def risk_epsilon_upper(N: int, k: int, beta: float) -> float:
    """Garatti-Campi one-sided Level-1 risk bound (risk.epsilon_upper)."""
    _DM, risk, _RSD = _import_track_a()
    return float(risk.epsilon_upper(N, int(k), beta))


def risk_two_level(ks_inner, N_inner, tau, beta_inner, beta_outer):
    """Track A's two-level per-image -> population aggregation (risk.two_level)."""
    _DM, risk, _RSD = _import_track_a()
    return risk.two_level(list(ks_inner), N_inner, tau, beta_inner, beta_outer)


def track_a_reference(name: str, x01: torch.Tensor, eps: float, N: int, seed: int):
    """Convenience wrapper (loads the model itself); used by the smoke test.

    Returns ``(k, deltas, eps_up)`` with ``eps_up[j] = epsilon_upper(N, k[j], 1e-6)``.
    Prefer :func:`track_a_counts` in loops to avoid reloading the model.
    """
    DM, risk, _RSD = _import_track_a()
    forward_fn, _H, _W = DM.load(name)
    k, deltas = track_a_counts(forward_fn, x01, eps, N, seed, DM.DEV)
    eps_up = [risk.epsilon_upper(N, k[j], 1e-6) for j in range(len(deltas))]
    return k, deltas, eps_up

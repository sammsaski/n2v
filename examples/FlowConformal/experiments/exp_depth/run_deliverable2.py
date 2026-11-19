"""Deliverable 2 — spatial per-region ℓ∞ depth-deviation certificate (flow + PCA).

Certifies a property the scalar median metric (Deliverable 1) *cannot* express:

    "within a safety-critical region R of the image, the depth deviation
     Δ(x) = f(x) - d0 stays within ±δ (per coarse cell), under random ℓ∞ input
     noise, with probability ≥ 1 - ε_total."

Pipeline: coarse deviation map (32×32 regions) -> flow_reach + PCA (reduced reach
set with a calibrated residual bound) -> verify_specification against the unsafe
spec "∃ cell in R with |Δ_cell| > δ" (an OR of per-cell halfspaces). A verdict of
UNSAT means the reach set is certified disjoint from that unsafe region, i.e. the
region is robust at tolerance δ.

We sweep δ (metres) and report where the certificate flips, both with the sound
full-output residual inflation and (diagnostic) without it, to show the cost of
soundness.

    python -m examples.FlowConformal.experiments.exp_depth.run_deliverable2 \
        --frame 0 --eps 8 --coarse 32 --pca 64 --region 4
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from n2v.sets import HalfSpace
from n2v.probabilistic import flow_reach, FlowReachConfig
from n2v.probabilistic.flow.sets import PCATransform
from n2v.utils.verify_specification import verify_specification, ProbVerifyConfig

from examples.FlowConformal.experiments.exp_depth import _depth_models as B

SEED = 47


def central_block_cells(coarse: int, block: int):
    """Cell indices of a centred ``block × block`` region on a coarse grid."""
    r0 = (coarse - block) // 2
    rows = range(r0, r0 + block)
    return [r * coarse + c for r in rows for c in rows], (r0, r0 + block)


def region_linf_spec(coarse_dim: int, cells, delta: float):
    """Unsafe spec: OR over region cells of ``|Δ_cell| > δ`` (per-cell halfspaces).

    Each cell contributes two single-row halfspaces (``Δ_i ≥ δ`` and
    ``Δ_i ≤ -δ``) in the ``G y ≤ g`` = unsafe convention. Returned as a flat
    list => a single OR-group for ``verify_specification``.
    """
    hss = []
    for i in cells:
        Gu = np.zeros((1, coarse_dim)); Gu[0, i] = -1.0   # -Δ_i ≤ -δ  ⇔  Δ_i ≥ δ
        hss.append(HalfSpace(Gu, np.array([[-delta]])))
        Gl = np.zeros((1, coarse_dim)); Gl[0, i] = 1.0    #  Δ_i ≤ -δ
        hss.append(HalfSpace(Gl, np.array([[-delta]])))
    return hss


def _zero_residual(tr: PCATransform) -> PCATransform:
    return PCATransform(mean=tr.mean, std=tr.std, components=tr.components,
                        pca_mean=tr.pca_mean, residual_bound=0.0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="depthanything", choices=B.MODEL_KEYS)
    p.add_argument("--dataset", default="nyu", choices=B.DATASETS)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--eps", type=int, default=8, help="ℓ∞ radius over 255")
    p.add_argument("--coarse", type=int, default=32)
    p.add_argument("--pca", type=int, default=64)
    p.add_argument("--region", type=int, default=4, help="centred block size")
    p.add_argument("--alpha", type=float, default=0.05, help="conformal miscoverage")
    p.add_argument("--n-verify", type=int, default=5000)
    p.add_argument("--deltas", default="0.05,0.1,0.2,0.5,1.0,2.0",
                   help="deviation tolerances in metres")
    args = p.parse_args()

    eps = args.eps / 255.0
    deltas = [float(x) for x in args.deltas.split(",")]
    torch.manual_seed(SEED); np.random.seed(SEED)

    forward_fn, res, device = B.load_depth_model(args.model)
    frames = {i: x for i, x in B.load_cached_frames(args.dataset, args.frame + 1,
                                                    res, device)}
    x01 = frames[args.frame]
    d0, _ = B.clean_depth(forward_fn, x01)
    dev = B.make_deviation_callable(forward_fn, d0, res, device, coarse=args.coarse)
    coarse_dim = args.coarse * args.coarse
    cells, (r0, r1) = central_block_cells(args.coarse, args.region)

    print(f"[d2] {args.model}/{args.dataset} frame {args.frame} eps={args.eps}/255 "
          f"coarse={args.coarse}² pca={args.pca} region=[{r0}:{r1}]² "
          f"({len(cells)} cells) alpha={args.alpha}", flush=True)

    # ---- flow + PCA reach set on the coarse deviation map ----
    torch.manual_seed(SEED); np.random.seed(SEED)
    box = B.make_input_box(x01, eps)
    ps = flow_reach(dev, box, FlowReachConfig(
        epsilon=args.alpha, m=2000, n_train=3000, flow_epochs=800,
        flow_config="base", pca_components=args.pca, pca_backend="randomized",
        seed=SEED))
    cov = ps.estimate_coverage(dev, box, n_test=2000, seed=SEED + 12345)
    print(f"[d2] reach set: reduced_dim={ps.dim} resid_bound="
          f"{ps.affine_transform.residual_bound:.3f} coverage={cov:.4f} "
          f"(target {1-args.alpha:.2f})", flush=True)

    # empirical per-cell deviation magnitude in the region (context)
    Xs = np.random.uniform(box.lb.reshape(-1), box.ub.reshape(-1),
                           size=(1000, box.dim)).astype(np.float32)
    Dev = dev(Xs)[:, cells]                     # (1000, |R|)
    emp_max = float(np.abs(Dev).max())
    emp_p95 = float(np.percentile(np.abs(Dev), 95))
    print(f"[d2] region |Δ| (empirical over 1000 draws): p95={emp_p95:.3f} m  "
          f"max={emp_max:.3f} m\n", flush=True)

    # ---- sweep δ, verify (sound + residual-free diagnostic) ----
    cfg = ProbVerifyConfig(method="scenario", n_samples=args.n_verify,
                           beta=0.001, seed=SEED)
    tr_sound = ps.affine_transform
    tr_diag = _zero_residual(tr_sound)
    print(f"    {'δ (m)':>7} {'sound':>10} {'eps_tot':>9}   "
          f"{'reduced-only(diag)':>18} {'eps_tot':>9}", flush=True)
    for d in deltas:
        spec = region_linf_spec(coarse_dim, cells, d)
        ps.affine_transform = tr_sound
        r1v = verify_specification(ps, spec, config=cfg)
        ps.affine_transform = tr_diag
        r2v = verify_specification(ps, spec, config=cfg)
        ps.affine_transform = tr_sound
        e1 = getattr(r1v, "epsilon_total", None)
        e2 = getattr(r2v, "epsilon_total", None)
        print(f"    {d:>7.2f} {r1v.verdict:>10} "
              f"{(f'{e1:.3f}' if e1 is not None else '-'):>9}   "
              f"{r2v.verdict:>18} "
              f"{(f'{e2:.3f}' if e2 is not None else '-'):>9}", flush=True)

    print("\n[d2] verdict key: UNSAT = region certified robust at δ; "
          "UNKNOWN = not certified; SAT = counterexample found.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

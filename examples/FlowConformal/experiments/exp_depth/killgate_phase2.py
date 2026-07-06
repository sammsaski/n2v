"""Phase-2b coverage kill-gate for flow + PCA on the dense depth deviation map.

The make-or-break check for Deliverable 2: does the flow-matching reach set,
trained on a PCA-reduced *coarse* deviation map Δ(x)=f(x)-d0, actually achieve
its claimed conformal coverage (1-ε) on held-out perturbations? If not, we stop
and reassess before building certificates on top of it.

Reports, per frame, the empirical reduced-frame coverage (should be >= 1-ε),
plus the calibrated residual bound and reduced dimension.

    python -m examples.FlowConformal.experiments.exp_depth.killgate_phase2
"""
import sys

import numpy as np
import torch

from n2v.probabilistic import flow_reach, FlowReachConfig

from examples.FlowConformal.experiments.exp_depth import _depth_models as B

SEED = 47
MODEL = "depthanything"
DATASET = "nyu"
EPS = 8 / 255
COARSE = 32          # 32x32 = 1024-dim coarse deviation map (regions)
PCA_K = 64           # reduce to 64 dims for the flow
ALPHA = 0.05         # conformal miscoverage ε₁
N_FRAMES = 3


def main() -> int:
    torch.manual_seed(SEED); np.random.seed(SEED)
    forward_fn, res, device = B.load_depth_model(MODEL)
    frames = B.load_cached_frames(DATASET, N_FRAMES, res, device)
    print(f"[kg] model={MODEL} res={res} coarse={COARSE}x{COARSE} "
          f"pca_k={PCA_K} alpha={ALPHA} eps={EPS:.4f}", flush=True)
    print(f"[kg] target coverage 1-alpha = {1-ALPHA:.3f}\n", flush=True)

    ok = True
    for idx, x01 in frames:
        torch.manual_seed(SEED); np.random.seed(SEED)
        d0, _ = B.clean_depth(forward_fn, x01)
        dev = B.make_deviation_callable(forward_fn, d0, res, device, coarse=COARSE)
        box = B.make_input_box(x01, EPS)
        ps = flow_reach(
            dev, box,
            FlowReachConfig(epsilon=ALPHA, m=2000, n_train=3000,
                            flow_epochs=800, flow_config="base",
                            pca_components=PCA_K, pca_backend="randomized",
                            seed=SEED),
        )
        tr = ps.affine_transform
        # Score-layer coverage (projection in the flow set).
        cov = ps.estimate_coverage(dev, box, n_test=2000, seed=SEED + 12345)
        # JOINT full-output coverage: projection in set AND residual <= bound.
        rng = np.random.default_rng(SEED + 777)
        Xt = rng.uniform(box.lb.reshape(-1), box.ub.reshape(-1),
                         size=(2000, box.dim)).astype(np.float32)
        Y = dev(Xt)
        z = tr.whiten(Y)
        in_score = ps.contains(torch.as_tensor(np.asarray(z),
                                               dtype=torch.float32)).numpy()
        in_resid = tr.residual_norm(Y) <= tr.residual_bound
        joint = float((in_score & in_resid).mean())
        passed = joint >= (1 - ALPHA - 0.02)
        ok = ok and passed
        print(f"[kg] frame {idx:3d}: reduced_dim={ps.dim} "
              f"resid_bound(ℓ∞)={tr.residual_bound:.4f} "
              f"score_cov={cov:.4f} joint_cov={joint:.4f}  "
              f"-> {'PASS' if passed else 'LOW'}", flush=True)

    print(f"\n[kg] {'ALL PASS' if ok else 'SOME LOW — investigate'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

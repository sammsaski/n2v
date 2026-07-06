"""Phase-0 smoke test: end-to-end sanity on one cached frame.

Confirms the whole bridge works in the n2v env:
  1. a depth transformer loads and forwards (transformers-5.x API check);
  2. the scalar median-absrel callable behaves (0 on the clean image, >0 under
     perturbation);
  3. n2v's ``conformal_reach`` runs on it and yields a probabilistic box on g;
  4. Track A's ``median_absrel`` + ``risk.epsilon_upper`` reference computes.

Tiny budgets — this is a plumbing check, not a real certificate.

    python -m examples.FlowConformal.experiments.exp_depth.smoke_phase0
"""
import sys

import numpy as np
import torch

from n2v.probabilistic import conformal_reach, ConformalReachConfig

from examples.FlowConformal.experiments.exp_depth import _depth_models as B

SEED = 47
MODEL = "depthanything"
DATASET = "nyu"
EPS = 8 / 255


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[smoke] loading model {MODEL!r} ...", flush=True)
    forward_fn, res, device = B.load_depth_model(MODEL)
    print(f"[smoke]   res={res} device={device}", flush=True)

    frames = B.load_cached_frames(DATASET, n=1, res=res, device=device)
    idx, x01 = frames[0]
    print(f"[smoke] frame {idx}: x01 {tuple(x01.shape)} "
          f"range=[{x01.min():.3f},{x01.max():.3f}]", flush=True)

    d0, scale_floor = B.clean_depth(forward_fn, x01)
    print(f"[smoke] d0 {tuple(d0.shape)} range=[{d0.min():.3f},{d0.max():.3f}] "
          f"scale_floor={float(scale_floor):.4g}", flush=True)

    g = B.make_scalar_absrel_callable(forward_fn, d0, scale_floor, res, device)
    print("[smoke] built g; calling g(clean) ...", flush=True)

    # Sanity: g(clean) == 0 exactly; g(perturbed) > 0.
    x_flat = x01.detach().cpu().numpy().reshape(1, -1)
    g_clean = float(g(x_flat)[0, 0])
    print(f"[smoke] g(clean)={g_clean:.3e}; building input box ...", flush=True)
    rng = np.random.default_rng(SEED)
    box = B.make_input_box(x01, EPS)
    print(f"[smoke] box dim={box.dim}; sampling 4 perturbations ...", flush=True)
    xp = rng.uniform(box.lb.reshape(-1), box.ub.reshape(-1), size=(4, box.dim)).astype(np.float32)
    print("[smoke] sampled; calling g(perturbed) ...", flush=True)
    g_pert = g(xp).reshape(-1)
    print(f"[smoke] g(clean)={g_clean:.3e} (expect ~0); "
          f"g(perturbed)={np.round(g_pert, 4).tolist()}", flush=True)
    assert g_clean < 1e-5, f"g(clean) should be ~0, got {g_clean}"
    assert np.all(g_pert >= 0), "median-absrel must be non-negative"

    # n2v conformal certificate on the scalar (tiny budget).
    print("[smoke] conformal_reach on scalar g ...", flush=True)
    pbox = conformal_reach(
        g, box,
        ConformalReachConfig(surrogate="naive", m=200, training_samples=100,
                             epsilon=0.05, batch_size=200, seed=SEED),
    )
    lb = float(np.asarray(pbox.lb).reshape(-1)[0])
    ub = float(np.asarray(pbox.ub).reshape(-1)[0])
    print(f"[smoke]   g-box=[{lb:.4f}, {ub:.4f}]  coverage={pbox.coverage:.4f} "
          f"confidence={pbox.confidence:.4f}", flush=True)
    for delta in (0.02, 0.05, 0.10, 0.20):
        verdict = "ROBUST" if ub <= delta else "not-certified"
        print(f"[smoke]     delta={delta:.2f}: {verdict} (ub {ub:.4f} vs {delta})",
              flush=True)

    # Track A reference (verbatim reuse).
    print("[smoke] Track A reference (median_absrel + epsilon_upper, N=200) ...",
          flush=True)
    k, deltas, eps_up = B.track_a_reference(MODEL, x01, EPS, N=200, seed=SEED)
    for j, delta in enumerate(deltas):
        print(f"[smoke]     delta={delta:.2f}: k={k[j]}/200  "
              f"eps_up={eps_up[j]:.4f}", flush=True)

    print("[smoke] PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

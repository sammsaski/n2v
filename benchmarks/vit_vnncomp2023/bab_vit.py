"""Branch-and-bound verification of the VNN-COMP 2023 ViT instances.

Wires the general n2v BaB engine (n2v/nn/bab.py) to the ViT reach driver: the
bounder is ViTReacher (sound), the falsifier is n2v.utils.falsify on the torch
model, and the property is argmax preservation. Branching is input-domain
splitting with sensitivity from the reach output's input-predicate generators
(remapped HWC->CHW to match the split space).
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from model import build_model
from reach import ViTReacher
from n2v.sets import ImageStar
from n2v.sets.halfspace import HalfSpace
from n2v.nn.bab import verify_bab, star_input_sensitivity
from n2v.utils.falsify import falsify

MEAN = np.array([0.4914, 0.4822, 0.4465]); STD = np.array([0.2023, 0.1994, 0.2010])


def robustness_spec(label, K=10):
    """Unsafe region (OR group): some class i beats the label -> Y_y - Y_i <= 0."""
    hs = []
    for i in range(K):
        if i == label:
            continue
        G = np.zeros((1, K)); G[0, label] = 1.0; G[0, i] = -1.0
        hs.append(HalfSpace(G, np.array([0.0])))
    return hs


def verify_instance_bab(reacher, torch_model, x, label, eps, *,
                        max_nodes=150, timeout_s=900.0, falsify_method="random",
                        verbose=False):
    spec = robustness_spec(label)
    lb = (np.clip(x - eps, 0, 1) - MEAN[:, None, None]) / STD[:, None, None]
    ub = (np.clip(x + eps, 0, 1) - MEAN[:, None, None]) / STD[:, None, None]
    lb = lb.astype(np.float64); ub = ub.astype(np.float64)
    H, W, C = 32, 32, 3
    best = {"margin": -np.inf}

    def reach_fn(l_chw, u_chw):
        img = ImageStar.from_bounds(np.transpose(l_chw, (1, 2, 0)).reshape(-1),
                                    np.transpose(u_chw, (1, 2, 0)).reshape(-1), H, W, C)
        return [reacher.reach(img)]

    def bound_fn(reach_sets, _spec):
        m = reacher.margins(reach_sets[0], label, margin_mode="estimate")
        mm = min(m.values())
        best["margin"] = max(best["margin"], mm)
        return mm > 0

    def falsify_fn(l_chw, u_chw):
        res, cex = falsify(torch_model, l_chw, u_chw, spec,
                           method=falsify_method, n_samples=300)
        return np.asarray(cex[0]) if (res == 0 and cex is not None) else None

    def sens_fn(reach_sets, _spec, n_input):
        s = star_input_sensitivity(reach_sets, _spec, n_input)
        if s is None:
            return None
        # star input preds are HWC-ordered; BaB splits CHW-flattened lb/ub.
        return s.reshape(H, W, C).transpose(2, 0, 1).reshape(-1)

    res = verify_bab(reach_fn, lb, ub, spec, falsify_fn=falsify_fn,
                     bound_fn=bound_fn, sensitivity_fn=sens_fn, branch="sensitivity",
                     max_nodes=max_nodes, timeout_s=timeout_s, verbose=verbose)
    res.reason += f" | best_margin={best['margin']:+.4f}"
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="ibp_3_3_8")
    ap.add_argument("idx", nargs="?", type=int, default=0)
    ap.add_argument("eps", nargs="?", type=float, default=1.0, help="x/255")
    ap.add_argument("--mode", default="symbolic-av")
    ap.add_argument("--max-nodes", type=int, default=150)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    z = np.load(os.path.join(HERE, "instances", f"{a.name}.npz"))
    x = z["images"][a.idx].astype(np.float64); y = int(z["labels"][a.idx])
    m = build_model(a.name, os.path.join(HERE, "onnx", f"{a.name}.onnx"))
    R = ViTReacher(m, mode=a.mode)
    print(f"BaB {a.name} idx={a.idx} label={y} eps={a.eps}/255 mode={a.mode} "
          f"max_nodes={a.max_nodes}", flush=True)
    t0 = time.time()
    res = verify_instance_bab(R, m, x, y, a.eps / 255, max_nodes=a.max_nodes,
                              timeout_s=a.timeout, verbose=a.verbose)
    print(f"-> {res.verdict} | nodes={res.nodes} splits={res.splits} "
          f"depth={res.max_depth} t={res.time_s:.1f}s | {res.reason}", flush=True)
    print("BAB_DONE", flush=True)


if __name__ == "__main__":
    main()

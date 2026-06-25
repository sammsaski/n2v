import sys, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from model import build_model
from reach import ViTReacher
from n2v.sets import ImageStar

MEAN = np.array([0.4914, 0.4822, 0.4465]); STD = np.array([0.2023, 0.1994, 0.2010])
name = sys.argv[1] if len(sys.argv) > 1 else "pgd_2_3_16"
idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
eps = (float(sys.argv[3]) if len(sys.argv) > 3 else 1.0) / 255

z = np.load(os.path.join(HERE, "instances", f"{name}.npz"))
x = z["images"][idx].astype(np.float64); y = int(z["labels"][idx])
m = build_model(name, os.path.join(HERE, "onnx", f"{name}.onnx"))

lb = (np.clip(x - eps, 0, 1) - MEAN[:, None, None]) / STD[:, None, None]
ub = (np.clip(x + eps, 0, 1) - MEAN[:, None, None]) / STD[:, None, None]
def img():
    return ImageStar.from_bounds(np.transpose(lb, (1, 2, 0)).reshape(-1),
                                 np.transpose(ub, (1, 2, 0)).reshape(-1), 32, 32, 3)
print(f"{name} idx={idx} label={y} eps={eps*255:.3f}/255", flush=True)

for mode in ["concretize", "symbolic-av"]:
    R = ViTReacher(m, mode=mode)
    t0 = time.time(); out = R.reach(img()); tr = time.time() - t0
    print(f"\n[{mode}] reach: nVar={out.nVar} nC={out.C.shape[0]} t={tr:.1f}s", flush=True)
    if mode == "symbolic-av":
        rng = np.random.default_rng(0); nv = 0
        for _ in range(3):
            xs = rng.uniform(lb, ub).astype(np.float32)
            with torch.no_grad():
                lg = m(torch.from_numpy(xs).unsqueeze(0)).numpy().reshape(-1).astype(np.float64)
            nv += (not out.contains(lg, method="lp"))
        print(f"[{mode}] containment: {nv}/3 violations (0=sound)", flush=True)
    t0 = time.time()
    mlp = R.margins(out, y, margin_mode="lp"); tlp = time.time() - t0
    mest = R.margins(out, y, margin_mode="estimate")
    print(f"[{mode}] LP   min-margin = {min(mlp.values()):+.4f} ({tlp:.1f}s, {sum(v>0 for v in mlp.values())}/9 classes>0)", flush=True)
    print(f"[{mode}] est  min-margin = {min(mest.values()):+.4f}", flush=True)
print("\nPROBE_DONE", flush=True)

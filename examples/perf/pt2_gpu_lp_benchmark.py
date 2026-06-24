"""PT-2 benchmark: batched-GPU sound LP vs CPU HiGHS for Star bound LPs.

Measures the two metrics the PT-2 plan calls for -- LP throughput (LPs/sec) and
end-to-end bound wall-time -- using the profiler's GPU instrumentation, with a
verdict/soundness check (the GPU+NS bounds must enclose the exact CPU bounds).

Workload = real n2v Stars: the output reach sets of an ACAS Xu network over many
input sub-boxes. This is the genuine LP path (``Star.get_ranges`` -> ``2*dim``
bound LPs); the default approx reach uses LP-free ``estimate_ranges``, so we
benchmark the LP path it would use when tight LP bounds are requested.

Run:  python examples/perf/pt2_gpu_lp_benchmark.py [--pop 256]
"""

import argparse
import time
from pathlib import Path

import numpy as np

import n2v
from n2v.nn import NeuralNetwork
from n2v.profiling import profile
from n2v.sets import Star
from n2v.utils.lpsolver_gpu import gpu_available
from n2v.utils.model_loader import load_onnx

# ACAS Xu nominal input box (normalized), 5 inputs.
_ACAS_LB = np.array([-0.3284, -0.5, -0.5, -0.5, -0.5]).reshape(-1, 1)
_ACAS_UB = np.array([0.6799, 0.5, 0.5, 0.5, 0.5]).reshape(-1, 1)


def _build_population(net, n_stars, rng):
    """Reach output Stars over n_stars random input sub-boxes (real n2v stars)."""
    stars = []
    span = _ACAS_UB - _ACAS_LB
    for _ in range(n_stars):
        # Random sub-box of the input region.
        c = _ACAS_LB + span * rng.uniform(0.1, 0.9, size=(5, 1))
        half = span * rng.uniform(0.02, 0.1, size=(5, 1))
        lb = np.maximum(c - half, _ACAS_LB)
        ub = np.minimum(c + half, _ACAS_UB)
        out = net.reach(Star.from_bounds(lb, ub), method="approx")
        s = out[0] if isinstance(out, list) else out
        if isinstance(s, Star) and s.nVar > 0:
            stars.append(s)
    return stars


def _time(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=256, help="population size")
    ap.add_argument("--net", type=str,
                    default="examples/ACASXu/onnx/ACASXU_run2a_1_1_batch_2000.onnx")
    args = ap.parse_args()

    if not gpu_available():
        print("No CUDA device available; GPU path will fall back to CPU.")
    print(f"Loading {args.net}")
    net = NeuralNetwork(load_onnx(Path(args.net)))

    rng = np.random.default_rng(47)
    stars = _build_population(net, args.pop, rng)
    if not stars:
        print("No usable stars produced; aborting.")
        return
    dim, nvar = stars[0].dim, stars[0].nVar
    nconstr = [int(s.C.shape[0]) if s.C.size > 0 else 0 for s in stars]
    n_lps = sum(2 * s.dim for s in stars)
    print(f"Population: {len(stars)} stars | dim={dim} nVar={nvar} "
          f"constraints~[{min(nconstr)},{max(nconstr)}] | total LPs={n_lps}")

    # ----- CPU baseline: per-star get_ranges (HiGHS) ----------------------- #
    n2v.set_gpu_lp(False)
    cpu_bounds = None

    def _cpu():
        nonlocal cpu_bounds
        cpu_bounds = [s.get_ranges() for s in stars]

    t_cpu = _time(_cpu)
    print(f"\nCPU  (HiGHS, per-star)        : {t_cpu*1e3:8.1f} ms  "
          f"{n_lps/t_cpu:10.0f} LPs/s")

    # ----- GPU within-star (4a-i): per-star get_ranges on GPU -------------- #
    n2v.set_gpu_lp(True)
    gpu_i_bounds = None

    def _gpu_i():
        nonlocal gpu_i_bounds
        gpu_i_bounds = [s.get_ranges() for s in stars]

    _gpu_i()  # warm up CUDA (kernel compile / context)
    with profile(level="operation") as p_i:
        t_gpu_i = _time(_gpu_i)
    n2v.set_gpu_lp(False)
    print(f"GPU  4a-i (within-star batch) : {t_gpu_i*1e3:8.1f} ms  "
          f"{n_lps/t_gpu_i:10.0f} LPs/s   speedup {t_cpu/t_gpu_i:5.2f}x  "
          f"gpu_time={p_i.rollup()['gpu_time']*1e3:.1f} ms")

    # ----- GPU cross-population (4a-ii): one batched solve ----------------- #
    pop_bounds = None

    def _gpu_ii():
        nonlocal pop_bounds
        pop_bounds = Star.get_ranges_population(stars)

    _gpu_ii()  # warm up
    with profile(level="operation") as p_ii:
        t_gpu_ii = _time(_gpu_ii)
    print(f"GPU  4a-ii (cross-population)  : {t_gpu_ii*1e3:8.1f} ms  "
          f"{n_lps/t_gpu_ii:10.0f} LPs/s   speedup {t_cpu/t_gpu_ii:5.2f}x  "
          f"gpu_time={p_ii.rollup()['gpu_time']*1e3:.1f} ms")

    # ----- Soundness / verdict-equivalence check --------------------------- #
    max_inward = 0.0
    max_loose = 0.0
    for (lb_c, ub_c), (lb_g, ub_g) in zip(cpu_bounds, pop_bounds):
        max_inward = max(max_inward,
                         float(np.max(lb_g - lb_c)),   # gpu_lb above cpu_lb = bad
                         float(np.max(ub_c - ub_g)))   # gpu_ub below cpu_ub = bad
        max_loose = max(max_loose,
                        float(np.max(lb_c - lb_g)),
                        float(np.max(ub_g - ub_c)))
    print(f"\nSoundness: worst inward (must be <=0): {max_inward:.2e}  "
          f"| worst outward looseness: {max_loose:.2e}")
    print("ENCLOSES exact CPU bounds" if max_inward < 1e-6
          else "!! SOUNDNESS VIOLATION !!")


if __name__ == "__main__":
    main()

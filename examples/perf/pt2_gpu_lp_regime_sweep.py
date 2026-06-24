"""PT-2 regime sweep: where does batched-GPU sound LP beat CPU HiGHS?

The real ACAS Xu approx-reach stars turn out to be *large, heterogeneous* LPs
with a tiny objective-batch (see pt2_gpu_lp_benchmark.py) -- not the regime
batched first-order GPU LP targets. This sweep characterizes the actual
crossover by varying var count, constraint count, and batch size on
structurally-identical LPs (one shared A,b,lb,ub, many objectives), reporting
CPU vs GPU throughput and confirming the GPU+NS bound encloses the exact CPU
optimum at every point.

Run:  python examples/perf/pt2_gpu_lp_regime_sweep.py
"""

import time

import numpy as np

from n2v.utils.lpsolver import solve_lp_batch
from n2v.utils.lpsolver_gpu import gpu_available, solve_lp_batch_gpu


def _make_lp(rng, n, m):
    A = rng.standard_normal((m, n))
    b = rng.uniform(1.0, 3.0, size=m) + np.abs(A).sum(1)
    lb = rng.uniform(-2.0, 0.0, size=n)
    ub = lb + rng.uniform(0.5, 3.0, size=n)
    return A, b, lb, ub


def _time(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    if not gpu_available():
        print("No CUDA device; aborting sweep.")
        return
    rng = np.random.default_rng(47)

    print(f"{'nVar':>5} {'mConstr':>7} {'batch':>6} "
          f"{'CPU LPs/s':>11} {'GPU LPs/s':>11} {'speedup':>8} "
          f"{'worst_inward':>13}")
    print("-" * 70)

    for n in (5, 20, 50):
        m = n
        for k in (100, 1000, 5000):
            A, b, lb, ub = _make_lp(rng, n, m)
            objs = [rng.standard_normal(n) for _ in range(k)]
            flags = [bool(i % 2) for i in range(k)]

            cpu_res = None

            def _cpu():
                nonlocal cpu_res
                cpu_res = solve_lp_batch(objs, A=A, b=b, lb=lb, ub=ub,
                                         minimize_flags=flags)

            gpu_res = None

            def _gpu():
                nonlocal gpu_res
                gpu_res = solve_lp_batch_gpu(objs, A=A, b=b, lb=lb, ub=ub,
                                             minimize_flags=flags, backend="pdhg")

            _gpu()  # warm up
            t_cpu = _time(_cpu)
            t_gpu = _time(_gpu)

            worst_inward = 0.0
            for i, do_min in enumerate(flags):
                if cpu_res[i] is None:
                    continue
                if do_min:
                    worst_inward = max(worst_inward, gpu_res[i] - cpu_res[i])
                else:
                    worst_inward = max(worst_inward, cpu_res[i] - gpu_res[i])

            print(f"{n:>5} {m:>7} {k:>6} "
                  f"{k/t_cpu:>11.0f} {k/t_gpu:>11.0f} {t_cpu/t_gpu:>7.2f}x "
                  f"{worst_inward:>13.2e}")


if __name__ == "__main__":
    main()

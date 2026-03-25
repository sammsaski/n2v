"""
Benchmark: LP Solver Ablation Study for n2v Exact Verification.

Compares three LP backends (CVXPY, SciPy linprog, highspy batch)
across multiple network sizes and reports mean ± std over N runs.

Usage:
    python benchmarks/benchmark_time.py
    python benchmarks/benchmark_time.py --runs 10
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from n2v.nn import NeuralNetwork
from n2v.sets import Star
from n2v.config import config
import n2v.utils.lpsolver as lps


def build_network(dims):
    """Build a simple feedforward ReLU network."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    model.eval()
    return model


def benchmark_solver(net, input_star, solver_name, n_runs, method='exact'):
    """Benchmark a single solver configuration. Returns (times_list, n_stars)."""
    orig_solver = config._default_lp_solver
    orig_highspy = lps._HAS_HIGHSPY

    try:
        if solver_name == 'CVXPY':
            config._default_lp_solver = 'default'
        elif solver_name == 'SciPy linprog':
            config._default_lp_solver = 'linprog'
            lps._HAS_HIGHSPY = False
        elif solver_name == 'highspy batch':
            # Verify highspy is actually importable before enabling
            try:
                import highspy  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "highspy is not installed. Install with: pip install highspy"
                )
            config._default_lp_solver = 'linprog'
            lps._HAS_HIGHSPY = True

        # Warm up
        result = net.reach(input_star, method=method)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            result = net.reach(input_star, method=method)
            times.append(time.perf_counter() - t0)

        return times, len(result)
    finally:
        # Always restore original solver configuration
        config._default_lp_solver = orig_solver
        lps._HAS_HIGHSPY = orig_highspy


def main():
    parser = argparse.ArgumentParser(description='n2v LP Solver Ablation')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per config')
    args = parser.parse_args()
    N_RUNS = args.runs

    print('=' * 90)
    print('n2v LP Solver Ablation Study — Exact Verification')
    print('=' * 90)
    print(f'Runs per config: {N_RUNS}')
    print(f'highspy available: {lps._HAS_HIGHSPY}')
    print()

    # ── Network configurations ──
    # (label, layer dims, input perturbation epsilon)
    networks = [
        # Tiny: few LP calls, tests dispatch overhead
        ('Tiny   (2->5->2)',             [2, 5, 2],           0.5),
        # Small: moderate LP calls
        ('Small  (5->20->10->2)',        [5, 20, 10, 2],      0.1),
        # Medium: many LP calls, crossing ReLUs
        ('Medium (5->50->30->5)',        [5, 50, 30, 5],      0.1),
        # Large: stress test LP solving
        ('Large  (10->100->50->10)',     [10, 100, 50, 10],   0.05),
        # Deep: 4 hidden layers
        ('Deep   (5->30->20->15->5)',    [5, 30, 20, 15, 5],  0.1),
        # XL: wide layers (skip CVXPY — too slow)
        ('XL     (10->200->100->10)',    [10, 200, 100, 10],  0.02),
    ]

    solvers = ['CVXPY', 'SciPy linprog', 'highspy batch']

    # Print header
    print(f'{"Network":<32} {"Solver":<16} {"Mean (ms)":>10} {"Std (ms)":>10} '
          f'{"Speedup":>10} {"Stars":>6} {"LPs (est)":>10}')
    print('-' * 96)

    for net_label, dims, eps in networks:
        np.random.seed(42)
        torch.manual_seed(42)

        model = build_network(dims)
        net = NeuralNetwork(model)

        lb = -eps * np.ones((dims[0], 1))
        ub = eps * np.ones((dims[0], 1))
        input_star = Star.from_bounds(lb, ub)

        results = {}
        for solver in solvers:
            # Skip CVXPY for XL (would take >30s)
            if solver == 'CVXPY' and 'XL' in net_label:
                results[solver] = (None, None, None)
                continue

            times, n_stars = benchmark_solver(net, input_star, solver, N_RUNS)
            results[solver] = (np.mean(times), np.std(times), n_stars)

        # Get CVXPY baseline for speedup calculation
        cvxpy_mean = results['CVXPY'][0]

        for solver in solvers:
            mean_t, std_t, n_stars = results[solver]

            if mean_t is None:
                print(f'{net_label:<32} {solver:<16} {"(skipped)":>10} {"":>10} '
                      f'{"":>10} {"":>6} {"":>10}')
                net_label = ''  # Don't repeat label
                continue

            if cvxpy_mean is not None:
                speedup = f'{cvxpy_mean / mean_t:.1f}x'
            else:
                speedup = '-'

            # Estimate LP calls: ~2 * n_stars * avg_crossing_neurons
            lp_est = f'~{n_stars * 2 * (dims[1] // 3)}'

            print(f'{net_label:<32} {solver:<16} {mean_t*1000:>10.1f} {std_t*1000:>10.1f} '
                  f'{speedup:>10} {n_stars:>6} {lp_est:>10}')
            net_label = ''  # Don't repeat label for subsequent solvers

        print()

    # ── Approx mode comparison (no LP splitting) ──
    print('=' * 90)
    print('Approx Mode (no LP splitting — baseline for non-LP overhead)')
    print('=' * 90)
    print()

    approx_nets = [
        ('Small  (5->20->10->2)',     [5, 20, 10, 2],    0.1),
        ('Medium (5->50->30->5)',     [5, 50, 30, 5],    0.1),
        ('Large  (10->100->50->10)', [10, 100, 50, 10], 0.05),
    ]

    print(f'{"Network":<32} {"Mean (ms)":>10} {"Std (ms)":>10}')
    print('-' * 54)

    for net_label, dims, eps in approx_nets:
        np.random.seed(42)
        torch.manual_seed(42)

        model = build_network(dims)
        net = NeuralNetwork(model)

        lb = -eps * np.ones((dims[0], 1))
        ub = eps * np.ones((dims[0], 1))
        input_star = Star.from_bounds(lb, ub)

        config._default_lp_solver = 'linprog'
        _ = net.reach(input_star, method='approx')

        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            net.reach(input_star, method='approx')
            times.append(time.perf_counter() - t0)

        print(f'{net_label:<32} {np.mean(times)*1000:>10.2f} {np.std(times)*1000:>10.2f}')

    print()
    print('Done.')


if __name__ == '__main__':
    main()

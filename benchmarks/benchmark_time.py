"""
Benchmark: LP Solver Ablation Study for n2v Verification.

Compares three LP backends (CVXPY, SciPy linprog, highspy batch)
across multiple network sizes and reports mean +/- std over N runs.

Runs approx mode first (all sizes), then exact mode (smaller nets only).

Usage:
    python benchmarks/benchmark_time.py
    python benchmarks/benchmark_time.py --runs 10
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from n2v.config import config
from n2v.nn import NeuralNetwork
from n2v.sets import Star
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


def benchmark_solver(
    net, input_star, solver_name, n_runs, method='exact',
):
    """Benchmark a single solver configuration.

    Returns (times_list, n_stars).
    """
    orig_solver = config._default_lp_solver
    orig_highspy = lps._HAS_HIGHSPY

    try:
        if solver_name == 'CVXPY':
            config._default_lp_solver = 'CLARABEL'
        elif solver_name == 'SciPy linprog':
            config._default_lp_solver = 'linprog'
            lps._HAS_HIGHSPY = False
        elif solver_name == 'highspy batch':
            try:
                import highspy  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "highspy is not installed. "
                    "Install with: pip install highspy"
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
        config._default_lp_solver = orig_solver
        lps._HAS_HIGHSPY = orig_highspy


# (label, layer dims, input perturbation epsilon)
NETWORKS = [
    ('Tiny   (2->5->2)', [2, 5, 2], 0.5),
    ('Small  (5->20->10->2)', [5, 20, 10, 2], 0.1),
    ('Medium (5->50->30->5)', [5, 50, 30, 5], 0.1),
    ('Large  (10->100->50->10)', [10, 100, 50, 10], 0.05),
    ('Deep   (5->30->20->15->5)', [5, 30, 20, 15, 5], 0.1),
    ('XL     (10->200->100->10)', [10, 200, 100, 10], 0.02),
]

SOLVERS = ['CVXPY', 'SciPy linprog', 'highspy batch']


def make_input(dims, eps):
    """Create input star from bounds."""
    lb = -eps * np.ones((dims[0], 1))
    ub = eps * np.ones((dims[0], 1))
    return Star.from_bounds(lb, ub)


def print_solver_table(networks, n_runs, method):
    """Run and print solver comparison table."""
    header = (
        f'{"Network":<32} {"Solver":<16} {"Mean (ms)":>10} '
        f'{"Std (ms)":>10} {"Speedup":>10} {"Stars":>6}'
    )
    print(header)
    print('-' * len(header))

    for net_label, dims, eps in networks:
        np.random.seed(42)
        torch.manual_seed(42)

        model = build_network(dims)
        net = NeuralNetwork(model)
        input_star = make_input(dims, eps)

        results = {}
        for solver in SOLVERS:
            # Skip CVXPY for XL (too slow)
            if solver == 'CVXPY' and 'XL' in net_label:
                results[solver] = (None, None, None)
                continue

            times, n_stars = benchmark_solver(
                net, input_star, solver, n_runs, method,
            )
            results[solver] = (
                np.mean(times), np.std(times), n_stars,
            )

        cvxpy_mean = results['CVXPY'][0]
        label = net_label

        for solver in SOLVERS:
            mean_t, std_t, n_stars = results[solver]

            if mean_t is None:
                print(
                    f'{label:<32} {solver:<16} '
                    f'{"(skipped)":>10} {"":>10} '
                    f'{"":>10} {"":>6}'
                )
                label = ''
                continue

            if cvxpy_mean is not None:
                speedup = f'{cvxpy_mean / mean_t:.1f}x'
            else:
                speedup = '-'

            print(
                f'{label:<32} {solver:<16} '
                f'{mean_t*1000:>10.1f} {std_t*1000:>10.1f} '
                f'{speedup:>10} {n_stars:>6}'
            )
            label = ''

        print()


def main():
    parser = argparse.ArgumentParser(
        description='n2v LP Solver Ablation',
    )
    parser.add_argument(
        '--runs', type=int, default=5,
        help='Number of runs per config',
    )
    args = parser.parse_args()
    n_runs = args.runs

    print('=' * 90)
    print('n2v LP Solver Ablation Study')
    print('=' * 90)
    print(f'Runs per config: {n_runs}')
    print(f'highspy available: {lps._HAS_HIGHSPY}')
    print()

    # -- Approx mode (all networks, no LP splitting) --
    print('=' * 90)
    print('Approx Mode (no LP splitting)')
    print('=' * 90)
    print()
    print_solver_table(NETWORKS, n_runs, method='approx')

    # -- Exact mode (smaller networks only) --
    exact_networks = [
        n for n in NETWORKS
        if 'Large' not in n[0] and 'XL' not in n[0] and 'Deep' not in n[0]
    ]

    print('=' * 90)
    print('Exact Mode (LP splitting)')
    print('=' * 90)
    print()
    print_solver_table(exact_networks, n_runs, method='exact')

    print('Done.')


if __name__ == '__main__':
    main()

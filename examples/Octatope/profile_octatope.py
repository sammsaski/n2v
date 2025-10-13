#!/usr/bin/env python3
"""
Profile Octatope Reachability to Find Bottlenecks

This script uses Python's cProfile to identify where time is being spent
during Octatope reachability analysis.
"""

import sys
import cProfile
import pstats
from io import StringIO
import numpy as np
import torch.nn as nn
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from n2v.sets import Octatope
from n2v.nn import NeuralNetwork


def create_test_model():
    """Create a small test model for profiling."""
    return nn.Sequential(
        nn.Linear(2, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )


def run_octatope_reach():
    """Run a simple Octatope reachability analysis."""
    # Create model
    model = create_test_model()
    net = NeuralNetwork(model)

    # Create input Octatope
    lb = np.array([[-0.5], [-0.5]], dtype=np.float32)
    ub = np.array([[0.5], [0.5]], dtype=np.float32)
    input_oct = Octatope.from_bounds(lb, ub)

    print(f"Input Octatope: dim={input_oct.dim}")
    print(f"Running exact reachability...")

    # Run reachability
    output_octs = net.reach(input_oct, method='exact')

    print(f"Output: {len(output_octs)} Octatope sets")
    return output_octs


def profile_octatope():
    """Profile Octatope reachability and print top time consumers."""
    print("="*60)
    print("PROFILING OCTATOPE REACHABILITY")
    print("="*60)

    # Create profiler
    profiler = cProfile.Profile()

    # Run with profiling
    profiler.enable()
    try:
        result = run_octatope_reach()
    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
    finally:
        profiler.disable()

    # Print statistics
    print("\n" + "="*60)
    print("TOP 30 TIME-CONSUMING FUNCTIONS")
    print("="*60)

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    # Print call statistics for specific functions
    print("\n" + "="*60)
    print("OCTATOPE-SPECIFIC FUNCTIONS")
    print("="*60)

    s2 = StringIO()
    ps2 = pstats.Stats(profiler, stream=s2)
    ps2.strip_dirs()
    ps2.sort_stats('cumulative')
    ps2.print_stats('octatope')
    print(s2.getvalue())

    # Print LP solver statistics
    print("\n" + "="*60)
    print("LP SOLVER FUNCTIONS")
    print("="*60)

    s3 = StringIO()
    ps3 = pstats.Stats(profiler, stream=s3)
    ps3.strip_dirs()
    ps3.sort_stats('cumulative')
    ps3.print_stats('cvxpy|optimize')
    print(s3.getvalue())


def compare_estimate_ranges():
    """Compare estimate_ranges performance between Star and Octatope."""
    from n2v.sets import Star

    print("\n" + "="*60)
    print("COMPARING estimate_ranges() PERFORMANCE")
    print("="*60)

    lb = np.array([[-0.5], [-0.5]], dtype=np.float32)
    ub = np.array([[0.5], [0.5]], dtype=np.float32)

    # Create Star
    star = Star.from_bounds(lb, ub)
    print(f"\nStar set: dim={star.dim}, nVar={star.nVar}")

    # Time Star estimate_ranges
    import time
    t_start = time.time()
    for _ in range(10):
        star.estimate_ranges()
    star_time = time.time() - t_start
    print(f"Star estimate_ranges (10 calls): {star_time:.4f}s ({star_time/10*1000:.2f}ms per call)")

    # Create Octatope
    oct = Octatope.from_bounds(lb, ub)
    print(f"\nOctatope set: dim={oct.dim}, num_vars={oct.utvpi.num_vars}")

    # Time Octatope estimate_ranges
    t_start = time.time()
    for _ in range(10):
        oct.estimate_ranges()
    oct_time = time.time() - t_start
    print(f"Octatope estimate_ranges (10 calls): {oct_time:.4f}s ({oct_time/10*1000:.2f}ms per call)")

    # Comparison
    if oct_time > star_time * 2:
        print(f"\n⚠️  Octatope estimate_ranges is {oct_time/star_time:.1f}x SLOWER than Star!")
        print("    This is a major bottleneck since estimate_ranges is called frequently.")
    else:
        print(f"\n✓ Octatope estimate_ranges is only {oct_time/star_time:.1f}x slower (acceptable)")


def main():
    """Run profiling analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Profile Octatope reachability to identify bottlenecks'
    )
    parser.add_argument('--compare-ranges', action='store_true',
                        help='Compare estimate_ranges() performance')
    parser.add_argument('--full-profile', action='store_true',
                        help='Run full cProfile analysis')

    args = parser.parse_args()

    if args.compare_ranges:
        compare_estimate_ranges()
    elif args.full_profile:
        profile_octatope()
    else:
        # Run both by default
        compare_estimate_ranges()
        profile_octatope()

    print("\n" + "="*60)
    print("PROFILING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test Octatope Reachability Analysis

This script creates small benchmark tests to verify and profile Octatope
reachability analysis. It compares performance against Star sets and helps
identify performance bottlenecks.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from n2v.sets import Star, Octatope
from n2v.nn import NeuralNetwork


def create_simple_network(input_dim=2, hidden_dims=[5], output_dim=1):
    """
    Create a simple feedforward network for testing.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension

    Returns:
        PyTorch Sequential model
    """
    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)


def run_benchmark(model, input_lb, input_ub, set_type='star', method='exact',
                  test_name='Test', verbose=True):
    """
    Run a single benchmark test.

    Args:
        model: PyTorch model
        input_lb: Lower bounds for input
        input_ub: Upper bounds for input
        set_type: 'star' or 'octatope'
        method: Reachability method
        test_name: Name for this test
        verbose: Print detailed output

    Returns:
        Dictionary with timing and result information
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"{test_name}")
        print(f"{'='*60}")
        print(f"Set type: {set_type.upper()}")
        print(f"Method: {method}")
        print(f"Input dimension: {len(input_lb)}")

    # Create neural network wrapper
    net = NeuralNetwork(model)

    # Create input set
    lb = input_lb.reshape(-1, 1).astype(np.float32)
    ub = input_ub.reshape(-1, 1).astype(np.float32)

    if set_type == 'star':
        input_set = Star.from_bounds(lb, ub)
    elif set_type == 'octatope':
        input_set = Octatope.from_bounds(lb, ub)
    else:
        raise ValueError(f"Unknown set type: {set_type}")

    if verbose:
        print(f"Input set created: {type(input_set).__name__}")
        if hasattr(input_set, 'nVar'):
            print(f"Number of variables: {input_set.nVar}")

    # Run reachability analysis
    if verbose:
        print(f"\nRunning reachability analysis...")

    t_start = time.time()
    try:
        output_sets = net.reach(input_set, method=method)
        elapsed = time.time() - t_start

        if verbose:
            print(f"✓ Completed in {elapsed:.3f} seconds")
            print(f"Number of output sets: {len(output_sets)}")

        # Get output bounds
        if output_sets:
            lb_out = np.ones(output_sets[0].dim) * 1000
            ub_out = np.ones(output_sets[0].dim) * -1000

            for out_set in output_sets:
                if set_type == 'octatope':
                    lb_temp, ub_temp = out_set.estimate_ranges()
                else:
                    lb_temp, ub_temp = out_set.estimate_ranges()
                lb_temp = lb_temp.flatten()
                ub_temp = ub_temp.flatten()
                lb_out = np.minimum(lb_temp, lb_out)
                ub_out = np.maximum(ub_temp, ub_out)

            if verbose:
                print(f"\nOutput bounds:")
                for i in range(len(lb_out)):
                    print(f"  Y_{i}: [{lb_out[i]:.6f}, {ub_out[i]:.6f}]")

        return {
            'success': True,
            'time': elapsed,
            'num_output_sets': len(output_sets),
            'set_type': set_type,
            'method': method,
            'test_name': test_name
        }

    except Exception as e:
        elapsed = time.time() - t_start
        if verbose:
            print(f"✗ Failed after {elapsed:.3f} seconds")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        return {
            'success': False,
            'time': elapsed,
            'error': str(e),
            'set_type': set_type,
            'method': method,
            'test_name': test_name
        }


def test_tiny_network():
    """Test 1: Tiny network (2-3-1) with small input region."""
    print("\n" + "="*60)
    print("TEST 1: Tiny Network (2-3-1)")
    print("="*60)

    # Create tiny network
    model = create_simple_network(input_dim=2, hidden_dims=[3], output_dim=1)

    # Small input region
    input_lb = np.array([-0.1, -0.1])
    input_ub = np.array([0.1, 0.1])

    # Run with Star (baseline)
    result_star = run_benchmark(model, input_lb, input_ub,
                                set_type='star', method='exact',
                                test_name='Baseline: Star Exact')

    # Run with Octatope exact
    result_oct_exact = run_benchmark(model, input_lb, input_ub,
                                     set_type='octatope', method='exact',
                                     test_name='Octatope Exact')

    # Run with Octatope exact-differentiable
    result_oct_exact_diff = run_benchmark(model, input_lb, input_ub,
                                          set_type='octatope', method='exact-differentiable',
                                          test_name='Octatope Exact (Differentiable)')

    # Run with Octatope approx
    result_oct_approx = run_benchmark(model, input_lb, input_ub,
                                      set_type='octatope', method='approx',
                                      test_name='Octatope Approx')

    return {
        'test': 'tiny_network',
        'star_exact': result_star,
        'oct_exact': result_oct_exact,
        'oct_exact_diff': result_oct_exact_diff,
        'oct_approx': result_oct_approx
    }


def test_small_network():
    """Test 2: Small network (2-5-1) with moderate input region."""
    print("\n" + "="*60)
    print("TEST 2: Small Network (2-5-1)")
    print("="*60)

    # Create small network
    model = create_simple_network(input_dim=2, hidden_dims=[5], output_dim=1)

    # Moderate input region
    input_lb = np.array([-0.5, -0.5])
    input_ub = np.array([0.5, 0.5])

    # Run with Star (baseline)
    result_star = run_benchmark(model, input_lb, input_ub,
                                set_type='star', method='exact',
                                test_name='Baseline: Star Exact')

    # Run with Octatope exact
    result_oct_exact = run_benchmark(model, input_lb, input_ub,
                                     set_type='octatope', method='exact',
                                     test_name='Octatope Exact')

    # Run with Octatope exact-differentiable
    result_oct_exact_diff = run_benchmark(model, input_lb, input_ub,
                                          set_type='octatope', method='exact-differentiable',
                                          test_name='Octatope Exact (Differentiable)')

    # Run with Octatope approx
    result_oct_approx = run_benchmark(model, input_lb, input_ub,
                                      set_type='octatope', method='approx',
                                      test_name='Octatope Approx')

    return {
        'test': 'small_network',
        'star_exact': result_star,
        'oct_exact': result_oct_exact,
        'oct_exact_diff': result_oct_exact_diff,
        'oct_approx': result_oct_approx
    }


def test_medium_network():
    """Test 3: Medium network (5-10-5-1) with ACAS-like input."""
    print("\n" + "="*60)
    print("TEST 3: Medium Network (5-10-5-1)")
    print("="*60)

    # Create medium network (similar to small ACAS Xu)
    model = create_simple_network(input_dim=5, hidden_dims=[10, 5], output_dim=1)

    # ACAS-like input bounds (scaled)
    input_lb = np.array([-0.3, -0.01, 0.4, 0.3, 0.3])
    input_ub = np.array([-0.29, 0.01, 0.5, 0.5, 0.5])

    # Run with Star (baseline) - set timeout
    print("\n⏱ Running Star exact (baseline)...")
    result_star = run_benchmark(model, input_lb, input_ub,
                                set_type='star', method='exact',
                                test_name='Baseline: Star Exact')

    if result_star['success'] and result_star['time'] > 60:
        print("\n⚠️  Star exact took > 60s, skipping Octatope exact methods to save time")
        print("    (Octatope exact would likely take much longer)")
        result_oct_exact = {
            'success': False,
            'skipped': True,
            'reason': 'Star baseline too slow',
            'test_name': 'Octatope Exact (Skipped)'
        }
        result_oct_exact_diff = {
            'success': False,
            'skipped': True,
            'reason': 'Star baseline too slow',
            'test_name': 'Octatope Exact Differentiable (Skipped)'
        }
    else:
        print("\n⏱ Running Octatope exact...")
        result_oct_exact = run_benchmark(model, input_lb, input_ub,
                                         set_type='octatope', method='exact',
                                         test_name='Octatope Exact')

        print("\n⏱ Running Octatope exact-differentiable...")
        result_oct_exact_diff = run_benchmark(model, input_lb, input_ub,
                                              set_type='octatope', method='exact-differentiable',
                                              test_name='Octatope Exact (Differentiable)')

    # Run with Octatope approx
    print("\n⏱ Running Octatope approx...")
    result_oct_approx = run_benchmark(model, input_lb, input_ub,
                                      set_type='octatope', method='approx',
                                      test_name='Octatope Approx')

    return {
        'test': 'medium_network',
        'star_exact': result_star,
        'oct_exact': result_oct_exact,
        'oct_exact_diff': result_oct_exact_diff,
        'oct_approx': result_oct_approx
    }


def print_summary(all_results):
    """Print summary of all test results."""
    print("\n" + "="*60)
    print("SUMMARY OF ALL TESTS")
    print("="*60)

    for test_results in all_results:
        test_name = test_results['test']
        print(f"\n{test_name.upper().replace('_', ' ')}:")
        print("-" * 40)

        # Star exact (baseline)
        star_result = test_results.get('star_exact', {})
        if star_result.get('success'):
            print(f"  Star Exact:        {star_result['time']:>8.3f}s  ({star_result['num_output_sets']} sets)")
        else:
            print(f"  Star Exact:        FAILED - {star_result.get('error', 'Unknown error')}")

        # Octatope exact
        oct_exact = test_results.get('oct_exact', {})
        if oct_exact.get('skipped'):
            print(f"  Octatope Exact:             SKIPPED - {oct_exact.get('reason', '')}")
        elif oct_exact.get('success'):
            speedup = star_result['time'] / oct_exact['time'] if star_result.get('success') else 0
            print(f"  Octatope Exact:             {oct_exact['time']:>8.3f}s  ({oct_exact['num_output_sets']} sets)  [{speedup:.2f}x vs Star]")
        else:
            print(f"  Octatope Exact:             FAILED - {oct_exact.get('error', 'Unknown error')}")

        # Octatope exact-differentiable
        oct_exact_diff = test_results.get('oct_exact_diff', {})
        if oct_exact_diff.get('skipped'):
            print(f"  Octatope Exact (Diff):      SKIPPED - {oct_exact_diff.get('reason', '')}")
        elif oct_exact_diff.get('success'):
            speedup = star_result['time'] / oct_exact_diff['time'] if star_result.get('success') else 0
            print(f"  Octatope Exact (Diff):      {oct_exact_diff['time']:>8.3f}s  ({oct_exact_diff['num_output_sets']} sets)  [{speedup:.2f}x vs Star]")
        else:
            print(f"  Octatope Exact (Diff):      FAILED - {oct_exact_diff.get('error', 'Unknown error')}")

        # Octatope approx
        oct_approx = test_results.get('oct_approx', {})
        if oct_approx.get('success'):
            speedup = star_result['time'] / oct_approx['time'] if star_result.get('success') else 0
            print(f"  Octatope Approx:            {oct_approx['time']:>8.3f}s  ({oct_approx['num_output_sets']} sets)  [{speedup:.2f}x vs Star]")
        else:
            print(f"  Octatope Approx:            FAILED - {oct_approx.get('error', 'Unknown error')}")

    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)

    # Analyze patterns
    for test_results in all_results:
        star_time = test_results.get('star_exact', {}).get('time', 0)
        oct_exact_time = test_results.get('oct_exact', {}).get('time', 0)
        oct_exact_diff_time = test_results.get('oct_exact_diff', {}).get('time', 0)
        oct_approx_time = test_results.get('oct_approx', {}).get('time', 0)

        if star_time > 0 and oct_exact_time > 0:
            if oct_exact_time > star_time * 10:
                print(f"\n⚠️  {test_results['test']}: Octatope exact is {oct_exact_time/star_time:.1f}x SLOWER than Star!")
                print(f"    This suggests a performance bottleneck in Octatope exact reachability.")
            elif oct_exact_time > star_time * 2:
                print(f"\n⚠️  {test_results['test']}: Octatope exact is {oct_exact_time/star_time:.1f}x slower than Star.")

        # Compare exact vs exact-differentiable
        if oct_exact_time > 0 and oct_exact_diff_time > 0:
            if oct_exact_diff_time < oct_exact_time:
                speedup = oct_exact_time / oct_exact_diff_time
                print(f"\n✓  {test_results['test']}: Differentiable solver is {speedup:.1f}x FASTER than CVXPY!")
            elif oct_exact_diff_time > oct_exact_time * 1.5:
                slowdown = oct_exact_diff_time / oct_exact_time
                print(f"\n⚠️  {test_results['test']}: Differentiable solver is {slowdown:.1f}x slower than CVXPY.")


def main():
    """Run all benchmark tests."""
    print("\n" + "="*60)
    print("OCTATOPE REACHABILITY BENCHMARK SUITE")
    print("="*60)
    print("\nThis suite tests Octatope reachability performance on")
    print("progressively larger networks to identify bottlenecks.")
    print("\nTests:")
    print("  1. Tiny network (2-3-1) - Should be fast")
    print("  2. Small network (2-5-1) - Should complete in seconds")
    print("  3. Medium network (5-10-5-1) - Similar to small ACAS Xu")

    all_results = []

    # Run tests
    try:
        # Test 1: Tiny network
        result1 = test_tiny_network()
        all_results.append(result1)

        # Test 2: Small network
        result2 = test_small_network()
        all_results.append(result2)

        # Test 3: Medium network (may be slow)
        result3 = test_medium_network()
        all_results.append(result3)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    if all_results:
        print_summary(all_results)

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
If Octatope exact is much slower than Star exact:
  1. Check if estimate_ranges() is being called efficiently
  2. Profile the ReLU splitting code in relu_octatope_exact
  3. Check if LP solving is using CVXPY efficiently
  4. Consider if differentiable solver would help (use exact-differentiable)

If Octatope approx is also slow:
  1. Check the approximation algorithm implementation
  2. Profile generator/constraint updates
  3. Check UTVPI constraint handling
    """)


if __name__ == "__main__":
    main()

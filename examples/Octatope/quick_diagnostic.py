#!/usr/bin/env python3
"""
Quick Diagnostic for Octatope Performance Issue

This script runs a minimal test to quickly identify the most likely bottleneck.
Should complete in under 30 seconds.
"""

import sys
import time
import numpy as np
import torch.nn as nn
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from n2v.sets import Star, Octatope
from n2v.nn import NeuralNetwork


def diagnostic_test():
    """Run quick diagnostic test."""
    print("="*60)
    print("QUICK OCTATOPE DIAGNOSTIC")
    print("="*60)
    print("\nThis test identifies the most likely performance bottleneck.")
    print("Expected completion time: < 30 seconds\n")

    # Create a minimal network
    print("[1/5] Creating test network (2-3-1)...")
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.ReLU(),
        nn.Linear(3, 1)
    )
    net = NeuralNetwork(model)
    print("      ✓ Network created")

    # Create input sets
    print("\n[2/5] Creating input sets...")
    lb = np.array([[-0.1], [-0.1]], dtype=np.float32)
    ub = np.array([[0.1], [0.1]], dtype=np.float32)

    star = Star.from_bounds(lb, ub)
    oct = Octatope.from_bounds(lb, ub)
    print(f"      ✓ Star: dim={star.dim}, nVar={star.nVar}")
    print(f"      ✓ Octatope: dim={oct.dim}, num_vars={oct.utvpi.num_vars}")

    # Test 1: estimate_ranges() speed
    print("\n[3/5] Testing estimate_ranges() performance...")
    print("      (This is called many times during exact reachability)")

    t_start = time.time()
    for i in range(20):
        star.estimate_ranges()
    star_time = time.time() - t_start

    t_start = time.time()
    for i in range(20):
        oct.estimate_ranges()
    oct_time = time.time() - t_start

    print(f"      Star (20 calls):     {star_time*1000:7.2f}ms ({star_time/20*1000:.2f}ms per call)")
    print(f"      Octatope (20 calls): {oct_time*1000:7.2f}ms ({oct_time/20*1000:.2f}ms per call)")
    print(f"      Ratio: {oct_time/star_time:.1f}x slower")

    estimate_ranges_slow = oct_time > star_time * 5

    # Test 2: Simple reachability
    print("\n[4/5] Testing simple reachability (approx method)...")

    t_start = time.time()
    star_output = net.reach(star, method='approx')
    star_reach_time = time.time() - t_start
    print(f"      Star approx:     {star_reach_time*1000:7.2f}ms ({len(star_output)} output sets)")

    t_start = time.time()
    oct_output = net.reach(oct, method='approx')
    oct_reach_time = time.time() - t_start
    print(f"      Octatope approx: {oct_reach_time*1000:7.2f}ms ({len(oct_output)} output sets)")
    print(f"      Ratio: {oct_reach_time/star_reach_time:.1f}x slower")

    approx_slow = oct_reach_time > star_reach_time * 5

    # Test 3: Exact reachability (small test)
    print("\n[5/5] Testing exact reachability...")
    print("      (This may take a few seconds)")

    t_start = time.time()
    star_exact = net.reach(star, method='exact')
    star_exact_time = time.time() - t_start
    print(f"      Star exact:     {star_exact_time:7.3f}s ({len(star_exact)} output sets)")

    t_start = time.time()
    try:
        # Set a reasonable timeout
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(20)  # 20 second timeout

        oct_exact = net.reach(oct, method='exact')
        signal.alarm(0)  # Cancel alarm

        oct_exact_time = time.time() - t_start
        print(f"      Octatope exact: {oct_exact_time:7.3f}s ({len(oct_exact)} output sets)")
        print(f"      Ratio: {oct_exact_time/star_exact_time:.1f}x slower")

        exact_slow = oct_exact_time > star_exact_time * 10
        timed_out = False

    except (TimeoutError, Exception) as e:
        signal.alarm(0)
        elapsed = time.time() - t_start
        print(f"      Octatope exact: TIMEOUT after {elapsed:.1f}s")
        print(f"      (Star completed in {star_exact_time:.3f}s)")
        exact_slow = True
        timed_out = True

    # Diagnosis
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if estimate_ranges_slow:
        print("\n🔴 PRIMARY ISSUE: estimate_ranges() is too slow")
        print(f"   Octatope.estimate_ranges() is {oct_time/star_time:.1f}x slower than Star")
        print("\n   Root Cause: Likely inefficient LP solving in _optimize_utvpi_lp()")
        print("\n   Impact: This function is called once per neuron during exact")
        print("   reachability. For a 5-50-50-5 network, that's ~105 calls per")
        print("   output set. If each call takes 50ms instead of 5ms, that's an")
        print("   extra ~4.7 seconds per output set!")
        print("\n   Recommended Fixes:")
        print("   1. Cache LP solver instances to avoid repeated setup")
        print("   2. Optimize UTVPI constraint handling")
        print("   3. Use differentiable solver (method='exact-differentiable')")
        print("   4. Consider tighter bounds to reduce LP problem size")

    elif approx_slow:
        print("\n🔴 PRIMARY ISSUE: Approximate reachability is slow")
        print(f"   Octatope approx is {oct_reach_time/star_reach_time:.1f}x slower")
        print("\n   Root Cause: Likely inefficient constraint or generator updates")
        print("\n   Recommended Fixes:")
        print("   1. Profile the approximation algorithms")
        print("   2. Check UTVPI constraint system operations")
        print("   3. Optimize generator matrix operations")

    elif exact_slow or timed_out:
        print("\n🔴 PRIMARY ISSUE: Exact reachability is extremely slow")
        if timed_out:
            print("   Octatope exact timed out (>20s) while Star completed quickly")
        else:
            print(f"   Octatope exact is {oct_exact_time/star_exact_time:.1f}x slower")
        print("\n   Root Cause: Combination of factors:")
        print("   - estimate_ranges() called many times (see above)")
        print("   - Possibly more splitting due to tighter bounds")
        print("   - LP solver overhead accumulates")
        print("\n   Recommended Fixes:")
        print("   1. Fix estimate_ranges() performance first (see above)")
        print("   2. Profile ReLU splitting logic")
        print("   3. Consider parallel processing")

    else:
        print("\n✅ No obvious performance issues detected")
        print("   Octatope is performing reasonably compared to Star")
        print("\n   If ACAS Xu is still slow, the issue may be:")
        print("   - Larger network size amplifies small inefficiencies")
        print("   - More ReLU splits due to tighter Octatope bounds")
        print("   - Try profiling on actual ACAS Xu network")

    # Recommendations
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)

    if estimate_ranges_slow:
        print("\n1. Look at sets/octatope.py, method estimate_ranges()")
        print("   Specifically check _optimize_utvpi_lp() and LP solver usage")
        print("\n2. Try the differentiable solver:")
        print("   cd ../AcasXu")
        print("   python verify_acasxu.py ... --set octatope --method exact-differentiable")
        print("\n3. Run full profiling:")
        print("   python profile_octatope.py")

    else:
        print("\n1. Run full benchmark suite:")
        print("   python test_octatope_reach.py")
        print("\n2. Profile with larger network:")
        print("   python profile_octatope.py --full-profile")
        print("\n3. Test on actual ACAS Xu with verbose output")

    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        diagnostic_test()
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Diagnostic failed with error: {e}")
        import traceback
        traceback.print_exc()

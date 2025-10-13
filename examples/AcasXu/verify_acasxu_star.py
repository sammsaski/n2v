#!/usr/bin/env python3
"""
ACAS Xu Verification Example

This script demonstrates verification of ACAS Xu neural network properties
using the NNV-Python toolkit with VNN-LIB format properties.

ACAS Xu is an airborne collision avoidance system that uses neural networks
to recommend advisory actions.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import n2v
from n2v.sets import Star
from n2v.nn import NeuralNetwork
from n2v.utils import load_vnnlib, verify_specification


def verify_acasxu_property(network_file: str, property_file: str,
                           reach_method: str = 'approx', timeout: float = 300.0,
                           use_parallel: bool = False, n_workers: int = None):
    """
    Verify an ACAS Xu property.

    Args:
        network_file: Path to ONNX network file
        property_file: Path to VNN-LIB property file
        reach_method: Reachability method ('exact' or 'approx')
        timeout: Timeout in seconds
        use_parallel: Enable parallel LP solving for better performance
        n_workers: Number of parallel workers (None = auto-detect)

    Returns:
        result: Verification result (0=violated, 1=verified, 2=unknown)
        time_elapsed: Computation time in seconds
        info: Dictionary with additional information
    """
    # Configure parallel processing if requested
    if use_parallel:
        # Enable both LP-level and Star-level parallelization
        n2v.set_parallel(True, n_workers=n_workers)
    else:
        n2v.set_parallel(False)

    print("="*80)
    print(f"Verifying: {os.path.basename(network_file)} with {os.path.basename(property_file)}")
    print("="*80)

    # Display parallel configuration
    if use_parallel:
        workers = n_workers if n_workers else "auto"
        print(f"\n⚡ Parallel processing enabled (workers: {workers})")
        print(f"   - LP-level parallelization: ON (within Stars)")
        print(f"   - Star-level parallelization: ON (across Stars)")

    # Load network
    print("\n1. Loading network...")
    from n2v.utils.model_loader import load_onnx
    model = load_onnx(network_file)
    net = NeuralNetwork(model)

    # Load property
    print("\n2. Loading property...")
    prop = load_vnnlib(property_file)
    print(f"   ✓ Property loaded: {property_file}")
    print(f"   Input dimension: {len(prop['lb'])}")
    print(f"   Input bounds:")
    for i in range(len(prop['lb'])):
        print(f"     X_{i}: [{prop['lb'][i]:.6f}, {prop['ub'][i]:.6f}]")

    print(f"   Output properties: {len(prop['prop'])}")
    if prop['prop']:
        print(f"   Property type: {'Single halfspace' if len(prop['prop']) == 1 else 'Multiple halfspaces (OR)'}")

    # Create input Star set
    print("\n3. Creating input set...")
    lb = prop['lb'].reshape(-1, 1).astype(np.float32)
    ub = prop['ub'].reshape(-1, 1).astype(np.float32)
    input_star = Star.from_bounds(lb, ub)
    print(f"   ✓ Input Star created:")
    print(f"     Dimension: {input_star.dim}")
    print(f"     Number of variables: {input_star.nVar}")

    # Perform reachability analysis
    print(f"\n4. Computing reachable set (method: {reach_method})...")
    t_start = time.time()

    try:
        # Use the new unified reachability interface
        reach_sets = net.reach(
            input_star,
            method=reach_method,
            parallel=use_parallel,
            n_workers=n_workers
        )

        time_reach = time.time() - t_start

        print(f"   ✓ Reachability completed in {time_reach:.2f} seconds")
        print(f"   Number of output stars: {len(reach_sets)}")

        # Get output bounds
        if reach_sets:
            lb_out = np.ones(5) * 1000
            ub_out = np.ones(5) * -1000

            for star in reach_sets:
                lb_temp, ub_temp = star.estimate_ranges()
                lb_temp = lb_temp.flatten()
                ub_temp = ub_temp.flatten()
                lb_out = np.minimum(lb_temp, lb_out)
                ub_out = np.maximum(ub_temp, ub_out)

            print(f"\n   Output bounds:")
            for i in range(5):
                print(f"     Y_{i}: [{lb_out[i]:.6f}, {ub_out[i]:.6f}]")

    except Exception as e:
        print(f"   ✗ Reachability failed: {e}")
        import traceback
        traceback.print_exc()
        return 2, time.time() - t_start, {'error': str(e), 'reach_method': reach_method, 'num_output_stars': 0}

    # Verify specification
    print(f"\n5. Verifying specification...")
    t_verify_start = time.time()

    try:
        result = verify_specification(reach_sets, prop['prop'])
        time_verify = time.time() - t_verify_start
        time_total = time.time() - t_start

        print(f"   ✓ Verification completed in {time_verify:.2f} seconds")

    except Exception as e:
        print(f"   ✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 2, time.time() - t_start, {'error': str(e)}

    # Report result
    print(f"\n" + "="*80)
    print("VERIFICATION RESULT")
    print("="*80)

    if result == 1:
        print("  Result: UNSAT")
        print("  Status: ✅ Property holds (no intersection with unsafe region)")
    elif result == 2:
        print("  Result: UNKNOWN")
        print("  Status: ⚠️  Cannot determine (possible intersection with unsafe region)")
    else:  # result == 0
        print("  Result: SAT")
        print("  Status: ❌ Property violated (counterexample exists)")

    print(f"\nTiming:")
    print(f"  Reachability: {time_reach:.2f}s")
    print(f"  Verification: {time_verify:.2f}s")
    print(f"  Total: {time_total:.2f}s")
    print("="*80 + "\n")

    info = {
        'num_output_stars': len(reach_sets),
        'time_reach': time_reach,
        'time_verify': time_verify,
        'time_total': time_total,
        'reach_method': reach_method
    }

    return result, time_total, info


def main():
    """Main function to run ACAS Xu verification."""
    import argparse

    # Get script directory
    script_dir = Path(__file__).parent

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Verify ACAS Xu neural network properties using VNN-LIB format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib
  %(prog)s onnx/ACASXU_run2a_1_5_batch_2000.onnx vnnlib/prop_3.vnnlib --method approx
  %(prog)s path/to/network.onnx path/to/property.vnnlib --timeout 600
  %(prog)s onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib --parallel --workers 4
        """
    )
    parser.add_argument('network', type=str,
                        help='Path to ONNX network file (relative to script dir or absolute)')
    parser.add_argument('property', type=str,
                        help='Path to VNN-LIB property file (relative to script dir or absolute)')
    parser.add_argument('--method', type=str, choices=['exact', 'approx'], default='exact',
                        help='Reachability method: exact or approx (default: exact)')
    parser.add_argument('--timeout', type=float, default=300.0,
                        help='Timeout in seconds (default: 300.0)')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel LP solving for better performance')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto-detect based on CPU cores)')

    args = parser.parse_args()

    # Resolve file paths (try relative to script dir first, then absolute)
    network_file = Path(args.network)
    if not network_file.is_absolute():
        network_file = script_dir / args.network

    property_file = Path(args.property)
    if not property_file.is_absolute():
        property_file = script_dir / args.property

    # Check files exist
    if not network_file.exists():
        print(f"Error: Network file not found: {network_file}")
        return 1

    if not property_file.exists():
        print(f"Error: Property file not found: {property_file}")
        return 1

    # Run verification
    print("\n" + "="*80)
    print("ACAS Xu Verification")
    print("="*80)
    print(f"Network: {network_file.name}")
    print(f"Property: {property_file.name}")
    print(f"Method: {args.method}")
    if args.parallel:
        workers = args.workers if args.workers else "auto"
        print(f"Parallel: enabled (workers: {workers})")
    else:
        print(f"Parallel: disabled")
    print("="*80 + "\n")

    try:
        result, time_elapsed, info = verify_acasxu_property(
            str(network_file),
            str(property_file),
            reach_method=args.method,
            timeout=args.timeout,
            use_parallel=args.parallel,
            n_workers=args.workers
        )

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Network: {network_file.name}")
        print(f"Property: {property_file.name}")
        print(f"Result: {['SAT', 'UNSAT', 'UNKNOWN'][result]}")
        print(f"Time: {time_elapsed:.2f}s")
        print(f"Method: {info['reach_method']}")
        print(f"Output stars: {info['num_output_stars']}")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

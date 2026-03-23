#!/usr/bin/env python3
"""
ACAS Xu Verification Example - Generalized for All Set Types

This script demonstrates verification of ACAS Xu neural network properties
using the NNV-Python toolkit with support for all set representations:
- Box: Interval arithmetic (fastest, least precise)
- Zono: Zonotopes (fast, good precision)
- Star: Star sets (precise, slower)
- Hexatope: Hexatope sets with DCS constraints (very precise)
- Octatope: Octatope sets with UTVPI constraints (very precise)

ACAS Xu is an airborne collision avoidance system that uses neural networks
to recommend advisory actions.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

import n2v
from n2v.sets import Star, Zono, Box, Hexatope, Octatope
from n2v.nn import NeuralNetwork
from n2v.utils import load_vnnlib
from n2v.utils.verify_specification import verify_specification
from n2v.utils.model_loader import load_onnx


def create_input_set(set_type: str, lb: np.ndarray, ub: np.ndarray):
    """
    Create input set of the specified type.

    Args:
        set_type: Type of set ('star', 'zono', 'box', 'hexatope', 'octatope')
        lb: Lower bounds
        ub: Upper bounds

    Returns:
        Input set of the specified type
    """
    if set_type == 'star':
        return Star.from_bounds(lb, ub)
    elif set_type == 'zono':
        return Zono.from_bounds(lb, ub)
    elif set_type == 'box':
        return Box(lb, ub)
    elif set_type == 'hexatope':
        return Hexatope.from_bounds(lb, ub)
    elif set_type == 'octatope':
        return Octatope.from_bounds(lb, ub)
    else:
        raise ValueError(f"Unknown set type: {set_type}")


def get_supported_methods(set_type: str):
    """
    Get supported reachability methods for a given set type.

    Args:
        set_type: Type of set

    Returns:
        List of supported method names
    """
    if set_type == 'star':
        return ['exact', 'approx']
    elif set_type in ['box', 'zono']:
        return ['approx']
    elif set_type in ['hexatope', 'octatope']:
        return ['approx']
    else:
        raise ValueError(f"Unknown set type: {set_type}")


def verify_acasxu_property(network_file: str, property_file: str,
                           set_type: str = 'star',
                           reach_method: str = 'exact',
                           timeout: float = 300.0,
                           use_parallel: bool = False,
                           n_workers: int = None):
    """
    Verify an ACAS Xu property using specified set type and method.

    Args:
        network_file: Path to ONNX network file
        property_file: Path to VNN-LIB property file
        set_type: Set representation ('star', 'zono', 'box', 'hexatope', 'octatope')
        reach_method: Reachability method:
            - For Star: 'exact' or 'approx'
            - For Box/Zono: 'approx'
            - For Hexatope/Octatope: 'approx'
        timeout: Timeout in seconds
        use_parallel: Enable parallel processing (Star only)
        n_workers: Number of parallel workers (None = use default of 4)

    Returns:
        result: Verification result (0=violated, 1=verified, 2=unknown)
        time_elapsed: Computation time in seconds
        info: Dictionary with additional information
    """
    # Validate method for set type
    supported_methods = get_supported_methods(set_type)
    if reach_method not in supported_methods:
        raise ValueError(
            f"Method '{reach_method}' not supported for {set_type}. "
            f"Supported methods: {', '.join(supported_methods)}"
        )

    # Configure parallel processing if requested (Star only)
    if use_parallel and set_type == 'star':
        n2v.set_parallel(True, n_workers=n_workers)
    else:
        n2v.set_parallel(False)

    print("="*80)
    print(f"Verifying: {os.path.basename(network_file)} with {os.path.basename(property_file)}")
    print("="*80)
    print(f"Set type: {set_type.upper()}")
    print(f"Method: {reach_method}")

    # Display parallel configuration
    if use_parallel and set_type == 'star':
        workers = n_workers if n_workers else "auto"
        print(f"\n⚡ Parallel processing enabled (workers: {workers})")
        print(f"   - LP-level parallelization: ON (within Stars)")
        print(f"   - Star-level parallelization: ON (across Stars)")
    elif use_parallel and set_type != 'star':
        print(f"\n⚠️  Parallel processing only supported for Star sets")

    # Load network
    print("\n1. Loading network...")
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

    # Create input set
    print(f"\n3. Creating input set ({set_type})...")
    lb = prop['lb'].reshape(-1, 1).astype(np.float32)
    ub = prop['ub'].reshape(-1, 1).astype(np.float32)
    input_set = create_input_set(set_type, lb, ub)
    print(f"   ✓ Input {set_type.capitalize()} created:")
    print(f"     Dimension: {input_set.dim}")
    if hasattr(input_set, 'nVar'):
        print(f"     Number of variables: {input_set.nVar}")

    # Perform reachability analysis
    print(f"\n4. Computing reachable set (method: {reach_method})...")
    t_start = time.time()

    try:
        # Use the unified reachability interface
        kwargs = {}
        if set_type == 'star' and use_parallel:
            kwargs['parallel'] = use_parallel
            kwargs['n_workers'] = n_workers

        reach_sets = net.reach(input_set, method=reach_method, **kwargs)

        time_reach = time.time() - t_start

        print(f"   ✓ Reachability completed in {time_reach:.2f} seconds")
        print(f"   Number of output sets: {len(reach_sets)}")

        # Get output bounds
        if reach_sets:
            lb_out = np.ones(5) * 1000
            ub_out = np.ones(5) * -1000

            for output_set in reach_sets:
                if set_type == 'box':
                    lb_temp, ub_temp = output_set.get_range()
                else:
                    lb_temp, ub_temp = output_set.estimate_ranges()
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
        return 2, time.time() - t_start, {
            'error': str(e),
            'set_type': set_type,
            'reach_method': reach_method,
            'num_output_sets': 0
        }

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
        'num_output_sets': len(reach_sets),
        'time_reach': time_reach,
        'time_verify': time_verify,
        'time_total': time_total,
        'set_type': set_type,
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
        description='Verify ACAS Xu neural network properties using various set representations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Set Types and Supported Methods:
  - box:      Fast interval arithmetic [Methods: approx]
  - zono:     Zonotope representation [Methods: approx]
  - star:     Star set representation [Methods: exact, approx]
  - hexatope: Hexatope with DCS [Methods: approx]
  - octatope: Octatope with UTVPI [Methods: approx]

Examples:
  # Use Star sets with exact method
  %(prog)s onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib --set star --method exact

  # Use Box for fast approximate verification
  %(prog)s onnx/ACASXU_run2a_1_5_batch_2000.onnx vnnlib/prop_3.vnnlib --set box --method approx

  # Use Star with parallel processing
  %(prog)s onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib --set star --method exact --parallel --workers 4

  # Compare different set types
  %(prog)s onnx/ACASXU_run2a_1_4_batch_2000.onnx vnnlib/prop_3.vnnlib --set octatope --method approx
        """
    )
    parser.add_argument('network', type=str,
                        help='Path to ONNX network file (relative to script dir or absolute)')
    parser.add_argument('property', type=str,
                        help='Path to VNN-LIB property file (relative to script dir or absolute)')
    parser.add_argument('--set', type=str, dest='set_type',
                        choices=['box', 'zono', 'star', 'hexatope', 'octatope'],
                        default='star',
                        help='Set representation type (default: star)')
    parser.add_argument('--method', type=str,
                        choices=['exact', 'approx'],
                        default='exact',
                        help='Reachability method (default: exact). Note: not all methods supported by all set types.')
    parser.add_argument('--timeout', type=float, default=300.0,
                        help='Timeout in seconds (default: 300.0)')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel processing (Star only)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: 4; set to CPU count for best performance)')

    args = parser.parse_args()

    # Validate that the method is supported for the chosen set type
    supported_methods = get_supported_methods(args.set_type)
    if args.method not in supported_methods:
        print(f"Error: Method '{args.method}' not supported for {args.set_type} sets.")
        print(f"Supported methods for {args.set_type}: {', '.join(supported_methods)}")
        return 1

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
    print("ACAS Xu Verification - Generalized")
    print("="*80)
    print(f"Network: {network_file.name}")
    print(f"Property: {property_file.name}")
    print(f"Set type: {args.set_type.upper()}")
    print(f"Method: {args.method}")
    if args.parallel and args.set_type == 'star':
        workers = args.workers if args.workers else "auto"
        print(f"Parallel: enabled (workers: {workers})")
    else:
        print(f"Parallel: disabled")
    print("="*80 + "\n")

    try:
        result, time_elapsed, info = verify_acasxu_property(
            str(network_file),
            str(property_file),
            set_type=args.set_type,
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
        print(f"Set type: {info['set_type'].upper()}")
        print(f"Method: {info['reach_method']}")
        print(f"Result: {['SAT', 'UNSAT', 'UNKNOWN'][result]}")
        print(f"Time: {time_elapsed:.2f}s")
        print(f"Output sets: {info['num_output_sets']}")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

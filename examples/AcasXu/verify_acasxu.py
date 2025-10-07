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

from n2v.sets import Star
from n2v.nn.reach.reach_star import reach_star_exact, reach_star_approx
from n2v.utils import load_vnnlib, verify_specification


def verify_acasxu_property(network_file: str, property_file: str,
                           reach_method: str = 'approx', timeout: float = 300.0):
    """
    Verify an ACAS Xu property.

    Args:
        network_file: Path to ONNX network file
        property_file: Path to VNN-LIB property file
        reach_method: Reachability method ('exact' or 'approx')
        timeout: Timeout in seconds

    Returns:
        result: Verification result (0=violated, 1=verified, 2=unknown)
        time_elapsed: Computation time in seconds
        info: Dictionary with additional information
    """
    print("="*80)
    print(f"Verifying: {os.path.basename(network_file)} with {os.path.basename(property_file)}")
    print("="*80)

    # Load network
    print("\n1. Loading network...")
    print(f"   NOTE: ACAS Xu ONNX models have embedded weights as inputs.")
    print(f"   Current limitation: onnx2torch doesn't support this format.")
    print(f"   TODO: Implement custom ONNX loader or use ONNX Runtime wrapper.")
    print(f"   ")
    print(f"   For now, this is a demonstration of the verification workflow.")
    print(f"   Skipping actual network loading...")

    # TODO: Implement proper ONNX loading for ACAS Xu format
    # For now, we'll create a dummy network for demonstration
    import torch.nn as nn
    net = nn.Sequential(
        nn.Linear(5, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 5)
    )
    net.eval()
    print(f"   ⚠️  Using dummy network (same architecture as ACAS Xu)")

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
        if reach_method == 'exact':
            reach_sets = reach_star_exact(net, [input_star])
        else:  # approx
            reach_sets = reach_star_approx(net, [input_star])

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
        return 2, time.time() - t_start, {'error': str(e)}

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
    print(f"Result: {result}")

    if result == 1:
        print("  Status: ✅ VERIFIED (property holds)")
    elif result == 2:
        print("  Status: ⚠️  UNKNOWN (property may not hold)")
    else:
        print("  Status: ❌ VIOLATED (property does not hold)")

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
    # Get script directory
    script_dir = Path(__file__).parent

    # Example: Verify property 1 with network 1_1
    network_file = script_dir / "onnx" / "ACASXU_run2a_1_1_batch_2000.onnx"
    property_file = script_dir / "vnnlib" / "prop_1.vnnlib"

    # Check files exist
    if not network_file.exists():
        print(f"Error: Network file not found: {network_file}")
        return

    if not property_file.exists():
        print(f"Error: Property file not found: {property_file}")
        return

    # Run verification
    print("\n" + "="*80)
    print("ACAS Xu Verification")
    print("="*80)
    print(f"Network: {network_file.name}")
    print(f"Property: {property_file.name}")
    print("="*80 + "\n")

    try:
        result, time_elapsed, info = verify_acasxu_property(
            str(network_file),
            str(property_file),
            reach_method='approx',  # Start with approx for speed
            timeout=300.0
        )

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Network: {network_file.name}")
        print(f"Property: {property_file.name}")
        print(f"Result: {['VIOLATED', 'VERIFIED', 'UNKNOWN'][result]}")
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

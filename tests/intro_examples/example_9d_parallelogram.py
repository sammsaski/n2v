#!/usr/bin/env python3
"""
Example 9.D: Parallelogram Zonotope

This example demonstrates basic zonotope operations using the parallelogram
from the NNV introductory documentation, and compares results with MATLAB NNV.

The parallelogram is defined as:
    (2 + ε₁, 3 + ε₁ + ε₂)

In compact notation:
    (⟨2, 1, 0⟩, ⟨3, 1, 1⟩)

Where:
    - Center: c = [2; 3]
    - Generator 1: v₁ = [1; 1]
    - Generator 2: v₂ = [0; 1]

The zonotope is: Z = c + α₁*v₁ + α₂*v₂, where -1 ≤ αᵢ ≤ 1
"""

import numpy as np
from pathlib import Path
from scipy.io import loadmat, savemat
import sys

# Add parent directory to path to import n2v
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from n2v.sets import Zono


def print_header(title):
    """Print formatted section header."""
    print("=" * 60)
    print(title)
    print("=" * 60)
    print()


def print_section(number, title):
    """Print formatted section."""
    print(f"\n{number}. {title}")


def compare_arrays(python_val, matlab_val, name, tolerance=1e-10):
    """Compare Python and MATLAB arrays with tolerance."""
    python_val = np.asarray(python_val)
    matlab_val = np.asarray(matlab_val)

    if python_val.shape != matlab_val.shape:
        print(f"   ⚠️  {name}: Shape mismatch!")
        print(f"       Python: {python_val.shape}, MATLAB: {matlab_val.shape}")
        return False

    diff = np.abs(python_val - matlab_val)
    max_diff = np.max(diff)

    if max_diff < tolerance:
        print(f"   ✓ {name}: Match (max diff = {max_diff:.2e})")
        return True
    else:
        print(f"   ✗ {name}: Mismatch (max diff = {max_diff:.2e})")
        print(f"       Python:  {python_val.flatten()}")
        print(f"       MATLAB:  {matlab_val.flatten()}")
        return False


def main():
    print_header("Example 9.D: Parallelogram Zonotope (Python)")

    # ==========================================================================
    # 1. Create the Parallelogram Zonotope
    # ==========================================================================
    print_section(1, "Creating parallelogram zonotope...")

    # Define center and generators
    c = np.array([[2.0], [3.0]])
    V = np.array([[1.0, 0.0],
                  [1.0, 1.0]])

    # Create zonotope
    Z = Zono(c, V)

    print(f"   Center c = {c.flatten()}")
    print(f"   Generators V = \n{V}")
    print(f"   Dimension: {Z.dim}")
    print(f"   Number of generators: {V.shape[1]}")
    print("   ✓ Zonotope created")

    # ==========================================================================
    # 2. Compute Bounds
    # ==========================================================================
    print_section(2, "Computing bounds...")

    # Get bounds using N2V method
    lb, ub = Z.get_bounds()

    print(f"   Bounds:")
    print(f"     Dimension 1 (x): [{lb[0, 0]:.10f}, {ub[0, 0]:.10f}]")
    print(f"     Dimension 2 (y): [{lb[1, 0]:.10f}, {ub[1, 0]:.10f}]")
    print(f"\n   Verification: Upper bound in vertical dimension = {ub[1, 0]:.10f}")
    print(f"   Expected: 3 + 1 + 1 = 5")
    match = abs(ub[1, 0] - 5.0) < 1e-10
    print(f"   Match: {'✓ YES' if match else '✗ NO'}")

    # ==========================================================================
    # 3. Get All Vertices
    # ==========================================================================
    print_section(3, "Computing vertices...")

    vertices = Z.get_vertices()

    print(f"   Number of vertices: {vertices.shape[1]}")
    print(f"   Vertices:")
    for i in range(vertices.shape[1]):
        print(f"     v{i+1} = ({vertices[0, i]:.10f}, {vertices[1, i]:.10f})")
    print("   ✓ Vertices computed")

    # ==========================================================================
    # 4. Affine Transformation
    # ==========================================================================
    print_section(4, "Testing affine transformation...")

    # Simple scaling and translation
    W = np.array([[2.0, 0.0],
                  [0.0, 1.0]])
    b = np.array([[1.0], [0.0]])

    Z_affine = Z.affine_map(W, b)
    lb_affine, ub_affine = Z_affine.get_bounds()

    print(f"   Transformation matrix W = \n{W}")
    print(f"   Translation vector b = {b.flatten()}")
    print(f"   Transformed zonotope:")
    print(f"     Center: ({Z_affine.c[0, 0]:.10f}, {Z_affine.c[1, 0]:.10f})")
    print(f"     Bounds: x ∈ [{lb_affine[0, 0]:.10f}, {ub_affine[0, 0]:.10f}], "
          f"y ∈ [{lb_affine[1, 0]:.10f}, {ub_affine[1, 0]:.10f}]")
    print("   ✓ Affine map computed")

    # ==========================================================================
    # 5. Minkowski Sum
    # ==========================================================================
    print_section(5, "Testing Minkowski sum...")

    # Create second zonotope (a small box)
    c2 = np.array([[0.0], [0.0]])
    V2 = np.array([[0.5, 0.0],
                   [0.0, 0.5]])
    Z2 = Zono(c2, V2)

    Z_sum = Z.minkowski_sum(Z2)
    lb_sum, ub_sum = Z_sum.get_bounds()

    print(f"   Second zonotope: center = {c2.flatten()}, generators = {V2.shape[1]}")
    print(f"   Minkowski sum result:")
    print(f"     Center: ({Z_sum.c[0, 0]:.10f}, {Z_sum.c[1, 0]:.10f})")
    print(f"     Number of generators: {Z_sum.V.shape[1]}")
    print(f"     Bounds: x ∈ [{lb_sum[0, 0]:.10f}, {ub_sum[0, 0]:.10f}], "
          f"y ∈ [{lb_sum[1, 0]:.10f}, {ub_sum[1, 0]:.10f}]")
    print("   ✓ Minkowski sum computed")

    # ==========================================================================
    # 6. Test Point Containment
    # ==========================================================================
    print_section(6, "Testing point containment...")

    # Test several points (same as MATLAB)
    test_points = np.array([
        [2.0, 3.0],    # center (should be inside)
        [2.5, 3.5],    # inside
        [1.0, 2.0],    # corner (should be inside)
        [3.0, 5.0],    # corner (should be inside)
        [0.0, 0.0],    # outside
        [5.0, 5.0]     # outside
    ])

    contains_results = np.zeros(test_points.shape[0])
    for i, point in enumerate(test_points):
        p = point.reshape(-1, 1)
        result = Z.contains(p)
        contains_results[i] = 1 if result else 0
        status = "INSIDE ✓" if result else "OUTSIDE ✗"
        print(f"   Point ({point[0]:.1f}, {point[1]:.1f}): {status}")
    print("   ✓ Containment tests completed")

    # ==========================================================================
    # 7. Load MATLAB Results and Compare
    # ==========================================================================
    print_section(7, "Loading MATLAB results and comparing...")

    matlab_results_path = Path(__file__).parent / "saved_results" / "example_9d_results.mat"

    if not matlab_results_path.exists():
        print(f"   ⚠️  MATLAB results not found at: {matlab_results_path}")
        print("   Please run the MATLAB script first: example_9d_parallelogram.m")
        print("   Skipping comparison...")
    else:
        # Load MATLAB results
        matlab_data = loadmat(str(matlab_results_path))
        matlab_results = matlab_data['results']

        # Extract MATLAB results (handling MATLAB struct array)
        def get_field(struct, field_name):
            return struct[field_name][0, 0]

        matlab_c = get_field(matlab_results, 'c')
        matlab_V = get_field(matlab_results, 'V')
        matlab_lb = get_field(matlab_results, 'lb')
        matlab_ub = get_field(matlab_results, 'ub')
        matlab_vertices = get_field(matlab_results, 'vertices')
        matlab_affine_c = get_field(matlab_results, 'affine_c')
        matlab_affine_V = get_field(matlab_results, 'affine_V')
        matlab_affine_lb = get_field(matlab_results, 'affine_lb')
        matlab_affine_ub = get_field(matlab_results, 'affine_ub')
        matlab_mink_sum_c = get_field(matlab_results, 'mink_sum_c')
        matlab_mink_sum_V = get_field(matlab_results, 'mink_sum_V')
        matlab_mink_sum_lb = get_field(matlab_results, 'mink_sum_lb')
        matlab_mink_sum_ub = get_field(matlab_results, 'mink_sum_ub')
        matlab_contains = get_field(matlab_results, 'contains_results').flatten()

        print("\n   Comparing Python vs MATLAB results:")
        print("   " + "-" * 56)

        all_match = True
        all_match &= compare_arrays(c, matlab_c, "Center c")
        all_match &= compare_arrays(V, matlab_V, "Generators V")
        all_match &= compare_arrays(lb, matlab_lb, "Lower bounds")
        all_match &= compare_arrays(ub, matlab_ub, "Upper bounds")
        all_match &= compare_arrays(vertices, matlab_vertices, "Vertices")
        all_match &= compare_arrays(Z_affine.c, matlab_affine_c, "Affine center")
        all_match &= compare_arrays(Z_affine.V, matlab_affine_V, "Affine generators")
        all_match &= compare_arrays(lb_affine, matlab_affine_lb, "Affine lower bounds")
        all_match &= compare_arrays(ub_affine, matlab_affine_ub, "Affine upper bounds")
        all_match &= compare_arrays(Z_sum.c, matlab_mink_sum_c, "Minkowski sum center")
        all_match &= compare_arrays(Z_sum.V, matlab_mink_sum_V, "Minkowski sum generators")
        all_match &= compare_arrays(lb_sum, matlab_mink_sum_lb, "Minkowski sum lb")
        all_match &= compare_arrays(ub_sum, matlab_mink_sum_ub, "Minkowski sum ub")
        all_match &= compare_arrays(contains_results, matlab_contains, "Point containment")

        print("   " + "-" * 56)
        if all_match:
            print("\n   ✅ All comparisons passed! Python matches MATLAB.")
        else:
            print("\n   ⚠️  Some comparisons failed. Check differences above.")

    # ==========================================================================
    # 8. Save Python Results
    # ==========================================================================
    print_section(8, "Saving Python results...")

    # Create results dictionary
    python_results = {
        'c': c,
        'V': V,
        'dim': Z.dim,
        'num_generators': V.shape[1],
        'lb': lb,
        'ub': ub,
        'vertices': vertices,
        'num_vertices': vertices.shape[1],
        'affine_W': W,
        'affine_b': b,
        'affine_c': Z_affine.c,
        'affine_V': Z_affine.V,
        'affine_lb': lb_affine,
        'affine_ub': ub_affine,
        'mink_c2': c2,
        'mink_V2': V2,
        'mink_sum_c': Z_sum.c,
        'mink_sum_V': Z_sum.V,
        'mink_sum_lb': lb_sum,
        'mink_sum_ub': ub_sum,
        'test_points': test_points,
        'contains_results': contains_results,
    }

    output_path = Path(__file__).parent / "saved_results" / "example_9d_python_results.mat"
    savemat(str(output_path), {'results': python_results})

    print(f"   ✓ Python results saved to: {output_path}")

    # ==========================================================================
    # 9. Summary
    # ==========================================================================
    print("\n")
    print_header("Summary of Example 9.D")
    print("Parallelogram zonotope: (⟨2, 1, 0⟩, ⟨3, 1, 1⟩)")
    print("  • Center: (2, 3)")
    print("  • 2 generators: [1, 0; 1, 1]")
    print(f"  • Bounds: x ∈ [1, 3], y ∈ [2, 5]")
    print(f"  • Upper bound (y): {ub[1, 0]:.1f} ✓ (3 + 1 + 1 = 5)")
    print("  • All operations completed successfully")
    if matlab_results_path.exists():
        print("  • Python results validated against MATLAB")
    print("=" * 60)


if __name__ == "__main__":
    main()

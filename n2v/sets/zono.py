"""
Zonotope set representation.

Represents a zonotope: Z = c + sum_{i=1}^n alpha_i * v_i, where -1 <= alpha_i <= 1
Translated from MATLAB NNV Zono.m
"""

import numpy as np
from typing import Optional, Tuple, List, TYPE_CHECKING
from scipy.linalg import svd

# TYPE_CHECKING imports for type hints (avoid circular import at runtime)
if TYPE_CHECKING:
    from n2v.sets.star import Star
    from n2v.sets.box import Box
    from n2v.sets.image_zono import ImageZono
    from n2v.sets.image_star import ImageStar

# NOTE: Runtime imports of n2v.sets.* modules are kept inline to avoid circular dependencies


class Zono:
    """
    Zonotope class.

    A Zonotope is defined by:
        Z = {c + V*alpha | -1 <= alpha_i <= 1 for all i}
    where c is the center and V = [v1, v2, ..., vn] are generators.

    Attributes:
        c: Center vector (n, 1)
        V: Generator matrix (n, m) where m is number of generators
        dim: Dimension of the zonotope
    """

    def __init__(self, c: Optional[np.ndarray] = None, V: Optional[np.ndarray] = None):
        """
        Initialize a Zonotope.

        Args:
            c: Center vector (n,) or (n, 1)
            V: Generator matrix (n, m)

        Raises:
            ValueError: If dimensions are inconsistent
        """
        if c is None and V is None:
            # Empty constructor
            self.c = np.array([]).reshape(-1, 1)
            self.V = np.array([]).reshape(0, 0)
            self.dim = 0
        elif c is not None and V is not None:
            # Full constructor - preserve dtype if float, otherwise default to float64
            c = np.asarray(c)
            V = np.asarray(V)
            if not np.issubdtype(c.dtype, np.floating):
                c = c.astype(np.float64)
            if not np.issubdtype(V.dtype, np.floating):
                V = V.astype(np.float64)

            # Ensure c is column vector
            if c.ndim == 1:
                c = c.reshape(-1, 1)
            if c.shape[1] != 1:
                raise ValueError("Center must be a column vector")

            # Ensure V is 2D
            if V.ndim == 1:
                V = V.reshape(-1, 1)

            # Validate dimensions
            if c.shape[0] != V.shape[0]:
                raise ValueError(
                    f"Dimension mismatch: center has {c.shape[0]} dims, "
                    f"generators have {V.shape[0]} dims"
                )

            self.c = c
            self.V = V
            self.dim = c.shape[0]
        else:
            raise ValueError("Must provide both c and V, or neither")

    def __repr__(self) -> str:
        """Return string representation of the Zonotope."""
        return f"Zono(dim={self.dim}, n_generators={self.V.shape[1] if self.V.size > 0 else 0})"

    @classmethod
    def from_bounds(cls, lb: np.ndarray, ub: np.ndarray) -> 'Zono':
        """
        Create Zonotope from axis-aligned box bounds.

        Args:
            lb: Lower bounds (n,) or (n, 1)
            ub: Upper bounds (n,) or (n, 1)

        Returns:
            Zonotope with center=(lb+ub)/2 and generators for each dimension
        """
        lb = np.asarray(lb, dtype=np.float64).reshape(-1, 1)
        ub = np.asarray(ub, dtype=np.float64).reshape(-1, 1)

        if lb.shape != ub.shape:
            raise ValueError(f"lb and ub must have same shape, got {lb.shape} and {ub.shape}")

        # Center at midpoint
        c = (lb + ub) / 2.0

        # Generators: one per dimension, each with range (ub-lb)/2
        n = lb.shape[0]
        V = np.diag(((ub - lb) / 2.0).flatten())

        return cls(c, V)

    # ======================== Geometric Operations ========================

    def affine_map(self, W: np.ndarray, b: Optional[np.ndarray] = None) -> 'Zono':
        """
        Apply affine transformation: W*Z + b.

        Args:
            W: Affine mapping matrix (m, n)
            b: Mapping vector (m,) or (m, 1), optional

        Returns:
            New Zono object
        """
        W = np.asarray(W, dtype=np.float64)

        # Validate dimensions
        if W.shape[1] != self.dim:
            raise ValueError(f"Matrix W has {W.shape[1]} columns, expected {self.dim}")

        # Transform center
        new_c = W @ self.c
        if b is not None:
            b = np.asarray(b, dtype=np.float64).reshape(-1, 1)
            new_c = new_c + b

        # Transform generators
        new_V = W @ self.V

        return Zono(new_c, new_V)

    def minkowski_sum(self, other: 'Zono') -> 'Zono':
        """
        Compute Minkowski sum with another zonotope.

        Args:
            other: Another Zono object

        Returns:
            New Zono representing Z1 ⊕ Z2
        """
        if not isinstance(other, Zono):
            raise TypeError("Can only compute Minkowski sum with another Zono")
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")

        new_c = self.c + other.c
        new_V = np.hstack([self.V, other.V])
        return Zono(new_c, new_V)

    def convex_hull(self, other: 'Zono') -> 'Zono':
        """
        Compute over-approximation of convex hull with another zonotope.

        Based on Girard (HSCC 2005). Note: true convex hull is generally
        not a zonotope; this returns an over-approximation.

        Args:
            other: Another Zono object

        Returns:
            New Zono over-approximating the convex hull
        """
        if not isinstance(other, Zono):
            raise TypeError("Can only compute convex hull with another Zono")
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")

        new_c = 0.5 * (self.c + other.c)
        new_V = np.hstack([self.V, other.V, 0.5 * (self.c - other.c)])
        return Zono(new_c, new_V)

    def convex_hull_with_linear_transform(self, L: np.ndarray) -> 'Zono':
        """
        Convex hull of zonotope with its linear transformation.

        Args:
            L: Square transformation matrix (n, n)

        Returns:
            New Zono over-approximating hull(Z, L*Z)
        """
        L = np.asarray(L, dtype=np.float64)

        if L.shape[0] != L.shape[1]:
            raise ValueError("Transformation matrix L must be square")
        if L.shape[0] != self.dim:
            raise ValueError(f"Matrix dimension {L.shape[0]} doesn't match zonotope dim {self.dim}")

        I = np.eye(self.dim)
        M1 = I + L
        M2 = I - L

        new_c = 0.5 * M1 @ self.c
        new_V = 0.5 * np.hstack([M1 @ self.V, M2 @ self.c, M2 @ self.V])
        return Zono(new_c, new_V)

    def intersect_half_space(self, H: np.ndarray, g: np.ndarray) -> 'Star':
        """
        Intersect zonotope with half-space H*x <= g.

        Args:
            H: Half-space matrix
            g: Half-space vector

        Returns:
            Star object (result is not a zonotope in general)
        """
        # Convert to Star and use Star's intersection method
        star = self.to_star()
        return star.intersect_half_space(H, g)

    # ======================== Order Reduction ========================

    def order_reduction_box(self, n_max: int) -> 'Zono':
        """
        Reduce zonotope order (number of generators).

        Based on: Girard (HSCC 2008), Althoff (CDC 2017), Combastel (ECC 2003)

        Args:
            n_max: Maximum number of generators to keep

        Returns:
            New Zono with at most n_max generators
        """
        if n_max < self.dim:
            raise ValueError(f"n_max ({n_max}) must be >= dimension ({self.dim})")

        n_gens = self.V.shape[1]
        if n_gens <= n_max:
            return Zono(self.c, self.V)

        # Compute 2-norm of each generator
        gen_norms = np.linalg.norm(self.V, axis=0)

        # Sort generators by norm (descending)
        sorted_indices = np.argsort(-gen_norms)

        # Keep n_max largest generators
        kept_indices = sorted_indices[:n_max]
        removed_indices = sorted_indices[n_max:]

        V_kept = self.V[:, kept_indices]

        # Over-approximate removed generators with interval hull
        V_removed = self.V[:, removed_indices]
        interval_hull = np.sum(np.abs(V_removed), axis=1, keepdims=True)
        V_hull = np.diag(interval_hull.flatten())

        # Remove zero columns from hull
        non_zero = np.any(V_hull != 0, axis=0)
        V_hull = V_hull[:, non_zero]

        # Combine kept generators and hull
        new_V = np.hstack([V_kept, V_hull])
        return Zono(self.c, new_V)

    def reduce_order(self, target_order: int) -> 'Zono':
        """
        Reduce zonotope to target order (generators per dimension).

        Order = n_generators / dim. Target order of k means at most k*dim generators.

        The reduction keeps the largest generators and over-approximates the
        removed ones with an interval hull. Since the hull can add up to dim
        generators, we keep (target_order - 1) * dim generators to ensure the
        final result has at most target_order * dim generators.

        Args:
            target_order: Target order (max generators = target_order * dim)

        Returns:
            New Zono with at most target_order * dim generators
        """
        if target_order < 1:
            raise ValueError(f"target_order must be >= 1, got {target_order}")

        # Current order
        current_order = self.V.shape[1] / self.dim if self.dim > 0 else 0
        if current_order <= target_order:
            return Zono(self.c.copy(), self.V.copy())

        # Keep (target_order - 1) * dim generators, hull adds up to dim more
        # This ensures final result has at most target_order * dim generators
        n_keep = max(self.dim, (target_order - 1) * self.dim)
        return self.order_reduction_box(n_keep)

    # ======================== Conversion Methods ========================

    def to_star(self) -> 'Star':
        """
        Convert zonotope to Star set.

        Returns:
            Star object with constraints -1 <= alpha_i <= 1
        """
        from .star import Star

        n_gens = self.V.shape[1]

        # V matrix: [c, v1, v2, ..., vn]
        V_star = np.hstack([self.c, self.V])

        # Constraints: -1 <= alpha_i <= 1 for all generators
        # This gives: -alpha_i <= 1 and alpha_i <= 1
        C = np.vstack([np.eye(n_gens), -np.eye(n_gens)])
        d = np.ones((2 * n_gens, 1))

        pred_lb = -np.ones((n_gens, 1))
        pred_ub = np.ones((n_gens, 1))

        return Star(V_star, C, d, pred_lb, pred_ub)

    def to_image_zono(self, height: int, width: int, num_channels: int) -> 'ImageZono':
        """
        Convert zonotope to ImageZono format.

        Args:
            height: Image height
            width: Image width
            num_channels: Number of channels

        Returns:
            ImageZono object
        """
        from .image_zono import ImageZono

        if height * width * num_channels != self.dim:
            raise ValueError(
                f"Image dimensions {height}x{width}x{num_channels} = "
                f"{height * width * num_channels} don't match zonotope dim {self.dim}"
            )

        return ImageZono(self.c, self.V, height, width, num_channels)

    def to_image_star(self, height: int, width: int, num_channels: int) -> 'ImageStar':
        """Convert zonotope to ImageStar format."""
        star = self.to_star()
        return star.to_image_star(height, width, num_channels)

    # ======================== Bounds and Range Methods ========================

    def get_box(self) -> 'Box':
        """
        Get axis-aligned bounding box.

        Returns:
            Box object
        """
        from .box import Box

        # For each dimension: lb[i] = c[i] - ||V[i,:]||_1
        #                     ub[i] = c[i] + ||V[i,:]||_1
        radius = np.sum(np.abs(self.V), axis=1, keepdims=True)
        lb = self.c - radius
        ub = self.c + radius

        return Box(lb, ub)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounds of zonotope.

        For Z = {c + V*alpha | alpha in [-1, 1]^m}, the bounds are:
        lb[i] = c[i] - sum_j |V[i,j]|
        ub[i] = c[i] + sum_j |V[i,j]|

        Returns:
            Tuple of (lb, ub) arrays
        """
        # Sum absolute values of generators
        sum_abs_generators = np.sum(np.abs(self.V), axis=1, keepdims=True)

        lb = self.c - sum_abs_generators
        ub = self.c + sum_abs_generators

        return lb, ub

    def get_ranges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get ranges (same as get_box bounds)."""
        box = self.get_box()
        return box.lb, box.ub

    def estimate_ranges(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate lower and upper bounds for all dimensions.

        For zonotopes, this is exact (not an estimate) since bounds
        can be computed analytically from the generators.

        Returns:
            Tuple of (lb, ub) arrays
        """
        return self.get_bounds()

    def get_range(self, index: int) -> Tuple[float, float]:
        """
        Get range at specific dimension.

        Args:
            index: Dimension index (0-based)

        Returns:
            Tuple of (lb, ub) for that dimension
        """
        if index < 0 or index >= self.dim:
            raise ValueError(f"Invalid index {index}, dimension is {self.dim}")

        radius = np.sum(np.abs(self.V[index, :]))
        lb = self.c[index, 0] - radius
        ub = self.c[index, 0] + radius

        return lb, ub

    def contains(self, x: np.ndarray) -> bool:
        """
        Check if zonotope contains a point.

        Args:
            x: Point vector (n,) or (n, 1)

        Returns:
            True if x is in the zonotope
        """
        x = np.asarray(x).reshape(-1, 1)

        if x.shape[0] != self.dim:
            raise ValueError(f"Point dimension {x.shape[0]} doesn't match zonotope dim {self.dim}")

        d = x - self.c
        d1 = np.sum(np.abs(self.V), axis=1, keepdims=True)

        return np.all(np.abs(d) <= d1)

    def get_oriented_box(self) -> 'Zono':
        """
        Get oriented rectangular hull using SVD.

        Based on MATTISE (Prof. Girard, 2005).

        Returns:
            Zono object representing oriented box
        """
        if self.V.shape[1] == 0:
            return Zono(self.c, self.V)

        # SVD decomposition
        Q, _, _ = svd(self.V, full_matrices=False)

        # Project generators onto principal directions
        P = Q.T @ self.V

        # Compute interval hull in rotated space
        D = np.diag(np.sum(np.abs(P), axis=1))

        # Rotate back
        new_V = Q @ D
        return Zono(self.c, new_V)

    def get_interval_hull(self) -> 'Zono':
        """
        Get interval hull (axis-aligned zonotope).

        Returns:
            Zono with dim generators (axis-aligned)
        """
        box = self.get_box()
        return box.to_zono()

    def get_vertices(self) -> np.ndarray:
        """
        Get all vertices of the zonotope.

        Warning: Exponential in number of generators (2^n vertices).

        Returns:
            Array where each column is a vertex
        """
        n_gens = self.V.shape[1]
        n_vertices = 2 ** n_gens

        if n_gens > 20:
            raise ValueError(
                f"Too many generators ({n_gens}) for vertex enumeration. "
                "This would create 2^{n_gens} = {n_vertices} vertices."
            )

        vertices = np.zeros((self.dim, n_vertices))

        for i in range(n_vertices):
            # Convert i to binary: 0 -> -1, 1 -> +1
            binary = np.array([(i >> j) & 1 for j in range(n_gens)])
            alpha = (2 * binary - 1).reshape(-1, 1)  # Map {0, 1} to {-1, +1}, column vector

            # Compute vertex: c is (dim, 1), V @ alpha is (dim, 1)
            vertices[:, i] = (self.c + self.V @ alpha).flatten()

        return vertices

    # ======================== Utility Methods ========================

    def change_vars_precision(self, precision: str) -> 'Zono':
        """
        Change numerical precision.

        Args:
            precision: 'float32' or 'float64'

        Returns:
            New Zono with converted precision
        """
        if precision == 'float32':
            dtype = np.float32
        elif precision == 'float64':
            dtype = np.float64
        else:
            raise ValueError(f"Unknown precision: {precision}")

        new_c = self.c.astype(dtype)
        new_V = self.V.astype(dtype)
        return Zono(new_c, new_V)

    # ======================== Reachability Analysis ========================
    # Note: Reachability analysis should be performed through NeuralNetwork.reach()
    # instead of calling reach() on set objects directly. This maintains proper
    # separation of concerns where sets represent geometric objects and reachability
    # is a neural network operation.
    #
    # Example usage:
    #     from n2v.nn import NeuralNetwork
    #     from n2v.sets import Zono
    #     import torch.nn as nn
    #
    #     model = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 1))
    #     net = NeuralNetwork(model)
    #     input_zono = Zono.from_bounds(lb, ub)
    #     output_zonos = net.reach(input_zono, method='approx')

"""
Box (Hyper-rectangle) set representation.

Represents an axis-aligned bounding box defined by lower and upper bounds.
Translated from MATLAB NNV Box.m
"""

import numpy as np
from typing import Optional, Tuple, List, Union


class Box:
    """
    Hyper-rectangle class for representing bounded regions.

    A Box is defined by lower and upper bound vectors:
        B = {x | lb <= x <= ub}

    Attributes:
        lb: Lower-bound vector (n,) or (n, 1)
        ub: Upper-bound vector (n,) or (n, 1)
        dim: Dimensionality of the box
        center: Center point of the box
        generators: Generator matrix (zonotope representation)
    """

    def __init__(self, lb: np.ndarray, ub: np.ndarray):
        """
        Initialize a Box from lower and upper bounds.

        Args:
            lb: Lower-bound vector (n,) or (n, 1)
            ub: Upper-bound vector (n,) or (n, 1)

        Raises:
            ValueError: If bounds have inconsistent dimensions
        """
        # Convert to numpy arrays and ensure column vectors
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)

        # Ensure column vectors
        if lb.ndim == 1:
            lb = lb.reshape(-1, 1)
        if ub.ndim == 1:
            ub = ub.reshape(-1, 1)

        # Validate dimensions
        if lb.shape[1] != 1 or ub.shape[1] != 1:
            raise ValueError("Lower and upper bounds must be column vectors")
        if lb.shape[0] != ub.shape[0]:
            raise ValueError("Lower and upper bounds must have same dimension")

        # Validate bounds
        if np.any(lb > ub):
            raise ValueError("Lower bounds must be less than or equal to upper bounds")

        self.lb = lb
        self.ub = ub
        self.dim = lb.shape[0]

        # Compute center
        self.center = 0.5 * (self.lb + self.ub)

        # Compute generators (zonotope representation)
        # Generator matrix: diagonal with half-widths, zero columns removed
        vec = 0.5 * (self.ub - self.lb)

        try:
            # Fast method: create diagonal matrix and remove zero columns
            G = np.diag(vec.flatten())
            non_zero_cols = np.any(G != 0, axis=0)
            self.generators = G[:, non_zero_cols]
        except MemoryError:
            # Memory-efficient method for large sparse perturbations
            non_zero_indices = np.where(vec.flatten() != 0)[0]
            self.generators = np.zeros((self.dim, len(non_zero_indices)))
            for i, idx in enumerate(non_zero_indices):
                self.generators[idx, i] = vec[idx]

    def __repr__(self) -> str:
        return f"Box(dim={self.dim}, lb_range=[{self.lb.min():.3f}, {self.lb.max():.3f}], " \
               f"ub_range=[{self.ub.min():.3f}, {self.ub.max():.3f}])"

    # ======================== Partitioning Methods ========================

    def single_partition(self, part_id: int, part_num: int) -> List['Box']:
        """
        Partition a single dimension into smaller boxes.

        Args:
            part_id: Index of dimension to partition (0-based)
            part_num: Number of partitions

        Returns:
            List of Box objects
        """
        if part_id < 0 or part_id >= self.dim:
            raise ValueError(f"Invalid partition index: {part_id}")
        if part_num < 1:
            raise ValueError("Number of partitions must be >= 1")

        if part_num == 1:
            return [self]

        boxes = []
        interval = (self.ub[part_id] - self.lb[part_id]) / part_num

        for i in range(part_num):
            new_lb = self.lb.copy()
            new_ub = self.ub.copy()
            new_lb[part_id] = self.lb[part_id] + i * interval
            new_ub[part_id] = self.lb[part_id] + (i + 1) * interval
            boxes.append(Box(new_lb, new_ub))

        return boxes

    def partition(self, part_indexes: List[int], part_numbers: List[int]) -> List['Box']:
        """
        Partition multiple dimensions of the box.

        Args:
            part_indexes: Indices of dimensions to partition
            part_numbers: Number of partitions for each dimension

        Returns:
            List of Box objects (product of all partitions)
        """
        if len(part_indexes) != len(part_numbers):
            raise ValueError("part_indexes and part_numbers must have same length")

        boxes = [self]
        for idx, num in zip(part_indexes, part_numbers):
            new_boxes = []
            for box in boxes:
                new_boxes.extend(box.single_partition(idx, num))
            boxes = new_boxes

        return boxes

    # ======================== Transformation Methods ========================

    def affine_map(self, W: np.ndarray, b: Optional[np.ndarray] = None) -> 'Box':
        """
        Apply affine transformation to the box: W*x + b.

        Args:
            W: Mapping matrix (m, n)
            b: Mapping vector (m,) or (m, 1), optional

        Returns:
            New Box object
        """
        W = np.asarray(W)

        # Transform center
        new_center = W @ self.center
        if b is not None:
            b = np.asarray(b).reshape(-1, 1)
            new_center = new_center + b

        # Transform generators
        new_generators = W @ self.generators

        # Compute new bounds using L1 norm
        new_dim = new_center.shape[0]
        new_lb = np.zeros((new_dim, 1))
        new_ub = np.zeros((new_dim, 1))

        for i in range(new_dim):
            radius = np.sum(np.abs(new_generators[i, :]))
            new_lb[i] = new_center[i] - radius
            new_ub[i] = new_center[i] + radius

        return Box(new_lb, new_ub)

    # ======================== Conversion Methods ========================

    def to_zono(self) -> 'Zono':
        """Convert Box to Zonotope representation."""
        from .zono import Zono
        return Zono(self.center, self.generators)

    def to_star(self) -> 'Star':
        """Convert Box to Star set representation."""
        return self.to_zono().to_star()

    # ======================== Sampling Methods ========================

    def sample(self, N: int) -> np.ndarray:
        """
        Generate random samples within the box.

        Args:
            N: Number of samples

        Returns:
            Array of shape (dim, N) with sampled points
        """
        samples = (self.ub - self.lb) * np.random.rand(self.dim, N) + self.lb
        return samples

    # ======================== Getter Methods ========================

    def get_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get lower and upper bounds."""
        return self.lb, self.ub

    def get_vertices(self) -> np.ndarray:
        """
        Get all vertices of the box.

        Returns:
            Array where each column is a vertex (dim, 2^dim)
        """
        # Generate all 2^dim vertices using binary enumeration
        n_vertices = 2 ** self.dim
        vertices = np.zeros((self.dim, n_vertices))

        for i in range(n_vertices):
            # Convert i to binary and use as selector for lb/ub
            binary = [(i >> j) & 1 for j in range(self.dim)]
            for j in range(self.dim):
                vertices[j, i] = self.ub[j] if binary[j] else self.lb[j]

        # Remove duplicate vertices
        vertices = np.unique(vertices, axis=1)
        return vertices

    # ======================== Static Methods ========================

    @staticmethod
    def box_hull(boxes: List['Box']) -> 'Box':
        """
        Compute bounding box that contains all input boxes.

        Args:
            boxes: List of Box objects

        Returns:
            Single Box containing all input boxes
        """
        if not boxes:
            raise ValueError("Cannot compute hull of empty box list")

        all_lb = np.hstack([box.lb for box in boxes])
        all_ub = np.hstack([box.ub for box in boxes])

        hull_lb = np.min(all_lb, axis=1, keepdims=True)
        hull_ub = np.max(all_ub, axis=1, keepdims=True)

        return Box(hull_lb, hull_ub)

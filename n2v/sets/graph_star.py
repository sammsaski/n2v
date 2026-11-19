"""
GraphStar set representation.

A GraphStar represents a set of graph node-feature matrices X of shape (N, F)
under affine perturbations:

    X = V[..., 0] + sum_i alpha_i * V[..., i+1],   subject to C * alpha <= d

where V has shape (N, F, n_basis+1), the predicate alpha is shared across
nodes and features, and the constraint structure mirrors the Star set.

Translated from NNV's MATLAB GraphStar.m.
"""

from typing import Optional, Tuple

import numpy as np

from n2v.sets.star import Star


class GraphStar:
    """
    GraphStar set: a Star whose state is an N x F node-feature matrix.

    Attributes:
        V: Basis tensor of shape (N, F, nVar+1). V[:, :, 0] is the center;
           V[:, :, k+1] is the k-th basis matrix.
        C: Predicate constraint matrix (nConstr, nVar).
        d: Predicate constraint vector (nConstr, 1).
        predicate_lb / predicate_ub: per-predicate bounds (nVar, 1).
        N: Number of nodes.
        F: Number of node-feature channels.
        nVar: Number of predicate variables.
        adjacency: Optional adjacency matrix (N, N) for the underlying graph.
            Stored on the set so per-layer reach ops do not need it as a
            separate argument.
        edge_basis: Optional edge-feature basis tensor of shape (E, D, nVar+1)
            describing edge-feature perturbations sharing the same predicate
            alpha. None when only node features are perturbed.
        edge_index: Optional (2, E) array describing edges if edge features
            are present.
    """

    def __init__(
        self,
        V: Optional[np.ndarray] = None,
        C: Optional[np.ndarray] = None,
        d: Optional[np.ndarray] = None,
        pred_lb: Optional[np.ndarray] = None,
        pred_ub: Optional[np.ndarray] = None,
        adjacency: Optional[np.ndarray] = None,
        edge_basis: Optional[np.ndarray] = None,
        edge_index: Optional[np.ndarray] = None,
    ):
        if V is None:
            self.V = np.array([]).reshape(0, 0, 0)
            self.C = np.array([]).reshape(0, 0)
            self.d = np.array([]).reshape(0, 1)
            self.N = 0
            self.F = 0
            self.nVar = 0
            self.predicate_lb = None
            self.predicate_ub = None
            self.adjacency = None
            self.edge_basis = None
            self.edge_index = None
            return

        V = np.asarray(V, dtype=np.float64)
        if V.ndim != 3:
            raise ValueError(f"GraphStar V must be 3D (N, F, nVar+1); got shape {V.shape}")

        N, F, B = V.shape
        nVar = B - 1
        if nVar < 0:
            raise ValueError("GraphStar V must have at least a center column on its last axis")

        if C is None:
            C = np.zeros((0, nVar))
        else:
            C = np.asarray(C, dtype=np.float64)
        if d is None:
            d = np.zeros((0, 1))
        else:
            d = np.asarray(d, dtype=np.float64).reshape(-1, 1)

        if C.size > 0 and C.shape[1] != nVar:
            raise ValueError(
                f"Constraint matrix has {C.shape[1]} cols; expected nVar={nVar}"
            )
        if C.shape[0] != d.shape[0]:
            raise ValueError(
                f"Constraint matrix rows {C.shape[0]} != constraint vector rows {d.shape[0]}"
            )

        if pred_lb is not None:
            pred_lb = np.asarray(pred_lb, dtype=np.float64).reshape(-1, 1)
            if pred_lb.shape[0] != nVar:
                raise ValueError(f"pred_lb size {pred_lb.shape[0]} != nVar {nVar}")
        if pred_ub is not None:
            pred_ub = np.asarray(pred_ub, dtype=np.float64).reshape(-1, 1)
            if pred_ub.shape[0] != nVar:
                raise ValueError(f"pred_ub size {pred_ub.shape[0]} != nVar {nVar}")

        if adjacency is not None:
            adjacency = np.asarray(adjacency, dtype=np.float64)
            if adjacency.shape != (N, N):
                raise ValueError(
                    f"Adjacency shape {adjacency.shape} != (N, N) = ({N}, {N})"
                )

        if edge_basis is not None:
            edge_basis = np.asarray(edge_basis, dtype=np.float64)
            if edge_basis.ndim != 3 or edge_basis.shape[2] != B:
                raise ValueError(
                    f"edge_basis must have shape (E, D, nVar+1={B}); got {edge_basis.shape}"
                )
            if edge_index is None:
                raise ValueError("edge_basis provided without edge_index")
            edge_index = np.asarray(edge_index, dtype=np.int64)
            if edge_index.shape != (2, edge_basis.shape[0]):
                raise ValueError(
                    f"edge_index shape {edge_index.shape} != (2, E={edge_basis.shape[0]})"
                )

        self.V = V
        self.C = C
        self.d = d
        self.N = N
        self.F = F
        self.nVar = nVar
        self.predicate_lb = pred_lb
        self.predicate_ub = pred_ub
        self.adjacency = adjacency
        self.edge_basis = edge_basis
        self.edge_index = edge_index

    def __repr__(self) -> str:
        return (
            f"GraphStar(N={self.N}, F={self.F}, nVar={self.nVar}, "
            f"nConstraints={self.C.shape[0]}, edges={'yes' if self.edge_basis is not None else 'no'})"
        )

    @property
    def dim(self) -> int:
        return self.N * self.F

    # ============================================================== Constructors

    @classmethod
    def from_bounds(
        cls,
        lb: np.ndarray,
        ub: np.ndarray,
        adjacency: Optional[np.ndarray] = None,
    ) -> "GraphStar":
        """
        Create a GraphStar from element-wise (N, F) bounds on node features.

        One predicate variable is created per (node, feature) entry whose
        lb < ub.  Equal bounds are baked into the center.
        """
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)
        if lb.shape != ub.shape or lb.ndim != 2:
            raise ValueError(
                f"lb/ub must be 2D arrays of shape (N, F); got {lb.shape}, {ub.shape}"
            )
        if np.any(ub < lb):
            raise ValueError("Upper bound below lower bound on at least one entry")

        N, F = lb.shape
        center = 0.5 * (lb + ub)
        half = 0.5 * (ub - lb)

        active = half > 0
        idx = np.argwhere(active)
        nVar = idx.shape[0]

        V = np.zeros((N, F, nVar + 1))
        V[..., 0] = center
        for k, (n, f) in enumerate(idx):
            V[n, f, k + 1] = half[n, f]

        C = np.vstack([np.eye(nVar), -np.eye(nVar)]) if nVar > 0 else np.zeros((0, 0))
        d = np.ones((2 * nVar, 1)) if nVar > 0 else np.zeros((0, 1))
        pred_lb = -np.ones((nVar, 1)) if nVar > 0 else None
        pred_ub = np.ones((nVar, 1)) if nVar > 0 else None

        return cls(V, C, d, pred_lb, pred_ub, adjacency=adjacency)

    @classmethod
    def from_star(cls, star: Star, N: int, F: int, adjacency: Optional[np.ndarray] = None) -> "GraphStar":
        """Reshape a Star of dim N*F back into a GraphStar of shape (N, F)."""
        if star.dim != N * F:
            raise ValueError(f"Star dim {star.dim} != N*F = {N * F}")
        V = star.V.reshape(N, F, star.V.shape[1])
        return cls(
            V,
            star.C if star.C.size else None,
            star.d if star.d.size else None,
            star.predicate_lb,
            star.predicate_ub,
            adjacency=adjacency,
        )

    # ====================================================================== Views

    def to_star(self) -> Star:
        """Flatten the (N, F) state into a Star of dim N*F.

        Row-major flatten: index = n * F + f.
        """
        flat_V = self.V.reshape(self.N * self.F, self.nVar + 1)
        return Star(
            flat_V,
            self.C if self.C.size else None,
            self.d if self.d.size else None,
            self.predicate_lb,
            self.predicate_ub,
        )

    flatten = to_star  # alias

    def with_state(self, new_V: np.ndarray) -> "GraphStar":
        """Return a copy with replaced basis tensor; constraints unchanged."""
        return GraphStar(
            new_V,
            self.C if self.C.size else None,
            self.d if self.d.size else None,
            self.predicate_lb,
            self.predicate_ub,
            adjacency=self.adjacency,
            edge_basis=self.edge_basis,
            edge_index=self.edge_index,
        )

    # ============================================================ Affine maps

    def affine_map_right(self, W: np.ndarray, b: Optional[np.ndarray] = None) -> "GraphStar":
        """Apply X -> X @ W + b broadcast across nodes (per-node feature map).

        W: (F, K).  b: (K,) or (1, K), broadcast to all nodes.
        """
        W = np.asarray(W, dtype=np.float64)
        if W.ndim != 2 or W.shape[0] != self.F:
            raise ValueError(f"W shape {W.shape} incompatible with F={self.F}")

        # V is (N, F, B).  Move basis to second axis to use matmul over (F, K).
        new_V = (self.V.transpose(0, 2, 1) @ W).transpose(0, 2, 1)  # (N, K, B)

        if b is not None:
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            if b.shape[0] != W.shape[1]:
                raise ValueError(f"Bias size {b.shape[0]} != output features {W.shape[1]}")
            new_V[..., 0] = new_V[..., 0] + b[np.newaxis, :]

        return self.with_state(new_V)

    def affine_map_left(self, A: np.ndarray) -> "GraphStar":
        """Apply X -> A @ X (graph aggregation; e.g. normalized adjacency)."""
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2 or A.shape[1] != self.N:
            raise ValueError(f"A shape {A.shape} incompatible with N={self.N}")
        # tensordot over the node axis: result shape (M, F, B)
        new_V = np.tensordot(A, self.V, axes=([1], [0]))
        return self.with_state(new_V)

    def add(self, other: np.ndarray) -> "GraphStar":
        """Add a constant (N, F) matrix to the state (modifies center only)."""
        other = np.asarray(other, dtype=np.float64)
        if other.shape != (self.N, self.F):
            raise ValueError(f"add: shape {other.shape} != (N, F) = ({self.N}, {self.F})")
        new_V = self.V.copy()
        new_V[..., 0] = new_V[..., 0] + other
        return self.with_state(new_V)

    def add_set(self, other: "GraphStar") -> "GraphStar":
        """Add another GraphStar that shares this set's predicate variables.

        Both operands must descend from the same input set via predicate-
        preserving affine maps, so they have identical shape, the same number
        of predicates, and the same constraint region (C, d, predicate bounds).
        Their basis tensors are then summed element-wise; the shared
        constraints are carried through unchanged.
        """
        if not isinstance(other, GraphStar):
            raise TypeError(f"add_set expects GraphStar, got {type(other).__name__}")
        if self.V.shape != other.V.shape:
            raise ValueError(
                f"add_set: basis shapes differ {self.V.shape} vs {other.V.shape}"
            )
        if self.nVar != other.nVar:
            raise ValueError(f"add_set: nVar differs {self.nVar} vs {other.nVar}")
        return self.with_state(self.V + other.V)

    # ============================================================ Subgraph slice

    def extract_subgraph(
        self,
        node_indices: np.ndarray,
        sub_adjacency: Optional[np.ndarray] = None,
    ) -> "GraphStar":
        """Slice the GraphStar to a node subset, pruning dead predicates.

        Keeps only the rows of the basis tensor for ``node_indices`` and drops
        predicate variables whose generators are all-zero over that subset, so
        downstream LP sizes scale with the subgraph rather than the full graph.
        Used by k-hop subgraph verification (:meth:`GraphNeuralNetwork.reach_subgraph`).

        Translated from NNV's GraphStar.extractSubgraph.m.

        Args:
            node_indices: 0-indexed node indices to keep (order is preserved).
            sub_adjacency: Optional adjacency to attach to the sub-GraphStar
                (e.g. the A_norm submatrix for GCN/SAGE).

        Returns:
            A GraphStar over the selected nodes with pruned predicates.
        """
        node_indices = np.asarray(node_indices, dtype=np.int64)
        sub_V = self.V[node_indices, :, :]                 # (n_sub, F, nVar+1)

        if self.nVar == 0:
            return GraphStar(sub_V, adjacency=sub_adjacency)

        gens = sub_V[:, :, 1:].reshape(-1, self.nVar)      # (n_sub*F, nVar)
        active_idx = np.flatnonzero(np.any(gens != 0.0, axis=0))

        if active_idx.size == 0:
            # No live predicate over this subset -> point set (center only).
            return GraphStar(sub_V[:, :, :1], adjacency=sub_adjacency)

        keep_cols = np.concatenate([[0], 1 + active_idx])
        sub_V_pruned = sub_V[:, :, keep_cols]

        if self.C.size:
            # Keep a constraint row ONLY if it touches no pruned predicate.
            # Dropping a row that couples a kept predicate to a pruned one
            # relaxes the predicate region (a sound over-approximation).
            # Deleting just the pruned column while keeping the row would
            # silently pin that predicate to 0, which can shrink the set below
            # the true reachable set (an unsoundness).  For box inputs each row
            # references a single predicate, so this drops exactly the pruned
            # predicates' bound rows and is behaviourally identical.
            pruned_mask = np.ones(self.nVar, dtype=bool)
            pruned_mask[active_idx] = False
            references_active = np.any(self.C[:, active_idx] != 0.0, axis=1)
            references_pruned = np.any(self.C[:, pruned_mask] != 0.0, axis=1)
            keep_rows = references_active & ~references_pruned
            sub_C = self.C[np.ix_(keep_rows, active_idx)]
            sub_d = self.d[keep_rows]
        else:
            sub_C, sub_d = None, None

        sub_lb = self.predicate_lb[active_idx]
        sub_ub = self.predicate_ub[active_idx]
        return GraphStar(
            sub_V_pruned,
            sub_C if (sub_C is not None and sub_C.size) else None,
            sub_d if (sub_d is not None and sub_d.size) else None,
            sub_lb, sub_ub, adjacency=sub_adjacency,
        )

    # =========================================================== Range queries

    def get_ranges(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute element-wise (N, F) lower and upper bounds via Star LP."""
        lb_flat, ub_flat = self.to_star().get_ranges(**kwargs)
        return lb_flat.reshape(self.N, self.F), ub_flat.reshape(self.N, self.F)

    def estimate_ranges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cheap interval bound from per-predicate box without solving LPs."""
        if self.nVar == 0:
            X = self.V[..., 0]
            return X.copy(), X.copy()

        if self.predicate_lb is None or self.predicate_ub is None:
            raise RuntimeError("estimate_ranges requires predicate_lb/predicate_ub")

        lb_a = self.predicate_lb.reshape(-1)
        ub_a = self.predicate_ub.reshape(-1)
        basis = self.V[..., 1:]                        # (N, F, nVar)
        pos = np.maximum(basis, 0.0)
        neg = np.minimum(basis, 0.0)
        lb = self.V[..., 0] + pos @ lb_a + neg @ ub_a
        ub = self.V[..., 0] + pos @ ub_a + neg @ lb_a
        return lb, ub

    # =================================================================== Membership

    def center(self) -> np.ndarray:
        """Return the center matrix (N, F)."""
        return self.V[..., 0].copy()

    def evaluate(self, alpha: np.ndarray) -> np.ndarray:
        """Materialize the (N, F) state for a given predicate vector."""
        alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
        if alpha.shape[0] != self.nVar:
            raise ValueError(f"alpha size {alpha.shape[0]} != nVar {self.nVar}")
        return self.V[..., 0] + self.V[..., 1:] @ alpha

    def contains(self, X: np.ndarray, **kwargs) -> bool:
        """Check whether an (N, F) matrix is in the GraphStar."""
        X = np.asarray(X, dtype=np.float64)
        if X.shape != (self.N, self.F):
            raise ValueError(f"X shape {X.shape} != (N, F) = ({self.N}, {self.F})")
        return self.to_star().contains(X.reshape(-1, 1), **kwargs)

    def sample(self, N: int) -> np.ndarray:
        """Draw up to N samples; returns an array of shape (k, self.N, self.F)."""
        flat = self.to_star().sample(N)        # (dim, k)
        if flat.size == 0:
            return np.zeros((0, self.N, self.F))
        k = flat.shape[1]
        return flat.T.reshape(k, self.N, self.F)

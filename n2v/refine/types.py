"""
Data types for the star-set branch-and-bound verifiers in ``n2v.refine``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class Status(Enum):
    """Verification verdict. Convention: SAT = unsafe (counterexample), UNSAT = safe."""

    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"


class Phase(Enum):
    """Fixed phase of a split ReLU neuron."""

    INACTIVE = 0  # output forced to 0   (pre-activation <= 0)
    ACTIVE = 1    # output forced to identity (pre-activation >= 0)


@dataclass
class LinearSpec:
    """
    Unsafe region as a conjunction of half-spaces: ``{y : G y <= g}``.

    A point ``y`` is *unsafe* iff every row holds (``G @ y <= g`` elementwise).
    The property is UNSAT (safe) iff no reachable output lies in this region.

    Attributes:
        G: (R, out_dim) constraint matrix.
        g: (R,) constraint vector.
    """

    G: np.ndarray
    g: np.ndarray

    def __post_init__(self) -> None:
        # atleast_2d promotes a 1-D G (a single margin row) to shape (1, out_dim);
        # a 2-D G is left as (R, out_dim). (The earlier reshape mis-read a flat
        # multi-row G as one wide row.)
        self.G = np.atleast_2d(np.asarray(self.G, dtype=np.float64))
        self.g = np.asarray(self.g, dtype=np.float64).flatten()
        if self.G.shape[0] != self.g.shape[0]:
            raise ValueError(f"G has {self.G.shape[0]} rows but g has {self.g.shape[0]}")

    def margins(self, y: np.ndarray) -> np.ndarray:
        """Row margins ``G y - g``; ``y`` is unsafe iff all entries <= 0."""
        return self.G @ np.asarray(y, dtype=np.float64).flatten() - self.g

    def is_unsafe(self, y: np.ndarray, tol: float = 0.0) -> bool:
        """True iff ``y`` lies in the closed unsafe region (``tol`` > 0 widens
        it; the counterexample contract is zero tolerance)."""
        return bool(np.all(self.margins(y) <= tol))


@dataclass(frozen=True)
class NeuronKey:
    """Identifies a ReLU neuron by its layer index and position."""

    layer: int
    neuron: int


@dataclass
class NeuronMeta:
    """
    Metadata for one *relaxed* (triangle-approximated) unstable neuron that
    survives to the output star, captured at the moment of relaxation.

    The pre-activation read ``x_hat_j = preact_center + preact_gens . alpha[:k]``
    is over the predicate *prefix* at this neuron's layer (k = len(preact_gens)).
    Because affine layers leave the predicate untouched and ReLU only *appends*
    variables, that prefix is preserved verbatim in every descendant star, so
    the read evaluates correctly against the final witness ``alpha``.

    Attributes:
        key: (layer, neuron) identifier.
        pred_col: index of this neuron's fresh predicate variable in the FINAL
            star predicate (0-based into ``alpha``; column ``pred_col + 1`` of V).
        preact_center: constant term of the pre-activation affine read.
        preact_gens: generator row of the pre-activation read over the prefix.
        l, u: estimated pre-activation bounds at relaxation time (l < 0 < u).
    """

    key: NeuronKey
    pred_col: int
    preact_center: float
    preact_gens: np.ndarray
    l: float
    u: float


@dataclass
class Witness:
    """
    A witness from the epigraph violation LP over a feasible set (P or its box).

    ``alpha`` minimizes the worst spec margin; ``t`` is that minimal worst
    margin (``t = min over set of max_r margin_r``). The set intersects the
    unsafe region iff ``t <= 0``.

    Attributes:
        alpha: (nVar,) predicate point.
        t: minimal worst-row margin attained at ``alpha``.
        binding_row: r* = argmin row margin at ``alpha`` (the most-violated spec row).
        mode: "faithful" (minimized over P, incl. C/d) or "box" (box only).
    """

    alpha: np.ndarray
    t: float
    binding_row: int
    mode: str


@dataclass
class RefineResult:
    """Outcome of a refinement search."""

    status: Status
    nodes: int
    max_depth: int
    counterexample_x: Optional[np.ndarray] = None

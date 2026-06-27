"""
Witness-guided star-set refinement (Phase 1, CPU kill experiment).

Public API is intentionally small while Phase 1 is exploratory; see
``.claude/research/29-RefineStar/`` for the design.
"""

from n2v.refine.types import (
    LinearSpec,
    NeuronKey,
    NeuronMeta,
    Phase,
    RefineNode,
    RefineResult,
    Status,
    Witness,
)
from n2v.refine.bab import verify_refine
from n2v.refine.selectors import (
    BoundWidthSelector,
    BoxCornerSelector,
    FaithfulSelector,
    RandomSelector,
    Selector,
)

__all__ = [
    "LinearSpec",
    "NeuronKey",
    "NeuronMeta",
    "Phase",
    "RefineNode",
    "RefineResult",
    "Status",
    "Witness",
    "verify_refine",
    "Selector",
    "FaithfulSelector",
    "BoxCornerSelector",
    "RandomSelector",
    "BoundWidthSelector",
]

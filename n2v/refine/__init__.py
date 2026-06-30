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
from n2v.refine.operations import refine, split, witness_phase
from n2v.refine.bab import verify_refine
from n2v.refine.input_split import verify_refine_input, verify_refine_input_parallel, verify_refine_input_sb
from n2v.refine.input_split_gpu import verify_refine_input_gpu
from n2v.refine.selectors import (
    BoundWidthSelector,
    BoxCornerSelector,
    ExactStarSBSelector,
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
    "refine",
    "split",
    "witness_phase",
    "verify_refine",
    "verify_refine_input",
    "verify_refine_input_parallel",
    "verify_refine_input_sb",
    "verify_refine_input_gpu",
    "Selector",
    "FaithfulSelector",
    "BoxCornerSelector",
    "ExactStarSBSelector",
    "RandomSelector",
    "BoundWidthSelector",
]

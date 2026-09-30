"""Bundled GNN test data: trimmed IEEE power-flow checkpoints and NNV references.

Set ``N2V_GNN_CHECKPOINTS`` to a full training ``outputs/`` directory (same
``<grid>_pf/<arch>_pf_<grid>.mat`` layout) to run the checkpoint tests against
the untrimmed models instead.  See ``make_fixtures.py`` for provenance.
"""

import os
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent


def checkpoint(arch: str, grid: str = "ieee24") -> Path:
    """Path to the ``<arch>_pf_<grid>.mat`` checkpoint (e.g. arch='sage')."""
    root = Path(os.environ.get("N2V_GNN_CHECKPOINTS", FIXTURE_DIR))
    return root / f"{grid}_pf" / f"{arch}_pf_{grid}.mat"


def matlab_reference(arch: str) -> Path:
    """Path to the NNV (MATLAB) IEEE-24 reach reference for ``arch``."""
    return FIXTURE_DIR / "matlab_refs" / f"{arch}_ieee24_reference.mat"

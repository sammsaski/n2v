"""DropPath reachability.

At inference (``model.eval()``) DropPath is identity, so every set type
is passed through unchanged.
"""

from __future__ import annotations

from typing import List


def drop_path_star(_layer, input_sets: List) -> List:
    return input_sets


def drop_path_zono(_layer, input_sets: List) -> List:
    return input_sets


def drop_path_box(_layer, input_sets: List) -> List:
    return input_sets

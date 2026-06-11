"""DagConcat reachability: concatenate two or more input set streams.

Coverage per nnVLA: Box only.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box


def dag_concat_box(input_boxes: List[Box], extras: List[List[Box]]) -> List[Box]:
    out: List[Box] = []
    for i, b in enumerate(input_boxes):
        lbs = [b.lb]
        ubs = [b.ub]
        for stream in extras:
            other = stream[i]
            lbs.append(other.lb)
            ubs.append(other.ub)
        out.append(Box(np.vstack(lbs), np.vstack(ubs)))
    return out

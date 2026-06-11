"""DagAdd reachability: elementwise add of two (or more) input set streams.

Each function consumes ``primary`` plus a list ``extras`` of additional
set streams; all streams must have the same length and dimension.

Coverage per nnVLA: Box only (IBP). Star and Zono would require joint
predicate alignment across streams which is not generally sound without
graph-level dependency tracking; ship Box, leave the higher set types
to be added once n2v has a multi-input dispatcher.
"""

from __future__ import annotations

from typing import List

import numpy as np

from n2v.sets import Box


def dag_add_box(input_boxes: List[Box], extras: List[List[Box]]) -> List[Box]:
    if not extras:
        return input_boxes
    out: List[Box] = []
    for i, b in enumerate(input_boxes):
        lb = b.lb.copy()
        ub = b.ub.copy()
        for stream in extras:
            other = stream[i]
            lb = lb + other.lb
            ub = ub + other.ub
        out.append(Box(lb, ub))
    return out

"""Point-agreement tests: parsed representation vs naive text evaluation.

Default: representative corpus subset (every benchmark's first instance in
both encodings + all regression files), 120 points each. N2V_DIFF_FULL=1
sweeps every corpus spec. Skips without the corpus.

Known, deliberate representation deviation (documented, not a bug):
strict ``<``/``>`` are represented non-strictly — random continuous
sampling hits the measure-zero boundary with probability 0.
"""

import gzip
import os

import numpy as np
import pytest

from n2v.utils import load_vnnlib
from n2v.utils.vnnlib2 import evaluate_nonlinear
from tests.differential import naive_eval, point_agreement
from tests.differential.test_parser_differential import (
    BENCH, _resolve, _subset, needs_corpus,
)
from tests.differential.canon import materialize


def _check_nonlinear(text, parsed, n_points=120, seed=0):
    amap = point_agreement.atom_index_map(text)
    rng = np.random.default_rng(seed)
    lb = np.where(np.isfinite(parsed["lb"]), parsed["lb"], -50.0)
    ub = np.where(np.isfinite(parsed["ub"]), parsed["ub"], 50.0)
    out_dim = max((i for k, i in amap.values() if k == "out"),
                  default=-1) + 1
    spec = naive_eval.NaiveSpec(text)
    disagreements = []
    for _ in range(n_points):
        x = lb + (ub - lb) * rng.random(lb.size)
        if rng.random() < 0.3:
            d = int(rng.integers(lb.size))
            x[d] += rng.normal() * 20
        y = rng.normal(size=out_dim) * rng.choice([1.0, 50.0, 2000.0])
        values = {a: (x[i] if k == "in" else y[i])
                  for a, (k, i) in amap.items()}
        naive = spec.satisfied(values)
        ours = evaluate_nonlinear(parsed, x, y)
        if naive != ours:
            disagreements.append((x.tolist(), y.tolist(), naive, ours))
    return n_points, disagreements


@needs_corpus
def test_point_agreement_corpus():
    checked, skipped, failures = 0, 0, []
    for rel in _subset():
        path = _resolve(rel)
        if path is None:
            continue
        local, cleanup = materialize(path)
        try:
            try:
                parsed = load_vnnlib(local)
            except Exception:  # noqa: BLE001 — rejected specs not in scope
                skipped += 1
                continue
            with open(local) as f:
                text = f.read()
            fmt = parsed.get("format", "pairs")
            if fmt == "relational":
                n, disagreements = point_agreement.check_relational_agreement(
                    text, parsed)
            elif fmt == "nonlinear":
                n, disagreements = _check_nonlinear(text, parsed)
            else:
                n, disagreements = point_agreement.check_file_agreement(
                    text, parsed["pairs"])
            checked += 1
            if disagreements:
                x, y, naive, ours = disagreements[0]
                failures.append(
                    f"{rel}: {len(disagreements)}/{n} points disagree "
                    f"(first: naive={naive} ours={ours})")
        finally:
            cleanup()
    assert checked > 0
    assert not failures, (
        f"representation disagrees with naive evaluation on "
        f"{len(failures)} file(s):\n  " + "\n  ".join(failures))

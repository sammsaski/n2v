"""Golden-snapshot regression gate for the VNNLIB parsers.

The parser's canonical output for every corpus spec was validated once
against independent references (alpha-beta-CROWN + NeuralSAT on 1.0,
3,002/3,002 agreement; the official VNNLIB-Python package on 2.0) and
frozen as hashes in golden_specs.txt. This test detects ANY behavior
change without running third-party code.

A mismatch means one of:
  - an unintended parser regression (fix the parser), or
  - a deliberate parser change / upstream corpus update — re-validate the
    changed files, then bless via tests/differential/update_golden.py and
    review the golden diff.

Default: representative subset. N2V_DIFF_FULL=1: the entire corpus.
"""

import hashlib
import json
import os

import pytest

from tests.differential import canon
from tests.differential.test_parser_differential import (
    _resolve, _subset, needs_corpus,
)

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "golden_specs.txt")


def _load_golden():
    golden = {}
    with open(GOLDEN) as f:
        for line in f:
            rel, h = line.rstrip("\n").split("\t")
            golden[rel] = h
    return golden


@needs_corpus
@pytest.mark.skipif(not os.path.isfile(GOLDEN),
                    reason="golden_specs.txt not generated yet")
def test_parser_matches_golden_snapshot():
    golden = _load_golden()
    mismatches = []
    checked = 0
    for rel in _subset():
        if rel not in golden:
            continue  # new corpus file: flagged by full mode / update flow
        path = _resolve(rel)
        if path is None:
            continue
        try:
            cases = canon.parse_ours(path)
            h = hashlib.sha256(
                json.dumps(cases, sort_keys=True).encode()).hexdigest()
        except Exception:  # noqa: BLE001
            h = "reject"
        checked += 1
        if h != golden[rel]:
            mismatches.append(f"{rel}: {golden[rel][:12]} -> {h[:12]}")
    assert checked > 0
    assert not mismatches, (
        f"parser behavior changed vs golden snapshot on {len(mismatches)} "
        f"file(s) — if deliberate, re-validate and re-bless via "
        f"update_golden.py:\n  " + "\n  ".join(mismatches))

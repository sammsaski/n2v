#!/usr/bin/env python3
"""Regenerate the golden parse snapshot (tests/differential/golden_specs.txt).

One line per corpus spec file:  <relpath>\t<sha256-of-canonical-parse>
Rejected specs record           <relpath>\treject

Run after a DELIBERATE parser-behavior change or a benchmark-corpus
update, after re-validating the changed files (fixtures / reference
parsers / point agreement). Review the diff of golden_specs.txt — every
changed line is a behavior change you are blessing.

Usage: python tests/differential/update_golden.py [corpus_root]
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from tests.differential import canon  # noqa: E402
from tests.differential.test_parser_differential import (  # noqa: E402
    BENCH, _all_specs, _resolve,
)


def canonical_hash(path):
    try:
        cases = canon.parse_ours(path)
    except Exception:  # noqa: BLE001
        return "reject"
    payload = json.dumps(cases, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else BENCH
    assert os.path.isdir(root), f"corpus not found: {root}"
    lines = []
    rels = _all_specs()
    for i, rel in enumerate(rels):
        path = _resolve(rel)
        if path is None:
            lines.append(f"{rel}\tmissing")
            continue
        lines.append(f"{rel}\t{canonical_hash(path)}")
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(rels)}", flush=True)
    out = os.path.join(HERE, "golden_specs.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} lines to {out}")


if __name__ == "__main__":
    main()

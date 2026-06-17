"""Canonicalization + comparison helpers for the parser differential tests.

Every parser under comparison is reduced to the same canonical form::

    [ {"lb": [...], "ub": [...],
       "disjuncts": [ {"rows": [[(col, coeff), ...], ...], "rhs": [...]} ]},
      ... ]

Semantics (identical on all sides): a counterexample exists iff for some
case, an input in [lb, ub] yields an output satisfying some disjunct
(AND of its ``rows . y <= rhs``). Cases with identical boxes are merged
(disjunct union) before comparison so representational differences
(per-disjunct pairs vs merged boxes) don't count as disagreements.
"""

import gzip
import itertools
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np

DISJUNCT_CAP = 20000
RTOL, ATOL = 1e-5, 1e-9


# ---------------------------------------------------------------------------
# canonical form
# ---------------------------------------------------------------------------

def _sparse_row(row):
    return [[int(i), float(v)] for i, v in enumerate(row) if v != 0.0]


def _canon_polytope(mat, rhs):
    rows = [_sparse_row(r) for r in mat]
    order = sorted(range(len(rows)), key=lambda i: (rows[i], float(rhs[i])))
    return {"rows": [rows[i] for i in order],
            "rhs": [float(rhs[i]) for i in order]}


def _canon_cases(cases):
    """cases: list of (lb, ub, polytopes). Merge same-box, sort, dedupe."""
    merged = {}
    for lb, ub, polys in cases:
        key = (tuple(round(float(x), 12) for x in lb),
               tuple(round(float(x), 12) for x in ub))
        d = merged.setdefault(key, {})
        for poly in polys:
            d[json.dumps(poly, sort_keys=True)] = poly
    return [
        {"lb": list(k[0]), "ub": list(k[1]),
         "disjuncts": [d[s] for s in sorted(d)]}
        for k, d in sorted(merged.items())
    ]


def _close(a, b):
    return abs(a - b) <= ATOL + RTOL * max(abs(a), abs(b))


def compare_canonical(ca, cb):
    """Return (agree: bool, reason: str)."""
    if len(ca) != len(cb):
        return False, f"n_cases {len(ca)} vs {len(cb)}"
    for case_a, case_b in zip(ca, cb):
        if len(case_a["lb"]) != len(case_b["lb"]):
            return False, "in_dim"
        for fa, fb in (("lb", "lb"), ("ub", "ub")):
            if not all(_close(x, y)
                       for x, y in zip(case_a[fa], case_b[fb])):
                return False, f"{fa} mismatch"
        da, db = case_a["disjuncts"], case_b["disjuncts"]
        if len(da) != len(db):
            return False, f"n_disjuncts {len(da)} vs {len(db)}"
        for pa, pb in zip(da, db):
            if len(pa["rows"]) != len(pb["rows"]):
                return False, "n_rows"
            for ra, rb, ha, hb in zip(pa["rows"], pb["rows"],
                                      pa["rhs"], pb["rhs"]):
                if [c for c, _ in ra] != [c for c, _ in rb]:
                    return False, "sparsity pattern"
                if not all(_close(va, vb)
                           for (_, va), (_, vb) in zip(ra, rb)):
                    return False, "coefficient"
                if not _close(ha, hb):
                    return False, "rhs"
    return True, ""


# ---------------------------------------------------------------------------
# file handling
# ---------------------------------------------------------------------------

def materialize(path):
    """Return (usable_path, cleanup_fn). Always a temp copy so reference
    parsers that write sidecar caches never touch the corpus."""
    tmpd = tempfile.mkdtemp()
    local = os.path.join(tmpd, "spec.vnnlib")
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as fi, open(local, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    else:
        shutil.copyfile(path, local)
    return local, lambda: shutil.rmtree(tmpd, ignore_errors=True)


# ---------------------------------------------------------------------------
# parser adapters -> canonical
# ---------------------------------------------------------------------------

def parse_ours(path):
    from n2v.utils import load_vnnlib
    local, cleanup = materialize(path)
    try:
        p = load_vnnlib(local)
        fmt = p.get("format")
        if fmt == "relational":
            C = p["input_coupling"]
            return {
                "format": "relational",
                "networks": [(n["name"], n["relation"]) for n in p["networks"]],
                "lb": [float(v) for v in np.asarray(p["lb"]).flatten()],
                "ub": [float(v) for v in np.asarray(p["ub"]).flatten()],
                "coupling": (_canon_polytope(
                    np.asarray(C.G).tolist(),
                    np.asarray(C.g).flatten().tolist())
                    if C is not None else None),
                "prop": [
                    [_canon_polytope(np.asarray(h.G).tolist(),
                                     np.asarray(h.g).flatten().tolist())
                     for h in (g["Hg"] if isinstance(g["Hg"], list)
                               else [g["Hg"]])]
                    for g in p["prop"]
                ],
            }
        if fmt == "nonlinear":
            return {
                "format": "nonlinear",
                "assertions": json.loads(json.dumps(p["assertions"])),
                "lb": [float(v) for v in np.asarray(p["lb"]).flatten()],
                "ub": [float(v) for v in np.asarray(p["ub"]).flatten()],
            }
        cases = []
        for pair in p["pairs"]:
            groups = [(g["Hg"] if isinstance(g["Hg"], list) else [g["Hg"]])
                      for g in pair["prop"]]
            n_combos = 1
            for g in groups:
                n_combos *= max(len(g), 1)
            if n_combos > DISJUNCT_CAP:
                raise RuntimeError("disjunct cap exceeded (skip file)")
            polys = []
            for combo in itertools.product(*groups):
                mat = np.concatenate(
                    [np.asarray(h.G, dtype=np.float64) for h in combo])
                rhs = np.concatenate(
                    [np.asarray(h.g, dtype=np.float64).flatten()
                     for h in combo])
                polys.append(_canon_polytope(mat.tolist(), rhs.tolist()))
            lb = np.asarray(pair["lb"], dtype=np.float64).flatten()
            ub = np.asarray(pair["ub"], dtype=np.float64).flatten()
            cases.append((lb.tolist(), ub.tolist(), polys))
        return _canon_cases(cases)
    finally:
        cleanup()


def _bak_style_to_canonical(rv):
    cases = []
    for box, props in rv:
        lb = [float(b[0]) for b in box]
        ub = [float(b[1]) for b in box]
        polys = [_canon_polytope([list(map(float, row)) for row in mat],
                                 [float(x) for x in rhs])
                 for mat, rhs in props]
        cases.append((lb, ub, polys))
    return _canon_cases(cases)


def parse_abcrown(path, abcrown_dir):
    """alpha-beta-CROWN's read_vnnlib (Bak lineage), config stubbed."""
    stub = types.ModuleType("arguments")
    stub.Config = {"debug": {"rescale_vnnlib_ptb": None}}
    sys.modules["arguments"] = stub
    cv = os.path.join(abcrown_dir, "complete_verifier")
    if cv not in sys.path:
        sys.path.insert(0, cv)
    import read_vnnlib as _abcrown_rv  # noqa: PLC0415
    local, cleanup = materialize(path)
    try:
        return _bak_style_to_canonical(_abcrown_rv.read_vnnlib(local))
    finally:
        cleanup()


def parse_neuralsat(path, neuralsat_dir):
    """NeuralSAT's read_vnnlib (independent Bak descendant)."""
    import logging
    hm = types.ModuleType("helper")
    hmm = types.ModuleType("helper.misc")
    hml = types.ModuleType("helper.misc.logger")
    hml.logger = logging.getLogger("nsat")
    sys.modules.setdefault("helper", hm)
    sys.modules.setdefault("helper.misc", hmm)
    sys.modules.setdefault("helper.misc.logger", hml)
    spec_dir = os.path.join(neuralsat_dir, "src", "helper", "spec")
    if spec_dir not in sys.path:
        sys.path.insert(0, spec_dir)
    import read_vnnlib as _nsat_rv  # noqa: PLC0415
    local, cleanup = materialize(path)
    try:
        return _bak_style_to_canonical(_nsat_rv.read_vnnlib(local))
    finally:
        cleanup()


def parse_official(path):
    """The official VNNLIB-Python package (2.0 only)."""
    import vnnlib as _vnnlib  # noqa: PLC0415
    from vnnlib import compat  # noqa: PLC0415
    local, cleanup = materialize(path)
    try:
        q = _vnnlib.parse_query_file(local)
        spec_cases = compat.transform(q)
        cases = []
        for c in spec_cases:
            box = np.asarray(c.input_box, dtype=np.float64)
            polys = [
                _canon_polytope(
                    np.asarray(p.coeff_matrix, dtype=np.float64).tolist(),
                    np.asarray(p.rhs, dtype=np.float64).flatten().tolist())
                for p in c.output_constraints
            ]
            cases.append((box[:, 0].tolist(), box[:, 1].tolist(), polys))
        return _canon_cases(cases)
    finally:
        cleanup()

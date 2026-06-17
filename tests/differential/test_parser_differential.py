"""Differential tests: n2v's VNNLIB parsers vs reference implementations.

These are environment-gated (they need the VNN-COMP benchmark corpus and,
per reference, an external checkout or package). Missing resources skip,
never fail — so the suite stays green on CI boxes without the corpus.

Resources (env overrides in parentheses):
  - benchmark corpus  (N2V_VNNCOMP_BENCHMARKS)
        default ~/v/other/VNNCOMP/vnncomp2026_benchmarks/benchmarks
  - alpha-beta-CROWN checkout (N2V_ABCROWN)   default ~/v/other/alpha-beta-CROWN
  - NeuralSAT checkout (N2V_NEURALSAT)        default ~/v/other/neuralsat
  - official 'vnnlib' package (pip install vnnlib) for the 2.0 gate

By default a fixed representative subset runs (every benchmark's first
instance in both encodings + every historically-problematic file).
Set N2V_DIFF_FULL=1 to sweep the entire corpus (minutes).
"""

import csv
import os

import pytest

from tests.differential import canon

BENCH = os.environ.get(
    "N2V_VNNCOMP_BENCHMARKS",
    os.path.expanduser("~/v/other/VNNCOMP/vnncomp2026_benchmarks/benchmarks"),
)
ABCROWN = os.environ.get(
    "N2V_ABCROWN", os.path.expanduser("~/v/other/alpha-beta-CROWN"))
NEURALSAT = os.environ.get(
    "N2V_NEURALSAT", os.path.expanduser("~/v/other/neuralsat"))
FULL = os.environ.get("N2V_DIFF_FULL") == "1"

needs_corpus = pytest.mark.skipif(
    not os.path.isdir(BENCH), reason="benchmark corpus not available")

# Reference-parser comparisons are validation tools, not default CI: they
# were used to validate the parser once (and get re-used when the corpus
# changes). Day-to-day regression is covered by test_golden_snapshot.py.
needs_references_optin = pytest.mark.skipif(
    os.environ.get("N2V_DIFF_REFERENCES") != "1",
    reason="reference differentials are opt-in: set N2V_DIFF_REFERENCES=1")

# Files that exposed real parser bugs — always in the subset.
REGRESSION_FILES = [
    "nn4sys/1.0/vnnlib/lindex_1.vnnlib",
    "nn4sys/1.0/vnnlib/lindex_200.vnnlib",
    "nn4sys/1.0/vnnlib/cardinality_0_100_128.vnnlib",
    "test/1.0/vnnlib/test_tiny.vnnlib",
    "test/1.0/vnnlib/test_small.vnnlib",
    "test/2.0/vnnlib/test_tiny.vnnlib",
    "test/2.0/vnnlib/test_small.vnnlib",
    "ml4acopf_2024/1.0/vnnlib/118_ieee_prop3.vnnlib",
    "ml4acopf_2024/1.0/vnnlib/14_ieee_prop1.vnnlib",
    "acasxu_2023/2.0/vnnlib/prop_6.vnnlib",
    "acasxu_2023/1.0/vnnlib/prop_6.vnnlib",
]

# Specs the official package mis-lowers (its confirmed compat.transform
# conjunction bug); ours is correct per the standard — excluded from the
# 2.0 agreement assertion.
KNOWN_OFFICIAL_BUGS = {"acasxu_2023/2.0/vnnlib/prop_6.vnnlib"}


def _resolve(rel):
    p = os.path.join(BENCH, rel)
    if os.path.exists(p):
        return p
    if os.path.exists(p + ".gz"):
        return p + ".gz"
    return None


def _first_instance_specs():
    """First instance's vnnlib of every benchmark/version dir."""
    out = []
    if not os.path.isdir(BENCH):
        return out
    for bench in sorted(os.listdir(BENCH)):
        bdir = os.path.join(BENCH, bench)
        if not os.path.isdir(bdir):
            continue
        for ver in sorted(os.listdir(bdir)):
            csvp = os.path.join(bdir, ver, "instances.csv")
            if not os.path.isfile(csvp):
                continue
            with open(csvp, newline="") as f:
                for row in csv.reader(f):
                    if row and row[0].strip():
                        out.append(f"{bench}/{ver}/"
                                   f"{row[1].strip().lstrip('./')}")
                        break
    return out


def _all_specs(version=None):
    out = set()
    for bench in sorted(os.listdir(BENCH)):
        bdir = os.path.join(BENCH, bench)
        if not os.path.isdir(bdir):
            continue
        for ver in sorted(os.listdir(bdir)):
            if version and ver != version:
                continue
            csvp = os.path.join(bdir, ver, "instances.csv")
            if not os.path.isfile(csvp):
                continue
            with open(csvp, newline="") as f:
                for row in csv.reader(f):
                    if row and row[0].strip():
                        out.add(f"{bench}/{ver}/"
                                f"{row[1].strip().lstrip('./')}")
    return sorted(out)


def _subset(version=None):
    if FULL:
        rels = _all_specs(version)
    else:
        rels = [r for r in _first_instance_specs()
                if version is None or f"/{version}/" in r]
        rels += [r for r in REGRESSION_FILES
                 if version is None or f"/{version}/" in r]
    seen, out = set(), []
    for r in rels:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _run_differential(rels, reference_fn, reference_name,
                      known_disagreements=frozenset()):
    disagreements = []
    compared = 0
    for rel in rels:
        path = _resolve(rel)
        if path is None:
            continue
        try:
            ours = canon.parse_ours(path)
        except Exception:  # noqa: BLE001 - rejected specs aren't compared
            ours = None
        try:
            ref = reference_fn(path)
        except Exception:  # noqa: BLE001
            ref = None
        if ours is None or ref is None:
            continue  # at least one side rejects: not a comparison case
        compared += 1
        agree, why = canon.compare_canonical(ours, ref)
        if not agree and rel not in known_disagreements:
            disagreements.append(f"{rel}: {why}")
    assert compared > 0, "differential compared zero files — check resources"
    assert not disagreements, (
        f"parser disagrees with {reference_name} on "
        f"{len(disagreements)} file(s):\n  " + "\n  ".join(disagreements)
    )


@needs_references_optin
@needs_corpus
@pytest.mark.skipif(
    not os.path.isfile(os.path.join(
        ABCROWN, "complete_verifier", "read_vnnlib.py")),
    reason="alpha-beta-CROWN checkout not available")
def test_v1_matches_abcrown():
    _run_differential(
        _subset("1.0"),
        lambda p: canon.parse_abcrown(p, ABCROWN),
        "alpha-beta-CROWN read_vnnlib",
    )


@needs_references_optin
@needs_corpus
@pytest.mark.skipif(
    not os.path.isfile(os.path.join(
        NEURALSAT, "src", "helper", "spec", "read_vnnlib.py")),
    reason="NeuralSAT checkout not available")
def test_v1_matches_neuralsat():
    pytest.importorskip("beartype")
    _run_differential(
        _subset("1.0"),
        lambda p: canon.parse_neuralsat(p, NEURALSAT),
        "NeuralSAT read_vnnlib",
    )


@needs_references_optin
@needs_corpus
def test_v2_matches_official_vnnlib():
    pytest.importorskip("vnnlib")
    _run_differential(
        _subset("2.0"),
        canon.parse_official,
        "official VNNLIB-Python",
        known_disagreements=KNOWN_OFFICIAL_BUGS,
    )


@needs_corpus
def test_prop6_against_official_bug():
    """Pin OUR prop_6 lowering (the official lib mis-lowers it): the
    conjunction of a 2-box input disjunction and an output assert must
    produce exactly 2 cases, each carrying the output constraints."""
    path = _resolve("acasxu_2023/2.0/vnnlib/prop_6.vnnlib")
    if path is None:
        pytest.skip("prop_6 not present")
    ours = canon.parse_ours(path)
    assert len(ours) == 2
    for case in ours:
        assert len(case["disjuncts"]) == 4  # OR of 4 output halfspaces
        assert all(len(d["rows"]) == 1 for d in case["disjuncts"])

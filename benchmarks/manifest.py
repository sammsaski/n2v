"""Instance lists (manifests) for benchmark runs: seeded selection, read, write.

A manifest is a CSV with columns ``benchmark, onnx, vnnlib, timeout``. ``onnx`` and
``vnnlib`` are copied verbatim from ``<bench-root>/benchmarks/<benchmark>/<version>/
instances.csv``, so they are relative to that folder (a relational row's ``onnx``
is the list literal of the original file). ``timeout`` is the official
per-instance timeout. Lines starting with ``#`` record how the list was made.

Usage:
    python benchmarks/manifest.py --bench-root DIR --per-benchmark 3 --seed 0 --out FILE
"""
import argparse
import csv
import hashlib
import os

import numpy as np

# The 34 voted VNN-COMP 2026 benchmarks, all at VNN-LIB 2.0.
BENCHMARKS_2026 = [
    "acasxu_2023", "adaptive_cruise_control_non_linear_2026", "cctsdb_yolo_2023",
    "cersyve", "cgan2026", "cgan_2023", "challenging_certified_training_2026",
    "cifar100_2024", "collins_aerospace_benchmark", "collins_rul_cnn_2022", "cora_2024",
    "dist_shift_2023", "isomorphic_acasxu_2026", "linearizenn_2024", "lsnc_relu",
    "malbeware", "metaroom_2023", "ml4acopf_2024", "monotonic_acasxu_2026", "nn4sys",
    "relusplitter", "relusplitter_2026", "safenlp_2024", "sat_relu",
    "smart_turn_multimodal_2026", "soundnessbench", "soundnessbench_2026", "test",
    "tinyimagenet_2024", "tllverifybench_2023", "traffic_signs_recognition_2023",
    "vggnet16_2022", "vit_2023", "yolo_2023",
]
COLUMNS = ["benchmark", "onnx", "vnnlib", "timeout"]


def instances_csv(bench_root, benchmark, version="2.0"):
    """Path of a benchmark's instances.csv: the version folder if it has one, else the benchmark folder."""
    versioned = os.path.join(bench_root, "benchmarks", benchmark, version, "instances.csv")
    if os.path.isfile(versioned):
        return versioned
    return os.path.join(bench_root, "benchmarks", benchmark, "instances.csv")


def read_instances(bench_root, benchmark, version="2.0"):
    """The benchmark's rows, in file order, as (onnx, vnnlib, timeout); blank and short rows are skipped."""
    rows = []
    with open(instances_csv(bench_root, benchmark, version), newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 3 and any(c.strip() for c in row):
                rows.append((row[0].strip(), row[1].strip(), float(row[2])))
    return rows


def _rng(seed, benchmark):
    """Generator seeded by (seed, benchmark name) only, so adding a benchmark never changes another's draw."""
    digest = hashlib.sha256(f"{seed}:{benchmark}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def select(bench_root, benchmarks=None, per_benchmark=None, seed=0, version="2.0"):
    """Rows (benchmark, onnx, vnnlib, timeout): `per_benchmark` random instances of each benchmark (all if None), in file order."""
    out = []
    for b in benchmarks or BENCHMARKS_2026:
        rows = read_instances(bench_root, b, version)
        idx = range(len(rows))
        if per_benchmark is not None and per_benchmark < len(rows):
            idx = sorted(_rng(seed, b).choice(len(rows), size=per_benchmark, replace=False))
        out.extend((b, *rows[i]) for i in idx)
    return out


def write_manifest(path, rows, meta):
    """Write `rows` with one `# key: value` line per entry of `meta` above the CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for b, onnx, vnnlib, timeout in rows:
            w.writerow([b, onnx, vnnlib, f"{timeout:g}"])


def read_manifest(path):
    """(meta, rows) of a manifest written by write_manifest."""
    meta, lines = {}, []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#"):
                k, _, v = line[1:].partition(":")
                meta[k.strip()] = v.strip()
            elif line.strip():
                lines.append(line)
    reader = csv.reader(lines)
    if next(reader) != COLUMNS:
        raise ValueError(f"{path}: expected columns {COLUMNS}")
    return meta, [(b, onnx, vnnlib, float(t)) for b, onnx, vnnlib, t in reader]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-root", required=True, help="vnncomp2026_benchmarks checkout")
    ap.add_argument("--benchmarks", nargs="*", help="default: all 34 voted 2026 benchmarks")
    ap.add_argument("--per-benchmark", type=int, help="instances per benchmark (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--version", default="2.0", help="VNN-LIB version folder")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = select(a.bench_root, a.benchmarks, a.per_benchmark, a.seed, a.version)
    meta = {"selection": "seeded per benchmark (benchmarks/manifest.py)", "seed": a.seed,
            "per_benchmark": a.per_benchmark or "all", "vnnlib_version": a.version,
            "benchmarks": len(a.benchmarks or BENCHMARKS_2026)}
    write_manifest(a.out, rows, meta)
    print(f"{len(rows)} instances -> {a.out}")


if __name__ == "__main__":
    main()

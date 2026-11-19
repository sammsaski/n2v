"""Regenerate the bundled GNN test fixtures from full training checkpoints.

The fixtures are IEEE power-flow GNN checkpoints (GCN, SAGE, GINE) trimmed to
the first ``--instances`` test instances, plus NNV (MATLAB) reach references for
the IEEE-24 models.  Weights and graph structure are kept intact; only the
per-instance arrays are cut, which shrinks each file from ~0.3-1.3 MB to
~10-50 KB.

Usage::

    python tests/fixtures/gnn/make_fixtures.py SRC_OUTPUTS [--refs REF_DIR]

``SRC_OUTPUTS`` is the ``outputs/`` directory of the training pipeline that
exported the ``.mat`` checkpoints (``<SRC_OUTPUTS>/ieee24_pf/gcn_pf_ieee24.mat``
etc.).  ``REF_DIR`` holds the NNV reference files produced by
``generate_ieee24_reference.m``; their embedded absolute checkpoint path is
rewritten to the fixture-relative one.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

HERE = Path(__file__).resolve().parent
PER_INSTANCE_KEYS = ("X_test_g", "Y_test_g", "python_predictions")
CHECKPOINTS = [
    f"{grid}_pf/{arch}_pf_{grid}.mat"
    for grid in ("ieee24", "ieee118")
    for arch in ("gcn", "sage", "gine_pretrain")
]
# reference file in REF_DIR -> (fixture name, checkpoint it was computed on)
REFERENCES = {
    "ieee24_pf_reference.mat": ("gcn_ieee24_reference.mat", "ieee24_pf/gcn_pf_ieee24.mat"),
    "sage_ieee24_reference.mat": ("sage_ieee24_reference.mat", "ieee24_pf/sage_pf_ieee24.mat"),
    "gine_pretrain_ieee24_reference.mat": (
        "gine_pretrain_ieee24_reference.mat", "ieee24_pf/gine_pretrain_pf_ieee24.mat"),
}


def _load(path):
    return {k: v for k, v in loadmat(str(path)).items() if not k.startswith("__")}


def trim_checkpoint(src: Path, dst: Path, instances: int) -> None:
    mat = _load(src)
    for key in PER_INSTANCE_KEYS:
        mat[key] = mat[key][:instances]
    dst.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(dst), mat, do_compression=True)


def copy_reference(src: Path, dst: Path, checkpoint: str, instances: int) -> None:
    mat = _load(src)
    used = np.asarray(mat["instances"]).flatten().astype(int)
    if used.max() > instances:
        raise ValueError(f"{src.name} uses instance {used.max()} > --instances {instances}")
    mat["checkpoint"] = checkpoint
    dst.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(dst), mat, do_compression=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src_outputs", type=Path)
    parser.add_argument("--refs", type=Path, default=None)
    parser.add_argument("--instances", type=int, default=5)
    args = parser.parse_args()

    for rel in CHECKPOINTS:
        trim_checkpoint(args.src_outputs / rel, HERE / rel, args.instances)
        print(f"wrote {rel} ({(HERE / rel).stat().st_size // 1024} KB)")
    if args.refs is not None:
        for src_name, (dst_name, checkpoint) in REFERENCES.items():
            dst = HERE / "matlab_refs" / dst_name
            copy_reference(args.refs / src_name, dst, checkpoint, args.instances)
            print(f"wrote matlab_refs/{dst_name} ({dst.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

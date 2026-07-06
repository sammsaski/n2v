"""Deliverable 1 — low-dim conformal robustness certificate, head-to-head with Track A.

For each cached frame and ℓ∞ radius ε we certify the SAME per-image property two
ways on the SAME scalar functional
``g(x) = median_pixels |f(x) - d0| / max(|d0|, scale_floor)``:

  * **n2v conformal** (`conformal_reach` on the scalar g): a probabilistic box
    ``[lb, ub]`` on g with coverage ``1 - ε₁`` (we set ε₁ = τ so both methods
    target ``P[g ≤ δ] ≥ 1 - τ``). The frame is "n2v-robust at δ" iff ``ub ≤ δ``.
  * **Track A** (Garatti-Campi): violation count k over N uniform-ℓ∞ draws of the
    metric, per-image risk ``eps_up = epsilon_upper(N, k, β)``; the frame is
    "τ-robust at δ" iff ``eps_up ≤ τ``. (`median_absrel` reused verbatim.)

Population layer (identical for both): Track A via ``risk.two_level``; n2v via the
same Level-2 binomial ``epsilon_upper(M, #not-robust, β_out)`` — so the two G(ε)
curves differ ONLY in the Level-1 per-frame certificate (conformal vs GC).

Writes one CSV row per (frame, ε) (resumable) and prints a population summary.

    python -m examples.FlowConformal.experiments.exp_depth.run_deliverable1 \
        --model depthanything --dataset nyu --frames 50 --eps 2,4,8
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from n2v.probabilistic import conformal_reach, ConformalReachConfig

from examples.FlowConformal.experiments.exp_depth import _depth_models as B

SEED = 47
DELTAS = (0.02, 0.05, 0.10, 0.20)   # Track A's tolerance grid
TAU = 0.05                          # per-image risk / conformal miscoverage (ε₁)
BETA = 1e-6                         # confidence budget (Track A convention)
_OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _dk(d: float) -> str:
    return str(int(round(d * 100)))


def _fields():
    base = ["model", "dataset", "frame", "eps255", "N", "m", "train",
            "n2v_lb", "n2v_ub", "n2v_coverage", "n2v_confidence",
            "n2v_time_s", "ta_time_s"]
    for d in DELTAS:
        k = _dk(d)
        base += [f"ta_k_{k}", f"ta_epsup_{k}", f"ta_rob_{k}", f"n2v_rob_{k}"]
    base += ["timestamp"]
    return base


def _now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _done_keys(csv_path: Path):
    done = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((int(row["frame"]), int(row["eps255"])))
    return done


def _run_cell(model, dataset, frame_idx, x01, g, eps255, forward_fn, device,
              N, m, train):
    """One (frame, eps) cell -> dict row (both methods)."""
    eps = eps255 / 255.0
    row = {"model": model, "dataset": dataset, "frame": frame_idx,
           "eps255": eps255, "N": N, "m": m, "train": train,
           "timestamp": _now()}

    # ---- n2v conformal certificate on scalar g ----
    box = B.make_input_box(x01, eps)
    t0 = time.time()
    pbox = conformal_reach(
        g, box,
        ConformalReachConfig(surrogate="naive", m=m, training_samples=train,
                             epsilon=TAU, batch_size=256, seed=SEED),
    )
    row["n2v_time_s"] = f"{time.time() - t0:.1f}"
    n2v_ub = float(np.asarray(pbox.ub).reshape(-1)[0])
    n2v_lb = float(np.asarray(pbox.lb).reshape(-1)[0])
    row.update({"n2v_lb": f"{n2v_lb:.5f}", "n2v_ub": f"{n2v_ub:.5f}",
                "n2v_coverage": f"{pbox.coverage:.4f}",
                "n2v_confidence": f"{pbox.confidence:.4f}"})

    # ---- Track A Garatti-Campi certificate ----
    t0 = time.time()
    k, ta_deltas = B.track_a_counts(forward_fn, x01, eps, N, SEED, device)
    row["ta_time_s"] = f"{time.time() - t0:.1f}"
    kmap = {d: k[j] for j, d in enumerate(ta_deltas)}
    for d in DELTAS:
        dk = _dk(d)
        kd = int(kmap[d])
        epsup = B.risk_epsilon_upper(N, kd, BETA)
        row[f"ta_k_{dk}"] = kd
        row[f"ta_epsup_{dk}"] = f"{epsup:.4f}"
        row[f"ta_rob_{dk}"] = int(epsup <= TAU)
        row[f"n2v_rob_{dk}"] = int(n2v_ub <= d)
    return row


def _population_summary(csv_path: Path, N: int):
    """Per (eps, delta): Track A vs n2v G(eps), agreement — from the CSV rows."""
    rows = list(csv.DictReader(open(csv_path)))
    eps_vals = sorted({int(r["eps255"]) for r in rows})
    summary = {}
    for e in eps_vals:
        er = [r for r in rows if int(r["eps255"]) == e]
        M = len(er)
        per_delta = {}
        for d in DELTAS:
            dk = _dk(d)
            ks = [int(r[f"ta_k_{dk}"]) for r in er]
            # Track A two-level (matches Track A's beta bookkeeping).
            ta = B.risk_two_level(ks, N, TAU, BETA / (2 * M), BETA / 2)
            G_ta = 1.0 - ta["frac_nonrobust_up"]
            # n2v: same Level-2 binomial on the per-frame conformal verdicts.
            k2_n = sum(1 for r in er if int(r[f"n2v_rob_{dk}"]) == 0)
            G_n = 1.0 - B.risk_epsilon_upper(M, k2_n, BETA / 2)
            agree = sum(1 for r in er
                        if int(r[f"ta_rob_{dk}"]) == int(r[f"n2v_rob_{dk}"]))
            per_delta[dk] = dict(
                M=M, G_trackA=round(G_ta, 4), G_n2v=round(G_n, 4),
                k2_trackA=int(ta["k2"]), k2_n2v=int(k2_n),
                agree=agree, agree_frac=round(agree / M, 3),
                trackA_robust_frames=M - int(ta["k2"]),
                n2v_robust_frames=M - int(k2_n),
            )
        summary[str(e)] = per_delta
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="depthanything", choices=B.MODEL_KEYS)
    p.add_argument("--dataset", default="nyu", choices=B.DATASETS)
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--eps", default="2,4,8",
                   help="comma-separated ℓ∞ radii over 255")
    p.add_argument("--N", type=int, default=1000, help="Track A perturbations/frame")
    p.add_argument("--m", type=int, default=1000, help="n2v calibration samples")
    p.add_argument("--train", type=int, default=200, help="n2v surrogate-fit samples")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    eps_list = [int(x) for x in args.eps.split(",")]
    out_csv = args.out or (_OUT_DIR /
                           f"d1_{args.model}_{args.dataset}.csv")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[d1] model={args.model} dataset={args.dataset} "
          f"eps={eps_list}/255 N={args.N} m={args.m} -> {out_csv.name}", flush=True)
    forward_fn, res, device = B.load_depth_model(args.model)
    frames = B.load_cached_frames(args.dataset, args.frames, res, device)
    print(f"[d1] loaded {len(frames)} frames at res={res} on {device}", flush=True)

    done = _done_keys(out_csv)
    fields = _fields()
    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    fh = open(out_csv, "a" if file_exists else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if not file_exists:
        writer.writeheader(); fh.flush()

    t_start = time.time()
    n_new = 0
    for frame_idx, x01 in frames:
        pending = [e for e in eps_list if (frame_idx, e) not in done]
        if not pending:
            continue
        torch.manual_seed(SEED); np.random.seed(SEED)
        d0, scale_floor = B.clean_depth(forward_fn, x01)
        g = B.make_scalar_absrel_callable(forward_fn, d0, scale_floor, res, device)
        for e in pending:
            row = _run_cell(args.model, args.dataset, frame_idx, x01, g, e,
                            forward_fn, device, args.N, args.m, args.train)
            writer.writerow(row); fh.flush()
            n_new += 1
            print(f"[d1] frame {frame_idx:3d} eps {e}/255 | "
                  f"n2v_ub={row['n2v_ub']} "
                  f"ta_k(.05)={row[f'ta_k_5']} "
                  f"({n_new} cells, {time.time()-t_start:.0f}s)", flush=True)
    fh.close()

    # ---- population summary ----
    summary = _population_summary(out_csv, args.N)
    sum_path = out_csv.with_suffix(".summary.json")
    json.dump(summary, open(sum_path, "w"), indent=2)
    print(f"\n[d1] population summary (G = certified lower bound on "
          f"τ-robust frame fraction, β={BETA}):", flush=True)
    for e in sorted(summary, key=int):
        print(f"\n  ε={e}/255:", flush=True)
        print(f"    {'δ':>5} {'G_trackA':>9} {'G_n2v':>7} "
              f"{'robФ_TA':>8} {'robФ_n2v':>9} {'agree':>7}", flush=True)
        for dk in (_dk(d) for d in DELTAS):
            r = summary[e][dk]
            print(f"    {int(dk)/100:>5.2f} {r['G_trackA']:>9.3f} "
                  f"{r['G_n2v']:>7.3f} {r['trackA_robust_frames']:>8d} "
                  f"{r['n2v_robust_frames']:>9d} {r['agree_frac']:>6.0%}",
                  flush=True)
    print(f"\n[d1] wrote {out_csv.name} and {sum_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

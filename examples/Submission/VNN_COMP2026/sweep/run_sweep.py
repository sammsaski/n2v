#!/usr/bin/env python3
"""Full VNN-COMP 2026 sweep driver for n2v: bounded-parallel, one-instance-per-
distinct-network, honoring each instance's EXACT timeout from instances.csv.

Each job calls the real competition runner (../vnncomp_runner.py) — the same
runner the official harness uses — so verdicts are faithful. The runner
self-times via SIGALRM; we add the official hard kill at timeout+60 so a hung
instance can't stall the pool. Results are written per-instance (resumable) plus
a master CSV.

NOTE on timing fidelity (bounded-parallel mode): K instances share the machine,
each pinned to N2V_WORKERS=W cores. Runtimes are therefore NOT directly
competition-comparable (the competition runs one instance with all cores) — but
verdicts are. Keep K*W <= physical cores to avoid oversubscription.

The benchmark corpus is located via N2V_VNNCOMP_BENCHMARKS (repo root or its
``benchmarks/`` subdir); falls back to the local checkout path.

Usage:
  run_sweep.py [--mode different|all|first] [--jobs K] [--workers W]
               [--only CAT[,CAT...]] [--exclude CAT[,CAT...]]
               [--out DIR] [--resume] [--dry-run]
"""
import ast
import csv
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(os.path.dirname(_HERE), "vnncomp_runner.py")
PY = sys.executable


def resolve_bench():
    env = os.environ.get("N2V_VNNCOMP_BENCHMARKS")
    if env:
        env = env.rstrip("/")
        return env if os.path.basename(env) == "benchmarks" else os.path.join(env, "benchmarks")
    return "/isis/home/sasakis/v/other/VNNCOMP/vnncomp2026_benchmarks/benchmarks"


BENCH = resolve_bench()
DEFAULT_OUT = os.path.join(_HERE, "results", "sweep")


def pick_version_dir(cat):
    catdir = os.path.join(BENCH, cat)
    for v in ("2.0", "1.0"):
        if os.path.isfile(os.path.join(catdir, v, "instances.csv")):
            return os.path.join(catdir, v)
    if os.path.isfile(os.path.join(catdir, "instances.csv")):
        return catdir
    return None


def read_instances(vdir):
    rows = []
    with open(os.path.join(vdir, "instances.csv")) as f:
        for r in csv.reader(f):
            if not r or all(not c.strip() for c in r):
                continue
            rows.append(tuple(c.strip() for c in r[:3]))
    return rows


def prefixed(vdir, path):
    path = path.strip()
    if path.startswith("["):            # relational list literal: leave as-is
        return path
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(vdir, path))


def stem(field):
    """Filesystem-safe stem for a result filename (handles relational lists)."""
    field = field.strip()
    if field.startswith("["):
        try:
            parts = ast.literal_eval(field)
            return "_".join(os.path.splitext(os.path.basename(str(p)))[0]
                            for _n, p in parts)[:80]
        except Exception:
            return "relational"
    return os.path.splitext(os.path.basename(field))[0]


def select(rows, mode):
    if mode == "all":
        return rows
    if mode == "first":
        return rows[:1]
    # different: first instance per distinct onnx field
    seen, out = set(), []
    for onnx, vnnlib, to in rows:
        if onnx not in seen:
            seen.add(onnx)
            out.append((onnx, vnnlib, to))
    return out


def main():
    args = sys.argv[1:]
    mode, jobs, workers = "different", 6, None
    only = exclude = None
    out_dir, resume, dry = DEFAULT_OUT, False, False
    for i, a in enumerate(args):
        if a == "--mode":
            mode = args[i + 1]
        elif a == "--jobs":
            jobs = int(args[i + 1])
        elif a == "--workers":
            workers = int(args[i + 1])
        elif a == "--only":
            only = set(args[i + 1].split(","))
        elif a == "--exclude":
            exclude = set(args[i + 1].split(","))
        elif a == "--out":
            out_dir = args[i + 1]
        elif a == "--resume":
            resume = True
        elif a == "--dry-run":
            dry = True
    if workers is None:
        workers = max(1, (os.cpu_count() or 8) // jobs)

    cats = sorted(d for d in os.listdir(BENCH) if pick_version_dir(d))
    if only:
        cats = [c for c in cats if c in only]
    if exclude:
        cats = [c for c in cats if c not in exclude]

    res_dir = os.path.join(out_dir, "results")
    log_dir = os.path.join(out_dir, "logs")
    cex_dir = os.path.join(out_dir, "counterexamples")
    for d in (out_dir, res_dir, log_dir, cex_dir):
        os.makedirs(d, exist_ok=True)

    # Build job list.
    jobs_list, budget = [], 0.0
    for cat in cats:
        vdir = pick_version_dir(cat)
        for onnx, vnnlib, to in select(read_instances(vdir), mode):
            try:
                tsec = float(to)
            except ValueError:
                tsec = 0.0
            budget += tsec
            rf = os.path.join(res_dir, cat, f"{stem(onnx)}__{stem(vnnlib)}.txt")
            jobs_list.append({
                "cat": cat, "vdir": vdir,
                "onnx_arg": prefixed(vdir, onnx),
                "vnnlib_arg": prefixed(vdir, vnnlib),
                "onnx_raw": onnx, "vnnlib_raw": vnnlib,
                "timeout": tsec, "result_file": rf,
            })

    print(f"Sweep: mode={mode}  {len(cats)} categories  {len(jobs_list)} instances")
    print(f"  parallel: {jobs} instances x {workers} workers ({jobs*workers} cores; "
          f"machine has {os.cpu_count()})")
    print(f"  timeout budget (worst case, serial-equiv): {budget/3600:.1f} h")
    print(f"  corpus: {BENCH}")
    print(f"  out: {out_dir}\n", flush=True)
    if dry:
        c = Counter(j["cat"] for j in jobs_list)
        for cat in cats:
            print(f"  {cat:<42}{c[cat]:>5} instances")
        return

    def run_job(j):
        os.makedirs(os.path.dirname(j["result_file"]), exist_ok=True)
        if resume and os.path.isfile(j["result_file"]) and os.path.getsize(j["result_file"]):
            with open(j["result_file"]) as f:
                verdict = f.readline().strip()
            return {**j, "result": verdict, "runtime": None, "skipped": True}
        env = dict(os.environ, N2V_WORKERS=str(workers))
        log = os.path.join(log_dir, j["cat"],
                           os.path.basename(j["result_file"]).replace(".txt", ".log"))
        os.makedirs(os.path.dirname(log), exist_ok=True)
        hard = j["timeout"] + 60.0
        t0 = time.time()
        try:
            with open(log, "w") as lf:
                p = subprocess.run(
                    [PY, RUNNER, j["cat"], j["onnx_arg"], j["vnnlib_arg"],
                     j["result_file"], str(j["timeout"])],
                    env=env, stdout=lf, stderr=subprocess.STDOUT,
                    timeout=hard)
            verdict = "no_result"
            if os.path.isfile(j["result_file"]) and os.path.getsize(j["result_file"]):
                with open(j["result_file"]) as f:
                    verdict = f.readline().strip()
            if p.returncode != 0 and verdict == "no_result":
                verdict = f"error_exit_{p.returncode}"
        except subprocess.TimeoutExpired:
            verdict = "timeout"            # hard kill at timeout+60 (matches harness)
            with open(j["result_file"], "w") as f:
                f.write("timeout\n")
        except Exception as e:  # noqa: BLE001
            verdict = f"driver_error: {e}"
        rt = time.time() - t0
        # Mirror a sat counterexample out to the counterexamples dir.
        if verdict == "sat":
            cf = os.path.join(cex_dir, j["cat"],
                              os.path.basename(j["result_file"]).replace(".txt", ".counterexample"))
            os.makedirs(os.path.dirname(cf), exist_ok=True)
            with open(j["result_file"]) as f:
                lines = f.read().splitlines()
            with open(cf, "w") as f:
                f.write("\n".join(lines[1:]))
        return {**j, "result": verdict, "runtime": round(rt, 1), "skipped": False}

    results = []
    start, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs_list}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            tag = "SKIP" if r.get("skipped") else f"{r.get('runtime')}s"
            print(f"[{done}/{len(jobs_list)}] {r['cat']:<34} "
                  f"{os.path.basename(r['result_file'])[:42]:<42} -> "
                  f"{r['result']:<10} {tag}", flush=True)

    # Master CSV + per-category summary.
    master = os.path.join(out_dir, "results.csv")
    with open(master, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "onnx", "vnnlib", "result", "runtime_s", "timeout_s"])
        for r in sorted(results, key=lambda x: (x["cat"], x["result_file"])):
            w.writerow([r["cat"], r["onnx_raw"], r["vnnlib_raw"], r["result"],
                        r.get("runtime"), r["timeout"]])

    by_cat = defaultdict(Counter)
    for r in results:
        v = r["result"]
        key = v if v in ("sat", "unsat", "unknown", "timeout") else "error"
        by_cat[r["cat"]][key] += 1
    print(f"\n{'category':<42}{'sat':>5}{'unsat':>6}{'unkn':>6}{'t/o':>5}{'err':>5}")
    g = Counter()
    for cat in cats:
        c = by_cat[cat]
        g.update(c)
        print(f"{cat:<42}{c['sat']:>5}{c['unsat']:>6}{c['unknown']:>6}"
              f"{c['timeout']:>5}{c['error']:>5}")
    print("-" * 64)
    print(f"{'TOTAL':<42}{g['sat']:>5}{g['unsat']:>6}{g['unknown']:>6}"
          f"{g['timeout']:>5}{g['error']:>5}")
    print(f"\nwall-clock: {(time.time()-start)/60:.1f} min   master CSV: {master}")


if __name__ == "__main__":
    main()

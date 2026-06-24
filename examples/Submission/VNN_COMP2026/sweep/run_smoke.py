#!/usr/bin/env python3
"""Smoke-test the entire VNN-COMP 2026 competition: every distinct model loads,
its spec parses, and an input set builds. No reachability is run.

Per category: pick the 2.0 instances.csv (fall back to 1.0, then flat), then for
each DISTINCT onnx field take its first instance and probe it in an isolated
subprocess with a hard timeout (giant specs / bad models can hang or segfault;
subprocess isolation is the durable pattern). Writes a JSONL of every probe and
prints a per-category summary.

The benchmark corpus is located via the N2V_VNNCOMP_BENCHMARKS env var (the same
one the differential tests use); it may point at the repo root or its
``benchmarks/`` subdir. Falls back to the local checkout path.

Usage: run_smoke.py [--all-instances] [--timeout SECS] [--jobs N]
                    [--only CAT[,CAT...]] [--out DIR]
"""
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_bench():
    env = os.environ.get("N2V_VNNCOMP_BENCHMARKS")
    if env:
        env = env.rstrip("/")
        return env if os.path.basename(env) == "benchmarks" else os.path.join(env, "benchmarks")
    return "/isis/home/sasakis/v/other/VNNCOMP/vnncomp2026_benchmarks/benchmarks"


BENCH = resolve_bench()
PROBE = os.path.join(_HERE, "smoke_probe.py")
PY = sys.executable


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
            onnx, vnnlib, timeout = (c.strip() for c in r[:3])
            rows.append((onnx, vnnlib, timeout))
    return rows


def prefixed(vdir, path):
    path = path.strip()
    if path.startswith("["):          # relational list literal: leave as-is
        return path
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(vdir, path))


def main():
    args = sys.argv[1:]
    all_instances = "--all-instances" in args
    timeout, jobs, only = 300, 24, None
    out_dir = os.path.join(_HERE, "results")
    for i, a in enumerate(args):
        if a == "--timeout":
            timeout = int(args[i + 1])
        elif a == "--jobs":
            jobs = int(args[i + 1])
        elif a == "--only":
            only = set(args[i + 1].split(","))
        elif a == "--out":
            out_dir = args[i + 1]

    cats = sorted(d for d in os.listdir(BENCH) if pick_version_dir(d))
    if only:
        cats = [c for c in cats if c in only]

    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "smoke_results.jsonl")

    # Build the flat job list (cat, ver, onnx, vnnlib).
    jobs_list, cat_ver = [], {}
    for cat in cats:
        vdir = pick_version_dir(cat)
        ver = os.path.basename(vdir) if os.path.basename(vdir) in ("2.0", "1.0") else "flat"
        cat_ver[cat] = ver
        rows = read_instances(vdir)
        if all_instances:
            probe_rows = rows
        else:
            seen, probe_rows = set(), []
            for onnx, vnnlib, to in rows:
                if onnx not in seen:
                    seen.add(onnx)
                    probe_rows.append((onnx, vnnlib, to))
        for onnx, vnnlib, to in probe_rows:
            jobs_list.append((cat, vdir, onnx, vnnlib))

    print(f"Smoke-testing {len(cats)} categories, {len(jobs_list)} probes "
          f"({'ALL instances' if all_instances else 'distinct models'}); "
          f"{jobs} parallel, {timeout}s timeout each\n  corpus: {BENCH}\n", flush=True)

    def run_probe(job):
        cat, vdir, onnx, vnnlib = job
        op, vp = prefixed(vdir, onnx), prefixed(vdir, vnnlib)
        t0 = time.time()
        try:
            p = subprocess.run([PY, PROBE, cat, op, vp],
                               capture_output=True, text=True, timeout=timeout)
            line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
            info = json.loads(line) if line else {
                "status": "fail", "error": "no output", "stderr": p.stderr.strip()[-300:]}
        except subprocess.TimeoutExpired:
            info = {"status": "fail", "error": f"TIMEOUT>{timeout}s", "stage": "subprocess"}
        except Exception as e:  # noqa: BLE001
            info = {"status": "fail", "error": f"driver: {e}"}
        info["category"], info["onnx"], info["vnnlib"] = cat, onnx, vnnlib
        info.setdefault("secs", round(time.time() - t0, 1))
        return info

    results = {c: [] for c in cats}
    done, start = 0, time.time()
    with open(out_jsonl, "w") as out, ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_probe, j): j for j in jobs_list}
        for fut in as_completed(futs):
            info = fut.result()
            results[info["category"]].append(info)
            out.write(json.dumps(info) + "\n")
            out.flush()
            done += 1
            if done % 25 == 0 or done == len(jobs_list):
                print(f"  ... {done}/{len(jobs_list)} probes "
                      f"({time.time() - start:.0f}s elapsed)", flush=True)

    print(f"\n{'category':<40}{'ver':<5}{'models':>7}{'ok':>5}{'fail':>6}  notes")
    g_ok = g_fail = 0
    for cat in cats:
        infos = results[cat]
        oks = [i for i in infos if i.get("status") == "ok"]
        fails = [i for i in infos if i.get("status") != "ok"]
        g_ok += len(oks); g_fail += len(fails)
        fmsgs = [f"{os.path.basename(str(i.get('onnx')))[:30]} [{i.get('stage')}] {i.get('error')}"
                 for i in fails]
        note = fmsgs[0][:70] if fmsgs else ""
        print(f"{cat:<40}{cat_ver[cat]:<5}{len(infos):>7}{len(oks):>5}{len(fails):>6}  {note}")
        for fmsg in fmsgs[1:]:
            print(f"{'':<57}  {fmsg[:70]}")
    print(f"\nTOTAL probes={g_ok + g_fail}  ok={g_ok}  fail={g_fail}  "
          f"({time.time() - start:.0f}s)")
    print(f"JSONL: {out_jsonl}")


if __name__ == "__main__":
    main()

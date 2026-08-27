#!/usr/bin/env python3
"""Probe whether SOUND reach (approx, then exact) decides an instance, so we can
tell whether dropping a probabilistic config concedes the hold or keeps it
soundly. Mirrors the runner's stage-2 reach (per-pair, IBP precompute,
verify_specification), but forces a sound method list and skips falsification.

Usage: sound_reach_probe.py CATEGORY ONNX VNNLIB [METHODS] [TIMEOUT]
  METHODS default "approx,exact"; TIMEOUT (s) self-enforced via SIGALRM.
Prints: VERDICT secs
"""
import os
import signal
import sys
import time

import numpy as np

_RUNNER_DIR = "/isis/home/sasakis/v/tools/n2v/examples/Submission/VNN_COMP2026"
sys.path.insert(0, _RUNNER_DIR)

import n2v  # noqa: E402
from n2v.nn import NeuralNetwork  # noqa: E402
from n2v.utils import load_vnnlib  # noqa: E402
from n2v.utils.model_loader import load_onnx  # noqa: E402
from n2v.utils.verify_specification import verify_specification  # noqa: E402
import vnncomp_runner as R  # noqa: E402


class _TO(BaseException):
    pass


def _alarm(s, f):
    raise _TO()


def main():
    cat, onnx_path, vnnlib_path = sys.argv[1], sys.argv[2], sys.argv[3]
    methods = (sys.argv[4] if len(sys.argv) > 4 else "approx,exact").split(",")
    timeout = float(sys.argv[5]) if len(sys.argv) > 5 else 300.0

    n2v.set_parallel(True, n_workers=int(os.environ.get("N2V_WORKERS", "16")))
    n2v.set_lp_solver("linprog")

    model = load_onnx(onnx_path)
    prop = load_vnnlib(vnnlib_path)
    ishape = R.get_input_shape(onnx_path)
    pairs = prop["pairs"]
    net = NeuralNetwork(model)

    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, max(1.0, timeout))
    verdict = "unknown"
    try:
        for method in methods:
            all_unsat = True
            decided_sat = False
            for pair in pairs:
                iset = R.create_input_set(pair["lb"], pair["ub"], ishape)
                extra = {"input_shape": ishape, "precompute_bounds": "ibp"}
                rs = net.reach(iset, method=method, **extra)
                v = verify_specification(rs, pair["prop"]).verdict
                if v == "SAT":
                    decided_sat = True
                    break
                elif v == "UNSAT":
                    continue
                else:
                    all_unsat = False
            if decided_sat:
                verdict = "sat"
                break
            if all_unsat:
                verdict = "unsat"
                break
    except _TO:
        verdict = "timeout"
    except Exception as e:  # noqa: BLE001
        verdict = f"error:{type(e).__name__}:{str(e)[:80]}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(f"{verdict} {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""VNN-COMP 2026 instance runner for n2v.

Called by ``run_instance.sh`` with::

    vnncomp_runner.py CATEGORY ONNX VNNLIB RESULTS_FILE TIMEOUT

Verifies one ONNX model against one VNNLIB property (1.0 or 2.0 — the
format is auto-detected by ``n2v.utils.load_vnnlib``) and writes the
VNN-COMP result to ``RESULTS_FILE``:

    sat | unsat | unknown | timeout

For ``sat`` the counterexample follows on the next lines. The runner
self-enforces ``TIMEOUT`` (the harness applies its own hard kill at
TIMEOUT+60). The strategy mirrors the 2025 runner: falsification first,
then the per-benchmark reachability methods, short-circuiting on the
first definitive result.

The ONNX argument is normally a single path. For the two-network
relational benchmarks (monotonic_acasxu, isomorphic_acasxu) the harness
passes a python list literal ``[('f', path), ('g', path)]``; those are
detected and reported ``unknown`` (relational verification is not yet
implemented).
"""

import ast
import gzip
import logging
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import time

import numpy as np

logger = logging.getLogger("vnncomp_runner")

# --- repo-relative imports -------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# Reuse the per-benchmark reachability strategies from the 2025 infra.
sys.path.insert(0, os.path.join(_REPO, "examples", "VNN-COMP"))

import n2v  # noqa: E402
from n2v.nn import NeuralNetwork  # noqa: E402
from n2v.utils import load_vnnlib, falsify  # noqa: E402
from n2v.utils.verify_specification import verify_specification  # noqa: E402
from n2v.utils.model_loader import load_onnx  # noqa: E402
from n2v.sets import Star  # noqa: E402
from n2v.sets.image_star import ImageStar  # noqa: E402

try:
    from benchmark_configs import get_config  # type: ignore
except Exception:  # pragma: no cover - fallback if examples module moves
    def get_config(category, onnx_path=None, vnnlib_path=None):
        return {
            "reach_methods": [("approx", {}), ("exact", {})],
            "n_rand": 100,
            "falsify_method": "random+pgd",
        }

RESULT_SAT = "sat"
RESULT_UNSAT = "unsat"
RESULT_UNKNOWN = "unknown"
RESULT_TIMEOUT = "timeout"


class _Timeout(BaseException):
    """Raised by the SIGALRM handler. Subclasses BaseException so the
    broad ``except Exception`` guards inside verification do NOT swallow it."""


def _on_alarm(signum, frame):
    raise _Timeout()


# ---------------------------------------------------------------------------
# Model / input-set helpers
# ---------------------------------------------------------------------------

def _maybe_decompress(path):
    """Return (usable_path, tmp_created). The harness usually decompresses,
    but handle .gz defensively."""
    if path.endswith(".gz"):
        fd, tmp = tempfile.mkstemp(suffix=os.path.basename(path)[:-3][-20:])
        os.close(fd)
        with gzip.open(path, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        return tmp, True
    return path, False


def get_input_shape(onnx_path):
    """Input tensor shape with the batch dimension stripped."""
    import onnx
    model = onnx.load(onnx_path)
    init = {i.name for i in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in init]
    if not inputs:
        raise ValueError(f"no true input tensor in {onnx_path}")
    dims = inputs[0].type.tensor_type.shape.dim
    vals = tuple(d.dim_value for d in dims)
    # Strip the leading dim only when it looks like a batch dim
    # (1, or 0 = dynamic). A rank-1 input (e.g. a flat vector packing
    # image + spec params) has no batch dim to strip.
    if len(vals) > 1 and vals[0] in (0, 1):
        vals = vals[1:]
    return vals


def create_input_set(lb, ub, input_shape):
    """Build a Star (flat) or ImageStar (spatial) from VNNLIB bounds.

    Star vs ImageStar is decided by the *non-singleton* structure of the
    batch-stripped input shape, so degenerate spatial shapes like ACAS Xu's
    ``[1,1,1,5]`` -> stripped ``(1,1,5)`` become a flat Star (a genuine
    5-vector) instead of a 1x5 single-channel "image". Real images
    ((C,H,W) or grayscale (1,H,W)) still become ImageStars.
    """
    lb = np.asarray(lb, dtype=np.float64).flatten().reshape(-1, 1)
    ub = np.asarray(ub, dtype=np.float64).flatten().reshape(-1, 1)

    nontrivial = [d for d in input_shape if d != 1]
    if len(input_shape) >= 3 and len(nontrivial) >= 2:
        H, W = nontrivial[-2], nontrivial[-1]
        C = int(np.prod(nontrivial[:-2])) if len(nontrivial) > 2 else 1
        # VNN-LIB X variables follow the ONNX input tensor order, i.e.
        # (C, H, W) row-major; ImageStar.from_bounds expects HWC. For
        # C == 1 the permutation is the identity.
        lb = lb.reshape(C, H, W).transpose(1, 2, 0).reshape(-1, 1)
        ub = ub.reshape(C, H, W).transpose(1, 2, 0).reshape(-1, 1)
        return ImageStar.from_bounds(lb, ub, height=H, width=W, num_channels=C)
    return Star.from_bounds(lb, ub)


def format_counterexample(input_vec, output_vec):
    input_vec = np.asarray(input_vec).flatten()
    output_vec = np.asarray(output_vec).flatten()
    lines = [f"(X_{i}  {v})" for i, v in enumerate(input_vec)]
    lines += [f"(Y_{i}  {v})" for i, v in enumerate(output_vec)]
    return "(" + "\n".join(lines) + ")"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _resolve_workers(workers):
    """Worker count for parallel LP. Defaults to all cores (one instance at a
    time on the competition machine); ``N2V_WORKERS`` overrides it, which is
    useful when running many instances concurrently for a smoke test."""
    if workers is not None:
        return workers
    env = os.environ.get("N2V_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return multiprocessing.cpu_count()


def verify_instance(onnx_path, vnnlib_path, category, workers=None):
    t0 = time.time()
    model = load_onnx(onnx_path)
    prop = load_vnnlib(vnnlib_path)
    input_shape = get_input_shape(onnx_path)

    # Normalized (region, prop) pairs: a counterexample exists iff SOME
    # pair has an input in its region whose output satisfies that pair's
    # own prop. Verifying any region against another pair's prop (the old
    # global-prop behavior) is unsound for combined-form specs.
    pairs = prop["pairs"]

    cfg = get_config(category, onnx_path, vnnlib_path)
    n_rand = cfg.get("n_rand", 100)
    falsify_method = cfg.get("falsify_method", "random+pgd")

    workers = _resolve_workers(workers)
    n2v.set_parallel(True, n_workers=workers)
    n2v.set_lp_solver("linprog")

    # Stage 1: falsification (counterexample search) per pair. The sample
    # budget is shared across pairs so many-region specs (e.g. lindex_200
    # with 200 pairs) don't multiply the falsification cost by the region
    # count.
    n_rand_per_pair = max(20, n_rand // max(len(pairs), 1))
    for pair in pairs:
        try:
            lb_s = np.asarray(pair["lb"], dtype=np.float64).reshape(input_shape)
            ub_s = np.asarray(pair["ub"], dtype=np.float64).reshape(input_shape)
            res, cex = falsify(model, lb_s, ub_s, pair["prop"],
                               method=falsify_method,
                               n_samples=n_rand_per_pair, seed=42)
            if res == 0 and cex is not None:
                return {"result": RESULT_SAT, "time": time.time() - t0,
                        "counterexample": format_counterexample(cex[0], cex[1])}
        except Exception as e:  # noqa: BLE001
            logger.debug("falsification failed: %s", e)

    # Stage 2: reachability methods, in configured order; each pair is
    # verified against its own prop, UNSAT only if every pair is disjoint.
    net = NeuralNetwork(model)
    for method, kwargs in cfg["reach_methods"]:
        all_unsat = True
        for pair in pairs:
            input_set = create_input_set(pair["lb"], pair["ub"], input_shape)
            try:
                extra = dict(kwargs)
                extra["input_shape"] = input_shape
                if method != "probabilistic":
                    extra["precompute_bounds"] = "ibp"
                reach_sets = net.reach(input_set, method=method, **extra)
                verdict = verify_specification(reach_sets, pair["prop"])
                if verdict.verdict == "SAT":
                    return {"result": RESULT_SAT, "time": time.time() - t0,
                            "counterexample": None}
                elif verdict.verdict == "UNSAT":
                    continue
                else:
                    all_unsat = False
            except NotImplementedError as e:
                logger.warning("unsupported in %s: %s", method, e)
                all_unsat = False
                break
            except Exception as e:  # noqa: BLE001
                logger.warning("error in %s: %s", method, e)
                all_unsat = False
        if all_unsat:
            return {"result": RESULT_UNSAT, "time": time.time() - t0,
                    "counterexample": None}

    return {"result": RESULT_UNKNOWN, "time": time.time() - t0,
            "counterexample": None}


def write_result(results_file, result_str, counterexample=None):
    with open(results_file, "w") as f:
        f.write(result_str)
        if result_str == RESULT_SAT and counterexample:
            f.write("\n")
            f.write(counterexample)
            f.write("\n")


def _resolve_relational_onnx(base, rel):
    """Resolve a tuple ONNX path against the benchmark version dir. The
    instances.csv path may not match packaging (monotonic lists
    onnx/original/X while its files are flat under onnx/), so fall back
    to a basename search under base/onnx."""
    import glob
    for cand in (os.path.join(base, rel), os.path.join(base, rel) + ".gz"):
        if os.path.exists(cand):
            return cand
    bn = os.path.basename(rel)
    hits = (glob.glob(os.path.join(base, "onnx", "**", bn), recursive=True)
            + glob.glob(os.path.join(base, "onnx", "**", bn + ".gz"),
                        recursive=True))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"relational ONNX not found: {rel}")


def verify_relational_instance(onnx_arg, vnnlib_arg, category):
    """Verify a two-network relational instance via self-composition
    (falsify -> sound joint reach)."""
    from n2v.nn.relational import solve_relational

    t0 = time.time()
    pairs = ast.literal_eval(onnx_arg.strip())   # [('f', path), ('g', path)]
    # ONNX tuple paths are relative to the benchmark version dir, which is
    # two levels up from the vnnlib (<bench>/<ver>/vnnlib/<file>).
    base = os.path.dirname(os.path.dirname(os.path.abspath(vnnlib_arg)))
    models = []
    tmps = []
    for _role, rel in pairs[:2]:
        p = _resolve_relational_onnx(base, rel)
        dp, tmp = _maybe_decompress(p)
        if tmp:
            tmps.append(dp)
        models.append(load_onnx(dp))

    vnnlib_path, vnnlib_tmp = _maybe_decompress(vnnlib_arg)
    try:
        spec = load_vnnlib(vnnlib_path)
        if spec.get("format") != "relational":
            return {"result": RESULT_UNKNOWN, "counterexample": None}

        n2v.set_parallel(False)
        n2v.set_lp_solver("linprog")
        cfg = get_config(category, str(pairs), vnnlib_arg)
        n_rand = cfg.get("n_rand", 200)
        verdict, cex = solve_relational(
            models[0], models[1], spec, method="approx", n_rand=n_rand,
            seed=42)
        if verdict == "sat":
            x, y = cex
            return {"result": RESULT_SAT, "time": time.time() - t0,
                    "counterexample": format_counterexample(x, y)}
        if verdict == "unsat":
            return {"result": RESULT_UNSAT, "time": time.time() - t0,
                    "counterexample": None}
        return {"result": RESULT_UNKNOWN, "counterexample": None}
    finally:
        if vnnlib_tmp and os.path.exists(vnnlib_path):
            os.remove(vnnlib_path)
        for t in tmps:
            if os.path.exists(t):
                os.remove(t)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 6:
        print("usage: vnncomp_runner.py CATEGORY ONNX VNNLIB RESULTS_FILE TIMEOUT",
              file=sys.stderr)
        return 2

    category = sys.argv[1]
    onnx_arg = sys.argv[2]
    vnnlib_arg = sys.argv[3]
    results_file = sys.argv[4]
    timeout = float(sys.argv[5])

    # Two-network relational instances: ONNX is a python list literal of
    # (role, path) tuples over a coupled joint input space. Tolerate a
    # stray surrounding quote in case the CSV cell reaches us unparsed.
    onnx_arg = onnx_arg.strip()
    if onnx_arg.startswith('"') and onnx_arg.endswith('"'):
        onnx_arg = onnx_arg[1:-1]
    if onnx_arg.strip().startswith("["):
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(1.0, timeout))
        try:
            result = verify_relational_instance(onnx_arg, vnnlib_arg, category)
        except _Timeout:
            result = {"result": RESULT_TIMEOUT, "counterexample": None}
        except Exception as e:  # noqa: BLE001
            logger.error("relational verification error: %s", e)
            result = {"result": RESULT_UNKNOWN, "counterexample": None}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        write_result(results_file, result["result"],
                     result.get("counterexample"))
        print(result["result"])
        if result["result"] == RESULT_SAT and result.get("counterexample"):
            print(result["counterexample"])
        return 0

    onnx_path, onnx_tmp = _maybe_decompress(onnx_arg)
    vnnlib_path, vnnlib_tmp = _maybe_decompress(vnnlib_arg)

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, max(1.0, timeout))
    try:
        result = verify_instance(onnx_path, vnnlib_path, category)
    except _Timeout:
        result = {"result": RESULT_TIMEOUT, "counterexample": None}
    except Exception as e:  # noqa: BLE001
        logger.error("verification error: %s", e)
        result = {"result": RESULT_UNKNOWN, "counterexample": None}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if onnx_tmp and os.path.exists(onnx_path):
            os.remove(onnx_path)
        if vnnlib_tmp and os.path.exists(vnnlib_path):
            os.remove(vnnlib_path)

    write_result(results_file, result["result"], result.get("counterexample"))
    print(result["result"])
    if result["result"] == RESULT_SAT and result.get("counterexample"):
        print(result["counterexample"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

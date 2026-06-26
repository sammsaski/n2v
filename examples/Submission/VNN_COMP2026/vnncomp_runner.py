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
passes a python list literal ``[('f', path), ('g', path)]``; these are
verified via self-composition (sound joint reach -> UNSAT; falsification
-> SAT). The relational SAT *witness* is currently conceded to ``unknown``
because the 2026 relational witness format (per-network ``X_f[i]``/``X_g[i]``
variables) is unconfirmed -- see ``verify_relational_instance``.

Every single-network ``sat`` witness is re-validated on the RAW ONNX via
onnxruntime before it is emitted (``n2v.utils.onnx_validate``), so an
onnx2torch conversion divergence downgrades to ``unknown`` instead of
producing a witness the official checker would reject.
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
from n2v.utils.onnx_validate import (  # noqa: E402
    onnx_forward, in_unsafe_region, make_onnx_forward,
)
from n2v.utils.falsify import _extract_halfspace_groups  # noqa: E402
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

# Per-method falsify budget knobs that a category config may tune via
# 'falsify_kwargs'. n_samples/method/seed are passed to falsify() EXPLICITLY by
# the runner, so they must never be threaded here (would raise a duplicate-arg
# TypeError); only these budget knobs are honored.
_FALSIFY_KWARG_WHITELIST = {"n_iters", "batch", "p_init",
                            "n_restarts", "n_steps", "step_size"}


def _whitelist_falsify_kwargs(d):
    """Keep only the whitelisted per-method budget knobs from a config's
    falsify_kwargs (dropping anything that falsify() already receives
    explicitly, or any unknown key)."""
    return {k: v for k, v in (d or {}).items()
            if k in _FALSIFY_KWARG_WHITELIST}


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

    # Nonlinear output property (Pow/Mul over the variables): no linear
    # half-space `pairs`, so it has its own falsify -> sound-reach path.
    if prop.get("format") == "nonlinear":
        return verify_nonlinear_instance(model, onnx_path, prop, input_shape,
                                         category, vnnlib_path, t0,
                                         workers=workers)

    # Normalized (region, prop) pairs: a counterexample exists iff SOME
    # pair has an input in its region whose output satisfies that pair's
    # own prop. Verifying any region against another pair's prop (the old
    # global-prop behavior) is unsound for combined-form specs.
    pairs = prop["pairs"]

    cfg = get_config(category, onnx_path, vnnlib_path)
    n_rand = cfg.get("n_rand", 100)
    falsify_method = cfg.get("falsify_method", "random+pgd")
    # Per-method falsify budgets from config (e.g. {'n_iters': 20000} for square,
    # {'n_restarts': 1, 'n_steps': 30} for apgd), whitelisted to budget knobs only.
    falsify_kwargs = _whitelist_falsify_kwargs(cfg.get("falsify_kwargs", {}))

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
                               n_samples=n_rand_per_pair, seed=42,
                               **falsify_kwargs)
            if res == 0 and cex is not None:
                # Re-execute the witness on the RAW ONNX in onnxruntime and
                # confirm the replayed output satisfies the (unsafe) spec with
                # NO output tolerance. This mirrors the VNN-COMP 2026 CORRECT
                # criterion exactly (scoring `counterexamples.py`,
                # `is_specification_vio(input_tol=0, output_tol=0)`): the
                # solver-provided Y is ignored and the ONNX CPU replay is the
                # sole authority for the output, with input bounds at zero
                # tolerance (the falsifier samples inside [lb, ub]). If ORT
                # disagrees (onnx2torch conversion divergence), do NOT emit a
                # `sat` the grader would score INCORRECT — keep searching the
                # remaining pairs, then fall to reach. NOTE: we do not gate on a
                # float64 re-forward — float64 is explicitly NOT the grader's
                # oracle, and gating on it discards witnesses the official ONNX
                # replay accepts (see resolved.md I-45).
                try:
                    y_ort = onnx_forward(onnx_path, cex[0])
                    groups = _extract_halfspace_groups(pair["prop"])
                    if in_unsafe_region(y_ort, groups, tol=0.0):
                        return {"result": RESULT_SAT, "time": time.time() - t0,
                                "counterexample": format_counterexample(cex[0], y_ort)}
                    logger.warning("falsify CE rejected by onnxruntime re-check "
                                   "(onnx2torch divergence); not emitting sat")
                except Exception as e:  # noqa: BLE001
                    logger.warning("onnxruntime CE re-validation error: %s", e)
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


# Falsification-sample floor for nonlinear specs. The shared DEFAULT_CONFIG
# n_rand (100) is tuned for linear half-space falsification; the nonlinear
# search relies on box corners + interior coverage near saturation boundaries,
# so it needs more samples. A per-benchmark config value, if set, wins.
_NONLINEAR_FALSIFY_SAMPLES = 2000


def verify_nonlinear_instance(model, onnx_path, prop, input_shape, category,
                              vnnlib_path, t0, workers=None):
    """Verify a ``format='nonlinear'`` single-network instance.

    Stage 1 (falsify): sample inputs in the box and test the resolved
    assertion AST for a concrete violation. A candidate is emitted as
    ``sat`` only after the witness input, replayed through onnxruntime,
    still satisfies the spec (the VNN-COMP 2026 CORRECT criterion: ONNX
    CPU replay is authoritative for the output, input bounds at zero
    tolerance; :func:`evaluate_nonlinear` evaluates both the input
    constraints on the witness and the output constraints on the replayed
    output exactly). The replayed output is finite-checked first so a
    NaN/Inf forward can never satisfy a ``!=`` conjunct and emit a spurious
    ``sat``. Stage 2 (sound reach): propagate the box through the network and
    evaluate the assertion conjunction over the reach set with three-valued
    affine/interval arithmetic -> ``unsat`` when provably violation-free,
    else ``unknown``.
    """
    from n2v.utils.verify_nonlinear import (
        verify_nonlinear_reach, falsify_nonlinear,
    )
    from n2v.utils.vnnlib2 import evaluate_nonlinear

    lb = np.asarray(prop["lb"], dtype=np.float64)
    ub = np.asarray(prop["ub"], dtype=np.float64)
    if not (np.isfinite(lb).all() and np.isfinite(ub).all()):
        # The best-effort box left an input dimension unbounded; a sound
        # reach needs a bounded input region, so concede unknown.
        logger.warning("nonlinear spec has unbounded input box; unknown")
        return {"result": RESULT_UNKNOWN, "time": time.time() - t0,
                "counterexample": None}

    cfg = get_config(category, onnx_path, vnnlib_path)
    # Use the machine the same way the linear path does (parallel LP across
    # all cores, one instance at a time) -- the sound reach is the bottleneck.
    workers = _resolve_workers(workers)
    n2v.set_parallel(True, n_workers=workers)
    n2v.set_lp_solver("linprog")

    # Stage 1: falsification. A candidate is emitted only if its input,
    # replayed through onnxruntime, still satisfies the spec — the VNN-COMP
    # 2026 CORRECT criterion (ONNX CPU replay is the output authority; the
    # input constraints, including the nonlinear ones, are checked exactly on
    # the witness at zero tolerance by ``evaluate_nonlinear``). We do NOT gate
    # on a float64 re-forward: float64 is not the grader's oracle, and gating
    # on it discards witnesses the official ONNX replay accepts (resolved.md
    # I-45) — e.g. acc instances whose output saturates on the spec boundary.
    try:
        n_rand = max(int(cfg.get("n_rand", _NONLINEAR_FALSIFY_SAMPLES)),
                     _NONLINEAR_FALSIFY_SAMPLES)
        # Search with the SAME backend the grader replays on (onnxruntime), so
        # a witness on a spec boundary isn't missed because the onnx2torch
        # float32 forward disagrees with onnxruntime there. One reused session,
        # batched, handles every search forward.
        ort_forward = make_onnx_forward(onnx_path)
        x = falsify_nonlinear(model, lb, ub, prop, input_shape,
                              n_samples=n_rand, seed=42, forward=ort_forward)
        if x is not None:
            try:
                y_ort = ort_forward(x.reshape(1, -1))[0]
                if np.all(np.isfinite(y_ort)) and \
                        evaluate_nonlinear(prop, x.ravel(), y_ort):
                    return {"result": RESULT_SAT, "time": time.time() - t0,
                            "counterexample": format_counterexample(x, y_ort)}
                logger.warning("nonlinear CE rejected by onnxruntime re-check "
                               "(onnx2torch divergence / non-finite); not "
                               "emitting sat")
            except Exception as e:  # noqa: BLE001
                logger.warning("onnxruntime CE re-validation error: %s", e)
    except Exception as e:  # noqa: BLE001
        logger.debug("nonlinear falsification failed: %s", e)

    # Stage 2: sound reachability. Each configured sound method is tried in
    # turn; an op unsupported in one method (NotImplementedError) skips to the
    # next rather than abandoning the rest (a later method may still prove it).
    net = NeuralNetwork(model)
    input_set = create_input_set(lb, ub, input_shape)
    for method, kwargs in cfg["reach_methods"]:
        if method == "probabilistic":
            continue  # sound-only path for nonlinear specs
        try:
            extra = dict(kwargs)
            extra["input_shape"] = input_shape
            extra["precompute_bounds"] = "ibp"
            reach_sets = net.reach(input_set, method=method, **extra)
            verdict = verify_nonlinear_reach(reach_sets, input_set, prop)
            if verdict == "UNSAT":
                return {"result": RESULT_UNSAT, "time": time.time() - t0,
                        "counterexample": None}
        except NotImplementedError as e:
            logger.warning("unsupported in %s: %s", method, e)
            continue
        except Exception as e:  # noqa: BLE001
            logger.warning("error in %s: %s", method, e)

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
            # The 2026 relational vnnlib declares PER-NETWORK variables
            # (X_f[i]/X_g[i], Y_f[i]/Y_g[i]); our flat (X_i, Y_i) witness format
            # does NOT match, and the relational witness syntax is unspecified in
            # the rules, so an emitted relational `sat` risks being scored
            # incorrect (−150). Until the format is confirmed with the organizers
            # (then re-validate on the raw ONNX like verify_instance), CONCEDE
            # relational SAT to `unknown` (0, sound) rather than emit a
            # likely-rejected witness. The sound UNSAT path below is unaffected.
            logger.warning("relational counterexample found but conceded to "
                           "unknown (2026 relational witness format unconfirmed)")
            return {"result": RESULT_UNKNOWN, "time": time.time() - t0,
                    "counterexample": None}
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

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
import functools
import gzip
import logging
import multiprocessing
import os
import re
import shutil
import signal
import sys
import tempfile
import time

import numpy as np
import torch

logger = logging.getLogger("vnncomp_runner")

# --- repo-relative imports -------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# Reuse the per-benchmark reachability strategies from the 2025 infra.
sys.path.insert(0, os.path.join(_REPO, "examples", "VNN-COMP"))

import n2v  # noqa: E402
from n2v.config import config as _n2v_config  # noqa: E402
from n2v.nn import NeuralNetwork  # noqa: E402
from n2v.utils import load_vnnlib, falsify  # noqa: E402
from n2v.utils.verify_specification import verify_specification  # noqa: E402
from n2v.utils.model_loader import load_onnx  # noqa: E402
from n2v.utils.onnx_validate import (  # noqa: E402
    onnx_forward, in_unsafe_region, make_onnx_forward,
)
from n2v.utils.falsify import _extract_halfspace_groups  # noqa: E402
from n2v.utils.vnnlib2 import evaluate_nonlinear  # noqa: E402
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


def _arm_timeout(timeout):
    """Self-enforce TIMEOUT via SIGALRM where available (POSIX competition host).
    On platforms without setitimer/SIGALRM (e.g. Windows, for local testing) this
    is a no-op and we rely on the harness's hard kill at TIMEOUT+60."""
    if hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer"):
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(1.0, timeout))


def _disarm_timeout():
    if hasattr(signal, "setitimer"):
        signal.setitimer(signal.ITIMER_REAL, 0)


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


def _onnx_input_dims(onnx_path):
    """Declared dims of the first true ONNX input (0 = dynamic)."""
    import onnx
    model = onnx.load(onnx_path)
    init = {i.name for i in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in init]
    if not inputs:
        raise ValueError(f"no true input tensor in {onnx_path}")
    dims = inputs[0].type.tensor_type.shape.dim
    return tuple(d.dim_value for d in dims)


def _strip_batch_dim(vals):
    # Strip the leading dim only when it looks like a batch dim
    # (1, or 0 = dynamic). A rank-1 input (e.g. a flat vector packing
    # image + spec params) has no batch dim to strip.
    if len(vals) > 1 and vals[0] in (0, 1):
        vals = vals[1:]
    return vals


def get_input_shape(onnx_path):
    """Input tensor shape with the batch dimension stripped."""
    return _strip_batch_dim(_onnx_input_dims(onnx_path))


class _PerSampleModel(torch.nn.Module):
    """Run a converted graph whose ONNX input has no batch dimension one sample
    at a time.

    The falsifiers call the model on ``(N, *input_shape)``. A graph declared on
    the bare input shape (cctsdb_yolo_2023's flat ``[12296]``, which reshapes
    it internally to ``[1, 3, 64, 64]``) rejects that, so each sample is run at
    the declared shape and the flattened outputs are stacked. Autograd flows
    through, so the gradient attacks still work."""

    def __init__(self, model, input_shape):
        super().__init__()
        self.model = model
        self.input_shape = tuple(input_shape)

    def forward(self, x):
        xs = x.reshape(-1, *self.input_shape)
        return torch.stack([self.model(xi).reshape(-1) for xi in xs])


def _falsify_model(model, input_shape, raw_dims, x_probe):
    """The model to hand to ``falsify``. Unchanged unless the ONNX input
    declares no batch dimension (nothing was stripped from ``raw_dims``) and
    the converted graph rejects a batch-prefixed input but accepts the bare
    one; then it is wrapped in :class:`_PerSampleModel`."""
    if len(input_shape) != len(raw_dims):
        return model
    x = torch.as_tensor(np.asarray(x_probe, dtype=np.float32)
                        .reshape(input_shape))
    with torch.no_grad():
        try:
            model(x.unsqueeze(0))
            return model
        except Exception:  # noqa: BLE001
            pass
        try:
            model(x)
        except Exception:  # noqa: BLE001
            return model
    logger.info("ONNX input %s has no batch dim; falsifying one sample at a "
                "time", tuple(raw_dims))
    return _PerSampleModel(model, input_shape)


def _snap_into_box(x, lb, ub):
    """Clip a candidate witness into the ORIGINAL float64 VNNLIB box.

    Witness producers reconstruct inputs from a normalised star (c + rad*alpha)
    or optimise in float32; both can land 1 ulp outside [lb, ub], and the
    grader replays input membership at ZERO tolerance. The snap happens
    BEFORE the onnxruntime re-check so the emitted (x, y) pair is consistent.
    Accepts a flat vector, a shaped vector, or a (N, d) batch.
    """
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64)
    if x.size % lb.size != 0:
        return x
    shp = x.shape
    xf = x.reshape(-1, lb.size)
    return np.clip(xf, lb[None, :], ub[None, :]).reshape(shp)


def _physical_memory_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 64 * 2**30


# Largest dense input-set basis create_input_set builds: half the machine.
_MAX_INPUT_BASIS_BYTES = _physical_memory_bytes() // 2


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

    # The set's basis is dense: dim x (perturbed dims + 1) float64. A
    # full-image box on a large input (e.g. 640x640x3) would need terabytes
    # and get the process killed before it could report; refuse it instead
    # (the caller concedes unknown for this method).
    n_pert = int(np.count_nonzero(ub > lb))
    basis_bytes = lb.size * (n_pert + 1) * 8
    if basis_bytes > _MAX_INPUT_BASIS_BYTES:
        raise MemoryError(
            f"input set basis would take {basis_bytes / 2**30:.0f} GiB "
            f"({lb.size} dims, {n_pert} perturbed)")

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


def _v2_ce_meta(vnnlib_path):
    """Declared I/O tensors of a VNNLIB 2.0 spec, in declaration order, for the
    section-5.3 textual counterexample format: a list of ``(kind, name, dtype,
    dims)`` with ``kind`` ``"input"`` or ``"output"``. Returns ``None`` for a
    1.0 spec (no ``(vnnlib-version ...)`` header), so the caller falls back to
    the legacy s-expr. Follows NNV's ``i_vnnlib2_ce_meta`` regex so our
    witnesses are byte-compatible with what the official 2.0 checker
    (``counterexamples_v2.py``) accepts."""
    if not vnnlib_path:
        return None
    try:
        with open(vnnlib_path) as f:
            txt = f.read()
    except Exception:  # noqa: BLE001
        return None
    if not re.search(r"\(\s*vnnlib-version", txt):
        return None
    meta = []
    for m in re.finditer(
            r"\(\s*declare-(input|output)\s+(\S+)\s+(\S+)\s+\[([0-9,\s]*)\]",
            txt):
        dims = m.group(4).strip()
        shape = [int(d) for d in dims.split(",") if d.strip()] if dims else []
        meta.append((m.group(1), m.group(2), m.group(3), shape))
    return meta or None


def _format_ce_v2(meta, input_vec, output_vec):
    """VNNLIB 2.0 section-5.3 textual witness: for each declared tensor, a
    ``name dtype [d0,d1,...]`` header then its C-order values one per line, in
    declaration order (``declare-output`` blocks drawn from the output witness,
    ``declare-input`` blocks from the input witness). The header repeats the
    declared dims verbatim, so a scalar ``[]`` stays ``[]`` with one value:
    the checker requires the header shape to equal the declared shape. The
    1.0 ``(X_i v)`` s-expr is scored ``malformed_ce`` for a 2.0 spec, which is
    why our 2.0-track SAT witnesses came back invalid. Raises ``ValueError``
    when a declared tensor has more values than the witness holds (the file
    would be malformed)."""
    out = []
    xi = yi = 0
    for kind, name, dtype, shape in meta:
        n = int(np.prod(shape))                 # 1 for a scalar []
        dimstr = ",".join(str(d) for d in shape)
        out.append(f"{name} {dtype} [{dimstr}]")
        if kind == "output":
            blk = output_vec[yi:yi + n]; yi += n
        else:
            blk = input_vec[xi:xi + n]; xi += n
        if len(blk) != n:
            raise ValueError(f"witness has {len(blk)} values for {kind} "
                             f"{name} {shape}, expected {n}")
        # repr() is the shortest ROUND-TRIP representation; ".16g" drops the
        # 17th significant digit, so a witness sitting exactly on a 17-digit
        # bound re-parsed outside the box (measured; invalid at zero tolerance).
        out.extend(repr(float(v)) for v in blk)
    return "\n".join(out)


def format_counterexample(input_vec, output_vec, vnnlib_path=None):
    """Emit the SAT witness in the format the grader expects for this spec's
    VNNLIB version: the section-5.3 textual assignment format for 2.0 specs,
    else the legacy ``((X_i v)...(Y_j v))`` s-expr for 1.0. The official 2.0
    checker rejects a 1.0-format witness as ``malformed_ce`` and vice-versa."""
    input_vec = np.asarray(input_vec).flatten()
    output_vec = np.asarray(output_vec).flatten()
    meta = _v2_ce_meta(vnnlib_path)
    if meta:
        return _format_ce_v2(meta, input_vec, output_vec)
    lines = [f"(X_{i}  {v})" for i, v in enumerate(input_vec)]
    lines += [f"(Y_{i}  {v})" for i, v in enumerate(output_vec)]
    return "(" + "\n".join(lines) + ")"


def _assertions_hold(prop, x, y):
    """Do the spec's own assertions hold at the witness ``(x, y)``? They are
    evaluated in float64 and again in float32, the precision in which the
    VNN-COMP 2026 checker evaluates the witness of a float32 network (it
    parses the input into float32 and compares the float32 replay output with
    NumPy's scalar promotion rules); both must hold."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    return (evaluate_nonlinear(prop, x, y)
            and evaluate_nonlinear(prop, x, y, dtype=np.float32))


def _witness_satisfies_spec(prop, groups, x, y):
    """Final SAT gate for a linear (single-network) spec: the replayed output
    ``y`` lies in the unsafe region ``groups`` of the witness's (region, prop)
    pair at zero tolerance. The parser lowers a strict ``<``/``>`` to the
    closed row, and the grader evaluates arithmetic (sums, coefficients) in
    float32, so the float64 rows can admit points the grader rejects; for
    such specs the parser also keeps the assertions (``prop['assertions']``),
    and they must hold at ``(x, y)`` as well (:func:`_assertions_hold`)."""
    if not in_unsafe_region(y, groups, tol=0.0):
        return False
    if prop.get("assertions") is None:
        return True
    return _assertions_hold(prop, x, y)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _restores_global_config(fn):
    """Restore ``n2v.config`` (parallel LP, workers, LP solver) after ``fn``.

    The runner switches the process-wide configuration to use every core;
    callers that run several instances in one process (the test suite, the
    internal harness) must get it back unchanged.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        c = _n2v_config
        saved = (c.parallel_lp, c.n_workers, c.auto_parallel,
                 c.parallel_threshold, c.lp_solver)
        try:
            return fn(*args, **kwargs)
        finally:
            (c.parallel_lp, c.n_workers, c.auto_parallel,
             c.parallel_threshold, c.lp_solver) = saved
    return wrapper


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


@_restores_global_config
def verify_instance(onnx_path, vnnlib_path, category, workers=None):
    t0 = time.time()
    prop = load_vnnlib(vnnlib_path)

    # Multimodal / multi-input specs (e.g. smart_turn: X1 [1,80,800] + X2
    # [1,3,32,112,112] -> a ~1.27M-dim joint input). The sound-reach path builds
    # one input set over the CONCATENATED joint space, whose dense generator is
    # infeasible (an n-by-n diagonal at n ~ 1.27M is 11.7 TiB); there is no
    # multi-input set construction in the reach path. Concede `unknown` (sound,
    # -150-safe) from the spec header BEFORE loading the (124 MB) model, instead
    # of attempting -- and OOMing on -- the dense set. Mirrors NNV's header-level
    # multimodal gate.
    if len(prop.get("input_tensors", [])) > 1:
        logger.info("multi-input/multimodal spec (%d input tensors): sound reach "
                    "over the joint input is intractable; conceding unknown",
                    len(prop["input_tensors"]))
        return {"result": RESULT_UNKNOWN, "time": time.time() - t0,
                "counterexample": None}

    # ViT (transformer self-attention): onnx2torch cannot ingest the exported
    # graph (Slice v1 + per-token BatchNorm), and a star+LP attention reach does
    # not scale. Route to the LP-free CROWN attention verifier (method translated from NNV
    # ViTCrown): reconstruct ViT_BN from the ONNX initializers, lower to the CROWN
    # op DAG, and prove robustness with backward linear bounds + refinement + α.
    if category and "vit" in category.lower():
        return verify_vit_instance(onnx_path, prop, get_input_shape(onnx_path),
                                   category, t0, vnnlib_path, workers=workers)

    model = load_onnx(onnx_path)
    raw_dims = _onnx_input_dims(onnx_path)
    input_shape = _strip_batch_dim(raw_dims)

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
    f_model = model
    if pairs:
        try:
            f_model = _falsify_model(
                model, input_shape, raw_dims,
                (np.asarray(pairs[0]["lb"], dtype=np.float64)
                 + np.asarray(pairs[0]["ub"], dtype=np.float64)) / 2)
        except Exception as e:  # noqa: BLE001
            logger.warning("falsify model probe failed: %s", e)
    for pair in pairs:
        try:
            lb_s = np.asarray(pair["lb"], dtype=np.float64).reshape(input_shape)
            ub_s = np.asarray(pair["ub"], dtype=np.float64).reshape(input_shape)
            res, cex = falsify(f_model, lb_s, ub_s, pair["prop"],
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
                # replay accepts (see resolved.md I-45). A strict comparison or
                # arithmetic is checked here as the checker evaluates it
                # (_witness_satisfies_spec), although the pair's half-spaces
                # are a float64 closure.
                try:
                    cx0 = _snap_into_box(cex[0], pair["lb"], pair["ub"])
                    y_ort = onnx_forward(onnx_path, cx0)
                    groups = _extract_halfspace_groups(pair["prop"])
                    if _witness_satisfies_spec(prop, groups, cx0, y_ort):
                        return {"result": RESULT_SAT, "time": time.time() - t0,
                                "counterexample": format_counterexample(
                                    cx0, y_ort, vnnlib_path)}
                    logger.warning("falsify CE rejected by onnxruntime re-check "
                                   "(onnx2torch divergence, or a strict bound "
                                   "or float32 arithmetic in the spec); "
                                   "not emitting sat")
                except Exception as e:  # noqa: BLE001
                    logger.warning("onnxruntime CE re-validation error: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.warning("falsification failed: %s", e)

    # Stage 2: reachability methods, in configured order; each pair is
    # verified against its own prop, UNSAT only if every pair is disjoint.
    net = NeuralNetwork(model)
    for method, kwargs in cfg["reach_methods"]:
        all_unsat = True
        for pair in pairs:
            try:
                input_set = create_input_set(pair["lb"], pair["ub"],
                                             input_shape)
                extra = dict(kwargs)
                extra["input_shape"] = input_shape
                if method != "probabilistic":
                    extra["precompute_bounds"] = "ibp"
                reach_sets = net.reach(input_set, method=method, **extra)
                verdict = verify_specification(reach_sets, pair["prop"])
                if verdict.verdict == "SAT":
                    # Sound reach proved a counterexample EXISTS (only EXACT
                    # reach can soundly return SAT; approx over-approx cannot).
                    # A `sat` WITHOUT a witness is penalized -150 by the grader,
                    # so emit the reach witness, re-validated on the raw ONNX at
                    # zero output tolerance exactly like the falsification path.
                    # If no witness can be validated, concede `unknown` (0, safe)
                    # rather than a witness-less sat or an (unsound) unsat.
                    cx = getattr(verdict, "counterexample_x", None)
                    if cx is not None:
                        try:
                            cx = np.asarray(cx, dtype=np.float64).flatten()
                            cx = _snap_into_box(cx, pair["lb"], pair["ub"])
                            y_ort = onnx_forward(onnx_path, cx)
                            groups = _extract_halfspace_groups(pair["prop"])
                            if _witness_satisfies_spec(prop, groups, cx,
                                                       y_ort):
                                return {"result": RESULT_SAT,
                                        "time": time.time() - t0,
                                        "counterexample":
                                            format_counterexample(
                                                cx, y_ort, vnnlib_path)}
                        except Exception as e:  # noqa: BLE001
                            logger.warning("reach-SAT witness ORT re-validation "
                                           "error: %s", e)
                    logger.warning("reach returned SAT without a validatable "
                                   "witness; conceding unknown (sat-without-"
                                   "witness is penalized)")
                    return {"result": RESULT_UNKNOWN, "time": time.time() - t0,
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


@_restores_global_config
def verify_nonlinear_instance(model, onnx_path, prop, input_shape, category,
                              vnnlib_path, t0, workers=None):
    """Verify a ``format='nonlinear'`` single-network instance.

    Stage 1 (falsify): sample inputs in the box and test the resolved
    assertion AST for a concrete violation. A candidate is emitted as
    ``sat`` only after the witness input, replayed through onnxruntime,
    still satisfies the spec (the VNN-COMP 2026 CORRECT criterion: ONNX
    CPU replay is authoritative for the output, input bounds at zero
    tolerance; :func:`_assertions_hold` evaluates both the input
    constraints on the witness and the output constraints on the replayed
    output exactly, in float64 and in the checker's float32). The replayed
    output is finite-checked first so a NaN/Inf forward can never satisfy a
    ``!=`` conjunct and emit a spurious ``sat``. Stage 2 (sound reach): propagate the box through the network and
    evaluate the assertion conjunction over the reach set with three-valued
    affine/interval arithmetic -> ``unsat`` when provably violation-free,
    else ``unknown``.
    """
    from n2v.utils.verify_nonlinear import (
        verify_nonlinear_reach, falsify_nonlinear,
        network_keeps_input_predicates,
    )

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
    # the witness at zero tolerance by ``_assertions_hold``). We do NOT gate
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
                x = _snap_into_box(x, lb, ub)
                y_ort = ort_forward(x.reshape(1, -1))[0]
                if np.all(np.isfinite(y_ort)) and \
                        _assertions_hold(prop, x, y_ort):
                    return {"result": RESULT_SAT, "time": time.time() - t0,
                            "counterexample": format_counterexample(
                                x, y_ort, vnnlib_path)}
                logger.warning("nonlinear CE rejected by onnxruntime re-check "
                               "(onnx2torch divergence / non-finite); not "
                               "emitting sat")
            except Exception as e:  # noqa: BLE001
                logger.warning("onnxruntime CE re-validation error: %s", e)
    except Exception as e:  # noqa: BLE001
        logger.warning("nonlinear falsification failed: %s", e)

    # Stage 2: sound reachability. Each configured sound method is tried in
    # turn; an op unsupported in one method (NotImplementedError) skips to the
    # next rather than abandoning the rest (a later method may still prove it).
    net = NeuralNetwork(model)
    input_set = create_input_set(lb, ub, input_shape)
    # A pure Linear/ReLU network keeps the input predicates as the prefix of
    # every output star, so x and y may share them (a conjunct relating x and
    # y is then decided exactly); any other network uses disjoint variables.
    shared = network_keeps_input_predicates(net)
    for method, kwargs in cfg["reach_methods"]:
        if method == "probabilistic":
            continue  # sound-only path for nonlinear specs
        try:
            extra = dict(kwargs)
            extra["input_shape"] = input_shape
            extra["precompute_bounds"] = "ibp"
            reach_sets = net.reach(input_set, method=method, **extra)
            verdict = verify_nonlinear_reach(
                reach_sets, input_set, prop, input_predicates_shared=shared)
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


# Output slack for the batched ViT candidate screen (absolute, on G @ y - g).
# Rows inside it are replayed at batch 1 and checked at zero tolerance; it only
# has to exceed the batched-vs-batch-1 difference (~2e-6 measured).
_VIT_SCREEN_TOL = 1e-4


def _vit_pgd(md, lb, ub, group, rng, n_starts=8, steps=60):
    """Gradient counterexample search on the reconstructed ViT (torch): drive the
    worst OR-halfspace value ``min_k (G_k @ Y - g_k)`` <= 0 inside the box."""
    import torch
    G = torch.tensor(np.vstack([np.atleast_2d(h.G) for h in group]), dtype=torch.float64)
    g = torch.tensor(np.concatenate([np.asarray(h.g).reshape(-1) for h in group]),
                     dtype=torch.float64)
    lo = torch.tensor(lb.reshape(3, 32, 32)); hi = torch.tensor(ub.reshape(3, 32, 32))
    outs = []
    for _ in range(n_starts):
        x0 = lb + (ub - lb) * rng.random(lb.size)
        x = torch.tensor(x0.reshape(3, 32, 32), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([x], lr=1e-2)
        for _ in range(steps):
            opt.zero_grad()
            y = md(x.reshape(1, 3, 32, 32))[0]
            (G @ y - g).min().backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(lo, hi)
        outs.append(x.detach().reshape(-1).cpu().numpy())
    return np.asarray(outs)


def verify_vit_instance(onnx_path, prop, input_shape, category, t0,
                        vnnlib_path=None, workers=None):
    """Verify a ViT robustness instance LP-free (method translated from NNV ViTCrown, no NNV dep).

    SAT iff a falsifier (random + ViT-gradient PGD), re-validated on the RAW ONNX
    in onnxruntime at batch 1 and zero tolerance, lands in the unsafe region.
    UNSAT iff the LP-free CROWN reach proves every (region, prop) pair disjoint
    from its unsafe region (a pair is safe iff SOME OR-group is fully
    unreachable, i.e. CROWN bounds ``G_k @ Y > g_k`` for every row). Else
    unknown (sound)."""
    import torch
    from n2v.nn.vit_crown import load_vit_onnx, to_ops, verify_halfspace_group_safe

    try:
        model = load_vit_onnx(onnx_path)
        ops = to_ops(model)
    except Exception as e:  # noqa: BLE001 — unfamiliar ViT graph: concede unknown
        logger.warning("ViT reconstruction failed (%s); unknown", e)
        return {"result": RESULT_UNKNOWN, "time": time.time() - t0, "counterexample": None}

    pairs = prop["pairs"]
    cfg = get_config(category, onnx_path, None)
    n_rand = cfg.get("n_rand", 200)
    rng = np.random.default_rng(42)
    fwd = make_onnx_forward(onnx_path)

    # Stage 1: falsification (counterexample search), grader-validated on raw ONNX.
    for pair in pairs:
        lb = np.asarray(pair["lb"], dtype=np.float64).reshape(-1)
        ub = np.asarray(pair["ub"], dtype=np.float64).reshape(-1)
        groups = _extract_halfspace_groups(pair["prop"])
        X = lb + (ub - lb) * rng.random((n_rand, lb.size))
        try:
            X = np.vstack([X] + [_vit_pgd(model, lb, ub, gp, rng) for gp in groups])
        except Exception as e:  # noqa: BLE001
            logger.debug("ViT PGD failed: %s", e)
        X = _snap_into_box(X, lb, ub)
        # The batched run is a screen only: its rows can differ from the
        # batch-1 replay the grader performs by float reordering (up to ~2e-6
        # on ibp_3_3_8). A row within a small slack of the unsafe region is
        # replayed alone, and the witness is judged and emitted on that output.
        Y = fwd(X)
        for i in range(X.shape[0]):
            if not in_unsafe_region(Y[i], groups, tol=_VIT_SCREEN_TOL):
                continue
            try:
                y1 = fwd(X[i:i + 1])[0]
                if np.all(np.isfinite(y1)) and \
                        _witness_satisfies_spec(prop, groups, X[i], y1):
                    return {"result": RESULT_SAT, "time": time.time() - t0,
                            "counterexample": format_counterexample(
                                X[i], y1, vnnlib_path)}
            except Exception as e:  # noqa: BLE001
                logger.warning("ViT CE re-validation error: %s", e)

    # Stage 2: LP-free CROWN reach. UNSAT iff every pair is provably safe.
    all_unsat = True
    for pair in pairs:
        lb = np.asarray(pair["lb"], dtype=np.float64).reshape(-1)
        ub = np.asarray(pair["ub"], dtype=np.float64).reshape(-1)
        groups = _extract_halfspace_groups(pair["prop"])
        pair_safe = False
        for group in groups:           # AND across groups: safe if ANY is unreachable
            G = np.vstack([np.atleast_2d(h.G) for h in group])
            g = np.concatenate([np.asarray(h.g).reshape(-1) for h in group])
            try:
                safe, _ = verify_halfspace_group_safe(
                    ops, lb, ub, G, g, refine=True, refine_iters=2, alpha=True)
            except Exception as e:  # noqa: BLE001
                logger.warning("CROWN reach error: %s", e)
                safe = False
            if safe:
                pair_safe = True
                break
        if not pair_safe:
            all_unsat = False
    if all_unsat:
        return {"result": RESULT_UNSAT, "time": time.time() - t0, "counterexample": None}
    return {"result": RESULT_UNKNOWN, "time": time.time() - t0, "counterexample": None}


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


@_restores_global_config
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
        _arm_timeout(timeout)
        try:
            result = verify_relational_instance(onnx_arg, vnnlib_arg, category)
        except _Timeout:
            result = {"result": RESULT_TIMEOUT, "counterexample": None}
        except Exception as e:  # noqa: BLE001
            logger.error("relational verification error: %s", e)
            result = {"result": RESULT_UNKNOWN, "counterexample": None}
        finally:
            _disarm_timeout()
        write_result(results_file, result["result"],
                     result.get("counterexample"))
        print(result["result"])
        if result["result"] == RESULT_SAT and result.get("counterexample"):
            print(result["counterexample"])
        return 0

    onnx_path, onnx_tmp = _maybe_decompress(onnx_arg)
    vnnlib_path, vnnlib_tmp = _maybe_decompress(vnnlib_arg)

    _arm_timeout(timeout)
    try:
        result = verify_instance(onnx_path, vnnlib_path, category)
    except _Timeout:
        result = {"result": RESULT_TIMEOUT, "counterexample": None}
    except Exception as e:  # noqa: BLE001
        logger.error("verification error: %s", e)
        result = {"result": RESULT_UNKNOWN, "counterexample": None}
    finally:
        _disarm_timeout()
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

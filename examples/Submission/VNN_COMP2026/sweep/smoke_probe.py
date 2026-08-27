#!/usr/bin/env python3
"""Smoke-probe ONE VNN-COMP instance: load model + parse spec + build input set.

Usage: smoke_probe.py CATEGORY ONNX VNNLIB

Does NOT run reachability. Surfaces load/parse/input-set errors that the real
runner would otherwise swallow into 'unknown'. Reuses the competition runner's
own helpers so the smoke result reflects exactly what the real run does at load
time. Prints a single JSON line to stdout. Driven by run_smoke.py.
"""
import ast
import json
import os
import sys
import time
import traceback

# The competition runner lives one directory up.
_RUNNER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RUNNER_DIR)

import vnncomp_runner as R  # noqa: E402
from n2v.utils import load_vnnlib  # noqa: E402
from n2v.utils.model_loader import load_onnx  # noqa: E402


def probe(category, onnx_arg, vnnlib_arg):
    info = {"category": category, "onnx": onnx_arg, "vnnlib": vnnlib_arg,
            "status": "ok", "stage": None, "error": None}
    onnx_arg = onnx_arg.strip()
    if onnx_arg.startswith('"') and onnx_arg.endswith('"'):
        onnx_arg = onnx_arg[1:-1]

    # --- relational (two-network) -----------------------------------------
    if onnx_arg.startswith("["):
        info["kind"] = "relational"
        pairs = ast.literal_eval(onnx_arg)
        base = os.path.dirname(os.path.dirname(os.path.abspath(vnnlib_arg)))
        info["stage"] = "load_onnx"
        for _role, rel in pairs[:2]:
            p = R._resolve_relational_onnx(base, rel)
            dp, tmp = R._maybe_decompress(p)
            try:
                load_onnx(dp)
            finally:
                if tmp and os.path.exists(dp):
                    os.remove(dp)
        info["stage"] = "load_vnnlib"
        vp, vtmp = R._maybe_decompress(vnnlib_arg)
        try:
            spec = load_vnnlib(vp)
        finally:
            if vtmp and os.path.exists(vp):
                os.remove(vp)
        info["spec_format"] = spec.get("format")
        info["n_models"] = len(pairs)
        return info

    # --- single network ---------------------------------------------------
    info["kind"] = "single"
    onnx_path, otmp = R._maybe_decompress(onnx_arg)
    vnnlib_path, vtmp = R._maybe_decompress(vnnlib_arg)
    try:
        info["stage"] = "load_onnx"
        load_onnx(onnx_path)
        info["stage"] = "load_vnnlib"
        prop = load_vnnlib(vnnlib_path)
        info["spec_format"] = prop.get("format", "pairs")
        info["stage"] = "get_input_shape"
        ishape = R.get_input_shape(onnx_path)
        info["input_shape"] = list(ishape)
        info["stage"] = "create_input_set"
        if "pairs" in prop and prop["pairs"]:
            pr = prop["pairs"][0]
            s = R.create_input_set(pr["lb"], pr["ub"], ishape)
            info["set_type"] = type(s).__name__
            info["n_pairs"] = len(prop["pairs"])
        info["stage"] = "done"
    finally:
        if otmp and os.path.exists(onnx_path):
            os.remove(onnx_path)
        if vtmp and os.path.exists(vnnlib_path):
            os.remove(vnnlib_path)
    return info


def main():
    t0 = time.time()
    category, onnx_arg, vnnlib_arg = sys.argv[1], sys.argv[2], sys.argv[3]
    info = {"category": category, "onnx": onnx_arg, "vnnlib": vnnlib_arg,
            "status": "ok", "stage": None, "error": None}
    try:
        info = probe(category, onnx_arg, vnnlib_arg)
        info["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        info["status"] = "fail"
        info["error"] = f"{type(e).__name__}: {e}"
        info["traceback"] = traceback.format_exc().splitlines()[-4:]
    info["secs"] = round(time.time() - t0, 1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()

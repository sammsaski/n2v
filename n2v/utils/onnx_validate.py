"""Re-validate a falsification counterexample on the RAW ONNX via onnxruntime.

VNN-COMP 2026 grades a ``sat`` witness by replaying the witness *input* through
the ORIGINAL ONNX model in onnxruntime (the tool's competition CPU; solver
output values are ignored) and checking that the input satisfies the VNN-LIB
input bounds at **zero tolerance** and the replayed output satisfies the output
constraints with **output tolerance 0** (vnncomp2026 issue #2). n2v finds
counterexamples on the onnx2torch-CONVERTED model, so a converted-vs-raw
divergence (lossy/shimmed op conversion) could produce a witness the grader
rejects — scored *incorrect* (a catastrophic −150).

This module re-runs the witness input on the raw ONNX and reports whether it
genuinely lands in the unsafe region. The caller should emit the returned ORT
output as the witness ``Y`` so the grader's reproduction check passes by
construction. Conservative by design: any error or non-violation means "not a
sound counterexample", so the caller downgrades ``sat`` → ``unknown`` (0 points)
rather than emit a witness that would score −150.
"""

import numpy as np


def _concrete_input_shape(inp) -> list:
    """Declared ONNX input shape with dynamic/symbolic/batch dims coerced to 1."""
    return [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]


def onnx_forward(onnx_path: str, x_flat) -> np.ndarray:
    """Run ``x_flat`` (flat, in ONNX input-variable order) through the raw ONNX
    in onnxruntime; return the output flattened. Reshapes ``x_flat`` to the
    model's declared input shape (dynamic/batch dims coerced to 1).

    For a single witness re-check. For many forwards over one model (a sampling
    search), build a reusable session once via :func:`make_onnx_forward`."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = np.asarray(x_flat, dtype=np.float32).reshape(_concrete_input_shape(inp))
    out = sess.run(None, {inp.name: x})[0]
    return np.asarray(out).reshape(-1)


def make_onnx_forward(onnx_path: str):
    """Build a reusable BATCHED onnxruntime forward over one model, backed by a
    single CPU ``InferenceSession`` (the backend VNN-COMP 2026 replays against).

    The returned callable maps ``X`` of shape ``(N, d)`` (each row a flat input
    in ONNX input-variable order, concatenated across all input tensors) to
    ``(N, out)`` (flat outputs concatenated across all output tensors, matching
    the joint flat indexing the VNNLIB resolver uses). A 1-D ``(d,)`` input is
    accepted and treated as a single row. When the model's batch dim is dynamic
    it runs the whole batch in ONE ``sess.run`` call; otherwise it loops rows
    over the single cached session (still far cheaper than rebuilding a session
    per sample as :func:`onnx_forward` would)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    inps = sess.get_inputs()
    shapes = [_concrete_input_shape(i) for i in inps]
    sizes = [int(np.prod(s)) for s in shapes]
    names = [i.name for i in inps]
    # True batching is available only for a single input whose leading dim is
    # dynamic (not a fixed positive int).
    can_batch = (len(inps) == 1 and len(inps[0].shape) >= 1
                 and not (isinstance(inps[0].shape[0], int) and inps[0].shape[0] > 0))

    def _row(xrow):
        feeds, off = {}, 0
        for nm, s, sz in zip(names, shapes, sizes):
            feeds[nm] = xrow[off:off + sz].astype(np.float32).reshape(s)
            off += sz
        outs = sess.run(None, feeds)
        return np.concatenate([np.asarray(o, dtype=np.float64).ravel() for o in outs])

    def forward(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N = X.shape[0]
        if can_batch and N > 1:
            batched = X.astype(np.float32).reshape([N] + shapes[0][1:])
            outs = sess.run(None, {names[0]: batched})
            return np.concatenate(
                [np.asarray(o, dtype=np.float64).reshape(N, -1) for o in outs], axis=1)
        return np.stack([_row(X[i]) for i in range(N)], axis=0)

    return forward


def in_unsafe_region(y, groups, tol: float = 1e-4) -> bool:
    """AND-of-OR unsafe-region membership for output ``y`` (``G @ y <= g + tol``).

    ``groups`` is the ``List[List[HalfSpace]]`` produced by
    ``_extract_halfspace_groups``: AND across groups, OR within a group, and all
    rows of a halfspace must hold. The default ``tol=1e-4`` is a permissive
    membership check; the VNN-COMP 2026 grader uses **output tolerance 0**, so
    the runner's ``sat`` gate passes ``tol=0.0`` to match the grader exactly
    (emitting a witness only the raw-ONNX replay confirms with no slack).
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    for group in groups:                       # AND across groups
        hit = False
        for hs in group:                       # OR within a group
            G = np.asarray(hs.G, dtype=np.float64)
            g = np.asarray(hs.g, dtype=np.float64).reshape(-1, 1)
            if bool(np.all(G @ y <= g + tol)):
                hit = True
                break
        if not hit:
            return False
    return True

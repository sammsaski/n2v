"""Soundness of the relational (two-network) reachability engine.

Self-contained: builds tiny ReLU MLPs and synthetic relational specs
(no benchmark-data dependency). Checks (a) the joint reach star encloses
every true [f(x_f); g(x_g)] over the coupled input — the core soundness
property — and (b) the engine's verdicts on controlled SAT/UNSAT cases.
"""

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

from n2v.sets.halfspace import HalfSpace
from n2v.utils.model_loader import load_onnx
from n2v.nn.relational import (
    build_joint_input_star, relational_reach, verify_relational,
    solve_relational, _sample_coupled,
)


def _mlp(tmp_path, name, seed, n_in=2, n_hidden=4, n_out=2):
    """Tiny Gemm-Relu-Gemm ONNX model; returns the loaded fx model."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((n_hidden, n_in)).astype(np.float32)
    b1 = rng.standard_normal(n_hidden).astype(np.float32)
    W2 = rng.standard_normal((n_out, n_hidden)).astype(np.float32)
    b2 = rng.standard_normal(n_out).astype(np.float32)

    def _t(nm, a):
        return helper.make_tensor(nm, TensorProto.FLOAT, list(a.shape),
                                  a.flatten().tolist())
    nodes = [
        helper.make_node("Gemm", ["x", "W1", "b1"], ["h"], transB=1),
        helper.make_node("Relu", ["h"], ["r"]),
        helper.make_node("Gemm", ["r", "W2", "b2"], ["y"], transB=1),
    ]
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, n_in])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, n_out])],
        initializer=[_t("W1", W1), _t("b1", b1), _t("W2", W2), _t("b2", b2)])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    path = str(tmp_path / f"{name}.onnx")
    onnx.save(model, path)
    return load_onnx(path)


def _equal_input_spec(prop_G, prop_g, n=2):
    """Spec over [X_f(n); X_g(n)] with X_f boxed in [-1,1], coupling
    X_f == X_g, and a single output HalfSpace over [Y_f(n); Y_g(n)]."""
    lb = np.concatenate([-np.ones(n), np.full(n, -np.inf)])
    ub = np.concatenate([np.ones(n), np.full(n, np.inf)])
    rows = []
    for i in range(n):                       # X_f[i] == X_g[i]
        r = np.zeros(2 * n); r[i] = 1.0; r[n + i] = -1.0
        rows.append(r.copy())
        rows.append(-r)
    coupling = HalfSpace(np.array(rows), np.zeros(len(rows)))
    return {
        "format": "relational",
        "networks": [
            {"name": "f", "relation": None, "input_offset": 0,
             "input_size": n, "output_offset": 0, "output_size": n},
            {"name": "g", "relation": ("equal-to", "f"), "input_offset": n,
             "input_size": n, "output_offset": n, "output_size": n},
        ],
        "lb": lb, "ub": ub,
        "input_coupling": coupling,
        "prop": [{"Hg": HalfSpace(np.asarray(prop_G, dtype=float),
                                  np.asarray(prop_g, dtype=float))}],
    }


def _fwd(model, x, n_in):
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)
                     .reshape(1, n_in)).numpy().flatten()


class TestRelationalSoundness:
    def test_join_encloses_true_joint_output(self, tmp_path):
        f = _mlp(tmp_path, "f", 1)
        g = _mlp(tmp_path, "g", 2)        # different network
        # unsafe region is irrelevant for the containment check
        spec = _equal_input_spec([[1, 0, -1, 0]], [0.0])
        joint = relational_reach(f, g, spec, method="approx")
        lo, hi = joint.get_ranges()
        lo, hi = np.asarray(lo).flatten(), np.asarray(hi).flatten()

        S = build_joint_input_star(spec)
        plb = S.predicate_lb.flatten()
        pub = S.predicate_ub.flatten()
        rng = np.random.default_rng(0)
        for _ in range(400):
            x = _sample_coupled(spec, plb, pub, rng)
            y = np.concatenate([_fwd(f, x[:2], 2), _fwd(g, x[2:4], 2)])
            assert np.all(y >= lo - 1e-5) and np.all(y <= hi + 1e-5), (
                f"joint reach not sound: y={y}\n lo={lo}\n hi={hi}")

    def test_inequality_coupling_sound(self, tmp_path):
        # X_g[0] <= X_f[0], X_f[1] == X_g[1]
        f = _mlp(tmp_path, "f", 3)
        spec = _equal_input_spec([[1, 0, -1, 0]], [0.0])
        # X_g[0] <= X_f[0], X_f[1] == X_g[1]
        spec["lb"] = np.array([-1., -1., -1., -np.inf])
        spec["ub"] = np.array([1., 1., np.inf, np.inf])
        spec["input_coupling"] = HalfSpace(
            np.array([[-1., 0., 1., 0.],          # X_g[0] <= X_f[0]
                      [0., 1., 0., -1.],          # X_f[1] == X_g[1]
                      [0., -1., 0., 1.]]),
            np.zeros(3))
        joint = relational_reach(f, f, spec, method="approx")
        lo, hi = joint.get_ranges()
        lo, hi = np.asarray(lo).flatten(), np.asarray(hi).flatten()
        S = build_joint_input_star(spec)
        plb, pub = S.predicate_lb.flatten(), S.predicate_ub.flatten()
        rng = np.random.default_rng(1)
        for _ in range(400):
            x = _sample_coupled(spec, plb, pub, rng)
            assert x[2] <= x[0] + 1e-9 and abs(x[1] - x[3]) < 1e-9
            y = np.concatenate([_fwd(f, x[:2], 2), _fwd(f, x[2:4], 2)])
            assert np.all(y >= lo - 1e-5) and np.all(y <= hi + 1e-5)


class TestRelationalVerdicts:
    def test_unsat_unsafe_region_far_from_outputs(self, tmp_path):
        # Unsafe region Y_f[0] <= -1e6 is far below the reachable outputs
        # (O(1) for a tiny net on a [-1,1] box), so the sound marginal
        # reach bound proves disjointness => UNSAT. (A tight relational
        # UNSAT like "Y_f==Y_g" is NOT provable here — the product
        # relaxation loses the cross-network correlation; that is a
        # documented precision limit, not a soundness one.)
        f = _mlp(tmp_path, "f", 7)
        spec = _equal_input_spec([[1, 0, 0, 0]], [-1e6])
        res = verify_relational(f, f, spec, method="approx")
        assert res.verdict == "UNSAT"

    def test_sat_when_unsafe_region_reachable(self, tmp_path):
        # f == g, X_f == X_g => Y_f == Y_g. Unsafe region
        # Y_f[0] - Y_g[0] <= 0.001 always holds (=0) => falsifier SATs.
        f = _mlp(tmp_path, "f", 7)
        spec = _equal_input_spec([[1, 0, -1, 0]], [0.001])
        verdict, cex = solve_relational(f, f, spec, method="approx",
                                        n_rand=100)
        assert verdict == "sat" and cex is not None
        x, y = cex
        assert y[0] - y[2] <= 0.001 + 1e-9

"""
End-to-end GNN verification on a power-flow checkpoint (GCN / SAGE / GINE).

Demonstrates the full n2v GNN pipeline through the GraphNeuralNetwork wrapper:

  1. load any .mat checkpoint (model type auto-detected),
  2. forward parity vs the saved PyTorch predictions,
  3. full-graph reach under an L_inf node-feature perturbation,
  4. k-hop subgraph reach on a few target nodes (exact per target, scalable),
  5. per-target local robustness (|f(X') - f(X)| <= delta) via verify_node_bounds,
  6. verify-or-falsify on a spec the model violates,
  7. sample-and-contain soundness.

Works unchanged for gcn / sage / gine_pretrain checkpoints.  Defaults to the
IEEE-24 SAGE checkpoint bundled with the test fixtures.

Usage (from the repository root):
    python examples/GNN/verify_gnn.py
    python examples/GNN/verify_gnn.py --eps 0.01 --delta 0.05
    python examples/GNN/verify_gnn.py \
        --checkpoint tests/fixtures/gnn/ieee118_pf/gine_pretrain_pf_ieee118.mat --targets 0 40 80
"""

import argparse
from pathlib import Path

import numpy as np

from n2v.sets import GraphStar
from n2v.nn import GraphNeuralNetwork
from n2v.utils import load_gnn_mat, verify_node_bounds


DEFAULT_CHECKPOINT = (Path(__file__).resolve().parents[2]
                      / "tests" / "fixtures" / "gnn" / "ieee24_pf" / "sage_pf_ieee24.mat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.05,
                        help="robustness tolerance: target outputs must stay within "
                             "+/- delta of the nominal prediction")
    parser.add_argument("--targets", type=int, nargs="*", default=None,
                        help="target node indices for subgraph reach (default: first 3)")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--full-graph-max-nodes", type=int, default=30,
                        help="above this node count, skip full-graph bound extraction "
                             "(LP-per-output) and use per-target subgraphs only")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}")
        return 1

    model = load_gnn_mat(args.checkpoint)
    net = GraphNeuralNetwork.from_model(model)
    print(f"loaded {model.model_type}: {net}")
    print(f"  {model.num_layers} layers, {model.num_test_instances} test instances")

    X = model.X_test[args.instance]
    Y_pt = model.python_predictions[args.instance]
    N = X.shape[0]
    print(f"\ninstance {args.instance}: X shape {X.shape}")

    # ---- forward parity ----------------------------------------------------
    parity = float(np.max(np.abs(net.evaluate(X) - Y_pt)))
    print(f"forward parity vs PyTorch:  max |dY| = {parity:.3e}")

    # ---- full-graph reach --------------------------------------------------
    print(f"\nfull-graph reach, eps = {args.eps}")
    is_gine = model.gine_variant is not None
    adj = None if is_gine else model.adjacency
    gs_in = GraphStar.from_bounds(X - args.eps, X + args.eps, adjacency=adj)
    out = net.reach(gs_in)[0]
    print(f"  output shape ({N}, {out.F}), predicates {out.nVar}")

    # Extracting tight LP bounds for *all* outputs is one LP per output entry and
    # does not scale; only do it (and the full-graph spec/soundness) on small graphs.
    small = N <= args.full_graph_max_nodes
    lb = ub = None
    if small:
        lb, ub = out.get_ranges()
        print(f"  bound width: mean = {(ub - lb).mean():.4e}, max = {(ub - lb).max():.4e}")
    else:
        print(f"  full-graph get_ranges skipped (N={N} > {args.full_graph_max_nodes}; "
              f"use per-target subgraphs)")

    # ---- k-hop subgraph reach (exact per target, always scalable) ----------
    targets = args.targets if args.targets is not None else list(range(min(3, N)))
    print(f"\nk-hop subgraph reach on targets {targets}")
    sub_results = net.reach_subgraph(gs_in, targets)
    for r in sub_results:
        line = (f"  node {r.target_node:3d}: subgraph {r.n_sub_nodes:3d} nodes / "
                f"{r.n_sub_edges:3d} edges, predicates {r.output.nVar}")
        if small:
            slb, sub_ub = r.target_ranges()
            exact = max(float(np.max(np.abs(slb - lb[r.target_node]))),
                        float(np.max(np.abs(sub_ub - ub[r.target_node]))))
            line += f", |subgraph - full| = {exact:.1e}"
        print(line)

    # ---- local robustness, per target via subgraphs (the scalable path) ---
    # |f(X')[t] - f(X)[t]| <= delta for every X' in the eps-box.
    Y_nom = net.evaluate(X)
    print(f"\nlocal robustness: target outputs within +/- {args.delta} of f(X)")
    for r in sub_results:
        n_sub, F_out = r.output.N, r.output.F
        spec_lb, spec_ub = np.full((n_sub, F_out), -np.inf), np.full((n_sub, F_out), np.inf)
        spec_lb[r.target_local_idx] = Y_nom[r.target_node] - args.delta
        spec_ub[r.target_local_idx] = Y_nom[r.target_node] + args.delta
        verdict = verify_node_bounds(r.outputs, spec_lb, spec_ub,
                                     target_nodes=[r.target_local_idx])
        print(f"  node {r.target_node:3d}: {verdict}")

    # ---- verify-or-falsify on a deliberately-too-tight spec (small graphs) --
    if small:
        Yc = net.evaluate(X)
        res = net.verify(gs_in, Yc + 0.05, Yc + 100.0)   # excludes reachable center
        msg = res.status
        if res.status == "falsified":
            cx = res.counterexample
            msg += f" (counterexample found, violation margin {cx.margin:.3f})"
        print(f"\nverify+falsify on an unsatisfiable spec: {msg}")

    # ---- soundness check (full-graph box on small graphs only) -------------
    if not small:
        print("\nfull-graph soundness check skipped at scale (per-target reach is exact)")
        return 0
    rng = np.random.default_rng(args.seed)
    misses = sum(
        int(np.any((y := net.evaluate(X + rng.uniform(-args.eps, args.eps, size=X.shape)))
                   < lb - 1e-5) or np.any(y > ub + 1e-5))
        for _ in range(args.samples)
    )
    if misses == 0:
        print(f"\nsoundness: all {args.samples} samples contained")
        return 0
    print(f"\nWARNING: {misses}/{args.samples} samples outside the reach box")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

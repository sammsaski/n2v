"""Probabilistic verification of a single ACAS Xu VNN-COMP instance.

Phase 2 smoke test. Given a ``(network_id, property_id)`` pair, loads
the ONNX network and VNN-LIB spec, runs the flow-conformal verification
pipeline, and prints the verdict + certificate parameters.

Phase 3 will sweep across all 186 instances; this script is the single-
instance driver that Phase 3 calls in parallel.

Usage:
    python -m examples.FlowConformal.benchmarks.test_acasxu_single \
        --network 1_1 --property 1

Defaults to (1_1, 1) — the simplest case.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from examples.FlowConformal.benchmarks._common import run_verification_pipeline
from n2v.sets.halfspace import HalfSpace
from n2v.utils import load_vnnlib
from n2v.utils.model_loader import load_onnx


_ACASXU_ROOT = Path(__file__).resolve().parents[2] / 'ACASXu'
_ONNX_DIR = _ACASXU_ROOT / 'onnx'
_VNNLIB_DIR = _ACASXU_ROOT / 'vnnlib'


class _ACASXuWrapper(nn.Module):
    """Wrap an ACAS Xu ONNX-loaded net so it accepts plain ``(batch, 5)``
    input. The upstream ONNX is exported with a ``(batch, 1, 1, 5)`` layout
    (two singleton spatial dims for a framework-specific flatten), so
    callers that produce flat batches need an adapter.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept both (batch, 5) and (batch, 1, 1, 5).
        if x.dim() == 2:
            x = x.unsqueeze(1).unsqueeze(1)  # (batch, 5) -> (batch, 1, 1, 5)
        y = self.inner(x)
        # Outputs come out (batch, 5); keep as-is.
        return y


def _extract_spec(prop_field):
    """Normalize the ``prop['prop']`` field to a single HalfSpace.

    ``load_vnnlib`` returns ``prop`` as ``list[dict]`` where each dict has
    keys ``{'dim', 'Hg', 'H', 'g'}`` — the HalfSpace is under ``Hg``.
    For Phase 2 we support only the len-1 list case (one conjunct);
    OR-of-ANDs is deferred.
    """
    if isinstance(prop_field, list):
        if len(prop_field) != 1:
            raise NotImplementedError(
                f'OR-of-ANDs output specs (len {len(prop_field)} conjuncts) '
                'are deferred to Phase 3.')
        entry = prop_field[0]
        if isinstance(entry, dict) and 'Hg' in entry:
            return entry['Hg']
        if isinstance(entry, HalfSpace):
            return entry
        raise TypeError(
            f'unexpected entry in prop list: {type(entry).__name__}')
    if isinstance(prop_field, HalfSpace):
        return prop_field
    raise TypeError(f'unsupported prop field type: {type(prop_field).__name__}')


def _resolve_network(network_id: str) -> Path:
    """Map '1_1' (or 'N_1_1') to the corresponding ONNX file."""
    nid = network_id.strip().lstrip('N_')
    return _ONNX_DIR / f'ACASXU_run2a_{nid}_batch_2000.onnx'


def _resolve_property(property_id: int) -> Path:
    return _VNNLIB_DIR / f'prop_{property_id}.vnnlib'


def main():
    parser = argparse.ArgumentParser(
        description='Probabilistic verification of a single ACAS Xu instance.')
    parser.add_argument('--network', default='1_1',
                        help='ACAS Xu network id (e.g. "1_1").')
    parser.add_argument('--property', type=int, default=1,
                        help='ACAS Xu property number (1-10).')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='Conformal miscoverage.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--flow-epochs', type=int, default=5000,
                        help='Flow training epochs.')
    parser.add_argument('--n-train', type=int, default=10_000)
    parser.add_argument('--scenario-n', type=int, default=10_000)
    args = parser.parse_args()

    onnx_path = _resolve_network(args.network)
    vnn_path = _resolve_property(args.property)
    if not onnx_path.exists():
        print(f'network not found: {onnx_path}', file=sys.stderr)
        sys.exit(2)
    if not vnn_path.exists():
        print(f'property not found: {vnn_path}', file=sys.stderr)
        sys.exit(2)

    print(f'Loading network  {onnx_path.name}')
    network = load_onnx(str(onnx_path))
    network.eval()
    # ACAS Xu ONNX nets want (batch, 1, 1, 5). Wrap so callers can pass
    # plain (batch, 5) — matches what sample_box / preimage_search produce.
    network = _ACASXuWrapper(network)

    print(f'Loading property {vnn_path.name}')
    prop = load_vnnlib(str(vnn_path))
    # If prop_k uses OR-of-ANDs input regions, `lb`/`ub` may come back
    # as a list — flag that and bail in Phase 2.
    if isinstance(prop['lb'], list) or isinstance(prop['ub'], list):
        print('OR-of-ANDs input regions not supported in Phase 2', file=sys.stderr)
        sys.exit(3)
    input_lb = np.asarray(prop['lb']).flatten()
    input_ub = np.asarray(prop['ub']).flatten()
    try:
        spec = _extract_spec(prop['prop'])
    except NotImplementedError as e:
        print(str(e), file=sys.stderr)
        sys.exit(3)

    print(f'Input box:  lb={input_lb.tolist()}')
    print(f'            ub={input_ub.tolist()}')
    print(f'Spec:       {spec}')
    print()

    print('Running verification pipeline...')
    t0 = time.time()
    result = run_verification_pipeline(
        network=network,
        input_lb=input_lb,
        input_ub=input_ub,
        spec=spec,
        alpha=args.alpha,
        n_train=args.n_train, flow_epochs=args.flow_epochs,
        scenario_n_samples=args.scenario_n,
        seed=args.seed,
    )
    total = time.time() - t0

    print()
    print('=' * 60)
    print(f'Instance:      ACAS Xu N_{args.network}  prop_{args.property}')
    print(f'Verdict:       {result["verdict"]}')
    print(f'spec:          {result["spec_summary"]}')
    print(f'coverage:      {result["coverage_empirical"]:.4f} (target >= {1 - args.alpha:.4f})')
    print(f'eps_total:     {result["epsilon_total"]:.4f}')
    print(f'delta_total:   {result["delta_total"]:.4f}  '
          f'(eps1={result["epsilon_1"]:.4f} d1={result["delta_1"]:.4f}, '
          f'eps2={result["epsilon_2"]:.4f} d2={result["delta_2"]:.4f})')
    print(f'train(s):      {result["flow_train_time_s"]:.1f}')
    print(f'verify(s):     {result["verification_time_s"]:.1f}')
    print(f'total(s):      {total:.1f}')
    if result['counterexample'] is not None:
        print(f'counterexample: x={result["counterexample"]["x"]}')
        print(f'                y={result["counterexample"]["y"]}')


if __name__ == '__main__':
    main()

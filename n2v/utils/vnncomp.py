"""
VNN-COMP benchmark infrastructure for n2v.

Provides:
- ReachOptions: Dataclass capturing a single verification attempt configuration
- analyze_difficulty(): Estimate verification complexity from pre-pass bounds
- get_benchmark_config(): Per-benchmark reachability configurations
- verify_regions_parallel(): Parallel verification of disjunctive input regions
"""

import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn

from n2v.sets import Star, HalfSpace
from n2v.sets.image_star import ImageStar


# =============================================================================
# Strategy: ReachOptions, difficulty analysis, benchmark configs
# =============================================================================

@dataclass
class ReachOptions:
    """Configuration for a single verification attempt."""
    method: str                         # 'exact', 'approx', 'relax-star-area', etc.
    relax_factor: Optional[float] = None
    relax_method: str = 'standard'
    precompute_bounds: bool = False
    parallel: bool = True
    n_workers: Optional[int] = None
    lp_solver: str = 'linprog'
    extra_kwargs: dict = field(default_factory=dict)

    def to_kwargs(self) -> dict:
        """Convert to kwargs dict for reach_pytorch_model()."""
        kw = {
            'lp_solver': self.lp_solver,
            'precompute_bounds': self.precompute_bounds,
        }
        if self.relax_factor is not None:
            kw['relax_factor'] = self.relax_factor
        if self.relax_method != 'standard':
            kw['relax_method'] = self.relax_method
        kw.update(self.extra_kwargs)
        return kw


def analyze_difficulty(
    layer_bounds: Dict[Union[int, str], Tuple[np.ndarray, np.ndarray]],
) -> dict:
    """
    Analyze network verification difficulty from pre-pass bounds.

    Counts stable (always active/inactive) and uncertain (crossing zero)
    neurons at each nonlinear layer.

    Args:
        layer_bounds: Dict from compute_intermediate_bounds().
            Maps layer_id -> (lb, ub) numpy arrays.

    Returns:
        Dictionary with:
        - 'per_layer': list of dicts with {layer_id, total, stable_active,
          stable_inactive, uncertain}
        - 'total_uncertain': sum of uncertain neurons across all layers
        - 'max_layer_uncertain': max uncertain in any single layer
        - 'estimated_exact_cost': rough string estimate ('low', 'medium', 'high')
    """
    per_layer = []
    total_uncertain = 0
    max_layer_uncertain = 0

    for layer_id, (lb, ub) in layer_bounds.items():
        lb_flat = lb.flatten()
        ub_flat = ub.flatten()
        total = len(lb_flat)

        stable_active = int(np.sum(lb_flat >= 0))
        stable_inactive = int(np.sum(ub_flat <= 0))
        uncertain = total - stable_active - stable_inactive

        per_layer.append({
            'layer_id': layer_id,
            'total': total,
            'stable_active': stable_active,
            'stable_inactive': stable_inactive,
            'uncertain': uncertain,
        })

        total_uncertain += uncertain
        max_layer_uncertain = max(max_layer_uncertain, uncertain)

    # Rough cost estimate
    if max_layer_uncertain <= 5:
        cost = 'low'
    elif max_layer_uncertain <= 20:
        cost = 'medium'
    else:
        cost = 'high'

    return {
        'per_layer': per_layer,
        'total_uncertain': total_uncertain,
        'max_layer_uncertain': max_layer_uncertain,
        'estimated_exact_cost': cost,
    }


def get_benchmark_config(category: str) -> List[ReachOptions]:
    """
    Return ordered list of ReachOptions for a benchmark category.

    The verification runner should try each configuration in order,
    stopping at the first conclusive result.

    Args:
        category: Benchmark name (e.g., 'acasxu', 'safenlp').

    Returns:
        Ordered list of ReachOptions to try.
    """
    category = category.lower()

    configs = {
        'acasxu': [
            ReachOptions(method='approx', precompute_bounds=True),
            ReachOptions(method='exact', precompute_bounds=True),
        ],
        'safenlp': [
            ReachOptions(method='approx', precompute_bounds=True),
            ReachOptions(method='exact', precompute_bounds=True),
        ],
        'tllverify': [
            ReachOptions(
                method='approx', relax_factor=0.9,
                relax_method='area', precompute_bounds=True,
            ),
            ReachOptions(method='approx', precompute_bounds=True),
        ],
        'sat_relu': [
            ReachOptions(method='approx', precompute_bounds=True),
            ReachOptions(method='exact', precompute_bounds=True),
        ],
        'cersyve': [
            ReachOptions(method='approx', precompute_bounds=True),
        ],
        'malbeware': [
            ReachOptions(method='approx', precompute_bounds=True),
            ReachOptions(method='exact', precompute_bounds=True),
        ],
        'metaroom': [
            ReachOptions(method='approx', precompute_bounds=True),
        ],
        'cora': [
            ReachOptions(
                method='approx', relax_factor=0.5,
                relax_method='area', precompute_bounds=True,
            ),
            ReachOptions(method='approx', precompute_bounds=True),
        ],
        'cifar100': [
            ReachOptions(method='approx', precompute_bounds=True),
        ],
    }

    # Default: approx then exact
    return configs.get(category, [
        ReachOptions(method='approx', precompute_bounds=True),
        ReachOptions(method='exact', precompute_bounds=True),
    ])


# =============================================================================
# Parallel region verification
# =============================================================================

def _verify_single_region(args):
    """Worker function for multiprocessing. Verifies one input region.

    Args:
        args: tuple of (model, lb, ub, property_spec, method, kwargs)

    Returns:
        dict with 'result' (int: 0=sat, 1=unsat, 2=unknown) and optional 'counterexample'
    """
    model, lb, ub, property_spec, method, kwargs = args

    try:
        from n2v.nn import NeuralNetwork
        from n2v.utils.verify_specification import verify_specification

        input_set = Star.from_bounds(
            lb.reshape(-1, 1),
            ub.reshape(-1, 1),
        )

        net = NeuralNetwork(model)
        reach_sets = net.reach(input_set, method=method, **kwargs)
        verdict = verify_specification(reach_sets, property_spec)

        return {'result': verdict, 'counterexample': None}

    except Exception as e:
        return {'result': 2, 'counterexample': None, 'error': str(e)}


def verify_regions_parallel(
    model: nn.Module,
    regions: List[Tuple[np.ndarray, np.ndarray]],
    property_spec,
    method: str = 'approx',
    n_workers: Optional[int] = None,
    **kwargs,
) -> Dict:
    """
    Verify multiple input regions in parallel using multiprocessing.

    Args:
        model: PyTorch model (nn.Module).
        regions: List of (lb, ub) tuples defining input regions.
        property_spec: Output property (HalfSpace or list of HalfSpace).
        method: Reachability method ('approx', 'exact', etc.).
        n_workers: Number of parallel workers (default: cpu_count).
        **kwargs: Additional kwargs passed to reach().

    Returns:
        Dictionary with:
        - 'result': 'sat' | 'unsat' | 'unknown'
        - 'counterexample': (input, output) if SAT, else None
        - 'per_region': list of per-region result dicts
    """
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), len(regions))

    # For single region or single worker, run sequentially
    if len(regions) <= 1 or n_workers <= 1:
        return _verify_regions_sequential(model, regions, property_spec, method, **kwargs)

    # Prepare worker arguments
    # Note: model is passed directly -- multiprocessing will pickle it
    worker_args = [
        (model, lb, ub, property_spec, method, kwargs)
        for lb, ub in regions
    ]

    per_region = []

    # Use spawn context for safety with torch
    ctx = multiprocessing.get_context('spawn')

    try:
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_verify_single_region, worker_args)

        per_region = results

    except Exception:
        # Fallback to sequential on any multiprocessing error
        return _verify_regions_sequential(model, regions, property_spec, method, **kwargs)

    return _aggregate_results(per_region)


def _verify_regions_sequential(model, regions, property_spec, method, **kwargs):
    """Sequential fallback for region verification."""
    per_region = []
    for lb, ub in regions:
        result = _verify_single_region(
            (model, lb, ub, property_spec, method, kwargs)
        )
        per_region.append(result)

        # Early termination on SAT
        if result['result'] == 0:
            return {
                'result': 'sat',
                'counterexample': result.get('counterexample'),
                'per_region': per_region,
            }

    return _aggregate_results(per_region)


def _aggregate_results(per_region: List[Dict]) -> Dict:
    """Aggregate per-region results into overall verdict."""
    verdicts = [r['result'] for r in per_region]

    if 0 in verdicts:
        # Any SAT -> overall SAT
        sat_idx = verdicts.index(0)
        return {
            'result': 'sat',
            'counterexample': per_region[sat_idx].get('counterexample'),
            'per_region': per_region,
        }
    elif all(v == 1 for v in verdicts):
        # All UNSAT -> overall UNSAT
        return {
            'result': 'unsat',
            'counterexample': None,
            'per_region': per_region,
        }
    else:
        # Some UNKNOWN -> overall UNKNOWN
        return {
            'result': 'unknown',
            'counterexample': None,
            'per_region': per_region,
        }

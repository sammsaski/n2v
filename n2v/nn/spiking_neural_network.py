"""
SpikingNeuralNetwork wrapper for formal verification.

Wraps F2FMLP (snntorch-based) models to enable reachability analysis using
set-based methods (Star, Box), mirroring the NeuralNetwork interface.

Loading a trained SNN:
    model = torch.load("path/to/snn_model.pt")   # saved by SNNVerifier.train()
    snn = SpikingNeuralNetwork(model)
    output_boxes = snn.reach(input_set, method='approx')

The model file is the full F2FMLP object (not a state-dict checkpoint). It is
produced by SNNVerifier.train(), which saves it alongside snn_checkpoint.pt.

reach() returns List[Box] (not List[Star]) because the SNN verification LP
produces axis-aligned output score bounds — exactly a Box per output class.
The layers property is not available for F2FMLP due to its time-loop structure
(torch.fx.symbolic_trace cannot trace data-dependent at-most-once masking).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from n2v.config import config as global_config
from n2v.sets.box import Box
from n2v.sets.star import Star
from n2v.snn.encoding import encode_batch
from n2v.snn.lp import (
    build_symbolic_relaxation_lp,
    build_symbolic_relaxation_lp_split,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SNNReachConfig:
    """Configuration for SpikingNeuralNetwork.reach().

    Attributes:
        method:           'approx' (depth-0 LP relaxation) or 'exact' (full latency
                          enumeration via branch-and-bound). Mirrors ReachConfig.method.
        parallel_workers: Number of worker threads for LP solving. 0 defers to the
                          global n2v config (n2v.set_parallel / n2v.config).
        singleton_bounds: Add equality constraints for neurons with a single feasible
                          firing time (can improve tightness).
        split_strategy:   How to order dimensions for the branch-and-bound split.
                          One of: 'selected', 'influence', 'choice', 'choice-influence',
                          'random'.
        label:            Optional true class label. When provided, the LP also
                          computes the certification gap (score[label] - score[competitor]).
    """
    method: Literal['approx', 'exact'] = 'approx'
    parallel_workers: int = 0
    singleton_bounds: bool = False
    split_strategy: str = 'choice-influence'
    label: Optional[int] = None

    def __post_init__(self):
        if self.method not in ('approx', 'exact'):
            raise ValueError(f"method must be 'approx' or 'exact', got {self.method!r}")
        if self.parallel_workers < 0:
            raise ValueError(f"parallel_workers must be >= 0, got {self.parallel_workers}")
        valid_strategies = {'selected', 'influence', 'choice', 'choice-influence', 'random'}
        if self.split_strategy not in valid_strategies and not self.split_strategy.startswith('split_'):
            raise ValueError(
                f"split_strategy must be one of {valid_strategies}, got {self.split_strategy!r}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_snn_reach_config(method: str, config: Optional[SNNReachConfig], **kwargs):
    """Validate and build an SNNReachConfig from method + optional config + kwargs."""
    if config is not None and kwargs:
        raise TypeError("Pass either config= or keyword arguments, not both")
    if config is not None:
        return config  # config is the authority; config.method overrides the method param
    return SNNReachConfig(method=method, **kwargs)


def _set_to_bounds(input_set, lp_solver: str = 'default',
                   parallel: bool = False, n_workers: int = 1):
    """Extract per-dimension (lb, ub) arrays from a Star or Box input set.

    For Star: solves 2*dim LPs to get the exact axis-aligned bounding box.
    For Box: reads lb/ub directly (no LP needed).

    Returns flat float64 arrays of shape (dim,) each.
    """
    if isinstance(input_set, Star):
        lb, ub = input_set.get_ranges(
            lp_solver=lp_solver, parallel=parallel, n_workers=n_workers
        )
    elif isinstance(input_set, Box):
        lb, ub = input_set.lb, input_set.ub
    else:
        raise TypeError(
            f"input_set must be a Star or Box, got {type(input_set).__name__}. "
            f"SpikingNeuralNetwork.reach() does not support {type(input_set).__name__}."
        )
    return lb.flatten().astype(np.float64), ub.flatten().astype(np.float64)


# ---------------------------------------------------------------------------
# SpikingNeuralNetwork
# ---------------------------------------------------------------------------

class SpikingNeuralNetwork:
    """
    Spiking Neural Network wrapper for formal verification.

    Wraps an F2FMLP (snntorch) model to enable reachability analysis using
    set-based methods (Star, Box), with the same interface as NeuralNetwork.

    Attributes:
        model:      The underlying F2FMLP nn.Module.
        input_size: Expected flat input size (inferred from model if not given).
        output_size: Number of output classes.

    The layers property is not available for F2FMLP: its time loop and
    at-most-once masking are not traceable by torch.fx.symbolic_trace.
    Only reach() and forward() are supported.
    """

    def __init__(self, model: nn.Module, input_size: Optional[int] = None) -> None:
        """
        Initialize SpikingNeuralNetwork wrapper.

        Args:
            model:      An F2FMLP instance (or any nn.Module with the same interface).
            input_size: Expected flat input size. Inferred from model.fcs[0].in_features
                        if not provided.

        Raises:
            TypeError: If model is not an nn.Module.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be a PyTorch nn.Module")

        self.model = model
        self.model.eval()

        # Infer input/output sizes from the model architecture.
        if hasattr(model, 'fcs') and len(model.fcs) > 0:
            self.input_size = model.fcs[0].in_features
            self.output_size = (model.fcs[-1].out_features,)
        else:
            self.input_size = input_size
            self.output_size = None

        if input_size is not None:
            if self.input_size is not None and input_size != self.input_size:
                raise ValueError(
                    f"input_size={input_size} conflicts with model's "
                    f"inferred input_size={self.input_size}"
                )
            self.input_size = input_size
            self._validate_input_size(input_size)

    def _validate_input_size(self, input_size: int) -> None:
        """Verify input_size matches the model's first layer."""
        if not hasattr(self.model, 'fcs') or len(self.model.fcs) == 0:
            return
        expected = self.model.fcs[0].in_features
        if input_size != expected:
            raise ValueError(
                f"input_size={input_size} does not match model's first layer "
                f"input size {expected}"
            )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass: convert inputs to spike train, return class scores.

        Args:
            x: Input tensor of shape (B, input_size) with values in [0, 1].

        Returns:
            Class scores of shape (B, num_classes).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if not hasattr(self.model, 'num_steps'):
            raise AttributeError("model must have a num_steps attribute (F2FMLP)")
        spikes = encode_batch(x, self.model.num_steps)
        return self.model(spikes)

    def reach(
        self,
        input_set: Union[Star, Box],
        method: Literal['approx', 'exact'] = 'exact',
        config: Optional[SNNReachConfig] = None,
        **kwargs,
    ) -> List[Box]:
        """Compute a reachable output set for the given input set.

        Args:
            input_set: A Star or Box representing the set of possible inputs.
            method:    'approx' — depth-0 LP relaxation (Algorithm 1, d=0).
                       'exact'  — full latency enumeration (Algorithm 1, d=k).
            config:    Optional SNNReachConfig (overrides method and kwargs).
            **kwargs:  SNNReachConfig fields as keyword arguments.

        Returns:
            List[Box] — a single Box giving per-class output score bounds.
            The box has shape (num_classes, 1) for lb and ub.

        The 'approx' result is a sound over-approximation: the true reachable
        output set is contained in the returned Box. 'exact' gives the exact
        reachable output set via complete enumeration of all feasible latency
        combinations.
        """
        cfg = _validate_snn_reach_config(method, config, **kwargs)

        if not hasattr(self.model, 'num_steps'):
            raise AttributeError("model must have a num_steps attribute (F2FMLP)")
        num_steps = self.model.num_steps

        # ---- Extract per-dimension bounds from the input set ----
        # (Must happen first so n_dims is known for parallel worker resolution.)
        lb_arr, ub_arr = _set_to_bounds(
            input_set,
            lp_solver=global_config.lp_solver,
            parallel=global_config.parallel_lp,
            n_workers=global_config.n_workers,
        )

        n_dims = len(lb_arr)
        if self.input_size is not None and n_dims != self.input_size:
            raise ValueError(
                f"input_set has dimension {n_dims} but model expects {self.input_size}"
            )

        # ---- Resolve parallel_workers for the LP enumeration ----
        parallel_workers = cfg.parallel_workers
        if parallel_workers == 0:
            if global_config.should_use_parallel(n_dims):
                parallel_workers = global_config.get_n_workers(n_dims)
            else:
                parallel_workers = 1

        # Build midpoint array as the nominal input (for base_lat computation).
        midpoint = (lb_arr + ub_arr) / 2.0

        # Determine which dimensions are symbolic (have non-zero range).
        symbolic_dims = np.where(ub_arr - lb_arr > 1e-12)[0]
        n_symbolic = len(symbolic_dims)

        # ---- Call the LP engine ----
        if cfg.method == 'approx':
            # Depth-0 relaxation: single LP over all symbolic dimensions.
            result = build_symbolic_relaxation_lp(
                self.model,
                image_flat=midpoint,
                epsilon=0.0,              # unused; input_bounds overrides this
                k=n_symbolic,
                num_steps=num_steps,
                tight_bounds=False,
                label=cfg.label,
                cert_only=False,
                pixel_indices=symbolic_dims,
                singleton_bounds=cfg.singleton_bounds,
                parallel_workers=parallel_workers,
                parallel_backend='thread',  # never 'process' when using input_bounds
                input_bounds=(lb_arr, ub_arr),
            )
        else:
            # Exact: branch-and-bound over all symbolic dimensions (split_depth = k).
            result = build_symbolic_relaxation_lp_split(
                self.model,
                image_flat=midpoint,
                epsilon=0.0,
                k=n_symbolic,
                num_steps=num_steps,
                split_depth=n_symbolic,
                label=cfg.label,
                tight_bounds=False,
                parallel_workers=parallel_workers,
                cert_only=False,
                parallel_backend='thread',
                split_strategy=cfg.split_strategy,
                pixel_indices=symbolic_dims,
                singleton_bounds=cfg.singleton_bounds,
                input_bounds=(lb_arr, ub_arr),
            )

        lb_y = result.get("lb")
        ub_y = result.get("ub")

        if lb_y is None or ub_y is None:
            raise RuntimeError(
                "SNN LP verification failed (infeasible sub-problem). "
                "Try method='approx' or a smaller input set."
            )

        return [Box(lb_y.reshape(-1, 1), ub_y.reshape(-1, 1))]

    def __repr__(self) -> str:
        arch = ""
        if hasattr(self.model, 'hidden_sizes') and hasattr(self.model, 'num_steps'):
            arch = (f", hidden_sizes={self.model.hidden_sizes}, "
                    f"num_steps={self.model.num_steps}")
        return (f"SpikingNeuralNetwork("
                f"input_size={self.input_size}, "
                f"output_size={self.output_size}"
                f"{arch})")

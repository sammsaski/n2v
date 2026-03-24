"""
Neural Network wrapper for verification.

Wraps PyTorch models to enable reachability analysis and verification.
"""

from typing import Union, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.fx as fx

from n2v.nn.reach import _function_node_to_module, reach_pytorch_model


class NeuralNetwork:
    """
    Neural Network wrapper for formal verification.

    Wraps a PyTorch nn.Module to enable reachability analysis using
    set-based methods (Star, Zono, Box).

    Attributes:
        model: PyTorch model
        layers: List of individual layers
        input_size: Expected input size
        output_size: Output size
    """

    def __init__(self, model: nn.Module, input_size: Optional[tuple] = None) -> None:
        """
        Initialize NeuralNetwork wrapper.

        Args:
            model: PyTorch model (nn.Module)
            input_size: Expected input size (excluding batch dim)
        """
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be a PyTorch nn.Module")

        self.model = model
        self.model.eval()  # Set to evaluation mode

        # Extract layers
        self.layers = self._extract_layers(model)

        # Determine input/output sizes
        self.input_size = input_size
        self.output_size = None

        if input_size is not None:
            self._validate_input_size(input_size)

    def _extract_layers(self, model: nn.Module) -> List[nn.Module]:
        """
        Extract individual layers from the model using torch.fx tracing.

        Traces the model's forward() to capture all operations including
        functional calls (e.g., ``F.relu``).

        Args:
            model: PyTorch model

        Returns:
            List of layers

        Raises:
            TypeError: If the model cannot be traced by torch.fx
        """
        # GraphModules (from onnx2torch or prior tracing)
        #   already have a graph
        if isinstance(model, fx.GraphModule):
            return self._extract_layers_from_graph(model)

        try:
            gm = torch.fx.symbolic_trace(model)
            return self._extract_layers_from_graph(gm)
        except Exception as e:
            raise TypeError(
                f"n2v requires models to be traceable by torch.fx. "
                f"Models with data-dependent control flow (e.g., "
                f"'if x.sum() > 0') or inline module instantiation "
                f"(e.g., 'nn.ReLU()(x)') are not supported. "
                f"For inline activations, use functional equivalents "
                f"(e.g., F.relu(x) instead of nn.ReLU()(x)). "
                f"Tracing failed with: {e}"
            ) from e

    def _extract_layers_from_graph(
        self, gm: fx.GraphModule,
    ) -> List[nn.Module]:
        """Extract layers from a torch.fx.GraphModule's graph.

        Walks the graph nodes and collects call_module targets
        plus call_function / call_method ops that map to
        nn.Module equivalents.
        """
        named_modules = dict(gm.named_modules())
        layers = []

        for node in gm.graph.nodes:
            if node.op == 'call_module':
                mod = named_modules.get(node.target)
                if mod is not None:
                    layers.append(mod)
            elif node.op == 'call_function':
                equiv = _function_node_to_module(node)
                if equiv is not None:
                    layers.append(equiv)
            elif node.op == 'call_method':
                if node.target == 'flatten':
                    start_dim = (
                        node.args[1]
                        if len(node.args) > 1
                        else node.kwargs.get('start_dim', 1)
                    )
                    end_dim = (
                        node.args[2]
                        if len(node.args) > 2
                        else node.kwargs.get('end_dim', -1)
                    )
                    layers.append(
                        nn.Flatten(
                            start_dim=start_dim,
                            end_dim=end_dim,
                        )
                    )

        return layers

    def _validate_input_size(self, input_size: tuple) -> None:
        """Validate input size by running a forward pass."""
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, *input_size)
                output = self.model(dummy_input)
                self.output_size = tuple(output.shape[1:])
        except Exception as e:
            raise ValueError(f"Model forward pass failed with input size {input_size}: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        with torch.no_grad():
            return self.model(x)

    def reach(
        self,
        input_set: Union['Star', 'Zono', 'Box', 'Hexatope', 'Octatope'],
        method: str = 'exact',
        **kwargs
    ) -> List:
        """
        Perform reachability analysis.

        This is the primary interface for reachability analysis. It automatically
        dispatches to the appropriate reachability method based on the input set type.

        Args:
            input_set: Input specification (Star, Zono, Box, Hexatope, or Octatope)
            method: Reachability method to use:
                For Star:
                    - 'exact': Exact reachability with splitting
                    - 'approx': Over-approximate reachability with relaxation
                For Box/Zono:
                    - 'approx': Over-approximate reachability (only option)
                For Hexatope/Octatope:
                    - 'approx': Over-approximate reachability
            **kwargs: Additional method-specific arguments:
                For Star 'exact':
                    - lp_solver: LP solver to use (default: 'default')
                    - verbose: 'display' to show progress
                    - parallel: Enable parallel Star processing
                    - n_workers: Number of parallel workers
                For Star 'approx':
                    - relax_factor: Relaxation factor (0=exact, 1=max, default: 0.5)
                    - relax_method: Relaxation strategy (default: 'standard')
                    - lp_solver: LP solver to use
                    - verbose: 'display' to show progress
                For Zono/Box:
                    - verbose: 'display' to show progress

        Returns:
            List of output sets (same type as input_set)

        Example:
            >>> from n2v.nn import NeuralNetwork
            >>> from n2v.sets import Star
            >>> import torch.nn as nn
            >>> model = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 1))
            >>> net = NeuralNetwork(model)
            >>> input_star = Star.from_bounds(lb, ub)
            >>> output_stars = net.reach(input_star, method='exact')
        """
        return reach_pytorch_model(
            self.model,
            input_set,
            method=method,
            **kwargs
        )

    def __repr__(self) -> str:
        return f"NeuralNetwork(layers={len(self.layers)}, input_size={self.input_size})"

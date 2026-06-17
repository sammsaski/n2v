"""
Model loading utilities for PyTorch and ONNX models.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
from pathlib import Path
from collections import OrderedDict

# ONNX support (required dependencies)
import onnx
from onnx2torch import convert

# Opset to upgrade old models to when onnx2torch lacks a converter for
# their version (onnx2torch's converters target modern opsets, ~9-17).
_ONNX2TORCH_TARGET_OPSET = 13

# Shape-computation ops the set-based reach cannot propagate; folded away
# by onnx-simplifier under a fixed input shape when present.
_SHAPE_SUBGRAPH_OPS = {"Shape", "ConstantOfShape"}


def _fold_shape_subgraph(onnx_model):
    """If the model computes shapes with Shape/ConstantOfShape, fix the
    input shape (dynamic dims -> 1) and constant-fold via onnx-simplifier
    so the reach never sees those ops. Returns the original model
    unchanged on any failure or when those ops are absent."""
    ops = {n.op_type for n in onnx_model.graph.node}
    if not (ops & _SHAPE_SUBGRAPH_OPS):
        return onnx_model
    try:
        import onnxsim
        inp = onnx_model.graph.input[0]
        shape = [d.dim_value if d.dim_value > 0 else 1
                 for d in inp.type.tensor_type.shape.dim]
        simplified, ok = onnxsim.simplify(
            onnx_model, overwrite_input_shapes={inp.name: shape})
        if ok and not ({n.op_type for n in simplified.graph.node}
                       & _SHAPE_SUBGRAPH_OPS):
            return simplified
    except Exception:  # noqa: BLE001 — fall back to the original model
        pass
    return onnx_model


def load_pytorch(
    model_path: Optional[Union[str, Path]] = None,
    model: Optional[nn.Module] = None,
    input_shape: Optional[tuple] = None,
) -> nn.Module:
    """
    Load a PyTorch model.

    Args:
        model_path: Path to saved model (.pt or .pth file)
        model: Pre-loaded PyTorch model
        input_shape: Expected input shape for validation

    Returns:
        PyTorch model in eval mode

    Raises:
        ValueError: If neither model_path nor model is provided
    """
    if model is not None:
        # Use provided model
        loaded_model = model
    elif model_path is not None:
        # Load from file
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Try loading as state dict or full model
        try:
            loaded_model = torch.load(model_path, map_location='cpu')
            if isinstance(loaded_model, dict):
                # It's a state dict, need model architecture
                raise ValueError(
                    "Loaded a state dict but no model architecture provided. "
                    "Please provide the model object."
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    else:
        raise ValueError("Either model_path or model must be provided")

    # Set to eval mode
    loaded_model.eval()

    # Validate input shape if provided
    if input_shape is not None:
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, *input_shape)
                loaded_model(dummy_input)
        except Exception as e:
            raise ValueError(f"Model failed with provided input_shape {input_shape}: {e}")

    return loaded_model


def load_onnx(
    onnx_path: Union[str, Path],
) -> nn.Module:
    """
    Load an ONNX model and convert to PyTorch.

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        PyTorch model (if backend='pytorch')

    Note:
        Requires onnx and onnx2torch packages.
        Install with: pip install onnx onnx2torch
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Transformer-style models (vit) compute reshape/transpose targets
    # with Shape / ConstantOfShape ops that the set-based reach cannot
    # propagate. With a fixed batch they are pure shape arithmetic and
    # constant-fold away; onnx-simplifier does this. Gated on those ops
    # being present so ordinary models are untouched.
    onnx_model = _fold_shape_subgraph(onnx_model)

    # Convert to PyTorch. onnx2torch registers converters per opset
    # version; old models (e.g. vgg16-7 is opset 8, but onnx2torch's Gemm
    # only covers 9/11/13) raise "Converter is not implemented (... Gemm,
    # version=8)". Upgrade such models to a supported opset and retry —
    # an n2v-side compatibility shim, not a change to the converter.
    try:
        pytorch_model = convert(onnx_model)
    except NotImplementedError as exc:
        if "Converter is not implemented" not in str(exc):
            raise
        try:
            from onnx import version_converter
            upgraded = version_converter.convert_version(
                onnx_model, _ONNX2TORCH_TARGET_OPSET)
        except Exception as up_exc:  # noqa: BLE001
            raise NotImplementedError(
                f"{exc}; opset upgrade to {_ONNX2TORCH_TARGET_OPSET} also "
                f"failed: {up_exc}") from exc
        pytorch_model = convert(upgraded)
    pytorch_model.eval()

    return pytorch_model


def get_model_summary(model: nn.Module, input_shape: tuple) -> dict:
    """
    Get summary information about a PyTorch model.

    Args:
        model: PyTorch model
        input_shape: Input shape (excluding batch dimension)

    Returns:
        Dictionary with model information
    """
    summary = OrderedDict()
    hooks = []

    def register_hook(module):
        """Register a forward hook on a module to capture shape info."""
        def hook(module, input, output):
            """Forward hook that records input/output shapes and param counts."""
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size()) if input else []
            summary[m_key]["output_shape"] = list(output.size()) if hasattr(output, 'size') else []

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size()))).item()
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size()))).item()
            summary[m_key]["nb_params"] = params

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))

    # Register hooks
    model.apply(register_hook)

    # Run forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        model(dummy_input)

    # Remove hooks
    for h in hooks:
        h.remove()

    return dict(summary)

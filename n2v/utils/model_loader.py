"""
Model loading utilities for PyTorch and ONNX models.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
from pathlib import Path
from collections import OrderedDict

# Optional ONNX support
try:
    import onnx
    from onnx2torch import convert
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


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
    if not ONNX_AVAILABLE:
        raise ImportError(
            "ONNX support requires 'onnx' and 'onnx2torch' packages. "
            "Install with: pip install onnx onnx2torch"
        )

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Convert to PyTorch
    pytorch_model = convert(onnx_model)
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

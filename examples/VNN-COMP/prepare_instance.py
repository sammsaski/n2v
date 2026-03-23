"""
ONNX model introspection and input set creation for VNN-COMP.

Provides utilities to:
- Detect input shape from ONNX model
- Create appropriate n2v input sets (Star or ImageStar) from VNNLIB bounds
- Load and prepare a complete verification instance
"""

import numpy as np

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def get_input_shape(onnx_path: str) -> tuple:
    """
    Extract input tensor shape from an ONNX model, stripping the batch dimension.

    Args:
        onnx_path: Path to the ONNX model file

    Returns:
        Tuple of input dimensions, e.g. (5,) for FC or (1, 28, 28) for CNN

    Raises:
        ImportError: If onnx package is not installed
        FileNotFoundError: If onnx_path doesn't exist
    """
    if not ONNX_AVAILABLE:
        raise ImportError("onnx package is required: pip install onnx")

    model = onnx.load(onnx_path)

    # Filter out initializers (weights) — only keep true inputs
    init_names = {init.name for init in model.graph.initializer}
    true_inputs = [inp for inp in model.graph.input if inp.name not in init_names]

    if not true_inputs:
        raise ValueError(f"No true input tensors found in {onnx_path}")

    input_tensor = true_inputs[0]
    dims = input_tensor.type.tensor_type.shape.dim

    # Strip batch dimension (first dim), extract remaining
    shape = []
    for i, d in enumerate(dims):
        if i == 0:
            continue  # skip batch dim
        shape.append(d.dim_value)

    return tuple(shape)


from n2v.sets import Star
from n2v.sets.image_star import ImageStar


def create_input_set(lb: np.ndarray, ub: np.ndarray, input_shape: tuple):
    """
    Create an n2v input set from bounds and detected input shape.

    Args:
        lb: Lower bounds, 1D numpy array of length prod(input_shape)
        ub: Upper bounds, 1D numpy array of length prod(input_shape)
        input_shape: Shape from get_input_shape(), e.g. (5,) or (1, 28, 28)

    Returns:
        Star for flat inputs (1D or 2D shape),
        ImageStar for spatial inputs (3D shape: C, H, W)
    """
    lb = np.asarray(lb, dtype=np.float64).flatten()
    ub = np.asarray(ub, dtype=np.float64).flatten()

    if len(input_shape) == 3:
        # Spatial input (C, H, W) -> ImageStar
        C, H, W = input_shape
        lb_col = lb.reshape(-1, 1)
        ub_col = ub.reshape(-1, 1)
        return ImageStar.from_bounds(lb_col, ub_col, height=H, width=W, num_channels=C)
    else:
        # Flat input (1D or 2D flattened) -> Star
        lb_col = lb.reshape(-1, 1)
        ub_col = ub.reshape(-1, 1)
        return Star.from_bounds(lb_col, ub_col)


from n2v.utils import load_onnx, load_vnnlib


def load_and_prepare(onnx_path: str, vnnlib_path: str) -> dict:
    """
    Load ONNX model and VNNLIB spec, create input sets.

    Args:
        onnx_path: Path to ONNX model file
        vnnlib_path: Path to VNNLIB specification file

    Returns:
        Dictionary with keys:
        - 'model': PyTorch model (torch.nn.Module)
        - 'input_shape': Detected input shape tuple
        - 'regions': List of dicts, each with 'lb', 'ub', 'input_set'
        - 'property_spec': Output property specification from VNNLIB
    """
    model = load_onnx(onnx_path)
    prop = load_vnnlib(vnnlib_path)
    input_shape = get_input_shape(onnx_path)

    lb_raw = prop['lb']
    ub_raw = prop['ub']
    property_spec = prop['prop']

    # Normalize to list of regions
    if not isinstance(lb_raw, list):
        lb_list = [lb_raw]
        ub_list = [ub_raw]
    else:
        lb_list = lb_raw
        ub_list = ub_raw

    regions = []
    for lb, ub in zip(lb_list, ub_list):
        input_set = create_input_set(lb, ub, input_shape)
        regions.append({
            'lb': lb,
            'ub': ub,
            'input_set': input_set,
        })

    return {
        'model': model,
        'input_shape': input_shape,
        'regions': regions,
        'property_spec': property_spec,
    }

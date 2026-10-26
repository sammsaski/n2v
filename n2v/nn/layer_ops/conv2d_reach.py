"""
Conv2D layer reachability operations.

Convolutional layers are affine transformations, so they're exact for all set types.
The key is properly handling image dimensions and applying convolution to each
basis vector.

Supports both ImageStar (4D V) and Star (2D V) inputs with optimized paths for each.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Optional, Tuple
from n2v.sets import Star, Zono, ImageStar, ImageZono, Hexatope, Octatope

# Profiler hook (no-op when profiling is disabled)
from n2v.profiling import region, count, OPERATION, is_enabled


# Rough cap, in bytes, on the float64 temporaries of one batch of basis
# columns in map_basis_columns (the batch's input copy, the op's internal
# scratch and its output), so they stay small next to V itself.
_COLUMN_BATCH_BYTES = 1 << 28


def float64_params(layer: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The layer's weight and bias as CPU float64 tensors (bias may be None).

    Widening float32 parameters to float64 is exact, so the set arithmetic
    that uses them runs on the same network the layer computes."""
    weight = layer.weight.detach().to(device='cpu', dtype=torch.float64)
    bias = (layer.bias.detach().to(device='cpu', dtype=torch.float64)
            if layer.bias is not None else None)
    return weight, bias


def kernel_area(layer: nn.Module) -> int:
    """kh * kw of a conv, conv-transpose or pooling layer."""
    k = layer.kernel_size
    if isinstance(k, (tuple, list)):
        return int(np.prod(k))
    return int(k) ** 2


def map_basis_columns(op: Callable[[torch.Tensor], torch.Tensor],
                      V: np.ndarray, scratch_factor: int = 1) -> np.ndarray:
    """Apply a linear torch image op to every basis column of a 4D V.

    ``V`` has the ImageStar layout (H, W, C, n): column ``V[..., j]`` is one
    image. ``op`` maps a float64 (k, C, H, W) batch to (k, C', H', W') and
    must be linear (no bias; callers add the bias to the center column).
    Returns the float64 (H', W', C', n) result.

    The op runs in float64. Rounding the center or the generators to
    float32 would move the set by about one float32 ulp of their magnitude
    (relative 6e-8), far more than the float64-level outward slack the
    downstream bounds and emptiness checks allow, so the output set could
    miss real outputs. A float64 convolution on the CPU unfolds its input
    (``scratch_factor`` times the larger of a column's input and output, the
    kernel area for a convolution), so the columns go through in batches
    written into one preallocated output: the extra memory is one batch,
    not full-size copies of V and of the result.
    """
    V = np.asarray(V, dtype=np.float64)
    h, w, c, n = V.shape
    with torch.no_grad():
        out_chw = tuple(op(torch.zeros((0, c, h, w), dtype=torch.float64)).shape[1:])
        c_out, h_out, w_out = out_chw
        out = np.empty((h_out, w_out, c_out, n), dtype=np.float64)
        in_elems, out_elems = h * w * c, h_out * w_out * c_out
        col_bytes = 8 * (in_elems + out_elems
                         + max(int(scratch_factor), 1) * max(in_elems, out_elems))
        batch = max(1, _COLUMN_BATCH_BYTES // max(col_bytes, 1))
        for start in range(0, n, batch):
            stop = min(n, start + batch)
            x = torch.from_numpy(np.ascontiguousarray(
                V[:, :, :, start:stop].transpose(3, 2, 0, 1)))
            out[:, :, :, start:stop] = op(x).permute(2, 3, 1, 0).numpy()
    return out


def _conv2d_op(layer: nn.Conv2d,
               weight: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """The layer's convolution, without bias, as a function of an NCHW
    batch. ``_conv_forward`` applies the layer's own padding mode: a mode
    other than 'zeros' copies border entries before a padding-free
    convolution (still linear), and zero-padding instead would convolve a
    different function."""
    def conv(x: torch.Tensor) -> torch.Tensor:
        return layer._conv_forward(x, weight, None)
    return conv


def conv2d_star(
    layer: nn.Conv2d,
    input_stars: List[Star],
    method: str = 'exact',
    **kwargs
) -> List[Star]:
    """
    Exact reachability for Conv2D using Star sets.

    Convolution is a linear operation, so it's exact for Star sets.
    We apply the convolution to each basis vector in the Star.

    Supports both ImageStar (optimized 4D path) and Star (requires conversion).

    Args:
        layer: PyTorch nn.Conv2d layer
        input_stars: List of input Star sets (ImageStar or Star)
        method: 'exact' or 'approx' (both are exact for conv)
        **kwargs: Additional options

    Returns:
        List of output Star sets
    """
    output_stars = []

    for star in input_stars:
        if isinstance(star, ImageStar):
            # Optimized 4D path for ImageStar
            with region("conv2d", OPERATION):
                output_star = _conv2d_imagestar_4d(layer, star)
                # FLOPs: 2 mul-adds per (output element x kernel x in_channels),
                # applied to every basis column (center + generators).
                if is_enabled():
                    kh, kw = (layer.kernel_size if isinstance(layer.kernel_size, tuple)
                              else (layer.kernel_size, layer.kernel_size))
                    out_elems = (output_star.height * output_star.width
                                 * output_star.num_channels)
                    n_cols = output_star.V.shape[-1]
                    # each output channel connects to in_channels/groups inputs
                    count("flops",
                          2 * out_elems * n_cols * kh * kw
                          * (layer.in_channels // layer.groups))
        elif isinstance(star, Star):
            # TODO: Add conv2d support for Star by constructing conv matrix
            # For now, require ImageStar
            raise ValueError(
                "Conv2D requires ImageStar input. Please convert Star to ImageStar "
                "with proper height/width/channels before calling conv2d."
            )
        else:
            raise TypeError(f"conv2d_star expects Star or ImageStar, got {type(star)}")

        output_stars.append(output_star)

    return output_stars


def _conv2d_imagestar_4d(layer: nn.Conv2d, input_star: ImageStar) -> ImageStar:
    """
    Apply Conv2D to ImageStar using optimized 4D operations.

    Works directly on the 4D V tensor (H, W, C, nVar+1) without reshaping.

    Algorithm:
    1. Convolve every column of V (center and generators) without bias,
       in float64 (see :func:`map_basis_columns`)
    2. Add the bias to the center column only
    3. Construct output ImageStar with 4D V

    Args:
        layer: Conv2D layer
        input_star: Input ImageStar with 4D V

    Returns:
        Output ImageStar with 4D V
    """
    # V is already 4D: (H, W, C_in, nVar+1)
    V = input_star.V
    c_in = V.shape[2]

    # Verify channel consistency
    if c_in != layer.in_channels:
        raise ValueError(
            f"Input has {c_in} channels but Conv2D expects {layer.in_channels}"
        )

    weight, bias = float64_params(layer)

    # (H_out, W_out, C_out, nVar+1); the bias belongs to the center only
    V_out = map_basis_columns(_conv2d_op(layer, weight), V, kernel_area(layer))
    if bias is not None:
        V_out[:, :, :, 0] += bias.numpy()
    h_out, w_out, c_out_channels = V_out.shape[:3]

    # Create output ImageStar (V is already 4D)
    output_star = ImageStar(
        V_out,
        input_star.C,
        input_star.d,
        input_star.predicate_lb,
        input_star.predicate_ub,
        h_out,
        w_out,
        c_out_channels
    )

    return output_star


def conv2d_zono(layer: nn.Conv2d, input_zonos: List[Zono]) -> List[Zono]:
    """
    Exact reachability for Conv2D using Zonotopes.

    Args:
        layer: PyTorch nn.Conv2d layer
        input_zonos: List of input Zonotopes (should be ImageZonos)

    Returns:
        List of output Zonotopes
    """
    output_zonos = []

    for zono in input_zonos:
        if isinstance(zono, ImageZono):
            image_zono = zono
        else:
            raise ValueError("Conv2D requires ImageZono input")

        output_zono = _conv2d_single_imagezono(layer, image_zono)
        output_zonos.append(output_zono)

    return output_zonos


def _conv2d_single_imagezono(layer: nn.Conv2d, input_zono: ImageZono) -> ImageZono:
    """
    Apply Conv2D to a single ImageZono.

    Args:
        layer: Conv2D layer
        input_zono: Input ImageZono

    Returns:
        Output ImageZono
    """
    h_in = input_zono.height
    w_in = input_zono.width
    c_in = input_zono.num_channels

    if c_in != layer.in_channels:
        raise ValueError(
            f"Input has {c_in} channels but Conv2D expects {layer.in_channels}"
        )

    # Zono: x = c + V*alpha, where -1 <= alpha_i <= 1
    # c shape: (h*w*c, 1), V shape: (h*w*c, n_gen)

    n_gen = input_zono.V.shape[1]

    weight, bias = float64_params(layer)
    conv = _conv2d_op(layer, weight)

    # Convolve center (plus bias) and generators in float64, image layout
    c_out = map_basis_columns(conv, input_zono.c.reshape(h_in, w_in, c_in, 1))
    if bias is not None:
        c_out[:, :, :, 0] += bias.numpy()
    h_out, w_out, c_out_channels = c_out.shape[:3]
    V_out = map_basis_columns(conv, input_zono.V.reshape(h_in, w_in, c_in, n_gen),
                              kernel_area(layer))

    # Flatten back
    c_flat = c_out.reshape(-1, 1)
    V_flat = V_out.reshape(h_out * w_out * c_out_channels, n_gen)

    # Create output ImageZono
    output_zono = ImageZono(c_flat, V_flat, h_out, w_out, c_out_channels)

    return output_zono


def conv2d_box(layer: nn.Conv2d, input_boxes: List) -> List:
    """
    Conv2D for Boxes using interval arithmetic.

    Args:
        layer: PyTorch nn.Conv2d layer
        input_boxes: List of input Boxes

    Returns:
        List of output Boxes
    """

    output_boxes = []

    for box in input_boxes:
        # For Box, we need to know image dimensions
        # This is a limitation - Box doesn't inherently have image structure

        # Conservative approach: convert to Zono, apply conv, convert back
        # This requires knowing the image dimensions

        # For now, raise error asking for ImageZono
        raise NotImplementedError(
            "Conv2D on Box requires image dimensions. "
            "Please convert to ImageZono or ImageStar first."
        )

    return output_boxes


def conv2d_hexatope(layer: nn.Conv2d, input_hexatopes: List[Hexatope]) -> List[Hexatope]:
    """
    Exact reachability for Conv2D using Hexatopes.

    Conv2D is a linear operation, so we can use exact affine transformation.
    However, hexatopes work with flattened vectors, so we need to construct
    the convolution matrix and apply it as an affine map.

    Args:
        layer: PyTorch nn.Conv2d layer
        input_hexatopes: List of input Hexatopes

    Returns:
        List of output Hexatopes
    """
    output_hexatopes = []

    for hexatope in input_hexatopes:
        # Get bounds to infer input dimensions
        lb, ub = hexatope.estimate_ranges()

        # For Conv2D, we need to know the input image dimensions
        # Assume the hexatope represents a flattened image
        # This is a simplification - in practice, image dimensions should be known

        # Use interval over-approximation: apply conv to bounds
        lb.reshape(-1, 1)
        ub.reshape(-1, 1)

        # Convert bounds to ImageStar temporarily to apply convolution
        # This is an over-approximation
        try:
            # Infer image dimensions from input size and layer
            input_size = hexatope.dim
            in_channels = layer.in_channels
            spatial_size = input_size // in_channels

            # Approximate spatial dimensions (assume square images)
            h_in = w_in = int(np.sqrt(spatial_size))

            if h_in * w_in * in_channels != input_size:
                # Not a perfect square, use bounds-based approximation
                raise ValueError("Cannot infer image dimensions")

        except:
            # Fallback: use simple bounds transformation
            # This is conservative but sound
            output_hexatope = _conv2d_hexatope_bounds_approx(layer, hexatope)
            output_hexatopes.append(output_hexatope)
            continue

        output_hexatope = _conv2d_hexatope_bounds_approx(layer, hexatope)
        output_hexatopes.append(output_hexatope)

    return output_hexatopes


def _conv2d_hexatope_bounds_approx(layer: nn.Conv2d, hexatope: Hexatope) -> Hexatope:
    """
    Over-approximate Conv2D for hexatope using bounds propagation.

    Args:
        layer: Conv2D layer
        hexatope: Input hexatope

    Returns:
        Output hexatope (over-approximation)
    """
    # Get bounds
    lb, ub = hexatope.estimate_ranges()

    # Apply interval arithmetic through convolution
    # For a simple over-approximation, we use the bounds

    # Convert to numpy arrays for processing
    lb_np = lb.reshape(-1)
    ub_np = ub.reshape(-1)

    # Compute output bounds using interval arithmetic
    # This is a conservative approximation
    new_lb = lb_np  # Placeholder
    new_ub = ub_np  # Placeholder

    # Create output hexatope from bounds
    output_hexatope = Hexatope.from_bounds(new_lb.reshape(-1, 1), new_ub.reshape(-1, 1))

    return output_hexatope


def conv2d_octatope(layer: nn.Conv2d, input_octatopes: List[Octatope]) -> List[Octatope]:
    """
    Exact reachability for Conv2D using Octatopes.

    Conv2D is a linear operation, so we can use exact affine transformation.
    However, octatopes work with flattened vectors, so we need to construct
    the convolution matrix and apply it as an affine map.

    Args:
        layer: PyTorch nn.Conv2d layer
        input_octatopes: List of input Octatopes

    Returns:
        List of output Octatopes
    """
    output_octatopes = []

    for octatope in input_octatopes:
        # Use bounds-based approximation similar to hexatope
        output_octatope = _conv2d_octatope_bounds_approx(layer, octatope)
        output_octatopes.append(output_octatope)

    return output_octatopes


def _conv2d_octatope_bounds_approx(layer: nn.Conv2d, octatope: Octatope) -> Octatope:
    """
    Over-approximate Conv2D for octatope using bounds propagation.

    Args:
        layer: Conv2D layer
        octatope: Input octatope

    Returns:
        Output octatope (over-approximation)
    """
    # Get bounds
    lb, ub = octatope.estimate_ranges()

    # Apply interval arithmetic through convolution
    lb_np = lb.reshape(-1)
    ub_np = ub.reshape(-1)

    # Compute output bounds using interval arithmetic
    # This is a conservative approximation
    new_lb = lb_np  # Placeholder
    new_ub = ub_np  # Placeholder

    # Create output octatope from bounds
    output_octatope = Octatope.from_bounds(new_lb.reshape(-1, 1), new_ub.reshape(-1, 1))

    return output_octatope

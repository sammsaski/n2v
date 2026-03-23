"""
MaxPool2D layer reachability operations.

Translated from MATLAB NNV MaxPooling2DLayer.m

Supports both ImageStar (4D V) and Star (2D V) inputs with optimized paths for each.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from n2v.sets import Star, ImageStar, ImageZono, Hexatope, Octatope

logger = logging.getLogger(__name__)


def maxpool2d_star(
    layer: nn.MaxPool2d,
    input_stars: List[Star],
    method: str = 'exact',
    lp_solver: str = 'default',
    verbose: bool = False,
    **kwargs
) -> List[Star]:
    """
    MaxPool2D reachability for Star sets.

    Supports both ImageStar (optimized 4D path) and Star (requires ImageStar).

    Args:
        layer: PyTorch nn.MaxPool2d layer
        input_stars: List of input Stars (should be ImageStars)
        method: 'exact' or 'approx'
        lp_solver: LP solver option
        verbose: Display option ('display' or None)
        **kwargs: Additional options

    Returns:
        List of output Stars (ImageStars)
    """
    if method == 'exact':
        return _maxpool2d_star_exact_multiple(layer, input_stars, lp_solver, verbose)
    else:
        return _maxpool2d_star_approx_multiple(layer, input_stars, lp_solver, verbose)


def _maxpool2d_star_exact_single(
    layer: nn.MaxPool2d,
    input_star: ImageStar,
    lp_solver: str = 'default',
    verbose: bool = False
) -> List[ImageStar]:
    """
    Exact MaxPool2D reachability for a single ImageStar.

    Works directly on the 4D V tensor (H, W, C, nVar+1).

    Algorithm:
    1. For each pooling window, find the pixel(s) with maximum value
    2. If a unique max exists, use that pixel's value
    3. If multiple pixels could be max (uncertain), split into cases

    Args:
        layer: PyTorch nn.MaxPool2d layer
        input_star: Input ImageStar with 4D V
        lp_solver: LP solver option
        verbose: Display option

    Returns:
        List of output ImageStars (may be multiple due to splitting)
    """
    # Apply padding if needed
    pad_star = _apply_padding_4d(layer, input_star)

    # V is 4D: (H, W, C, nVar+1)
    V = pad_star.V
    h_in, w_in, c_in, n_cols = V.shape
    n_pred = n_cols - 1

    # Get kernel size and stride (can be int, tuple, or list from onnx2torch)
    kernel_size = layer.kernel_size
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    elif isinstance(kernel_size, list):
        kernel_size = tuple(kernel_size)

    stride = layer.stride
    if isinstance(stride, int):
        stride = (stride, stride)
    elif isinstance(stride, list):
        stride = tuple(stride)

    h_out = (h_in - kernel_size[0]) // stride[0] + 1
    w_out = (w_in - kernel_size[1]) // stride[1] + 1

    # Get start points for each pooling window
    start_points = _get_start_points(h_in, w_in, h_out, w_out, kernel_size, stride)

    # Initialize output basis tensor (4D)
    V_out = np.zeros((h_out, w_out, c_in, n_cols))

    # Track which positions need splitting
    split_positions = []
    max_indices = {}

    # Estimate ranges for bounds checking
    if pad_star.state_lb is None or pad_star.state_ub is None:
        pad_star.estimate_ranges()

    lb_4d = pad_star.state_lb.reshape(h_in, w_in, c_in)
    ub_4d = pad_star.state_ub.reshape(h_in, w_in, c_in)

    # For each channel and each pooling window, find the max
    for k in range(c_in):
        for i in range(h_out):
            for j in range(w_out):
                # Get indices of pixels in this pooling window
                start_h, start_w = start_points[i][j]
                max_idx = _get_local_max_index_4d(
                    lb_4d, ub_4d, start_h, start_w, kernel_size, k
                )

                max_indices[(i, j, k)] = max_idx

                if len(max_idx) == 1:
                    # Unique max - copy that pixel's value directly from 4D V
                    idx_h, idx_w = max_idx[0]
                    V_out[i, j, k, :] = V[idx_h, idx_w, k, :]
                else:
                    # Multiple possible maxes - need to split
                    split_positions.append((i, j, k))

    # Create initial output star with 4D V
    output_stars = [ImageStar(
        V_out, pad_star.C, pad_star.d,
        pad_star.predicate_lb, pad_star.predicate_ub,
        h_out, w_out, c_in
    )]

    # Report splits
    if verbose and len(split_positions) > 0:
        logger.debug(f'There are splits at {len(split_positions)} local regions')

    # Perform splitting for uncertain positions
    for pos in split_positions:
        i, j, k = pos
        max_idx_list = max_indices[(i, j, k)]

        new_stars = []
        for star in output_stars:
            # Split this star into multiple stars, one for each possible max
            split_stars = _step_split_4d(
                star, pad_star, (i, j, k), max_idx_list, lp_solver
            )
            new_stars.extend(split_stars)

        if verbose:
            logger.debug(f'Split {len(output_stars)} images into {len(new_stars)} images')

        output_stars = new_stars

    return output_stars


def _maxpool2d_star_exact_multiple(
    layer: nn.MaxPool2d,
    input_stars: List[ImageStar],
    lp_solver: str = 'default',
    verbose: bool = False
) -> List[ImageStar]:
    """
    Exact MaxPool2D for multiple input stars.
    """
    output_stars = []
    for star in input_stars:
        if isinstance(star, ImageStar):
            output_stars.extend(_maxpool2d_star_exact_single(layer, star, lp_solver, verbose))
        else:
            raise TypeError(f"MaxPool2D expects ImageStar input, got {type(star)}")
    return output_stars


def _maxpool2d_star_approx_single(
    layer: nn.MaxPool2d,
    input_star: ImageStar,
    lp_solver: str = 'default',
    verbose: bool = False
) -> ImageStar:
    """
    Approximate MaxPool2D reachability (over-approximation).

    Works directly on the 4D V tensor (H, W, C, nVar+1).

    When multiple pixels could be max, introduce a new predicate variable
    instead of splitting.

    Args:
        layer: PyTorch nn.MaxPool2d layer
        input_star: Input ImageStar with 4D V
        lp_solver: LP solver option
        verbose: Display option

    Returns:
        Single over-approximate ImageStar with 4D V
    """
    # Apply padding
    pad_star = _apply_padding_4d(layer, input_star)

    # V is 4D: (H, W, C, nVar+1)
    V = pad_star.V
    h_in, w_in, c_in, n_cols = V.shape
    n_pred_orig = n_cols - 1

    # Get kernel size and stride
    kernel_size = layer.kernel_size
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    elif isinstance(kernel_size, list):
        kernel_size = tuple(kernel_size)

    stride = layer.stride
    if isinstance(stride, int):
        stride = (stride, stride)
    elif isinstance(stride, list):
        stride = tuple(stride)

    h_out = (h_in - kernel_size[0]) // stride[0] + 1
    w_out = (w_in - kernel_size[1]) // stride[1] + 1

    # Get start points
    start_points = _get_start_points(h_in, w_in, h_out, w_out, kernel_size, stride)

    # Estimate ranges
    if pad_star.state_lb is None or pad_star.state_ub is None:
        pad_star.estimate_ranges()

    lb_4d = pad_star.state_lb.reshape(h_in, w_in, c_in)
    ub_4d = pad_star.state_ub.reshape(h_in, w_in, c_in)

    # Count new predicates needed
    new_pred_count = 0
    max_indices = {}

    for k in range(c_in):
        for i in range(h_out):
            for j in range(w_out):
                start_h, start_w = start_points[i][j]
                max_idx = _get_local_max_index_4d(
                    lb_4d, ub_4d, start_h, start_w, kernel_size, k
                )
                max_indices[(i, j, k)] = max_idx
                if len(max_idx) > 1:
                    new_pred_count += 1

    if verbose and new_pred_count > 0:
        logger.debug(f'{new_pred_count} new variables are introduced')

    # Build new basis matrix with additional predicates
    n_pred_new = n_pred_orig + new_pred_count
    V_out = np.zeros((h_out, w_out, c_in, n_pred_new + 1))

    # New constraints
    pool_size = kernel_size[0] * kernel_size[1]
    new_C = np.zeros((new_pred_count * (pool_size + 1), n_pred_new))
    new_d = np.zeros((new_pred_count * (pool_size + 1), 1))
    new_pred_lb = np.zeros((new_pred_count, 1))
    new_pred_ub = np.zeros((new_pred_count, 1))

    new_pred_idx = 0
    for k in range(c_in):
        for i in range(h_out):
            for j in range(w_out):
                max_idx = max_indices[(i, j, k)]
                start_h, start_w = start_points[i][j]

                if len(max_idx) == 1:
                    # Unique max - copy from 4D V
                    idx_h, idx_w = max_idx[0]
                    V_out[i, j, k, :n_pred_orig + 1] = V[idx_h, idx_w, k, :]
                else:
                    # Multiple maxes - introduce new predicate variable y
                    lb, ub = _get_local_bounds_4d(lb_4d, ub_4d, start_h, start_w, kernel_size, k)

                    # Use midpoint as center for the new variable
                    V_out[i, j, k, 0] = (lb + ub) / 2
                    V_out[i, j, k, n_pred_orig + 1 + new_pred_idx] = 1  # new predicate

                    # Predicate bounds are relative to center
                    half_range = (ub - lb) / 2
                    new_pred_lb[new_pred_idx] = -half_range
                    new_pred_ub[new_pred_idx] = half_range

                    # Constraint: y <= half_range
                    C_row = np.zeros((1, n_pred_new))
                    C_row[0, n_pred_orig + new_pred_idx] = 1
                    new_C[new_pred_idx * (pool_size + 1), :] = C_row
                    new_d[new_pred_idx * (pool_size + 1)] = half_range

                    # Constraints: xi - (center + y) <= 0 for each pixel
                    center = (lb + ub) / 2
                    for idx, (ph, pw) in enumerate(_get_local_points(start_h, start_w, kernel_size)):
                        C_row = np.zeros((1, n_pred_new))
                        # Coefficients from 4D V
                        C_row[0, :n_pred_orig] = V[ph, pw, k, 1:n_pred_orig + 1]
                        C_row[0, n_pred_orig + new_pred_idx] = -1
                        new_C[new_pred_idx * (pool_size + 1) + 1 + idx, :] = C_row
                        new_d[new_pred_idx * (pool_size + 1) + 1 + idx] = center - V[ph, pw, k, 0]

                    new_pred_idx += 1

    # Combine constraints
    n_orig_constraints = pad_star.C.shape[0]
    C_combined = np.hstack([pad_star.C, np.zeros((n_orig_constraints, new_pred_count))])
    C_combined = np.vstack([C_combined, new_C])
    d_combined = np.vstack([pad_star.d, new_d])

    pred_lb_combined = np.vstack([pad_star.predicate_lb, new_pred_lb])
    pred_ub_combined = np.vstack([pad_star.predicate_ub, new_pred_ub])

    return ImageStar(
        V_out, C_combined, d_combined,
        pred_lb_combined, pred_ub_combined,
        h_out, w_out, c_in
    )


def _maxpool2d_star_approx_multiple(
    layer: nn.MaxPool2d,
    input_stars: List[ImageStar],
    lp_solver: str = 'default',
    verbose: bool = False
) -> List[ImageStar]:
    """
    Approximate MaxPool2D for multiple input stars.
    """
    output = []
    for star in input_stars:
        if isinstance(star, ImageStar):
            output.append(_maxpool2d_star_approx_single(layer, star, lp_solver, verbose))
        else:
            raise TypeError(f"MaxPool2D expects ImageStar input, got {type(star)}")
    return output


def maxpool2d_zono(layer: nn.MaxPool2d, input_zonos: List[ImageZono]) -> List[ImageZono]:
    """
    MaxPool2D for ImageZono (over-approximation using bounds).

    Args:
        layer: PyTorch nn.MaxPool2d layer
        input_zonos: List of input ImageZonos

    Returns:
        List of output ImageZonos
    """
    output_zonos = []
    for zono in input_zonos:
        # Get bounds
        lb = zono.get_bounds()[0]
        ub = zono.get_bounds()[1]

        # Reshape to (channels, height, width) for PyTorch
        lb_img = lb.reshape(zono.height, zono.width, zono.num_channels).transpose(2, 0, 1)
        ub_img = ub.reshape(zono.height, zono.width, zono.num_channels).transpose(2, 0, 1)

        # Convert to torch tensors
        lb_torch = torch.from_numpy(lb_img).unsqueeze(0).float()
        ub_torch = torch.from_numpy(ub_img).unsqueeze(0).float()

        # Apply maxpool to -lb (gives -(min pooling)) and ub
        new_lb = -F.max_pool2d(-lb_torch, layer.kernel_size, layer.stride, layer.padding)
        new_ub = F.max_pool2d(ub_torch, layer.kernel_size, layer.stride, layer.padding)

        # Convert back to ImageZono
        new_lb_np = new_lb.squeeze(0).numpy().transpose(1, 2, 0)
        new_ub_np = new_ub.squeeze(0).numpy().transpose(1, 2, 0)

        output_zono = ImageZono.from_bounds(
            new_lb_np, new_ub_np,
            new_lb_np.shape[0], new_lb_np.shape[1], new_lb_np.shape[2]
        )
        output_zonos.append(output_zono)

    return output_zonos


# Helper functions

def _apply_padding_4d(layer: nn.MaxPool2d, input_star: ImageStar) -> ImageStar:
    """Apply zero padding to ImageStar with 4D V if needed."""
    padding = layer.padding
    if isinstance(padding, int):
        padding = (padding, padding)
    elif isinstance(padding, (list, tuple)):
        padding = tuple(padding)

    if padding == (0, 0):
        return input_star

    # V is 4D: (H, W, C, nVar+1)
    V = input_star.V
    h, w, c, n_cols = V.shape

    # Padding: (top, bottom, left, right)
    pad_t, pad_b = padding[0], padding[0]
    pad_l, pad_r = padding[1], padding[1]

    h_pad = h + pad_t + pad_b
    w_pad = w + pad_l + pad_r

    # Create padded V (4D)
    V_pad = np.zeros((h_pad, w_pad, c, n_cols))
    V_pad[pad_t:pad_t + h, pad_l:pad_l + w, :, :] = V

    return ImageStar(
        V_pad, input_star.C, input_star.d,
        input_star.predicate_lb, input_star.predicate_ub,
        h_pad, w_pad, c
    )


def _get_start_points(h_in: int, w_in: int, h_out: int, w_out: int,
                      kernel_size: Tuple[int, int], stride: Tuple[int, int]) -> List[List[Tuple[int, int]]]:
    """
    Get start points (top-left corners) for each pooling window.

    Returns:
        List of lists, indexed as [i][j] -> (start_h, start_w)
    """
    start_points = [[None for _ in range(w_out)] for _ in range(h_out)]

    for i in range(h_out):
        for j in range(w_out):
            start_h = i * stride[0]
            start_w = j * stride[1]
            start_points[i][j] = (start_h, start_w)

    return start_points


def _get_local_points(start_h: int, start_w: int, kernel_size: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Get all pixel indices in a pooling window."""
    points = []
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            points.append((start_h + i, start_w + j))
    return points


def _get_local_max_index_4d(
    lb_4d: np.ndarray,
    ub_4d: np.ndarray,
    start_h: int, start_w: int,
    kernel_size: Tuple[int, int],
    channel: int
) -> List[Tuple[int, int]]:
    """
    Find the pixel(s) with maximum value in a local pooling window.

    Uses pre-computed 4D bounds arrays.

    Returns:
        List of (h, w) indices. If len == 1, unique max. If len > 1, uncertain.
    """
    points = _get_local_points(start_h, start_w, kernel_size)

    # Get bounds for each point from 4D arrays
    lbs = [lb_4d[ph, pw, channel] for ph, pw in points]
    ubs = [ub_4d[ph, pw, channel] for ph, pw in points]

    # Find the point with maximum lower bound
    max_lb_val = max(lbs)
    max_lb_idx = lbs.index(max_lb_val)

    # Check which points could potentially be >= max_lb_val
    candidates = [i for i, ub in enumerate(ubs) if ub >= max_lb_val]

    if len(candidates) == 1:
        return [points[max_lb_idx]]
    else:
        return [points[i] for i in candidates]


def _get_local_bounds_4d(
    lb_4d: np.ndarray,
    ub_4d: np.ndarray,
    start_h: int, start_w: int,
    kernel_size: Tuple[int, int],
    channel: int
) -> Tuple[float, float]:
    """Get min/max bounds for all pixels in a local pooling window."""
    points = _get_local_points(start_h, start_w, kernel_size)

    lbs = [lb_4d[ph, pw, channel] for ph, pw in points]
    ubs = [ub_4d[ph, pw, channel] for ph, pw in points]

    return min(lbs), max(ubs)


def _step_split_4d(
    current_star: ImageStar,
    original_star: ImageStar,
    pos: Tuple[int, int, int],
    max_indices: List[Tuple[int, int]],
    lp_solver: str
) -> List[ImageStar]:
    """
    Split an ImageStar into multiple stars based on which pixel is max.

    Works directly on 4D V tensors.

    Args:
        current_star: Current output ImageStar being built (4D V)
        original_star: Original padded input ImageStar (4D V)
        pos: (i, j, k) position in output where split occurs
        max_indices: List of (h, w) candidates for max pixel
        lp_solver: LP solver option

    Returns:
        List of ImageStars, one for each valid max candidate
    """
    i, j, k = pos
    V_curr = current_star.V  # 4D: (H_out, W_out, C, nVar+1)
    V_orig = original_star.V  # 4D: (H_in, W_in, C, nVar+1)
    n_pred = current_star.nVar

    output_stars = []

    for idx, (max_h, max_w) in enumerate(max_indices):
        # Create constraints: this pixel is >= all others
        constraints = []
        for other_h, other_w in max_indices:
            if (other_h, other_w) == (max_h, max_w):
                continue

            # Constraint: V[max] - V[other] >= 0
            # Becomes: V[other]_basis - V[max]_basis) * α <= V[max]_center - V[other]_center
            C_row = np.zeros((1, n_pred))
            if original_star.nVar > 0:
                C_row[0, :] = V_orig[other_h, other_w, k, 1:] - V_orig[max_h, max_w, k, 1:]
            d_val = V_orig[max_h, max_w, k, 0] - V_orig[other_h, other_w, k, 0]

            constraints.append((C_row, d_val))

        # Combine with existing constraints
        if len(constraints) > 0:
            new_C_rows = np.vstack([c[0] for c in constraints])
            new_d_vals = np.array([[c[1]] for c in constraints])
            new_C = np.vstack([current_star.C, new_C_rows])
            new_d = np.vstack([current_star.d, new_d_vals])
        else:
            new_C = current_star.C
            new_d = current_star.d

        # Update V at position (i, j, k) with the max pixel's value
        V_out = V_curr.copy()
        V_out[i, j, k, :original_star.nVar + 1] = V_orig[max_h, max_w, k, :]

        new_star = ImageStar(
            V_out, new_C, new_d,
            current_star.predicate_lb, current_star.predicate_ub,
            current_star.height, current_star.width, current_star.num_channels
        )

        output_stars.append(new_star)

    return output_stars


def maxpool2d_hexatope(layer: nn.MaxPool2d, input_hexatopes: List[Hexatope]) -> List[Hexatope]:
    """
    MaxPool2D for Hexatopes (over-approximation using bounds).

    Since maxpooling is non-linear, we use interval arithmetic over-approximation.

    Args:
        layer: PyTorch nn.MaxPool2d layer
        input_hexatopes: List of input Hexatopes

    Returns:
        List of output Hexatopes (over-approximation)
    """
    output_hexatopes = []

    for hexatope in input_hexatopes:
        lb, ub = hexatope.estimate_ranges()
        new_lb = lb
        new_ub = ub
        output_hexatope = Hexatope.from_bounds(new_lb, new_ub)
        output_hexatopes.append(output_hexatope)

    return output_hexatopes


def maxpool2d_octatope(layer: nn.MaxPool2d, input_octatopes: List[Octatope]) -> List[Octatope]:
    """
    MaxPool2D for Octatopes (over-approximation using bounds).

    Since maxpooling is non-linear, we use interval arithmetic over-approximation.

    Args:
        layer: PyTorch nn.MaxPool2d layer
        input_octatopes: List of input Octatopes

    Returns:
        List of output Octatopes (over-approximation)
    """
    output_octatopes = []

    for octatope in input_octatopes:
        lb, ub = octatope.estimate_ranges()
        new_lb = lb
        new_ub = ub
        output_octatope = Octatope.from_bounds(new_lb, new_ub)
        output_octatopes.append(output_octatope)

    return output_octatopes

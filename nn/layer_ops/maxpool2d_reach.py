"""
MaxPool2D layer reachability operations.

Translated from MATLAB NNV MaxPooling2DLayer.m
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from nnv_py.sets import Star, ImageStar, ImageZono


def maxpool2d_star(
    layer: nn.MaxPool2d,
    input_stars: List[Star],
    method: str = 'exact',
    lp_solver: str = 'default',
    dis_opt: Optional[str] = None,
    **kwargs
) -> List[Star]:
    """
    MaxPool2D reachability for Star sets.

    Args:
        layer: PyTorch nn.MaxPool2d layer
        input_stars: List of input Stars (should be ImageStars)
        method: 'exact' or 'approx'
        lp_solver: LP solver option
        dis_opt: Display option ('display' or None)
        **kwargs: Additional options

    Returns:
        List of output Stars (ImageStars)
    """
    if method == 'exact':
        return _maxpool2d_star_exact_multiple(layer, input_stars, lp_solver, dis_opt)
    else:
        return _maxpool2d_star_approx_multiple(layer, input_stars, lp_solver, dis_opt)


def _maxpool2d_star_exact_single(
    layer: nn.MaxPool2d,
    input_star: ImageStar,
    lp_solver: str = 'default',
    dis_opt: Optional[str] = None
) -> List[ImageStar]:
    """
    Exact MaxPool2D reachability for a single ImageStar.

    Algorithm:
    1. For each pooling window, find the pixel(s) with maximum value
    2. If a unique max exists, use that pixel's value
    3. If multiple pixels could be max (uncertain), split into cases

    Args:
        layer: PyTorch nn.MaxPool2d layer
        input_star: Input ImageStar
        lp_solver: LP solver option
        dis_opt: Display option

    Returns:
        List of output ImageStars (may be multiple due to splitting)
    """
    # Apply padding if needed
    pad_star = _apply_padding(layer, input_star)

    # Get output dimensions
    h_in, w_in, c_in = pad_star.height, pad_star.width, pad_star.num_channels
    kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
    stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)

    h_out = (h_in - kernel_size[0]) // stride[0] + 1
    w_out = (w_in - kernel_size[1]) // stride[1] + 1

    # Get start points for each pooling window
    start_points = _get_start_points(h_in, w_in, h_out, w_out, kernel_size, stride)

    # Initialize output basis matrix
    n_pred = pad_star.nVar
    V_out = np.zeros((h_out, w_out, c_in, n_pred + 1))

    # Track which positions need splitting
    split_positions = []
    max_indices = {}  # Store max indices for each position

    # For each channel and each pooling window, find the max
    for k in range(c_in):
        for i in range(h_out):
            for j in range(w_out):
                # Get indices of pixels in this pooling window
                start_h, start_w = start_points[i][j]
                max_idx = _get_local_max_index(
                    pad_star, start_h, start_w, kernel_size, k, lp_solver
                )

                max_indices[(i, j, k)] = max_idx

                if len(max_idx) == 1:
                    # Unique max - copy that pixel's value
                    idx_h, idx_w = max_idx[0]
                    # V is stored flattened, so reshape to access
                    V_img = pad_star.V.reshape(h_in, w_in, c_in, n_pred + 1)
                    V_out[i, j, k, :] = V_img[idx_h, idx_w, k, :]
                else:
                    # Multiple possible maxes - need to split
                    split_positions.append((i, j, k))

    # Create initial output star
    V_out_flat = V_out.reshape(-1, n_pred + 1)
    output_stars = [ImageStar(
        V_out_flat, pad_star.C, pad_star.d,
        pad_star.predicate_lb, pad_star.predicate_ub,
        h_out, w_out, c_in
    )]

    # Report splits
    if dis_opt == 'display' and len(split_positions) > 0:
        print(f'There are splits at {len(split_positions)} local regions')

    # Perform splitting for uncertain positions
    for pos in split_positions:
        i, j, k = pos
        max_idx_list = max_indices[(i, j, k)]

        new_stars = []
        for star in output_stars:
            # Split this star into multiple stars, one for each possible max
            split_stars = _step_split(
                star, pad_star, (i, j, k), max_idx_list, lp_solver
            )
            new_stars.extend(split_stars)

        if dis_opt == 'display':
            print(f'Split {len(output_stars)} images into {len(new_stars)} images')

        output_stars = new_stars

    return output_stars


def _maxpool2d_star_exact_multiple(
    layer: nn.MaxPool2d,
    input_stars: List[ImageStar],
    lp_solver: str = 'default',
    dis_opt: Optional[str] = None
) -> List[ImageStar]:
    """
    Exact MaxPool2D for multiple input stars.
    """
    output_stars = []
    for star in input_stars:
        output_stars.extend(_maxpool2d_star_exact_single(layer, star, lp_solver, dis_opt))
    return output_stars


def _maxpool2d_star_approx_single(
    layer: nn.MaxPool2d,
    input_star: ImageStar,
    lp_solver: str = 'default',
    dis_opt: Optional[str] = None
) -> ImageStar:
    """
    Approximate MaxPool2D reachability (over-approximation).

    When multiple pixels could be max, introduce a new predicate variable
    instead of splitting.

    Args:
        layer: PyTorch nn.MaxPool2d layer
        input_star: Input ImageStar
        lp_solver: LP solver option
        dis_opt: Display option

    Returns:
        Single over-approximate ImageStar
    """
    # Apply padding
    pad_star = _apply_padding(layer, input_star)

    # Get dimensions
    h_in, w_in, c_in = pad_star.height, pad_star.width, pad_star.num_channels
    kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
    stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)

    h_out = (h_in - kernel_size[0]) // stride[0] + 1
    w_out = (w_in - kernel_size[1]) // stride[1] + 1

    # Get start points
    start_points = _get_start_points(h_in, w_in, h_out, w_out, kernel_size, stride)

    # Count new predicates needed
    n_pred_orig = pad_star.nVar
    new_pred_count = 0

    # First pass: determine how many new predicates we need
    max_indices = {}
    for k in range(c_in):
        for i in range(h_out):
            for j in range(w_out):
                start_h, start_w = start_points[i][j]
                max_idx = _get_local_max_index(
                    pad_star, start_h, start_w, kernel_size, k, lp_solver
                )
                max_indices[(i, j, k)] = max_idx
                if len(max_idx) > 1:
                    new_pred_count += 1

    if dis_opt == 'display' and new_pred_count > 0:
        print(f'{new_pred_count} new variables are introduced')

    # Build new basis matrix with additional predicates
    n_pred_new = n_pred_orig + new_pred_count
    V_out = np.zeros((h_out, w_out, c_in, n_pred_new + 1))
    V_in = pad_star.V.reshape(h_in, w_in, c_in, n_pred_orig + 1)

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
                    # Unique max
                    idx_h, idx_w = max_idx[0]
                    V_out[i, j, k, :n_pred_orig + 1] = V_in[idx_h, idx_w, k, :]
                else:
                    # Multiple maxes - introduce new predicate variable y
                    # Constraints: y <= ub(local region), xi - y <= 0 for all i in region
                    V_out[i, j, k, 0] = 0  # center
                    V_out[i, j, k, n_pred_orig + 1 + new_pred_idx] = 1  # new predicate

                    # Get local bounds
                    lb, ub = _get_local_bounds(
                        pad_star, start_h, start_w, kernel_size, k, lp_solver
                    )
                    new_pred_lb[new_pred_idx] = lb
                    new_pred_ub[new_pred_idx] = ub

                    # Constraint: y <= ub
                    C_row = np.zeros((1, n_pred_new))
                    C_row[0, n_pred_orig + new_pred_idx] = 1
                    new_C[new_pred_idx * (pool_size + 1), :] = C_row
                    new_d[new_pred_idx * (pool_size + 1)] = ub

                    # Constraints: xi - y <= 0 for each pixel in pooling window
                    for idx, (ph, pw) in enumerate(_get_local_points(start_h, start_w, kernel_size)):
                        C_row = np.zeros((1, n_pred_new))
                        # xi coefficient (from V)
                        C_row[0, :n_pred_orig] = V_in[ph, pw, k, 1:n_pred_orig + 1]
                        # -y coefficient
                        C_row[0, n_pred_orig + new_pred_idx] = -1
                        new_C[new_pred_idx * (pool_size + 1) + 1 + idx, :] = C_row
                        new_d[new_pred_idx * (pool_size + 1) + 1 + idx] = -V_in[ph, pw, k, 0]

                    new_pred_idx += 1

    # Combine constraints
    n_orig_constraints = pad_star.C.shape[0]
    C_combined = np.hstack([pad_star.C, np.zeros((n_orig_constraints, new_pred_count))])
    C_combined = np.vstack([C_combined, new_C])
    d_combined = np.vstack([pad_star.d, new_d])

    pred_lb_combined = np.vstack([pad_star.predicate_lb, new_pred_lb])
    pred_ub_combined = np.vstack([pad_star.predicate_ub, new_pred_ub])

    V_out_flat = V_out.reshape(-1, n_pred_new + 1)
    return ImageStar(
        V_out_flat, C_combined, d_combined,
        pred_lb_combined, pred_ub_combined,
        h_out, w_out, c_in
    )


def _maxpool2d_star_approx_multiple(
    layer: nn.MaxPool2d,
    input_stars: List[ImageStar],
    lp_solver: str = 'default',
    dis_opt: Optional[str] = None
) -> List[ImageStar]:
    """
    Approximate MaxPool2D for multiple input stars.
    """
    return [_maxpool2d_star_approx_single(layer, star, lp_solver, dis_opt) for star in input_stars]


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
        lb = zono.get_bounds()[0]  # Lower bound image
        ub = zono.get_bounds()[1]  # Upper bound image

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

def _apply_padding(layer: nn.MaxPool2d, input_star: ImageStar) -> ImageStar:
    """Apply zero padding to ImageStar if needed."""
    padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)

    if padding == (0, 0):
        return input_star

    # Pad the ImageStar
    h, w, c = input_star.height, input_star.width, input_star.num_channels
    n_pred = input_star.nVar

    # Padding: (top, bottom, left, right)
    pad_t, pad_b = padding[0], padding[0]
    pad_l, pad_r = padding[1], padding[1]

    h_pad = h + pad_t + pad_b
    w_pad = w + pad_l + pad_r

    # Reshape V to image format
    V_img = input_star.V.reshape(h, w, c, n_pred + 1)

    # Create padded V
    V_pad = np.zeros((h_pad, w_pad, c, n_pred + 1))
    V_pad[pad_t:pad_t + h, pad_l:pad_l + w, :, :] = V_img

    # Flatten back
    V_pad_flat = V_pad.reshape(-1, n_pred + 1)

    return ImageStar(
        V_pad_flat, input_star.C, input_star.d,
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


def _get_local_max_index(
    image_star: ImageStar,
    start_h: int, start_w: int,
    kernel_size: Tuple[int, int],
    channel: int,
    lp_solver: str
) -> List[Tuple[int, int]]:
    """
    Find the pixel(s) with maximum value in a local pooling window.

    Returns:
        List of (h, w) indices. If len == 1, unique max. If len > 1, uncertain.
    """
    # Get bounds for this ImageStar
    if image_star.state_lb is None or image_star.state_ub is None:
        image_star.estimate_ranges(lp_solver)

    # Reshape bounds to image format
    h, w, c = image_star.height, image_star.width, image_star.num_channels
    lb_img = image_star.state_lb.reshape(h, w, c)
    ub_img = image_star.state_ub.reshape(h, w, c)

    # Get points in pooling window
    points = _get_local_points(start_h, start_w, kernel_size)

    # Get bounds for each point
    lbs = [lb_img[ph, pw, channel] for ph, pw in points]
    ubs = [ub_img[ph, pw, channel] for ph, pw in points]

    # Find the point with maximum lower bound
    max_lb_val = max(lbs)
    max_lb_idx = lbs.index(max_lb_val)

    # Check which points could potentially be >= max_lb_val
    candidates = [i for i, ub in enumerate(ubs) if ub >= max_lb_val]

    if len(candidates) == 1:
        # Unique maximum
        return [points[max_lb_idx]]
    else:
        # Multiple candidates - need more precise LP check
        # For now, return all candidates (could be refined with LP)
        return [points[i] for i in candidates]


def _get_local_bounds(
    image_star: ImageStar,
    start_h: int, start_w: int,
    kernel_size: Tuple[int, int],
    channel: int,
    lp_solver: str
) -> Tuple[float, float]:
    """Get bounds for all pixels in a local pooling window."""
    if image_star.state_lb is None or image_star.state_ub is None:
        image_star.estimate_ranges(lp_solver)

    h, w, c = image_star.height, image_star.width, image_star.num_channels
    lb_img = image_star.state_lb.reshape(h, w, c)
    ub_img = image_star.state_ub.reshape(h, w, c)

    points = _get_local_points(start_h, start_w, kernel_size)

    lbs = [lb_img[ph, pw, channel] for ph, pw in points]
    ubs = [ub_img[ph, pw, channel] for ph, pw in points]

    return min(lbs), max(ubs)


def _step_split(
    current_star: ImageStar,
    original_star: ImageStar,
    pos: Tuple[int, int, int],
    max_indices: List[Tuple[int, int]],
    lp_solver: str
) -> List[ImageStar]:
    """
    Split an ImageStar into multiple stars based on which pixel is max.

    Args:
        current_star: Current output ImageStar being built
        original_star: Original padded input ImageStar
        pos: (i, j, k) position in output where split occurs
        max_indices: List of (h, w) candidates for max pixel
        lp_solver: LP solver option

    Returns:
        List of ImageStars, one for each valid max candidate
    """
    i, j, k = pos
    h_out, w_out, c_out = current_star.height, current_star.width, current_star.num_channels
    h_in, w_in, c_in = original_star.height, original_star.width, original_star.num_channels
    n_pred = current_star.nVar

    output_stars = []

    for idx, (max_h, max_w) in enumerate(max_indices):
        # Create constraints: this pixel is >= all others
        # For each other pixel p: center_pixel - p >= 0
        constraints = []
        for other_h, other_w in max_indices:
            if (other_h, other_w) == (max_h, max_w):
                continue

            # Add constraint: V[max_h, max_w] >= V[other_h, other_w]
            # This becomes: V[max_h, max_w] - V[other_h, other_w] >= 0
            # Or: -V[max_h, max_w] + V[other_h, other_w] <= 0

            V_in = original_star.V.reshape(h_in, w_in, c_in, original_star.nVar + 1)

            # Construct constraint row
            C_row = np.zeros((1, n_pred))
            # Coefficients for: V[other] - V[max] <= 0 becomes C*α <= d
            # V = V0*1 + V1*α1 + V2*α2 + ...
            # V[max] - V[other] >= 0
            # (V[max]_basis * α) - (V[other]_basis * α) >= 0
            # (V[max]_basis - V[other]_basis) * α >= V[other]_center - V[max]_center
            # Flip: (V[other]_basis - V[max]_basis) * α <= V[max]_center - V[other]_center

            if original_star.nVar > 0:
                C_row[0, :] = V_in[other_h, other_w, k, 1:] - V_in[max_h, max_w, k, 1:]

            d_val = V_in[max_h, max_w, k, 0] - V_in[other_h, other_w, k, 0]

            constraints.append((C_row, d_val))

        # Check if constraints are feasible
        if len(constraints) > 0:
            new_C_rows = np.vstack([c[0] for c in constraints])
            new_d_vals = np.array([[c[1]] for c in constraints])

            # Combine with existing constraints
            new_C = np.vstack([current_star.C, new_C_rows])
            new_d = np.vstack([current_star.d, new_d_vals])
        else:
            new_C = current_star.C
            new_d = current_star.d

        # Update V at position (i, j, k) with the max pixel's value
        V_out = current_star.V.reshape(h_out, w_out, c_out, n_pred + 1).copy()
        V_in = original_star.V.reshape(h_in, w_in, c_in, original_star.nVar + 1)
        V_out[i, j, k, :original_star.nVar + 1] = V_in[max_h, max_w, k, :]

        V_out_flat = V_out.reshape(-1, n_pred + 1)

        new_star = ImageStar(
            V_out_flat, new_C, new_d,
            current_star.predicate_lb, current_star.predicate_ub,
            h_out, w_out, c_out
        )

        # Check feasibility (optional, can be expensive)
        # For now, assume it's feasible
        output_stars.append(new_star)

    return output_stars

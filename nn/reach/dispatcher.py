"""
Dispatcher for reachability analysis based on method and set type.
"""

import torch.nn as nn
from typing import Union, List


def reach_pytorch_model(
    model: nn.Module,
    input_set: Union['Star', 'Zono', 'Box'],
    method: str = 'approx-star',
    num_cores: int = 1,
) -> List:
    """
    Dispatch reachability computation based on method and input set type.

    Args:
        model: PyTorch model
        input_set: Input specification
        method: Reachability method
        num_cores: Number of cores for parallel computation

    Returns:
        List of output sets

    Raises:
        NotImplementedError: For unsupported method/set combinations
    """
    from ...sets import Star, Zono, Box

    if method == 'exact-star':
        if not isinstance(input_set, Star):
            # Convert to Star
            if hasattr(input_set, 'to_star'):
                input_set = input_set.to_star()
            else:
                raise TypeError(f"Cannot convert {type(input_set)} to Star")

        from .reach_star import reach_star_exact
        return reach_star_exact(model, [input_set], num_cores=num_cores)

    elif method == 'approx-star':
        if not isinstance(input_set, Star):
            if hasattr(input_set, 'to_star'):
                input_set = input_set.to_star()
            else:
                raise TypeError(f"Cannot convert {type(input_set)} to Star")

        from .reach_star import reach_star_approx
        return reach_star_approx(model, [input_set], num_cores=num_cores)

    elif method == 'approx-zono':
        if not isinstance(input_set, Zono):
            if hasattr(input_set, 'to_zono'):
                input_set = input_set.to_zono()
            else:
                raise TypeError(f"Cannot convert {type(input_set)} to Zono")

        from .reach_zono import reach_zono_approx
        return reach_zono_approx(model, [input_set])

    elif method == 'approx-box':
        if not isinstance(input_set, Box):
            if hasattr(input_set, 'get_box'):
                input_set = input_set.get_box()
            else:
                raise TypeError(f"Cannot convert {type(input_set)} to Box")

        from .reach_box import reach_box_approx
        return reach_box_approx(model, [input_set])

    else:
        raise ValueError(f"Unknown reachability method: {method}")

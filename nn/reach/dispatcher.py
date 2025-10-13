"""
Dispatcher for reachability analysis based on method and set type.
"""

import torch.nn as nn
from typing import Union, List


def reach_pytorch_model(
    model: nn.Module,
    input_set: Union['Star', 'Zono', 'Box', 'Hexatope', 'Octatope'],
    method: str = 'exact',
    **kwargs
) -> List:
    """
    Dispatch reachability computation based on method and input set type.

    This is the primary dispatcher that routes to the appropriate reachability
    method based on the input set type and the requested method.

    Args:
        model: PyTorch model
        input_set: Input specification (Star, Zono, Box, Hexatope, or Octatope)
        method: Reachability method:
            - 'exact': Exact reachability (Star, Hexatope, Octatope)
            - 'exact-differentiable': Exact with differentiable solver (Hexatope, Octatope)
            - 'approx': Over-approximate reachability (all set types)
        **kwargs: Additional method-specific arguments passed to underlying methods

    Returns:
        List of output sets (same type as input)

    Raises:
        TypeError: If input_set type is not supported
        ValueError: If method is not valid for the given set type

    Note:
        For backward compatibility, the old method names like 'exact-star',
        'approx-star', 'approx-zono', etc. are still supported but the new
        unified naming is preferred: just use 'exact' or 'approx' and the
        dispatcher will determine the appropriate method based on set type.
    """
    from ...sets import Star, Zono, Box, Hexatope, Octatope

    # Handle backward compatibility with old method names
    legacy_methods = {
        'exact-star': ('exact', Star),
        'approx-star': ('approx', Star),
        'approx-zono': ('approx', Zono),
        'approx-box': ('approx', Box),
        'exact-hexatope': ('exact', Hexatope),
        'approx-hexatope': ('approx', Hexatope),
        'exact-octatope': ('exact', Octatope),
        'approx-octatope': ('approx', Octatope),
    }

    if method in legacy_methods:
        new_method, expected_type = legacy_methods[method]
        if not isinstance(input_set, expected_type):
            # Try to convert
            if hasattr(input_set, f'to_{expected_type.__name__.lower()}'):
                input_set = getattr(input_set, f'to_{expected_type.__name__.lower()}')()
            else:
                raise TypeError(
                    f"Method '{method}' requires {expected_type.__name__} input, "
                    f"got {type(input_set).__name__}"
                )
        method = new_method

    # Dispatch based on input set type
    if isinstance(input_set, Star):
        if method == 'exact':
            from .reach_star import reach_star_exact
            return reach_star_exact(model, [input_set], **kwargs)
        elif method == 'approx':
            from .reach_star import reach_star_approx
            return reach_star_approx(model, [input_set], **kwargs)
        else:
            raise ValueError(
                f"Unknown method '{method}' for Star. "
                f"Supported: 'exact', 'approx'"
            )

    elif isinstance(input_set, Zono):
        if method != 'approx':
            raise ValueError(
                f"Only 'approx' method is supported for Zono, got '{method}'"
            )
        from .reach_zono import reach_zono_approx
        return reach_zono_approx(model, [input_set], **kwargs)

    elif isinstance(input_set, Box):
        if method != 'approx':
            raise ValueError(
                f"Only 'approx' method is supported for Box, got '{method}'"
            )
        from .reach_box import reach_box_approx
        return reach_box_approx(model, [input_set])

    elif isinstance(input_set, Hexatope):
        use_differentiable = (method == 'exact-differentiable')

        if method in ('exact', 'exact-differentiable'):
            from .reach_hexatope import reach_hexatope_exact
            return reach_hexatope_exact(
                model, [input_set],
                use_differentiable=use_differentiable,
                **kwargs
            )
        elif method == 'approx':
            from .reach_hexatope import reach_hexatope_approx
            return reach_hexatope_approx(model, [input_set], **kwargs)
        else:
            raise ValueError(
                f"Unknown method '{method}' for Hexatope. "
                f"Supported: 'exact', 'exact-differentiable', 'approx'"
            )

    elif isinstance(input_set, Octatope):
        use_differentiable = (method == 'exact-differentiable')

        if method in ('exact', 'exact-differentiable'):
            from .reach_octatope import reach_octatope_exact
            return reach_octatope_exact(
                model, [input_set],
                use_differentiable=use_differentiable,
                **kwargs
            )
        elif method == 'approx':
            from .reach_octatope import reach_octatope_approx
            return reach_octatope_approx(model, [input_set], **kwargs)
        else:
            raise ValueError(
                f"Unknown method '{method}' for Octatope. "
                f"Supported: 'exact', 'exact-differentiable', 'approx'"
            )

    else:
        raise TypeError(
            f"Unsupported input set type: {type(input_set).__name__}. "
            f"Supported types: Star, Zono, Box, Hexatope, Octatope"
        )

"""GELU activation reachability.

Two approximation modes are supported, matching ``nn.GELU(approximate=...)``:

* ``approximate='none'`` (default) -- the exact ``erf``-based form
  ``GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))``. Global minimum
  ``f ≈ -0.169971207`` at ``x ≈ -0.7517918``.
* ``approximate='tanh'`` -- the GPT-2 / HF default form
  ``GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.
  Global minimum ``f ≈ -0.170041`` at ``x ≈ -0.7517``.

Both forms are smooth and non-monotone (small dip near ``x = -0.75``). Box reach
clamps the lower bound to the global ``F_MIN`` constant whenever the input
interval brackets the dip; Star reach lifts the per-element Box bound into a
fresh predicate. Floor constants are rounded **away from zero** so the Box
floor is a true lower bound (T0-3, audit C5: the previous constants were
rounded toward zero by ~1.2e-6 and produced a tiny under-approximation).
"""

from __future__ import annotations

from math import erf, sqrt, tanh
from typing import List

import numpy as np

from n2v.sets import Box, Star
from n2v.nn.layer_ops._image_shape import apply_box_lift_star


# Exact erf-form constants.
#
# PR-1 audit I1: the previous X_MIN constants were the true argmin rounded
# to only 4-7 digits. The ``contains_min = (lb <= x_min) & (ub >= x_min)``
# point check then missed narrow intervals that actually bracket the true
# argmin but exclude the rounded constant, producing an above-floor lower
# bound -- unsound by ~5e-8 on Box [-0.7530, -0.7520] for the tanh form.
#
# Fix: use the true argmin to ~16 digits (verified against scipy's bounded
# Brent solver with xatol = 1e-12) for the point check, and use F_MIN
# rounded AWAY from zero from the true value as the floor. Adopt a tiny
# guard band around X_MIN so floating-point representation of nearby
# interval endpoints can't slip past the check.
_GELU_X_MIN = -0.7517915237434403   # true erf-GELU argmin (scipy-verified)
_GELU_F_MIN = -0.16997120748        # erf-GELU(x_min) rounded AWAY from zero
                                    # from the true -0.16997120747990369.

# Tanh-approximation constants. The tanh form's minimum is slightly LOWER
# than the erf form's, so a model authored with ``nn.GELU(approximate='tanh')``
# (GPT-2 and many HF transformers) needs its own floor.
_GELU_TANH_X_MIN = -0.7524614212122555  # true tanh-GELU argmin (scipy)
_GELU_TANH_F_MIN = -0.17004075058       # rounded AWAY from zero from
                                        # true -0.17004075057125403.

# Tiny inward-guard band: a box ``(lb, ub)`` is considered to bracket the
# argmin if ``lb <= x_min + GUARD`` and ``ub >= x_min - GUARD``. Without
# this, an endpoint exactly equal to ``x_min`` modulo float rounding could
# slip outside the check. ``GUARD = 5e-12`` is well below double-precision
# ULP at this magnitude and well below any practical reach tolerance.
_GELU_XMIN_GUARD = 5e-12

_TANH_SQRT_2_OVER_PI = sqrt(2.0 / np.pi)


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact erf-form GELU forward."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    """Tanh-approximation GELU forward (GPT-2 / HuggingFace default)."""
    x = np.asarray(x, dtype=np.float64)
    inner = _TANH_SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.vectorize(tanh)(inner))


def _gelu_box_impl(
    input_boxes: List[Box],
    forward,
    x_min: float,
    f_min: float,
) -> List[Box]:
    out: List[Box] = []
    for b in input_boxes:
        lb = b.lb.flatten()
        ub = b.ub.flatten()
        fl = forward(lb)
        fu = forward(ub)
        # PR-1 audit I1: inward-guard the point check so float rounding at
        # box endpoints can't move them past the true argmin.
        contains_min = (
            (lb <= x_min + _GELU_XMIN_GUARD)
            & (ub >= x_min - _GELU_XMIN_GUARD)
        )
        out_lb = np.where(contains_min, f_min, np.minimum(fl, fu))
        out_ub = np.maximum(fl, fu)
        out.append(Box(out_lb.reshape(-1, 1), out_ub.reshape(-1, 1)))
    return out


def gelu_box(input_boxes: List[Box]) -> List[Box]:
    """Sound Box reach for the exact (erf) GELU."""
    return _gelu_box_impl(input_boxes, _gelu, _GELU_X_MIN, _GELU_F_MIN)


def gelu_tanh_box(input_boxes: List[Box]) -> List[Box]:
    """Sound Box reach for the tanh-approximation GELU (GPT-2 form).

    T0-3 (audit C5): nn.GELU(approximate='tanh') was previously routed to
    ``gelu_box`` which uses the erf-form ``F_MIN = -0.16997``. The tanh form
    dips lower (~-0.170041) so the erf floor was strictly ABOVE the true
    minimum, producing an unsound (under-approximating) box reach that
    excluded true outputs near the dip.
    """
    return _gelu_box_impl(
        input_boxes, _gelu_tanh, _GELU_TANH_X_MIN, _GELU_TANH_F_MIN,
    )


def gelu_star_approx(input_stars: List[Star]) -> List[Star]:
    """Box-lifted Star reach for the exact (erf) GELU, preserving ImageStar shape."""

    def _box(lb: np.ndarray, ub: np.ndarray):
        box = gelu_box([Box(lb, ub)])[0]
        return box.lb, box.ub

    return apply_box_lift_star(input_stars, _box)


def gelu_zono(input_zonos):
    """Sound (box-lifted) Zono reach for the exact (erf) GELU."""
    from n2v.sets import Zono
    out = []
    for z in input_zonos:
        lb, ub = z.get_bounds()
        box = gelu_box([Box(
            np.asarray(lb).reshape(-1, 1), np.asarray(ub).reshape(-1, 1),
        )])[0]
        out.append(Zono.from_bounds(box.lb, box.ub))
    return out


def gelu_tanh_zono(input_zonos):
    """Sound (box-lifted) Zono reach for the tanh-approximation GELU."""
    from n2v.sets import Zono
    out = []
    for z in input_zonos:
        lb, ub = z.get_bounds()
        box = gelu_tanh_box([Box(
            np.asarray(lb).reshape(-1, 1), np.asarray(ub).reshape(-1, 1),
        )])[0]
        out.append(Zono.from_bounds(box.lb, box.ub))
    return out


def _box_lift_lb_ub(input_set):
    """Fast IBP (lb, ub). Hex/Oct take a solver arg on get_bounds; use
    their zero-arg ``estimate_ranges`` IBP method instead."""
    from n2v.sets import Hexatope, Octatope

    if isinstance(input_set, (Hexatope, Octatope)):
        lb, ub = input_set.estimate_ranges()
    elif hasattr(input_set, "get_bounds"):
        lb, ub = input_set.get_bounds()
    else:
        lb, ub = input_set.get_ranges()
    return np.asarray(lb).reshape(-1, 1), np.asarray(ub).reshape(-1, 1)


def gelu_hexatope(input_hexatopes):
    """Sound (box-lifted) Hexatope reach for the exact (erf) GELU."""
    from n2v.sets import Hexatope
    out = []
    for h in input_hexatopes:
        lb, ub = _box_lift_lb_ub(h)
        box = gelu_box([Box(lb, ub)])[0]
        out.append(Hexatope.from_bounds(box.lb, box.ub))
    return out


def gelu_tanh_hexatope(input_hexatopes):
    """Sound (box-lifted) Hexatope reach for the tanh-approximation GELU."""
    from n2v.sets import Hexatope
    out = []
    for h in input_hexatopes:
        lb, ub = _box_lift_lb_ub(h)
        box = gelu_tanh_box([Box(lb, ub)])[0]
        out.append(Hexatope.from_bounds(box.lb, box.ub))
    return out


def gelu_octatope(input_octatopes):
    """Sound (box-lifted) Octatope reach for the exact (erf) GELU."""
    from n2v.sets import Octatope
    out = []
    for o in input_octatopes:
        lb, ub = _box_lift_lb_ub(o)
        box = gelu_box([Box(lb, ub)])[0]
        out.append(Octatope.from_bounds(box.lb, box.ub))
    return out


def gelu_tanh_octatope(input_octatopes):
    """Sound (box-lifted) Octatope reach for the tanh-approximation GELU."""
    from n2v.sets import Octatope
    out = []
    for o in input_octatopes:
        lb, ub = _box_lift_lb_ub(o)
        box = gelu_tanh_box([Box(lb, ub)])[0]
        out.append(Octatope.from_bounds(box.lb, box.ub))
    return out


def gelu_tanh_star_approx(input_stars: List[Star]) -> List[Star]:
    """Box-lifted Star reach for the tanh-approximation GELU.

    Star path is box-lifted today; a CROWN-style linear relaxation would be a
    tighter follow-up (see PR12_FIX_LIST T4 polish).
    """

    def _box(lb: np.ndarray, ub: np.ndarray):
        box = gelu_tanh_box([Box(lb, ub)])[0]
        return box.lb, box.ub

    return apply_box_lift_star(input_stars, _box)

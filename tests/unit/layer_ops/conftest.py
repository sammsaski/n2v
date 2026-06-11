"""Pytest configuration and fixtures for layer operation tests."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from n2v.sets import Star, Zono, Box, ImageStar, ImageZono, Hexatope, Octatope


# ============================================================================
# Set Fixtures
# ============================================================================

@pytest.fixture
def simple_star():
    """Create a simple 3D Star set."""
    V = np.array([[1.0, 0.1, 0.0],
                  [0.0, 0.0, 0.2],
                  [0.0, 0.1, 0.0]])
    C = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    d = np.array([[1.0], [1.0]])
    pred_lb = np.array([[0.0], [0.0]])
    pred_ub = np.array([[1.0], [1.0]])
    return Star(V, C, d, pred_lb, pred_ub)


@pytest.fixture
def simple_zono():
    """Create a simple 3D Zonotope."""
    c = np.array([[0.5], [0.5], [0.5]])
    V = np.array([[0.1, 0.0],
                  [0.0, 0.1],
                  [0.05, 0.05]])
    return Zono(c, V)


@pytest.fixture
def simple_box():
    """Create a simple 3D Box."""
    lb = np.array([[0.0], [0.0], [0.0]])
    ub = np.array([[1.0], [1.0], [1.0]])
    return Box(lb, ub)


@pytest.fixture
def simple_hexatope():
    """Create a simple 3D Hexatope."""
    lb = np.array([[0.0], [0.0], [0.0]])
    ub = np.array([[1.0], [1.0], [1.0]])
    return Hexatope.from_bounds(lb, ub)


@pytest.fixture
def simple_octatope():
    """Create a simple 3D Octatope."""
    lb = np.array([[0.0], [0.0], [0.0]])
    ub = np.array([[1.0], [1.0], [1.0]])
    return Octatope.from_bounds(lb, ub)


@pytest.fixture
def simple_image_star():
    """Create a simple 4x4x1 ImageStar."""
    lb = np.zeros((4, 4, 1))
    ub = np.ones((4, 4, 1))
    return ImageStar.from_bounds(lb, ub, height=4, width=4, num_channels=1)


@pytest.fixture
def simple_image_zono():
    """Create a simple 4x4x1 ImageZono."""
    lb = np.zeros((4, 4, 1))
    ub = np.ones((4, 4, 1))
    return ImageZono.from_bounds(lb, ub, height=4, width=4, num_channels=1)


# ============================================================================
# Assertion Helpers
# ============================================================================

def assert_star_valid(star):
    """Assert that a Star set is valid."""
    assert star.V is not None
    assert star.C is not None
    assert star.d is not None
    assert star.V.shape[1] == star.nVar + 1
    assert star.C.shape[1] == star.nVar


def assert_zono_valid(zono):
    """Assert that a Zonotope is valid."""
    assert zono.c is not None
    assert zono.V is not None
    assert len(zono.c.shape) == 2
    assert zono.c.shape[1] == 1


def assert_hexatope_valid(hexatope):
    """Assert that a Hexatope is valid."""
    assert hexatope.center is not None
    assert hexatope.generators is not None
    assert hexatope.dcs is not None
    assert hexatope.generators.shape[0] == hexatope.dim
    assert hexatope.generators.shape[1] == hexatope.nVar


def assert_octatope_valid(octatope):
    """Assert that an Octatope is valid."""
    assert octatope.center is not None
    assert octatope.generators is not None
    assert octatope.utvpi is not None
    assert octatope.generators.shape[0] == octatope.dim
    assert octatope.generators.shape[1] == octatope.nVar


# Make helpers available via pytest
pytest.assert_star_valid = assert_star_valid
pytest.assert_zono_valid = assert_zono_valid
pytest.assert_hexatope_valid = assert_hexatope_valid
pytest.assert_octatope_valid = assert_octatope_valid


# ============================================================================
# Reach-vs-forward Monte-Carlo containment helper (PR-1 audit N11 / M4)
# ============================================================================
#
# Many existing tests only assert output SHAPE or FINITENESS -- a stub that
# returned a constant Box([0], [1]) of the right shape would pass. The audit's
# central N-series criticism is that we lack a shared way to check
# *concrete-forward containment*: random inputs in the reach-set domain must
# map under ``layer.forward`` to outputs that lie inside the reach-set's
# get_bounds envelope.
#
# This helper makes the check a one-liner. Callers provide the layer (a torch
# nn.Module), the input box (per-element bounds), the reach function
# (e.g. ``layer_ops.linear_box``), and optional ``input_reshape`` /
# ``output_reshape`` callables for layers that operate on shapes other than
# flat (B, dim).


def assert_reach_contains_forward(
    layer,
    input_lb_flat,
    input_ub_flat,
    reach_fn,
    *,
    n_samples: int = 32,
    input_shape=None,
    set_type=None,
    output_flat_dim=None,
    rtol: float = 0.0,
    atol: float = 1e-6,
    seed: int = 0,
    reach_kwargs=None,
):
    """Monte-Carlo soundness check: ``reach_fn`` output must contain
    every ``layer.forward(sample)`` for samples drawn uniformly from
    ``[input_lb_flat, input_ub_flat]``.

    Parameters
    ----------
    layer : torch.nn.Module
        The forward function under test.
    input_lb_flat, input_ub_flat : 1-D arrays of identical length
        Per-element bounds on the flat input.
    reach_fn : callable
        Signature ``reach_fn(layer, [input_set]) -> [output_set]`` (matches
        every helper in ``n2v/nn/layer_ops/*_reach.py``).
    n_samples : int
        Number of Monte-Carlo samples. 32 is a sensible default for tests.
    input_shape : tuple[int, ...] or None
        Shape to ``view()`` the flat sample into before calling ``layer``.
        E.g. ``(1, n_tokens, dim)`` for a transformer block. If None, the
        sample is fed flat (1, total_dim).
    set_type : type, default Box
        Set class to build the input via ``from_bounds``. Box covers most
        cases; pass ``Star`` or ``Zono`` when the reach function only
        accepts those.
    output_flat_dim : int or None
        If the layer's output isn't flat-equivalent (e.g. has spatial
        dimensions), provide the expected flat dim to flatten with.
    reach_kwargs : dict or None
        Extra kwargs forwarded to ``reach_fn`` (e.g. ``expected_n_tokens``,
        ``image_shape``).
    """
    if set_type is None:
        from n2v.sets import Box as _Box
        set_type = _Box
    if reach_kwargs is None:
        reach_kwargs = {}

    input_lb_flat = np.asarray(input_lb_flat, dtype=np.float64).reshape(-1)
    input_ub_flat = np.asarray(input_ub_flat, dtype=np.float64).reshape(-1)
    assert input_lb_flat.shape == input_ub_flat.shape, (
        "lb/ub must have identical shape"
    )

    lb_col = input_lb_flat.reshape(-1, 1)
    ub_col = input_ub_flat.reshape(-1, 1)
    from n2v.sets import Box as _Box  # noqa: WPS433 local import
    if set_type is _Box:
        input_set = _Box(lb_col, ub_col)
    else:
        input_set = set_type.from_bounds(lb_col, ub_col)
    out_sets = reach_fn(layer, [input_set], **reach_kwargs)
    assert len(out_sets) == 1, (
        f"reach_fn returned {len(out_sets)} sets; expected 1"
    )
    out_set = out_sets[0]
    if hasattr(out_set, "get_bounds"):
        out_lb, out_ub = out_set.get_bounds()
    else:
        out_lb, out_ub = out_set.estimate_ranges()
    out_lb = np.asarray(out_lb).flatten()
    out_ub = np.asarray(out_ub).flatten()

    rng = np.random.default_rng(seed)
    layer.eval()
    with torch.no_grad():
        for k in range(n_samples):
            x = rng.uniform(input_lb_flat, input_ub_flat).astype(np.float32)
            if input_shape is not None:
                x_t = torch.from_numpy(x).reshape(input_shape)
            else:
                x_t = torch.from_numpy(x).unsqueeze(0)
            y_t = layer(x_t).detach().cpu().numpy().flatten()
            if output_flat_dim is not None:
                y_t = y_t[:output_flat_dim]
            tol = atol + rtol * np.maximum(np.abs(out_lb), np.abs(out_ub))
            below = (y_t + tol) < out_lb
            above = (y_t - tol) > out_ub
            if np.any(below) or np.any(above):
                bad_idx = int(np.argmax(below | above))
                raise AssertionError(
                    f"sample {k}: forward output index {bad_idx} = "
                    f"{y_t[bad_idx]:.6g} outside reach bounds "
                    f"[{out_lb[bad_idx]:.6g}, {out_ub[bad_idx]:.6g}] "
                    f"(tol {tol[bad_idx]:.2g})"
                )


pytest.assert_reach_contains_forward = assert_reach_contains_forward

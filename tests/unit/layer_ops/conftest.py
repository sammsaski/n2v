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

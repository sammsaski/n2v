"""
Tests for VNN-LIB property file loading.
"""

import pytest
import numpy as np
import tempfile
import os
from n2v.utils.load_vnnlib import load_vnnlib
from n2v.sets import HalfSpace


class TestLoadVNNLib:
    """Tests for load_vnnlib function."""

    def test_simple_input_bounds(self):
        """Test loading simple input bounds."""
        vnnlib_content = """
; Simple test property
; Input: 2D, Output: 2D

; Declare input variables
(declare-const X_0 Real)
(declare-const X_1 Real)

; Declare output variables
(declare-const Y_0 Real)
(declare-const Y_1 Real)

; Define input bounds: 0 <= X_0 <= 1, 0 <= X_1 <= 1
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
(assert (>= X_1 0.0))
(assert (<= X_1 1.0))

; Define output property: Y_0 <= 10
(assert (<= Y_0 10.0))
"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vnnlib', delete=False) as f:
            f.write(vnnlib_content)
            temp_file = f.name

        try:
            prop = load_vnnlib(temp_file)

            # Check input bounds
            assert prop['lb'].shape == (2,)
            assert prop['ub'].shape == (2,)
            np.testing.assert_array_almost_equal(prop['lb'], [0.0, 0.0])
            np.testing.assert_array_almost_equal(prop['ub'], [1.0, 1.0])

            # Check output property
            assert len(prop['prop']) == 1
            assert prop['prop'][0]['Hg'] is not None
            assert isinstance(prop['prop'][0]['Hg'], HalfSpace)

        finally:
            os.unlink(temp_file)

    def test_multiple_output_constraints(self):
        """Test loading multiple output constraints (AND)."""
        vnnlib_content = """
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0.0))
(assert (<= X_0 1.0))

; Multiple output constraints (implicit AND)
(assert (<= Y_0 10.0))
(assert (<= Y_1 5.0))
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vnnlib', delete=False) as f:
            f.write(vnnlib_content)
            temp_file = f.name

        try:
            prop = load_vnnlib(temp_file)

            # Check input bounds
            assert prop['lb'].shape == (1,)
            assert prop['ub'].shape == (1,)

            # Check output property - should have 2 constraints combined
            assert len(prop['prop']) == 1
            halfspace = prop['prop'][0]['Hg']
            assert halfspace.G.shape[0] == 2  # 2 constraints

        finally:
            os.unlink(temp_file)

    def test_output_comparison_constraints(self):
        """Test output constraints comparing two output variables."""
        vnnlib_content = """
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0.0))
(assert (<= X_0 1.0))

; Y_0 should be greater than Y_1: Y_1 <= Y_0 or Y_1 - Y_0 <= 0
(assert (<= Y_1 Y_0))
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vnnlib', delete=False) as f:
            f.write(vnnlib_content)
            temp_file = f.name

        try:
            prop = load_vnnlib(temp_file)

            # Check that constraint involves both Y_0 and Y_1
            halfspace = prop['prop'][0]['Hg']
            # The constraint Y_1 <= Y_0 becomes Y_1 - Y_0 <= 0
            # So G should have form [?, ?] where not both are zero
            assert halfspace.G.shape[1] == 2  # 2D output space
            assert not np.all(halfspace.G == 0)

        finally:
            os.unlink(temp_file)

    def test_comments_and_empty_lines(self):
        """Test that comments and empty lines are properly ignored."""
        vnnlib_content = """
; This is a comment

; More comments
(declare-const X_0 Real)

; Another comment
(declare-const Y_0 Real)

; Input bounds
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))

; Output constraint
(assert (<= Y_0 5.0))
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vnnlib', delete=False) as f:
            f.write(vnnlib_content)
            temp_file = f.name

        try:
            prop = load_vnnlib(temp_file)

            # Should work despite comments
            assert prop['lb'].shape == (1,)
            assert prop['ub'].shape == (1,)
            assert len(prop['prop']) == 1

        finally:
            os.unlink(temp_file)

    def test_multiline_assertion(self):
        """Test assertions that span multiple lines."""
        vnnlib_content = """
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 0.0))
(assert
    (<= X_0 1.0)
)

(assert
    (<= Y_0
        5.0
    )
)
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vnnlib', delete=False) as f:
            f.write(vnnlib_content)
            temp_file = f.name

        try:
            prop = load_vnnlib(temp_file)

            # Should properly handle multiline
            np.testing.assert_array_almost_equal(prop['lb'], [0.0])
            np.testing.assert_array_almost_equal(prop['ub'], [1.0])
            assert len(prop['prop']) == 1

        finally:
            os.unlink(temp_file)

    def test_higher_dimensional_input(self):
        """Test with higher dimensional input space."""
        vnnlib_content = """
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)
(declare-const X_3 Real)

(declare-const Y_0 Real)

(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (>= X_1 -2.0))
(assert (<= X_1 2.0))
(assert (>= X_2 0.0))
(assert (<= X_2 0.5))
(assert (>= X_3 -0.5))
(assert (<= X_3 0.5))

(assert (<= Y_0 10.0))
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vnnlib', delete=False) as f:
            f.write(vnnlib_content)
            temp_file = f.name

        try:
            prop = load_vnnlib(temp_file)

            # Check 4D input space
            assert prop['lb'].shape == (4,)
            assert prop['ub'].shape == (4,)
            np.testing.assert_array_almost_equal(prop['lb'], [-1.0, -2.0, 0.0, -0.5])
            np.testing.assert_array_almost_equal(prop['ub'], [1.0, 2.0, 0.5, 0.5])

        finally:
            os.unlink(temp_file)

    def test_output_halfspace_structure(self):
        """Test that output HalfSpace has correct structure."""
        vnnlib_content = """
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (>= X_0 0.0))
(assert (<= X_0 1.0))

(assert (<= Y_0 10.0))
(assert (>= Y_1 -5.0))
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vnnlib', delete=False) as f:
            f.write(vnnlib_content)
            temp_file = f.name

        try:
            prop = load_vnnlib(temp_file)

            halfspace = prop['prop'][0]['Hg']

            # Should have 2 constraints (Y_0 <= 10 and Y_1 >= -5)
            assert halfspace.G.shape == (2, 2)  # 2 constraints, 2D output
            assert halfspace.g.shape == (2, 1)  # 2 constraint values
            assert halfspace.dim == 2

            # Check constraint values
            # Y_0 <= 10 becomes [1, 0] @ y <= 10
            # Y_1 >= -5 becomes [-1, 0] @ y <= 5
            expected_G = np.array([[1, 0], [0, -1]], dtype=np.float32)
            expected_g = np.array([[10], [5]], dtype=np.float32)

            np.testing.assert_array_almost_equal(halfspace.G, expected_G)
            np.testing.assert_array_almost_equal(halfspace.g, expected_g)

        finally:
            os.unlink(temp_file)


class TestLoadVNNLibErrorHandling:
    """Test error handling in load_vnnlib."""

    def test_nonexistent_file(self):
        """Test that nonexistent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            load_vnnlib('/nonexistent/path/to/file.vnnlib')

    def test_empty_file(self):
        """Test handling of empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vnnlib', delete=False) as f:
            f.write("")
            temp_file = f.name

        try:
            prop = load_vnnlib(temp_file)
            # Should return empty/default structure
            assert prop['lb'] is None or len(prop['lb']) == 0
        finally:
            os.unlink(temp_file)

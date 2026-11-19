"""
Tests for VNN-LIB property file loading.
"""

import pytest
import numpy as np
import tempfile
import os
from n2v.utils.load_vnnlib import load_vnnlib
from n2v.sets import HalfSpace
from n2v.utils.vnnlib2 import VNNLibParseError, evaluate_nonlinear
from tests.known_bugs import known_bug


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
        """An empty/unparseable spec must raise, not silently return empty.

        Silently returning an empty spec means 'verify nothing' — the
        verifier would report a vacuous result. ``load_vnnlib`` guards
        against this and raises.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vnnlib', delete=False) as f:
            f.write("")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match='empty'):
                load_vnnlib(temp_file)
        finally:
            os.unlink(temp_file)


def _load_text(tmp_path, text):
    path = tmp_path / "spec.vnnlib"
    path.write_text(text)
    return load_vnnlib(str(path))


class TestFormatDispatch:
    """``load_vnnlib`` picks the reader from the text and returns the 2.0
    nonlinear and relational forms as they are. The expected values are
    written out by hand from each small file."""

    def test_v2_file_without_a_version_line_is_read_by_the_2_0_reader(
            self, tmp_path):
        """The docstring routes a file with either marker, ``vnnlib-version``
        or ``declare-network``, to the 2.0 reader. A test that needs both
        hands this file to the 1.0 reader, which finds no ``declare-const``
        and refuses it as empty."""
        prop = _load_text(tmp_path, (
            "(declare-network N\n"
            "  (declare-input X float32 [1,2])\n"
            "  (declare-output Y float32 [1,1])\n)\n"
            "(assert (>= X[0,0] 0.25))\n(assert (<= X[0,0] 0.75))\n"
            "(assert (>= X[0,1] -2.0))\n(assert (<= X[0,1] 3.0))\n"
            "(assert (>= Y[0,0] 0.875))\n"))
        (pair,) = prop["pairs"]
        assert np.asarray(pair["lb"], dtype=np.float64).ravel().tolist() == [
            0.25, -2.0]
        assert np.asarray(pair["ub"], dtype=np.float64).ravel().tolist() == [
            0.75, 3.0]
        (group,) = pair["prop"]
        assert np.asarray(group["Hg"].G).tolist() == [[-1.0]]
        assert np.asarray(group["Hg"].g).ravel().tolist() == [-0.875]

    @known_bug(6, "a 1.0 file whose comment holds the text 'vnnlib-version' "
                  "or 'declare-network' is handed to the 2.0 reader, which "
                  "refuses it (no declare-input): the version is detected by "
                  "searching the raw text, comments included, so a legal "
                  "1.0 file gets no verdict", raises=VNNLibParseError)
    def test_v1_file_whose_comment_names_a_2_0_keyword_is_read_by_the_1_0_reader(
            self, tmp_path):
        """A comment has no meaning; the reader is chosen by what the file
        states, here ``declare-const``."""
        prop = _load_text(tmp_path, (
            "; written by a tool that also writes declare-network files\n"
            "(declare-const X_0 Real)\n(declare-const Y_0 Real)\n"
            "(assert (>= X_0 0.25))\n(assert (<= X_0 0.75))\n"
            "(assert (>= Y_0 0.875))\n"))
        (pair,) = prop["pairs"]
        assert np.asarray(pair["lb"], dtype=np.float64).ravel().tolist() == [
            0.25]
        assert np.asarray(pair["ub"], dtype=np.float64).ravel().tolist() == [
            0.75]
        (group,) = pair["prop"]
        assert np.asarray(group["Hg"].G).tolist() == [[-1.0]]
        assert np.asarray(group["Hg"].g).ravel().tolist() == [-0.875]

    def test_nonlinear_spec_is_returned_as_parsed(self, tmp_path):
        """A product of variables has no half-space reading: ``load_vnnlib``
        returns the 2.0 nonlinear parse (``prop`` None, the assertions kept)
        before its emptiness guard and its pairs, which would refuse it. The
        kept assertions must decide y*y >= 4 as the text does."""
        prop = _load_text(tmp_path, (
            "(vnnlib-version <2.0>)\n(declare-network N\n"
            "  (declare-input X real [1,1])\n"
            "  (declare-output Y real [1,1])\n)\n"
            "(assert (>= X[0,0] 1.0))\n(assert (<= X[0,0] 2.0))\n"
            "(assert (>= (* Y[0,0] Y[0,0]) 4.0))\n"))
        assert prop["format"] == "nonlinear"
        assert prop["prop"] is None and "pairs" not in prop
        assert np.asarray(prop["lb"]).ravel().tolist() == [1.0]
        assert np.asarray(prop["ub"]).ravel().tolist() == [2.0]
        for y in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
            assert evaluate_nonlinear(prop, np.array([1.5]), np.array([y])) \
                is (y * y >= 4.0), y

    def test_relational_spec_is_returned_with_its_coupled_input_unbounded(
            self, tmp_path):
        """A two-network file bounds X_g only through the coupling
        X_f == X_g, so its box leaves X_g unbounded. ``load_vnnlib`` returns
        it after its emptiness guard and before the pairs validator, which
        would refuse the infinite bound; the box is the one stated."""
        prop = _load_text(tmp_path, (
            "(vnnlib-version <2.0>)\n"
            "(declare-network f\n  (declare-input X_f real [1])\n"
            "  (declare-output Y_f real [1])\n)\n"
            "(declare-network g\n  (equal-to f)\n"
            "  (declare-input X_g real [1])\n"
            "  (declare-output Y_g real [1])\n)\n"
            "(assert (and (>= X_f[0] 0.25) (<= X_f[0] 0.75)))\n"
            "(assert (== X_f[0] X_g[0]))\n"
            "(assert (<= Y_f[0] Y_g[0]))\n"))
        assert prop["format"] == "relational" and "pairs" not in prop
        assert np.asarray(prop["lb"]).ravel().tolist() == [0.25, -np.inf]
        assert np.asarray(prop["ub"]).ravel().tolist() == [0.75, np.inf]

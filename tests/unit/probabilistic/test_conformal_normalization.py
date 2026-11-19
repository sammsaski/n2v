"""Tests for compute_normalization: the formula of Hashemi et al. 2025, eq. 6.

These tests pin the formula as the paper writes it: τ_k is the largest error
in output k over the training AND the calibration samples (Finding 1 of the
Hashemi audit moved the code to it from a training-only τ).

Status: that formula is DEFERRED-BUGS.md bug 1. A τ computed from the
calibration samples breaks the exchangeability the coverage statement rests
on, and the box under-covers with the naive surrogate at 50 and 200 outputs.
The owner deferred the fix (2026-10-08: raise it with the paper's authors
first), so today's formula is kept and these tests document it. They say what
the code computes, not that the result is a valid guarantee; the guarantee is
tested in tests/contracts/test_probabilistic_contract.py
(``test_conformal_box_misses_no_more_often_than_it_states``). When bug 1 is
fixed, the tests here that assert a τ taken from the calibration errors change
with it, or move to the explicitly named paper option if one is kept.

Audit reference:
  .claude/research/flow-matching-probabilistic-reach/_archive/audits/
    2026-04-13-hashemi-clipping-block-audit.md  (Finding 1)
"""
import warnings

import numpy as np
import pytest

from n2v.probabilistic.conformal import (
    compute_normalization,
    compute_nonconformity_scores,
    compute_threshold,
    compute_inflation,
    conformal_inference,
)


def test_tau_uses_calibration_errors_when_training_errors_zero():
    """Clipping-block case: training errors are zero, so the paper's τ comes
    from the calibration errors alone (the formula of deferred bug 1)."""
    train_err = np.zeros((10, 2))
    calib_err = np.array([[1.0, 0.10], [0.5, 0.05], [0.3, 0.02]])
    tau = compute_normalization(
        training_errors=train_err, calibration_errors=calib_err
    )
    # Per paper spec: τ_k = max(τ*, max_train_k, max_calib_k) = max(τ*, max_calib_k)
    expected = np.array([1.0, 0.10])
    np.testing.assert_allclose(tau, expected, rtol=1e-6)


def test_tau_uses_max_of_train_and_calib_per_dim():
    """Naive case: nonzero training errors. The paper's τ_k is the max over
    both per dim (the formula of deferred bug 1)."""
    train_err = np.array([[0.8, 0.20], [0.4, 0.15]])
    calib_err = np.array([[1.0, 0.10], [0.5, 0.05]])
    tau = compute_normalization(
        training_errors=train_err, calibration_errors=calib_err
    )
    # Per dim: max(0.8, 0.4, 1.0, 0.5) = 1.0; max(0.20, 0.15, 0.10, 0.05) = 0.20
    expected = np.array([1.0, 0.20])
    np.testing.assert_allclose(tau, expected, rtol=1e-6)


def test_tau_respects_tau_star_floor():
    """If both train and calib errors are zero, τ_k = τ* per dim (no zero-div)."""
    train_err = np.zeros((10, 2))
    calib_err = np.zeros((10, 2))
    tau = compute_normalization(
        training_errors=train_err, calibration_errors=calib_err
    )
    assert np.all(tau >= 1e-10)
    # And finite — no infs or NaNs
    assert np.all(np.isfinite(tau))


def test_tau_calibration_only_does_not_collapse_to_floor():
    """Regression: with zero training errors, τ must NOT collapse to a uniform
    1e-10 floor when calibration errors carry the per-dim structure."""
    train_err = np.zeros((10, 2))
    calib_err = np.array([[1.0, 0.10], [0.5, 0.05], [0.3, 0.02]])
    tau = compute_normalization(
        training_errors=train_err, calibration_errors=calib_err
    )
    # The buggy pre-fix behavior would give tau == [1e-10, 1e-10].
    assert tau[0] > 1e-3, "tau_0 should reflect calibration-error scale, not 1e-10"
    assert tau[1] > 1e-3, "tau_1 should reflect calibration-error scale, not 1e-10"
    # And tau must be anisotropic when the calibration errors are.
    assert tau[0] > tau[1], "per-dim structure must be preserved"


def test_legacy_training_only_signature_raises_deprecation_warning():
    """Backward compat: callers passing only `training_errors` get a
    DeprecationWarning. The buggy training-only τ is still returned so
    external code does not crash, but the warning surfaces the issue.
    """
    train_err = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tau = compute_normalization(train_err)
    assert any(
        issubclass(rec.category, DeprecationWarning) for rec in w
    ), "expected DeprecationWarning for training-only call"
    # The training-only fallback still returns the per-dim training maxima.
    assert tau.shape == (3,)
    np.testing.assert_allclose(tau, np.array([1.0, 2.0, 3.0]), rtol=1e-6)


def test_internal_callers_dont_emit_deprecation_warning():
    """`conformal_inference` already passes both args; verify no warning leaks
    through the supported path."""
    rng = np.random.default_rng(42)
    train_err = rng.standard_normal((50, 5)) * 0.1
    calib_err = rng.standard_normal((100, 5)) * 0.1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = conformal_inference(
            training_errors=train_err,
            calibration_errors=calib_err,
            m=100,
            ell=99,
            epsilon=0.01,
        )
    deprecation_warnings = [
        rec for rec in w if issubclass(rec.category, DeprecationWarning)
    ]
    assert not deprecation_warnings, (
        f"conformal_inference emitted DeprecationWarning(s): "
        f"{[str(r.message) for r in deprecation_warnings]}"
    )


def test_audit_worked_example_2d():
    """Reproduce the audit's worked 2D example end-to-end (Finding 1).

    Three calibration errors (clipping-block: zero training errors):
        err = [[1.0, 0.10], [0.5, 0.05], [0.3, 0.02]]
    Paper-correct flow:
        τ = [1.0, 0.10]
        scores: R_1=1.0, R_2=0.5, R_3=0.3
        ell=2 -> threshold = 0.5
        sigma = [1.0*0.5, 0.10*0.5] = [0.5, 0.05]   (per-component)
    Buggy (pre-fix) flow gave [0.5, 0.5]            (uniform).
    """
    train_err = np.zeros((10, 2))
    calib_err = np.array([[1.0, 0.10], [0.5, 0.05], [0.3, 0.02]])

    tau = compute_normalization(
        training_errors=train_err, calibration_errors=calib_err
    )
    np.testing.assert_allclose(tau, [1.0, 0.10], rtol=1e-6)

    scores = compute_nonconformity_scores(calib_err, tau)
    np.testing.assert_allclose(np.sort(scores), [0.3, 0.5, 1.0], rtol=1e-6)

    threshold = compute_threshold(scores, ell=2)
    assert threshold == pytest.approx(0.5, rel=1e-6)

    inflation = compute_inflation(tau, threshold)
    np.testing.assert_allclose(inflation, [0.5, 0.05], rtol=1e-6)
    # And specifically NOT the buggy uniform [0.5, 0.5]
    assert inflation[1] < 0.5


@pytest.mark.parametrize("factor, expected", [
    (0.5, [6.0, 1.015625]),
    (None, [6.0, 0.25]),
], ids=["factor-binds", "default-factor"])
def test_floor_is_the_factor_times_the_mean_absolute_training_error(
        factor, expected):
    """tau* = tau_star_factor * mean |training error| and tau = max(tau*,
    per-output maxima), by hand: the mean absolute training error is 2.03125,
    so with the factor 0.5 the floor 1.015625 binds in output 1 (whose
    largest error is 0.25) and not in output 0 (6); with the default factor
    1e-5 it binds nowhere. Dyadic data: exact. Catches a factor that is
    ignored or replaced, a floor taken from the largest instead of the mean
    error, and a floor that is not applied."""
    train_err = np.array([[2.0, -0.0625], [-6.0, 0.0625]])
    calib_err = np.array([[1.0, 0.125], [-0.5, -0.25]])
    options = {} if factor is None else {"tau_star_factor": factor}
    tau = compute_normalization(train_err, calib_err, **options)
    assert tau.tolist() == expected


def test_one_dimensional_calibration_errors_are_one_sample():
    """A 1-D calibration error array is one sample with one error per
    output: tau = per-output max(|train|, |calib|) = [1, 2, 0.5], by hand.
    Catches a 1-D array read as a column, which takes the largest calibration
    error of all outputs (2) for every output."""
    train_err = np.array([[1.0, 0.0, 0.5], [0.0, 0.25, 0.0]])
    calib_err = np.array([0.5, -2.0, -0.25])
    tau = compute_normalization(train_err, calib_err)
    assert tau.tolist() == [1.0, 2.0, 0.5]

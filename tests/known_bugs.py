"""Markers for tests that fail on today's code because of a known defect.

A contract test states the behaviour DESIGN promises. Where the code does not
keep the promise yet, the test is still written to the promise and carries a
strict ``xfail`` that names the defect and the PLAN.md step that fixes it. When
that step lands the test passes, the strict marker turns the pass into a
failure, and the marker is removed in the same change.

``raises`` names the one way the test is allowed to fail today, so the marker
absorbs the documented failure and nothing else: a wrong answer is an
``AssertionError``, a call that should refuse and does not is ``DID_NOT_RAISE``
(what ``pytest.raises`` raises when nothing was raised), and a crash is its own
exception type. Any other failure is reported as a failure.
"""

import pytest

# What ``pytest.raises`` raises when the block raised nothing.
DID_NOT_RAISE = pytest.fail.Exception


def known_bug(step, defect, raises=AssertionError):
    """A strict xfail for ``defect``, fixed by PLAN step ``step``."""
    return pytest.mark.xfail(
        strict=True, raises=raises,
        reason=f"{defect} (PLAN step {step})")


def deferred_bug(number, defect, raises=AssertionError):
    """A strict xfail for a bug the owner deferred (DEFERRED-BUGS.md)."""
    return pytest.mark.xfail(
        strict=True, raises=raises,
        reason=f"{defect} (DEFERRED-BUGS.md bug {number}; owner decision "
               f"2026-10-08: documented, not fixed in the redesign)")

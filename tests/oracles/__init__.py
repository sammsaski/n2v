"""Oracle test framework for layer reachability.

Each ``<layer>.py`` module in this package provides one or more reachability
oracles. An oracle samples concrete points from an input set, pushes them
through the concrete PyTorch layer, and asserts every sample lies inside
the (union of) reachable output set(s).

This complements the existing ``tests/unit/layer_ops`` and
``tests/soundness`` suites: unit tests check shape/numeric expectations on
known-small problems, soundness tests check ground-truth bounds, oracles
provide random-sample containment evidence.

Imported by ``tests/soundness/test_soundness_<name>.py`` for new layers
added during the nnVLA port.
"""

from tests.oracles._framework import (
    assert_set_contains_pushforward,
    sample_from_set,
    contains,
)

__all__ = [
    "assert_set_contains_pushforward",
    "sample_from_set",
    "contains",
]


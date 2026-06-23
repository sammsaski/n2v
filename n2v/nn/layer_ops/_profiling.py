"""Shared profiler-counter helpers for layer reachability ops.

Single home for deriving profiler counters from reach state, so the derivation
isn't copy-pasted (and silently drifting) across relu/leakyrelu/sigmoid/tanh and
the zono/hexatope/octatope paths. All are no-ops when profiling is disabled.
"""

from n2v.profiling import count, is_enabled


def record_layer_neurons(sets) -> None:
    """Record the *static* neuron count once per layer (== set dimension).

    Every set representation (Star, ImageStar, Zono, Box, Hexatope, Octatope)
    exposes ``.dim`` directly, so no conversion is needed.
    """
    if is_enabled() and sets:
        count("n_neurons", sets[0].dim)


def record_classification(dim: int, n_inactive: int, n_unstable: int) -> None:
    """Record the ReLU-style per-set neuron classification (no-op when off).

    inactive + unstable + (stable-)active == dim, so active is derived here once
    rather than re-computed by hand at each call site.
    """
    count("n_stable_inactive", n_inactive)
    count("n_unstable", n_unstable)
    count("n_stable_active", dim - n_inactive - n_unstable)


def record_relax_outcome(n_unstable: int, n_relaxed: int) -> None:
    """Partition the estimated-unstable neurons into relaxed vs LP-resolved.

    ``n_relaxed`` got triangle constraints; ``n_resolved`` were proved stable by
    the exact LP. The two partition the estimated-unstable set (sum to
    ``n_unstable``).
    """
    count("n_relaxed", n_relaxed)
    count("n_resolved", n_unstable - n_relaxed)

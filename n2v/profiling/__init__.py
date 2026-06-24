"""n2v runtime profiler — observation-only, multi-granularity timing + counters.

Public API::

    from n2v.profiling import profile, region, count, set_meta
    with profile(level="layer") as p:
        net.reach(input_set, method="approx")
    p.summary()

See ``.claude/research/perf-improvements/plans/2026-06-22-profiler-design.md``.
"""

from n2v.profiling.profiler import (
    ITEM,
    LAYER,
    OPERATION,
    PHASE,
    RUN,
    Profiler,
    ProfileResult,
    Record,
    count,
    is_enabled,
    profile,
    region,
    set_meta,
)

__all__ = [
    "profile",
    "region",
    "count",
    "set_meta",
    "is_enabled",
    "ProfileResult",
    "Record",
    "Profiler",
    "RUN",
    "PHASE",
    "LAYER",
    "OPERATION",
    "ITEM",
]

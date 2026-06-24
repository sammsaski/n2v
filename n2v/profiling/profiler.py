"""Observation-only, multi-granularity runtime profiler for n2v.

Records nested timed regions (run / phase / layer / operation / item) plus
domain counters (LP solves, population size, neuron classification, ...). It is
designed to be a true no-op when disabled: ``region(...)`` returns a shared
singleton context manager after a single global check, so the instrumented hot
path pays ~nothing when profiling is off.

See ``.claude/research/perf-improvements/plans/2026-06-22-profiler-design.md``
for the full spec and metric catalog, and the companion ``-test-spec.md``.

The profiler is purely observational: it must never change a verdict, a bound,
or control flow (pinned by the non-interference test T3).
"""

from __future__ import annotations

import csv
import io
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Granularity levels (ordered; a deeper level is skipped when the active
# profile's ``max_level`` is shallower).
# --------------------------------------------------------------------------- #
RUN, PHASE, LAYER, OPERATION, ITEM = 0, 1, 2, 3, 4

_LEVEL_NAMES = {
    RUN: "run",
    PHASE: "phase",
    LAYER: "layer",
    OPERATION: "operation",
    ITEM: "item",
}
_LEVEL_BY_NAME = {name: lvl for lvl, name in _LEVEL_NAMES.items()}


def _resolve_level(level) -> int:
    """Accept an int level or its name ('run'/'phase'/'layer'/...)."""
    if isinstance(level, str):
        try:
            return _LEVEL_BY_NAME[level]
        except KeyError:
            raise ValueError(
                f"Unknown profiler level {level!r}; "
                f"expected one of {sorted(_LEVEL_BY_NAME)}"
            )
    return int(level)


# --------------------------------------------------------------------------- #
# Record: one node of the region tree.
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    """A single timed region.

    ``wall_time``/``cpu_time`` are filled on close; ``counters`` and ``meta``
    accumulate during the region's lifetime.
    """

    name: str
    level: int
    parent: Optional["Record"] = None
    children: List["Record"] = field(default_factory=list)
    wall_time: float = 0.0
    cpu_time: float = 0.0
    counters: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    errored: bool = False
    # internal start markers
    _t0: float = 0.0
    _c0: float = 0.0

    @property
    def level_name(self) -> str:
        return _LEVEL_NAMES.get(self.level, str(self.level))

    @property
    def self_time(self) -> float:
        """Wall time not attributed to child regions."""
        return self.wall_time - sum(c.wall_time for c in self.children)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Record({self.name!r}, {self.level_name}, "
            f"wall={self.wall_time:.6g}s, children={len(self.children)})"
        )


# --------------------------------------------------------------------------- #
# Profiler: owns the region tree for one ``profile(...)`` scope.
# --------------------------------------------------------------------------- #
class Profiler:
    """Holds the region tree and a per-thread region stack for one run."""

    def __init__(self, max_level: int = OPERATION):
        self.max_level = max_level
        self.root = Record(name="run", level=RUN)
        self.root._t0 = time.perf_counter()
        self.root._c0 = time.process_time()
        self._local = threading.local()
        self._lock = threading.Lock()

    # -- per-thread region stack -------------------------------------------- #
    def _stack(self) -> List[Record]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            # New thread (e.g. a ThreadPool worker): root is the default parent.
            # NOTE: cross-pool parent linkage is handled when T11 lands.
            stack = [self.root]
            self._local.stack = stack
        return stack

    def current(self) -> Record:
        return self._stack()[-1]

    # -- region lifecycle --------------------------------------------------- #
    def open(self, name: str, level: int, meta: Optional[Dict[str, Any]]) -> Record:
        rec = Record(
            name=name,
            level=level,
            parent=self.current(),
            meta=dict(meta) if meta else {},
        )
        with self._lock:
            rec.parent.children.append(rec)
        self._stack().append(rec)
        rec._t0 = time.perf_counter()
        rec._c0 = time.process_time()
        return rec

    def close(self, rec: Record, errored: bool = False) -> None:
        rec.wall_time = time.perf_counter() - rec._t0
        rec.cpu_time = time.process_time() - rec._c0
        rec.errored = errored or rec.errored
        stack = self._stack()
        # Defensive: keep the stack balanced even if regions were closed out of
        # order (should not happen with the context-manager API).
        if stack and stack[-1] is rec:
            stack.pop()
        else:
            while stack and stack[-1] is not rec:
                stack.pop()
            if stack:
                stack.pop()

    # -- counters / metadata on the current region -------------------------- #
    # Locked: ``count()`` can fire from ThreadPool workers (e.g. parallel
    # per-dimension LP solves in ``Star._get_ranges_parallel``), and
    # ``d[k] = d.get(k,0)+n`` is a non-atomic read-modify-write that would lose
    # updates under the GIL. The lock is uncontended on the common single-thread
    # path. (Worker-thread counts still attribute to the run root, since each
    # thread starts a fresh region stack — a documented limitation; the run
    # TOTAL is correct.)
    def add_counter(self, name: str, n: int = 1) -> None:
        with self._lock:
            c = self.current().counters
            c[name] = c.get(name, 0) + n

    def set_meta(self, **kwargs: Any) -> None:
        with self._lock:
            self.current().meta.update(kwargs)

    def finalize(self) -> "ProfileResult":
        self.root.wall_time = time.perf_counter() - self.root._t0
        self.root.cpu_time = time.process_time() - self.root._c0
        return ProfileResult(self.root, self.max_level)


# --------------------------------------------------------------------------- #
# Global active profiler (one ``profile(...)`` scope at a time for M1).
# The instrumented hot path checks ``_active`` and short-circuits when None.
# --------------------------------------------------------------------------- #
_active: Optional[Profiler] = None


def is_enabled() -> bool:
    """True iff a ``profile(...)`` scope is currently active."""
    return _active is not None


class _NullRegion:
    """Shared no-op context manager returned on the disabled fast path."""

    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NULL = _NullRegion()


class _Region:
    """Active context manager for one region (only created when profiling)."""

    __slots__ = ("_prof", "_name", "_level", "_meta", "rec")

    def __init__(self, prof: Profiler, name: str, level: int, meta: Dict[str, Any]):
        self._prof = prof
        self._name = name
        self._level = level
        self._meta = meta
        self.rec: Optional[Record] = None

    def __enter__(self) -> Record:
        self.rec = self._prof.open(self._name, self._level, self._meta)
        return self.rec

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._prof.close(self.rec, errored=exc_type is not None)
        return False  # never suppress exceptions


def region(name: str, level: int = OPERATION, **meta: Any):
    """Open a timed region. No-op (returns a shared singleton) when disabled or
    when ``level`` is deeper than the active profile's ``max_level``."""
    prof = _active
    if prof is None or level > prof.max_level:
        return _NULL
    return _Region(prof, name, level, meta)


def count(name: str, n: int = 1) -> None:
    """Increment a counter on the current region. No-op when disabled."""
    prof = _active
    if prof is None:
        return
    prof.add_counter(name, n)


def set_meta(**kwargs: Any) -> None:
    """Attach metadata to the current region. No-op when disabled."""
    prof = _active
    if prof is None:
        return
    prof.set_meta(**kwargs)


# --------------------------------------------------------------------------- #
# profile(...): the user-facing scope. ``with profile() as p: ...; p.summary()``
# --------------------------------------------------------------------------- #
class profile:
    """Context manager that activates profiling for its scope.

    Example::

        with profile(level="layer") as p:
            net.reach(input_set, method="approx")
        print(p.summary())
    """

    def __init__(self, level="operation"):
        self.max_level = _resolve_level(level)
        self._prof: Optional[Profiler] = None
        self._prev: Optional[Profiler] = None
        self.result: Optional["ProfileResult"] = None

    def __enter__(self) -> "profile":
        global _active
        self._prev = _active
        self._prof = Profiler(self.max_level)
        _active = self._prof
        return self

    def __exit__(self, *exc) -> bool:
        global _active
        self.result = self._prof.finalize()
        _active = self._prev
        return False

    # -- convenience: delegate to the finalized result --------------------- #
    @property
    def root(self) -> Record:
        return self.result.root if self.result is not None else self._prof.root

    def records(self) -> List[Record]:
        return self.result.records()

    def find(self, name: str) -> List[Record]:
        return self.result.find(name)

    def subtree_counters(self, rec: Record) -> Dict[str, Any]:
        return self.result.subtree_counters(rec)

    def rollup(self) -> Dict[str, Any]:
        return self.result.rollup()

    def summary(self, rollup: bool = True) -> str:
        return self.result.summary(rollup=rollup)

    def to_dict(self) -> Dict[str, Any]:
        return self.result.to_dict()

    def to_json(self, indent: Optional[int] = 2) -> str:
        return self.result.to_json(indent=indent)

    def to_csv(self) -> str:
        return self.result.to_csv()


# --------------------------------------------------------------------------- #
# ProfileResult: read-only view over a finalized region tree.
# (summary()/to_json()/to_csv() arrive with tests T8/T9.)
# --------------------------------------------------------------------------- #
class ProfileResult:
    def __init__(self, root: Record, max_level: int):
        self.root = root
        self.max_level = max_level

    def records(self) -> List[Record]:
        """All records in pre-order (root first)."""
        out: List[Record] = []

        def _walk(r: Record) -> None:
            out.append(r)
            for c in r.children:
                _walk(c)

        _walk(self.root)
        return out

    def find(self, name: str) -> List[Record]:
        return [r for r in self.records() if r.name == name]

    def subtree_counters(self, rec: Record) -> Dict[str, Any]:
        """Sum a region's own counters with all of its descendants'."""
        agg: Dict[str, Any] = {}

        def _walk(r: Record) -> None:
            for k, v in r.counters.items():
                agg[k] = agg.get(k, 0) + v
            for c in r.children:
                _walk(c)

        _walk(rec)
        return agg

    def rollup(self) -> Dict[str, Any]:
        """Run-level aggregate: grand totals + a per-layer-type breakdown.

        Per-layer detail lives in the region tree (see ``summary``); this is the
        end-of-run view the tree rolls up to. Counters are summed across the
        whole run; ``n_neurons`` is the static per-layer size summed over the
        layer instances of each type; ``peak_population`` is the largest output
        population any single layer produced.
        """
        layers = [r for r in self.records() if r.level == LAYER]

        # Grand totals: each counter is recorded on exactly one region, so
        # summing every region's own counters double-counts nothing.
        totals: Dict[str, Any] = {}
        for r in self.records():
            for k, v in r.counters.items():
                totals[k] = totals.get(k, 0) + v

        by_type: Dict[str, Dict[str, Any]] = {}
        for r in layers:
            t = r.meta.get("layer_type", r.name)
            agg = by_type.setdefault(
                t, {"count": 0, "wall_time": 0.0, "counters": {}}
            )
            agg["count"] += 1
            agg["wall_time"] += r.wall_time
            for k, v in self.subtree_counters(r).items():
                agg["counters"][k] = agg["counters"].get(k, 0) + v

        # Share of total run wall time + per-type roofline intensity (the
        # meaningful per-kernel-ish number, vs the coarse run-level one below).
        total_wall = self.root.wall_time or 0.0
        for agg in by_type.values():
            agg["wall_pct"] = (
                100.0 * agg["wall_time"] / total_wall if total_wall > 0 else 0.0
            )
            tb = agg["counters"].get("set_bytes_out", 0)
            agg["arithmetic_intensity"] = (
                agg["counters"].get("flops", 0) / tb if tb > 0 else 0.0
            )

        peak_population = max(
            (r.counters.get("n_sets_out", 0) for r in layers), default=0
        )
        peak_set_bytes = max(
            (r.counters.get("set_bytes_out", 0) for r in layers), default=0
        )
        # Roofline arithmetic intensity (FLOPs per carried byte) -- a COARSE,
        # run-level aggregate over heterogeneous layers, not a per-kernel number:
        # the denominator sums every layer's carried bytes, so it mixes layers
        # of different intensities. For per-kernel roofline analysis, compute
        # flops/set_bytes_out per layer (both are in `by_layer_type`/the CSV
        # export). >0 guard avoids div-by-zero on LP-only (FLOP-free) runs.
        total_flops = totals.get("flops", 0)
        total_bytes = totals.get("set_bytes_out", 0)
        arithmetic_intensity = (
            total_flops / total_bytes if total_bytes > 0 else 0.0
        )
        return {
            "wall_time": self.root.wall_time,
            "n_layers": len(layers),
            "peak_population": peak_population,
            "peak_set_bytes": peak_set_bytes,
            "arithmetic_intensity": arithmetic_intensity,
            "totals": totals,
            "by_layer_type": by_type,
        }

    def summary(self, rollup: bool = True) -> str:
        """Indented region tree (wall/self time + each region's own counters),
        followed by a run-level rollup unless ``rollup=False``."""
        lines: List[str] = []

        def _fmt(rec: Record, depth: int) -> None:
            indent = "  " * depth
            ctr = " ".join(f"{k}={v}" for k, v in rec.counters.items())
            line = (
                f"{indent}{rec.name} [{rec.level_name}] "
                f"wall={rec.wall_time:.6f}s self={rec.self_time:.6f}s"
            )
            if ctr:
                line += f"  {ctr}"
            lines.append(line)
            for c in rec.children:
                _fmt(c, depth + 1)

        _fmt(self.root, 0)

        if rollup:
            lines.append(self._format_rollup(self.rollup()))
        return "\n".join(lines)

    @staticmethod
    def _format_rollup(ro: Dict[str, Any]) -> str:
        """Render rollup() as a compact, human-readable block."""
        out: List[str] = []
        out.append("=== run rollup ===")
        out.append(
            f"wall={ro['wall_time']:.6f}s  layers={ro['n_layers']}  "
            f"peak_population={ro['peak_population']}  "
            f"peak_set_bytes={ro['peak_set_bytes']}"
        )
        totals = ro["totals"]
        if totals:
            out.append(
                "totals: " + " ".join(f"{k}={v}" for k, v in totals.items())
            )
        for t, agg in ro["by_layer_type"].items():
            ctr = " ".join(f"{k}={v}" for k, v in agg["counters"].items())
            line = (
                f"  {t} x{agg['count']}  "
                f"wall={agg['wall_time']:.6f}s ({agg['wall_pct']:.1f}%)"
            )
            if ctr:
                line += f"  {ctr}"
            out.append(line)
        return "\n".join(out)

    # -- export (T9) -------------------------------------------------------- #
    def _record_to_dict(self, rec: Record) -> Dict[str, Any]:
        return {
            "name": rec.name,
            "level": rec.level,
            "level_name": rec.level_name,
            "wall_time": rec.wall_time,
            "cpu_time": rec.cpu_time,
            "self_time": rec.self_time,
            "errored": rec.errored,
            "counters": dict(rec.counters),
            "meta": dict(rec.meta),
            "children": [self._record_to_dict(c) for c in rec.children],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Nested, JSON-serializable view of the region tree + rollup."""
        return {
            "max_level": self.max_level,
            "root": self._record_to_dict(self.root),
            "rollup": self.rollup(),
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize the full result to a JSON string (meta coerced via str)."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_csv(self) -> str:
        """Flat one-row-per-region CSV. Fixed timing columns followed by one
        column per counter key (union across regions, first-seen order)."""
        rows: List[tuple] = []

        def _walk(r: Record, depth: int) -> None:
            rows.append((r, depth))
            for c in r.children:
                _walk(c, depth + 1)

        _walk(self.root, 0)

        counter_keys: List[str] = []
        seen = set()
        for r, _ in rows:
            for k in r.counters:
                if k not in seen:
                    seen.add(k)
                    counter_keys.append(k)

        fields = [
            "depth", "name", "level", "wall_time", "cpu_time",
            "self_time", "errored",
        ] + counter_keys

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(fields)
        for r, depth in rows:
            row = [
                depth, r.name, r.level_name,
                f"{r.wall_time:.9f}", f"{r.cpu_time:.9f}",
                f"{r.self_time:.9f}", int(r.errored),
            ] + [r.counters.get(k, "") for k in counter_keys]
            writer.writerow(row)
        return buf.getvalue()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ProfileResult(root={self.root!r}, n_records={len(self.records())})"

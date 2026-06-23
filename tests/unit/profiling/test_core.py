"""Core profiler engine tests (no n2v reach wiring yet).

T1 — region nesting / tree shape.
T4 — disabled = no-op (and level-skipping).

See plans/2026-06-22-profiler-test-spec.md.
"""

import csv
import io
import json
import threading

import pytest

from n2v.profiling import (
    LAYER,
    OPERATION,
    PHASE,
    RUN,
    count,
    is_enabled,
    profile,
    region,
    set_meta,
)
from n2v.profiling.profiler import _NULL


# --------------------------------------------------------------------------- #
# T1 — region nesting / tree shape
# --------------------------------------------------------------------------- #
def test_t1_region_nesting_tree_shape():
    with profile() as p:
        with region("reach", PHASE):
            with region("layer0", LAYER):
                with region("affine", OPERATION):
                    pass
            with region("layer1", LAYER):
                pass

    root = p.root
    assert root.level == RUN
    assert root.name == "run"

    # one phase child
    assert len(root.children) == 1
    reach = root.children[0]
    assert reach.name == "reach"
    assert reach.level == PHASE
    assert reach.parent is root

    # two layer children, order preserved
    assert [c.name for c in reach.children] == ["layer0", "layer1"]
    layer0, layer1 = reach.children
    assert layer0.level == LAYER and layer1.level == LAYER
    assert layer0.parent is reach and layer1.parent is reach

    # operation nested under layer0 only
    assert [c.name for c in layer0.children] == ["affine"]
    affine = layer0.children[0]
    assert affine.level == OPERATION
    assert affine.parent is layer0
    assert layer1.children == []


def test_t1_records_walk_preorder():
    with profile() as p:
        with region("reach", PHASE):
            with region("layer0", LAYER):
                pass
    names = [r.name for r in p.records()]
    assert names == ["run", "reach", "layer0"]
    assert p.find("layer0") == [p.root.children[0].children[0]]


# --------------------------------------------------------------------------- #
# T4 — disabled = no-op
# --------------------------------------------------------------------------- #
def test_t4_disabled_region_is_singleton_noop():
    # Outside any profile() scope, profiling is disabled.
    assert not is_enabled()
    r = region("x", OPERATION)
    assert r is _NULL  # no Record object created
    with region("y", LAYER):  # works, records nothing, raises nothing
        pass
    # counters / metadata are silent no-ops when disabled
    count("anything", 5)
    set_meta(foo="bar")


def test_t4_enabled_only_within_scope():
    assert not is_enabled()
    with profile():
        assert is_enabled()
    assert not is_enabled()


def test_t4_level_deeper_than_max_is_skipped():
    # profile only down to LAYER: OPERATION regions are skipped (no records).
    with profile(level="layer") as p:
        with region("reach", PHASE):
            with region("layer0", LAYER):
                with region("affine", OPERATION):  # deeper than max_level -> skipped
                    count("ignored", 1)
    names = [r.name for r in p.records()]
    assert names == ["run", "reach", "layer0"]
    # the skipped operation produced no node
    assert p.find("affine") == []


def test_t4_nested_profiles_restore_previous_active():
    with profile() as outer:
        assert is_enabled()
        with profile() as inner:
            assert is_enabled()
            with region("inner_phase", PHASE):
                pass
        # after inner closes, the outer profile is active again
        assert is_enabled()
        with region("outer_phase", PHASE):
            pass
    assert not is_enabled()
    assert [c.name for c in inner.root.children] == ["inner_phase"]
    assert [c.name for c in outer.root.children] == ["outer_phase"]


# --------------------------------------------------------------------------- #
# T7 — operation drill-down: counters attribute to the open region; a layer's
# subtree sums its own + its operations'. Skipping a region attributes its
# counts to the parent (coarser, never lost).
# --------------------------------------------------------------------------- #
def test_t7_operation_drilldown_sums():
    with profile(level="operation") as p:
        with region("L", LAYER):
            with region("op1", OPERATION):
                count("c", 1)
            with region("op2", OPERATION):
                count("c", 2)
            count("c", 4)  # lands on the layer itself

    L = p.find("L")[0]
    assert {r.name for r in L.children} == {"op1", "op2"}
    assert p.find("op1")[0].counters["c"] == 1
    assert p.find("op2")[0].counters["c"] == 2
    assert L.counters["c"] == 4
    # subtree rolls up own + descendants
    assert p.subtree_counters(L)["c"] == 7


def test_t7_skipped_operation_attributes_to_parent():
    # At layer level, OPERATION regions are skipped but their counts are not
    # lost -- they attribute to the enclosing layer.
    with profile(level="layer") as p:
        with region("L", LAYER):
            with region("op", OPERATION):  # skipped
                count("c", 3)
            count("c", 4)
    assert p.find("op") == []
    assert p.find("L")[0].counters["c"] == 7


# --------------------------------------------------------------------------- #
# T8 — rollup percentages: per-layer-type wall share is in [0,100] and sums to
# <= 100 (sequential siblings); summary renders a '%'.
# --------------------------------------------------------------------------- #
def test_t8_rollup_wall_percentages():
    with profile(level="layer") as p:
        with region("reach", PHASE):
            with region("Linear", LAYER, layer_type="Linear"):
                count("n_sets_out", 1)
            with region("ReLU", LAYER, layer_type="ReLU"):
                count("n_sets_out", 2)

    ro = p.rollup()
    pcts = [agg["wall_pct"] for agg in ro["by_layer_type"].values()]
    assert all(0.0 <= x <= 100.0 for x in pcts)
    assert sum(pcts) <= 100.0 + 1e-6
    assert ro["peak_population"] == 2

    text = p.summary()
    assert "%" in text
    assert "=== run rollup ===" in text


# --------------------------------------------------------------------------- #
# T9 — JSON / CSV export round-trips the tree + counters.
# --------------------------------------------------------------------------- #
def test_t9_to_json_roundtrip():
    with profile(level="operation") as p:
        with region("reach", PHASE):
            with region("L", LAYER, layer_type="Linear"):
                count("n_lp_solves", 5)
                with region("affine", OPERATION):
                    pass

    parsed = json.loads(p.to_json())
    assert parsed["max_level"] == OPERATION

    # walk the serialized tree; structure + counters match records()
    flat = []

    def _walk(node):
        flat.append((node["name"], node["level"], node["counters"]))
        for ch in node["children"]:
            _walk(ch)

    _walk(parsed["root"])
    expected = [(r.name, r.level, dict(r.counters)) for r in p.records()]
    assert flat == expected
    # meta survives
    L_node = parsed["root"]["children"][0]["children"][0]
    assert L_node["meta"]["layer_type"] == "Linear"
    assert L_node["counters"]["n_lp_solves"] == 5


def test_t9_to_csv_roundtrip():
    with profile(level="operation") as p:
        with region("reach", PHASE):
            with region("L", LAYER, layer_type="Linear"):
                count("n_lp_solves", 5)

    rows = list(csv.DictReader(io.StringIO(p.to_csv())))
    # one row per region (root + reach + L)
    assert len(rows) == len(p.records()) == 3
    assert [r["name"] for r in rows] == ["run", "reach", "L"]
    assert [r["level"] for r in rows] == ["run", "phase", "layer"]
    # the counter column is populated on L, empty elsewhere
    l_row = next(r for r in rows if r["name"] == "L")
    assert l_row["n_lp_solves"] == "5"
    assert l_row["depth"] == "2"


# --------------------------------------------------------------------------- #
# T15 — edge cases: empty run, single region, no counters.
# --------------------------------------------------------------------------- #
def test_t15_empty_run():
    with profile() as p:
        pass
    assert [r.name for r in p.records()] == ["run"]
    ro = p.rollup()
    assert ro["n_layers"] == 0
    assert ro["peak_population"] == 0
    assert ro["by_layer_type"] == {}
    # exports do not crash and are non-empty
    assert json.loads(p.to_json())["root"]["name"] == "run"
    csv_rows = list(csv.DictReader(io.StringIO(p.to_csv())))
    assert len(csv_rows) == 1 and csv_rows[0]["name"] == "run"
    assert "=== run rollup ===" in p.summary()


def test_t15_single_region_no_counters():
    with profile(level="layer") as p:
        with region("only", LAYER, layer_type="Linear"):
            pass
    assert [r.name for r in p.records()] == ["run", "only"]
    # no counters anywhere -> totals empty, csv has just the fixed columns
    assert p.rollup()["totals"] == {}
    header = p.to_csv().splitlines()[0]
    assert header.startswith("depth,name,level,wall_time,cpu_time,self_time,errored")


# --------------------------------------------------------------------------- #
# T13 — exception inside a region: region closes, partial record kept and
# flagged errored, and the exception PROPAGATES (observation-only).
# --------------------------------------------------------------------------- #
def test_t13_exception_closes_region_and_propagates():
    p = profile(level="operation")
    with pytest.raises(ValueError, match="boom"):
        with p:
            with region("L", LAYER):
                count("c", 1)
                raise ValueError("boom")

    # active profiler restored despite the exception
    assert not is_enabled()
    # the region was closed, its partial counter kept, and it is flagged
    L = p.find("L")[0]
    assert L.counters["c"] == 1
    assert L.errored is True
    assert L.wall_time >= 0.0


# --------------------------------------------------------------------------- #
# T11 — thread-safety: regions opened in worker threads aggregate without loss
# or races (worker threads start a fresh stack rooted at the run).
# --------------------------------------------------------------------------- #
def test_t11_thread_safe_aggregation():
    n = 16
    with profile() as p:
        def work(i):
            with region(f"t{i}", LAYER):
                count("c", 1)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    worker_regions = [r for r in p.records() if r.name.startswith("t")]
    assert len(worker_regions) == n  # none lost to races
    assert sum(r.counters.get("c", 0) for r in worker_regions) == n
    # all attached under the run root (fresh per-thread stacks)
    assert all(r.parent is p.root for r in worker_regions)


def test_t11_counter_increments_are_atomic():
    """Concurrent count() to the same region (worker threads all attribute to
    root) must not lose updates -- the add_counter lock makes the
    read-modify-write atomic. Without it, d[k]=d.get(k,0)+1 races under the GIL.
    """
    n_threads, per = 8, 20_000
    with profile() as p:
        def work():
            for _ in range(per):
                count("x", 1)

        threads = [threading.Thread(target=work) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert p.root.counters["x"] == n_threads * per

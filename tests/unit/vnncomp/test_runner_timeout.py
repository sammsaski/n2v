"""The 2026 runner and its deadline.

The runner arms an alarm for the instance timeout. Work the alarm cannot
interrupt (it has been seen swallowed inside a logging lock) returns late, and
whatever it returns is written to the results file. A verdict decided after
the deadline is not the instance's result: the file must say ``timeout``.

From the section "main, around the verification" on, the file tests the rest
of ``main`` on both branches (one network, and the relational list of two):
the alarm armed for the instance timeout and disarmed afterwards, an alarm
that fires inside a broad ``except Exception`` guard, an error written as
``unknown``, compressed arguments and their temporary copies, and the
relational instance end to end (the order of its networks, where its files
are found, the verdict it writes).
"""

import gzip
import os
import shutil
import signal
import sys
import tempfile
import time

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'examples', 'Submission', 'VNN_COMP2026'))
import vnncomp_runner as R  # noqa: E402
from tests.known_bugs import known_bug  # noqa: E402


@pytest.fixture
def alarm_restored():
    """``main`` installs its SIGALRM handler and never removes it."""
    if not hasattr(signal, "SIGALRM"):
        pytest.skip("no SIGALRM on this platform")
    saved = signal.getsignal(signal.SIGALRM)
    yield
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, saved)


def _run_main(monkeypatch, tmp_path, timeout, verify_instance):
    results = tmp_path / "out.txt"
    monkeypatch.setattr(R, "verify_instance", verify_instance)
    monkeypatch.setattr(sys, "argv", ["vnncomp_runner.py", "category",
                                      "model.onnx", "spec.vnnlib",
                                      str(results), str(timeout)])
    assert R.main() == 0
    return results.read_text().splitlines()[0]


@pytest.mark.parametrize("verdict", [R.RESULT_UNSAT, R.RESULT_SAT,
                                     R.RESULT_UNKNOWN])
def test_a_verdict_decided_in_time_is_written(monkeypatch, tmp_path, capsys,
                                              alarm_restored, verdict):
    def in_time(onnx_path, vnnlib_path, category):
        return {"result": verdict, "counterexample": None}

    assert _run_main(monkeypatch, tmp_path, 30, in_time) == verdict


@pytest.mark.parametrize("verdict", [R.RESULT_UNSAT, R.RESULT_SAT])
@known_bug(20, "the runner writes whatever verify_instance returns, with no "
               "check of the elapsed time against the timeout")
def test_a_verdict_decided_after_the_deadline_is_written_as_timeout(
        monkeypatch, tmp_path, capsys, alarm_restored, verdict):
    """The timeout is 0.05 s and the work takes 0.2 s. The alarm is armed for
    no less than one second, so it does not fire: this is the swallowed-alarm
    case without the lock."""
    def late(onnx_path, vnnlib_path, category):
        time.sleep(0.2)
        return {"result": verdict, "counterexample": "((X_0 0.0))"}

    assert _run_main(monkeypatch, tmp_path, 0.05, late) == R.RESULT_TIMEOUT


# --------------------------------------------------------------------------- #
# The promise, by cases                                                        #
# --------------------------------------------------------------------------- #
# A verdict that comes back later than the timeout by more than any reasonable
# margin is written as ``timeout``, whether or not the alarm fired; a verdict
# that comes back in time is written as decided. The competition harness kills
# the process 60 s after the timeout, so timeout + 61 s is late under any
# margin. A verdict decided a moment after the deadline (timeout + 1 s) is not
# asserted either way here: PLAN step 20 proposes (2026-11-13) that there be
# no margin, but the owner has left the margin open, so no number is fixed.
_RELATIONAL_ONNX = "[('f', 'f.onnx'), ('g', 'g.onnx')]"
_LATE = known_bug(
    20, "the runner writes whatever the verification returns, with no check "
        "of the elapsed time against the timeout: a verdict decided long "
        "after the deadline, with the alarm lost or swallowed, is written "
        "as decided")
_VERDICTS = [R.RESULT_UNSAT, R.RESULT_SAT, R.RESULT_UNKNOWN]
_BRANCHES = {"single": "verify_instance",
             "relational": "verify_relational_instance"}


class _SimulatedClock:
    """Every clock the runner could read, moved only by the patched work."""

    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now

    def ns(self):
        return int(self.now * 1e9)


@pytest.fixture
def lost_alarm(monkeypatch):
    """A simulated clock in place of ``time.time``, ``time.monotonic`` and
    ``time.perf_counter`` (and their ``_ns`` forms, and any of them the
    runner imported by name), so the test does not depend on which one a
    fixed runner reads. ``_arm_timeout`` and ``_disarm_timeout`` record and
    arm nothing: the alarm is lost, as when it was delivered inside native
    code and swallowed, or the platform has none."""
    clock = _SimulatedClock()
    for name in ("time", "monotonic", "perf_counter"):
        original = getattr(time, name)
        for attr, value in list(vars(R).items()):
            if value is original:
                monkeypatch.setattr(R, attr, clock)
        monkeypatch.setattr(time, name, clock)
    for name in ("time_ns", "monotonic_ns", "perf_counter_ns"):
        monkeypatch.setattr(time, name, clock.ns)
    armed = []
    monkeypatch.setattr(R, "_arm_timeout", armed.append)
    monkeypatch.setattr(R, "_disarm_timeout", lambda: None)
    return clock


def _run(monkeypatch, tmp_path, branch, timeout, work):
    """``main`` on one instance of ``branch`` with the verification replaced
    by ``work``; the first line of the results file."""
    results = tmp_path / "out.txt"
    monkeypatch.setattr(R, _BRANCHES[branch], work)
    onnx = _RELATIONAL_ONNX if branch == "relational" else "model.onnx"
    monkeypatch.setattr(sys, "argv", ["vnncomp_runner.py", "category", onnx,
                                      "spec.vnnlib", str(results),
                                      str(timeout)])
    if R.main() != 0:
        pytest.fail("main did not return 0")
    return results.read_text().splitlines()[0]


def _decided_after(clock, seconds, verdict):
    def work(*args, **kwargs):
        clock.now += seconds
        return {"result": verdict,
                "counterexample": "((X_0 0.0))" if verdict == R.RESULT_SAT
                else None}
    return work


@pytest.mark.parametrize("branch", _BRANCHES)
@pytest.mark.parametrize("verdict", _VERDICTS)
def test_a_verdict_decided_before_the_deadline_is_written_on_a_lost_alarm(
        monkeypatch, tmp_path, capsys, lost_alarm, verdict, branch):
    """Decided at 9 s of a 10 s timeout, on the simulated clock: written as
    decided. Neighbour of the late cells below (same clock, same runner
    call). Catches a deadline check that fires early or writes ``timeout``
    for an answer in time."""
    work = _decided_after(lost_alarm, 9.0, verdict)
    assert _run(monkeypatch, tmp_path, branch, 10, work) == verdict


@pytest.mark.parametrize("branch", _BRANCHES)
@pytest.mark.parametrize("seconds", [
    pytest.param(109.0, id="109s-of-10s"),
    pytest.param(10.0 + 61.0, id="timeout-plus-61s"),
])
@pytest.mark.parametrize("verdict", _VERDICTS)
@_LATE
def test_a_verdict_decided_long_after_the_deadline_is_written_as_timeout(
        monkeypatch, tmp_path, capsys, lost_alarm, verdict, seconds, branch):
    """The September observation (PLAN step 20): ACAS Xu ``1_1`` with
    ``prop_1`` and a 10 s timeout wrote ``unsat`` after 109 s, the alarm
    swallowed. On the simulated clock the alarm never fires and the work
    returns 109 s, or timeout + 61 s, after the start: the file must say
    ``timeout``. A late SAT or UNKNOWN is no more the instance's result than
    a late UNSAT."""
    work = _decided_after(lost_alarm, seconds, verdict)
    assert _run(monkeypatch, tmp_path, branch, 10, work) == R.RESULT_TIMEOUT


@pytest.mark.parametrize("branch", _BRANCHES)
@_LATE
def test_a_verdict_returned_after_a_swallowed_alarm_is_written_as_timeout(
        monkeypatch, tmp_path, capsys, alarm_restored, branch):
    """Real time and the real alarm: timeout 1.0 s, and work that runs 1.3 s
    in small steps inside ``try: ... except BaseException: pass``, as a broad
    handler in a library would, then returns UNSAT. The alarm fires at 1.0 s
    and is swallowed; the verdict comes back 0.3 s after the deadline."""
    def swallows_the_alarm(*args, **kwargs):
        start = time.monotonic()
        while time.monotonic() - start < 1.3:
            try:
                time.sleep(0.01)
            except BaseException:   # noqa: BLE001 - the swallowing handler
                pass
        return {"result": R.RESULT_UNSAT, "counterexample": None}

    assert _run(monkeypatch, tmp_path, branch, 1.0,
                swallows_the_alarm) == R.RESULT_TIMEOUT


@pytest.mark.parametrize("branch", _BRANCHES)
def test_the_alarm_interrupts_work_that_runs_past_the_deadline(
        monkeypatch, tmp_path, capsys, alarm_restored, branch):
    """Real time and the real alarm: timeout 1.0 s and interruptible work of
    3 s that would return UNSAT. The alarm interrupts it: the file says
    ``timeout``, written well before the work would have ended. Neighbour of
    the swallowed-alarm cell. Catches an alarm that is not armed, armed for
    the wrong time, or whose exception is written as anything else; an alarm
    that fires before the deadline (the lower bound on the elapsed time);
    ``test_the_alarm_is_armed_for_the_instance_timeout`` reads the armed
    time itself."""
    def interruptible(*args, **kwargs):
        time.sleep(3.0)
        return {"result": R.RESULT_UNSAT, "counterexample": None}

    start = time.monotonic()
    written = _run(monkeypatch, tmp_path, branch, 1.0, interruptible)
    elapsed = time.monotonic() - start
    assert written == R.RESULT_TIMEOUT
    assert 0.95 <= elapsed < 2.0, f"written after {elapsed:.2f} s"


# --------------------------------------------------------------------------- #
# main, around the verification                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("branch", _BRANCHES)
@pytest.mark.parametrize("timeout,armed", [(0.3, 1.0), (7.5, 7.5)],
                         ids=["0.3s-floored-to-1s", "7.5s"])
def test_the_alarm_is_armed_for_the_instance_timeout(
        monkeypatch, tmp_path, capsys, alarm_restored, branch, timeout,
        armed):
    """While the verification runs, the real-time alarm is armed for the
    instance timeout (at least one second) with the runner's handler.
    Read from inside the work, so no case waits for the alarm. Catches an
    alarm armed for one second whatever the timeout (an instance that needs
    two seconds of a 20 s budget is cut off), a floor dropped, and a branch
    that arms nothing."""
    seen = []

    def work(*args, **kwargs):
        seen.append((signal.getsignal(signal.SIGALRM),
                     signal.getitimer(signal.ITIMER_REAL)[0]))
        return {"result": R.RESULT_UNSAT, "counterexample": None}

    assert _run(monkeypatch, tmp_path, branch, timeout, work) == \
        R.RESULT_UNSAT
    handler, remaining = seen[0]
    assert handler is R._on_alarm
    assert armed - 0.25 < remaining <= armed


@pytest.mark.parametrize("branch", _BRANCHES)
def test_no_alarm_is_left_armed_after_main(monkeypatch, tmp_path, capsys,
                                           alarm_restored, branch):
    """``main`` returns well before a 30 s timeout: the real-time alarm is
    disarmed, so it cannot interrupt what the process does next. Catches a
    disarm of another timer."""
    def work(*args, **kwargs):
        return {"result": R.RESULT_UNSAT, "counterexample": None}

    assert _run(monkeypatch, tmp_path, branch, 30, work) == R.RESULT_UNSAT
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0.0


@pytest.mark.parametrize("branch", _BRANCHES)
def test_an_alarm_inside_a_broad_exception_guard_still_ends_the_run(
        monkeypatch, tmp_path, capsys, alarm_restored, branch):
    """The alarm's handler runs inside verification code wrapped in
    ``try: ... except Exception: pass`` (the falsify and reach stages have
    such guards), after which the work would go on and return UNSAT: the
    file says ``timeout``. Catches the runner's timeout exception derived
    from ``Exception``, which such a guard swallows."""
    def guarded(*args, **kwargs):
        try:
            R._on_alarm(signal.SIGALRM, None)
        except Exception:  # noqa: BLE001 - the guard of a verification stage
            pass
        return {"result": R.RESULT_UNSAT, "counterexample": None}

    assert _run(monkeypatch, tmp_path, branch, 30, guarded) == \
        R.RESULT_TIMEOUT


@pytest.mark.parametrize("branch", _BRANCHES)
def test_an_error_in_the_verification_is_written_unknown(
        monkeypatch, tmp_path, capsys, alarm_restored, branch):
    """The verification raises: ``main`` still returns 0 and writes
    ``unknown``, never a verdict."""
    def broken(*args, **kwargs):
        raise RuntimeError("the verification failed")

    assert _run(monkeypatch, tmp_path, branch, 30, broken) == R.RESULT_UNKNOWN


def test_main_refuses_too_few_arguments(monkeypatch, tmp_path, capsys):
    """Fewer than the five arguments: usage on stderr, exit code 2, no
    results file and no alarm."""
    results = tmp_path / "out.txt"
    monkeypatch.setattr(sys, "argv", ["vnncomp_runner.py", "category",
                                      "model.onnx", "spec.vnnlib",
                                      str(results)])
    assert R.main() == 2
    assert "usage" in capsys.readouterr().err
    assert not results.exists()
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0.0


# Small instances run through the real verification. y = x0 + x1 is written
# as a Gemm on the flattened input; every verdict below is known by hand.

def _affine_onnx(path, w, b):
    """``Y [1, m] = X @ w.T + b`` with X [1, n]."""
    w = np.asarray(w, np.float32)
    graph = helper.make_graph(
        [helper.make_node("Reshape", ["X", "sh"], ["f"]),
         helper.make_node("Gemm", ["f", "W", "B"], ["Y"], transB=1)], "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                       [1, w.shape[1]])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                       [1, w.shape[0]])],
        [numpy_helper.from_array(np.array([-1, w.shape[1]], np.int64), "sh"),
         numpy_helper.from_array(w, "W"),
         numpy_helper.from_array(np.asarray(b, np.float32), "B")])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def _gzip(path):
    with open(path, "rb") as fi, gzip.open(str(path) + ".gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(path)
    return str(path) + ".gz"


@pytest.fixture
def real_runs(monkeypatch):
    """The verification runs for real on one worker, falsification by 200
    random samples and approx reach."""
    monkeypatch.setenv("N2V_WORKERS", "1")
    monkeypatch.setattr(R, "get_config", lambda *a, **k: {
        "reach_methods": [("approx", {})], "n_rand": 200,
        "falsify_method": "random", "falsify_kwargs": {}})


def _main(monkeypatch, tmp_path, onnx_arg, spec, timeout=60):
    results = tmp_path / "results.txt"
    monkeypatch.setattr(sys, "argv", ["vnncomp_runner.py", "demo", onnx_arg,
                                      spec, str(results), str(timeout)])
    if R.main() != 0:
        pytest.fail("main did not return 0")
    return results.read_text().splitlines()[0]


def test_compressed_arguments_are_read_and_their_copies_removed(
        monkeypatch, tmp_path, capsys, alarm_restored, real_runs):
    """``main`` with ``model.onnx.gz`` and ``spec.vnnlib.gz``: y = x0 + x1 on
    ``[0, 1]^2`` and unsafe ``Y_0 >= 2.5``, proved by approx reach, so the
    file says ``unsat``; the decompressed copies, made in the temporary
    directory, are gone afterwards. Catches a ".gz" copied without
    decompressing (the model cannot be loaded) and copies reported as not
    created (never removed)."""
    model = _gzip(_affine_onnx(tmp_path / "model.onnx", [[1.0, 1.0]], [0.0]))
    spec = tmp_path / "spec.vnnlib"
    spec.write_text("(declare-const X_0 Real)\n(declare-const X_1 Real)\n"
                    "(declare-const Y_0 Real)\n(assert (>= X_0 0.0))\n"
                    "(assert (<= X_0 1.0))\n(assert (>= X_1 0.0))\n"
                    "(assert (<= X_1 1.0))\n(assert (>= Y_0 2.5))\n")
    spec = _gzip(spec)
    scratch = tmp_path / "tmpdir"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    assert _main(monkeypatch, tmp_path, model, spec) == R.RESULT_UNSAT
    assert sorted(os.listdir(scratch)) == []


# The relational instance: networks f and g of one input each, coupled inputs
# X_g = X_f in [0, 1], unsafe Y_g - Y_f >= 0.5. With g = f - 1 the difference
# is -1 everywhere (safe); with g = f + 1 it is 1 everywhere (every input is a
# counterexample, which the runner concedes to unknown).
_REL_SPEC = """(vnnlib-version <2.0>)
(declare-network f
    (declare-input X_f float32 [1, 1])
    (declare-output Y_f float32 [1, 1])
)
(declare-network g
    (declare-input X_g float32 [1, 1])
    (declare-output Y_g float32 [1, 1])
)
(assert (>= X_f[0,0] 0.0))
(assert (<= X_f[0,0] 1.0))
(assert (== X_g[0,0] X_f[0,0]))
(assert (>= (- Y_g[0,0] Y_f[0,0]) 0.5))
"""


def _relational(tmp_path, g_bias, layout="listed", spec_text=_REL_SPEC):
    """A benchmark version directory with ``onnx/`` and ``vnnlib/``: f(x) = x
    and g(x) = x + ``g_bias``. ``layout`` places the files: ``listed`` at the
    listed paths ``onnx/f.onnx`` and ``onnx/g.onnx``; ``flat`` directly under
    ``onnx/`` while the instance lists ``onnx/original/``; ``deeper`` under
    ``onnx/a/b/`` with the same listing; ``gz`` compressed at the listed
    paths; ``gz-with-spec`` as ``gz`` with the spec compressed too;
    ``flat-gz`` compressed directly under ``onnx/`` while the instance lists
    ``onnx/original/``; ``missing`` nowhere. Returns the ONNX argument and
    the spec."""
    base = tmp_path / "bench" / "2026"
    sub = {"listed": "", "flat": "", "gz": "", "gz-with-spec": "",
           "flat-gz": "", "missing": "", "deeper": "a/b"}[layout]
    where = base / "onnx" / sub
    where.mkdir(parents=True)
    (base / "vnnlib").mkdir()
    for name, bias in (("f", 0.0), ("g", g_bias)):
        p = _affine_onnx(where / f"{name}.onnx", [[1.0]], [bias])
        if layout in ("gz", "gz-with-spec", "flat-gz"):
            _gzip(p)
        elif layout == "missing":
            os.remove(p)
    folder = "onnx" if layout in ("listed", "gz", "gz-with-spec", "missing") \
        else "onnx/original"
    onnx_arg = f"[('f', '{folder}/f.onnx'), ('g', '{folder}/g.onnx')]"
    spec = base / "vnnlib" / "spec.vnnlib"
    spec.write_text(spec_text)
    if layout == "gz-with-spec":
        return onnx_arg, _gzip(spec)
    return onnx_arg, str(spec)


@pytest.mark.parametrize("quoted", [False, True], ids=["plain", "quoted"])
def test_a_relational_instance_proved_safe_is_written_unsat(
        monkeypatch, tmp_path, capsys, alarm_restored, real_runs, quoted):
    """g = f - 1, so Y_g - Y_f = -1 < 0.5 for every input: the joint reach
    proves it and the file says ``unsat``, also when the list literal comes
    wrapped in the CSV cell's double quotes. The caller's configuration
    ('highs', parallel LP on with 3 workers) is given back. Catches the
    relational verdict compared with the upper-case convention of
    ``verify_specification`` (a proof written as unknown), a quote left on
    the argument (it no longer parses), and a configuration not restored."""
    import n2v
    from n2v.config import config
    onnx_arg, spec = _relational(tmp_path, g_bias=-1.0)
    if quoted:
        onnx_arg = '"' + onnx_arg + '"'
    try:
        n2v.set_parallel(True, n_workers=3)
        n2v.set_lp_solver("highs")
        before = (config.parallel_lp, config.n_workers,
                  str(config.lp_solver.value))
        assert _main(monkeypatch, tmp_path, onnx_arg, spec) == R.RESULT_UNSAT
        assert (config.parallel_lp, config.n_workers,
                str(config.lp_solver.value)) == before
    finally:
        config.reset()


def test_a_relational_instance_with_a_counterexample_is_not_written_unsat(
        monkeypatch, tmp_path, capsys, alarm_restored, real_runs):
    """g = f + 1, so Y_g - Y_f = 1 >= 0.5 at every input: a counterexample
    everywhere. The runner concedes a relational counterexample (its witness
    format is unconfirmed), so the file says ``unknown``: never ``unsat``,
    and no ``sat`` without a checked witness. Catches f and g swapped
    against the spec's declaration order (the swapped composition is safe
    and is written ``unsat``)."""
    onnx_arg, spec = _relational(tmp_path, g_bias=1.0)
    assert _main(monkeypatch, tmp_path, onnx_arg, spec) == R.RESULT_UNKNOWN


@pytest.mark.parametrize("layout,expected", [
    ("flat", R.RESULT_UNSAT),
    ("deeper", R.RESULT_UNSAT),
    ("gz", R.RESULT_UNSAT),
    ("gz-with-spec", R.RESULT_UNSAT),
    ("flat-gz", R.RESULT_UNSAT),
    ("missing", R.RESULT_UNKNOWN),
])
def test_relational_networks_are_found_where_the_benchmark_ships_them(
        monkeypatch, tmp_path, capsys, alarm_restored, real_runs, layout,
        expected):
    """The safe relational instance (g = f - 1) with its files directly
    under ``onnx/`` or two folders deeper while the instance lists
    ``onnx/original/``, or compressed at the listed paths (with or without
    the spec compressed as well) or directly under ``onnx/``: found,
    proved, ``unsat``, and no decompressed copy is left in the temporary
    directory. Files that exist nowhere give ``unknown``. Catches a
    basename search that is not recursive (a file directly under ``onnx/``
    is missed) or that does not look for the compressed name, and
    compressed copies of the networks or of the spec that are not
    removed."""
    onnx_arg, spec = _relational(tmp_path, g_bias=-1.0, layout=layout)
    scratch = tmp_path / "tmpdir"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    assert _main(monkeypatch, tmp_path, onnx_arg, spec) == expected
    assert sorted(os.listdir(scratch)) == []


def test_a_list_of_networks_with_a_single_network_spec_is_unknown(
        monkeypatch, tmp_path, capsys, alarm_restored, real_runs):
    """Two networks with a spec that is not relational (one network's box
    and output): there is nothing the relational verifier can decide, the
    file says ``unknown``."""
    plain = ("(declare-const X_0 Real)\n(declare-const Y_0 Real)\n"
             "(assert (>= X_0 0.0))\n(assert (<= X_0 1.0))\n"
             "(assert (>= Y_0 5.0))\n")
    onnx_arg, spec = _relational(tmp_path, g_bias=-1.0, spec_text=plain)
    assert _main(monkeypatch, tmp_path, onnx_arg, spec) == R.RESULT_UNKNOWN

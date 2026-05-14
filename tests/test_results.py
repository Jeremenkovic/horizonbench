"""Tests for results/trace.py and results/store.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from horizonbench.results.trace import TraceEvent, TraceWriter, read_trace
from horizonbench.results.store import ResultStore
from horizonbench.tasks.base import Checkpoint, CheckpointResult, RunResult


# ── TraceWriter ───────────────────────────────────────────────────────────────


def test_trace_writer_creates_file(tmp_path):
    p = tmp_path / "trace.jsonl"
    with TraceWriter(p) as tw:
        tw.run_start("family/0.1/inst/k5", "gpt-4o", 5)
        tw.message("user", "hello")
        tw.tool_call("read_file", {"path": "a.py"}, "call-1")
        tw.tool_result("call-1", "contents")
        tw.run_end(True, 1.0, 0.01, 100, 3.5)
    assert p.exists()


def test_trace_writer_readable(tmp_path):
    p = tmp_path / "trace.jsonl"
    with TraceWriter(p) as tw:
        tw.run_start("t/0.1/i/k3", "m", 3)
        tw.message("assistant", "done")
        tw.run_end(False, 0.33, 0.0, 50, 1.0)

    events = read_trace(p)
    assert len(events) == 3
    assert events[0].event_type == "run_start"
    assert events[-1].event_type == "run_end"
    assert events[-1].data["partial_credit"] == pytest.approx(0.33)


def test_trace_tool_result_truncated(tmp_path):
    p = tmp_path / "trace.jsonl"
    with TraceWriter(p) as tw:
        tw.tool_result("c1", "x" * 5000)  # long result
    events = read_trace(p)
    assert len(events[0].data["result"]) <= 2000


# ── ResultStore ───────────────────────────────────────────────────────────────


def _make_run(
    task_id="multi-refactor/0.1/inst-001/k5",
    success=True,
    partial_credit=1.0,
    n_checkpoints=5,
) -> RunResult:
    cps = [
        CheckpointResult(
            checkpoint=Checkpoint(index=i, description=f"cp{i}"),
            passed=success,
        )
        for i in range(n_checkpoints)
    ]
    return RunResult(
        task_id=task_id,
        success=success,
        partial_credit=partial_credit,
        checkpoint_results=cps,
        cost_usd=0.01,
        total_tokens=200,
        wall_seconds=5.0,
    )


def test_store_save_and_retrieve():
    with ResultStore(":memory:") as store:
        result = _make_run()
        run_id = store.save_run(result, model="gpt-4o", family="multi-refactor", instance_id="inst-001", run_idx=0)
        assert run_id > 0
        runs = store.get_runs(model="gpt-4o", family="multi-refactor")
        assert len(runs) == 1
        assert runs[0]["success"] is True
        assert runs[0]["partial_credit"] == pytest.approx(1.0)


def test_store_filter_by_k():
    with ResultStore(":memory:") as store:
        store.save_run(_make_run("f/0.1/i/k5"), "m", "f", "i", 0)
        store.save_run(_make_run("f/0.1/i/k10", partial_credit=0.5, success=False), "m", "f", "i", 0)
        k5 = store.get_runs(k=5)
        k10 = store.get_runs(k=10)
        assert len(k5) == 1
        assert len(k10) == 1


def test_store_success_rates():
    with ResultStore(":memory:") as store:
        # 2 successes, 1 failure at k=5
        store.save_run(_make_run("f/0.1/a/k5", success=True), "m", "f", "a", 0)
        store.save_run(_make_run("f/0.1/b/k5", success=True), "m", "f", "b", 1)
        store.save_run(_make_run("f/0.1/c/k5", success=False, partial_credit=0.4), "m", "f", "c", 2)
        rates = store.success_rates("m", "f")
        assert rates[5] == pytest.approx(2 / 3)


def test_store_partial_credits():
    with ResultStore(":memory:") as store:
        store.save_run(_make_run("f/0.1/a/k5", partial_credit=1.0), "m", "f", "a", 0)
        store.save_run(_make_run("f/0.1/b/k5", partial_credit=0.6, success=False), "m", "f", "b", 1)
        pcs = store.partial_credits("m", "f")
        assert sorted(pcs[5]) == pytest.approx(sorted([1.0, 0.6]))


def test_store_multiple_models():
    with ResultStore(":memory:") as store:
        store.save_run(_make_run(), "model-a", "f", "i", 0)
        store.save_run(_make_run(), "model-b", "f", "i", 0)
        assert len(store.get_runs(model="model-a")) == 1
        assert len(store.get_runs(model="model-b")) == 1
        assert len(store.get_runs()) == 2

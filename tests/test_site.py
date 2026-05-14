"""Tests for the static site generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from horizonbench.results.store import ResultStore
from horizonbench.results.trace import TraceWriter
from horizonbench.site import build_site, _slug
from horizonbench.tasks.base import Checkpoint, CheckpointResult, RunResult


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_run(task_id="f/0.1/i/k5", success=True, partial_credit=1.0) -> RunResult:
    cps = [
        CheckpointResult(
            checkpoint=Checkpoint(index=i, description=f"cp{i}"),
            passed=success,
        )
        for i in range(5)
    ]
    return RunResult(
        task_id=task_id,
        success=success,
        partial_credit=partial_credit,
        checkpoint_results=cps,
        cost_usd=0.02,
        total_tokens=300,
        wall_seconds=10.0,
    )


def _populate_db(db_path: Path, n_models=2, n_families=2) -> None:
    models = [f"model-{i}" for i in range(n_models)]
    families = ["multi-refactor", "data-pipeline"][:n_families]
    with ResultStore(db_path) as store:
        run_idx = 0
        for model in models:
            for family in families:
                for k in [5, 10]:
                    for i in range(3):
                        success = (i % 2 == 0)
                        run = _make_run(
                            task_id=f"{family}/0.1/inst{i}/k{k}",
                            success=success,
                            partial_credit=1.0 if success else 0.4,
                        )
                        store.save_run(run, model=model, family=family,
                                       instance_id=f"inst{i}", run_idx=run_idx)
                        run_idx += 1


# ── _slug ─────────────────────────────────────────────────────────────────────


def test_slug_basic():
    assert _slug("claude-sonnet-4-6") == "claude_sonnet_4_6"


def test_slug_spaces():
    assert _slug("GPT 4o") == "gpt_4o"


# ── build_site: empty DB ──────────────────────────────────────────────────────


def test_build_site_empty_db(tmp_path):
    db = tmp_path / "empty.duckdb"
    with ResultStore(db):
        pass  # create empty DB
    out = tmp_path / "dist"
    n = build_site(db_path=db, output_dir=out)
    assert n >= 1
    assert (out / "index.html").exists()
    content = (out / "index.html").read_text()
    assert "horizonbench run" in content.lower() or "no results" in content.lower()


# ── build_site: with data ─────────────────────────────────────────────────────


def test_build_site_creates_index(tmp_path):
    db = tmp_path / "test.duckdb"
    _populate_db(db)
    out = tmp_path / "dist"
    n = build_site(db_path=db, output_dir=out)
    assert (out / "index.html").exists()
    content = (out / "index.html").read_text()
    assert "model-0" in content
    assert "multi-refactor" in content


def test_build_site_creates_model_pages(tmp_path):
    db = tmp_path / "test.duckdb"
    _populate_db(db, n_models=1, n_families=1)
    out = tmp_path / "dist"
    build_site(db_path=db, output_dir=out)
    model_pages = list(out.glob("model_*.html"))
    assert len(model_pages) >= 1
    content = model_pages[0].read_text()
    assert "RDC" in content
    assert "Plotly" in content or "plotly" in content.lower()


def test_build_site_page_count(tmp_path):
    db = tmp_path / "test.duckdb"
    _populate_db(db, n_models=2, n_families=2)
    out = tmp_path / "dist"
    n = build_site(db_path=db, output_dir=out)
    # index + 2 models × 2 families = 5 pages minimum
    assert n >= 5


def test_build_site_with_traces(tmp_path):
    db = tmp_path / "test.duckdb"
    _populate_db(db, n_models=1, n_families=1)
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()

    # Write a fake trace file
    tf = traces_dir / "multi-refactor_k5_run-0000.jsonl"
    with TraceWriter(tf) as tw:
        tw.run_start("multi-refactor/0.1/inst/k5", "model-0", 5)
        tw.message("user", "go")
        tw.message("assistant", "done")
        tw.run_end(False, 0.4, 0.01, 100, 5.0)

    out = tmp_path / "dist"
    n = build_site(db_path=db, output_dir=out, traces_dir=traces_dir)
    # Should have generated at least one trace page
    trace_pages = list(out.glob("trace_*.html"))
    assert len(trace_pages) >= 1
    content = trace_pages[0].read_text()
    assert "FAIL" in content or "fail" in content


def test_build_site_index_sorted_by_rdc_auc(tmp_path):
    """Model with higher success rate should appear first in index."""
    db = tmp_path / "test.duckdb"
    with ResultStore(db) as store:
        # model-good: always success
        for i in range(5):
            store.save_run(
                _make_run("f/0.1/i/k5", success=True, partial_credit=1.0),
                model="model-good", family="multi-refactor", instance_id=f"i{i}", run_idx=i
            )
        # model-bad: always fails
        for i in range(5):
            store.save_run(
                _make_run("f/0.1/j/k5", success=False, partial_credit=0.1),
                model="model-bad", family="multi-refactor", instance_id=f"j{i}", run_idx=i
            )
    out = tmp_path / "dist"
    build_site(db_path=db, output_dir=out)
    content = (out / "index.html").read_text()
    pos_good = content.find("model-good")
    pos_bad = content.find("model-bad")
    assert pos_good < pos_bad, "model-good should appear before model-bad in the leaderboard"


# ── site CLI command ──────────────────────────────────────────────────────────


def test_site_cli_command(tmp_path):
    from typer.testing import CliRunner
    from horizonbench.cli import app

    db = tmp_path / "test.duckdb"
    _populate_db(db, n_models=1, n_families=1)
    out = tmp_path / "dist"

    runner = CliRunner()
    result = runner.invoke(app, ["site", "--db", str(db), "--output-dir", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "index.html").exists()
    assert "Built" in result.output


def test_site_cli_missing_db(tmp_path):
    from typer.testing import CliRunner
    from horizonbench.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["site", "--db", str(tmp_path / "nope.duckdb")])
    assert result.exit_code != 0

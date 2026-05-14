"""Tests for CLI — all four task families, metrics command, generate command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from horizonbench.cli import app

runner = CliRunner()


def _mock_stop_response(content="done"):
    msg = MagicMock()
    msg.role = "assistant"
    msg.content = content
    msg.tool_calls = []

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model_dump.return_value = {}
    return resp


def _run_cli(family, k=2, n_runs=1, tmp_path=None):
    tmp = str(tmp_path) if tmp_path else "/tmp/hb_test"
    mock_resp = _mock_stop_response()
    with patch("litellm.completion", return_value=mock_resp):
        with patch("litellm.completion_cost", return_value=0.001):
            return runner.invoke(
                app,
                [
                    "run",
                    "--model", "claude-sonnet-4-6",
                    "--family", family,
                    "--k", str(k),
                    "--n-runs", str(n_runs),
                    "--output-dir", tmp,
                    "--db", f"{tmp}/test.duckdb",
                ],
            )


# ── list-families ─────────────────────────────────────────────────────────────


def test_list_families():
    result = runner.invoke(app, ["list-families"])
    assert result.exit_code == 0
    assert "multi-refactor" in result.output
    assert "data-pipeline" in result.output
    assert "research-synth" in result.output
    assert "constraint-plan" in result.output


# ── unknown family ────────────────────────────────────────────────────────────


def test_run_unknown_family(tmp_path):
    result = runner.invoke(
        app,
        ["run", "--model", "gpt-4o", "--family", "nonexistent",
         "--output-dir", str(tmp_path), "--db", str(tmp_path / "t.duckdb")]
    )
    assert result.exit_code != 0
    assert "Unknown task family" in result.output


# ── per-family smoke runs ─────────────────────────────────────────────────────


@pytest.mark.parametrize("family", ["multi-refactor", "data-pipeline", "research-synth", "constraint-plan"])
def test_run_each_family(tmp_path, family):
    result = _run_cli(family, k=2, n_runs=1, tmp_path=tmp_path)
    assert result.exit_code == 0, f"{family} failed:\n{result.output}"
    summary = json.loads((tmp_path / f"{family}_k2_summary.json").read_text())
    assert summary["family"] == family
    assert summary["n_runs"] == 1


# ── multi-run ─────────────────────────────────────────────────────────────────


def test_run_two_runs(tmp_path):
    result = _run_cli("multi-refactor", k=2, n_runs=2, tmp_path=tmp_path)
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "multi-refactor_k2_summary.json").read_text())
    assert len(data["runs"]) == 2


# ── trace written ─────────────────────────────────────────────────────────────


def test_run_writes_trace(tmp_path):
    result = _run_cli("multi-refactor", k=2, n_runs=1, tmp_path=tmp_path)
    assert result.exit_code == 0, result.output
    trace_files = list(tmp_path.glob("*.jsonl"))
    assert len(trace_files) == 1
    lines = trace_files[0].read_text().strip().splitlines()
    assert len(lines) >= 3  # run_start + messages + run_end


# ── DuckDB written ────────────────────────────────────────────────────────────


def test_run_writes_to_db(tmp_path):
    result = _run_cli("multi-refactor", k=2, n_runs=2, tmp_path=tmp_path)
    assert result.exit_code == 0, result.output
    from horizonbench.results.store import ResultStore
    with ResultStore(tmp_path / "test.duckdb") as store:
        runs = store.get_runs(model="claude-sonnet-4-6", family="multi-refactor")
    assert len(runs) == 2


# ── summary output ────────────────────────────────────────────────────────────


def test_run_output_contains_stats(tmp_path):
    result = _run_cli("multi-refactor", k=3, n_runs=1, tmp_path=tmp_path)
    assert result.exit_code == 0, result.output
    assert "Success rate" in result.output
    assert "Mean partial credit" in result.output
    assert "RDC(k)" in result.output


# ── metrics command ───────────────────────────────────────────────────────────


def test_metrics_no_db(tmp_path):
    result = runner.invoke(
        app,
        ["metrics", "--model", "m", "--family", "multi-refactor",
         "--db", str(tmp_path / "nonexistent.duckdb")]
    )
    assert result.exit_code != 0


def test_metrics_after_run(tmp_path):
    _run_cli("multi-refactor", k=2, n_runs=2, tmp_path=tmp_path)
    result = runner.invoke(
        app,
        ["metrics", "--model", "claude-sonnet-4-6", "--family", "multi-refactor",
         "--db", str(tmp_path / "test.duckdb")]
    )
    assert result.exit_code == 0, result.output
    assert "RDC" in result.output or "MOP" in result.output


# ── generate command ──────────────────────────────────────────────────────────


def test_generate_dry_run(tmp_path):
    result = runner.invoke(
        app,
        [
            "generate", "generate",
            "--release", "test-v0",
            "--out-dir", str(tmp_path),
            "--families", "multi-refactor",
            "--k-values", "5,10",
            "--n-instances", "2",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    # Dry run: no files written
    assert not (tmp_path / "test-v0").exists()


def test_generate_creates_files(tmp_path):
    result = runner.invoke(
        app,
        [
            "generate", "generate",
            "--release", "test-v0",
            "--out-dir", str(tmp_path),
            "--families", "multi-refactor",
            "--k-values", "5",
            "--n-instances", "2",
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = tmp_path / "test-v0" / "manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert "multi-refactor" in data["families"]
    assert len(data["families"]["multi-refactor"]) == 2  # 1 k * 2 instances

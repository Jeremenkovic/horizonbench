"""Tests for --extra and --sandbox CLI flags."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from horizonbench.cli import app, _build_adapter
from horizonbench.adapters.base import AdapterConfig

runner = CliRunner()


def _mock_resp():
    msg = MagicMock()
    msg.role = "assistant"
    msg.content = "done"
    msg.tool_calls = []
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 5
    usage.total_tokens = 10
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model_dump.return_value = {}
    return resp


# ── _build_adapter ────────────────────────────────────────────────────────────


def test_build_adapter_no_extra():
    adapter = _build_adapter("claude-sonnet-4-6", None)
    assert adapter.config.model == "claude-sonnet-4-6"
    assert adapter.config.extra == {}


def test_build_adapter_with_extra():
    adapter = _build_adapter("openai/llama-3", '{"api_base": "http://localhost:8000/v1"}')
    assert adapter.config.extra["api_base"] == "http://localhost:8000/v1"


def test_build_adapter_invalid_json():
    import click
    with pytest.raises((SystemExit, click.exceptions.Exit)):
        _build_adapter("model", "not-json")


# ── --extra flag in CLI ────────────────────────────────────────────────────────


def test_run_with_extra_json(tmp_path):
    mock_resp = _mock_resp()
    with patch("litellm.completion", return_value=mock_resp):
        with patch("litellm.completion_cost", return_value=0.0):
            result = runner.invoke(
                app,
                [
                    "run",
                    "--model", "openai/llama-3",
                    "--family", "multi-refactor",
                    "--k", "2",
                    "--n-runs", "1",
                    "--output-dir", str(tmp_path),
                    "--db", str(tmp_path / "t.duckdb"),
                    "--extra", '{"api_base": "http://localhost:8000/v1"}',
                ],
            )
    assert result.exit_code == 0, result.output


def test_run_with_invalid_extra(tmp_path):
    result = runner.invoke(
        app,
        [
            "run",
            "--model", "m",
            "--family", "multi-refactor",
            "--k", "2",
            "--output-dir", str(tmp_path),
            "--db", str(tmp_path / "t.duckdb"),
            "--extra", "not-valid-json",
        ],
    )
    assert result.exit_code != 0


# ── --sandbox flag ────────────────────────────────────────────────────────────


def test_run_sandbox_local(tmp_path):
    mock_resp = _mock_resp()
    with patch("litellm.completion", return_value=mock_resp):
        with patch("litellm.completion_cost", return_value=0.0):
            result = runner.invoke(
                app,
                [
                    "run", "--model", "claude-sonnet-4-6",
                    "--family", "multi-refactor", "--k", "2", "--n-runs", "1",
                    "--sandbox", "local",
                    "--output-dir", str(tmp_path), "--db", str(tmp_path / "t.duckdb"),
                ],
            )
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "multi-refactor_k2_summary.json").read_text())
    assert data["sandbox"] == "local"


def test_run_sandbox_invalid(tmp_path):
    result = runner.invoke(
        app,
        [
            "run", "--model", "m", "--family", "multi-refactor", "--k", "2",
            "--sandbox", "kubernetes",
            "--output-dir", str(tmp_path), "--db", str(tmp_path / "t.duckdb"),
        ],
    )
    assert result.exit_code != 0

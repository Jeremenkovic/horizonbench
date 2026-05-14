"""Smoke test: claude-sonnet-4-6 on multi-refactor k=5, n-runs=2.

Requires ANTHROPIC_API_KEY in the environment.  Skipped otherwise.
This test makes real API calls and will incur a small cost (~$0.10–0.50).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from horizonbench.cli import _run_single, app
from horizonbench.tasks.multi_refactor import MultiRefactorTask

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live smoke test",
)

runner = CliRunner()


def test_smoke_cli_multi_refactor_k5_n2(tmp_path):
    """Full CLI smoke test: 2 runs of multi-refactor at k=5."""
    result = runner.invoke(
        app,
        [
            "run",
            "--model", "claude-sonnet-4-6",
            "--family", "multi-refactor",
            "--k", "5",
            "--n-runs", "2",
            "--output-dir", str(tmp_path),
            "--max-turns", "30",
        ],
    )

    assert result.exit_code == 0, f"CLI failed:\n{result.output}"

    # Summary file exists and is well-formed
    summary_path = tmp_path / "multi-refactor_k5_summary.json"
    assert summary_path.exists(), "Summary JSON not written"
    data = json.loads(summary_path.read_text())

    assert data["model"] == "claude-sonnet-4-6"
    assert data["family"] == "multi-refactor"
    assert data["k"] == 5
    assert data["n_runs"] == 2
    assert len(data["runs"]) == 2

    # Each run has partial credit in [0, 1]
    for run in data["runs"]:
        assert 0.0 <= run["partial_credit"] <= 1.0, f"Bad partial credit: {run}"
        assert run["wall_seconds"] > 0

    # Trace files written for each run
    trace_files = sorted(tmp_path.glob("*.jsonl"))
    assert len(trace_files) == 2, f"Expected 2 trace files, got {len(trace_files)}"
    for tf in trace_files:
        lines = tf.read_text().strip().splitlines()
        assert len(lines) >= 2, f"Trace too short in {tf.name}"

    # Cost was tracked (may be 0.0 if LiteLLM cost estimate fails, but must be present)
    assert "total_cost_usd" in data

    print(f"\nSmoke test results:")
    print(f"  Success rate: {data['success_rate']:.0%}")
    print(f"  Mean partial credit: {data['mean_partial_credit']:.3f}")
    print(f"  Total cost: ${data['total_cost_usd']:.4f}")


def test_smoke_single_run_direct(tmp_path):
    """Direct _run_single smoke — no CLI overhead, easier to debug."""
    task_cls = MultiRefactorTask
    result = _run_single(
        task_cls=task_cls,
        model="claude-sonnet-4-6",
        k=5,
        instance_id="smoke-direct-001",
        workspace=tmp_path,
        max_turns=30,
    )

    assert result.error is None, f"Run errored: {result.error}"
    assert 0.0 <= result.partial_credit <= 1.0
    assert result.wall_seconds > 0
    assert result.total_tokens > 0
    print(f"\n  partial_credit={result.partial_credit:.2f}  tokens={result.total_tokens}  cost=${result.cost_usd:.4f}")

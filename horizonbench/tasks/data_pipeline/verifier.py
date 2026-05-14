"""Verifier for data_pipeline task.

Compares agent-produced CSV for each stage against ground-truth CSV.
Rows are compared after sorting by all columns to be order-independent.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _read_csv_sorted(path: Path) -> list[tuple]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return []
    keys = sorted(rows[0].keys())
    return sorted(tuple(str(r.get(k, "")) for k in keys) for r in rows)


def verify_stage(stage_spec: dict[str, Any], workspace: Path) -> tuple[bool, str]:
    """Compare agent output vs ground truth for one stage.

    Returns (passed, detail).
    """
    output_filename = stage_spec["output"]
    agent_path = workspace / "data" / output_filename
    gt_path = workspace / "ground_truth" / output_filename

    if not agent_path.exists():
        return False, f"output file missing: data/{output_filename}"
    if not gt_path.exists():
        return False, f"ground truth missing (internal error): {output_filename}"

    agent_rows = _read_csv_sorted(agent_path)
    gt_rows = _read_csv_sorted(gt_path)

    if agent_rows == gt_rows:
        return True, "exact match"
    return (
        False,
        f"mismatch: got {len(agent_rows)} rows, expected {len(gt_rows)} rows",
    )

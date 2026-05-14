"""Generator for data_pipeline task.

Produces a directory with:
  data/stage_00_input.csv   — initial raw data
  pipeline.json             — ordered list of transformation specs
  ground_truth/stage_NN.csv — expected output for each stage (hidden from agent)

Stage operations (cycling for k > 3):
  0 mod 3 == 0 → filter:   keep rows where `value` > threshold
  1 mod 3 == 1 → compute:  add column `doubled` = value * 2
  2 mod 3 == 2 → groupby:  sum `value` by `category`
"""

from __future__ import annotations

import csv
import json
import random
from io import StringIO
from pathlib import Path
from typing import Any

CATEGORIES = ["alpha", "beta", "gamma", "delta"]


def _generate_csv(rng: random.Random, n_rows: int = 30) -> list[dict[str, Any]]:
    return [
        {
            "id": i,
            "category": rng.choice(CATEGORIES),
            "value": rng.randint(10, 100),
            "label": f"item_{i:03d}",
        }
        for i in range(n_rows)
    ]


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _numeric_col(rows: list[dict]) -> str:
    """Return the first numeric-looking column to use for filter/compute."""
    if not rows:
        return "value"
    for col in ("value", "value_sum", "doubled"):
        if col in rows[0]:
            return col
    # Fallback: first column that looks numeric
    for col, val in rows[0].items():
        try:
            int(val)
            return col
        except (ValueError, TypeError):
            pass
    return list(rows[0].keys())[-1]


def _apply_filter(rows: list[dict], threshold: int) -> list[dict]:
    col = _numeric_col(rows)
    return [r for r in rows if int(r.get(col, 0)) > threshold]


def _apply_compute(rows: list[dict]) -> list[dict]:
    col = _numeric_col(rows)
    return [{**r, "doubled": int(r.get(col, 0)) * 2} for r in rows]


def _apply_groupby(rows: list[dict]) -> list[dict]:
    col = _numeric_col(rows)
    totals: dict[str, int] = {}
    for r in rows:
        cat = r.get("category", "unknown")
        totals[cat] = totals.get(cat, 0) + int(r.get(col, 0))
    return [{"category": k, "value_sum": v} for k, v in sorted(totals.items())]


def _stage_name(i: int) -> str:
    return f"stage_{i + 1:02d}_output.csv"


def generate_workspace(k: int, seed: int, workspace: Path) -> dict[str, Any]:
    rng = random.Random(seed)
    data_dir = workspace / "data"
    gt_dir = workspace / "ground_truth"
    data_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    initial_rows = _generate_csv(rng)
    _write_csv(initial_rows, data_dir / "stage_00_input.csv")

    stages = []
    current_rows = initial_rows
    for i in range(k):
        op_idx = i % 3
        if op_idx == 0:
            threshold = rng.randint(20, 60)
            spec = {
                "index": i,
                "op": "filter",
                "column": "value",
                "threshold": threshold,
                "input": f"stage_{i:02d}_input.csv" if i == 0 else _stage_name(i - 1),
                "output": _stage_name(i),
            }
            current_rows = _apply_filter(current_rows, threshold)
        elif op_idx == 1:
            spec = {
                "index": i,
                "op": "compute",
                "new_column": "doubled",
                "source_column": "value",
                "multiplier": 2,
                "input": f"stage_{i:02d}_input.csv" if i == 0 else _stage_name(i - 1),
                "output": _stage_name(i),
            }
            current_rows = _apply_compute(current_rows)
        else:
            spec = {
                "index": i,
                "op": "groupby",
                "group_column": "category",
                "agg_column": "value",
                "agg": "sum",
                "output_column": "value_sum",
                "input": f"stage_{i:02d}_input.csv" if i == 0 else _stage_name(i - 1),
                "output": _stage_name(i),
            }
            current_rows = _apply_groupby(current_rows)

        stages.append(spec)
        _write_csv(current_rows, gt_dir / _stage_name(i))

    (workspace / "pipeline.json").write_text(json.dumps({"stages": stages}, indent=2))
    return {"stages": stages, "k": k, "seed": seed}

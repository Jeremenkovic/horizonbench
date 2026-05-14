"""Generator for constraint_plan task.

Produces a scheduling problem with N tasks and k constraints.
The agent must produce workspace/schedule.json satisfying all constraints.

Problem JSON (workspace/problem.json):
{
  "horizon": 24,          # total time slots available (hours)
  "tasks": [
    {"id": "T0", "name": "...", "duration": 2}
  ],
  "constraints": [
    {"type": "time_window", "task": "T0", "earliest_start": 0, "latest_end": 8},
    {"type": "precedence",  "before": "T0", "after": "T1"},
    {"type": "no_overlap",  "tasks": ["T0", "T2"]},
    {"type": "max_parallel","tasks": ["T1", "T2", "T3"], "limit": 2}
  ]
}

Expected schedule JSON (workspace/schedule.json — agent writes this):
{
  "schedule": [
    {"task": "T0", "start": 0, "end": 2},
    ...
  ]
}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

TASK_NAMES = [
    "database migration", "cache warmup", "index rebuild", "backup creation",
    "log rotation", "config reload", "health check", "report generation",
    "batch export", "cleanup job", "sync task", "validation run",
]


def _gen_tasks(rng: random.Random, n: int) -> list[dict[str, Any]]:
    return [
        {"id": f"T{i}", "name": TASK_NAMES[i % len(TASK_NAMES)], "duration": rng.randint(1, 4)}
        for i in range(n)
    ]


def _gen_constraints(
    rng: random.Random, tasks: list[dict], k: int, horizon: int
) -> list[dict[str, Any]]:
    """Generate exactly k constraints, cycling through types."""
    task_ids = [t["id"] for t in tasks]
    constraints = []
    for i in range(k):
        ctype = i % 4
        if ctype == 0 and task_ids:
            tid = rng.choice(task_ids)
            dur = next(t["duration"] for t in tasks if t["id"] == tid)
            es = rng.randint(0, max(0, horizon // 2 - dur))
            le = rng.randint(es + dur, horizon)
            constraints.append(
                {"type": "time_window", "task": tid, "earliest_start": es, "latest_end": le}
            )
        elif ctype == 1 and len(task_ids) >= 2:
            before, after = rng.sample(task_ids, 2)
            constraints.append({"type": "precedence", "before": before, "after": after})
        elif ctype == 2 and len(task_ids) >= 2:
            n_overlap = rng.randint(2, min(3, len(task_ids)))
            group = rng.sample(task_ids, n_overlap)
            constraints.append({"type": "no_overlap", "tasks": group})
        else:
            n_group = rng.randint(2, min(4, len(task_ids)))
            group = rng.sample(task_ids, n_group)
            limit = rng.randint(1, max(1, n_group - 1))
            constraints.append({"type": "max_parallel", "tasks": group, "limit": limit})
    return constraints


def generate_workspace(k: int, seed: int, workspace: Path) -> dict[str, Any]:
    rng = random.Random(seed)
    horizon = 24
    n_tasks = max(k + 1, 4)   # always more tasks than constraints, min 4
    tasks = _gen_tasks(rng, n_tasks)
    constraints = _gen_constraints(rng, tasks, k, horizon)

    problem = {"horizon": horizon, "tasks": tasks, "constraints": constraints}
    (workspace / "problem.json").write_text(json.dumps(problem, indent=2))

    return {"tasks": tasks, "constraints": constraints, "horizon": horizon, "k": k, "seed": seed}

"""Verifier for constraint_plan task.

Loads agent-produced schedule.json and checks each constraint.
Returns per-constraint pass/fail and an overall result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_schedule(workspace: Path) -> dict[str, Any]:
    path = workspace / "schedule.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, TypeError):
        return {}


def _build_slot_map(schedule: dict) -> dict[str, tuple[int, int]]:
    """Return {task_id: (start, end)} from the agent's schedule."""
    result = {}
    for entry in schedule.get("schedule", []):
        tid = entry.get("task", "")
        start = entry.get("start", 0)
        end = entry.get("end", 0)
        result[tid] = (int(start), int(end))
    return result


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def check_constraint(constraint: dict, slot_map: dict, task_durations: dict) -> tuple[bool, str]:
    """Return (satisfied, detail) for a single constraint."""
    ctype = constraint.get("type", "")

    if ctype == "time_window":
        tid = constraint["task"]
        if tid not in slot_map:
            return False, f"{tid} not scheduled"
        start, end = slot_map[tid]
        es = constraint["earliest_start"]
        le = constraint["latest_end"]
        ok = start >= es and end <= le
        return ok, f"{tid} [{start},{end}] vs window [{es},{le}]"

    elif ctype == "precedence":
        before = constraint["before"]
        after = constraint["after"]
        if before not in slot_map:
            return False, f"{before} not scheduled"
        if after not in slot_map:
            return False, f"{after} not scheduled"
        ok = slot_map[before][1] <= slot_map[after][0]
        return ok, f"{before} ends {slot_map[before][1]}, {after} starts {slot_map[after][0]}"

    elif ctype == "no_overlap":
        tasks = constraint["tasks"]
        scheduled = [(t, slot_map[t]) for t in tasks if t in slot_map]
        for i, (ta, ia) in enumerate(scheduled):
            for tb, ib in scheduled[i + 1:]:
                if _intervals_overlap(ia, ib):
                    return False, f"{ta} and {tb} overlap"
        return True, "no overlaps"

    elif ctype == "max_parallel":
        tasks = constraint["tasks"]
        limit = constraint["limit"]
        scheduled = [(t, slot_map[t]) for t in tasks if t in slot_map]
        # collect all breakpoints
        points = set()
        for _, (s, e) in scheduled:
            points.add(s)
            points.add(e)
        for pt in sorted(points):
            parallel = sum(1 for _, (s, e) in scheduled if s <= pt < e)
            if parallel > limit:
                return False, f"{parallel} tasks running at t={pt} > limit {limit}"
        return True, f"max parallel ≤ {limit}"

    return True, f"unknown constraint type {ctype!r} (skipped)"


def verify_all_constraints(
    constraints: list[dict],
    schedule: dict,
    tasks: list[dict],
) -> list[tuple[bool, str]]:
    task_durations = {t["id"]: t["duration"] for t in tasks}
    slot_map = _build_slot_map(schedule)
    return [check_constraint(c, slot_map, task_durations) for c in constraints]

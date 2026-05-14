"""ConstraintPlanTask — schedule N tasks subject to k constraints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from horizonbench.tasks.base import Checkpoint, CheckpointResult, Task, TaskInstance
from horizonbench.tasks.constraint_plan.generator import generate_workspace
from horizonbench.tasks.constraint_plan.verifier import load_schedule, verify_all_constraints


class ConstraintPlanTask(Task):
    family = "constraint-plan"
    version = "0.1"

    def generate(self, k: int, instance_id: str, workspace: Path) -> TaskInstance:
        seed = self.seed_for(instance_id)
        meta = generate_workspace(k=k, seed=seed, workspace=workspace)
        return TaskInstance(
            family=self.family,
            version=self.version,
            instance_id=instance_id,
            k=k,
            seed=seed,
            workspace=workspace,
            metadata=meta,
        )

    def checkpoints(self, instance: TaskInstance) -> list[Checkpoint]:
        return [
            Checkpoint(
                index=i,
                description=f"Constraint {i}: {c['type']}",
            )
            for i, c in enumerate(instance.metadata.get("constraints", []))
        ]

    def verify_checkpoint(
        self, checkpoint: Checkpoint, instance: TaskInstance, workspace: Path
    ) -> CheckpointResult:
        constraints = instance.metadata.get("constraints", [])
        tasks = instance.metadata.get("tasks", [])
        if checkpoint.index >= len(constraints):
            return CheckpointResult(checkpoint=checkpoint, passed=False, detail="index OOB")
        schedule = load_schedule(workspace)
        if not schedule:
            return CheckpointResult(checkpoint=checkpoint, passed=False, detail="schedule.json missing or invalid")
        results = verify_all_constraints(constraints, schedule, tasks)
        passed, detail = results[checkpoint.index]
        return CheckpointResult(checkpoint=checkpoint, passed=passed, detail=detail)

    def system_prompt(self, instance: TaskInstance) -> str:
        return (
            "You are a scheduling assistant. "
            "You will be given a scheduling problem with tasks and constraints. "
            "Produce a valid schedule that satisfies all constraints. "
            "Write your schedule to schedule.json."
        )

    def user_message(self, instance: TaskInstance) -> str:
        problem = json.dumps(
            {
                "horizon": instance.metadata["horizon"],
                "tasks": instance.metadata["tasks"],
                "constraints": instance.metadata["constraints"],
            },
            indent=2,
        )
        return (
            f"Solve the following scheduling problem with {instance.k} constraints.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Write your solution to schedule.json as:\n"
            '{"schedule": [{"task": "T0", "start": 0, "end": 2}, ...]}'
        )

    def tools(self, instance: TaskInstance) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file to the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ]

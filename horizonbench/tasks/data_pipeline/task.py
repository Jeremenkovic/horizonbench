"""DataPipelineTask — multi-stage CSV ETL task family."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from horizonbench.tasks.base import Checkpoint, CheckpointResult, Task, TaskInstance
from horizonbench.tasks.data_pipeline.generator import generate_workspace
from horizonbench.tasks.data_pipeline.verifier import verify_stage


class DataPipelineTask(Task):
    """k CSV transformation stages that the agent must execute in sequence."""

    family = "data-pipeline"
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
                index=s["index"],
                description=f"Stage {s['index']}: {s['op']} → {s['output']}",
            )
            for s in instance.metadata.get("stages", [])
        ]

    def verify_checkpoint(
        self, checkpoint: Checkpoint, instance: TaskInstance, workspace: Path
    ) -> CheckpointResult:
        stages = instance.metadata.get("stages", [])
        if checkpoint.index >= len(stages):
            return CheckpointResult(checkpoint=checkpoint, passed=False, detail="index OOB")
        passed, detail = verify_stage(stages[checkpoint.index], workspace)
        return CheckpointResult(checkpoint=checkpoint, passed=passed, detail=detail)

    def system_prompt(self, instance: TaskInstance) -> str:
        return (
            "You are a data-engineering assistant. "
            "You will be given a CSV dataset and a pipeline specification. "
            "Execute each transformation stage in order, reading from the previous stage's output "
            "and writing the result to the specified output file in the data/ directory. "
            "Use only standard library Python (csv module) — no pandas."
        )

    def user_message(self, instance: TaskInstance) -> str:
        pipeline = json.dumps(instance.metadata.get("stages", []), indent=2)
        return (
            f"Execute the {instance.k}-stage data pipeline defined in pipeline.json.\n\n"
            f"The initial data is at data/stage_00_input.csv.\n"
            f"For each stage, read the input file, apply the transformation, "
            f"and write the result to data/<output> as a CSV.\n\n"
            f"Pipeline:\n{pipeline}"
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
            {
                "type": "function",
                "function": {
                    "name": "run_bash",
                    "description": "Run a shell command in the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
        ]

"""MultiRefactorTask — multi-file camelCase refactor task family."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from horizonbench.tasks.base import Checkpoint, CheckpointResult, Task, TaskInstance
from horizonbench.tasks.multi_refactor.generator import generate_workspace
from horizonbench.tasks.multi_refactor.verifier import run_pytest, verify_file


class MultiRefactorTask(Task):
    """Each instance has k Python source files that need a snake→camelCase rename."""

    family = "multi-refactor"
    version = "0.1"

    # ── generation ────────────────────────────────────────────────────────────

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

    # ── checkpoints ───────────────────────────────────────────────────────────

    def checkpoints(self, instance: TaskInstance) -> list[Checkpoint]:
        files = instance.metadata.get("files", [])
        return [
            Checkpoint(
                index=f["index"],
                description=f"Rename {f['old_name']} → {f['new_name']} in {f['filename']}",
                weight=1.0,
            )
            for f in files
        ]

    # ── per-checkpoint verification ───────────────────────────────────────────

    def verify_checkpoint(
        self,
        checkpoint: Checkpoint,
        instance: TaskInstance,
        workspace: Path,
    ) -> CheckpointResult:
        files = instance.metadata.get("files", [])
        if checkpoint.index >= len(files):
            return CheckpointResult(checkpoint=checkpoint, passed=False, detail="index OOB")
        file_meta = files[checkpoint.index]
        result = verify_file(checkpoint.index, file_meta, workspace)
        return CheckpointResult(
            checkpoint=checkpoint,
            passed=result["passed"],
            detail=result["ast_detail"],
        )

    # ── full verification shortcut (also runs pytest) ─────────────────────────

    def verify_with_pytest(
        self, instance: TaskInstance, workspace: Path
    ) -> tuple[list[CheckpointResult], bool, str]:
        """Verify all checkpoints AND run the embedded pytest suite.

        Returns (checkpoint_results, pytest_passed, pytest_output).
        """
        cp_results = self.verify_all(instance, workspace)
        pytest_passed, pytest_output = run_pytest(workspace)
        return cp_results, pytest_passed, pytest_output

    # ── agent interface ────────────────────────────────────────────────────────

    def system_prompt(self, instance: TaskInstance) -> str:
        return (
            "You are a precise code-refactoring assistant. "
            "You will be given a Python repository. "
            "Your job is to rename functions according to the instructions in each file. "
            "Make only the requested changes — do not alter logic, imports, or comments. "
            "All files are in the current working directory. Use relative paths (e.g. src/module_000.py). "
            "Never use 'find /' — use 'find .' or 'find src/' instead."
        )

    def user_message(self, instance: TaskInstance) -> str:
        files = instance.metadata.get("files", [])
        file_list = "\n".join(
            f"  - src/{f['filename']}: rename `{f['old_name']}` → `{f['new_name']}`"
            for f in files
        )
        return (
            f"Refactor the following {instance.k} files in the repository.\n"
            f"Files are at relative paths under src/. Use read_file with e.g. 'src/module_000.py'.\n"
            f"For each file, rename the specified function to its camelCase equivalent.\n"
            f"Also update any internal calls to the renamed function.\n\n"
            f"Files to refactor:\n{file_list}\n\n"
            f"After completing all renames, run `pytest tests/` to verify your changes."
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
                        "properties": {
                            "path": {"type": "string", "description": "Relative path from /work/"}
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write (overwrite) a file in the workspace.",
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
                    "description": "Run a shell command in the workspace and return stdout+stderr.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                        },
                        "required": ["command"],
                    },
                },
            },
        ]

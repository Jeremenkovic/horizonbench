"""ResearchSynthTask — structured fact extraction from N sandboxed documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from horizonbench.tasks.base import Checkpoint, CheckpointResult, Task, TaskInstance
from horizonbench.tasks.research_synth.generator import generate_workspace
from horizonbench.tasks.research_synth.verifier import load_extracted, verify_document


class ResearchSynthTask(Task):
    family = "research-synth"
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
            Checkpoint(index=i, description=f"Extract facts from {doc_id}")
            for i, doc_id in enumerate(instance.metadata.get("doc_ids", []))
        ]

    def verify_checkpoint(
        self, checkpoint: Checkpoint, instance: TaskInstance, workspace: Path
    ) -> CheckpointResult:
        doc_ids = instance.metadata.get("doc_ids", [])
        if checkpoint.index >= len(doc_ids):
            return CheckpointResult(checkpoint=checkpoint, passed=False, detail="index OOB")
        doc_id = doc_ids[checkpoint.index]
        gt = instance.metadata.get("ground_truth", {}).get(doc_id, {})
        extracted = load_extracted(workspace)
        f1, detail = verify_document(doc_id, gt, extracted)
        return CheckpointResult(checkpoint=checkpoint, passed=(f1 == 1.0), detail=detail)

    def partial_credit(self, results: list[CheckpointResult]) -> float:
        """Mean F1 across documents (extracted from the detail string)."""
        if not results:
            return 0.0
        total = 0.0
        for r in results:
            if r.passed:
                total += 1.0
            elif "F1=" in r.detail:
                try:
                    total += float(r.detail.split("F1=")[1].split(" ")[0])
                except (ValueError, IndexError):
                    pass
        return total / len(results)

    def system_prompt(self, instance: TaskInstance) -> str:
        return (
            "You are a research assistant that extracts structured facts from documents. "
            "For each document you read, extract all key-value facts of the form "
            "'entity:attribute → value unit'. "
            "Write your extracted facts to extracted_facts.json as a JSON object: "
            '{"doc_id": {"entity:attribute": "value unit", ...}, ...}.'
        )

    def user_message(self, instance: TaskInstance) -> str:
        doc_ids = instance.metadata.get("doc_ids", [])
        return (
            f"Read the following {instance.k} documents and extract all facts from each.\n\n"
            f"Documents available (use read_document tool):\n"
            + "\n".join(f"  - {d}" for d in doc_ids)
            + "\n\nWrite all extracted facts to extracted_facts.json."
        )

    def tools(self, instance: TaskInstance) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_documents",
                    "description": "List available document IDs.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_document",
                    "description": "Read a document by its ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {"doc_id": {"type": "string"}},
                        "required": ["doc_id"],
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

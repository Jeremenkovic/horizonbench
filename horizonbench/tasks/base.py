"""Task ABC — checkpoint protocol, partial-credit rubric, step parameter k."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Checkpoint:
    """A single verifiable milestone within a task."""

    index: int          # 0-based ordering
    description: str    # human-readable label
    weight: float = 1.0  # relative weight for partial credit (default uniform)


@dataclass
class CheckpointResult:
    checkpoint: Checkpoint
    passed: bool
    detail: str = ""


@dataclass
class TaskInstance:
    """A concrete, generated instance of a task family."""

    family: str
    version: str
    instance_id: str
    k: int                            # step-parameter (number of files, stages, etc.)
    seed: int                         # deterministic seed derived from (family, version, instance_id)
    workspace: Path                   # path to the generated sandbox workspace
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return f"{self.family}/{self.version}/{self.instance_id}/k{self.k}"


@dataclass
class RunResult:
    """Full result of a single task run."""

    task_id: str
    success: bool
    partial_credit: float             # [0, 1]
    checkpoint_results: list[CheckpointResult] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    total_tokens: int = 0
    wall_seconds: float = 0.0
    error: str | None = None


def _derive_seed(family: str, version: str, instance_id: str) -> int:
    """Deterministic seed from (family, version, instance_id) — SHA-256 fold to 32-bit."""
    raw = json.dumps([family, version, instance_id], sort_keys=True).encode()
    digest = hashlib.sha256(raw).digest()
    # fold 32 bytes into a single positive 32-bit integer
    value = int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF
    return value


class Task(ABC):
    """Abstract base for every HorizonBench task family.

    Subclass contract:
    1. ``generate`` — builds a TaskInstance (workspace on disk) from (k, instance_id).
    2. ``checkpoints`` — returns the ordered list of Checkpoints for an instance.
    3. ``verify_checkpoint`` — tests whether a single checkpoint is satisfied in
       the current workspace state.
    4. ``partial_credit`` — aggregates checkpoint results into [0, 1].
    5. ``system_prompt`` — returns the family-level system prompt (same for all models).
    6. ``tools`` — returns the LiteLLM-compatible tool schemas for this family.
    """

    # Subclasses set these as class attributes.
    family: str = ""
    version: str = "0.1"

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def generate(self, k: int, instance_id: str, workspace: Path) -> TaskInstance:
        """Materialise a task instance into *workspace* and return its descriptor."""

    # ------------------------------------------------------------------ #
    # Verification                                                         #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def checkpoints(self, instance: TaskInstance) -> list[Checkpoint]:
        """Return the ordered checkpoints for *instance*."""

    @abstractmethod
    def verify_checkpoint(
        self,
        checkpoint: Checkpoint,
        instance: TaskInstance,
        workspace: Path,
    ) -> CheckpointResult:
        """Test whether *checkpoint* is satisfied; workspace may have been modified by the agent."""

    def verify_all(self, instance: TaskInstance, workspace: Path) -> list[CheckpointResult]:
        """Run every checkpoint in order and return results."""
        return [
            self.verify_checkpoint(cp, instance, workspace)
            for cp in self.checkpoints(instance)
        ]

    # ------------------------------------------------------------------ #
    # Partial-credit rubric                                                #
    # ------------------------------------------------------------------ #

    def partial_credit(self, results: list[CheckpointResult]) -> float:
        """Weighted fraction of checkpoints passed. Override for custom rubrics."""
        if not results:
            return 0.0
        total_weight = sum(r.checkpoint.weight for r in results)
        if total_weight == 0:
            return 0.0
        passed_weight = sum(r.checkpoint.weight for r in results if r.passed)
        return passed_weight / total_weight

    # ------------------------------------------------------------------ #
    # Agent interface                                                       #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def system_prompt(self, instance: TaskInstance) -> str:
        """Return the system prompt for this task family (same for all models)."""

    @abstractmethod
    def user_message(self, instance: TaskInstance) -> str:
        """Return the initial user message describing the task to the agent."""

    def tools(self, instance: TaskInstance) -> list[dict[str, Any]]:
        """LiteLLM-compatible tool schemas. Default: empty (subclasses override)."""
        return []

    # ------------------------------------------------------------------ #
    # Seed helper                                                          #
    # ------------------------------------------------------------------ #

    def seed_for(self, instance_id: str) -> int:
        return _derive_seed(self.family, self.version, instance_id)

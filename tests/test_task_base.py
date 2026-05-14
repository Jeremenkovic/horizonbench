"""Tests for Task ABC and base data structures."""

from pathlib import Path

import pytest

from horizonbench.tasks.base import (
    Checkpoint,
    CheckpointResult,
    Task,
    TaskInstance,
    _derive_seed,
)


# ── concrete stub ────────────────────────────────────────────────────────────


class StubTask(Task):
    family = "stub"
    version = "0.1"

    def generate(self, k: int, instance_id: str, workspace: Path) -> TaskInstance:
        seed = self.seed_for(instance_id)
        return TaskInstance(
            family=self.family,
            version=self.version,
            instance_id=instance_id,
            k=k,
            seed=seed,
            workspace=workspace,
        )

    def checkpoints(self, instance: TaskInstance) -> list[Checkpoint]:
        return [Checkpoint(index=i, description=f"step {i}") for i in range(instance.k)]

    def verify_checkpoint(
        self, checkpoint: Checkpoint, instance: TaskInstance, workspace: Path
    ) -> CheckpointResult:
        marker = workspace / f"step_{checkpoint.index}.done"
        return CheckpointResult(checkpoint=checkpoint, passed=marker.exists())

    def system_prompt(self, instance: TaskInstance) -> str:
        return "You are a stub task agent."

    def user_message(self, instance: TaskInstance) -> str:
        return f"Complete {instance.k} steps."


# ── tests ─────────────────────────────────────────────────────────────────────


def test_seed_determinism():
    s1 = _derive_seed("multi-refactor", "0.1", "inst-001")
    s2 = _derive_seed("multi-refactor", "0.1", "inst-001")
    assert s1 == s2


def test_seed_differs_by_family():
    s1 = _derive_seed("family-a", "0.1", "inst-001")
    s2 = _derive_seed("family-b", "0.1", "inst-001")
    assert s1 != s2


def test_seed_differs_by_instance():
    s1 = _derive_seed("multi-refactor", "0.1", "inst-001")
    s2 = _derive_seed("multi-refactor", "0.1", "inst-002")
    assert s1 != s2


def test_seed_positive():
    assert _derive_seed("x", "0.1", "y") >= 0


def test_generate_returns_instance(tmp_path):
    task = StubTask()
    inst = task.generate(k=5, instance_id="test-001", workspace=tmp_path)
    assert inst.family == "stub"
    assert inst.k == 5
    assert inst.seed == _derive_seed("stub", "0.1", "test-001")


def test_task_id_format(tmp_path):
    task = StubTask()
    inst = task.generate(k=10, instance_id="abc", workspace=tmp_path)
    assert inst.task_id == "stub/0.1/abc/k10"


def test_partial_credit_all_pass(tmp_path):
    task = StubTask()
    inst = task.generate(k=3, instance_id="pc-test", workspace=tmp_path)
    # create all markers
    for i in range(3):
        (tmp_path / f"step_{i}.done").touch()
    results = task.verify_all(inst, tmp_path)
    assert all(r.passed for r in results)
    assert task.partial_credit(results) == 1.0


def test_partial_credit_none_pass(tmp_path):
    task = StubTask()
    inst = task.generate(k=3, instance_id="pc-test2", workspace=tmp_path)
    results = task.verify_all(inst, tmp_path)
    assert task.partial_credit(results) == 0.0


def test_partial_credit_half_pass(tmp_path):
    task = StubTask()
    inst = task.generate(k=4, instance_id="pc-test3", workspace=tmp_path)
    for i in range(2):
        (tmp_path / f"step_{i}.done").touch()
    results = task.verify_all(inst, tmp_path)
    assert task.partial_credit(results) == pytest.approx(0.5)


def test_partial_credit_custom_weights():
    cp_a = Checkpoint(index=0, description="a", weight=3.0)
    cp_b = Checkpoint(index=1, description="b", weight=1.0)
    results = [
        CheckpointResult(checkpoint=cp_a, passed=True),
        CheckpointResult(checkpoint=cp_b, passed=False),
    ]
    task = StubTask()
    assert task.partial_credit(results) == pytest.approx(0.75)


def test_partial_credit_empty():
    task = StubTask()
    assert task.partial_credit([]) == 0.0

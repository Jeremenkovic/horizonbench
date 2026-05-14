"""Tests for multi_refactor task — generator, verifier, partial credit."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from horizonbench.tasks.multi_refactor import MultiRefactorTask
from horizonbench.tasks.multi_refactor.generator import (
    _camel,
    _module_name,
    _snake,
    generate_workspace,
)
from horizonbench.tasks.multi_refactor.verifier import check_ast, run_pytest


# ── generator tests ───────────────────────────────────────────────────────────


def test_generate_creates_k_files(tmp_path):
    meta = generate_workspace(k=5, seed=42, workspace=tmp_path)
    src = tmp_path / "src"
    assert len(list(src.glob("module_*.py"))) == 5
    assert len(meta["files"]) == 5


def test_generate_files_have_old_name(tmp_path):
    meta = generate_workspace(k=3, seed=99, workspace=tmp_path)
    for f in meta["files"]:
        source = (tmp_path / "src" / f["filename"]).read_text()
        assert f"def {f['old_name']}(" in source


def test_generate_creates_test_suite(tmp_path):
    generate_workspace(k=3, seed=7, workspace=tmp_path)
    assert (tmp_path / "tests" / "test_refactor.py").exists()


def test_generate_deterministic(tmp_path):
    m1 = generate_workspace(k=4, seed=123, workspace=tmp_path / "a")
    m2 = generate_workspace(k=4, seed=123, workspace=tmp_path / "b")
    for f1, f2 in zip(m1["files"], m2["files"]):
        assert f1["constant"] == f2["constant"]


def test_generate_different_seeds_differ(tmp_path):
    m1 = generate_workspace(k=4, seed=1, workspace=tmp_path / "a")
    m2 = generate_workspace(k=4, seed=2, workspace=tmp_path / "b")
    constants_1 = [f["constant"] for f in m1["files"]]
    constants_2 = [f["constant"] for f in m2["files"]]
    assert constants_1 != constants_2


# ── verifier: check_ast ───────────────────────────────────────────────────────


def test_check_ast_old_name_present(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("def compute_value_0(x): return x\n")
    passed, detail = check_ast(f, "compute_value_0", "computeValue0")
    assert not passed
    assert "old name" in detail


def test_check_ast_new_name_present(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("def computeValue0(x): return x\n")
    passed, detail = check_ast(f, "compute_value_0", "computeValue0")
    assert passed


def test_check_ast_neither_present(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("def something_else(x): return x\n")
    passed, detail = check_ast(f, "compute_value_0", "computeValue0")
    assert not passed


def test_check_ast_missing_file(tmp_path):
    passed, detail = check_ast(tmp_path / "nonexistent.py", "a", "b")
    assert not passed


# ── task-level tests ──────────────────────────────────────────────────────────


def test_task_generate(tmp_path):
    task = MultiRefactorTask()
    inst = task.generate(k=3, instance_id="smoke", workspace=tmp_path)
    assert inst.k == 3
    assert inst.family == "multi-refactor"
    assert len(inst.metadata["files"]) == 3


def test_task_checkpoints_count(tmp_path):
    task = MultiRefactorTask()
    inst = task.generate(k=5, instance_id="cp-test", workspace=tmp_path)
    cps = task.checkpoints(inst)
    assert len(cps) == 5


def test_task_verify_all_before_refactor(tmp_path):
    task = MultiRefactorTask()
    inst = task.generate(k=3, instance_id="before", workspace=tmp_path)
    results = task.verify_all(inst, tmp_path)
    # Before refactor: all should fail (old names still present)
    assert all(not r.passed for r in results)
    assert task.partial_credit(results) == 0.0


def test_task_verify_after_full_refactor(tmp_path):
    task = MultiRefactorTask()
    inst = task.generate(k=3, instance_id="after-full", workspace=tmp_path)
    # Manually apply the refactor to all files
    for f in inst.metadata["files"]:
        src_path = tmp_path / "src" / f["filename"]
        source = src_path.read_text()
        refactored = source.replace(f"def {f['old_name']}(", f"def {f['new_name']}(")
        # also replace the RESULT call
        refactored = refactored.replace(f"{f['old_name']}(10)", f"{f['new_name']}(10)")
        src_path.write_text(refactored)

    results = task.verify_all(inst, tmp_path)
    assert all(r.passed for r in results)
    assert task.partial_credit(results) == 1.0


def test_task_partial_refactor(tmp_path):
    task = MultiRefactorTask()
    inst = task.generate(k=4, instance_id="partial", workspace=tmp_path)
    # Only refactor the first 2 files
    for f in inst.metadata["files"][:2]:
        src_path = tmp_path / "src" / f["filename"]
        source = src_path.read_text()
        refactored = source.replace(f"def {f['old_name']}(", f"def {f['new_name']}(")
        refactored = refactored.replace(f"{f['old_name']}(10)", f"{f['new_name']}(10)")
        src_path.write_text(refactored)

    results = task.verify_all(inst, tmp_path)
    assert task.partial_credit(results) == pytest.approx(0.5)


def test_pytest_passes_after_full_refactor(tmp_path):
    task = MultiRefactorTask()
    inst = task.generate(k=2, instance_id="pytest-check", workspace=tmp_path)
    for f in inst.metadata["files"]:
        src_path = tmp_path / "src" / f["filename"]
        source = src_path.read_text()
        refactored = source.replace(f"def {f['old_name']}(", f"def {f['new_name']}(")
        refactored = refactored.replace(f"{f['old_name']}(10)", f"{f['new_name']}(10)")
        src_path.write_text(refactored)

    pytest_passed, output = run_pytest(tmp_path)
    assert pytest_passed, f"pytest failed:\n{output}"


def test_pytest_fails_before_refactor(tmp_path):
    task = MultiRefactorTask()
    inst = task.generate(k=2, instance_id="pytest-fail", workspace=tmp_path)
    pytest_passed, _ = run_pytest(tmp_path)
    assert not pytest_passed

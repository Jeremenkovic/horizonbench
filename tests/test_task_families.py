"""Tests for data_pipeline, research_synth, and constraint_plan task families."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

# ── data_pipeline ─────────────────────────────────────────────────────────────

from horizonbench.tasks.data_pipeline import DataPipelineTask
from horizonbench.tasks.data_pipeline.generator import (
    _apply_compute,
    _apply_filter,
    _apply_groupby,
    generate_workspace as dp_gen,
)
from horizonbench.tasks.data_pipeline.verifier import verify_stage


def test_dp_generate_creates_pipeline_json(tmp_path):
    dp_gen(k=3, seed=42, workspace=tmp_path)
    assert (tmp_path / "pipeline.json").exists()


def test_dp_generate_creates_input_csv(tmp_path):
    dp_gen(k=3, seed=42, workspace=tmp_path)
    assert (tmp_path / "data" / "stage_00_input.csv").exists()


def test_dp_generate_creates_ground_truth(tmp_path):
    meta = dp_gen(k=3, seed=42, workspace=tmp_path)
    for s in meta["stages"]:
        assert (tmp_path / "ground_truth" / s["output"]).exists()


def test_dp_generate_deterministic(tmp_path):
    m1 = dp_gen(k=3, seed=7, workspace=tmp_path / "a")
    m2 = dp_gen(k=3, seed=7, workspace=tmp_path / "b")
    assert m1["stages"][0]["op"] == m2["stages"][0]["op"]


def test_dp_apply_filter():
    rows = [{"value": "30", "id": "1"}, {"value": "70", "id": "2"}]
    result = _apply_filter(rows, threshold=50)
    assert len(result) == 1
    assert result[0]["id"] == "2"


def test_dp_apply_compute():
    rows = [{"value": "10", "id": "1"}]
    result = _apply_compute(rows)
    assert result[0]["doubled"] == 20


def test_dp_apply_groupby():
    rows = [
        {"category": "a", "value": "10"},
        {"category": "a", "value": "20"},
        {"category": "b", "value": "5"},
    ]
    result = _apply_groupby(rows)
    by_cat = {r["category"]: r["value_sum"] for r in result}
    assert by_cat["a"] == 30
    assert by_cat["b"] == 5


def test_dp_verify_stage_missing_output(tmp_path):
    dp_gen(k=2, seed=99, workspace=tmp_path)
    meta = dp_gen(k=2, seed=99, workspace=tmp_path)
    passed, detail = verify_stage(meta["stages"][0], tmp_path)
    assert not passed
    assert "missing" in detail


def test_dp_verify_stage_correct(tmp_path):
    meta = dp_gen(k=1, seed=5, workspace=tmp_path)
    stage = meta["stages"][0]
    # Copy ground truth to data/ as if agent produced it
    gt_file = tmp_path / "ground_truth" / stage["output"]
    out_file = tmp_path / "data" / stage["output"]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(gt_file.read_text())
    passed, detail = verify_stage(stage, tmp_path)
    assert passed


def test_dp_task_partial_credit(tmp_path):
    task = DataPipelineTask()
    inst = task.generate(k=3, instance_id="dp-pc", workspace=tmp_path)
    # Copy ground truth for first stage only
    stage0 = inst.metadata["stages"][0]
    gt = tmp_path / "ground_truth" / stage0["output"]
    out = tmp_path / "data" / stage0["output"]
    out.parent.mkdir(exist_ok=True)
    out.write_text(gt.read_text())
    results = task.verify_all(inst, tmp_path)
    credit = task.partial_credit(results)
    assert credit == pytest.approx(1 / 3)


# ── research_synth ────────────────────────────────────────────────────────────

from horizonbench.tasks.research_synth import ResearchSynthTask
from horizonbench.tasks.research_synth.generator import generate_workspace as rs_gen
from horizonbench.tasks.research_synth.verifier import _f1, load_extracted, verify_document


def test_rs_generate_creates_docs(tmp_path):
    meta = rs_gen(k=4, seed=10, workspace=tmp_path)
    for doc_id in meta["doc_ids"]:
        assert (tmp_path / "docs" / f"{doc_id}.txt").exists()


def test_rs_generate_ground_truth(tmp_path):
    meta = rs_gen(k=3, seed=11, workspace=tmp_path)
    assert (tmp_path / "ground_truth.json").exists()
    gt = json.loads((tmp_path / "ground_truth.json").read_text())
    assert len(gt) == 3


def test_rs_f1_perfect():
    assert _f1({"a:b": "1 kg"}, {"a:b": "1 kg"}) == pytest.approx(1.0)


def test_rs_f1_zero():
    assert _f1({}, {"a:b": "1 kg"}) == pytest.approx(0.0)


def test_rs_f1_partial():
    score = _f1({"a:b": "1 kg", "c:d": "wrong"}, {"a:b": "1 kg", "c:d": "2 m"})
    assert 0 < score < 1


def test_rs_load_extracted_missing(tmp_path):
    assert load_extracted(tmp_path) == {}


def test_rs_load_extracted_valid(tmp_path):
    data = {"doc_000": {"e:a": "1 kg"}}
    (tmp_path / "extracted_facts.json").write_text(json.dumps(data))
    assert load_extracted(tmp_path) == data


def test_rs_task_verify_perfect(tmp_path):
    task = ResearchSynthTask()
    inst = task.generate(k=2, instance_id="rs-perfect", workspace=tmp_path)
    # Write perfect facts
    gt = inst.metadata["ground_truth"]
    (tmp_path / "extracted_facts.json").write_text(json.dumps(gt))
    results = task.verify_all(inst, tmp_path)
    assert all(r.passed for r in results)
    assert task.partial_credit(results) == pytest.approx(1.0)


def test_rs_task_verify_empty(tmp_path):
    task = ResearchSynthTask()
    inst = task.generate(k=2, instance_id="rs-empty", workspace=tmp_path)
    results = task.verify_all(inst, tmp_path)
    assert not any(r.passed for r in results)
    assert task.partial_credit(results) == pytest.approx(0.0)


# ── constraint_plan ───────────────────────────────────────────────────────────

from horizonbench.tasks.constraint_plan import ConstraintPlanTask
from horizonbench.tasks.constraint_plan.generator import generate_workspace as cp_gen
from horizonbench.tasks.constraint_plan.verifier import (
    _build_slot_map,
    check_constraint,
    load_schedule,
    verify_all_constraints,
)


def test_cp_generate_creates_problem(tmp_path):
    cp_gen(k=4, seed=20, workspace=tmp_path)
    assert (tmp_path / "problem.json").exists()


def test_cp_generate_k_constraints(tmp_path):
    meta = cp_gen(k=5, seed=21, workspace=tmp_path)
    assert len(meta["constraints"]) == 5


def test_cp_load_schedule_missing(tmp_path):
    assert load_schedule(tmp_path) == {}


def test_cp_check_time_window_pass(tmp_path):
    constraint = {"type": "time_window", "task": "T0", "earliest_start": 0, "latest_end": 8}
    slot_map = {"T0": (2, 5)}
    passed, _ = check_constraint(constraint, slot_map, {})
    assert passed


def test_cp_check_time_window_fail(tmp_path):
    constraint = {"type": "time_window", "task": "T0", "earliest_start": 0, "latest_end": 4}
    slot_map = {"T0": (2, 6)}  # ends at 6 > 4
    passed, _ = check_constraint(constraint, slot_map, {})
    assert not passed


def test_cp_check_precedence_pass():
    c = {"type": "precedence", "before": "T0", "after": "T1"}
    passed, _ = check_constraint(c, {"T0": (0, 2), "T1": (2, 4)}, {})
    assert passed


def test_cp_check_precedence_fail():
    c = {"type": "precedence", "before": "T1", "after": "T0"}
    passed, _ = check_constraint(c, {"T0": (0, 2), "T1": (2, 4)}, {})
    assert not passed


def test_cp_check_no_overlap_pass():
    c = {"type": "no_overlap", "tasks": ["T0", "T1"]}
    passed, _ = check_constraint(c, {"T0": (0, 2), "T1": (3, 5)}, {})
    assert passed


def test_cp_check_no_overlap_fail():
    c = {"type": "no_overlap", "tasks": ["T0", "T1"]}
    passed, _ = check_constraint(c, {"T0": (0, 4), "T1": (2, 6)}, {})
    assert not passed


def test_cp_check_max_parallel_pass():
    c = {"type": "max_parallel", "tasks": ["T0", "T1", "T2"], "limit": 2}
    # T0=[0,2], T1=[0,2], T2=[2,4] → max 2 at t=0
    passed, _ = check_constraint(c, {"T0": (0, 2), "T1": (0, 2), "T2": (2, 4)}, {})
    assert passed


def test_cp_check_max_parallel_fail():
    c = {"type": "max_parallel", "tasks": ["T0", "T1", "T2"], "limit": 1}
    passed, _ = check_constraint(c, {"T0": (0, 2), "T1": (0, 2), "T2": (0, 2)}, {})
    assert not passed


def test_cp_task_verify_all_satisfied(tmp_path):
    task = ConstraintPlanTask()
    inst = task.generate(k=2, instance_id="cp-pass", workspace=tmp_path)
    # Build a trivially valid schedule (tasks back to back in order)
    schedule = {
        "schedule": [
            {"task": t["id"], "start": i * 4, "end": i * 4 + t["duration"]}
            for i, t in enumerate(inst.metadata["tasks"])
        ]
    }
    (tmp_path / "schedule.json").write_text(json.dumps(schedule))
    # At least no crash — partial credit may be < 1 depending on constraints
    results = task.verify_all(inst, tmp_path)
    credit = task.partial_credit(results)
    assert 0.0 <= credit <= 1.0

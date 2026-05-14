"""Verifier for research_synth task.

Checks workspace/extracted_facts.json against workspace/ground_truth.json.

Per-document scoring:
  - precision = correctly extracted / total extracted
  - recall    = correctly extracted / total ground truth
  - f1        = harmonic mean

A checkpoint passes when per-document F1 == 1.0 (exact extraction).
Partial credit = mean F1 across all documents.
"""

from __future__ import annotations

import json
from pathlib import Path


def _f1(predicted: dict[str, str], ground_truth: dict[str, str]) -> float:
    if not ground_truth:
        return 1.0
    if not predicted:
        return 0.0
    correct = sum(1 for k, v in predicted.items() if ground_truth.get(k) == v)
    precision = correct / len(predicted)
    recall = correct / len(ground_truth)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def verify_document(
    doc_id: str,
    ground_truth: dict[str, str],
    extracted: dict[str, dict[str, str]],
) -> tuple[float, str]:
    """Return (f1_score, detail) for a single document."""
    predicted = extracted.get(doc_id, {})
    score = _f1(predicted, ground_truth)
    correct = sum(1 for k, v in predicted.items() if ground_truth.get(k) == v)
    detail = f"F1={score:.2f} ({correct}/{len(ground_truth)} facts)"
    return score, detail


def load_extracted(workspace: Path) -> dict[str, dict[str, str]]:
    """Load the agent-produced extracted_facts.json, or return empty dict."""
    path = workspace / "extracted_facts.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, TypeError):
        return {}

"""Generator for research_synth task.

Produces k plain-text documents each containing embedded key-value facts.
Ground truth stored at workspace/ground_truth.json (hidden from agent).
Agent must produce workspace/extracted_facts.json in the same format.

Fact format in prose:
  "The {entity} {attribute} is {value}."

Ground truth JSON:
  { "doc_000": {"entity:attribute": "value", ...}, ... }
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

ENTITIES = [
    "reactor", "satellite", "compound", "algorithm", "protocol",
    "observatory", "accelerator", "telescope", "genome", "circuit",
]
ATTRIBUTES = [
    "mass", "temperature", "frequency", "velocity", "density",
    "capacity", "latency", "pressure", "wavelength", "efficiency",
]
ADJECTIVES = ["primary", "secondary", "nominal", "maximum", "average"]
UNITS = ["kg", "K", "Hz", "m/s", "g/cm3", "TB", "ms", "Pa", "nm", "%"]

_TEMPLATES = [
    "In study {study_id}, researchers found that the {entity} {attr} is {value} {unit}.",
    "According to report {study_id}, the {entity} operates with a {attr} of {value} {unit}.",
    "Measurements from {study_id} indicate the {entity} {attr} reaches {value} {unit}.",
    "The {entity} described in {study_id} has a {attr} of {value} {unit}.",
]


def _make_doc(rng: random.Random, doc_idx: int, n_facts: int) -> tuple[str, dict[str, str]]:
    """Return (document_text, facts_dict)."""
    entity = rng.choice(ENTITIES)
    study_id = f"REP-{rng.randint(1000, 9999)}"
    facts: dict[str, str] = {}
    paragraphs = [f"Document {doc_idx:03d} — Research Summary ({study_id})\n"]

    for _ in range(n_facts):
        attr = rng.choice(ATTRIBUTES)
        value = str(rng.randint(1, 999))
        unit = rng.choice(UNITS)
        template = rng.choice(_TEMPLATES)
        sentence = template.format(
            study_id=study_id, entity=entity, attr=attr, value=value, unit=unit
        )
        paragraphs.append(sentence)
        key = f"{entity}:{attr}"
        facts[key] = f"{value} {unit}"

    return "\n".join(paragraphs) + "\n", facts


def generate_workspace(k: int, seed: int, workspace: Path) -> dict[str, Any]:
    rng = random.Random(seed)
    docs_dir = workspace / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    ground_truth: dict[str, dict[str, str]] = {}
    doc_ids = []

    for i in range(k):
        n_facts = rng.randint(2, 4)
        doc_id = f"doc_{i:03d}"
        text, facts = _make_doc(rng, i, n_facts)
        (docs_dir / f"{doc_id}.txt").write_text(text)
        ground_truth[doc_id] = facts
        doc_ids.append(doc_id)

    (workspace / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))

    return {"doc_ids": doc_ids, "k": k, "seed": seed, "ground_truth": ground_truth}

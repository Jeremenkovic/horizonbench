"""Task set generator — produces the frozen v1.0 task instances on disk.

Usage:
    uv run horizonbench generate --release v1.0 --out-dir tasks/
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import track

from horizonbench.tasks.constraint_plan import ConstraintPlanTask
from horizonbench.tasks.data_pipeline import DataPipelineTask
from horizonbench.tasks.multi_refactor import MultiRefactorTask
from horizonbench.tasks.research_synth import ResearchSynthTask

console = Console()

# Canonical step parameters from the spec
K_VALUES = [5, 10, 20, 30, 50, 75, 100]
INSTANCES_PER_K = 20

TASK_FAMILIES = {
    "multi-refactor": MultiRefactorTask,
    "data-pipeline": DataPipelineTask,
    "research-synth": ResearchSynthTask,
    "constraint-plan": ConstraintPlanTask,
}

generate_app = typer.Typer()


@generate_app.command()
def generate(
    release: Annotated[str, typer.Option("--release", "-r", help="Release tag, e.g. v1.0")] = "v1.0",
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o", help="Output directory")] = Path("tasks"),
    families: Annotated[str, typer.Option("--families", help="Comma-separated family IDs, or 'all'")] = "all",
    k_values: Annotated[str, typer.Option("--k-values", help="Comma-separated k values")] = "",
    n_instances: Annotated[int, typer.Option("--n-instances", help="Instances per k")] = INSTANCES_PER_K,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
):
    """Generate and freeze a task set for a release."""
    selected_families = (
        list(TASK_FAMILIES.keys())
        if families == "all"
        else [f.strip() for f in families.split(",")]
    )
    selected_k = (
        K_VALUES
        if not k_values
        else [int(k.strip()) for k in k_values.split(",")]
    )

    for fam in selected_families:
        if fam not in TASK_FAMILIES:
            console.print(f"[red]Unknown family: {fam}[/red]")
            raise typer.Exit(1)

    total = len(selected_families) * len(selected_k) * n_instances
    console.print(
        f"Generating {total} tasks "
        f"({len(selected_families)} families × {len(selected_k)} k-values × {n_instances} instances)"
    )

    manifest = {"release": release, "families": {}}

    for family_id in selected_families:
        task_cls = TASK_FAMILIES[family_id]
        task = task_cls()
        manifest["families"][family_id] = []

        for k in selected_k:
            for i in range(n_instances):
                instance_id = f"k{k:03d}-i{i:04d}"
                seed = task.seed_for(instance_id)
                record = {
                    "family": family_id,
                    "k": k,
                    "instance_id": instance_id,
                    "seed": seed,
                }
                manifest["families"][family_id].append(record)

                if not dry_run:
                    ws_dir = out_dir / release / family_id / f"k{k}" / instance_id
                    ws_dir.mkdir(parents=True, exist_ok=True)
                    task.generate(k=k, instance_id=instance_id, workspace=ws_dir)

    manifest_path = out_dir / release / "manifest.json"
    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    console.print(f"[green]Done.[/green] Manifest → {manifest_path}")
    if dry_run:
        console.print("[dim](dry run — no files written)[/dim]")

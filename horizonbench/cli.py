"""Typer CLI for HorizonBench."""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

import os

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from horizonbench.adapters.base import AdapterConfig, Message
from horizonbench.adapters.litellm import LiteLLMAdapter
from horizonbench.generate import generate_app
from horizonbench.metrics import gds, mop, rdc, rdc_auc, vaf
from horizonbench.site import build_site
from horizonbench.results.store import ResultStore
from horizonbench.results.trace import TraceWriter
from horizonbench.sandbox.runner import build_local_tool_handler
from horizonbench.tasks.base import RunResult, TaskInstance
from horizonbench.tasks.constraint_plan import ConstraintPlanTask
from horizonbench.tasks.data_pipeline import DataPipelineTask
from horizonbench.tasks.multi_refactor import MultiRefactorTask
from horizonbench.tasks.research_synth import ResearchSynthTask

app = typer.Typer(
    name="horizonbench",
    help="Long-horizon agent reliability benchmark.",
    add_completion=False,
    invoke_without_command=True,
)
app.add_typer(generate_app, name="generate", help="Generate frozen task sets.")

console = Console()

_BANNER = """\
 _   _            _                   ____                  _
| | | | ___  _ __(_)_______  _ __    | __ )  ___ _ __   ___| |__
| |_| |/ _ \\| '__| |_  / _ \\| '_ \\   |  _ \\ / _ \\ '_ \\ / __| '_ \\
|  _  | (_) | |  | |/ / (_) | | | |  | |_) |  __/ | | | (__| | | |
|_| |_|\\___/|_|  |_/___\\___/|_| |_|  |____/ \\___|_| |_|\\___|_| |_|"""


def _print_welcome() -> None:
    console.print()
    console.print(Text(_BANNER, style="bold cyan"))
    console.print(
        Text("  Long-horizon agent reliability benchmark  v0.1.0", style="dim"),
        justify="left",
    )
    console.print()

    # ── status checks ────────────────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    env_file = Path(__file__).parent.parent / ".env"
    db_path = Path("results/horizonbench.duckdb")

    rows = []
    rows.append(("horizonbench installed", True, ""))
    rows.append(("API key configured", bool(api_key), "run: horizonbench setup" if not api_key else ""))
    rows.append(("Results DB exists", db_path.exists(), "no runs yet" if not db_path.exists() else str(db_path)))

    if db_path.exists():
        try:
            from horizonbench.results.store import ResultStore
            with ResultStore(db_path) as store:
                runs = store.get_runs()
            n = len(runs)
            rows.append((f"Runs recorded", True, f"{n} total"))
        except Exception:
            pass

    status_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    status_table.add_column(width=3)
    status_table.add_column(width=32)
    status_table.add_column(style="dim")
    for label, ok, hint in rows:
        icon = Text("✓", style="bold green") if ok else Text("✗", style="bold red")
        status_table.add_row(icon, label, hint)

    console.print(Panel(status_table, title="[bold]Status[/bold]", border_style="cyan", padding=(0, 1)))

    # ── quick start ───────────────────────────────────────────────────────────
    qs = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=False)
    qs.add_column(style="bold green", no_wrap=True)
    qs.add_column(style="cyan", no_wrap=True)
    qs.add_column(style="dim", no_wrap=True)
    qs.add_row("$", "horizonbench setup", "→ configure API key interactively")
    qs.add_row("$", "horizonbench run --model claude-sonnet-4-6 --family multi-refactor --k 5", "→ run first benchmark")
    qs.add_row("$", "horizonbench list-families", "→ see all task families")
    qs.add_row("$", "horizonbench export --output summary.md", "→ export results for AI analysis")
    qs.add_row("$", "horizonbench site --output-dir site/dist", "→ build leaderboard site")
    console.print(Panel(qs, title="[bold]Quick start[/bold]", border_style="cyan", padding=(0, 1)))
    console.print()


@app.callback()
def _root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _print_welcome()
        raise typer.Exit(0)

TASK_REGISTRY: dict[str, type] = {
    "multi-refactor": MultiRefactorTask,
    "data-pipeline": DataPipelineTask,
    "research-synth": ResearchSynthTask,
    "constraint-plan": ConstraintPlanTask,
}


def _build_adapter(model: str, extra_json: str | None) -> LiteLLMAdapter:
    """Build a LiteLLMAdapter, merging any --extra JSON kwargs."""
    extra: dict[str, Any] = {}
    if extra_json:
        try:
            extra = json.loads(extra_json)
        except json.JSONDecodeError as exc:
            console.print(f"[red]--extra is not valid JSON: {exc}[/red]")
            raise typer.Exit(1) from exc
    return LiteLLMAdapter(AdapterConfig(model=model, temperature=0.0, extra=extra))


def _build_tool_handler(sandbox: str, workspace: Path):
    """Return the appropriate tool handler based on --sandbox mode."""
    if sandbox == "docker":
        try:
            from horizonbench.sandbox.runner import Sandbox, SandboxConfig
            # Start container for the lifetime of the run — caller must manage.
            # We return both the handler factory and the Sandbox object.
            return None, "docker"  # resolved per-run inside _run_single
        except Exception as exc:
            console.print(f"[yellow]Docker unavailable ({exc}), falling back to local[/yellow]")
    return build_local_tool_handler(workspace), "local"


def _run_single(
    task_cls: type,
    model: str,
    k: int,
    instance_id: str,
    workspace: Path,
    max_turns: int,
    trace_path: Path | None = None,
    extra_json: str | None = None,
    sandbox: str = "local",
) -> RunResult:
    """Execute one run and return a RunResult."""
    task = task_cls()
    instance: TaskInstance = task.generate(k=k, instance_id=instance_id, workspace=workspace)

    adapter = _build_adapter(model, extra_json)
    messages = [
        Message(role="system", content=task.system_prompt(instance)),
        Message(role="user", content=task.user_message(instance)),
    ]
    tools = task.tools(instance)

    # Resolve tool handler
    if sandbox == "docker":
        try:
            from horizonbench.sandbox.runner import Sandbox, SandboxConfig
            _sandbox_ctx = Sandbox(workspace, SandboxConfig(timeout_seconds=k * 15 + 60))
            _sandbox_ctx.__enter__()
            def _docker_handler(name: str, args: dict) -> str:
                if name == "read_file":
                    try:
                        return _sandbox_ctx.read_file(args.get("path", ""))
                    except FileNotFoundError as e:
                        return f"ERROR: {e}"
                elif name == "write_file":
                    _sandbox_ctx.write_file(args.get("path", ""), args.get("content", ""))
                    return "ok"
                elif name == "run_bash":
                    return _sandbox_ctx.run_bash(args.get("command", ""))
                else:
                    return build_local_tool_handler(workspace)(name, args)
            tool_handler = _docker_handler
        except Exception as exc:
            console.print(f"[yellow]Docker sandbox failed ({exc}), falling back to local[/yellow]")
            tool_handler = build_local_tool_handler(workspace)
            _sandbox_ctx = None
    else:
        tool_handler = build_local_tool_handler(workspace)
        _sandbox_ctx = None

    writer: TraceWriter | None = None
    if trace_path:
        writer = TraceWriter(trace_path)
        writer.run_start(instance.task_id, model, k)
        writer.message("system", messages[0].content)
        writer.message("user", messages[1].content)

    t0 = time.monotonic()
    try:
        final_messages, usage = adapter.run_tool_loop(
            messages, tools, tool_handler, max_turns=max_turns
        )
        wall = time.monotonic() - t0
    except Exception as exc:
        wall = time.monotonic() - t0
        if _sandbox_ctx:
            try:
                _sandbox_ctx.__exit__(None, None, None)
            except Exception:
                pass
        if writer:
            writer.run_end(False, 0.0, 0.0, 0, wall, error=str(exc))
            writer.close()
        return RunResult(
            task_id=instance.task_id,
            success=False,
            partial_credit=0.0,
            wall_seconds=wall,
            error=str(exc),
        )
    finally:
        if _sandbox_ctx:
            try:
                _sandbox_ctx.__exit__(None, None, None)
            except Exception:
                pass

    if writer:
        for m in final_messages[2:]:
            writer.message(m.role, m.content, m.tool_calls or None)

    cp_results = task.verify_all(instance, workspace)
    credit = task.partial_credit(cp_results)
    success = credit == 1.0

    if writer:
        writer.run_end(success, credit, usage.cost_usd, usage.total_tokens, wall)
        writer.close()

    return RunResult(
        task_id=instance.task_id,
        success=success,
        partial_credit=credit,
        checkpoint_results=cp_results,
        trace=[
            {"role": m.role, "content": m.content, "tool_calls": m.tool_calls}
            for m in final_messages
        ],
        cost_usd=usage.cost_usd,
        total_tokens=usage.total_tokens,
        wall_seconds=wall,
    )


@app.command()
def run(
    model: Annotated[str, typer.Option("--model", "-m", help="LiteLLM model string, e.g. claude-sonnet-4-6")],
    family: Annotated[str, typer.Option("--family", "-f", help="Task family ID")] = "multi-refactor",
    k: Annotated[int, typer.Option("--k", help="Step parameter")] = 5,
    n_runs: Annotated[int, typer.Option("--n-runs", "-n", help="Number of independent runs")] = 1,
    max_turns: Annotated[int, typer.Option("--max-turns", help="Max agent turns per run")] = 50,
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("results"),
    db_path: Annotated[Path, typer.Option("--db", help="DuckDB results file")] = Path("results/horizonbench.duckdb"),
    sandbox: Annotated[str, typer.Option("--sandbox", help="Execution sandbox: local or docker")] = "local",
    extra: Annotated[str | None, typer.Option("--extra", help='Extra adapter kwargs as JSON, e.g. \'{"api_base": "http://..."}\'', )] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
):
    """Run a HorizonBench evaluation."""
    if family not in TASK_REGISTRY:
        console.print(f"[red]Unknown task family: {family!r}[/red]")
        console.print(f"Available: {list(TASK_REGISTRY)}")
        raise typer.Exit(1)

    if sandbox not in ("local", "docker"):
        console.print(f"[red]--sandbox must be 'local' or 'docker', got {sandbox!r}[/red]")
        raise typer.Exit(1)

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    task_cls = TASK_REGISTRY[family]
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    sandbox_label = f"[sandbox={sandbox}]"
    console.print(f"[bold]Running[/bold] {n_runs}× {family} k={k} with {model} {sandbox_label}")

    with ResultStore(db_path) as store:
        for run_idx in range(n_runs):
            instance_id = f"run-{run_idx:04d}"
            trace_path = output_dir / f"{family}_k{k}_{instance_id}.jsonl"

            with tempfile.TemporaryDirectory(prefix=f"horizonbench_{family}_") as tmpdir:
                workspace = Path(tmpdir)
                console.print(f"  Run {run_idx + 1}/{n_runs}  instance={instance_id}", end=" ")
                result = _run_single(
                    task_cls, model, k, instance_id, workspace, max_turns,
                    trace_path=trace_path,
                    extra_json=extra,
                    sandbox=sandbox,
                )
                results.append(result)
                store.save_run(
                    result, model=model, family=family,
                    instance_id=instance_id, run_idx=run_idx,
                )

            icon = "✓" if result.success else "✗"
            console.print(
                f"{icon}  credit={result.partial_credit:.2f}  "
                f"${result.cost_usd:.4f}  {result.wall_seconds:.1f}s"
            )

    successes_by_k = {k: [int(r.success) for r in results]}
    credits_by_k = {k: [r.partial_credit for r in results]}
    rdc_curve = rdc(successes_by_k)
    gds_curve = gds(successes_by_k, credits_by_k)

    n_success = sum(1 for r in results if r.success)
    mean_credit = sum(r.partial_credit for r in results) / len(results)
    total_cost = sum(r.cost_usd for r in results)

    table = Table(title=f"Results — {family} k={k} ({model})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Runs", str(n_runs))
    table.add_row("Success rate", f"{n_success}/{n_runs} ({n_success/n_runs:.0%})")
    table.add_row("Mean partial credit", f"{mean_credit:.3f}")
    table.add_row("RDC(k)", f"{rdc_curve.get(k, 0):.3f}")
    gds_val = gds_curve.get(k)
    table.add_row("GDS(k)", f"{gds_val:.3f}" if gds_val is not None else "n/a (no failures)")
    table.add_row("Total cost", f"${total_cost:.4f}")
    console.print(table)

    summary = {
        "model": model,
        "family": family,
        "k": k,
        "n_runs": n_runs,
        "sandbox": sandbox,
        "success_rate": n_success / n_runs,
        "mean_partial_credit": mean_credit,
        "rdc": rdc_curve.get(k),
        "gds": gds_curve.get(k),
        "total_cost_usd": total_cost,
        "runs": [
            {
                "task_id": r.task_id,
                "success": r.success,
                "partial_credit": r.partial_credit,
                "cost_usd": r.cost_usd,
                "total_tokens": r.total_tokens,
                "wall_seconds": r.wall_seconds,
                "error": r.error,
            }
            for r in results
        ],
    }
    summary_path = output_dir / f"{family}_k{k}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    console.print(f"[dim]Summary → {summary_path}  |  DB → {db_path}[/dim]")


@app.command()
def metrics(
    model: Annotated[str, typer.Option("--model", "-m")],
    family: Annotated[str, typer.Option("--family", "-f")] = "multi-refactor",
    db_path: Annotated[Path, typer.Option("--db")] = Path("results/horizonbench.duckdb"),
):
    """Print RDC / VAF / GDS / MOP for a model+family from the results DB."""
    if not db_path.exists():
        console.print(f"[red]DB not found: {db_path}[/red]")
        raise typer.Exit(1)

    with ResultStore(db_path) as store:
        rates = store.success_rates(model, family)
        pcs = store.partial_credits(model, family)

    if not rates:
        console.print(f"[yellow]No data for model={model!r} family={family!r}[/yellow]")
        raise typer.Exit(0)

    successes_by_k = {k: [int(r >= 0.999) for r in vs] for k, vs in pcs.items()}
    rdc_curve = rdc(successes_by_k)
    vaf_curve = vaf(successes_by_k)
    gds_curve = gds(successes_by_k, pcs)
    mop_val = mop(rdc_curve)
    auc = rdc_auc(rdc_curve)

    table = Table(title=f"Metrics — {family} ({model})")
    table.add_column("k", style="cyan")
    table.add_column("RDC(k)", style="bold")
    table.add_column("VAF(k)")
    table.add_column("GDS(k)")
    for k_val in sorted(rdc_curve):
        vaf_val = vaf_curve.get(k_val)
        gds_val = gds_curve.get(k_val)
        table.add_row(
            str(k_val),
            f"{rdc_curve[k_val]:.3f}",
            f"{vaf_val:.2f}" if vaf_val is not None else "—",
            f"{gds_val:.3f}" if gds_val is not None else "—",
        )
    console.print(table)
    console.print(f"  RDC_AUC = {auc:.3f}")
    console.print(f"  MOP     = {mop_val if mop_val is not None else '>max_k'}")


@app.command("list-families")
def list_families():
    """List available task families."""
    for name in TASK_REGISTRY:
        console.print(f"  {name}")


@app.command("site")
def site_build(
    db_path: Annotated[Path, typer.Option("--db")] = Path("results/horizonbench.duckdb"),
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("site/dist"),
    traces_dir: Annotated[Path | None, typer.Option("--traces-dir")] = Path("results"),
):
    """Build the static leaderboard site from the results DB."""
    if not db_path.exists():
        console.print(f"[red]DB not found: {db_path}[/red]")
        raise typer.Exit(1)
    td = traces_dir if (traces_dir and traces_dir.exists()) else None
    n = build_site(db_path=db_path, output_dir=output_dir, traces_dir=td)
    console.print(f"[green]Built {n} pages → {output_dir}/[/green]")
    console.print(f"  Open: {output_dir}/index.html")


@app.command("export")
def export_summary(
    db_path: Annotated[Path, typer.Option("--db")] = Path("results/horizonbench.duckdb"),
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Write to file instead of stdout")] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="Output format: markdown or json")] = "markdown",
):
    """Export all results as a markdown or JSON summary ready to paste into any AI."""
    if not db_path.exists():
        console.print(f"[red]DB not found: {db_path}[/red]")
        raise typer.Exit(1)

    with ResultStore(db_path) as store:
        all_runs = store.get_runs()

    if not all_runs:
        console.print("[yellow]No results found in DB.[/yellow]")
        raise typer.Exit(0)

    pairs = sorted({(r["model"], r["family"]) for r in all_runs})

    if fmt == "json":
        out: dict = {"models": []}
        with ResultStore(db_path) as store:
            for model, family in pairs:
                pcs = store.partial_credits(model, family)
                if not pcs:
                    continue
                successes_by_k = {k: [int(v >= 0.999) for v in vs] for k, vs in pcs.items()}
                rdc_curve = rdc(successes_by_k)
                gds_curve = gds(successes_by_k, pcs)
                out["models"].append({
                    "model": model,
                    "family": family,
                    "rdc_auc": round(rdc_auc(rdc_curve), 4),
                    "mop": mop(rdc_curve),
                    "rdc_by_k": {str(k): round(v, 4) for k, v in sorted(rdc_curve.items())},
                    "avg_partial_credit_by_k": {
                        str(k): round(sum(vs) / len(vs), 4) for k, vs in sorted(pcs.items()) if vs
                    },
                    "n_runs": sum(len(vs) for vs in pcs.values()),
                })
        text = json.dumps(out, indent=2)
    else:
        lines: list[str] = [
            "# HorizonBench Results Summary",
            "",
            "Long-horizon agent reliability benchmark. "
            "RDC(k) = success rate at k steps. RDC_AUC = area under decay curve (higher = better). "
            "MOP = step count where success rate first drops below 50% (higher = better).",
            "",
        ]
        with ResultStore(db_path) as store:
            for model, family in pairs:
                pcs = store.partial_credits(model, family)
                if not pcs:
                    continue
                successes_by_k = {k: [int(v >= 0.999) for v in vs] for k, vs in pcs.items()}
                rdc_curve = rdc(successes_by_k)
                gds_curve = gds(successes_by_k, pcs)
                auc = rdc_auc(rdc_curve)
                mop_val = mop(rdc_curve)
                n_runs = sum(len(vs) for vs in pcs.values())

                lines += [
                    f"## {model} — {family}",
                    "",
                    f"- **RDC_AUC**: {auc:.3f}",
                    f"- **MOP**: {mop_val if mop_val is not None else '>max_k'}",
                    f"- **Total runs**: {n_runs}",
                    "",
                    "| k | RDC(k) | Avg partial credit |",
                    "|---|--------|--------------------|",
                ]
                for k_val in sorted(rdc_curve):
                    pc_list = pcs.get(k_val, [])
                    avg_pc = sum(pc_list) / len(pc_list) if pc_list else 0.0
                    lines.append(f"| {k_val} | {rdc_curve[k_val]:.3f} | {avg_pc:.3f} |")
                lines.append("")
        text = "\n".join(lines)

    if output:
        output.write_text(text)
        console.print(f"[green]Exported → {output}[/green]")
    else:
        print(text)


@app.command("setup")
def setup() -> None:
    """Interactive setup — configure API key and verify installation."""
    _print_welcome()

    console.print("[bold cyan]Setup wizard[/bold cyan]\n")

    env_file = Path(__file__).parent.parent / ".env"
    existing_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if existing_key:
        console.print(f"[green]✓[/green]  ANTHROPIC_API_KEY already set ({existing_key[:12]}...)\n")
        change = typer.confirm("  Change it?", default=False)
        if not change:
            console.print("[dim]  Keeping existing key.[/dim]")
            _finish_setup()
            return

    key = typer.prompt("  Paste your Anthropic API key (sk-ant-...)", hide_input=True)
    if not key.startswith("sk-"):
        console.print("[yellow]  Warning: key doesn't look like an Anthropic key (should start with sk-).[/yellow]")

    # Write to .env
    env_file.write_text(f"ANTHROPIC_API_KEY={key}\n")
    os.environ["ANTHROPIC_API_KEY"] = key
    console.print(f"[green]✓[/green]  Saved to {env_file}\n")

    _finish_setup()


def _finish_setup() -> None:
    console.print("[bold]Verifying installation...[/bold]\n")

    steps = [
        ("Import horizonbench.cli", lambda: __import__("horizonbench.cli")),
        ("Import horizonbench.tasks", lambda: __import__("horizonbench.tasks.base")),
        ("Import horizonbench.metrics", lambda: __import__("horizonbench.metrics")),
        ("Import horizonbench.results", lambda: __import__("horizonbench.results.store")),
    ]
    all_ok = True
    for label, fn in steps:
        try:
            fn()
            console.print(f"  [green]✓[/green]  {label}")
        except Exception as e:
            console.print(f"  [red]✗[/red]  {label} — {e}")
            all_ok = False

    console.print()
    if all_ok:
        console.print(Panel(
            "[bold green]HorizonBench is ready![/bold green]\n\n"
            "Run your first benchmark:\n"
            "[cyan]  horizonbench run --model claude-sonnet-4-6 --family multi-refactor --k 5[/cyan]",
            border_style="green",
            padding=(1, 2),
        ))
    else:
        console.print("[red]Some checks failed. Try reinstalling:[/red]  uv tool install . --force")


if __name__ == "__main__":
    app()

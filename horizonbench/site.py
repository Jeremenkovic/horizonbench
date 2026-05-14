"""Static site generator for HorizonBench leaderboard.

Reads from the DuckDB results store and JSONL trace files; outputs
self-contained HTML pages to the given output directory.

Pages produced
--------------
index.html                          — leaderboard (all models × families)
model_{model}_{family}.html         — per-model decay curve + per-k table
trace_{family}_{k}_{instance}.html  — failure trace viewer
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from horizonbench.metrics import gds, gds_overall, mop, rdc, rdc_auc, vaf
from horizonbench.results.store import ResultStore
from horizonbench.results.trace import read_trace

_TEMPLATES_DIR = Path(__file__).parent.parent / "site" / "templates"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["ts"] = lambda t: datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")
    env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
    return env


def _compute_model_family_metrics(
    store: ResultStore, model: str, family: str
) -> dict[str, Any]:
    """Compute all metrics for one (model, family) pair."""
    pcs = store.partial_credits(model, family)
    if not pcs:
        return {}

    # Use full-credit (1.0) as success threshold
    successes_by_k = {k: [int(v >= 0.999) for v in vs] for k, vs in pcs.items()}
    rdc_curve = rdc(successes_by_k)
    vaf_curve = vaf(successes_by_k)
    gds_curve = gds(successes_by_k, pcs)

    all_runs = store.get_runs(model=model, family=family)
    total_cost = sum(r["cost_usd"] for r in all_runs)
    n_runs = len(all_runs)

    return {
        "model": model,
        "family": family,
        "n_runs": n_runs,
        "rdc_auc": rdc_auc(rdc_curve),
        "mop": mop(rdc_curve),
        "gds_overall": gds_overall(gds_curve),
        "avg_cost": total_cost / n_runs if n_runs else 0.0,
        "rdc_curve": rdc_curve,
        "vaf_curve": vaf_curve,
        "gds_curve": gds_curve,
        "pcs": pcs,
        "successes_by_k": successes_by_k,
    }


def build_site(
    db_path: Path,
    output_dir: Path,
    traces_dir: Path | None = None,
) -> int:
    """Generate all site pages.  Returns number of pages written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = _jinja_env()
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    pages_written = 0

    with ResultStore(db_path) as store:
        all_runs = store.get_runs()
        if not all_runs:
            # Write empty index
            _write(
                env, "index.html",
                output_dir / "index.html",
                generated_at=generated_at, rows=[], families=[],
            )
            return 1

        # Collect unique (model, family) pairs
        pairs = sorted({(r["model"], r["family"]) for r in all_runs})
        families = sorted({r["family"] for r in all_runs})

        # Compute metrics for each pair
        metric_rows = []
        for model, family in pairs:
            m = _compute_model_family_metrics(store, model, family)
            if not m:
                continue
            metric_rows.append({
                **m,
                "gds": m.get("gds_overall"),   # template alias
                "model_slug": _slug(model),
                "family_slug": _slug(family),
            })

        # Sort leaderboard by RDC_AUC descending
        metric_rows.sort(key=lambda r: r["rdc_auc"], reverse=True)

        # ── index.html ─────────────────────────────────────────────────────────
        _write(
            env, "index.html",
            output_dir / "index.html",
            generated_at=generated_at,
            rows=metric_rows,
            families=families,
        )
        pages_written += 1

        # ── per-(model, family) pages ──────────────────────────────────────────
        for row in metric_rows:
            model = row["model"]
            family = row["family"]
            rdc_curve = row["rdc_curve"]
            vaf_curve = row["vaf_curve"]
            gds_curve = row["gds_curve"]
            pcs = row["pcs"]
            successes_by_k = row["successes_by_k"]

            k_rows = []
            for k_val in sorted(rdc_curve):
                ns = successes_by_k.get(k_val, [])
                pc_list = pcs.get(k_val, [])
                k_rows.append({
                    "k": k_val,
                    "n_runs": len(ns),
                    "rdc": rdc_curve[k_val],
                    "vaf": vaf_curve.get(k_val),
                    "gds": gds_curve.get(k_val),
                    "avg_credit": sum(pc_list) / len(pc_list) if pc_list else 0.0,
                    "avg_cost": sum(
                        r["cost_usd"]
                        for r in store.get_runs(model=model, family=family, k=k_val)
                    ) / len(ns) if ns else 0.0,
                })

            rdc_chart_json = json.dumps({
                "k": sorted(rdc_curve.keys()),
                "rdc": [rdc_curve[k] for k in sorted(rdc_curve.keys())],
            })

            # Collect trace links for this model+family
            trace_links = []
            if traces_dir and traces_dir.exists():
                pattern = f"{family}_k*_run-*.jsonl"
                for tf in sorted(traces_dir.glob(pattern)):
                    try:
                        events = read_trace(tf)
                        run_end = next(
                            (e for e in reversed(events) if e.event_type == "run_end"), None
                        )
                        run_start = next(
                            (e for e in events if e.event_type == "run_start"), None
                        )
                        if run_start:
                            trace_slug = _slug(tf.stem)
                            trace_links.append({
                                "task_id": run_start.data.get("task_id", tf.stem),
                                "success": run_end.data.get("success", False) if run_end else False,
                                "partial_credit": run_end.data.get("partial_credit", 0.0) if run_end else 0.0,
                                "slug": trace_slug,
                            })
                            # Write trace page
                            _write_trace_page(env, output_dir, tf, model, trace_slug, generated_at)
                            pages_written += 1
                    except Exception:
                        pass

            page_name = f"model_{row['model_slug']}_{row['family_slug']}.html"
            _write(
                env, "model.html",
                output_dir / page_name,
                generated_at=generated_at,
                model=model,
                family=family,
                n_runs=row["n_runs"],
                rdc_auc=row["rdc_auc"],
                mop=row["mop"],
                gds_overall=row["gds_overall"],
                k_rows=k_rows,
                rdc_chart_json=rdc_chart_json,
                traces=trace_links,
            )
            pages_written += 1

    return pages_written


def _write(env: Environment, template_name: str, out_path: Path, **ctx: Any) -> None:
    tpl = env.get_template(template_name)
    out_path.write_text(tpl.render(**ctx))


def _write_trace_page(
    env: Environment,
    output_dir: Path,
    trace_path: Path,
    model: str,
    slug: str,
    generated_at: str,
) -> None:
    try:
        events = read_trace(trace_path)
    except Exception:
        return

    run_start = next((e for e in events if e.event_type == "run_start"), None)
    run_end = next((e for e in reversed(events) if e.event_type == "run_end"), None)

    task_id = run_start.data.get("task_id", slug) if run_start else slug
    success = run_end.data.get("success", False) if run_end else False
    partial_credit = run_end.data.get("partial_credit", 0.0) if run_end else 0.0
    cost_usd = run_end.data.get("cost_usd", 0.0) if run_end else 0.0
    wall_seconds = run_end.data.get("wall_seconds", 0.0) if run_end else 0.0

    _write(
        env, "trace.html",
        output_dir / f"trace_{slug}.html",
        generated_at=generated_at,
        task_id=task_id,
        model=model,
        success=success,
        partial_credit=partial_credit,
        cost_usd=cost_usd,
        wall_seconds=wall_seconds,
        events=events,
    )

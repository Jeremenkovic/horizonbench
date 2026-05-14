# HorizonBench — Long-Horizon Agent Reliability Benchmark

**Project spec, v0.1**
**Owner:** Nemanja
**Status:** Pre-launch, ready to build

---

## 1. One-liner

A public benchmark that measures how agent success degrades as task length grows from 5 to 100+ steps, ranking the major frontier and open-weight models by *reliability* rather than peak capability — and exposing the gap between "smart on one shot" and "reliable for a 30-minute autonomous run."

---

## 2. Why this benchmark needs to exist

The dominant agent benchmarks (SWE-bench, WebArena, AgentBench, GAIA) all score on a single end-state success metric — did the agent finish or not. Recent reliability-science research (arXiv 2603.29231) showed that **capability rankings and reliability rankings substantially diverge**: the strongest pass@1 model is frequently *not* the most reliable over 50-step tasks, and there are multi-rank inversions between medium and long horizons.

In production, this matters more than peak capability. An architect picking a model for a multi-hour autonomous workflow needs to know:

- At what step count does the model's reliability collapse?
- How wide is the run-to-run variance, and does it grow with horizon?
- When the model fails, does it fail gracefully (partial progress preserved) or catastrophically (state corrupted, work lost)?

No public leaderboard answers these. HorizonBench will.

---

## 3. Goals and non-goals

**Goals (v1):**
- Quantify reliability decay vs. task length for at least 6 frontier and 4 open-weight models.
- Publish four formal reliability metrics per model: RDC, VAF, GDS, MOP (defined below).
- Run every model under a single fair harness with full cost and token logging.
- Release a living leaderboard, the eval set, the harness, and the failure traces under permissive license.
- Be cited as the *de facto* long-horizon reliability reference within 6 months.

**Non-goals (v1):**
- Not trying to replace SWE-bench, GAIA, or capability benchmarks — orthogonal axis.
- Not benchmarking custom scaffolding/multi-agent systems. v1 is single-agent, single-tool-loop. (v2 may add scaffold comparisons.)
- Not benchmarking training-time properties (fine-tuning, distillation). Inference-only.
- Not aiming for breadth across every modality. Text + tool use only in v1.

---

## 4. Core metrics — formal definitions

Let `T(k)` denote a task instance parameterized to require approximately `k` sequential steps. Let `S(k, i)` ∈ {0, 1} be the binary success of run `i` on a `k`-step task. Let `P(k, i)` ∈ [0, 1] be the partial credit on run `i`.

### 4.1 Reliability Decay Curve (RDC)

`RDC(k) = mean over i of S(k, i)`

Plotted as success rate vs. step count `k ∈ {5, 10, 20, 30, 50, 75, 100}`. Each model is run `n ≥ 10` times per `k` to control variance. The curve is the model's signature artifact.

**Single-number summary:** `RDC_AUC` — area under the curve from `k=5` to `k=100`, normalized to [0, 1]. Higher = more reliable across horizons.

### 4.2 Meltdown Onset Point (MOP)

The smallest `k` at which `RDC(k) < 0.5`. If no such `k` exists in the tested range, report `MOP > 100`. This is the headline "where does this model break" number.

### 4.3 Variance Amplification Factor (VAF)

`VAF(k) = std_i(S(k, i)) / std_i(S(5, i))`

How much does run-to-run outcome variance grow from short (5-step) tasks to a given horizon `k`? `VAF(50)` is the headline value — interpretable as "this model's outcomes are X times less predictable at 50 steps than at 5."

### 4.4 Graceful Degradation Score (GDS)

For each failed run, measure partial credit `P(k, i)`. Then:

`GDS(k) = mean over failed runs of P(k, i)`

A high GDS means failures preserve partial progress (good for human-handoff systems). A low GDS means failures are catastrophic (bad for autonomy). Headline value is `GDS_overall` averaged across all `k`.

---

## 5. Task design

### 5.1 Design principles

1. **Parameterizable length.** Each task family must scale cleanly to k ∈ {5, 10, 20, 30, 50, 75, 100} sequential steps without changing the underlying skill.
2. **Programmatic verification.** No LLM-as-judge for final correctness in v1 — every task has a deterministic grader plus a partial-credit rubric.
3. **Contamination resistance.** Tasks are procedurally generated from seed templates, with fresh random parameters per release. The held-out test set rotates quarterly.
4. **Tool-use grounded.** Tasks require real tool calls (file ops, code execution, web fetch from a sandboxed mirror) — not pure chain-of-thought.
5. **Partial-credit instrumented.** Every task defines a list of "checkpoints" the agent must hit in order. Partial credit = checkpoints reached / total.

### 5.2 Task families (v1 — pick 4 of these 6)

| ID | Family | Step parameter | Verifier |
|---|---|---|---|
| `multi-refactor` | Multi-file code refactor in a sandboxed repo | # of files touched | Test suite must pass; AST diff must match required transformation |
| `data-pipeline` | Build an ETL chain over synthetic CSVs | # of transformation stages | Output CSV exact match against ground truth |
| `research-synth` | Synthesize a structured answer from N sandboxed documents | # of source docs | Extracted facts must match key-value ground truth (F1) |
| `constraint-plan` | Generate a plan satisfying N constraints (scheduling, packing) | # of constraints | Constraint solver verifies plan |
| `ticket-resolution` | Resolve a synthetic support ticket through a mock CRM API | # of sub-steps required | Final CRM state must match expected schema |
| `long-doc-edit` | Apply N structured edits to a long document | # of edits | Diff exact-match per edit |

**Recommended v1 set:** `multi-refactor`, `data-pipeline`, `research-synth`, `constraint-plan`. These cover four distinct cognitive profiles (code, data, retrieval, search) and have well-defined verifiers.

### 5.3 Per-family scale

- 7 step parameters × 20 task instances per parameter × 4 families = 560 tasks per release.
- 10 runs per task per model for variance estimation = 5,600 runs per model per release.
- At ~$0.50 average cost per run, that's ~$2,800 per model per release. Budget ~$30k/release across 10 models. Build a sponsor or apply for compute grants for this.

---

## 6. Models to evaluate (v1 target list)

**Frontier (closed):**
- Anthropic: Claude Opus 4.6, Sonnet 4.6, Haiku 4.5
- OpenAI: latest GPT-5.x flagship + mid-tier
- Google: latest Gemini 3.x Pro + Flash
- xAI: latest Grok 4.x

**Open weights:**
- Meta Llama 4.x flagship
- Qwen 3.x flagship
- DeepSeek latest
- Mistral latest

Use the published API endpoints for closed models. For open-weights, run on Together/Fireworks/Groq to remove infra-skill bias from results. **Always report inference provider on the leaderboard** — same model on different providers can differ measurably.

---

## 7. Eval harness architecture

```
┌─────────────────────────────────────────────────┐
│              CLI orchestrator                    │
│        (horizonbench run --model X --family Y)   │
└─────────────────┬───────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌───────▼─────────┐
│  Task loader   │  │  Model adapter   │
│  (yaml specs)  │  │  (LiteLLM-based) │
└───────┬────────┘  └───────┬─────────┘
        │                   │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │   Sandbox runner   │
        │  (Docker per run)  │
        │  - filesystem      │
        │  - mock APIs       │
        │  - tool definitions│
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │     Verifier      │
        │  - exact match    │
        │  - test runner    │
        │  - constraint solver│
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │   Results store    │
        │  (DuckDB + JSONL)  │
        │  every run, every  │
        │  message, every    │
        │  token, every $    │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │   Metrics engine   │
        │  computes RDC,     │
        │  VAF, GDS, MOP     │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │   Static site gen  │
        │  (Astro/Next.js)   │
        │  leaderboard +     │
        │  per-model report  │
        └────────────────────┘
```

### 7.1 Sandbox contract

Every task runs in an ephemeral Docker container with:
- Read-only base filesystem with task inputs at `/task/`
- Writable scratch at `/work/`
- Mock tool servers (HTTP) on `localhost` for "web," "CRM," "search," "filesystem" — all deterministic, all logged.
- Wall-clock timeout = `k * 15s` (scales with horizon).
- Network egress disabled (only mocks reachable).

### 7.2 Tool interface

Use the OpenAI/Anthropic-compatible function-calling schema, normalized via LiteLLM. Tools defined per family in `tools.yaml`. Same tool schemas given to every model — no per-model prompt tuning beyond the system prompt template.

### 7.3 Cost & token logging

Every run produces a `trace.jsonl` with: every message, every tool call, every token count, every wall-clock timestamp, every dollar (computed from the public price sheet of the inference provider at run time). The leaderboard surfaces **cost-normalized reliability** as a secondary axis.

---

## 8. Methodology rules

These are the project's *credibility moat*. Write them down, enforce them in code, defend them publicly.

1. **No prompt tuning per model.** One system prompt template per task family, same for all models. Publish the template.
2. **No re-runs on failure.** First N runs are first N runs. No cherry-picking.
3. **Fixed seeds for task generation per release.** Same tasks for every model. Seeds published.
4. **Quarterly test-set rotation.** Tasks regenerated with new seeds. Old set published as historical / contaminated.
5. **Failure traces are public.** Every failure has a publicly-linkable trace. This is what hiring managers actually click through.
6. **Inference provider is part of the model identity.** "Llama-4-405B (Together)" and "Llama-4-405B (Fireworks)" are separate leaderboard rows.
7. **Vendor pre-publication review window.** Each vendor gets 5 business days to flag harness bugs before publication, but cannot block publication or request score changes.
8. **Reproducibility check.** Anyone with $X and the public repo can reproduce any leaderboard row to within stated variance bounds.

---

## 9. Public artifacts

1. **Leaderboard site** — `horizonbench.ai` (or chosen name). Live decay curves. Filter by task family, step count, cost ceiling.
2. **Methodology paper / long-form blog post** — published with v1 launch. ~6,000 words. Frame as a *reliability science* contribution, not just a benchmark.
3. **GitHub repo** — Apache 2.0, harness + task generators + verifiers. README leads with "run your own model in 15 minutes."
4. **HuggingFace dataset** — current and historical task sets.
5. **Failure trace explorer** — clickable per-task per-model traces. *This is the secret weapon* — it makes the benchmark useful to engineers, not just a number.
6. **Quarterly release notes** — what changed, what the new top models did, notable inversions.

---

## 10. Repo structure

```
horizonbench/
├── README.md
├── SPEC.md                  ← this doc
├── LICENSE                  (Apache 2.0)
├── pyproject.toml
├── horizonbench/
│   ├── __init__.py
│   ├── cli.py               (Typer-based)
│   ├── adapters/
│   │   ├── base.py
│   │   ├── litellm.py       (unified API)
│   │   └── local.py         (vLLM/Ollama)
│   ├── tasks/
│   │   ├── base.py          (Task ABC, checkpoint protocol)
│   │   ├── multi_refactor/
│   │   ├── data_pipeline/
│   │   ├── research_synth/
│   │   └── constraint_plan/
│   ├── sandbox/
│   │   ├── runner.py        (Docker orchestration)
│   │   └── mock_servers/
│   ├── verifiers/
│   ├── metrics/
│   │   ├── rdc.py
│   │   ├── vaf.py
│   │   ├── gds.py
│   │   └── mop.py
│   └── results/
│       ├── store.py         (DuckDB)
│       └── trace.py
├── tasks/                   (generated task instances, by release)
│   └── v1.0/
├── results/                 (run outputs, gitignored except sample)
├── site/                    (Astro static site)
│   ├── src/
│   └── public/
└── tests/
```

---

## 11. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Standard for evals; ecosystem |
| CLI | Typer | Clean, typed, modern |
| Model API | LiteLLM | Single interface across vendors |
| Validation | Pydantic v2 | Typed tasks, typed traces |
| Sandbox | Docker + Python `docker` SDK | Strong isolation, reproducible |
| Mock servers | FastAPI | Trivial to spin up per-run |
| Results store | DuckDB + Parquet | Fast analytical queries on millions of runs |
| Metrics | NumPy / Polars | Standard |
| Plots | Plotly (interactive on site) + Matplotlib (paper figures) | Both |
| Site | Astro + Tailwind | Fast static site, easy to deploy |
| Hosting | Cloudflare Pages | Free, fast, supports analytics |
| Trace viewer | Custom React component or Inspect AI's viewer | TBD in milestone 3 |
| CI | GitHub Actions | Standard |

Consider building the eval harness on top of [**Inspect AI**](https://inspect.aisi.org.uk/) (UK AISI's open eval framework) instead of from scratch. Pros: free reliability, free trace viewer, free community. Cons: less differentiated as your own engineering artifact. Recommendation for v1: **fork Inspect AI's primitives but wrap them in HorizonBench's own task and metric layer** so the contribution is the methodology + tasks + metrics, not the runner.

---

## 12. Build plan — 4-week launch sprint

### Week 1 — skeleton + one task family
- Repo skeleton, CI, lint, type-check
- Task ABC, checkpoint protocol, partial-credit rubric
- `multi-refactor` family — generator, verifier, 5 task instances
- LiteLLM adapter for Claude + GPT
- Docker sandbox runner
- One end-to-end run for Claude Sonnet at k=10

### Week 2 — full task suite + metrics
- Implement remaining 3 task families
- Generate full v1.0 task set (560 tasks, fixed seeds)
- Implement RDC, VAF, GDS, MOP metric calculators
- DuckDB results store + replay tooling
- Smoke test: 1 model × 4 families × full step range, ~$200 in spend

### Week 3 — full model sweep + site
- Run all 10+ models on full task set
- Astro static site, leaderboard view, per-model decay-curve page
- Cost-normalized leaderboard
- Failure trace explorer (basic)

### Week 4 — launch
- Methodology blog post / paper
- 24h vendor pre-review window
- Cross-post: HN, Twitter, r/MachineLearning, AI eval Slack/Discords
- Email frontier-lab eval contacts (Anthropic, OpenAI, Google have public eval@ addresses)

---

## 13. Naming options

Pick one — first three are my top picks:

- **HorizonBench** — describes what it measures; clean, ownable
- **DecayBench** — pointed, slightly negative-coded but memorable
- **Marathon** — vivid metaphor; high recall; check trademark
- **LongRun** — clean but generic
- **Endurance** — descriptive; check for collisions
- **ReliaBench** — pure description; less memorable
- **MOP-Bench** — leans into the meltdown metric; insider-coded

Check domain availability and prior art on Google Scholar before committing. Reserve `.ai`, `.org`, and the GitHub org name on day 1.

---

## 14. Risks & open questions

1. **Cost.** Full sweep is $20–30k. Mitigation: launch with 2 task families and 6 models, expand on traction. Apply for compute credits from Anthropic, OpenAI, Google's research access programs.
2. **Vendors gaming the benchmark.** Mitigation: quarterly task rotation + procedurally-generated tasks. Make it hard to overfit.
3. **Inspect AI / METR / Vals AI overlap.** Mitigation: be clear that HorizonBench's contribution is the *reliability-decay framing and metrics*, not the harness. Cite related work generously.
4. **Step-count is a leaky proxy for "horizon."** Real autonomy is measured in wall-clock and decision-count, not step-count. Mitigation: report both step-count and median wall-clock in results.
5. **Partial credit is hard to define fairly across families.** Mitigation: every family's rubric is published; partial credit is a secondary metric, not the headline.
6. **Frontier vendors release new models monthly.** Mitigation: design for a "rolling release" — new models can be added to the current leaderboard between quarterly task rotations.

---

## 15. Definition of "v1 launched"

- [ ] `horizonbench/` Apache-2.0 repo with full harness running on at least 4 task families
- [ ] v1.0 task set frozen, seeded, published on HuggingFace
- [ ] At least 10 models scored on full task set
- [ ] Leaderboard live at chosen domain
- [ ] 6,000-word methodology post published
- [ ] Failure trace explorer live
- [ ] At least one frontier lab linked to the project publicly (tweet, repo cite, internal use)

---

## Appendix A — Prior art to read before building

- Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents (arXiv 2603.29231) — *the source of RDC/VAF/GDS/MOP*
- SWE-bench, SWE-bench Pro (Princeton) — *gold standard for verifier-grounded benchmarks*
- GAIA (Meta/HF) — *long-horizon retrieval-grounded benchmark*
- WebArena, VisualWebArena — *agent-task benchmark design*
- Beyond Accuracy: Multi-Dimensional Framework for Enterprise Agentic AI Systems (arXiv 2511.14136) — *cost/reliability/SLA metrics*
- Inspect AI documentation — *eval framework primitives*
- Vals AI's methodology pages — *how to run a credible public eval*

---

## Appendix B — First Claude Code prompt to run

```
Read SPEC.md in full. Then:

1. Scaffold the repo per section 10. Use uv for dependency management.
2. Implement horizonbench/tasks/base.py with the Task ABC, including:
   - checkpoint protocol
   - partial-credit rubric interface
   - step-parameter k
   - deterministic seed from (family, version, instance_id)
3. Implement horizonbench/tasks/multi_refactor/ with:
   - generator that produces a sandbox repo with k files needing a parameterized refactor
   - verifier that runs the included pytest suite + AST diff check
   - partial-credit rubric: fraction of files correctly refactored
4. Implement horizonbench/sandbox/runner.py using the docker SDK.
5. Implement horizonbench/adapters/litellm.py.
6. Implement horizonbench/cli.py with `horizonbench run --model X --family Y --k N --n-runs M`.
7. Add a smoke test that runs claude-sonnet-4-6 on multi-refactor with k=5, n-runs=2.

After each step, run tests and stop for me to review. Don't proceed past failing tests.
```

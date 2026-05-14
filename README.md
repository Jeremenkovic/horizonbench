# HorizonBench

**Long-horizon agent reliability benchmark.**  
Measures how success degrades as task length grows from 5 to 100+ steps, ranking frontier and open-weight models by *reliability* rather than peak capability.

---

## Why

Dominant benchmarks (SWE-bench, GAIA) score on a single end-state. HorizonBench asks: **at what step count does the model's reliability collapse?** It exposes the gap between "smart on one shot" and "reliable for a 30-minute autonomous run."

---

## Metrics

| Metric | Definition |
|--------|-----------|
| **RDC(k)** | Mean success rate at step count k |
| **RDC_AUC** | Area under the decay curve, normalised to [0,1] |
| **MOP** | Smallest k where RDC(k) < 0.5 ("meltdown onset point") |
| **VAF(k)** | std(S(k)) / std(S(5)) — variance amplification vs. short tasks |
| **GDS(k)** | Mean partial credit on *failed* runs — graceful degradation score |

---

## Task families (v1)

| ID | What the agent does | Step parameter |
|----|---------------------|----------------|
| `multi-refactor` | Rename snake_case to camelCase in k Python files | files |
| `data-pipeline` | Execute k CSV transformation stages in sequence | stages |
| `research-synth` | Extract facts from k sandboxed documents | documents |
| `constraint-plan` | Schedule tasks satisfying k constraints | constraints |

---

## Quick start

```bash
# Install
pip install uv
uv sync

# Run one model on one task family
export ANTHROPIC_API_KEY=sk-ant-...
uv run horizonbench run \
  --model claude-sonnet-4-6 \
  --family multi-refactor \
  --k 10 \
  --n-runs 5

# View metrics from the results DB
uv run horizonbench metrics \
  --model claude-sonnet-4-6 \
  --family multi-refactor

# Generate a frozen task set (560 tasks)
uv run horizonbench generate generate \
  --release v1.0 \
  --out-dir tasks/

# List available task families
uv run horizonbench list-families
```

---

## Repo structure

```
horizonbench/
  cli.py                   Typer CLI (run / metrics / list-families)
  generate.py              Task set generator (horizonbench generate)
  adapters/
    base.py                ModelAdapter ABC
    litellm.py             LiteLLM unified adapter (all cloud models)
    local.py               vLLM / Ollama adapters
  tasks/
    base.py                Task ABC, Checkpoint, RunResult
    multi_refactor/        Python code refactoring
    data_pipeline/         CSV ETL pipeline
    research_synth/        Document fact extraction
    constraint_plan/       Constraint satisfaction scheduling
  sandbox/
    runner.py              Docker SDK runner + local tool handler
    mock_servers/
      doc_server.py        FastAPI mock document server
  metrics/
    rdc.py                 Reliability Decay Curve
    vaf.py                 Variance Amplification Factor
    gds.py                 Graceful Degradation Score
    mop.py                 Meltdown Onset Point
  results/
    store.py               DuckDB results store
    trace.py               JSONL trace writer
tasks/v1.0/                Frozen task instances (generated)
results/                   Run outputs (gitignored)
tests/                     129 unit tests
```

---

## Methodology rules

1. No prompt tuning per model — one system prompt template per family.
2. No re-runs on failure — first N runs are the results.
3. Fixed seeds per release — same tasks for every model.
4. Quarterly test-set rotation — seeds published on rotation.
5. Failure traces are public — every failure has a linkable trace.
6. Inference provider is part of model identity.
7. Vendor pre-publication review window — 5 business days.
8. Reproducibility — anyone with the repo can reproduce any row.

---

## Adding a new model

Any model accessible via LiteLLM works out of the box:

```bash
# OpenAI
uv run horizonbench run --model gpt-4o --family multi-refactor --k 10

# Google
uv run horizonbench run --model gemini/gemini-2.0-flash --k 10

# Local vLLM
uv run horizonbench run --model openai/llama-3-70b --k 10
```

---

## License

Apache 2.0 — see LICENSE.

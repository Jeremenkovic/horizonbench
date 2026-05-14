# HorizonBench

**Long-horizon agent reliability benchmark.**  
Measures how AI agent success rate decays as task length grows from 5 to 100+ steps — ranking models by *reliability* rather than peak capability.

---

## Install

**Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) — install it first if you don't have it:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install HorizonBench as a global CLI tool:

```bash
uv tool install git+https://github.com/Jeremenkovic/horizonbench.git
```

Verify it works:

```bash
horizonbench
```

You should see a welcome screen with status checks and quick-start commands.

---

## Setup

Configure your API key interactively:

```bash
horizonbench setup
```

Or set it manually as an environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Anthropic
export OPENAI_API_KEY=sk-...          # OpenAI
export GOOGLE_API_KEY=...             # Google
```

---

## Usage

### Run a benchmark

```bash
horizonbench run --model claude-sonnet-4-6 --family multi-refactor --k 10 --n-runs 20
```

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Any LiteLLM model string | required |
| `--family` | Task family ID | `multi-refactor` |
| `--k` | Step count | `5` |
| `--n-runs` | Number of independent runs | `1` |
| `--sandbox` | `local` or `docker` | `local` |

### View metrics

```bash
horizonbench metrics --model claude-sonnet-4-6 --family multi-refactor
```

### Export results (paste into any AI for analysis)

```bash
horizonbench export                          # print to terminal
horizonbench export --output summary.md     # save as markdown
horizonbench export --format json           # machine-readable
```

### Build the leaderboard site

```bash
horizonbench site --output-dir site/dist
# then open site/dist/index.html
```

### List available task families

```bash
horizonbench list-families
```

---

## Task families

| ID | What the agent does | Step parameter |
|----|---------------------|----------------|
| `multi-refactor` | Rename snake_case → camelCase in k Python files | files |
| `data-pipeline` | Execute k CSV transformation stages in sequence | stages |
| `research-synth` | Extract facts from k sandboxed documents | documents |
| `constraint-plan` | Schedule tasks satisfying k constraints | constraints |

---

## Metrics

| Metric | Meaning |
|--------|---------|
| **RDC(k)** | Success rate at step count k |
| **RDC_AUC** | Area under the decay curve — overall reliability score (higher = better) |
| **MOP** | Step count where success first drops below 50% — meltdown onset point (higher = better) |
| **VAF(k)** | Variance amplification vs. short tasks |
| **GDS(k)** | Mean partial credit on failed runs — how gracefully the model degrades |

---

## Supported models

Any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers) works out of the box:

```bash
# Anthropic
horizonbench run --model claude-sonnet-4-6 --family multi-refactor --k 10

# OpenAI
horizonbench run --model gpt-4o --family multi-refactor --k 10

# Google
horizonbench run --model gemini/gemini-2.0-flash --family multi-refactor --k 10

# Local (Ollama)
horizonbench run --model ollama/llama3 --family multi-refactor --k 10
```

---

## Why

Most benchmarks (SWE-bench, MMLU, GAIA) measure a single end-state. They don't show how reliability *degrades* as tasks get longer. HorizonBench asks: **at what step count does the model start failing?**

This matters for anyone building autonomous agents — a model that scores 95% at k=5 might be at 20% by k=30.

---

## License

Apache 2.0 — see LICENSE.

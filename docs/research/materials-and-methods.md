# Materials and Methods

## Overview

We benchmark three categories of models against human decision-making data:
1. **Decision models** (Jev, Laya, OpenJev) — non-autoregressive classifiers that return structured probabilities
2. **Autoregressive LLMs** (llm-low, llm-high) — text-generation models prompted to return JSON
3. **Human data** — aggregated choice frequencies from published datasets

All models receive identical information about each decision problem. The ground truth is the observed human choice rate.

---

## Datasets

### choices13k

- **Source:** Peterson et al. (2019), https://github.com/jcpeterson/choices13k
- **Size:** 13,006 risky choice problems with 240,000+ human judgments
- **Format:** Each problem presents two gambles (A and B). Human choice rate (`bRate`) is the fraction of participants who chose Gamble B.
- **Gamble A:** Two-outcome lottery: outcome H_a with probability p_a, otherwise L_a
- **Gamble B:** Lottery with expected value H_b, probability p_b, otherwise L_b
- **Ambiguity:** Some problems hide probabilities for Gamble B (Amb = 1)

### CPC18

- **Source:** Erev et al. (2017), https://zenodo.org/records/2571510
- **Size:** 270 aggregated problems (694,500 individual trials)
- **Format:** Similar to choices13k — two gambles, human choice rate for Gamble B
- **Conditions:** Decisions under risk (known probabilities) and ambiguity (unknown probabilities)

---

## Prompt Structure

### Decision Models (Jev, Laya, OpenJev)

Decision models receive information in a **structured, split format**:

- **State** — the context/description of the decision problem
- **Question** — the instruction about what to decide
- **Criteria/Options** — the possible choices

For Jev (API format):
```json
{
  "state": "Gamble A: Outcome 26 with probability 0.95, otherwise -1\nGamble B: Lottery with expected value 23, probability 0.05, otherwise 21",
  "model": "jev-latest",
  "questions": {
    "choice": {
      "type": "choice",
      "instructions": "Which gamble should be chosen?",
      "criteria": {"A": null, "B": null}
    }
  }
}
```

For Laya (local model):
- State, question, and options are concatenated into a single sequence
- The model scores each option independently using option-marker scoring
- Returns a probability distribution over options

### Autoregressive LLMs (llm-low, llm-high)

LLMs receive the same information as a **single natural-language prompt**:

```
Given the following state:
Gamble A: Outcome 26 with probability 0.95, otherwise -1
Gamble B: Lottery with expected value 23, probability 0.05, otherwise 21

Question: Which gamble should be chosen?

Options:
- A
- B

Respond with JSON only: {"choice": "<option>", "probability": <0-1>}
```

The prompt is sent as the initial user message. The model responds with JSON containing:
- `choice` — the selected option (A or B)
- `probability` — confidence in that choice (0–1)

---

## Model Arms

| Arm | Type | Model | Access |
|-----|------|-------|--------|
| jev | Decision model | jev-latest | TypeSafe API |
| laya | Decision model | ModernBERT-large (421M) | Local (HuggingFace) |
| openjev | Decision model | DiffusionGemma 26B-A4B | Local (vLLM/MLX) |
| llm-low | LLM | Configurable (default: Gemini 2.0 Flash) | OpenRouter API |
| llm-high | LLM | Configurable (default: Claude Sonnet 4) | OpenRouter API |

---

## Evaluation Metrics

### Accuracy
Fraction of problems where the model's choice matches the human majority choice (bRate > 0.5 → B, else A).

### Brier Score
Mean squared error between predicted probability and observed human choice rate:
```
Brier = (1/N) Σ (p_i - bRate_i)²
```

### Log Loss
Cross-entropy of predicted probability against human choice rate:
```
LogLoss = -(1/N) Σ [bRate_i · log(p_i) + (1 - bRate_i) · log(1 - p_i)]
```

### Expected Calibration Error (ECE)
Weighted average of |predicted probability - observed frequency| across 10 probability bins:
```
ECE = Σ (|bin_i| / N) · |p̄_i - bRatē_i|
```

### Mean Confidence
Average confidence score reported by the model (when available).

---

## Harness Architecture

```
harness/
├── pyproject.toml          # Package config, entry point
├── configs/default.yaml    # Default experiment config
├── src/decision_bench/
│   ├── models/             # Model adapters (one per arm)
│   ├── datasets/           # Dataset loaders
│   ├── metrics.py          # Evaluation metrics
│   └── runner.py           # Experiment orchestration
└── tests/
```

### Execution Flow

1. Load dataset → list of `DecisionProblem` (state, question, options, human_choice_rate)
2. For each problem, call `model.predict(state, question, options)`
3. Record prediction incrementally to `runs/{arm}/{dataset}_predictions.jsonl`
4. Compute aggregate metrics → `runs/{arm}/{dataset}_summary.json`

### Configuration

Per-arm YAML files in `runs/`:
- `api_key: ""` → arm is skipped (not configured)
- `api_key: "<key>"` → arm runs

Command:
```bash
decision-bench --runs-dir runs --models jev llm-high
```

---

## Key Design Decisions

1. **Identical information** — All models see the same state description, question, and options. No model receives additional context.
2. **Structured vs. natural language** — Decision models use structured APIs (state/question/options split); LLMs use a single natural-language prompt. This reflects each model's native interface.
3. **Incremental writes** — Predictions are flushed to disk after each problem to prevent data loss on interruption.
4. **Raw output capture** — Full model responses are recorded for auditability and post-hoc analysis.
5. **Human choice rate as soft label** — We compare against the full distribution (bRate), not just the majority choice, enabling calibration analysis.

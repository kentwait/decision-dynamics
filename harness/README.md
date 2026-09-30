# decision-bench

Benchmark harness for comparing decision models (Jev, Laya, OpenJev) and LLMs (local, frontier) against human decision-making data.

## Setup

```bash
cd harness
uv pip install -e ".[laya,frontier,local]"
```

## Data

Download datasets to `datasets/`:

```bash
python datasets/download.py choices13k cpc18
```

## Run

```bash
decision-bench --config configs/default.yaml --output-dir results
```

Or with a subset:

```bash
decision-bench --config configs/default.yaml --limit 100
```

## Results

Results are written to `results/`:
- `{model}_{dataset}_summary.json` — aggregate metrics
- `{model}_{dataset}_predictions.jsonl` — per-problem predictions

## Metrics

- **Accuracy** — fraction matching human majority choice
- **Brier score** — mean squared error of predicted probability vs. human choice rate
- **Log loss** — cross-entropy of predicted probability vs. human choice rate
- **ECE** — expected calibration error (10 bins)
- **Mean confidence** — average model confidence

import argparse
import json
import os
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from decision_bench.datasets.choices13k import Choices13kDataset
from decision_bench.datasets.cpc18 import CPC18Dataset
from decision_bench.metrics import Metrics
from decision_bench.models.jev import JevModel
from decision_bench.models.laya import LayaModel
from decision_bench.models.llm_frontier import FrontierLLMModel
from decision_bench.models.llm_local import LocalLLMModel
from decision_bench.models.openjev import OpenJevModel


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(config: dict):
    model_type = config["type"]
    if model_type == "laya":
        return LayaModel(
            checkpoint=config.get("checkpoint", "convaiinnovations/laya-typed-decisions"),
            device=config.get("device", "cpu"),
        )
    elif model_type == "jev":
        return JevModel(
            api_key=config.get("api_key"),
            base_url=config.get("base_url", "https://api.typesafe.ai/v1"),
        )
    elif model_type == "openjev":
        return OpenJevModel(base_url=config.get("base_url", "http://localhost:8000"))
    elif model_type == "llm_local":
        return LocalLLMModel(
            model=config.get("model", "llama3.1:8b"),
            base_url=config.get("base_url", "http://localhost:11434"),
        )
    elif model_type == "llm_frontier":
        return FrontierLLMModel(
            model=config.get("model", "gpt-4o"),
            provider=config.get("provider", "openai"),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def build_dataset(config: dict):
    dataset_type = config["type"]
    if dataset_type == "choices13k":
        return Choices13kDataset(config["path"])
    elif dataset_type == "cpc18":
        return CPC18Dataset(config["path"])
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def run_experiment(model_config: dict, dataset_config: dict, output_dir: str, limit: int | None = None):
    model = build_model(model_config)
    dataset = build_dataset(dataset_config)
    problems = dataset.load()

    if limit:
        problems = problems[:limit]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = Metrics()
    predictions_file = output_path / f"{model.name()}_{dataset.name()}_predictions.jsonl"

    with open(predictions_file, "w") as f:
        for prob in tqdm(problems, desc=f"{model.name()} on {dataset.name()}"):
            pred = model.predict(prob.state, prob.question, prob.options)

            if pred.error:
                record = {
                    "problem_id": prob.problem_id,
                    "error": pred.error,
                    "latency_ms": pred.latency_ms,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                continue

            human_preferred = prob.options[0] if prob.human_choice_rate > 0.5 else prob.options[1]
            correct = pred.choice == human_preferred

            metrics.update(
                predicted_prob=pred.probability or 0.5,
                human_choice_rate=prob.human_choice_rate,
                confidence=pred.confidence or 0.5,
                correct=correct,
            )

            record = {
                "problem_id": prob.problem_id,
                "predicted_choice": pred.choice,
                "predicted_probability": pred.probability,
                "confidence": pred.confidence,
                "human_choice_rate": prob.human_choice_rate,
                "correct": correct,
                "latency_ms": pred.latency_ms,
                "raw_output": pred.raw_output,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

    summary = {
        "model": model.name(),
        "dataset": dataset.name(),
        "metrics": metrics.compute(),
        "config": {"model": model_config, "dataset": dataset_config},
    }

    with open(output_path / f"{model.name()}_{dataset.name()}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def discover_run_configs(runs_dir: str = "runs", model_filter: list[str] | None = None) -> list[dict]:
    configs = []
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return configs
    for yaml_file in sorted(runs_path.glob("*.yaml")):
        config = load_config(str(yaml_file))
        if model_filter and config.get("type") not in model_filter:
            print(f"Skipping {yaml_file.name}: not requested")
            continue
        if not config.get("api_key"):
            print(f"Skipping {yaml_file.name}: api_key is empty")
            continue
        config["_name"] = yaml_file.stem
        configs.append(config)
    return configs


def main():
    parser = argparse.ArgumentParser(description="Benchmark decision models vs LLMs vs humans")
    parser.add_argument("--config", help="Path to config YAML (legacy mode)")
    parser.add_argument("--runs-dir", default="runs", help="Directory with per-arm YAML configs")
    parser.add_argument("--output-dir", default="results", help="Output directory (legacy mode)")
    parser.add_argument("--limit", type=int, help="Limit number of problems")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model types to run (default: all). Choices: laya, jev, openjev, llm_local, llm_frontier",
    )
    args = parser.parse_args()

    dataset_configs = [
        {"type": "choices13k", "path": "datasets/c13k_selections.csv"},
        {"type": "cpc18", "path": "datasets/cpc18_aggregated.csv"},
    ]

    if args.config:
        config = load_config(args.config)
        models = config["models"]
        if args.models:
            models = [m for m in models if m["type"] in args.models]
        for model_config in models:
            for dataset_config in config["datasets"]:
                summary = run_experiment(model_config, dataset_config, args.output_dir, args.limit)
                print(f"\n{summary['model']} on {summary['dataset']}:")
                for k, v in summary["metrics"].items():
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        run_configs = discover_run_configs(args.runs_dir, model_filter=args.models)

        for model_config in run_configs:
            model_name = model_config["_name"]
            output_dir = str(Path(args.runs_dir) / model_name)
            for dataset_config in dataset_configs:
                summary = run_experiment(model_config, dataset_config, output_dir, args.limit)
                print(f"\n{summary['model']} on {summary['dataset']}:")
                for k, v in summary["metrics"].items():
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()

import csv
from collections import defaultdict
from pathlib import Path


def preprocess(input_path: str, output_path: str):
    trials = defaultdict(lambda: {"L": 0, "R": 0, "metadata": {}})

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_id = row["GameID"]
            choice = row["Button"]
            if choice in ("L", "R"):
                trials[game_id][choice] += 1
            trials[game_id]["metadata"] = {
                "Ha": row["Ha"], "pHa": row["pHa"], "La": row["La"],
                "Hb": row["Hb"], "pHb": row["pHb"], "Lb": row["Lb"],
                "LotShapeA": row.get("LotShapeA", ""),
                "LotNumA": row.get("LotNumA", ""),
                "LotShapeB": row.get("LotShapeB", ""),
                "LotNumB": row.get("LotNumB", ""),
                "Amb": row["Amb"], "Corr": row["Corr"],
                "Condition": row.get("Condition", ""),
                "Set": row.get("Set", ""),
            }

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "problem", "n", "bRate", "bRate_std",
            "Ha", "pHa", "La", "Hb", "pHb", "Lb",
            "LotShapeA", "LotNumA", "LotShapeB", "LotNumB", "Amb", "Corr", "Condition", "Set",
        ])
        writer.writeheader()
        for game_id, data in sorted(trials.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            total = data["L"] + data["R"]
            if total == 0:
                continue
            b_rate = data["R"] / total
            n = total
            b_rate_std = (b_rate * (1 - b_rate) / n) ** 0.5 if n > 1 else 0.0
            meta = data["metadata"]
            writer.writerow({
                "problem": game_id,
                "n": n,
                "bRate": b_rate,
                "bRate_std": b_rate_std,
                **meta,
            })


if __name__ == "__main__":
    preprocess("datasets/cpc18.csv", "datasets/cpc18_aggregated.csv")
    print("Wrote datasets/cpc18_aggregated.csv")

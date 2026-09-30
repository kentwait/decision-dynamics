import csv
from decision_bench.datasets.base import BaseDataset, DecisionProblem


class CPC18Dataset(BaseDataset):
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> list[DecisionProblem]:
        problems = []
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gamble_a = f"Outcome {row['Ha']} with probability {row['pHa']}, otherwise {row['La']}"
                gamble_b = f"Outcome {row['Hb']} with probability {row['pHb']}, otherwise {row['Lb']}"
                state = f"Gamble A: {gamble_a}\nGamble B: {gamble_b}"

                problems.append(DecisionProblem(
                    problem_id=row.get("problem", str(len(problems))),
                    state=state,
                    question="Which gamble should be chosen?",
                    options=["A", "B"],
                    human_choice_rate=float(row["bRate"]),
                    metadata={
                        "Ha": row["Ha"], "pHa": row["pHa"], "La": row["La"],
                        "Hb": row["Hb"], "pHb": row["pHb"], "Lb": row["Lb"],
                    },
                ))
        return problems

    def name(self) -> str:
        return "cpc18"

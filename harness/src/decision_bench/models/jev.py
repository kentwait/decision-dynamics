import os
import time
import requests
from decision_bench.models.base import BaseModel, ModelPrediction


class JevModel(BaseModel):
    def __init__(self, api_key: str | None = None, base_url: str = "https://api.typesafe.ai/v1"):
        self.api_key = api_key or os.environ.get("JEV_API_KEY", "")
        self.base_url = base_url

    def predict(self, state: str, question: str, options: list[str]) -> ModelPrediction:
        start = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/evaluate",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "state": state,
                    "questions": [{"type": "choice", "text": question, "options": options}],
                },
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()["results"][0]
            latency = (time.time() - start) * 1000

            return ModelPrediction(
                choice=result.get("choice"),
                probability=result.get("probabilities", {}).get(result.get("choice", ""), 0.5),
                confidence=result.get("confidence"),
                raw_output=str(result),
                latency_ms=latency,
            )
        except Exception as e:
            return ModelPrediction(error=str(e), latency_ms=(time.time() - start) * 1000)

    def name(self) -> str:
        return "jev"

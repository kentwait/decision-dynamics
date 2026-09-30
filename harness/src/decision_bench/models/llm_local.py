import json
import time
from decision_bench.models.base import BaseModel, ModelPrediction


class LocalLLMModel(BaseModel):
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def predict(self, state: str, question: str, options: list[str]) -> ModelPrediction:
        start = time.time()
        try:
            import ollama

            options_text = "\n".join(f"- {opt}" for opt in options)
            prompt = (
                f"Given the following state:\n{state}\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"Respond with JSON only: {{\"choice\": \"<option>\", \"probability\": <0-1>}}"
            )

            resp = ollama.generate(model=self.model, prompt=prompt, format="json")
            parsed = json.loads(resp["response"])
            latency = (time.time() - start) * 1000

            return ModelPrediction(
                choice=parsed.get("choice"),
                probability=float(parsed.get("probability", 0.5)),
                confidence=float(parsed.get("probability", 0.5)),
                raw_output=resp["response"],
                latency_ms=latency,
            )
        except Exception as e:
            return ModelPrediction(error=str(e), latency_ms=(time.time() - start) * 1000)

    def name(self) -> str:
        return f"local-llm:{self.model}"

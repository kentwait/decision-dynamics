import json
import os
import time
from decision_bench.models.base import BaseModel, ModelPrediction


class FrontierLLMModel(BaseModel):
    def __init__(self, model: str = "gpt-4o", provider: str = "openai"):
        self.model = model
        self.provider = provider

    def predict(self, state: str, question: str, options: list[str]) -> ModelPrediction:
        start = time.time()
        try:
            options_text = "\n".join(f"- {opt}" for opt in options)
            prompt = (
                f"Given the following state:\n{state}\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"Respond with JSON only: {{\"choice\": \"<option>\", \"probability\": <0-1>}}"
            )

            if self.provider == "openai":
                from openai import OpenAI

                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content
            elif self.provider == "anthropic":
                import anthropic

                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            parsed = json.loads(raw)
            latency = (time.time() - start) * 1000

            return ModelPrediction(
                choice=parsed.get("choice"),
                probability=float(parsed.get("probability", 0.5)),
                confidence=float(parsed.get("probability", 0.5)),
                raw_output=raw,
                latency_ms=latency,
            )
        except Exception as e:
            return ModelPrediction(error=str(e), latency_ms=(time.time() - start) * 1000)

    def name(self) -> str:
        return f"frontier-llm:{self.model}"

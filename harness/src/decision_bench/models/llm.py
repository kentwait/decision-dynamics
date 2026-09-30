import json
import os
import time
from decision_bench.models.base import BaseModel, ModelPrediction


class LLMModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_type: str = "chat_completions",
        api_key: str = "",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_type = api_type
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

    def _build_prompt(self, state: str, question: str, options: list[str]) -> str:
        options_text = "\n".join(f"- {opt}" for opt in options)
        return (
            f"Given the following state:\n{state}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Respond with JSON only: {{\"choice\": \"<option>\", \"probability\": <0-1>}}"
        )

    def _call_chat_completions(self, prompt: str) -> str:
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_responses(self, prompt: str) -> str:
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.base_url}/responses",
            headers=headers,
            json={
                "model": self.model_name,
                "input": prompt,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if "output" in data and isinstance(data["output"], list):
            for item in data["output"]:
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            return content["text"]
        return str(data)

    def _call_messages(self, prompt: str) -> str:
        import requests

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        resp = requests.post(
            f"{self.base_url}/messages",
            headers=headers,
            json={
                "model": self.model_name,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    def predict(self, state: str, question: str, options: list[str]) -> ModelPrediction:
        start = time.time()
        try:
            prompt = self._build_prompt(state, question, options)

            if self.api_type == "chat_completions":
                raw = self._call_chat_completions(prompt)
            elif self.api_type == "responses":
                raw = self._call_responses(prompt)
            elif self.api_type == "messages":
                raw = self._call_messages(prompt)
            else:
                raise ValueError(f"Unknown api_type: {self.api_type}")

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
        return f"llm:{self.model_name}"

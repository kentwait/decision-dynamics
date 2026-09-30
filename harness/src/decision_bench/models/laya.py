import time
from decision_bench.models.base import BaseModel, ModelPrediction


class LayaModel(BaseModel):
    def __init__(self, checkpoint: str = "convaiinnovations/laya-typed-decisions", device: str = "cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, state: str, question: str, options: list[str]) -> ModelPrediction:
        start = time.time()
        try:
            import torch

            texts = [f"{state}\n{question}\n{opt}" for opt in options]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            best_idx = int(probs[0].argmax()) if len(probs.shape) > 1 else int(probs.argmax())
            confidence = float(probs[0][best_idx]) if len(probs.shape) > 1 else float(probs[best_idx])
            latency = (time.time() - start) * 1000

            return ModelPrediction(
                choice=options[best_idx],
                probability=confidence,
                confidence=confidence,
                latency_ms=latency,
            )
        except Exception as e:
            return ModelPrediction(error=str(e), latency_ms=(time.time() - start) * 1000)

    def name(self) -> str:
        return "laya"

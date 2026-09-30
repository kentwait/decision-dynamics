from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelPrediction:
    choice: Optional[str] = None
    probability: Optional[float] = None
    confidence: Optional[float] = None
    raw_output: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class BaseModel(ABC):
    @abstractmethod
    def predict(self, state: str, question: str, options: list[str]) -> ModelPrediction:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

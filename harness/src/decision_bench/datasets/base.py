from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DecisionProblem:
    problem_id: str
    state: str
    question: str
    options: list[str]
    human_choice_rate: float
    metadata: dict = field(default_factory=dict)


class BaseDataset(ABC):
    @abstractmethod
    def load(self) -> list[DecisionProblem]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

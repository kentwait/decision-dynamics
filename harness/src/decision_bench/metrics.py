import numpy as np
from dataclasses import dataclass, field


@dataclass
class Metrics:
    n: int = 0
    correct: int = 0
    brier_sum: float = 0.0
    log_loss_sum: float = 0.0
    confidence_sum: float = 0.0
    calibration_bins: list[float] = field(default_factory=lambda: [0.0] * 10)
    calibration_counts: list[int] = field(default_factory=lambda: [0] * 10)

    def update(self, predicted_prob: float, human_choice_rate: float, confidence: float, correct: bool):
        self.n += 1
        self.correct += int(correct)
        self.brier_sum += (predicted_prob - human_choice_rate) ** 2
        eps = 1e-15
        p = np.clip(predicted_prob, eps, 1 - eps)
        self.log_loss_sum += -(human_choice_rate * np.log(p) + (1 - human_choice_rate) * np.log(1 - p))
        self.confidence_sum += confidence
        bin_idx = min(int(predicted_prob * 10), 9)
        self.calibration_bins[bin_idx] += abs(predicted_prob - human_choice_rate)
        self.calibration_counts[bin_idx] += 1

    def compute(self) -> dict:
        if self.n == 0:
            return {}
        ece = sum(
            (self.calibration_bins[i] / self.calibration_counts[i]) * (self.calibration_counts[i] / self.n)
            for i in range(10)
            if self.calibration_counts[i] > 0
        )
        return {
            "n": self.n,
            "accuracy": self.correct / self.n,
            "brier_score": self.brier_sum / self.n,
            "log_loss": self.log_loss_sum / self.n,
            "mean_confidence": self.confidence_sum / self.n,
            "ece": ece,
        }

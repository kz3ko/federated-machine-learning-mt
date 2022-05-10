from dataclasses import dataclass


@dataclass
class SingleTestMetrics:
    accuracy: float
    loss: float

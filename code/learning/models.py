from dataclasses import dataclass

from numpy import array, int64


@dataclass
class SingleTestMetrics:
    accuracy: float
    loss: float


@dataclass
class PredictionMetrics:
    labels: array
    predicted_labels: array
    max_label: int64
    predicted_max_label: int64

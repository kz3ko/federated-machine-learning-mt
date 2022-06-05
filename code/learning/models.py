from typing import Optional
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


@dataclass
class EarlyStoppingBestMetric:
    type: str
    value: Optional[float] = None
    iteration: Optional[int] = None

    def __post_init__(self):
        self.compare_method = self.__get_compare_method()

    def __get_compare_method(self):
        if self.type.endswith('loss'):
            return '__lt__'
        elif self.type.endswith('accuracy'):
            return '__gt__'

        raise ValueError(f'Unknown metric "{self.type}" provided to "{self.__class__.__name__}", could not deteremine '
                         f'compare function.')

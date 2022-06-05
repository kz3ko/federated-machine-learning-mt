from abc import ABC, abstractmethod
from logging import info

from config.config import EarlyStoppingConfig
from learning.models import EarlyStoppingBestMetric, SingleTestMetrics


class Algorithm(ABC):
    triggered: bool

    @abstractmethod
    def validate(self, metrics: SingleTestMetrics):
        pass


class EarlyStopping:

    def __init__(self, config: EarlyStoppingConfig):
        self.enabled = config.enabled
        self.metric_type = config.metric_type
        self.patience = config.patience
        self.iteration = 0
        self.triggered = False
        self.best_metric = EarlyStoppingBestMetric(self.metric_type)

    def validate(self, metrics: SingleTestMetrics):
        self.iteration += 1
        recent_value = metrics.__getattribute__(self.best_metric.type)
        best_metric_changed = recent_value.__getattribute__(self.best_metric.compare_method)(self.best_metric.value)
        if self.best_metric.value is None or best_metric_changed:
            self.best_metric.iteration = self.iteration
            self.best_metric.value = recent_value
        elif self.iteration - self.best_metric.iteration >= self.patience:
            self.triggered = True
            info(f'Early stopping triggered, stopping learning process after {self.iteration} iteration.')

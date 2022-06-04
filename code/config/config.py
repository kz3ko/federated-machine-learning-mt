from enum import Enum
from typing import Any
from dataclasses import dataclass, field

from utilities.utils import get_data_from_json


class ConfigPath(Enum):
    MAIN_CONFIG = './config/config.json'


@dataclass
class DataDistributionConfig:
    dataset_name: str
    test_data_ratio: float
    clients_number: int
    main_classes_per_client_number: int
    main_class_ownership_per_client_ratio: float


@dataclass
class LearningCycleConfig:
    iterations: int
    iterations_to_aggregate: int


@dataclass
class WeightsSendingConfig:
    send_only_changed_weights: bool
    minimum_weight_difference_to_send: float


@dataclass
class ClientLearningConfig:
    weights_sending: WeightsSendingConfig

    def __init__(self, client_learning_config_json: dict[Any]):
        self.weights_sending = WeightsSendingConfig(**client_learning_config_json['weights_sending'])


@dataclass
class EarlyStoppingConfig:
    enabled: bool
    patience: int
    metric_type: str
    available_metrics: list[str] = field(repr=False)

    def __post_init__(self):
        self.__validate_metric_used()

    def __validate_metric_used(self):
        if self.metric_type not in self.available_metrics:
            metrics_string = f'", "'.join(self.available_metrics)
            raise ValueError(f'"EarlyStopping" can be used only only with "{metrics_string}", but there was '
                             f'"{self.metric_type}" provided.')


@dataclass
class ServerLearningConfig:
    early_stopping: EarlyStoppingConfig

    def __init__(self, server_learning_config_json: dict[Any]):
        early_stopping_metrics = ['accuracy', 'loss']
        self.early_stopping = EarlyStoppingConfig(
            **server_learning_config_json['early_stopping'],
            available_metrics=early_stopping_metrics
        )


@dataclass
class LearningConfig:
    cycle: LearningCycleConfig = field(init=False)
    client: ClientLearningConfig = field(init=False)
    server: ServerLearningConfig = field(init=False)

    def __init__(self, learning_config_json: dict[Any]):
        self.cycle = LearningCycleConfig(**learning_config_json['cycle'])
        self.client = ClientLearningConfig(learning_config_json['client'])
        self.server = ServerLearningConfig(learning_config_json['server'])


@dataclass
class Config:
    data_distribution: DataDistributionConfig = field(init=False)
    learning: LearningConfig = field(init=False)

    def __post_init__(self):
        main_config = get_data_from_json(ConfigPath.MAIN_CONFIG.value)
        self.data_distribution = DataDistributionConfig(**main_config['data_distribution'])
        self.learning = LearningConfig(main_config['learning'])

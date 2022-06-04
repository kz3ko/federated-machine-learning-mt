from enum import Enum
from typing import Any
from dataclasses import dataclass, field

from utilities.utils import get_data_from_json


class ConfigPath(Enum):
    MAIN_CONFIG = './config/config.json'


@dataclass
class DatasetConfig:
    name: str
    test_data_ratio: float


@dataclass
class DataDistributionConfig:
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
class FederatedLearningConfig:
    data_distribution: DataDistributionConfig = field(init=False)
    cycle: LearningCycleConfig = field(init=False)
    client: ClientLearningConfig = field(init=False)
    server: ServerLearningConfig = field(init=False)

    def __init__(self, federated_learning_config_json: dict[Any]):
        self.data_distribution = DataDistributionConfig(**federated_learning_config_json['data_distribution'])
        self.cycle = LearningCycleConfig(**federated_learning_config_json['cycle'])
        self.client = ClientLearningConfig(federated_learning_config_json['client'])
        self.server = ServerLearningConfig(federated_learning_config_json['server'])


@dataclass
class TraditionalLearningConfig:
    epochs: int
    early_stopping: EarlyStoppingConfig

    def __init__(self, traditional_learning_config_json: dict[Any]):
        self.epochs = traditional_learning_config_json['epochs']
        early_stopping_metrics = ['accuracy', 'loss', 'val_accuracy', 'val_loss']
        self.early_stopping = EarlyStoppingConfig(
            **traditional_learning_config_json['early_stopping'],
            available_metrics=early_stopping_metrics
        )


@dataclass
class Config:
    dataset: DatasetConfig = field(init=False)
    federated_learning: FederatedLearningConfig = field(init=False)
    traditional_learning: TraditionalLearningConfig = field(init=False)

    def __post_init__(self):
        main_config = get_data_from_json(ConfigPath.MAIN_CONFIG.value)
        self.learning_type = self.__get_learning_type(main_config)
        self.dataset = DatasetConfig(**main_config['dataset'])
        self.federated_learning = FederatedLearningConfig(main_config['federated_learning'])
        self.traditional_learning = TraditionalLearningConfig(main_config['traditional_learning'])

    @staticmethod
    def __get_learning_type(main_config: dict[Any, Any]):
        available_learning_types = ['federated', 'traditional']
        learning_type = main_config['learning_type'].lower()
        if learning_type not in available_learning_types:
            available_learning_types_string = '", "'.join(available_learning_types)
            raise ValueError(f'Provided "{learning_type}" which is inccorrect, available learning types: '
                             f'"{available_learning_types_string}".')

        return learning_type

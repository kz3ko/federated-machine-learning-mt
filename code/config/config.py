from enum import Enum
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
class LearningConfig:
    iterations: int
    iterations_to_aggregate: int


@dataclass
class Config:
    data_distribution: DataDistributionConfig = field(init=False)
    learning: LearningConfig = field(init=False)

    def __post_init__(self):
        main_config = get_data_from_json(ConfigPath.MAIN_CONFIG.value)
        self.data_distribution = DataDistributionConfig(**main_config['data_distribution'])
        self.learning = LearningConfig(**main_config['learning'])


config = Config()

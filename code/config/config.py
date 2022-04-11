from dataclasses import dataclass, field

from utilities.utils import get_data_from_json

CONFIG_PATH = './config/config.json'


@dataclass
class DataDistributionConfig:
    dataset_name: str
    clients_number: int
    main_classes_per_client_number: int
    main_class_ownership_per_client_ratio: float


@dataclass
class Config:
    data_distribution: DataDistributionConfig = field(init=False)

    def __post_init__(self):
        __config = get_data_from_json(CONFIG_PATH)
        self.data_distribution = DataDistributionConfig(**__config['data_distribution'])


config = Config()

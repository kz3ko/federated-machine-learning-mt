from typing import ClassVar
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class GeneratedDataPath:
    timestamp: ClassVar[str] = field(default=None)
    root: str = field(init=False)
    models: str = field(init=False)
    plots: str = field(init=False)
    metrics: str = field(init=False)

    def __new__(cls, *args, **kwargs):
        cls.timestamp = datetime.now().strftime('%H_%M_%s__%d_%m_%Y') if not cls.timestamp else cls.timestamp
        return super(GeneratedDataPath, cls).__new__(cls, *args, **kwargs)

    def __post_init__(self):
        self.root = f'generated_data/{self.timestamp}'
        self.models = f'{self.root}/models'
        self.plots = f'{self.root}/plots'
        self.metrics = f'{self.root}/metrics'

    def get_models_path_for_timestamp(self, timestamp: str) -> str:
        return self.__get_generated_data_path_for_timestamp(self.models, timestamp)

    def get_plots_path_for_timestamp(self, timestamp: str) -> str:
        return self.__get_generated_data_path_for_timestamp(self.plots, timestamp)

    def get_logs_path_for_timestamp(self, timestamp: str) -> str:
        return self.__get_generated_data_path_for_timestamp(self.metrics, timestamp)

    @staticmethod
    def __get_generated_data_path_for_timestamp(data_path: str, timestamp) -> str:
        data_path_without_timestamp = data_path.rsplit('/', 1)[0]

        return f'{data_path_without_timestamp}/{timestamp}'


generated_data_path = GeneratedDataPath()

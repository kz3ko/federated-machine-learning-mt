from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class GeneratedDataPath:
    models: str = field(init=False)
    plots: str = field(init=False)

    def __post_init__(self):
        timestamp = datetime.now().strftime('%H%M_%d%m%Y')
        self.models = f'generated_data/models/{timestamp}'
        self.plots = f'generated_data/plots/{timestamp}'

    def get_models_path_for_timestamp(self, timestamp: str) -> str:
        return self.__get_generated_data_path_for_timestamp(self.models, timestamp)

    def get_plots_path_for_timestamp(self, timestamp: str) -> str:
        return self.__get_generated_data_path_for_timestamp(self.plots, timestamp)

    @staticmethod
    def __get_generated_data_path_for_timestamp(data_path: str, timestamp) -> str:
        data_path_without_timestamp = data_path.rsplit('/', 1)[0]

        return f'{data_path_without_timestamp}/{timestamp}'


generated_data_path = GeneratedDataPath()

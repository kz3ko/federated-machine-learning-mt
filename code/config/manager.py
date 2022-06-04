from typing import Type
from shutil import copy

from config.config import Config, ConfigPath
from generated_data.path import generated_data_path
from learning.neural_network import NeuralNetworkModel


class ConfigManager:

    def __init__(self, model_class: Type[NeuralNetworkModel]):
        self.config_path = ConfigPath.MAIN_CONFIG
        self.config = Config()
        self.model_class_name = model_class.__name__

    def save_used_config(self):
        target_path = generated_data_path.root
        copied_config_path = f'{target_path}/config_used.{self.model_class_name}.json'
        copy(self.config_path.value, copied_config_path)

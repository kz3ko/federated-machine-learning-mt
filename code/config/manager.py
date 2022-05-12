from shutil import copy

from config.config import Config, ConfigPath
from generated_data.path import generated_data_path


class ConfigManager:

    def __init__(self):
        self.config_path = ConfigPath.MAIN_CONFIG
        self.config = Config()

    def save_used_config(self):
        target_path = generated_data_path.root
        copied_config_path = f'{target_path}/config_used.json'
        copy(self.config_path.value, copied_config_path)

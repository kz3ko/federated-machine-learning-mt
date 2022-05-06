from logging import info
from typing import Union
from abc import ABC, abstractmethod

from tensorflow.keras.callbacks import History
from tensorflow.keras.models import load_model

from learning.neural_network import NeuralNetworkModel
from data_provider.dataset import CustomDataset, ClientDataset, TestDataset
from generated_data.path import generated_data_path


class LearningParticipant(ABC):

    id: Union[str, int]
    history: History

    def __init__(self, dataset: CustomDataset, model: NeuralNetworkModel):
        self.dataset = dataset
        self.model = model
        self.model_name = self._get_model_name_to_save()

    def train_model(self) -> NeuralNetworkModel:
        info(f'Training model for participant with id "{self.id}".')
        self.history = self.model.train()

        return self.model

    def save_model(self):
        saved_model_path = f'{generated_data_path.models}/{self.model_name}.h5'
        info(f'Saving model for participant with id "{self.id}" in path: "{saved_model_path}".')
        self.model.base_model.save(saved_model_path)

    def read_model(self, timestamp: str):
        models_path = generated_data_path.get_models_path_for_timestamp(timestamp)
        model_to_read_path = f'{models_path}/{self.model_name}.h5'
        info(f'Reading model for participant with id "{self.id}" from path: "{model_to_read_path}".')
        self.model.base_model = load_model(model_to_read_path)

    @abstractmethod
    def _get_model_name_to_save(self) -> str:
        pass


class Server(LearningParticipant):

    def __init__(self, dataset: TestDataset, model: NeuralNetworkModel):
        self.id = 'server'
        super().__init__(dataset, model)

    def _get_model_name_to_save(self) -> str:
        return f'{self.id}_model'


class Client(LearningParticipant):

    def __init__(self, client_id: int, dataset: ClientDataset, model: NeuralNetworkModel):
        self.id = client_id
        super().__init__(dataset, model)

    def _get_model_name_to_save(self) -> str:
        return f'client_{self.id}_model'

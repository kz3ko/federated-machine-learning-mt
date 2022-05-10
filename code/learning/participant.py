from __future__ import annotations

from logging import info
from typing import Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tensorflow.keras.callbacks import History
from numpy import array

from learning.neural_network import NeuralNetworkModel
from learning.models import SingleTestMetrics
from data_provider.dataset import CustomDataset, ClientDataset, TestDataset
from generated_data.path import generated_data_path


@dataclass
class Participants:
    server: Server
    clients: list[Client]


class LearningParticipant(ABC):

    id: Union[str, int]
    latest_learning_history: History

    def __init__(self, dataset: CustomDataset, model: NeuralNetworkModel):
        self.dataset = dataset
        self.model = model
        self.full_name = self._get_participant_full_name()

    def train_model(self):
        info(f'Training model for participant with id "{self.id}".')
        self.latest_learning_history = self.model.train(self.dataset)

        return self.latest_learning_history

    def test_model(self, dataset: CustomDataset) -> SingleTestMetrics:
        return self.model.test(dataset)

    def get_model_weights(self) -> list[array]:
        return self.model.get_weights()

    def set_model_weights(self, new_weights: list[array]):
        self.model.set_weights(new_weights)

    def save_model(self):
        target_path = f'{generated_data_path.models}/{self.full_name}.h5'
        info(f'Saving model for participant with id "{self.id}" in path: "{target_path}".')
        self.model.save(target_path)

    def read_model(self, timestamp: str):
        models_directory_path = generated_data_path.get_models_path_for_timestamp(timestamp)
        model_path = f'{models_directory_path}/{self.full_name}.h5'
        info(f'Reading model for participant with id "{self.id}" from path: "{model_path}".')
        self.model.load(model_path)

    @abstractmethod
    def _get_participant_full_name(self) -> str:
        pass


class Server(LearningParticipant):

    def __init__(self, dataset: TestDataset, model: NeuralNetworkModel):
        self.id = 'server'
        super().__init__(dataset, model)

    def _get_participant_full_name(self) -> str:
        return self.id


class Client(LearningParticipant):

    def __init__(self, client_id: int, dataset: ClientDataset, model: NeuralNetworkModel):
        self.id = client_id
        super().__init__(dataset, model)

    def _get_participant_full_name(self) -> str:
        return f'client_{self.id}'

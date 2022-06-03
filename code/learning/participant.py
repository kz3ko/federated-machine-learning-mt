from __future__ import annotations

from logging import info
from typing import Union, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from copy import deepcopy

from tensorflow.keras.callbacks import History
from numpy import array, product

from learning.neural_network import NeuralNetworkModel
from learning.models import SingleTestMetrics, PredictionMetrics
from data_provider.dataset import CustomDataset, ClientDataset, TestDataset
from generated_data.path import generated_data_path


@dataclass
class Participants:
    server: Server
    clients: list[Client]

    def __iter__(self) -> Iterator[LearningParticipant]:
        for participant in [self.server, *self.clients]:
            yield participant


class LearningParticipant(ABC):

    id: Union[str, int]
    latest_learning_history: History
    latest_predictions: PredictionMetrics
    dataset_used_for_predictions: CustomDataset

    def __init__(self, dataset: CustomDataset, model: NeuralNetworkModel):
        self.dataset = dataset
        self.model = model
        self.full_name = self._get_participant_full_name()

    def train_model(self) -> History:
        info(f'Training model for participant with id "{self.id}".')
        self.latest_learning_history = self.model.train(self.dataset)

        return self.latest_learning_history

    def test_model(self, dataset: CustomDataset) -> SingleTestMetrics:
        return self.model.test(dataset)

    def make_predictions(self, dataset: CustomDataset) -> PredictionMetrics:
        self.dataset_used_for_predictions = dataset
        self.latest_predictions = self.model.make_predictions(self.dataset_used_for_predictions)

        return self.latest_predictions

    def get_all_model_weights(self) -> list[array]:
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

    def __init__(self, client_id: int, dataset: ClientDataset, model: NeuralNetworkModel,
                 minimum_weight_difference_to_send: float):
        self.id = client_id
        self.current_model_weights = None
        self.previous_model_weights = None
        self.minimum_weight_difference_to_send = minimum_weight_difference_to_send
        super().__init__(dataset, model)

    def train_model(self) -> History:
        self.previous_model_weights = self.current_model_weights
        super().train_model()
        self.current_model_weights = self.get_all_model_weights()

        return self.latest_learning_history

    def get_changed_model_weights(self) -> list[array]:
        if not self.minimum_weight_difference_to_send:
            return self.current_model_weights

        if not (self.previous_model_weights or self.current_model_weights):
            return self.get_all_model_weights()
        elif not self.previous_model_weights and self.current_model_weights:
            return self.current_model_weights
        else:
            return self.__count_changed_model_weights()

    def _get_participant_full_name(self) -> str:
        return f'client_{self.id}'

    def __count_changed_model_weights(self) -> list[array]:
        changed_model_weights = deepcopy(self.current_model_weights)
        layers_weights_iterator = enumerate(zip(self.previous_model_weights, self.current_model_weights))
        for layer_idx, (previous_layer_weights, current_layer_weights) in layers_weights_iterator:
            layer_shape = current_layer_weights.shape
            flatten_shape = (product(layer_shape, ))
            flattened_previous_layer_weights = previous_layer_weights.reshape(flatten_shape)
            flattened_current_layer_weights = current_layer_weights.reshape(flatten_shape)
            single_layer_weights_indexed_iterator = enumerate(zip(flattened_previous_layer_weights,
                                                                  flattened_current_layer_weights))

            for weight_idx, (previous_weight, current_weight) in single_layer_weights_indexed_iterator:
                weight_difference = abs(current_weight - previous_weight)
                if weight_difference > self.minimum_weight_difference_to_send:
                    continue
                flattened_current_layer_weights[weight_idx] = previous_weight

            changed_model_weights[layer_idx] = flattened_current_layer_weights.reshape(layer_shape)

        return changed_model_weights

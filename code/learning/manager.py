from logging import info

from numpy import array, average

from learning.participant import Client, Server
from config.config import LearningConfig
from data_provider.dataset import TestDataset


class FederatedLearningManager:

    def __init__(self, config: LearningConfig, test_dataset: TestDataset, clients: list[Client], server: Server):
        self.iterations = config.iterations
        self.iterations_to_aggregate = config.iterations_to_aggregate
        self.test_dataset = test_dataset
        self.server = server
        self.clients = clients

    def run_learning_cycle(self):
        for iteration in range(1, self.iterations + 1):
            global_weights = self.server.get_model_weights()
            for client in self.clients:
                client.set_model_weights(global_weights)
                client.train_model()

            averaged_weights = self.__get_clients_models_averaged_weights()
            self.server.set_model_weights(averaged_weights)
            loss, accuracy = self.server.test_model(self.test_dataset)
            info(f'Server metrics after {iteration} iteration: loss = {loss}; accuracy = {accuracy}')

    def __get_clients_models_averaged_weights(self) -> list[array]:
        client_models_weights = [client.get_model_weights() for client in self.clients]
        relative_weights = [1/len(self.clients) for _ in self.clients]
        clients_models_averaged_weights = []
        for weights_list_tuple in zip(*client_models_weights):
            averaged_weights = array(
                [average(array(weights), axis=0, weights=relative_weights) for weights in zip(*weights_list_tuple)]
            )
            clients_models_averaged_weights.append(averaged_weights)

        return clients_models_averaged_weights

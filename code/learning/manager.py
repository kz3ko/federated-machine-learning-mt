from logging import info

from numpy import array

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

            averaged_weights = self.__get_clients_average_weights()
            if iteration % self.iterations_to_aggregate == 0:
                self.server.set_model_weights(averaged_weights)
                loss, accuracy = self.server.test_model(self.test_dataset)
                info(f'Server test loss after {iteration} iteration: {loss}\nServer test accuracy after {iteration} '
                     f'iteration: {accuracy}')

    def __get_clients_average_weights(self):
        client_models_weights = [client.get_model_weights() for client in self.clients]
        averaged_weights = []
        for weights_list_tuple in zip(*client_models_weights):
            averaged_weights.append(array([array(weights).mean(axis=0) for weights in zip(*weights_list_tuple)]))

        return averaged_weights

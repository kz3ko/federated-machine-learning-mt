from typing import Type

from config.config import LearningConfig
from data_provider.dataset import TestDataset, ClientDataset
from learning.neural_network import NeuralNetworkModel
from learning.participant import Server, Client


class ParticipantCreator:

    def __init__(self, config: LearningConfig, test_dataset: TestDataset, client_datasets: dict[int, ClientDataset],
                 model_class: Type[NeuralNetworkModel]):
        self.model_epochs = config.iterations_to_aggregate
        self.test_dataset = test_dataset
        self.client_datasets = client_datasets
        self.model_class = model_class

    def create_server(self) -> Server:
        server_model = self.model_class(self.model_epochs)
        server = Server(self.test_dataset, server_model)

        return server

    def create_clients(self) -> list[Client]:
        clients = []
        for client_id, client_dataset in self.client_datasets.items():
            client_model = self.model_class(self.model_epochs)
            client = Client(client_id, client_dataset, client_model)
            clients.append(client)

        return clients

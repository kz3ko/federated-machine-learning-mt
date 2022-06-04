from typing import Type

from config.config import ClientLearningConfig, ServerLearningConfig, TraditionalLearningConfig
from data_provider.dataset import TestDataset, ClientDataset, TrainTraditionalDataset
from learning.neural_network import NeuralNetworkModel
from learning.participant import Participants, Server, Client, TraditionalParticipant


class ParticipantCreator:

    def __init__(self, model_class: Type[NeuralNetworkModel]):
        self.model_class = model_class

    def create_federated_participants(
            self,
            test_dataset: TestDataset,
            client_datasets: dict[int, ClientDataset],
            client_learning_config: ClientLearningConfig,
            server_learning_config: ServerLearningConfig,
    ) -> Participants:
        server = self.__create_server(test_dataset, server_learning_config)
        clients = self.__create_clients(client_datasets, client_learning_config)
        participants = Participants(server=server, clients=clients)

        return participants

    def create_traditional_participants(
            self,
            dataset: TrainTraditionalDataset,
            learning_config: TraditionalLearningConfig
    ) -> Participants:
        model = self.model_class()
        model.epochs = learning_config.epochs
        model.update_early_stopping(learning_config.early_stopping)

        participant = TraditionalParticipant(dataset, model, learning_config)
        participants = Participants(traditional_participant=participant)

        return participants

    def __create_server(self, test_dataset: TestDataset, server_learning_config: ServerLearningConfig) -> Server:
        server_model = self.model_class()
        server = Server(test_dataset, server_model, server_learning_config)

        return server

    def __create_clients(self, client_datasets: dict[int, ClientDataset], client_learning_config: ClientLearningConfig
                         ) -> list[Client]:
        clients = []
        for client_id, client_dataset in client_datasets.items():
            client_model = self.model_class()
            client = Client(client_id, client_dataset, client_model, client_learning_config)
            clients.append(client)

        return clients

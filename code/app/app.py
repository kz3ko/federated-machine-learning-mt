from config.config import config
from data_provider.distributor import DataDistributor
from learning.manager import FederatedLearningManager
from learning.participant_creator import ParticipantCreator
from learning.neural_network import FirstNeuralNetworkModel


class App:

    def __init__(self):
        self.config = config

        data_distributor = DataDistributor(config.data_distribution)
        test_dataset = data_distributor.create_test_dataset()
        client_datasets = data_distributor.create_client_datasets()

        model_class = FirstNeuralNetworkModel
        participant_creator = ParticipantCreator(test_dataset, client_datasets, model_class)
        server = participant_creator.create_server()
        clients = participant_creator.create_clients()

        self.learning_manager = FederatedLearningManager(config.learning, test_dataset, clients, server)

    def run(self):
        self.learning_manager.run_learning_cycle()

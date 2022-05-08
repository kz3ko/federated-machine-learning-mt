from config.config import config
from data_provider.distributor import DataDistributor
from learning.manager import FederatedLearningManager
from learning.participant_creator import ParticipantCreator
from learning.neural_network import FirstNeuralNetworkModel
from analytics.statistics_collector import StatisticsCollector


class App:

    def __init__(self):
        data_distributor = DataDistributor(config.data_distribution)
        test_dataset = data_distributor.create_test_dataset()
        client_datasets = data_distributor.create_client_datasets()

        model_class = FirstNeuralNetworkModel
        participant_creator = ParticipantCreator(config.learning, test_dataset, client_datasets, model_class)
        self.server = participant_creator.create_server()
        self.clients = participant_creator.create_clients()

        self.statistics_collector = StatisticsCollector(self.clients)
        self.learning_manager = FederatedLearningManager(config.learning, test_dataset, self.clients, self.server)

    def run(self):
        self.learning_manager.run_learning_cycle()
        self.server.save_model()
        for client in self.clients:
            client.save_model()

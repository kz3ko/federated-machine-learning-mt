from config.manager import ConfigManager
from data_provider.distributor import DataDistributor
from learning.manager import LearningManager
from learning.participant_creator import ParticipantCreator
from learning.neural_network import FirstNeuralNetworkModel
from analytics.manager import AnalyticsManager


class App:

    def __init__(self):
        self.config_manager = ConfigManager()
        config = self.config_manager.config

        data_distributor = DataDistributor(config.data_distribution)
        test_dataset = data_distributor.create_test_dataset()
        client_datasets = data_distributor.create_client_datasets()

        model_class = FirstNeuralNetworkModel
        participant_creator = ParticipantCreator(test_dataset, client_datasets, model_class)
        participants = participant_creator.create_participants()

        self.analytics_manager = AnalyticsManager(participants)
        self.learning_manager = LearningManager(config.learning, test_dataset, participants, self.analytics_manager)

    def run(self):
        self.learning_manager.run_learning_cycle()
        self.learning_manager.save_models()

        self.analytics_manager.create_plots()
        self.analytics_manager.save_plots()

        self.config_manager.save_used_config()

from config.manager import ConfigManager
from data_provider.distributor import DataDistributor
from learning.manager import LearningManager
from learning.participant_creator import ParticipantCreator
from learning.neural_network import FirstNeuralNetworkModel
from analytics.manager import AnalyticsManager


class App:

    def __init__(self):
        self.model_class = FirstNeuralNetworkModel

        self.config_manager = ConfigManager(self.model_class)
        self.config = self.config_manager.config
        self.learning_type = self.config.learning_type

    def run(self):
        print(self.learning_type)
        if self.learning_type == 'federated':
            self.__run_federated_learning()
        elif self.learning_type == 'traditional':
            self.__run_traditional_learning()

    def __run_federated_learning(self):
        data_distributor = DataDistributor(self.config.dataset, self.config.federated_learning.data_distribution)
        test_dataset = data_distributor.create_test_dataset()

        client_datasets = data_distributor.create_client_datasets()
        participant_creator = ParticipantCreator(test_dataset, client_datasets, self.model_class,
                                                 self.config.federated_learning.client,
                                                 self.config.federated_learning.server)
        participants = participant_creator.create_participants()

        self.analytics_manager = AnalyticsManager(participants)
        self.learning_manager = LearningManager(self.config.federated_learning, test_dataset, participants,
                                                self.analytics_manager)

        self.learning_manager.run_federated_learning_cycle()
        self.learning_manager.make_predictions()
        self.learning_manager.save_models()

        self.analytics_manager.prepare_best_metrics()
        self.analytics_manager.save_collected_metrics_to_files()
        self.analytics_manager.create_plots()
        self.analytics_manager.save_plots()

        self.config_manager.save_used_config()

    def __run_traditional_learning(self):
        print('RUNNING TRADITIONAL LEARNING')

from config.manager import ConfigManager
from data_provider.distributor import DataDistributor
from learning.manager import FederatedLearningManager, TraditionalLearningManager
from learning.participant_creator import ParticipantCreator
from learning.types import LearningType
from learning.neural_network import FirstNeuralNetworkModel, SecondNeuralNetworkModel, ThirdNeuralNetworkModel, \
    FourthNeuralNetworkModel
from analytics.manager import AnalyticsManager


class App:

    def __init__(self):
        self.model_class = ThirdNeuralNetworkModel

        self.config_manager = ConfigManager(self.model_class)
        self.config = self.config_manager.config
        self.learning_type = self.config.learning_type

    def run(self):
        if self.learning_type == LearningType.FEDERATED:
            self.__prepare_managers_for_federated_learning()
        elif self.learning_type == LearningType.TRADITIONAL:
            self.__prepare_managers_for_traditional_learning()

        self.learning_manager.run_learning_cycle()
        self.learning_manager.make_predictions()
        self.learning_manager.save_models()

        self.analytics_manager.prepare_best_metrics()
        self.analytics_manager.save_collected_metrics_to_files()
        self.analytics_manager.create_plots()
        self.analytics_manager.save_plots()

        self.config_manager.save_used_config()

    def __prepare_managers_for_federated_learning(self):
        data_distributor = DataDistributor(self.config.dataset, self.config.federated_learning.data_distribution)
        test_dataset = data_distributor.create_test_dataset()
        client_datasets = data_distributor.create_client_datasets()

        participant_creator = ParticipantCreator(self.model_class)
        participants = participant_creator.create_federated_participants(test_dataset, client_datasets,
                                                                         self.config.federated_learning.client,
                                                                         self.config.federated_learning.server)

        self.analytics_manager = AnalyticsManager(participants)
        self.learning_manager = FederatedLearningManager(self.config.federated_learning, test_dataset, participants,
                                                         self.analytics_manager)

    def __prepare_managers_for_traditional_learning(self):
        data_distributor = DataDistributor(self.config.dataset, self.config.federated_learning.data_distribution)
        test_dataset = data_distributor.create_test_dataset()
        train_dataset = data_distributor.create_train_traditional_dataset()

        participant_creator = ParticipantCreator(self.model_class)
        participants = participant_creator.create_traditional_participants(train_dataset,
                                                                           self.config.traditional_learning)

        self.analytics_manager = AnalyticsManager(participants)
        self.learning_manager = TraditionalLearningManager(self.config.traditional_learning, test_dataset, participants,
                                                           self.analytics_manager)


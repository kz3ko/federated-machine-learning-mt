from logging import info

from learning.participant import Participants
from learning.federated_averaging import FederatedAveraging
from config.config import LearningConfig
from data_provider.dataset import TestDataset
from analytics.manager import AnalyticsManager


class LearningManager:

    def __init__(self, config: LearningConfig, test_dataset: TestDataset, participants: Participants,
                 analytics_manager: AnalyticsManager):
        self.iterations = config.iterations
        self.iterations_to_aggregate = config.iterations_to_aggregate
        self.test_dataset = test_dataset
        self.participants = participants
        self.server = self.participants.server
        self.clients = self.participants.clients
        self.analytics_manager = analytics_manager
        self.federated_averaging = FederatedAveraging(self.clients)

    def run_learning_cycle(self):
        for iteration in range(1, self.iterations + 1):
            global_weights = self.server.get_all_model_weights()
            for client in self.clients:
                client.set_model_weights(global_weights)
                client.train_model()
                self.analytics_manager.save_client_metrics(iteration, client)

            if iteration % self.iterations_to_aggregate != 0:
                continue

            client_models_weights = [client.get_changed_model_weights() for client in self.clients]
            averaged_weights = self.federated_averaging.get_clients_models_averaged_weights(client_models_weights)
            self.server.set_model_weights(averaged_weights)
            metrics = self.server.test_model(self.test_dataset)
            info(f'Server metrics after {iteration} iteration: loss = {metrics.loss}; accuracy = {metrics.accuracy}')
            self.analytics_manager.save_server_metrics(iteration, metrics)

        self.analytics_manager.prepare_best_metrics()
        self.analytics_manager.save_collected_metrics_to_files()

    def make_predictions(self):
        for participant in self.participants:
            predictions = participant.make_predictions(self.test_dataset)
            self.analytics_manager.save_participant_predictions(participant, predictions)

    def save_models(self):
        for participant in self.participants:
            participant.save_model()

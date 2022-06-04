from logging import info

from keras.callbacks import EarlyStopping

from learning.participant import Participants
from config.config import FederatedLearningConfig
from data_provider.dataset import TestDataset
from analytics.manager import AnalyticsManager


class LearningManager:

    def __init__(self, config: FederatedLearningConfig, test_dataset: TestDataset, participants: Participants,
                 analytics_manager: AnalyticsManager):
        self.iterations = config.cycle.iterations
        self.iterations_to_aggregate = config.cycle.iterations_to_aggregate
        self.test_dataset = test_dataset
        self.participants = participants
        self.server = self.participants.server
        self.clients = self.participants.clients
        self.analytics_manager = analytics_manager

    def run_federated_learning_cycle(self):
        for iteration in range(1, self.iterations + 1):
            global_weights = self.server.get_all_model_weights()
            for client in self.clients:
                client.set_model_weights(global_weights)
                client.train_model()
                self.analytics_manager.save_client_metrics(iteration, client)

            if iteration % self.iterations_to_aggregate != 0:
                continue

            clients_models_weights = [client.get_changed_model_weights() for client in self.clients]
            self.server.update_global_weights(clients_models_weights)
            metrics = self.server.train_model()
            info(f'Server metrics after {iteration} iteration: loss = {metrics.loss}; accuracy = {metrics.accuracy}')
            self.analytics_manager.save_server_metrics(iteration, metrics)

            if not self.server.learning_enabled:
                break

    def run_traditional_learning_cycle(self):
        client = self.clients[0]
        client.model.epochs = 2
        client.model.callbacks = [EarlyStopping(patience=3, monitor='val_loss')]
        client.train_model()
        self.analytics_manager.save_traditional_learning_metrics(client)

    def make_predictions(self):
        for participant in self.participants:
            predictions = participant.make_predictions(self.test_dataset)
            self.analytics_manager.save_participant_predictions(participant, predictions)

    def save_models(self):
        for participant in self.participants:
            participant.save_model()

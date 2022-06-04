from analytics.metrics_collector import MetricsCollector
from analytics.plotter import ClientLearningPlotter, ServerTestingPlotter, ConfusionMatrixMaker
from learning.participant import Participants, LearningParticipant, Client
from learning.models import SingleTestMetrics, PredictionMetrics


class AnalyticsManager:

    def __init__(self, participants: Participants):
        self.metrics_collector = MetricsCollector(participants)
        self.plotters = [
            ClientLearningPlotter(self.metrics_collector),
            ServerTestingPlotter(self.metrics_collector),
            ConfusionMatrixMaker(self.metrics_collector)
        ]

    def save_client_metrics(self, iteration: int, client: Client):
        return self.metrics_collector.save_client_metrics(iteration, client)

    def save_server_metrics(self, iteration: int, single_test_metrics: SingleTestMetrics):
        return self.metrics_collector.save_server_metrics(iteration, single_test_metrics)

    def save_traditional_learning_metrics(self, client):
        return self.metrics_collector.save_traditional_learning_metrics(client)

    def save_collected_metrics_to_files(self):
        return self.metrics_collector.save_collected_metrics_to_files()

    def prepare_best_metrics(self):
        return self.metrics_collector.prepare_best_metrics()

    def save_participant_predictions(self, participant: LearningParticipant, predictions: PredictionMetrics):
        self.metrics_collector.save_participant_predictions(participant, predictions)

    def create_plots(self):
        for plotter in self.plotters:
            plotter.create_plots()

    def save_plots(self):
        for plotter in self.plotters:
            plotter.save_plots()

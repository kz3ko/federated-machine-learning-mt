from analytics.metrics_collector import MetricsCollector
from analytics.plotter_creator import PlotterCreator
from learning.participant import Participants, LearningParticipant, Client, TraditionalParticipant
from learning.models import SingleTestMetrics, PredictionMetrics


class AnalyticsManager:

    def __init__(self, participants: Participants):
        self.metrics_collector = MetricsCollector(participants)
        self.plotters_creator = PlotterCreator(participants, self.metrics_collector)
        self.plotters = self.plotters_creator.create_plotters()

    def save_client_metrics(self, iteration: int, client: Client):
        return self.metrics_collector.save_client_metrics(iteration, client)

    def save_server_metrics(self, iteration: int, single_test_metrics: SingleTestMetrics):
        return self.metrics_collector.save_server_metrics(iteration, single_test_metrics)

    def save_traditional_learning_metrics(self, participant: TraditionalParticipant):
        return self.metrics_collector.save_traditional_participant_metrics(participant)

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

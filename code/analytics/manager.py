from analytics.metrics_collector import MetricsCollector
from analytics.plotter import ClientLearningPlotter, ServerTestingPlotter, ConfusionMatrixMaker
from learning.participant import Participants, Client
from learning.models import SingleTestMetrics


class AnalyticsManager:

    def __init__(self, participants: Participants):
        self.statistics_collector = MetricsCollector(participants)
        self.plotters = [
            ClientLearningPlotter(self.statistics_collector),
            ServerTestingPlotter(self.statistics_collector),
            ConfusionMatrixMaker(self.statistics_collector)
        ]

    def save_client_metrics(self, iteration: int, client: Client):
        return self.statistics_collector.save_client_metrics(iteration, client)

    def save_server_metrics(self, iteration: int, single_test_metrics: SingleTestMetrics):
        return self.statistics_collector.save_server_metrics(iteration, single_test_metrics)

    def save_collected_metrics_to_files(self):
        return self.statistics_collector.save_collected_metrics_to_files()

    def create_plots(self):
        for plotter in self.plotters:
            plotter.create_plots()

    def save_plots(self):
        for plotter in self.plotters:
            plotter.save_plots()

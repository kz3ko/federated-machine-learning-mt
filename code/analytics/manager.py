from analytics.metrics_collector import MetricsCollector
from learning.participant import Participants, Client
from learning.models import SingleTestMetrics


class AnalyticsManager:

    def __init__(self, participants: Participants):
        self.statistics_collector = MetricsCollector(participants)

    def save_client_metrics(self, iteration: int, client: Client):
        return self.statistics_collector.save_client_metrics(iteration, client)

    def save_server_metrics(self, iteration: int, single_test_metrics: SingleTestMetrics):
        return self.statistics_collector.gather_server_metrics(iteration, single_test_metrics)

    def save_collected_metrics_to_files(self):
        return self.statistics_collector.save_collected_metrics_to_files()

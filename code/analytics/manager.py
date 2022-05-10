from analytics.statistics_collector import StatisticsCollector
from learning.participant import Participants, Client
from learning.models import SingleTestMetrics


class AnalyticsManager:

    def __init__(self, participants: Participants):
        self.statistics_collector = StatisticsCollector(participants)

    def save_client_metrics(self, iteration: int, client: Client):
        return self.statistics_collector.save_client_metrics(iteration, client)

    def save_server_metrics(self, iteration: int, single_test_metrics: SingleTestMetrics):
        return self.statistics_collector.save_server_metrics(iteration, single_test_metrics)

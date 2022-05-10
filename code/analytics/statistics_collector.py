from analytics.models import ClientMetrics, ServerMetrics
from learning.participant import Participants, Client
from learning.models import SingleTestMetrics


class StatisticsCollector:

    def __init__(self, participants: Participants):
        self.clients_metrics = {client.id: ClientMetrics() for client in participants.clients}
        self.server_metrics = ServerMetrics()

    def save_client_metrics(self, iteration: int, client: Client):
        client_metrics = self.clients_metrics[client.id]
        for metric, value in client.latest_learning_history.history.items():
            [client_metrics.__getattribute__(metric)[iteration]] = value

    def save_server_metrics(self, iteration: int, single_test_metrics: SingleTestMetrics):
        for metric, value in single_test_metrics.__dict__.items():
            self.server_metrics.__getattribute__(metric)[iteration] = value

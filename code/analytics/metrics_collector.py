from pandas import DataFrame

from analytics.models import ParticipantMetrics, ClientMetrics, ServerMetrics
from learning.participant import Participants, Client
from learning.models import SingleTestMetrics
from generated_data.path import generated_data_path
from utilities.utils import create_directory


class MetricsCollector:

    def __init__(self, participants: Participants):
        self.clients_metrics = {client.id: ClientMetrics(client.id) for client in participants.clients}
        self.server_metrics = ServerMetrics('server')
        self.all_metrics = [self.server_metrics, *self.clients_metrics.values()]

    def save_client_metrics(self, iteration: int, client: Client):
        client_metrics = self.clients_metrics[client.id]
        for metric, value in client.latest_learning_history.history.items():
            [client_metrics.__getattribute__(metric)[iteration]] = value

    def gather_server_metrics(self, iteration: int, single_test_metrics: SingleTestMetrics):
        for metric, value in single_test_metrics.__dict__.items():
            self.server_metrics.__getattribute__(metric)[iteration] = value

    def save_collected_metrics_to_files(self):
        metrics_path = generated_data_path.metrics
        create_directory(metrics_path)
        for metrics in self.all_metrics:
            target_path = f'{metrics_path}/{metrics.full_name}.csv'
            metrics_dataframe = self.__create_metrics_dataframe(metrics)
            metrics_dataframe.to_csv(target_path, index=False)

    @staticmethod
    def __create_metrics_dataframe(metrics: ParticipantMetrics) -> DataFrame:
        data = metrics.as_dict()
        metrics_dataframe = DataFrame(data)

        return metrics_dataframe
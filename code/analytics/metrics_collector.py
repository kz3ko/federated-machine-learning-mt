from pandas import DataFrame

from analytics.models import Metrics, ClientMetrics, ServerMetrics, ClientBestMetrics, ServerBestMetrics
from learning.participant import Participants, LearningParticipant, Client
from learning.models import SingleTestMetrics, PredictionMetrics
from generated_data.path import generated_data_path
from utilities.utils import create_directory


class MetricsCollector:

    def __init__(self, participants: Participants):
        self.clients_metrics = {client.id: ClientMetrics(client.id) for client in participants.clients}
        self.server_metrics = ServerMetrics('server')
        self.participants_metrics = [self.server_metrics, *self.clients_metrics.values()]
        self.best_metrics = []
        self.predictions = {}

    def save_client_metrics(self, iteration: int, client: Client):
        client_metrics = self.clients_metrics[client.id]
        client_metrics.iterations.append(iteration)
        for metric, values in client.latest_learning_history.history.items():
            client_metrics.__getattribute__(metric).extend(values)

    def save_server_metrics(self, iteration: int, single_test_metrics: SingleTestMetrics):
        self.server_metrics.iterations.append(iteration)
        for metric, value in single_test_metrics.__dict__.items():
            self.server_metrics.__getattribute__(metric).append(value)

    def prepare_best_metrics(self):
        for metrics in self.participants_metrics:
            max_accuracy = max(metrics.accuracy)
            max_accuracy_iteration = metrics.iterations[metrics.accuracy.index(max_accuracy)]
            min_loss = min(metrics.loss)
            min_loss_iteration = metrics.iterations[metrics.loss.index(min_loss)]
            if isinstance(metrics, ClientMetrics):
                max_val_accuracy = max(metrics.val_accuracy)
                max_val_accuracy_iteration = metrics.iterations[metrics.val_accuracy.index(max_val_accuracy)]
                min_val_loss = min(metrics.val_loss)
                min_val_loss_iteration = metrics.iterations[metrics.val_loss.index(min_val_loss)]
                single_best_metric = ClientBestMetrics(metrics.full_name, max_accuracy, max_accuracy_iteration,
                                                       min_loss, min_loss_iteration, max_val_accuracy,
                                                       max_val_accuracy_iteration, min_val_loss, min_val_loss_iteration)
            else:
                single_best_metric = ServerBestMetrics(metrics.full_name, max_accuracy, max_accuracy_iteration,
                                                       min_loss, min_loss_iteration)

            self.best_metrics.append(single_best_metric)

    def save_collected_metrics_to_files(self):
        all_metrics = [*self.participants_metrics, *self.best_metrics]
        metrics_path = generated_data_path.metrics
        create_directory(metrics_path)
        for metrics in all_metrics:
            target_path = f'{metrics_path}/{metrics.full_name}.csv'
            metrics_dataframe = self.__create_metrics_dataframe(metrics)
            metrics_dataframe.to_csv(target_path, index=False)

    def save_participant_predictions(self, participant: LearningParticipant, predictions: PredictionMetrics):
        self.predictions[participant] = predictions

    @staticmethod
    def __create_metrics_dataframe(metrics: Metrics) -> DataFrame:
        data = metrics.as_dict()
        try:
            metrics_dataframe = DataFrame(data)
        except ValueError:
            metrics_dataframe = DataFrame(data, index=[0])

        return metrics_dataframe

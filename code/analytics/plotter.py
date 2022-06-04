from abc import ABC, abstractmethod

from matplotlib.pyplot import figure, Figure, Axes, rcParams, rcParamsDefault
from sklearn.metrics import confusion_matrix
from seaborn import heatmap, set
from numpy import asarray, sum, array

from analytics.metrics_collector import MetricsCollector
from analytics.models import ClientMetrics, TraditionalParticipantMetrics
from generated_data.path import generated_data_path
from learning.participant import Client, Server
from utilities.utils import create_directory


class Plotter(ABC):

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.plots = {}
        self.figure_size = (6, 5)
        # self.font_scale = 1.0
        self.file_extension = 'jpg'
        self.target_directory = None

    def save_plots(self):
        create_directory(self.target_directory)
        for plot_name, figure_ in self.plots.items():
            target_file_name = f'{plot_name}.{self.file_extension}'
            target_file_path = f'{self.target_directory}/{target_file_name}'
            figure_.savefig(target_file_path, dpi=150)

    def _create_figure(self) -> tuple[Figure, Axes]:
        figure_ = figure(figsize=self.figure_size)
        figure_.set_tight_layout('True')
        axis = figure_.add_subplot()

        return figure_, axis

    @abstractmethod
    def create_plots(self):
        pass


class ClientLearningPlotter(Plotter):

    def __init__(self, metrics_collector: MetricsCollector):
        super().__init__(metrics_collector)
        self.target_directory = f'{generated_data_path.plots}/learning_plots'

    def create_plots(self):
        for client_id, client_metrics in self.metrics_collector.clients_metrics.items():
            client_accuracy_plot_name = f'client_{client_id}_accuracy_plot'
            client_loss_plot_name = f'client_{client_id}_loss_plot'
            self.plots[client_accuracy_plot_name] = self.__create_client_accuracy_plot(client_id, client_metrics)
            self.plots[client_loss_plot_name] = self.__create_client_loss_plot(client_id, client_metrics)

        all_clients_accuracy_plot_name = f'clients_accuracy_comparison_plot'
        all_clients_loss_plot_name = f'clients_loss_comparison_plot'
        self.plots[all_clients_accuracy_plot_name] = self.__create_all_clients_accuracy_comparison()
        self.plots[all_clients_loss_plot_name] = self.__create_all_clients_loss_comparison()

    def __create_client_accuracy_plot(self, client_id: int, client_metrics: ClientMetrics) -> Figure:
        figure_, axis = self._create_figure()
        accuracy_line = axis.plot(client_metrics.iterations, client_metrics.accuracy,
                                  label='Funkcja dokładności uczenia')
        val_accuracy_line = axis.plot(client_metrics.iterations, client_metrics.val_accuracy,
                                      label='Funkcja dokładności walidacji')
        axis.set_title(f'Wykres dokładności (ang. accuracy) procesu uczenia modelu dla klienta \n'
                       f'o identyfikatorze "{client_id}"')
        axis.set_xlabel('Iteracje')
        axis.set_ylabel('Dokładność modelu')
        axis.grid(True)
        axis.legend(handles=[*accuracy_line, *val_accuracy_line], loc='upper left')

        return figure_

    def __create_client_loss_plot(self, client_id: int, client_metrics: ClientMetrics) -> Figure:
        figure_, axis = self._create_figure()
        loss_line = axis.plot(client_metrics.iterations, client_metrics.loss, label='Funkcja strat uczenia')
        val_loss_line = axis.plot(client_metrics.iterations, client_metrics.val_loss, label='Funkcja strat walidacji')
        axis.set_title(f'Wykres strat (ang. loss) procesu uczenia modelu dla klienta \n'
                       f'o identyfikatorze "{client_id}"')
        axis.set_xlabel('Iteracje')
        axis.set_ylabel('Wartosc funkcji strat')
        axis.grid(True)
        axis.legend(handles=[*loss_line, *val_loss_line], loc='upper right')

        return figure_

    def __create_all_clients_accuracy_comparison(self):
        figure_, axis = self._create_figure()
        accuracy_lines = []
        for client_id, client_metrics in self.metrics_collector.clients_metrics.items():
            accuracy_lines.append(axis.plot(client_metrics.iterations, client_metrics.val_accuracy,
                                            label=f'Model klienta o id "{client_id}"'))
        axis.set_title(f'Wykres dokładności (ang. accuracy) procesu uczenia modelu dla \nwszystkich klientów.')
        axis.set_xlabel('Iteracje')
        axis.set_ylabel('Dokładność modelu')
        axis.grid(True)
        axis.legend(handles=[line[0] for line in accuracy_lines], loc='upper left')

        return figure_

    def __create_all_clients_loss_comparison(self):
        figure_, axis = self._create_figure()
        loss_lines = []
        for client_id, client_metrics in self.metrics_collector.clients_metrics.items():
            loss_lines.append(axis.plot(client_metrics.iterations, client_metrics.val_loss,
                                        label=f'Model klienta o id "{client_id}"'))
        axis.set_title(f'Wykres strat (ang. loss) procesu uczenia modelu dla \nwszystkich klientów.')
        axis.set_xlabel('Iteracje')
        axis.set_ylabel('Wartosc funkcji strat')
        axis.grid(True)
        axis.legend(handles=[line[0] for line in loss_lines], loc='upper right')

        return figure_


class TraditionalParticipantLearningPlotter(Plotter):

    def __init__(self, metrics_collector: MetricsCollector):
        super().__init__(metrics_collector)
        self.target_directory = f'{generated_data_path.plots}/learning_plots'

    def create_plots(self):
        metrics = self.metrics_collector.traditional_participant_metrics

        client_accuracy_plot_name = f'{metrics.full_name}_accuracy_plot'
        client_loss_plot_name = f'client_{metrics.full_name}_loss_plot'
        self.plots[client_accuracy_plot_name] = self._create_accuracy_plot(metrics)
        self.plots[client_loss_plot_name] = self._create_loss_plot(metrics)

    def _create_accuracy_plot(self, metrics: TraditionalParticipantMetrics) -> Figure:
        figure_, axis = self._create_figure()
        accuracy_line = axis.plot(metrics.iterations, metrics.accuracy, label='Funkcja dokładności uczenia')
        val_accuracy_line = axis.plot(metrics.iterations, metrics.val_accuracy, label='Funkcja dokładności walidacji')
        axis.set_title(f'Wykres dokładności (ang. accuracy) procesu uczenia')
        axis.set_xlabel('Iteracje')
        axis.set_ylabel('Dokładność modelu')
        axis.grid(True)
        axis.legend(handles=[*accuracy_line, *val_accuracy_line], loc='upper left')

        return figure_

    def _create_loss_plot(self, metrics: TraditionalParticipantMetrics) -> Figure:
        figure_, axis = self._create_figure()
        loss_line = axis.plot(metrics.iterations, metrics.loss, label='Funkcja strat uczenia')
        val_loss_line = axis.plot(metrics.iterations, metrics.val_loss, label='Funkcja strat walidacji')
        axis.set_title(f'Wykres strat (ang. loss) procesu uczenia')
        axis.set_xlabel('Iteracje')
        axis.set_ylabel('Wartosc funkcji strat')
        axis.grid(True)
        axis.legend(handles=[*loss_line, *val_loss_line], loc='upper right')

        return figure_


class ServerTestingPlotter(Plotter):

    def __init__(self, metrics_collector: MetricsCollector):
        super(ServerTestingPlotter, self).__init__(metrics_collector)
        self.server_metrics = self.metrics_collector.server_metrics
        self.target_directory = f'{generated_data_path.plots}/learning_plots'

    def create_plots(self):
        figure_, axis = self._create_figure()
        accuracy_line = axis.plot(self.server_metrics.iterations, self.server_metrics.accuracy,
                                  label='Funkcja dokładności')
        loss_line = axis.plot(self.server_metrics.iterations, self.server_metrics.loss, label='Funkcja strat')
        axis.set_title(f'Wykres dokładności (ang. accuracy) oraz strat (ang. loss) \ndla modelu globalnego')
        axis.set_xlabel('Iteracje')
        axis.set_ylabel('Wartość')
        axis.grid(True)
        axis.legend(handles=[*accuracy_line, *loss_line], loc='upper left')
        self.plots['server_accuracy_and_loss'] = figure_


class ConfusionMatrixMaker(Plotter):

    def __init__(self, metrics_collector: MetricsCollector):
        super(ConfusionMatrixMaker, self).__init__(metrics_collector)
        self.target_directory = f'{generated_data_path.plots}/confusion_matrixes'
        self.font_scale = 2.2
        self.figure_size = (6, 6)
        self.font_size = 22

    def create_plots(self):
        for participant, predictions in self.metrics_collector.predictions.items():
            classes = participant.dataset_used_for_predictions.classes
            number_of_classes = len(classes)

            matrix = confusion_matrix(predictions.max_label, predictions.predicted_max_label)
            self.figure_size = 2 * (2 * number_of_classes,)
            figure_, axis = self._create_figure()

            box_labels = self.__get_box_labels(matrix, number_of_classes)

            heat_map = heatmap(matrix, cmap='Blues', linecolor='black', linewidths=1, xticklabels=classes,
                               yticklabels=classes, annot=box_labels, fmt='', cbar=False,
                               annot_kws={"size": self.font_size},)
            heat_map.set_xticklabels(labels=heat_map.get_xticklabels(), fontsize=self.font_size)
            heat_map.set_yticklabels(labels=heat_map.get_yticklabels(), fontsize=self.font_size)

            if isinstance(participant, Server):
                axis.set_title(f'Macierz pomyłek dla modelu globalnego')
            elif isinstance(participant, Client):
                axis.set_title(f'Macierz pomyłek dla klienta o identyfikatorze "{participant.id}"')
            else:
                axis.set_title(f'Macierz pomyłek')

            axis.title.set_fontsize(self.font_size)
            axis.set_xlabel('Klasa prawdziwa', fontsize=self.font_size)
            axis.set_ylabel('Klasa wyznaczona przez model', fontsize=self.font_size)

            self.plots[participant.full_name] = figure_

    @staticmethod
    def __get_box_labels(matrix: array, number_of_classes: int) -> array:
        matrix_flatten = matrix.flatten()
        label_all_percentage = [f'{value:.2%}' for value in matrix_flatten / sum(matrix)]
        label_class_percentage = [
            f'{value / sum(matrix[i % number_of_classes, :]):.2%}' for i, value in enumerate(matrix_flatten)
        ]
        box_labels = [f'{number}\n{all_percentage}\n{class_percentage}' for number, all_percentage, class_percentage
                      in zip(matrix_flatten, label_all_percentage, label_class_percentage)]

        return asarray(box_labels).reshape(matrix.shape)

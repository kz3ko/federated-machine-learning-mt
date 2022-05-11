from abc import ABC, abstractmethod

from analytics.metrics_collector import MetricsCollector


class Plotter(ABC):

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector

    def save_plots(self):
        print('Saving created plots...')

    @abstractmethod
    def create_plots(self):
        pass


class ClientLearningPlotter(Plotter):

    def create_plots(self):
        print('Creating client learning plot...')


class ServerTestingPlotter(Plotter):

    def create_plots(self):
        print('Creating server testing plot...')


class ConfusionMatrixMaker(Plotter):

    def create_plots(self):
        print('Creating server confusion matrixes...')

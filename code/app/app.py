from typing import Type

from numpy.random import shuffle

from learning.client import Client
from config.config import config
from data_provider.distributor import DataDistributor
from data_provider.normalizer import Normalizer
from learning.neural_network import NeuralNetworkModel, FirstNeuralNetworkModel
from visualisation.plotter import plot_samples, plot_client_data_distribution, get_metrics, plot_accuracy_comparison, \
    plot_loss_comparison


class App:

    def __init__(self):
        self.normalizer = Normalizer()
        self.data_distributor = DataDistributor(config.data_distribution, self.normalizer)
        self.test_dataset = self.data_distributor.create_test_dataset()
        self.clients = self.__create_clients(self.data_distributor, FirstNeuralNetworkModel)

    def run(self):
        client_models = []
        for client in self.clients:
            client_model = client.train_model()
            client_models.append(client_model)
            break


    def plot(self):
        client = self.clients[0]
        samples = client.dataset.samples.copy()
        shuffle(samples)
        print(f'================== ID = {client.id} ==================')
        samples_string = ''.join([f'{class_label.name} - {len([sample for sample in client.dataset.samples if sample.class_label.name == class_label.name])} \n'for class_label in self.data_distributor.dataset_class_labels])
        print(samples_string)
        print('========================================')

        # plot_samples(samples[0:50])
        plot_client_data_distribution(client.dataset)
        print('Done!')

    @staticmethod
    def __create_clients(data_distributor: DataDistributor, model_class: Type[NeuralNetworkModel]) -> list[Client]:
        clients = []
        client_datasets = data_distributor.create_client_datasets()
        for client_id, client_dataset in client_datasets.items():
            client_model = model_class(client_dataset)
            client = Client(client_id, client_dataset, client_model)
            clients.append(client)

        return clients

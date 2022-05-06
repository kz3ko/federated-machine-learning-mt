from typing import Type

from numpy.random import shuffle

from learning.participants import Client, Server
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
        model_class = FirstNeuralNetworkModel
        self.server = self.__create_server(model_class)
        self.clients = self.__create_clients(model_class)

    def run(self):
        learning_timestamp = '2305_06422022'
        client_models = [client.read_model('2305_06422022') for client in self.clients]
        self.server.read_model('2305_06422022')

    def main_run(self):
        client_models = [client.train_model() for client in self.clients]
        for client in self.clients:
            client.save_model()

        self.server.save_model()

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

    def __create_server(self, model_class: Type[NeuralNetworkModel]) -> Server:
        test_dataset = self.data_distributor.create_test_dataset()
        server_model = model_class(test_dataset)
        server = Server(test_dataset, server_model)

        return server

    def __create_clients(self, model_class: Type[NeuralNetworkModel]) -> list[Client]:
        clients = []
        client_datasets = self.data_distributor.create_client_datasets()
        for client_id, client_dataset in client_datasets.items():
            client_model = model_class(client_dataset)
            client = Client(client_id, client_dataset, client_model)
            clients.append(client)

        return clients

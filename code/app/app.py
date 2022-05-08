from typing import Type

from numpy import array
from numpy.random import shuffle
from pandas import DataFrame

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
        self.test_dataset = self.data_distributor.create_test_dataset()
        model_class = FirstNeuralNetworkModel
        self.server = self.__create_server(model_class)
        self.clients = self.__create_clients(model_class)
        self.learning_rounds = 300

    def run(self):
        results = {'learning_round': [], 'accuracy': [], 'loss': []}
        for learning_round in range(1, self.learning_rounds + 1):
            server_weights = self.server.model.base_model.get_weights()
            for client in self.clients:
                client.model.base_model.set_weights(server_weights)
                client.train_model()

            client_models_weights = [client.model.base_model.get_weights() for client in self.clients]
            averaged_weights = list()
            for weights_list_tuple in zip(*client_models_weights):
                averaged_weights.append(array([array(weights).mean(axis=0) for weights in zip(*weights_list_tuple)]))

            self.server.model.base_model.set_weights(averaged_weights)
            loss, accuracy = self.server.test_model(self.test_dataset)
            print(f'Server test loss after {learning_round} learning round: {loss}')
            print(f'Server test accuracy after {learning_round} learning round: {accuracy}\n')

            results['learning_round'].append(learning_round)
            results['accuracy'].append(accuracy)
            results['loss'].append(loss)

        results_dataframe = DataFrame(results)
        results_dataframe.to_csv('results.csv')

        for client in self.clients:
            client.save_model()
        self.server.save_model()

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
        server_model = model_class()
        server = Server(self.test_dataset, server_model)

        return server

    def __create_clients(self, model_class: Type[NeuralNetworkModel]) -> list[Client]:
        clients = []
        client_datasets = self.data_distributor.create_client_datasets()
        for client_id, client_dataset in client_datasets.items():
            client_model = model_class()
            client = Client(client_id, client_dataset, client_model)
            clients.append(client)

        return clients

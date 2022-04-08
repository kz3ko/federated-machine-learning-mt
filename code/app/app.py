from random import shuffle

from learning_participants.client import Client
from config.config import config
from data_provider.distributor import DataDistributor
from visualisation.plotter import plot_samples, plot_client_data_distribution


class App:

    def __init__(self):
        self.data_distributor = DataDistributor(config.data_distribution)
        self.test_dataset = self.data_distributor.create_test_dataset()
        self.clients = self.__create_clients(self.data_distributor)

    def run(self):
        client = self.clients[0]
        samples = client.dataset.samples.copy()
        shuffle(samples)
        print(f'================== ID = {client.id} ==================')
        samples_string = ''.join([f'{class_label.name} - {len([sample for sample in client.dataset.samples if sample.class_label.name == class_label.name])} \n'for class_label in self.data_distributor.dataset_class_labels])
        print(samples_string)
        print('========================================')

        # plot_samples(samples[0:25])
        plot_client_data_distribution(client.dataset)
        print('Done!')

    @staticmethod
    def __create_clients(data_distributor: DataDistributor) -> list[Client]:
        client_datasets = data_distributor.create_client_datasets()

        return [Client(client_id, client_dataset) for client_id, client_dataset in client_datasets.items()]

from numpy.random import shuffle

from config.config import config
from data_provider.distributor import DataDistributor
from learning.manager import FederatedLearningManager
from learning.participant_creator import ParticipantCreator
from learning.neural_network import FirstNeuralNetworkModel
from visualisation.plotter import plot_samples, plot_client_data_distribution, get_metrics, plot_accuracy_comparison, \
    plot_loss_comparison


class App:

    def __init__(self):
        self.config = config
        self.data_distributor = DataDistributor(config.data_distribution)
        self.test_dataset = self.data_distributor.create_test_dataset()
        self.client_datasets = self.data_distributor.create_client_datasets()

        self.model_class = FirstNeuralNetworkModel
        self.participant_creator = ParticipantCreator(self.test_dataset, self.client_datasets, self.model_class)
        self.server = self.participant_creator.create_server()
        self.clients = self.participant_creator.create_clients()

        self.learning_manager = FederatedLearningManager(self.config.learning, self.test_dataset, self. clients,
                                                         self.server)

    def run(self):
        self.learning_manager.run_learning_cycle()
        # self.plot()

    def plot(self):
        client = self.clients[3]
        samples = client.dataset.samples.copy()
        shuffle(samples)
        print(f'================== ID = {client.id} ==================')
        samples_string = ''.join([f'{class_label.name} - {len([sample for sample in client.dataset.samples if sample.class_label.name == class_label.name])} \n'for class_label in self.data_distributor.dataset_class_labels])
        print(samples_string)
        print('========================================')

        plot_samples(samples[0:50])
        plot_client_data_distribution(client.dataset)
        print('Done!')

from numpy import array
from numpy.random import shuffle

from learning_participants.client import Client
from config.config import config
from data_provider.distributor import DataDistributor
from data_provider.normalizer import Normalizer
from visualisation.plotter import plot_samples, plot_client_data_distribution, get_metrics, plot_accuracy_comparison, \
    plot_loss_comparison


class App:

    def __init__(self):
        self.normalizer = Normalizer()
        self.data_distributor = DataDistributor(config.data_distribution, self.normalizer)
        self.test_dataset = self.data_distributor.create_test_dataset()
        self.clients = self.__create_clients(self.data_distributor)

    def run(self):
        number_of_epochs = 45
        batch_size = 64
        verbosity = 1
        validation_split = 0.2

        client_models = []
        for client in self.clients:
            client_model = client.create_model()
            client_models.append(client_model)
            history = client_model.fit(
                client.dataset.input_values,
                client.dataset.target_labels,
                batch_size=batch_size,
                epochs=number_of_epochs,
                verbose=verbosity,
                validation_split=validation_split,
                shuffle=True
            )
            accuracy, val_accuracy, loss, val_loss = get_metrics(history)
            plot_accuracy_comparison(accuracy, val_accuracy)
            plot_loss_comparison(loss, val_loss)
            client_model.evaluate(
                self.test_dataset.input_values,
                self.test_dataset.target_labels,
                batch_size=batch_size
            )

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
    def __create_clients(data_distributor: DataDistributor) -> list[Client]:
        client_datasets = data_distributor.create_client_datasets()

        return [Client(client_id, client_dataset) for client_id, client_dataset in client_datasets.items()]

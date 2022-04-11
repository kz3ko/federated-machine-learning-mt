from matplotlib import pyplot as plt
from tensorflow.python.data import Dataset

from data_provider.models import ClassLabel
from learning_participants.client import Client
from data_provider.distributor import DataDistributor
from config.config import config


def plot_samples(dataset: Dataset, class_: ClassLabel):
    plt.rcParams['figure.figsize'] = (2.5, 2.5)  # set default size of plots
    col1 = 10
    row1 = 1
    fig = plt.figure(figsize=(col1, row1))
    for index in range(0, col1 * row1):
        fig.add_subplot(row1, col1, index + 1)
        plt.axis('off')
        plt.imshow(dataset[index])  # index of the sample picture
        plt.title("Class " + class_.name)
    plt.show()


def main():
    data_distributor = DataDistributor(config.data_distribution)
    client_datasets = data_distributor.create_client_datasets()
    clients = [Client(client_id, client_dataset) for client_id, client_dataset in client_datasets.items()]

    client = clients[0]
    print(f'================== ID = {client.id} ==================')
    samples_string = ''.join([f'{class_label.name} - {len([sample for sample in client.dataset.samples if sample.class_label.name == class_label.name])} \n' for class_label in data_distributor.dataset_class_labels])
    print(samples_string)
    print('========================================')

    print('Done!')


if __name__ == '__main__':
    main()

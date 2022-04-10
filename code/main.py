from typing import Optional

import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from tensorflow.python.data import Dataset
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset
from tensorflow.python.framework.ops import EagerTensor
from tensorflow_datasets.core.dataset_info import DatasetInfo

from models import SampleClass, Sample
from dataset import RawDatasetClasses, ClassDataset, ClientDataset
from client import Client


def load_dataset(name: str, *args, **kwargs) -> tuple[PrefetchDataset, Optional[DatasetInfo]]:
    return tfds.load(name, *args, **kwargs)


def plot_samples(dataset: Dataset, class_: SampleClass):
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


def get_samples_per_class(dataset: PrefetchDataset, classes: RawDatasetClasses) -> dict[SampleClass, list[EagerTensor]]:
    samples_per_class = {class_: [] for class_ in classes}
    for sample_value, label in dataset:
        class_ = classes[int(label.numpy())]
        sample = Sample(sample_value, class_)
        samples_per_class[class_].append(sample)

    return samples_per_class


def get_class_datasets(samples_per_class: dict[SampleClass, list[EagerTensor]]) -> dict[SampleClass, ClassDataset]:
    class_datasets = {}
    for class_, samples in samples_per_class.items():
        class_datasets[class_] = ClassDataset(class_, samples)

    return class_datasets


def get_main_classes_per_client(
        classes: RawDatasetClasses,
        number_of_main_classes_per_client: int,
        class_datasets: RawDatasetClasses
) -> list[list[SampleClass]]:
    main_classes_per_client = []
    for i in range(0, len(classes), number_of_main_classes_per_client):
        main_classes_per_client.append(class_datasets[i: i + number_of_main_classes_per_client])

    return main_classes_per_client


def __distribute_main_class_samples_between_client_datasets(
        number_of_clients: int,
        main_classes_per_client: list[list[SampleClass]],
        class_datasets: dict[SampleClass, ClassDataset],
        client_main_class_ownership_ratio: float
) -> dict[int, list[Sample]]:
    client_samples = {client_id: [] for client_id in range(number_of_clients)}
    for client_id, client_classes in zip(client_samples, main_classes_per_client):
        for class_ in client_classes:
            dataset = class_datasets[class_]
            number_of_samples_per_main_class = int(len(dataset) * client_main_class_ownership_ratio)
            samples = [Sample(value, class_) for value in dataset[0: number_of_samples_per_main_class]]
            client_samples[client_id].extend(samples)
            dataset.truncate(number_of_samples_per_main_class)

    return client_samples


def __distribute_side_class_samples_between_client_datasets(
        client_samples: dict[int, list[Sample]],
        main_classes_per_client: list[list[SampleClass]],
        class_datasets: dict[SampleClass, ClassDataset],
        classes: RawDatasetClasses
) -> dict[int, list[Sample]]:
    samples_of_side_class_per_client = {
        class_: len(dataset) / len(client_samples) for class_, dataset in class_datasets.items()
    }
    for client_id, client_classes in zip(client_samples, main_classes_per_client):
        side_classes = list(set(classes).difference(set(client_classes)))
        for class_ in side_classes:
            dataset = class_datasets[class_]
            number_of_samples_per_side_class = int(samples_of_side_class_per_client[class_])
            samples = [Sample(value, class_) for value in dataset[0: number_of_samples_per_side_class]]
            client_samples[client_id].extend(samples)
            dataset.truncate(number_of_samples_per_side_class)

    return client_samples


def distribute_samples_between_client_datasets(
        number_of_clients: int,
        main_classes_per_client: list[list[SampleClass]],
        class_datasets: dict[SampleClass, ClassDataset],
        client_main_class_ownership_ratio: float,
        classes: RawDatasetClasses,
) -> dict[int, ClientDataset]:
    client_samples = __distribute_main_class_samples_between_client_datasets(number_of_clients, main_classes_per_client,
                                                                             class_datasets,
                                                                             client_main_class_ownership_ratio)

    client_samples = __distribute_side_class_samples_between_client_datasets(client_samples, main_classes_per_client,
                                                                             class_datasets, classes)

    client_datasets = {client_id: ClientDataset(samples) for client_id, samples in client_samples.items()}

    return client_datasets


def main():
    dataset_name = 'cifar100'
    number_of_clients = 10
    number_of_main_classes_per_client = 10
    main_class_ownership_per_client_ratio = 0.7

    whole_dataset, dataset_info = load_dataset(dataset_name, split='all', with_info=True, as_supervised=True)

    dataset_classes = RawDatasetClasses(dataset_info)

    samples_per_class = get_samples_per_class(whole_dataset, dataset_classes)

    main_classes_per_client = get_main_classes_per_client(dataset_classes, number_of_main_classes_per_client,
                                                          dataset_classes)

    class_datasets = get_class_datasets(samples_per_class)

    client_datasets = distribute_samples_between_client_datasets(number_of_clients, main_classes_per_client,
                                                                 class_datasets, main_class_ownership_per_client_ratio,
                                                                 dataset_classes)

    clients = [Client(client_id, client_dataset) for client_id, client_dataset in client_datasets.items()]

    for client in clients:
        if client.id == 1:
            print(f'================== ID = {client.id} ==================')
            samples_string = ''.join([f'{class_.name} - {len([sample for sample in client.dataset.samples if sample.class_.name == class_.name])} \n' for class_ in dataset_classes])
            print(samples_string)
            print('========================================')

    print('Done!')


if __name__ == '__main__':
    main()

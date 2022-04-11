from tensorflow.python.framework.ops import EagerTensor

from data_provider.loader import DatasetLoader
from data_provider.models import ClassLabel, Sample
from data_provider.dataset import ClassDataset, ClientDataset
from data_provider.class_labels import DatasetClassLabels
from config.config import DataDistributionConfig


class DataDistributor:

    def __init__(self, config: DataDistributionConfig):
        self.dataset_name = config.dataset_name
        self.clients_number = config.clients_number
        self.main_classes_per_client_number = config.main_classes_per_client_number
        self.main_class_ownership_per_client_ratio = config.main_class_ownership_per_client_ratio
        self.dataset, self.dataset_info = DatasetLoader.load(self.dataset_name, split='all', with_info=True,
                                                             as_supervised=True)
        self.dataset_class_labels = DatasetClassLabels(self.dataset_info)

    def create_client_datasets(self) -> dict[int, ClientDataset]:
        samples_per_class = self.__get_samples_per_class()
        class_datasets = self.__get_class_datasets(samples_per_class)
        main_classes_per_client = self.__get_main_classes_per_client()
        client_datasets = self.__distribute_samples_between_client_datasets(main_classes_per_client, class_datasets)

        return client_datasets

    def __get_samples_per_class(self) -> dict[ClassLabel, list[EagerTensor]]:
        samples_per_class = {class_label: [] for class_label in self.dataset_class_labels}
        for sample_value, label in self.dataset:
            class_label = self.dataset_class_labels[int(label.numpy())]
            sample = Sample(sample_value, class_label)
            samples_per_class[class_label].append(sample)

        return samples_per_class

    @staticmethod
    def __get_class_datasets(samples_per_class: dict[ClassLabel, list[EagerTensor]]) -> dict[ClassLabel, ClassDataset]:
        return {class_label: ClassDataset(class_label, samples) for class_label, samples in samples_per_class.items()}

    def __get_main_classes_per_client(self) -> list[list[ClassLabel]]:
        return [self.dataset_class_labels[i: i + self.main_classes_per_client_number] for i in range(
            0, len(self.dataset_class_labels), self.main_classes_per_client_number)]

    def __distribute_main_class_samples_between_client_datasets(
            self,
            main_class_labels_per_client: list[list[ClassLabel]],
            class_datasets: dict[ClassLabel, ClassDataset],
    ) -> dict[int, list[Sample]]:
        client_samples = {client_id: [] for client_id in range(self.clients_number)}
        for client_id, client_class_labels in zip(client_samples, main_class_labels_per_client):
            for class_label in client_class_labels:
                dataset = class_datasets[class_label]
                number_of_samples_per_main_class = int(len(dataset) * self.main_class_ownership_per_client_ratio)
                client_samples[client_id].extend(dataset.pop(number_of_samples_per_main_class))

        return client_samples

    def __distribute_side_class_samples_between_client_datasets(
            self,
            client_samples: dict[int, list[Sample]],
            main_class_labels_per_client: list[list[ClassLabel]],
            class_datasets: dict[ClassLabel, ClassDataset],
    ) -> dict[int, list[Sample]]:
        samples_of_side_class_per_client = {
            class_: len(dataset) / len(client_samples) for class_, dataset in class_datasets.items()
        }
        for client_id, client_class_labels in zip(client_samples, main_class_labels_per_client):
            side_classes = list(set(self.dataset_class_labels).difference(set(client_class_labels)))
            for class_label in side_classes:
                dataset = class_datasets[class_label]
                number_of_samples_per_side_class = int(samples_of_side_class_per_client[class_label])
                client_samples[client_id].extend(dataset.pop(number_of_samples_per_side_class))

        return client_samples

    def __distribute_samples_between_client_datasets(
            self,
            main_class_labels_per_client: list[list[ClassLabel]],
            class_datasets: dict[ClassLabel, ClassDataset],
    ) -> dict[int, ClientDataset]:
        client_samples = self.__distribute_main_class_samples_between_client_datasets(
            main_class_labels_per_client,
            class_datasets
        )
        client_samples = self.__distribute_side_class_samples_between_client_datasets(
            client_samples,
            main_class_labels_per_client,
            class_datasets
        )
        client_datasets = {client_id: ClientDataset(samples) for client_id, samples in client_samples.items()}

        return client_datasets

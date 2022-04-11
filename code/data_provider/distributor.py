from tensorflow.python.framework.ops import EagerTensor

from data_provider.loader import DatasetLoader
from data_provider.models import SampleClass, Sample
from data_provider.dataset import RawDatasetClasses, ClassDataset, ClientDataset
from config.config import DataDistributionConfig


class DataDistributor:

    def __init__(self, config: DataDistributionConfig):
        self.dataset_name = config.dataset_name
        self.clients_number = config.clients_number
        self.main_classes_per_client_number = config.main_classes_per_client_number
        self.main_class_ownership_per_client_ratio = config.main_class_ownership_per_client_ratio
        self.dataset, self.dataset_info = DatasetLoader.load(self.dataset_name, split='all', with_info=True,
                                                             as_supervised=True)
        self.dataset_classes = RawDatasetClasses(self.dataset_info)

    def get_client_datasets(self) -> dict[int, ClientDataset]:
        samples_per_class = self.__get_samples_per_class()
        class_datasets = self.__get_class_datasets(samples_per_class)
        main_classes_per_client = self.__get_main_classes_per_client()
        client_datasets = self.__distribute_samples_between_client_datasets(main_classes_per_client, class_datasets)

        return client_datasets

    def __get_samples_per_class(self) -> dict[SampleClass, list[EagerTensor]]:
        samples_per_class = {class_: [] for class_ in self.dataset_classes}
        for sample_value, label in self.dataset:
            class_ = self.dataset_classes[int(label.numpy())]
            sample = Sample(sample_value, class_)
            samples_per_class[class_].append(sample)

        return samples_per_class

    @staticmethod
    def __get_class_datasets(samples_per_class: dict[SampleClass, list[EagerTensor]]) -> dict[SampleClass, ClassDataset]:
        class_datasets = {}
        for class_, samples in samples_per_class.items():
            class_datasets[class_] = ClassDataset(class_, samples)

        return class_datasets

    def __get_main_classes_per_client(self) -> list[list[SampleClass]]:
        main_classes_per_client = []
        for i in range(0, len(self.dataset_classes), self.main_classes_per_client_number):
            main_classes_per_client.append(self.dataset_classes[i: i + self.main_classes_per_client_number])

        return main_classes_per_client

    def __distribute_main_class_samples_between_client_datasets(
            self,
            main_classes_per_client: list[list[SampleClass]],
            class_datasets: dict[SampleClass, ClassDataset],
    ) -> dict[int, list[Sample]]:
        client_samples = {client_id: [] for client_id in range(self.clients_number)}
        for client_id, client_classes in zip(client_samples, main_classes_per_client):
            for class_ in client_classes:
                dataset = class_datasets[class_]
                number_of_samples_per_main_class = int(len(dataset) * self.main_class_ownership_per_client_ratio)
                samples = [Sample(value, class_) for value in dataset[0: number_of_samples_per_main_class]]
                client_samples[client_id].extend(samples)
                dataset.truncate(number_of_samples_per_main_class)

        return client_samples

    def __distribute_side_class_samples_between_client_datasets(
            self,
            client_samples: dict[int, list[Sample]],
            main_classes_per_client: list[list[SampleClass]],
            class_datasets: dict[SampleClass, ClassDataset],
    ) -> dict[int, list[Sample]]:
        samples_of_side_class_per_client = {
            class_: len(dataset) / len(client_samples) for class_, dataset in class_datasets.items()
        }
        for client_id, client_classes in zip(client_samples, main_classes_per_client):
            side_classes = list(set(self.dataset_classes).difference(set(client_classes)))
            for class_ in side_classes:
                dataset = class_datasets[class_]
                number_of_samples_per_side_class = int(samples_of_side_class_per_client[class_])
                samples = [Sample(value, class_) for value in dataset[0: number_of_samples_per_side_class]]
                client_samples[client_id].extend(samples)
                dataset.truncate(number_of_samples_per_side_class)

        return client_samples

    def __distribute_samples_between_client_datasets(
            self,
            main_classes_per_client: list[list[SampleClass]],
            class_datasets: dict[SampleClass, ClassDataset],
    ) -> dict[int, ClientDataset]:
        client_samples = self.__distribute_main_class_samples_between_client_datasets(
            main_classes_per_client,
            class_datasets
        )
        client_samples = self.__distribute_side_class_samples_between_client_datasets(
            client_samples,
            main_classes_per_client,
            class_datasets
        )
        client_datasets = {client_id: ClientDataset(samples) for client_id, samples in client_samples.items()}

        return client_datasets

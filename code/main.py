from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Generator

import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from tensorflow.python.data import Dataset
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset
from tensorflow.python.framework.ops import EagerTensor
from tensorflow_datasets.core.dataset_info import DatasetInfo


@dataclass(unsafe_hash=True)
class SampleClass:
    id: int
    name: str


@dataclass(unsafe_hash=True)
class Sample:
    value: EagerTensor
    class_: SampleClass


class TruncatePoint(Enum):
    FIRST = 'first'
    LAST = 'last'


class RawDatasetClasses:

    def __init__(self, dataset_info: DatasetInfo):
        self.classes = self.__get_dataset_classes(dataset_info)

    def __repr__(self) -> str:
        return str(self.classes)

    def __len__(self) -> int:
        return len(self.classes)

    def __iter__(self) -> Generator[SampleClass, None, None]:
        for class_ in self.classes:
            yield class_

    def __getitem__(self, key: Union[int, str, slice]) -> SampleClass:
        if isinstance(key, int):
            [output] = filter(lambda cls: cls.id == key, self.classes)
        elif isinstance(key, str):
            [output] = filter(lambda cls: cls.name == key, self.classes)
        elif isinstance(key, slice):
            start = key.start if key.start else 0
            stop = key.stop
            step = key.step if key.step else 1
            output = self.classes[start: stop: step]
        else:
            raise TypeError(f'{self.__class__.__name__} indices must be "int", "str" or "slice", not {type(key)}.')

        return output

    @staticmethod
    def __get_dataset_classes(dataset_info: DatasetInfo) -> list[SampleClass]:
        return [SampleClass(dataset_info.features['label'].str2int(name), name) for name in
                dataset_info.features['label'].names]


class CustomDataset(ABC):

    def __init__(self, samples: list[EagerTensor]):
        self.samples = samples
        self.data = self.__get_data_from_samples(self.samples)

    def __getitem__(self, key: int) -> EagerTensor:
        if isinstance(key, int):
            [output] = [sample for sample in self.data.skip(key).take(1)]
        elif isinstance(key, slice):
            start = key.start if key.start else 0
            stop = key.stop
            output = [sample for sample in self.data.skip(start).take(stop - start)]
        else:
            raise TypeError(f'{self.__class__.__name__} indices must be "int" or "slice", not {type(key)}.')

        return output

    def __iter__(self) -> Generator[EagerTensor, None, None]:
        for sample in self.data:
            yield sample

    def __copy__(self):
        cls = self.__class__
        copied_dataset = cls.__new__(cls)
        copied_dataset.__dict__.update(self.__dict__)
        return copied_dataset

    def truncate(self, number: int, where: TruncatePoint = TruncatePoint.FIRST):
        if where == TruncatePoint.FIRST:
            self.samples = self.samples[number:]
        elif where == TruncatePoint.LAST:
            self.samples = self.samples[:-number]
        else:
            raise TypeError('You can only truncate first or last samples from dataset!')

        self.data = self.__get_data_from_samples(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def __get_data_from_samples(samples: list[EagerTensor]) -> Dataset:
        return Dataset.from_tensor_slices(samples)


class ClassDataset(CustomDataset):

    def __init__(self, class_: SampleClass, samples: list[EagerTensor]):
        super(ClassDataset, self).__init__(samples)
        self.class_ = class_


class ClientDataset(CustomDataset):

    def __init__(self, samples: list[EagerTensor]):
        super(ClientDataset, self).__init__(samples)


def load_dataset(name: str, *args, **kwargs) -> tuple[PrefetchDataset, Optional[DatasetInfo]]:
    return tfds.load(name, *args, **kwargs)


def get_sample_by_index(dataset: Dataset, index: int):
    for sample in dataset.skip(index).take(1):
        return sample


def plot_samples(dataset: Dataset, class_: SampleClass):
    plt.rcParams['figure.figsize'] = (2.5, 2.5)  # set default size of plots
    col1 = 10
    row1 = 1
    fig = plt.figure(figsize=(col1, row1))
    for index in range(0, col1 * row1):
        fig.add_subplot(row1, col1, index + 1)
        plt.axis('off')
        plt.imshow(get_sample_by_index(dataset, index))  # index of the sample picture
        plt.title("Class " + class_.name)
    plt.show()


def main():
    dataset_name = 'cifar100'
    number_of_clients = 10
    number_of_main_classes_per_client = 5
    main_class_ownership_per_client_ratio = 0.7

    whole_dataset, dataset_info = load_dataset(dataset_name, split='all', with_info=True, as_supervised=True)
    raw_dataset_classes = RawDatasetClasses(dataset_info)

    number_of_classes = len(raw_dataset_classes)
    number_of_samples = len(whole_dataset)

    client_ids = [client_id for client_id in range(number_of_clients)]

    samples_per_class = {class_: [] for class_ in raw_dataset_classes}
    for sample, label in whole_dataset:
        class_ = raw_dataset_classes[int(label.numpy())]
        samples_per_class[class_].append(sample)

    class_datasets = {}
    for class_, samples in samples_per_class.items():
        class_datasets[class_] = ClassDataset(class_, samples)

    main_classes_per_client = []
    for i in range(0, number_of_classes, number_of_main_classes_per_client):
        main_classes_per_client.append(raw_dataset_classes[i: i + number_of_main_classes_per_client])

    client_samples = {client_id: [] for client_id in client_ids}
    for client_id, client_classes in zip(client_samples, main_classes_per_client):
        for class_ in client_classes:
            dataset = class_datasets[class_]
            number_of_samples_per_main_class = int(len(dataset) * main_class_ownership_per_client_ratio)
            samples = [Sample(value, class_) for value in dataset[0: number_of_samples_per_main_class]]
            client_samples[client_id].extend(samples)
            dataset.truncate(number_of_samples_per_main_class)

    samples_of_side_class_per_client = {class_: len(dataset) / number_of_clients for class_, dataset in
                                        class_datasets.items()}

    for client_id, client_classes in zip(client_samples, main_classes_per_client):
        side_classes = list(set(raw_dataset_classes).difference(set(client_classes)))
        for class_ in side_classes:
            dataset = class_datasets[class_]
            number_of_samples_per_side_class = int(samples_of_side_class_per_client[class_])
            samples = [Sample(value, class_) for value in dataset[0: number_of_samples_per_side_class]]
            client_samples[client_id].extend(samples)
            dataset.truncate(number_of_samples_per_side_class)

    for client_id, samples in client_samples.items():
        if client_id == 1:
            print(f'================== ID = {client_id} ==================')
            samples_string = ''.join([f'{class_.name} - {len([sample for sample in samples if sample.class_.name == class_.name])} \n' for class_ in raw_dataset_classes])
            print(samples_string)
            print('========================================')

    print('Done!')


if __name__ == '__main__':
    main()

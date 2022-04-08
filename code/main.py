from dataclasses import dataclass
from typing import Optional, Union, Generator
from abc import ABC

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.data import Dataset
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset, TakeDataset
from tensorflow.python.framework.ops import EagerTensor
from tensorflow_datasets.core.dataset_info import DatasetInfo
from matplotlib import pyplot as plt

from config.config import CONFIG


@dataclass(unsafe_hash=True)
class SampleClass:
    id: int
    name: str


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

    def __getitem__(self, key: Union[int, str]) -> SampleClass:
        if isinstance(key, int):
            [class_] = filter(lambda cls: cls.id == key, self.classes)
        elif isinstance(key, str):
            [class_] = filter(lambda cls: cls.name == key, self.classes)
        else:
            raise TypeError(f'{self.__class__.__name__} indices must be "int" or "str", not {type(key)}.')

        return class_

    @staticmethod
    def __get_dataset_classes(dataset_info: DatasetInfo) -> list[SampleClass]:
        return [SampleClass(dataset_info.features['label'].str2int(name), name) for name in
                dataset_info.features['label'].names]


class CustomDataset(ABC):

    def __init__(self, samples: list[EagerTensor]):
        self.samples = samples
        self.data = self.__get_data_from_samples(self.samples)

    def __getitem__(self, key: int) -> EagerTensor:
        if isinstance(key, slice):
            indices = range(*key.indices(len(self.list)))
            return [self.list[i] for i in indices]
        elif isinstance(key, int)
            for sample in self.data.skip(key).take(1):
                return sample

    def __iter__(self) -> Generator[EagerTensor, None, None]:
        for sample in self.data:
            yield sample

    def __copy__(self):
        cls = self.__class__
        copied_dataset = cls.__new__(cls)
        copied_dataset.__dict__.update(self.__dict__)
        return copied_dataset

    def pop(self, sample_to_pop: EagerTensor) -> EagerTensor:
        [popped_sample] = [sample for sample in self.samples if sample == sample_to_pop]
        self.samples.remove(popped_sample)
        self.data = self.__get_data_from_samples(self.samples)

        return popped_sample

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
    main_classes_per_client = 5
    main_class_ownership_per_client = 0.7

    whole_dataset, dataset_info = load_dataset(dataset_name, split='all', with_info=True, as_supervised=True)
    raw_dataset_classes = RawDatasetClasses(dataset_info)

    classes_per_client = int(len(raw_dataset_classes)/10)
    samples_per_client = int(len(whole_dataset)/10)

    samples_per_class = {class_: [] for class_ in raw_dataset_classes}
    for sample, label in whole_dataset:
        class_ = raw_dataset_classes[int(label.numpy())]
        samples_per_class[class_].append(sample)

    class_datasets = {}
    for class_, samples in samples_per_class.items():
        class_datasets[class_] = ClassDataset(class_, samples)

    client_datasets = {}
    current_client = 0
    current_main_class_number = 0
    for class_ in raw_dataset_classes:
        dataset = class_datasets[class_]



    # class_ = raw_dataset_classes[5]
    # print(class_.name)
    # print(class_datasets[class_][123])

    print('Done!')


if __name__ == '__main__':
    main()

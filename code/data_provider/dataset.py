from __future__ import annotations

from typing import Union, Generator
from enum import Enum
from abc import ABC

from tensorflow.python.data import Dataset

from data_provider.models import ClassLabel, Sample


class CutOffSide(Enum):
    FIRST = 'first'
    LAST = 'last'


class CustomDataset(ABC):

    def __init__(self, samples: list[Sample]):
        self.samples = samples
        self.number_of_samples_per_class = self.__count_samples_per_class(self.samples)
        self.classic_dataset = self.__create_classic_dataset_from_samples(self.samples)

    def __getitem__(self, key: int) -> Union[Sample, list[Sample]]:
        if isinstance(key, int):
            output = self.samples[key]
        elif isinstance(key, slice):
            start = key.start if key.start else 0
            stop = key.stop
            output = self.samples[start: stop]
        else:
            raise TypeError(f'{self.__class__.__name__} indices must be "int" or "slice", not {type(key)}.')

        return output

    def __iter__(self) -> Generator[Sample, None, None]:
        for sample in self.samples:
            yield sample

    def __copy__(self) -> CustomDataset:
        cls = self.__class__
        copied_dataset = cls.__new__(cls)
        copied_dataset.__dict__.update(self.__dict__)
        return copied_dataset

    def __len__(self) -> int:
        return len(self.samples)

    def truncate(self, number: int, cut_off_side: CutOffSide = CutOffSide.FIRST):
        if cut_off_side == CutOffSide.FIRST:
            self.samples = self.samples[number:]
        elif cut_off_side == CutOffSide.LAST:
            self.samples = self.samples[:-number]
        else:
            raise TypeError('You can only truncate first or last samples from dataset!')

        self.classic_dataset = self.__create_classic_dataset_from_samples(self.samples)

    def pop(self, number: int, cut_off_side: CutOffSide = CutOffSide.FIRST) -> Union[Sample, list[Sample]]:
        if cut_off_side == CutOffSide.FIRST:
            popped_samples = self.samples[:number]
        elif cut_off_side == CutOffSide.LAST:
            popped_samples = self.samples[-number:]
        else:
            raise TypeError('You can only truncate first or last samples from dataset!')

        self.truncate(number, cut_off_side)

        return popped_samples

    @staticmethod
    def __create_classic_dataset_from_samples(samples: list[Sample]) -> Dataset:
        return Dataset.from_tensor_slices([sample.value for sample in samples])

    @staticmethod
    def __count_samples_per_class(samples: list[Sample]) -> dict[ClassLabel, int]:
        class_labels = list(set([sample.class_label for sample in samples]))
        sample_class_labels = [sample.class_label for sample in samples]

        return {class_label: sample_class_labels.count(class_label) for class_label in class_labels}


class ClassDataset(CustomDataset):

    def __init__(self, samples: list[Sample]):
        super(ClassDataset, self).__init__(samples)
        [self.class_label] = self.number_of_samples_per_class.keys()


class ClientDataset(CustomDataset):

    def __init__(self, samples: list[Sample]):
        super(ClientDataset, self).__init__(samples)


class TestDataset(CustomDataset):

    def __init__(self, samples: list[Sample]):
        super(TestDataset, self).__init__(samples)

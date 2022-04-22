from __future__ import annotations

from typing import Union, Generator, Tuple
from enum import Enum
from abc import ABC

from numpy import array
from keras.utils import to_categorical

from data_provider.models import ClassLabel, Sample
from utilities.utils import get_shuffled_data


class CutOffSide(Enum):
    FIRST = 'first'
    LAST = 'last'


class CustomDataset(ABC):

    def __init__(self, samples: list[Sample]):
        self.samples = get_shuffled_data(samples)
        self.number_of_samples_per_class = self.__count_samples_per_class(self.samples)
        self.input_shape = self.__get_input_shape(self.samples)
        self.input_values, self.target_labels = self.__get_model_inputs(self.samples)

    def __getitem__(self, key: int) -> Union[Sample, list[Sample]]:
        if isinstance(key, int):
            output = self.samples[key]
        elif isinstance(key, slice):
            start = key.start if key.start else 0
            stop = key.stop
            step = key.step
            output = self.samples[start: stop: step]
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
    def __get_input_shape(samples: list[Sample]) -> Tuple[int, int, int]:
        return samples[0].value.shape

    @staticmethod
    def __count_samples_per_class(samples: list[Sample]) -> dict[ClassLabel, int]:
        class_labels = list(set([sample.class_label for sample in samples]))
        sample_class_labels = [sample.class_label for sample in samples]

        return {class_label: sample_class_labels.count(class_label) for class_label in class_labels}

    @staticmethod
    def __get_model_inputs(samples: list[Sample]) -> tuple[array, array]:
        values, labels = [
            array(model_inputs) for model_inputs in zip(*[(sample.value, sample.class_label.id) for sample in samples])
        ]

        return array(values), array(labels)


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

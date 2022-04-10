from typing import Union, Generator
from enum import Enum
from abc import ABC

from tensorflow_datasets.core.dataset_info import DatasetInfo
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.data import Dataset

from models import SampleClass, Sample


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

    def __getitem__(self, key: Union[int, str, slice]) -> Union[SampleClass, list[SampleClass]]:
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

    def __getitem__(self, key: int) -> Union[EagerTensor, list[EagerTensor]]:
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

    def __len__(self) -> int:
        return len(self.samples)

    def truncate(self, number: int, where: TruncatePoint = TruncatePoint.FIRST):
        if where == TruncatePoint.FIRST:
            self.samples = self.samples[number:]
        elif where == TruncatePoint.LAST:
            self.samples = self.samples[:-number]
        else:
            raise TypeError('You can only truncate first or last samples from dataset!')

        self.data = self.__get_data_from_samples(self.samples)

    @staticmethod
    def __get_data_from_samples(samples: list[Sample]) -> Dataset:
        return Dataset.from_tensor_slices([sample.value for sample in samples])


class ClassDataset(CustomDataset):

    def __init__(self, class_: SampleClass, samples: list[EagerTensor]):
        super(ClassDataset, self).__init__(samples)
        self.class_ = class_


class ClientDataset(CustomDataset):

    def __init__(self, samples: list[EagerTensor]):
        super(ClientDataset, self).__init__(samples)

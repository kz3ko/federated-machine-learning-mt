from typing import Union, Generator

from tensorflow_datasets.core.dataset_info import DatasetInfo

from data_provider.models import ClassLabel


class DatasetClassLabels:

    def __init__(self, dataset_info: DatasetInfo):
        self.class_labels = self.__get_dataset_class_labels(dataset_info)

    def __repr__(self) -> str:
        return str(self.class_labels)

    def __len__(self) -> int:
        return len(self.class_labels)

    def __iter__(self) -> Generator[ClassLabel, None, None]:
        for class_label in self.class_labels:
            yield class_label

    def __getitem__(self, key: Union[int, str, slice]) -> Union[ClassLabel, list[ClassLabel]]:
        if isinstance(key, int):
            try:
                [output] = filter(lambda class_label: class_label.id == key, self.class_labels)
            except ValueError:
                raise IndexError(f'There is no class label with {key} id!')
        elif isinstance(key, str):
            try:
                [output] = filter(lambda class_label: class_label.name == key, self.class_labels)
            except ValueError:
                raise NameError(f'There is no class with label {key} name!')
        elif isinstance(key, slice):
            start = key.start if key.start else 0
            stop = key.stop
            step = key.step if key.step else 1
            output = self.class_labels[start: stop: step]
        else:
            raise TypeError(f'{self.__class__.__name__} indices must be "int", "str" or "slice", not {type(key)}.')

        return output

    @staticmethod
    def __get_dataset_class_labels(dataset_info: DatasetInfo) -> list[ClassLabel]:
        return [ClassLabel(dataset_info.features['label'].str2int(name), name) for name in
                dataset_info.features['label'].names]

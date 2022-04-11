from typing import Optional

from tensorflow_datasets import load
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset
from tensorflow_datasets.core.dataset_info import DatasetInfo


class DatasetLoader:

    @staticmethod
    def load(name: str, *args, **kwargs) -> tuple[PrefetchDataset, Optional[DatasetInfo]]:
        return load(name, *args, **kwargs)

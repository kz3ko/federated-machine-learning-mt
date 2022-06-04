from typing import Optional

from config.config import DatasetConfig

from tensorflow_datasets import load
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset
from tensorflow_datasets.core.dataset_info import DatasetInfo


class DatasetLoader:

    def __init__(self, config: DatasetConfig):
        self.name = config.name

    def load_dataset(self, *args, **kwargs) -> tuple[PrefetchDataset, Optional[DatasetInfo]]:
        return load(self.name, *args, **kwargs)

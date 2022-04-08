from dataclasses import dataclass

from tensorflow.python.framework.ops import EagerTensor


@dataclass(unsafe_hash=True)
class ClassLabel:
    id: int
    name: str


@dataclass(unsafe_hash=True)
class Sample:
    value: EagerTensor
    class_label: ClassLabel

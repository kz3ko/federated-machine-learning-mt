from typing import Union, Any

from abc import ABC
from dataclasses import dataclass, field, asdict


@dataclass
class Metrics(ABC):

    def __post_init__(self):
        self.fields_to_ignore = []

    def as_dict(self) -> dict[Any]:
        metrics_as_dict = asdict(self)
        for field_ in self.fields_to_ignore:
            metrics_as_dict.pop(field_)

        return metrics_as_dict


@dataclass
class ParticipantMetrics(Metrics, ABC):

    id: Union[str, int]
    full_name: str = field(init=False)
    iterations: list[int] = field(default_factory=lambda: [])
    accuracy: list[float] = field(default_factory=lambda: [])
    loss: list[float] = field(default_factory=lambda: [])

    def __post_init__(self):
        super().__post_init__()
        self.fields_to_ignore.extend(['id', 'full_name'])


@dataclass
class ClientMetrics(ParticipantMetrics):

    val_accuracy: list[float] = field(default_factory=lambda: [])
    val_loss: list[float] = field(default_factory=lambda: [])

    def __post_init__(self):
        super().__post_init__()
        self.full_name = f'client_{self.id}'


@dataclass
class ServerMetrics(ParticipantMetrics):

    def __post_init__(self):
        super().__post_init__()
        self.full_name = self.id


@dataclass
class ParticipantBestMetrics(Metrics, ABC):

    name: str = field(repr=False)
    max_accuracy: float
    max_accuracy_iteration: int
    min_loss: float
    min_loss_iteration: int
    full_name: str = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.full_name = f'{self.name}_best_metrics'
        self.fields_to_ignore.extend(['full_name', 'name'])


@dataclass
class ClientBestMetrics(ParticipantBestMetrics):

    max_val_accuracy: float
    max_val_accuracy_iteration: int
    min_val_loss: float
    min_val_loss_iteration: int


@dataclass
class ServerBestMetrics(ParticipantBestMetrics):

    pass

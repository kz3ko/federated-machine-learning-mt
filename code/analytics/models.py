from typing import Union, Any

from abc import ABC
from dataclasses import dataclass, field, asdict


@dataclass
class ParticipantMetrics(ABC):
    id: Union[str, int]
    full_name: str = field(init=False)
    iterations: list[int] = field(default_factory=lambda: [])
    accuracy: list[float] = field(default_factory=lambda: [])
    loss: list[float] = field(default_factory=lambda: [])

    def as_dict(self) -> dict[Any]:
        metrics_as_dict = asdict(self)
        fields_to_ignore = ['id', 'full_name']
        for field_ in fields_to_ignore:
            metrics_as_dict.pop(field_)

        return metrics_as_dict


@dataclass
class ClientMetrics(ParticipantMetrics):
    val_accuracy: dict[int, float] = field(default_factory=lambda: [])
    val_loss: dict[int, float] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.full_name = f'client_{self.id}'


@dataclass
class ServerMetrics(ParticipantMetrics):

    def __post_init__(self):
        self.full_name = self.id

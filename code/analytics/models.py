from typing import Union, Any

from abc import ABC
from dataclasses import dataclass, field, asdict


@dataclass
class ParticipantMetrics(ABC):
    id: Union[str, int]
    full_name: str = field(init=False)
    accuracy: dict[int, float] = field(default_factory=lambda: {})
    loss: dict[int, float] = field(default_factory=lambda: {})

    def as_dict(self) -> dict[Any]:
        metrics_as_dict = asdict(self)
        unnecessary_keys = ['id', 'full_name']
        for key in unnecessary_keys:
            metrics_as_dict.pop(key)

        return metrics_as_dict


@dataclass
class ClientMetrics(ParticipantMetrics):
    val_accuracy: dict[int, float] = field(default_factory=lambda: {})
    val_loss: dict[int, float] = field(default_factory=lambda: {})

    def __post_init__(self):
        self.full_name = f'client_{self.id}'


@dataclass
class ServerMetrics(ParticipantMetrics):

    def __post_init__(self):
        self.full_name = self.id

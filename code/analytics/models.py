from abc import ABC
from dataclasses import dataclass, field


@dataclass
class ParticipantMetrics(ABC):
    accuracy: dict[int, float] = field(default_factory=lambda: {})
    loss: dict[int, float] = field(default_factory=lambda: {})


@dataclass
class ClientMetrics(ParticipantMetrics):
    val_accuracy: dict[int, float] = field(default_factory=lambda: {})
    val_loss: dict[int, float] = field(default_factory=lambda: {})


@dataclass
class ServerMetrics(ParticipantMetrics):
    pass

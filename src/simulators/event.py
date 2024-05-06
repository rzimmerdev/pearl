from abc import ABC, abstractmethod
from dataclasses import dataclass
import dxlib as dx

from src.lob import LOB


@dataclass
class MarketEvent:
    interval: float
    signal: dx.Signal


class EventSampler(ABC):
    @abstractmethod
    def __call__(self, lob: LOB, *args, **kwargs) -> MarketEvent:
        pass


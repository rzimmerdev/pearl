from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import TypeVar, Generic, Type

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Layer initialization helper.
    """
    nn.init.xavier_uniform_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


@dataclass()
class ParamDataclass(ABC):
    a = 2


def paramclass(cls: Type) -> Type:
    if not issubclass(cls, ParamDataclass):
        cls = type(cls.__name__, (ParamDataclass, cls), dict(cls.__dict__))
    return dataclass(cls)


class Parametric(type):
    def __new__(cls, name, bases, dct, param_class=None):
        dct["_params_class"] = param_class
        return super().__new__(cls, name, bases, dct)


P = TypeVar("P", bound=ParamDataclass)


class Model(nn.Module, Generic[P], metaclass=Parametric):
    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def from_params(cls, params: P):
        return cls(**params.__dict__)

    @abstractmethod
    def params(self) -> P:
        pass

    def checkpoint(self):
        return {
            "state_dict": self.state_dict(),
            "params": asdict(self.params())
        }

    def save(self):
        torch.save(self.checkpoint(), self.path)

    def load(self, checkpoint):
        """
        Load should never handle params, only weights
        """
        if not "state_dict" in checkpoint:
            raise ValueError("Checkpoint does not contain state_dict")
        self.load_state_dict(checkpoint["state_dict"])

    @classmethod
    def from_checkpoint(cls, checkpoint):
        if not "params" in checkpoint:
            raise ValueError("Checkpoint does not contain params")
        if not "_params_class" in cls.__dict__:
            raise ValueError("Model does not have a params class")

        params = cls._params_class(**checkpoint["params"])
        model = cls.from_params(params)
        model.load_state_dict(checkpoint["state_dict"])
        return model

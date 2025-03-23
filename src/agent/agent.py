from abc import ABC, abstractmethod
from typing import Dict

from src.agent.models.model import Model


def default_reshape(action):
    return action


class Agent(ABC):
    def __init__(self):
        self.hyperparameters = {}

    def weights(self) -> Dict[str, dict]:
        """
        Compact method for getting current model weights, as a dictionary.

        Returns:

        """
        return {name: model.state_dict() for name, model in self.models.items()}

    def parameters(self):
        # return list(self.policy_network.parameters()) + list(self.value_network.parameters())
        # separate learning rates for policy and value networks
        return [
            {'params': model.parameters(), 'lr': self.hyperparameters.get('lr', 1e-3)}
            for model in self.models.values()
        ]

    @property
    @abstractmethod
    def models(self) -> Dict[str, Model]:
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def update(self, trajectories):
        pass

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)

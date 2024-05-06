import numpy as np

from .hmm import HMM
from ...lob.simulator import LOBSimulator


class Sampler:
    def __call__(self, *args, **kwargs):
        pass


class HMMSampler(Sampler):
    def __init__(self, hidden_states: int, space: dict = None):
        self.space = space or {
            "price": np.linspace(0, 1, 100),
            "quantity": np.linspace(0, 1, 100),
            "type": [-2, 2],
            "interval": np.linspace(0, 50, 1000),
        }
        self.hmm = HMM(hidden_states, self.space)

    def __call__(self, *args, **kwargs):
        pass


class HMMSimulator(LOBSimulator):
    def __init__(self, hidden_states: int, space=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = HMMSampler(hidden_states, space)

    def simulate(self):
        pass


def main():
    # using dash to plot realtime event times:
    import dash
    from dash import dcc

    # create a HMM object
    hmm = HMM(2, np.linspace(0, 1, 100), 1)

    # create a HMM simulator
    hmm_simulator = HMMSimulator(hmm, 1000, 1000)
    hmm_simulator.run(10)


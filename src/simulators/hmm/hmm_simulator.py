import numpy as np

from .hmm import HMM
from ...lob import LOBSimulator, EventSampler, ArrivalSampler


class HMMArrivalSampler(ArrivalSampler):
    def __init__(self, hmm: HMM, *args, **kwargs):
        self.hmm = hmm

    def __call__(self, state, *args, **kwargs) -> float:
        state = self.hmm.observation_index(state)
        observation = np.random.choice(self.hmm.observation_space, p=self.hmm.emission_matrix[state])
        return observation


class HMMEventSampler(EventSampler):
    def __call__(self, *args, **kwargs):
        pass


class HMMSimulator(LOBSimulator):
    def __init__(self, hmm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hmm = hmm
        self.state = np.random.choice(np.arange(self.hmm.hidden_states))
        self.arrival_sampler = HMMArrivalSampler(self.hmm)
        self.event_sampler = HMMEventSampler()

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

    # run the simulation


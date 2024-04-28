import numpy as np
import dxlib as dx

from .hmm import HMM
from ...lob.simulator import LOBSimulator, EventSampler, ArrivalSampler


class HMMArrivalSampler(ArrivalSampler):
    def __init__(self, hmm: HMM, *args, **kwargs):
        self.hmm = hmm
        self.current_state = np.random.choice(np.arange(self.hmm.hidden_states))

    def __call__(self, *args, **kwargs) -> float:
        self.current_state = np.random.choice(np.arange(self.hmm.hidden_states),
                                              p=self.hmm.transition_matrix[self.current_state])
        observation = np.random.choice(self.hmm.observation_space, p=self.hmm.emission_matrix[self.current_state])
        return observation


class HMMEventSampler(EventSampler):
    def __call__(self, *args, **kwargs) -> dx.Signal:
        return dx.Signal(
            side=dx.Side.BUY,
            price=1.0,
            quantity=1.0,
        )


class HMMSimulator(LOBSimulator):
    def __init__(self, hmm, *args, **kwargs):
        self.hmm = hmm
        self.state = np.random.choice(np.arange(self.hmm.hidden_states))
        self.arrival_sampler = HMMArrivalSampler(self.hmm)
        self.event_sampler = HMMEventSampler()

        super().__init__(self.arrival_sampler, self.event_sampler, *args, **kwargs)

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


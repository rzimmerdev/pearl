from .hmm import HMM
from ...lob import LOBSimulator, EventSampler, ArrivalSampler


class HMMArrivalSampler(ArrivalSampler):
    def __init__(self, hmm: HMM, *args, **kwargs):
        self.hmm = hmm

    def __call__(self, *args, **kwargs):
        pass


class HMMEventSampler(EventSampler):
    def __call__(self, *args, **kwargs):
        pass


class HMMSimulator(LOBSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

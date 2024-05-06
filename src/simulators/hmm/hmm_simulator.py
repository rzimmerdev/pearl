import numpy as np
import dxlib as dx

from .hmm import HMM
from ..event import MarketEvent, EventSampler
from ..simulator import LOBSimulator
from ...lob import LOB


class HMMSampler(EventSampler):
    def __init__(self, hidden_states: int, space: dict = None):
        self.space = space or {
            "spread": np.linspace(-10, 10, 1000),
            "quantity": np.linspace(0, 1, 1000),
            "side": np.array([-1, 1]),
            "interval": np.linspace(0, 50, 1000),
        }
        self.hmm = HMM(hidden_states, self.space)

    def __call__(self, lob: LOB, *args, **kwargs) -> MarketEvent:
        df = self.hmm.mixture_sampling(1)
        event = df.iloc[0]
        spread = event["spread"]
        price = np.maximum(lob.mid_price + spread, 0)
        print(lob.mid_price, spread, price)
        signal = dx.Signal(
            side=dx.Side(event["side"]),
            quantity=event["quantity"],
            price=price,
        )
        interval = event["interval"]

        return MarketEvent(interval, signal)


class HMMSimulator(LOBSimulator):
    def __init__(self, lob, hidden_states: int, space=None, *args, **kwargs):
        super().__init__(lob, HMMSampler(hidden_states, space), *args, **kwargs)

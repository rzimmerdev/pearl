from abc import ABC
from typing import List, Tuple

import dxlib as dx
import asyncio

from .lob import LOB


class EventSampler(ABC):
    def __call__(self, *args, **kwargs) -> dx.Signal:
        pass


class ArrivalSampler(ABC):
    def __call__(self, *args, **kwargs) -> float:
        pass


class TestEventSampler(EventSampler):
    def __call__(self, *args, **kwargs) -> dx.Signal:
        return dx.Signal(
            side=dx.Side.BUY,
            price=1.0,
            quantity=1.0,
        )


class TestArrivalSampler(ArrivalSampler):
    def __call__(self, *args, **kwargs) -> float:
        return 1.0


class LOBSimulator:
    def __init__(
        self,
        event_sampler: EventSampler,
        arrival_sampler: callable,
        logger: dx.LoggerMixin = None,
    ):
        self.lob = LOB()
        self.t = 0
        self.logger = logger or dx.InfoLogger()

        self.event_sampler = event_sampler
        self.arrival_sampler = arrival_sampler
        self.events: List[Tuple[float, dx.Signal]] = []

    async def _step(self):
        try:
            # sample arrival time and wait
            arrival_time = self.arrival_sampler()
            await asyncio.sleep(arrival_time)

            # sample event and register
            event = self.event_sampler()
            t = self.register(event, arrival_time)

            self.logger.info(f"Event {event} registered at time {t}")
            return t, event
        except asyncio.CancelledError:
            return None

    def register(self, event: dx.Signal, time: float):
        self.t += time
        self.events.append((time, event))
        self.lob.send(event)
        self.lob.aggregate()
        return self.t

    async def _run(self, T: float = None):
        while self.t < T:
            await self._step()

    def run(self, T: float = 10.0):
        self.logger.info("Starting simulation")
        asyncio.run(self._run(T))
        self.logger.info("Simulation finished")
        return self.lob

from dataclasses import dataclass

import dxlib as dx
import asyncio

from .lob import LOB


@dataclass
class MarketEvent:
    time: float
    signal: dx.Signal


class EventSampler:
    def __call__(self, *args, **kwargs) -> MarketEvent:
        pass


class LOBSimulator:
    def __init__(
        self,
        event_sampler: EventSampler,
        logger: dx.LoggerMixin = None,
    ):
        self.lob = LOB()
        self.t = 0
        self.logger = logger or dx.InfoLogger()

        self.event_sampler = event_sampler

    async def _step(self, wait=False):
        try:
            event = self.event_sampler()
            if wait:
                await asyncio.sleep(event.time)
            self.register(event)
            self.logger.info(f"Event {event} registered at time {event.time}")
        except asyncio.CancelledError:
            return None

    def register(self, event: MarketEvent):
        signal, time = event.signal, event.time
        self.t += time
        self.lob.send(event.signal)
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

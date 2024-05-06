import dxlib as dx
import asyncio

from ..lob import LOB
from .event import MarketEvent, EventSampler


class LOBSimulator:
    def __init__(
        self,
        lob: LOB,
        event_sampler: EventSampler,
        logger: dx.LoggerMixin = None,
    ):
        self.t = 0
        self.lob = lob
        self.logger = logger or dx.InfoLogger()

        self.event_sampler = event_sampler

    async def _step(self, wait=False):
        try:
            event = self.event_sampler(self.lob)
            if wait:
                await asyncio.sleep(event.interval)
            self.register(event)
            self.logger.info(f"Event {event} registered at time {event.interval}")
        except asyncio.CancelledError:
            return None

    def register(self, event: MarketEvent):
        signal, interval = event.signal, event.interval
        self.t += interval
        self.lob.send(event.signal)

        return self.t

    async def _run(self, T: float = None):
        while self.t < T:
            await self._step()

    def run(self, T):
        self.logger.info("Starting simulation")
        asyncio.run(self._run(T))
        self.logger.info("Simulation finished")
        return self.lob

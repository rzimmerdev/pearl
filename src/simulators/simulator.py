import dxlib as dx
import asyncio

import tqdm

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
            self.logger.debug(f"Event {event} registered at time {event.interval}")

            return event
        except asyncio.CancelledError:
            return None

    def register(self, event: MarketEvent):
        signal, interval = event.signal, event.interval
        self.t += interval
        self.lob.send(event.signal)
        self.lob.aggregate()

        return self.t

    async def _run(self, T, wait=False, d: float = None):
        if d is None:
            d = T // 10
        last_snapshot = 0
        snapshots = [self.lob.copy()]

        if d == 0:
            d = 1

        for _ in tqdm.tqdm(range(int(T // d))):
            await self._step(wait=wait)

            if self.t - last_snapshot >= d:
                snapshots.append(self.lob.copy())
                last_snapshot = self.t

        return snapshots

    def run(self, T: float, wait: bool = False, d: float = None):
        self.logger.info("Starting simulation")
        snapshots = asyncio.run(self._run(T, wait, d))
        self.logger.info("Simulation finished")
        return snapshots

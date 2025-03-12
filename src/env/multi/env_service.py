import asyncio
import threading
import time
from collections import defaultdict

from src.env.lib import Router
from src.env.multi.multi_env import MarketEnv


class Timer:
    def __init__(self, timeout):
        """Async timer to decrease time clock and call a callback function when time is up"""
        self.timeout = timeout
        self.start_time = time.time()

    def is_timeout(self):
        return time.time() - self.start_time > self.timeout

    def reset(self):
        self.start_time = time.time()

    # method to start the timer
    async def _start(self, callback, *args, **kwargs):
        while not self.is_timeout():
            await asyncio.sleep(1)

        await callback(*args, **kwargs)

    # method to run
    async def run(self, callback, *args, **kwargs):
        await self._start(callback, *args, **kwargs)

    def start(self, callback, *args, **kwargs):
        while True:
            time.sleep(self.timeout)
            callback(*args, **kwargs)


class MarketEnvService(MarketEnv):
    def __init__(self, host, port, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = Router(host, port)
        self.action_buffer = defaultdict(list)
        self.timer = Timer(1.0)
        self.connections = set()

    def _step(self, actions):
        self.step(actions)

    def handle(self, uuid, msg):
        # decide based on msg:
        # {"type": "action", "data": [0.1, 0.2, 0.3, 0.4]}
        # {"type": "close"}
        # {"type": "connect"}
        if msg["type"] == "close":
            self.connections.remove(uuid)
            return
        elif msg["type"] == "connect":
            self.connections.add(uuid)

        if msg["type"] != "action":
            return

        self.action_buffer[uuid].append(msg)

        if len(self.connections) == self.n_agents:
            self.execute_step()

    def execute_step(self):
        self._step(self.action_buffer)
        self.action_buffer.clear()

    def start(self):
        """Run router async and timer async"""
        self.router.bind()
        self.router.use_mesh("market", "localhost", 8000)

        # router.start and timer.start on separate threads
        t_service = threading.Thread(target=self.router.start, args=(self.handle,))
        t_timer = threading.Thread(target=self.timer.start, args=(self.execute_step,))

        # wait for both threads to finish
        t_service.start()
        print("Router thread started")
        t_timer.start()
        print("Timer thread started")

        t_service.join()
        print("Router thread finished")
        t_timer.join()
        print("Timer thread finished")


if __name__ == "__main__":
    env = MarketEnvService("localhost", 5000, n_agents=2, n_levels=10, starting_value=100)
    env.start()
    try:
        asyncio.run(asyncio.sleep(10))
    except KeyboardInterrupt:
        pass
    finally:
        env.router.remove_mesh()
        env.router.close()

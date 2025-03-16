import asyncio
import threading
import time


class Clock:
    def __init__(self, timeout):
        """Async timer to decrease time clock and call a callback function when time is up"""
        self.timeout = timeout
        self.start_time = time.time()
        self.thread = None
        self.running = False

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

    def start(self, callback, threaded=True, *args, **kwargs):
        def _start():
            while self.running:
                if self.is_timeout():
                    callback(*args, **kwargs)
                    self.reset()
                else:
                    time.sleep(1)

        self.running = True
        if threaded:
            self.thread = threading.Thread(target=_start)
            self.thread.start()
            return self.thread
        else:
            return _start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
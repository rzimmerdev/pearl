import asyncio
import time

class Timer:
    def __init__(self, timeout):
        """Async timer to decrease time clock and call a callback function when time is up."""
        self.timeout = timeout
        self.start_time = None
        self.running = False

    def is_timeout(self):
        """Check if the timeout has been reached."""
        return time.time() - self.start_time > self.timeout

    def reset(self):
        """Reset the timer to the current time."""
        self.start_time = time.time()

    async def start(self, callback, *args, **kwargs):
        """Start the timer and wait for the timeout to call the callback."""
        self.running = True
        self.reset()

        while self.running and not self.is_timeout():
            await asyncio.sleep(0.1)

        if self.running:
            await callback(*args, **kwargs)
        self.running = False

    def stop(self):
        """Stop the timer."""
        self.running = False

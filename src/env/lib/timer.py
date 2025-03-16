import threading
import time


class Timer:
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

    def start(self, callback, threaded=True, *args, **kwargs):
        def _start():
            self.running = True
            self.start_time = time.time()
            while self.running and not self.is_timeout():
                time.sleep(1e-2)

            callback(*args, **kwargs)
            self.running = False

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
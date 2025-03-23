import time
from collections import defaultdict
from dxlib.common import Singleton


class Runtime:
    def __init__(self, err=1e-5):
        self._timers = defaultdict()
        self._start = time.perf_counter()  # Use perf_counter for higher precision
        self._error = err

    def start(self, name):
        self._timers[name] = {
            "start": time.perf_counter() - self._error,  # Use perf_counter here as well
            "avg": 0,
            "count": 0
        }

    def stop(self, name):
        diff = time.perf_counter() - self._timers[name]["start"] - self._error
        self._timers[name]["avg"] = self._timers[name]["avg"] * self._timers[name]["count"] + diff
        self._timers[name]["count"] += 1
        self._timers[name]["avg"] /= self._timers[name]["count"]

    def get(self, name):
        return self._timers[name]["avg"]

    def names(self):
        return self._timers.keys()

class Benchmark(metaclass=Singleton):
    def __init__(self):
        super().__init__()
        self.err = time.perf_counter() - time.perf_counter()
        self._benchmarks = defaultdict()

    def _create(self, group):
        self._benchmarks[group] = Runtime(self.err)

    def start(self, group, name):
        if group not in self._benchmarks:
            self._create(group)
        self._benchmarks[group].start(name)

    def stop(self, group, name):
        self._benchmarks[group].stop(name)

    def get(self, group, name):
        return self._benchmarks[group].get(name)

    @staticmethod
    def measure(method, *args, **kwargs):
        start = time.perf_counter()
        method(*args, **kwargs)
        return time.perf_counter() - start

    def summary(self):
        summary = {}
        for group, runtime in self._benchmarks.items():
            summary[group] = {name: runtime.get(name) for name in runtime.names()}
        return summary


# Example usage
if __name__ == "__main__":
    benchmark = Benchmark()
    # note: time.sleep() has additional overhead due to internal clock synchronization, so the error is higher
    sleep_error = Benchmark.measure(time.sleep, 1) - 1
    print(sleep_error)  # 0.0001
    benchmark.start("group1", "timer1")
    time.sleep(1 - sleep_error)
    benchmark.stop("group1", "timer1")
    print(round(benchmark.get("group1", "timer1"), 5))  # 1.0
    benchmark.start("group2", "timer2")
    time.sleep(2 - sleep_error)
    benchmark.stop("group2", "timer2")
    print(round(benchmark.get("group2", "timer2"), 5))  # 2.0
    benchmark.start("group1", "timer1")
    time.sleep(3 - sleep_error)
    benchmark.stop("group1", "timer1")
    print(round(benchmark.get("group1", "timer1"), 5))  # 2.0

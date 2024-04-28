from lob.simulator import *


def main():
    simulator = LOBSimulator(
        event_sampler=TestEventSampler(),
        arrival_sampler=TestArrivalSampler(),
    )
    simulator.run()
    simulator.lob.plot()


if __name__ == "__main__":
    main()

import numpy as np

from src.simulators import HMM, HMMSimulator


def main():
    simulator = HMMSimulator(
        3,
    )
    simulator.run()
    simulator.lob.plot()


if __name__ == "__main__":
    main()

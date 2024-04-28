import numpy as np

from src.simulators import HMM, HMMSimulator


def main():
    hmm = HMM(2, np.linspace(0, 1, 100), 1)

    simulator = HMMSimulator(
        hmm
    )
    simulator.run()
    simulator.lob.plot()


if __name__ == "__main__":
    main()

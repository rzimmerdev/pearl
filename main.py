import dxlib as dx

from src.lob import LOB
from src.simulators import HMMSimulator


def main():
    lob = LOB()

    lob.send(dx.Signal(dx.Side.BUY, 1, 90))
    lob.send(dx.Signal(dx.Side.BUY, 1, 100))
    lob.send(dx.Signal(dx.Side.SELL, 1, 110))
    lob.send(dx.Signal(dx.Side.SELL, 1, 115))

    simulator = HMMSimulator(
        lob,
        3,
    )
    simulator.run(100)
    simulator.lob.plot()


if __name__ == "__main__":
    main()

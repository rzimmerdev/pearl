from typing import Iterable

import dxlib as dx

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist

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
    snapshots = simulator.run(4)

    # fig, ax = plt.subplots()
    # set plot size
    fig, ax = plt.subplots(figsize=(16, 8))
    axes = []

    def update(frame) -> Iterable[Artist]:
        ax.clear()  # Clear the previous plot
        axes[frame] = snapshots[frame].plot(ax=ax)
        return axes[frame],

    for snapshot in snapshots:
        ax = snapshot.plot(ax=ax)
        axes.append(ax)

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000)
    anim.save('animation.gif', writer='imagemagick')  # Save the animation as a GIF using imagemagick writer

    ax = snapshots[-1].plot()
    # save the plot as a png file
    plt.savefig('plot.png')


if __name__ == "__main__":
    main()

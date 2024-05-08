import dxlib as dx
import numpy as np
import matplotlib.pyplot as plt


class LOB:
    def __init__(self):
        self.bids = []
        self.asks = []

    def send(self, order: dx.Signal, order_type: dx.OrderType = dx.OrderType.LIMIT):
        if order_type == dx.OrderType.MARKET:
            self.send_market(order)
        if order.side == dx.Side.BUY:
            self.bids.append(order)
        elif order.side == dx.Side.SELL:
            self.asks.append(order)

    def aggregate(self):
        # make orders with the same price into one
        for i in range(len(self.bids)):
            for j in range(i + 1, len(self.bids)):
                if self.bids[i].price == self.bids[j].price:
                    self.bids[i].quantity += self.bids[j].quantity
                    self.bids.pop(j)

    def send_market(self, order: dx.Signal):
        if order.side == dx.Side.BUY:
            while order.quantity > 0:
                ask = self.asks[0]
                if ask.quantity > order.quantity:
                    ask.quantity -= order.quantity
                    order.quantity = 0
                else:
                    order.quantity -= ask.quantity
                    self.asks.pop(0)
        else:
            while order.quantity > 0:
                bid = self.bids[0]
                if bid.quantity > order.quantity:
                    bid.quantity -= order.quantity
                    order.quantity = 0
                else:
                    order.quantity -= bid.quantity
                    self.bids.pop(0)

    @property
    def mid_price(self):
        return (self.asks[0].price + self.bids[0].price) / 2 if self.asks and self.bids else None

    def plot(self, depth: int = None, ax=None) -> plt.Axes:
        depth_label = f"{depth} levels"
        if depth is None:
            depth = np.max([len(self.bids), len(self.asks)])
            depth_label = "Full"

        bids = sorted(self.bids, key=lambda x: x.price, reverse=True)[:depth]
        bid_prices = [x.price for x in bids]
        bid_volumes = np.array([x.quantity for x in bids])
        bid_volumes = np.cumsum(bid_volumes)

        asks = sorted(self.asks, key=lambda x: x.price, reverse=True)[:depth]
        ask_prices = [x.price for x in asks]
        ask_volumes = np.array([x.quantity for x in asks])
        ask_volumes = np.cumsum(ask_volumes[::-1])[::-1]

        if ax is None:
            _, ax = plt.subplots()

        ax.bar([f"{x:.2f}" for x in ask_prices], ask_volumes, color='#BDE4C7', label='Asks')
        ax.bar([f"{x:.2f}" for x in bid_prices], bid_volumes, color='#FDBFC0', label='Bids')

        ax.set_xlabel('Price')
        ax.set_ylabel('Volume')
        ax.set_title(f'Limit Order Book, depth={depth_label}')
        ax.legend()

        return ax

    def copy(self):
        new_lob = LOB()
        new_lob.bids = self.bids.copy()
        new_lob.asks = self.asks.copy()
        return new_lob


if __name__ == "__main__":
    lob = LOB()
    lob.send(dx.Signal(dx.Side.BUY, 10, 100))
    lob.send(dx.Signal(dx.Side.BUY, 10, 101))
    lob.send(dx.Signal(dx.Side.SELL, 10, 102))
    lob.send(dx.Signal(dx.Side.SELL, 10, 101), dx.OrderType.MARKET)
    lob.plot()

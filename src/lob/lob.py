import dxlib as dx
import plotly.graph_objects as go


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

    def plot(self, depth: int = None):
        depth_label = f"{depth} levels"
        if depth is None:
            depth = len(self.asks)
            depth_label = "Full"
        ask_prices = [x.price for x in self.asks][:depth]
        ask_volumes = [x.quantity for x in self.asks][:depth]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=["{:.4f}".format(x) for x in ask_prices],
                x=ask_volumes,
                orientation="h",
                marker=dict(
                    color="Gray",
                ),
                name="Asks",
            )
        )
        bid_prices = [x.price for x in self.bids]
        bid_volumes = [x.quantity for x in self.bids]
        fig.add_trace(
            go.Bar(
                y=["{:.4f}".format(x) for x in bid_prices],
                x=bid_volumes,
                orientation="h",
                marker=dict(
                    color="IndianRed",
                ),
                name="Bids",
            )
        )
        fig.update_layout(
            barmode="stack",
            xaxis_title="Volume",
            yaxis_title="Price",
            title="Limit Order Book, depth={}".format(depth_label),
        )
        fig.show()


if __name__ == "__main__":
    lob = LOB()
    lob.send(dx.Signal(dx.Side.BUY, 10, 100))
    lob.send(dx.Signal(dx.Side.BUY, 10, 101))
    lob.send(dx.Signal(dx.Side.SELL, 10, 102))
    lob.send(dx.Signal(dx.Side.SELL, 10, 101), dx.OrderType.MARKET)
    lob.plot()

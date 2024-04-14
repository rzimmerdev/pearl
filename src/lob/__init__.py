import pandas as pd
import jax.numpy as jnp
import jax
import plotly.graph_objects as go


class LOB:
    def __init__(self, levels=100):
        """deals with historical orders and trades"""
        self.orders = []
        self.trades = []

    def insert(self, orders):
        """add orders unsorted"""
        self.orders.append(orders)

    def sort(self):
        """sorts orders by price, stable sort"""
        self.orders = sorted(self.orders, key=lambda x: x["price"])

    def match(self):
        """matches orders, returns trades"""
        trades = []
        for i in range(len(self.orders) - 1):
            if self.orders[i]["type"] != self.orders[i + 1]["type"]:
                if self.orders[i]["price"] == self.orders[i + 1]["price"]:
                    trade = {
                        "time": self.orders[i]["time"],
                        "price": self.orders[i]["price"],
                        "quantity": min(self.orders[i]["quantity"], self.orders[i + 1]["quantity"]),
                    }
                    trades.append(trade)
        return trades

    def __repr__(self):
        return f"LOB with {len(self.orders)} orders and {len(self.trades)} trades"

    def plot(self, t=None):
        """plot order book at time t"""
        bids = []
        asks = []

        if t is None:
            t = self.orders[-1]["time"]

        for order in self.orders:
            if t and order["time"] > t:
                break
            if order["type"]:
                bids.append(order)
            else:
                asks.append(order)

        bids = pd.DataFrame(bids)
        asks = pd.DataFrame(asks)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bids["price"], y=bids["quantity"], mode="markers", name="bids"))
        fig.add_trace(go.Scatter(x=asks["price"], y=asks["quantity"], mode="markers", name="asks"))

        fig.show()


def main():

    key = jax.random.PRNGKey(0)

    events = 10

    # spreads are poisson, and spread is poisson process
    ask_lambda = 1
    bid_lambda = 1
    ask_distribution = jax.random.poisson(key, shape=(events,), lam=ask_lambda)
    bid_distribution = jax.random.poisson(key, shape=(events,), lam=bid_lambda)

    # mid price increment are normal, and mid price is weiner process
    mid_price = 100
    mid_price = jnp.cumsum(jax.random.normal(key, shape=(events,))) + mid_price

    # events times are sampled continuously from continuous increment process
    increments = jax.random.exponential(key, shape=(events,))
    event_times = jnp.cumsum(increments)

    # order types are sampled from uniform distribution
    order_types = jax.random.uniform(key, shape=(events,)) > 0.5

    # order quantities are sampled from normal distribution
    order_quantities = jax.random.normal(key, shape=(events,))

    # build order book
    lob = LOB()

    # for each event time, create new order of given type, at specific price
    print(event_times)
    for i, event_time in enumerate(event_times):
        order = {
            "time": event_time,
            "price": mid_price[i],
            "quantity": order_quantities[i],
            "type": order_types[i],
        }
        lob.insert(order)

    print(lob)
    lob.plot()


if __name__ == "__main__":
    main()

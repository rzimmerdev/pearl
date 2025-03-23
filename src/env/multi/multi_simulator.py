from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
from dxlib.orders import LimitOrderBook, Order, Transaction
from dxlib.dynamics import GeometricBrownianMotion, CoxIngersollRoss, OrnsteinUhlenbeck, Hawkes


@dataclass
class MarketVariables:
    last_transaction: float
    quotes: List[float]
    drift: float
    spread: float
    timestep: float = 0
    events: List[float] = field(default_factory=list)
    next_event: float = 0


@dataclass
class Portfolio:
    cash: float = 0
    inventory: float = 0

    def position(self, midprice):
        return self.cash, self.inventory * midprice

    def update(self, transaction, sign):
        self.cash -= transaction.price
        self.inventory += sign * transaction.quantity


@dataclass
class UserVariables:
    portfolio: Portfolio = field(default_factory=Portfolio)
    # currently quoted orders
    bid: Order = None
    ask: Order = None
    # currently placed orders
    ask_placed: Order | None = None
    bid_placed: Order | None = None

    def set_quote(self, order, side):
        setattr(self, side, order)

    def set_placed_order(self, order, side):
        setattr(self, f"{side}_placed", order)

    def update(self, transaction, sign):
        self.portfolio.update(transaction, sign)
        # if transaction fills a placed order, set it to None
        if self.ask_placed and self.ask_placed.uuid == transaction.uuid:
            self.ask_placed = None
        if self.bid_placed and self.bid_placed.uuid == transaction.uuid:
            self.bid_placed = None

class MarketSimulator:
    def __init__(
            self,
            starting_value,
            drift_rate=0.02,
            drift_std=0.01,
            volatility=0.1,
            spread_mean=0.1,
            spread_std=0.01,
            dt=1,
            base_event_intensity=0.5,
            event_size_mean=1,
            risk_free_reversion: float = 0.5,
            spread_reversion: float = 1e-2,
            order_eps: float = 2e-2,
            agent_ids: List[str] = None  # List of agent ids
    ):
        if agent_ids is None:
            agent_ids = []
        self.dt = dt
        self.drift_rate = drift_rate
        self.drift_std = drift_std
        self.spread_mean = spread_mean
        self.spread_std = spread_std

        bid_center = self.drift_rate - self.spread_mean / 2 / self.dt
        ask_center = self.drift_rate + self.spread_mean / 2 / self.dt

        self.bid_process = GeometricBrownianMotion(bid_center, volatility)
        self.ask_process = GeometricBrownianMotion(ask_center, volatility)

        self.risk_free_process = CoxIngersollRoss(risk_free_reversion, drift_rate, drift_std)
        self.spread_process = OrnsteinUhlenbeck(spread_reversion, spread_mean, spread_std)

        self.event_process = Hawkes(base_event_intensity, branching_ratio=1e-1, decay=0.3)

        self.quantity_distribution = np.random.poisson
        self.event_size_distribution = np.random.poisson
        self.event_size_mean = event_size_mean

        self.tick_size = 2
        self.lob: LimitOrderBook = LimitOrderBook(self.tick_size)

        self.starting_value = starting_value

        self.market_variables = MarketVariables(
            last_transaction=starting_value,
            quotes=[starting_value],
            drift=drift_rate,
            spread=spread_mean,
            next_event=self.next_event(0, [])
        )

        self.order_eps = order_eps
        self.agent_ids = agent_ids
        self.user_variables: Dict[str, UserVariables] = {
            agent_id: UserVariables()
            for agent_id in self.agent_ids
        }

        self.previous_event = 0
        self.uuid = "market"

    def reset(self):
        self.bid_process.mean = self.drift_rate - self.spread_mean / 2 / self.dt
        self.ask_process.mean = self.drift_rate + self.spread_mean / 2 / self.dt
        self.lob = LimitOrderBook(self.tick_size)
        self.market_variables = MarketVariables(
            last_transaction=self.starting_value,
            quotes=[self.starting_value],
            drift=self.drift_rate,
            spread=self.spread_mean,
            next_event=self.next_event(0, []),
            timestep=0,
        )
        self.user_variables = {agent_id: UserVariables() for agent_id in self.agent_ids}

    def add_user(self, agent_id):
        self.agent_ids.append(agent_id)
        self.user_variables[agent_id] = UserVariables()

    def fill(self, n):
        # generate n events to fill the order book
        for _ in range(n):
            orders = self._sample_orders(self.midprice())
            for order in orders:
                self.lob.send_order(order)
            self.market_variables.quotes.append(self.midprice())

    @property
    def best_ask(self):
        return self.lob.asks.bottom().value.price if self.lob.asks.bottom() is not None else None

    @property
    def best_bid(self):
        return self.lob.bids.top().value.price if self.lob.bids.top() is not None else None

    def midprice(self):
        best_ask = self.best_ask
        best_bid = self.best_bid

        if best_ask is not None and best_bid is not None:
            return (best_ask + best_bid) / 2
        return np.mean(self.market_variables.quotes[-5:])

    def spread(self):
        best_ask = self.best_ask
        best_bid = self.best_bid

        if best_ask is not None and best_bid is not None:
            return best_ask - best_bid
        return self.spread_mean

    def _sample_orders(self, price):
        bid_size = self.event_size_distribution(self.event_size_mean)
        ask_size = self.event_size_distribution(self.event_size_mean)

        bids = self.bid_process.sample(price, self.dt, bid_size)
        asks = self.ask_process.sample(price, self.dt, ask_size)

        bids_quantity = np.array(self.quantity_distribution(size=bid_size) + 1)
        asks_quantity = np.array(self.quantity_distribution(size=ask_size) + 1)

        orders = [Order(price, quantity, 'ask', client=self.uuid) for price, quantity in zip(asks, asks_quantity)] + \
                 [Order(price, quantity, 'bid', client=self.uuid)
                  for price, quantity in zip(bids, bids_quantity)]

        return np.array(orders)

    def set_order(self, user: UserVariables, bid=None, ask=None):
        def _set_order(order, placed_order, side):
            if order:
                user.set_quote(order, side)
                if placed_order is None:
                    self.lob.send_order(order)
                    user.set_placed_order(order, side)
                elif abs(placed_order.price - order.price) > self.order_eps:
                    self.lob.cancel_order(placed_order.uuid)
                    self.lob.send_order(order)
                    user.set_placed_order(order, side)

        _set_order(bid, user.bid_placed, 'bid')
        _set_order(ask, user.ask_placed, 'ask')

    def update_portfolios(self, transactions: List[Transaction]):
        delta = {}

        for transaction in transactions:
            buyer = transaction.buyer
            seller = transaction.seller

            if buyer in self.agent_ids:
                self.user_variables[buyer].portfolio.update(transaction, 1)
                delta.setdefault(buyer, Portfolio()).update(transaction, 1)

            if seller in self.agent_ids:
                self.user_variables[seller].portfolio.update(transaction, -1)
                delta.setdefault(seller, Portfolio()).update(transaction, -1)

        return delta

    def best(self, side):
        best = self.lob.bids.top() if side == 'bid' else self.lob.asks.bottom()
        return best.value.price if best is not None else self.midprice()

    def position(self, agent_id):
        return self.user_variables[agent_id].portfolio.position(self.midprice())

    def make_orders(self, action: list | tuple | np.ndarray):
        return Order(action[0], action[1], 'bid'), Order(action[2], action[3], 'ask')

    def next_event(self, timestep, events):
        return self.event_process.sample(timestep, self.dt, np.array(events))

    def step(self, action_dict: Dict[str, tuple] = None):
        transactions = []

        if action_dict is not None:
            for agent_id, action in action_dict.items():
                bid, ask = self.make_orders(action)
                self.set_order(self.user_variables[agent_id], bid, ask)

        if self.market_variables.timestep + self.dt >= self.market_variables.next_event:
            self.market_variables.events.append(self.market_variables.next_event)
            self.previous_event = self.market_variables.next_event
            orders = self._sample_orders(self.midprice())
            self.market_variables.next_event = self.next_event(self.market_variables.timestep,
                                                               self.market_variables.events)
            np.random.shuffle(orders)
            for order in orders:
                transactions += self.lob.send_order(order)

            self.market_variables.timestep = self.market_variables.next_event
        else:
            self.market_variables.timestep += self.dt

        self.market_variables.quotes.append(self.midprice())

        drift = self.risk_free_process.sample(self.market_variables.drift, self.dt)
        spread = self.spread_process.sample(self.market_variables.spread, self.dt)
        self.ask_process.mean = drift + spread / 2
        self.bid_process.mean = drift - spread / 2
        self.market_variables.drift = drift
        self.market_variables.spread = spread

        delta = {}

        if transactions:
            delta = self.update_portfolios(transactions)

        # Return observations as a dictionary (agent_id -> observation)
        midprice = self.midprice()
        positions, observations = {}, {}

        for agent_id in self.agent_ids:
            observations[agent_id] = self.get_observation(agent_id, midprice)
            positions[agent_id] = self.position(agent_id)

        return delta, transactions, positions, observations

    def get_observation(self, agent_id, midprice):
        observation = {
            "inventory": self.user_variables[agent_id].portfolio.inventory,
            "cash": self.user_variables[agent_id].portfolio.cash,
            "midprice": midprice,
            "spread": self.spread()
        }
        return observation

    def inventory(self, agent_id):
        return self.user_variables[agent_id].portfolio.inventory

    def quotes(self):
        return self.market_variables.quotes
from typing import List, Dict
import numpy as np
from dxlib.orders import LimitOrderBook, Order, Transaction
from dxlib.dynamics import GeometricBrownianMotion, CoxIngersollRoss, OrnsteinUhlenbeck, Hawkes


class MarketSimulator:
    def __init__(
            self,
            starting_value,
            risk_free_mean=0.02,
            risk_free_std=0.01,
            volatility=0.1,
            spread_mean=0.1,
            spread_std=0.01,
            dt=1,
            base_event_intensity=0.5,
            event_size_mean=1,
            risk_free_reversion: float = 0.5,
            spread_reversion: float = 1e-2,
            order_eps: float = 2e-2,
            starting_cash: float = 0,
            starting_inventory: float = 0,
            agent_ids: List[str] = None  # List of agent ids
    ):
        if agent_ids is None:
            agent_ids = []
        self.dt = dt
        self.risk_free_mean = risk_free_mean
        self.risk_free_std = risk_free_std
        self.spread_mean = spread_mean
        self.spread_std = spread_std

        bid_center = self.risk_free_mean - self.spread_mean / 2 / self.dt
        ask_center = self.risk_free_mean + self.spread_mean / 2 / self.dt

        self.bid_process = GeometricBrownianMotion(bid_center, volatility)
        self.ask_process = GeometricBrownianMotion(ask_center, volatility)

        self.risk_free_process = CoxIngersollRoss(
            risk_free_reversion, risk_free_mean, risk_free_std)
        self.spread_process = OrnsteinUhlenbeck(
            spread_reversion, spread_mean, spread_std)

        self.event_process = Hawkes(
            base_event_intensity, branching_ratio=1e-1, decay=0.3)

        self.quantity_distribution = np.random.poisson
        self.event_size_distribution = np.random.poisson
        self.event_size_mean = event_size_mean

        self.tick_size = 2
        self.lob: LimitOrderBook = LimitOrderBook(self.tick_size)

        self.starting_value = starting_value
        self.risk_free = risk_free_mean

        self.market_variables = {
            "last_transaction": self.starting_value,
            "midprice": [self.starting_value],
            "risk_free_rate": self.risk_free,
            "spread": self.spread_mean,
            "timestep": 0,
            "events": []
        }

        self.order_eps = order_eps
        self.agent_ids = agent_ids
        self.user_variables: Dict[str, Dict[str, any]] = {agent_id: {
            "bid_price": None,
            "ask_price": None,
            "bid_quantity": None,
            "ask_quantity": None,
            "ask_order": None,
            "bid_order": None,
            "cash": starting_cash,
            "inventory": starting_inventory
        } for agent_id in self.agent_ids}

        self.next_event = self.event_process.sample(
            0, self.dt, np.array(self.market_variables['events']))
        self.previous_event = 0

    def reset(self):
        self.bid_process.mean = self.risk_free_mean - self.spread_mean / 2 / self.dt
        self.ask_process.mean = self.risk_free_mean + self.spread_mean / 2 / self.dt
        self.lob = LimitOrderBook(self.tick_size)
        self.market_variables = {
            "last_transaction": self.starting_value,
            "midprice": [self.starting_value],
            "risk_free_rate": self.risk_free,
            "spread": self.spread_mean,
            "timestep": 0,
            "events": []
        }
        self.user_variables = {agent_id: {
            "bid_price": None,
            "ask_price": None,
            "bid_quantity": None,
            "ask_quantity": None,
            "ask_order": None,
            "bid_order": None,
            "cash": 0,
            "inventory": 0
        } for agent_id in self.agent_ids}

        self.next_event = self.event_process.sample(0, self.dt, np.array([]))

    def fill(self, n):
        # generate n events to fill the order book
        for _ in range(n):
            orders = self._sample_orders(self.midprice())
            for order in orders:
                self.lob.send_order(order)
            self.market_variables['midprice'].append(self.midprice())
            self.market_variables['timestep'] += self.dt

    def midprice(self):
        best_ask = self.lob.asks.bottom(
        ).value.price if self.lob.asks.bottom() is not None else None
        best_bid = self.lob.bids.top().value.price if self.lob.bids.top() is not None else None

        if best_ask is not None and best_bid is not None:
            return (best_ask + best_bid) / 2
        return np.mean(self.market_variables['midprice'][-5:])

    def spread(self):
        best_ask = self.lob.asks.bottom(
        ).value.price if self.lob.asks.bottom() is not None else None
        best_bid = self.lob.bids.top().value.price if self.lob.bids.top() is not None else None

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

        orders = [Order(price, quantity, 'ask') for price, quantity in zip(asks, asks_quantity)] + \
                 [Order(price, quantity, 'bid')
                  for price, quantity in zip(bids, bids_quantity)]

        return np.array(orders)

    def set_order(self, agent_id, bid=None, ask=None):
        if bid:
            self.user_variables[agent_id]["bid_price"] = bid.price
            self.user_variables[agent_id]["bid_quantity"] = bid.quantity

            if existing_order:= self.user_variables[agent_id].get("bid_order", None) is None:
                self.lob.send_order(bid)
                self.user_variables[agent_id]["bid_order"] = bid
            elif abs(existing_order.price - bid.price) > self.order_eps:
                self.lob.cancel_order(existing_order.uuid)
                self.lob.send_order(bid)
                self.user_variables[agent_id]["bid_order"] = bid

        if ask:
            self.user_variables[agent_id]["ask_price"] = ask.price
            self.user_variables[agent_id]["ask_quantity"] = ask.quantity

            if existing_order := self.user_variables[agent_id].get("ask_order", None) is None:
                self.lob.send_order(ask)
                self.user_variables[agent_id]["ask_order"] = ask
            elif abs(existing_order.price - ask.price) > self.order_eps:
                self.lob.cancel_order(existing_order.uuid)
                self.lob.send_order(ask)
                self.user_variables[agent_id]["ask_order"] = ask

    def _participated_side(self, transaction):
        for agent_id in self.agent_ids:
            if self.user_variables[agent_id]["bid_order"] and transaction.buyer == self.user_variables[agent_id][
                "bid_order"].uuid:
                return 'bid'
            elif self.user_variables[agent_id]["ask_order"] and transaction.seller == self.user_variables[agent_id][
                "ask_order"].uuid:
                return 'ask'
        return None

    def _calculate_pnl(self, transactions):
        bids, asks = [], []
        for transaction in transactions:
            side = self._participated_side(transaction)
            if side == 'bid':
                bids.append(transaction)
            elif side == 'ask':
                asks.append(transaction)
        if not bids and not asks:
            return 0, 0
        delta_inventory = sum([bid.quantity for bid in bids]) - \
                          sum([ask.quantity for ask in asks])
        transaction_pnl = -sum([bid.price * bid.quantity for bid in bids]) + sum(
            [ask.price * ask.quantity for ask in asks])
        return transaction_pnl, delta_inventory

    def best(self, side):
        best = self.lob.bids.top() if side == 'bid' else self.lob.asks.bottom()
        return best.value.price if best is not None else self.midprice()

    def virtual_pnl(self, pnl, quantity):
        best_price = self.best('bid' if quantity > 0 else 'ask')
        liquidation_pnl = quantity * best_price
        return pnl + liquidation_pnl

    def position(self, agent_id):
        return self.user_variables[agent_id]["inventory"] * self.midprice(), self.user_variables[agent_id]["cash"]

    def step(self, action_dict: Dict[int, tuple] = None):
        transactions = []

        if action_dict is not None:
            for agent_id, action in action_dict.items():
                bid = Order(action[0], action[1], 'bid')
                ask = Order(action[2], action[3], 'ask')
                self.set_order(agent_id, bid, ask)

        if self.market_variables['timestep'] + self.dt >= self.next_event:
            self.market_variables['events'].append(self.next_event)
            self.previous_event = self.next_event
            orders = self._sample_orders(self.midprice())
            self.next_event = self.event_process.sample(self.market_variables['timestep'], self.dt,
                                                        np.array(self.market_variables['events']))
            np.random.shuffle(orders)
            for order in orders:
                transactions += self.lob.send_order(order)

            self.market_variables['timestep'] = self.next_event
        else:
            self.market_variables['timestep'] += self.dt

        self.market_variables['midprice'].append(self.midprice())
        self.market_variables['risk_free_rate'] = self.risk_free_process.sample(self.market_variables['risk_free_rate'],
                                                                                self.dt)
        self.market_variables['spread'] = self.spread_process.sample(
            self.market_variables['spread'], self.dt)
        self.ask_process.mean = self.market_variables['risk_free_rate'] + \
                                self.market_variables['spread'] / 2
        self.bid_process.mean = self.market_variables['risk_free_rate'] - \
                                self.market_variables['spread'] / 2

        transaction_pnl = 0
        if transactions:
            transaction_pnl, delta_inventory = self._calculate_pnl(
                transactions)
            for agent_id in self.agent_ids:
                self.user_variables[agent_id]["cash"] += transaction_pnl
                self.user_variables[agent_id]["inventory"] += delta_inventory

        # Return observations as a dictionary (agent_id -> observation)
        observations = {agent_id: self.get_observation(agent_id) for agent_id in self.agent_ids}

        return transactions, {agent_id: self.position(agent_id) for agent_id in
                              self.agent_ids}, transaction_pnl, observations

    def get_observation(self, agent_id):
        observation = {
            "inventory": self.user_variables[agent_id]["inventory"],
            "cash": self.user_variables[agent_id]["cash"],
            "midprice": self.midprice(),
            "spread": self.spread()
        }
        return observation

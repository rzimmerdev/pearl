import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import islice

from src.env.multi.multi_simulator import MarketSimulator  # Import the multi-agent simulator


class Observation:
    @staticmethod
    def returns(quotes):
        if len(quotes) < 2:
            return 0
        return (quotes[-1] - quotes[-2]) / quotes[-2]

    @staticmethod
    def moving_average(quotes, n):
        if len(quotes) < n:
            return 0
        returns = np.diff(quotes)[-n:]
        return np.mean(returns)

    @staticmethod
    def rsi(quotes, window=30, interval=1):
        if len(quotes) < window:
            return interval / 2
        returns = np.diff(quotes)[-window:]
        up, down = returns.clip(min=0), -returns.clip(max=0)
        avg_up, avg_down = np.mean(up), np.mean(down)
        return interval if avg_down == 0 else interval * (1 - 1 / (1 + avg_up / avg_down))

    @staticmethod
    def volatility(quotes, window=30):
        if len(quotes) < window:
            return 0
        return np.std(np.diff(quotes)[-window:])

    @staticmethod
    def order_imbalance(bids, asks):
        if not bids or not asks:
            return 0
        num_bids = sum(order.quantity for order in bids[0].value.orders)
        num_asks = sum(order.quantity for order in asks[0].value.orders)
        return (num_bids - num_asks) / (num_bids + num_asks)

    @classmethod
    def state(cls, bids, asks, quotes, inventory):
        return [
            cls.returns(quotes),
            cls.moving_average(quotes, 5),
            cls.moving_average(quotes, 10),
            cls.moving_average(quotes, 50),
            cls.rsi(quotes),
            cls.volatility(quotes),
            inventory,
            cls.order_imbalance(bids, asks),
        ]


def reward_fn(virtual_pnl, inventory, delta_midprice, eta, alpha, starting_value):
    w = virtual_pnl + inventory * delta_midprice - np.max(eta * inventory * delta_midprice, 0)
    return 1 - np.exp(-alpha * w / starting_value)


class MarketEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
            self,
            n_levels=10,
            starting_value=100,
            *args, **kwargs
    ):
        super().__init__()
        self.n_levels = n_levels
        self.starting_value = starting_value
        self.simulator = MarketSimulator(
            starting_value=starting_value,
            *args, **kwargs,
        )

        self.quotes = []
        self.window = int(5e2)
        self.eta = 0.01
        self.alpha = 1
        self.duration = 1 / 252

        self.observation_space = spaces.Box(
            low=np.array([-100, -100, -100, -100, 0, 0, -1e3, -1e3] + [0, 0] * 2 * self.n_levels, dtype=np.float32),
            high=np.array([100, 100, 100, 100, 1, 1e3, 1e3, 1e3] + [1e4, 1e4] * 2 * self.n_levels, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1e2, 1e2, 1e2, 1e2], dtype=np.float32),
            dtype=np.float32,
        )
        self.agent_ids = set()

    def add_user(self, user_id):
        self.agent_ids.add(user_id)
        self.simulator.add_user(user_id)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulator.reset()
        self.simulator.fill(100)
        self.quotes = []

        # Get initial states for all agents
        states = {agent_id: self._get_state(agent_id) for agent_id in self.agent_ids}
        return states, {}

    def step(self, actions):
        previous_midprice = self.simulator.midprice()
        delta, transactions, positions, observations = self.simulator.step(actions)
        midprice = self.simulator.midprice()
        delta_midprice = midprice - previous_midprice
        self.quotes.append(midprice)

        next_state = {agent_id: self._get_state(agent_id) for agent_id in self.agent_ids}

        best_ask = self.simulator.best_ask or midprice
        best_bid = self.simulator.best_bid or midprice

        reward = {}

        for agent_id in self.agent_ids:
            virtual_pnl = self.virtual_pnl(best_ask, best_bid, delta[agent_id]) if agent_id in delta else 0
            reward[agent_id] = reward_fn(
                virtual_pnl,
                self.simulator.user_variables[agent_id].portfolio.inventory,
                delta_midprice,
                self.eta,
                self.alpha,
                self.starting_value,
            )

        done = self.done
        trunc = any(np.abs(r) > 1e4 for r in reward.values())

        return next_state, reward, done, trunc, {}

    def virtual_pnl(self, best_ask, best_bid, delta):
        if delta.inventory < 0:  # sold inventory, liquidation price is best bid
            return delta.inventory * best_bid + delta.cash
        else:
            return delta.inventory * best_ask + delta.cash

    def _get_state(self, agent_id):
        lob = self.simulator.lob
        bids_list = list(islice(lob.bids.ordered_traversal(reverse=True), self.n_levels))
        asks_list = list(islice(lob.asks.ordered_traversal(), self.n_levels))

        state = Observation.state(bids_list, asks_list, self.simulator.quotes(), self.simulator.inventory(agent_id))

        bids = self.order_side(bids_list)
        asks = self.order_side(asks_list, fill_value=1e4)
        state += list(bids) + list(asks)

        return np.array(state, dtype=np.float32)

    def state(self, agent_id):
        return self._get_state(agent_id)

    def order_side(self, side_list, fill_value=0.0):
        levels = []
        midprice = self.simulator.midprice()
        for price_level in side_list:
            level_quantity = sum(order.quantity for order in price_level.value.orders)
            levels.append([np.abs(price_level.key - midprice), level_quantity])
        levels = np.array(levels).flatten()
        levels = np.pad(levels, (0, 2 * self.n_levels - len(levels)), 'constant', constant_values=fill_value)
        return levels

    def render(self, mode="human"):
        print(f"Midprice: {self.simulator.midprice()}")

    @property
    def timestep(self):
        return self.simulator.market_variables.timestep

    @property
    def done(self):
        return self.timestep >= self.duration

    @property
    def events(self):
        return self.simulator.market_variables.events

    @property
    def snapshot_columns(self):
        return [
            "position",
            "midprice",
            "inventory",
            "events",
            "market_timestep",
        ]

    def snapshot(self):
        midprice = self.simulator.midprice()
        position = {
            agent_id: self.simulator.user_variables[agent_id].portfolio.position(midprice)
            for agent_id in self.agent_ids
        }
        return {
            "position": position,
            "midprice": midprice,
            "events": self.events[-1],
            "market_timestep": self.timestep,
        }

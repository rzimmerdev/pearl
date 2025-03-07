import time
import uuid

import zmq

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import islice

from src.env.multi.multi_simulator import MarketSimulator  # Import the multi-agent simulator
from src.env.multi.router import Router


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
    def rsi(quotes, window=30):
        if len(quotes) < window:
            return 50
        returns = np.diff(quotes)[-window:]
        up, down = returns.clip(min=0), -returns.clip(max=0)
        avg_up, avg_down = np.mean(up), np.mean(down)
        return 100 if avg_down == 0 else 100 - 100 / (1 + avg_up / avg_down)

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
    def state(cls, bids, asks, midprice, inventory):
        return [
            cls.returns(midprice),
            cls.moving_average(midprice, 5),
            cls.moving_average(midprice, 10),
            cls.moving_average(midprice, 50),
            cls.rsi(midprice),
            cls.volatility(midprice),
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
            n_agents=2,  # Added n_agents to handle multiple agents
            n_levels=10,
            starting_value=100,
            *args, **kwargs
    ):
        super().__init__()
        self.n_agents = n_agents  # Number of agents
        self.n_levels = n_levels
        self.starting_value = starting_value
        self.agent_ids = [str(uuid.uuid4()) for _ in range(n_agents)]
        self.simulator = MarketSimulator(
            starting_value=starting_value,
            *args, **kwargs,
            agent_ids=self.agent_ids,
        )

        self.quotes = []
        self.window = int(5e2)
        self.eta = 0.01
        self.alpha = 1
        self.duration = 1 / 252

        # Modify observation space for multiple agents
        self.observation_space = spaces.Box(
            low=np.array([-100, -100, -100, -100, 0, 0, -1e3, -1e3] + [0, 0] * 2 * self.n_levels, dtype=np.float32),
            high=np.array([100, 100, 100, 100, 100, 1e3, 1e3, 1e3] + [1e4, 1e4] * 2 * self.n_levels, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1e2, 1e2, 1e2, 1e2], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulator.reset()
        self.simulator.fill(100)
        self.quotes = []

        # Get initial states for all agents
        states = [self._get_state(agent_id) for agent_id in range(self.n_agents)]
        return states, {}

    def step(self, actions):
        previous_midprice = self.simulator.midprice()
        previous_inventory = self.inventory

        # Process actions for each agent
        transactions, positions, transaction_pnl = self.simulator.step(actions)

        self.quotes.append(self.simulator.midprice())
        inventory = [self.simulator.user_variables[agent_id]["inventory"] for agent_id in self.agent_ids]

        delta_midprice = self.simulator.midprice() - previous_midprice
        delta_inventory = [inventory[i] - previous_inventory[agent_id] for i, agent_id in enumerate(self.agent_ids)]

        next_state = [self._get_state(agent_id) for agent_id in range(self.n_agents)]
        virtual_pnl = [self.simulator.virtual_pnl(transaction_pnl[i], delta_inventory[i]) for i in range(self.n_agents)]
        reward = [reward_fn(virtual_pnl[i], inventory[i], delta_midprice, self.eta, self.alpha, self.starting_value)
                  for i, agent_id in enumerate(range(self.n_agents))]
        done = self.simulator.market_variables["timestep"] >= self.duration
        trunc = any(np.abs(r) > 1e4 for r in reward)

        return next_state, reward, done, trunc, {}

    def _calculate_reward(self, transaction_pnl, inventory, delta_inventory, delta_midprice):
        w = self.simulator.virtual_pnl(transaction_pnl, delta_inventory) + inventory * delta_midprice - np.max(
            self.eta * inventory * delta_midprice, 0)
        return 1 - np.exp(-self.alpha * w / self.starting_value)

    def _get_state(self, agent_id):
        lob = self.simulator.lob
        bids_list = list(islice(lob.bids.ordered_traversal(reverse=True), self.n_levels))
        asks_list = list(islice(lob.asks.ordered_traversal(), self.n_levels))

        state = Observation.state(bids_list, asks_list, self.simulator.midprice(), self.inventory[agent_id])

        bids = self.order_side(bids_list)
        asks = self.order_side(asks_list, fill_value=1e4)
        state += list(bids) + list(asks)

        return np.array(state, dtype=np.float32)

    def order_side(self, side_list, fill_value=0.0):
        levels = []
        midprice = self.simulator.midprice()
        for price_level in side_list:
            level_quantity = sum(order.quantity for order in price_level.value.orders)
            levels.append([np.abs(price_level.key - midprice), level_quantity])
        levels = np.array(levels).flatten()
        levels = np.pad(levels, (0, 2 * self.n_levels - len(levels)), 'constant', constant_values=fill_value)
        return levels

    @property
    def inventory(self):
        return {
            agent_id: self.simulator.user_variables[agent_id]["inventory"]
            for agent_id in self.agent_ids
        }

    def render(self, mode="human"):
        print(f"Midprice: {self.simulator.midprice()}, Inventory: {self.inventory}")

    @property
    def done(self):
        return self.simulator.market_variables["timestep"] >= self.duration

    @property
    def events(self):
        return self.simulator.market_variables["events"]

    @property
    def snapshot_columns(self):
        return [
            "financial_return",
            "midprice",
            "inventory",
            "events",
            "market_timestep",
        ]

    def snapshot(self):
        financial_return = {
            agent_id: self.simulator.user_variables[agent_id]["cash"] +
                      self.simulator.user_variables[agent_id]["inventory"] * self.simulator.midprice()
            for agent_id in self.agent_ids
        }
        return {
            "financial_return": financial_return,
            "midprice": self.simulator.midprice(),
            "inventory": self.inventory,
            "events": self.events[-1],
            "market_timestep": self.simulator.market_variables["timestep"],
        }


class AsyncMarketEnv(MarketEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # send to zeromq queue
        self.router = Router()

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        out = super().step(*args, **kwargs)
        self.socket.send_json(
            {
                "midprice": self.simulator.midprice(),
                "inventory": self.inventory,
                "events": self.events[-1],
                "market_timestep": self.simulator.market_variables["timestep"],
            }
        )
        return out

    def render(self, *args, **kwargs):
        return super().render(*args, **kwargs)

    @property
    def done(self):
        return super().done

    @property
    def snapshot_columns(self):
        return super().snapshot_columns

    def snapshot(self):
        return super().snapshot()

if __name__ == "__main__":
    env = AsyncMarketEnv()

    while not env.done:
        # simulate reading agent actions from a message queue
        actions = np.random.rand(2, 4)
        time.sleep(1)
        print(actions)
        next_state, reward, done, trunc, _ = env.step(actions)

        env.render()
        if trunc:
            break

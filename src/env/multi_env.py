import uuid

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import islice

from ray.rllib import MultiAgentEnv

from .multi_simulator import MarketSimulator  # Import the multi-agent simulator


class MarketEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
            self,
            n_agents=2,  # Added n_agents to handle multiple agents
            n_levels=10,
            starting_value=100,
            risk_free_mean=0.02,
            risk_free_std=0.01,
            volatility=0.4,
            spread_mean=.150,
            spread_std=0.01,
            dt=1 / 252 / 6.5 / 60,
            base_event_intensity=0.5,
            event_size_mean=1,
            risk_free_reversion=0.5,
            spread_reversion=1e-2,
            order_eps=2e-2,
    ):
        super().__init__()
        self.n_agents = n_agents  # Number of agents
        self.n_levels = n_levels
        self.starting_value = starting_value
        self.agent_ids = [str(uuid.uuid4()) for _ in range(n_agents)]
        self.simulator = MarketSimulator(
            starting_value=starting_value,
            risk_free_mean=risk_free_mean,
            risk_free_std=risk_free_std,
            volatility=volatility,
            spread_mean=spread_mean,
            spread_std=spread_std,
            dt=dt,
            base_event_intensity=base_event_intensity,
            event_size_mean=event_size_mean,
            risk_free_reversion=risk_free_reversion,
            spread_reversion=spread_reversion,
            order_eps=order_eps,
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
        previous_inventory = self._calculate_inventory()

        # Process actions for each agent
        transactions, positions, transaction_pnl = self.simulator.step(actions)

        self.quotes.append(self.simulator.midprice())
        inventory = [self.simulator.user_variables[agent_id]["inventory"] for agent_id in self.agent_ids]

        delta_midprice = self.simulator.midprice() - previous_midprice
        delta_inventory = [inventory[i] - previous_inventory[agent_id] for i, agent_id in enumerate(self.agent_ids)]

        next_state = [self._get_state(agent_id) for agent_id in range(self.n_agents)]
        reward = [self._calculate_reward(transaction_pnl[agent_id], inventory[i], delta_inventory[i], delta_midprice)
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

        state = [
            self._calculate_return(),
            self._calculate_ma(5),
            self._calculate_ma(10),
            self._calculate_ma(50),
            self._calculate_rsi(),
            self._calculate_volatility(),
            self._calculate_inventory(),
            self._calculate_order_imbalance(bids_list, asks_list),
        ]

        bids = self._extract_order_book_side(bids_list)
        asks = self._extract_order_book_side(asks_list, fill_value=1e4)
        state += list(bids) + list(asks)

        return np.array(state, dtype=np.float32)

    def _extract_order_book_side(self, side_list, fill_value=0.0):
        levels = []
        midprice = self.simulator.midprice()
        for price_level in side_list:
            level_quantity = sum(order.quantity for order in price_level.value.orders)
            levels.append([np.abs(price_level.key - midprice), level_quantity])
        levels = np.array(levels).flatten()
        levels = np.pad(levels, (0, 2 * self.n_levels - len(levels)), 'constant', constant_values=fill_value)
        return levels

    def _calculate_return(self):
        if len(self.quotes) < 2:
            return 0
        return (self.quotes[-1] - self.quotes[-2]) / self.quotes[-2]

    def _calculate_ma(self, n):
        if len(self.quotes) < n:
            return 0
        returns = np.diff(self.quotes)[-n:]
        return np.mean(returns)

    def _calculate_rsi(self):
        if len(self.quotes) < self.window:
            return 50
        returns = np.diff(self.quotes)[-self.window:]
        up, down = returns.clip(min=0), -returns.clip(max=0)
        avg_up, avg_down = np.mean(up), np.mean(down)
        return 100 if avg_down == 0 else 100 - 100 / (1 + avg_up / avg_down)

    def _calculate_volatility(self):
        if len(self.quotes) < self.window:
            return 0
        return np.std(np.diff(self.quotes)[-self.window:])

    def _calculate_inventory(self):
        return {
            agent_id: self.simulator.user_variables[agent_id]["inventory"]
            for agent_id in self.agent_ids
        }

    def _calculate_order_imbalance(self, bids, asks):
        if not bids or not asks:
            return 0
        num_bids = sum(order.quantity for order in bids[0].value.orders)
        num_asks = sum(order.quantity for order in asks[0].value.orders)
        return (num_bids - num_asks) / (num_bids + num_asks)

    def render(self, mode="human"):
        print(f"Midprice: {self.simulator.midprice()}, Inventory: {self._calculate_inventory()}")

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
            "inventory": self._calculate_inventory(),
            "events": self.events[-1],
            "market_timestep": self.simulator.market_variables["timestep"],
        }


class MultiMarketEnv(MultiAgentEnv, MarketEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def render(self, *args, **kwargs):
        return super().render(*args, **kwargs)

    @property
    def done(self):
        return super().done

    @property
    def observation_space(self):
        return super().observation_space

    @property
    def action_space(self):
        return super().action_space

    @property
    def snapshot_columns(self):
        return super().snapshot_columns

    def snapshot(self):
        return super().snapshot()



if __name__ == "__main__":
    env = MarketEnv(n_agents=3, spread_mean=1000, volatility=0.6)
    state, _ = env.reset()
    action = np.array(
        [[1.0, 10.0, 1.2, 5.0], [2.0, 20.0, 2.4, 10.0], [0.5, 5.0, 0.6, 2.0]])  # Example actions for 3 agents
    lob_state, _, _, _, _ = env.step(action)
    print(lob_state[0][5:5 + 10 * 2])  # Print asks for agent 0
    print(lob_state[1][5 + 10 * 2:])  # Print bids for agent 1

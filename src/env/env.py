import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import islice
from env.simulators import MarketSimulator


class MarketEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
            self,
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
        self.n_levels = n_levels
        self.starting_value = starting_value
        self.simulator = MarketSimulator(
            starting_value,
            risk_free_mean,
            risk_free_std,
            volatility,
            spread_mean,
            spread_std,
            dt,
            base_event_intensity,
            event_size_mean,
            risk_free_reversion,
            spread_reversion,
            order_eps,
        )

        self.quotes = []
        self.window = int(5e2)
        self.eta = 0.01
        self.alpha = 1
        self.duration = 1 / 252

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
        state = self._get_state()
        return state, {}

    def step(self, action):
        previous_midprice = self.simulator.midprice()
        previous_inventory = self._calculate_inventory()

        if action is not None:
            action[0] = self.simulator.midprice() - action[0]
            action[2] += self.simulator.midprice()

        transactions, position, transaction_pnl = self.simulator.step(action)

        self.quotes.append(self.simulator.midprice())
        inventory = self.simulator.user_variables["inventory"]

        delta_midprice = self.simulator.midprice() - previous_midprice
        delta_inventory = inventory - previous_inventory

        next_state = self._get_state()
        reward = self._calculate_reward(transaction_pnl, inventory, delta_inventory, delta_midprice)
        done = self.simulator.market_timestep >= self.duration
        trunc = np.abs(reward) > 1e4

        return next_state, reward, done, trunc, {}

    def _calculate_reward(self, transaction_pnl, inventory, delta_inventory, delta_midprice):
        w = self.simulator.virtual_pnl(transaction_pnl, delta_inventory) + inventory * delta_midprice - np.max(
            self.eta * inventory * delta_midprice, 0)
        return 1 - np.exp(-self.alpha * w / self.starting_value)

    def _get_state(self):
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
        # if no last midprice, return 0
        if len(self.quotes) < 2:
            return 0
        return (self.quotes[-1] - self.quotes[-2]) / self.quotes[-2]

    def _calculate_ma(self, n):
        # calculate n-tick return moving average
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
        return self.simulator.user_variables["inventory"]

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
        return self.simulator.market_timestep >= self.duration

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
        financial_return = self.simulator.user_variables["cash"] + self.simulator.user_variables[
            "inventory"] * self.simulator.midprice()
        return {
            "financial_return": financial_return,
            "midprice": self.simulator.midprice(),
            "inventory": self._calculate_inventory(),
            "events": self.events[-1],
            "market_timestep": self.simulator.market_timestep,
        }


if __name__ == "__main__":
    env = MarketEnv(spread_mean=1000, volatility=0.6)
    state, _ = env.reset()
    action = np.array([1.0, 10.0, 1.2, 5.0])  # Example action
    lob_state, _, _, _, _ = env.step(action)
    # print asks
    print(lob_state[5:5 + 10 * 2])
    # print bids
    print(lob_state[5 + 10 * 2:])

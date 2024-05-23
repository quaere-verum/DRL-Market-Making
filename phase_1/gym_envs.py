import numpy as np
import gymnasium as gym
from utils.history import History
from utils.portfolio import SimplePortfolio
from typing import Union, Tuple


def make_reward_function(rho=0.):
    def base_reward_function(info, benchmark=False):
        col = 'benchmark_stock_held' if benchmark else 'stock_held'
        stock_amount_delta = info[col, -1] - info[col, -2]
        option_delta = (max(info['stock_price', -1] - info['strike_price', 0], 0) -
                        max(info['stock_price', -2] - info['strike_price', 0], 0))
        transaction_cost = info['transaction_fees', 0] * info['stock_price', - 1] * np.abs(stock_amount_delta)
        stock_delta = info[col, -2] * (info['stock_price', -1] - info['stock_price', -2])
        portfolio_delta = stock_delta - option_delta
        return portfolio_delta - transaction_cost - rho*portfolio_delta ** 2
    return base_reward_function


class MarketMakingEnv(gym.Env):
    metadata = {'render_modes': ['logs']}

    def __init__(self,
                 epsilon: float = 0.1,
                 sigma: float = 0.1,
                 rho: float = 0.,
                 action_bins: int = 0,
                 duration_bounds: tuple = (2, 30),
                 transaction_fees: float = 0.001,
                 seed: int = None,
                 benchmark: bool = False):
        """
        Gymnasium environment for option pricing.
        :param epsilon: action space is [0, 1+epsilon]
        :param sigma: standard deviation of the asset returns
        :param rho: risk aversion in the reward function
        :param action_bins: if 0, a continuous action space is used. Else, must be >= 2 and the action space is
        discretised
        :param duration_bounds: minimum and maximum duration
        :param transaction_fees: transaction fees as a percentage of each transaction
        :param seed: random number generation seed for reproducible results
        :param benchmark: whether to benchmark the agent against Black-Scholes
        """
        super().__init__()
        assert epsilon >= 0
        assert action_bins == 0 or action_bins >= 2

        self.duration_bounds = duration_bounds
        self.transaction_fees = transaction_fees
        self.seed = seed
        self.benchmark = benchmark
        self.action_bins = None
        self.epsilon = epsilon

        if action_bins == 0:
            self.action_space = gym.spaces.Box(low=0, high=1 + epsilon)
        else:
            self.action_bins = action_bins
            self.action_space = gym.spaces.Discrete(action_bins)
        # stock price, remaining time, stock held, strike price, volatility forecast, Black-Scholes hedge
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0, 0]).astype(np.float32),
                                                high=np.array([np.inf, duration_bounds[1] + 2,
                                                               1 + epsilon, np.inf, np.inf, 1]).astype(np.float32),
                                                shape=(6,))

        self.sigma = sigma
        self.portfolio = None
        if benchmark:
            self.benchmark_portfolio = None
        self.reward_function = make_reward_function(rho)
        self.rng = None

        self.info = None

    def _process_action(self, action: np.ndarray) -> np.float32:
        if isinstance(self.action_space, gym.spaces.Discrete):
            return np.float32(action[0]/self.action_bins*(1+self.epsilon))
        else:
            return np.float32(action[0])

    def reset(self, seed: int = None, options: None = None) -> Tuple[np.ndarray, History]:
        if self.seed is not None:
            self.seed += 1
        self.rng = np.random.RandomState(self.seed)
        strike_price = np.clip(self.rng.normal(100,20), 50, 150)
        expiry_time = self.rng.randint(low=self.duration_bounds[0], high=self.duration_bounds[1] + 1)
        self.portfolio = SimplePortfolio(strike_price=strike_price,
                                         stock_price=100.,
                                         expiry_time=expiry_time,
                                         transaction_fees=self.transaction_fees)
        if self.benchmark:
            self.benchmark_portfolio = SimplePortfolio(strike_price=strike_price,
                                                       stock_price=100.,
                                                       expiry_time=expiry_time,
                                                       transaction_fees=self.transaction_fees)
        self.info = History(max_size=self.duration_bounds[1])

        black_scholes_hedge = self.portfolio.black_scholes_hedge(self.sigma)

        state = {
            'stock_price': self.portfolio.stock_price,
            'remaining_time': self.portfolio.remaining_time,
            'stock_held': self.portfolio.stock_held,
            'strike_price': self.portfolio.strike_price,
            'volatility_forecast': self.sigma,
            'black_scholes_hedge': black_scholes_hedge
        }
        self.info.set(
            transaction_fees=self.transaction_fees,
            reward=0,
            benchmark_stock_held=0 if self.benchmark else None,
            benchmark_reward=0 if self.benchmark else None,
            **state
        )
        return np.array(list(state.values()), dtype=np.float32), self.info[-1]

    def step(self, action: np.ndarray = None) -> Tuple[np.ndarray, float, bool, bool, History]:
        action = self._process_action(action)
        done, truncated = False, False
        new_price = self.portfolio.stock_price * np.exp(self.rng.normal(0, self.sigma))
        self.portfolio.update_position(new_price, action)
        black_scholes_hedge = self.portfolio.black_scholes_hedge(self.sigma)

        if self.benchmark:
            benchmark_hedge = self.benchmark_portfolio.black_scholes_hedge(self.sigma)
            self.benchmark_portfolio.update_position(new_price, benchmark_hedge)

        state = {
            'stock_price': self.portfolio.stock_price,
            'remaining_time': self.portfolio.remaining_time,
            'stock_held': self.portfolio.stock_held,
            'strike_price': self.portfolio.strike_price,
            'volatility_forecast': self.sigma,
            'black_scholes_hedge': black_scholes_hedge
        }
        self.info.add(
            transaction_fees=self.transaction_fees,
            reward=0,
            benchmark_stock_held=self.benchmark_portfolio.stock_held if self.benchmark else None,
            benchmark_reward=0 if self.benchmark else None,
            **state
        )
        reward = self.reward_function(self.info)
        if self.benchmark:
            benchmark_reward = self.reward_function(self.info, benchmark=True)

        if self.portfolio.remaining_time == 1:
            reward -= self.portfolio.stock_held * new_price * self.transaction_fees
            if self.benchmark:
                benchmark_reward -= self.portfolio.stock_held * new_price * self.transaction_fees
            done = True

        self.info['benchmark_reward', -1] = benchmark_reward if self.benchmark else None
        self.info['reward', -1] = reward

        if self.benchmark:
            reward = reward - benchmark_reward

        return np.array(list(state.values()), dtype=np.float32), reward, done, truncated, self.info[-1]

    def render(self) -> None:
        pass

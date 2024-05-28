import numpy as np
import gymnasium as gym
from utils.history import History
from utils.portfolio import SimplePortfolio
from typing import Union, Tuple, List, Callable

gym.envs.register('OptionHedgingEnv', 'option_hedging.gym_envs:OptionHedgingEnv')


def make_reward_function(rho: float = 0.) -> Callable[[History], float]:

    def reward_function(info):
        portfolio_delta = info['portfolio_value', -1] - info['portfolio_value', -2]
        transaction_costs = info['transaction_fees', -1] * np.abs((info['stock_held', -1] -
                                                                   info['stock_held', -2]) * info['stock_price', -1])
        return portfolio_delta - transaction_costs - rho / 2 * portfolio_delta ** 2

    return reward_function


class OptionHedgingEnv(gym.Env):
    metadata = {'render_modes': ['logs']}

    def __init__(self,
                 epsilon: float = 0.1,
                 sigma: float = 0.1,
                 rho: float = 0.,
                 action_bins: int = 0,
                 T: int = 1,
                 rebalance_frequency: int = 12,
                 transaction_fees: float = 0.001,
                 benchmark: bool = False):
        """
        Gymnasium environment for option hedging.
        :param epsilon: action space is [0, 1+epsilon]
        :param sigma: standard deviation of the asset returns
        :param rho: risk aversion in the reward function
        :param action_bins: if 0, a continuous action space is used. Else, must be >= 2 and the action space is
        discretised
        :param duration_bounds: minimum and maximum duration
        :param transaction_fees: transaction fees as a percentage of each transaction
        """
        super().__init__()
        assert epsilon >= 0
        assert action_bins == 0 or action_bins >= 2
        assert rebalance_frequency >= 1 and isinstance(rebalance_frequency, int)

        self.T = T
        self.dt = 1/rebalance_frequency
        self.transaction_fees = transaction_fees
        self.benchmark = benchmark
        self.action_bins = action_bins
        self.epsilon = epsilon

        if action_bins == 0:
            self.action_space = gym.spaces.Box(low=0, high=1 + epsilon)
        else:
            self.action_space = gym.spaces.Discrete(action_bins)
        # stock price, remaining time, stock held, strike price, option_valuation
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0]).astype(np.float32),
                                                high=np.array([1e3, T+1,
                                                               1 + epsilon, 1e3, 1e3]).astype(np.float32),
                                                shape=(5,))

        self.sigma = sigma
        self.portfolio = None
        self.reward_function = make_reward_function(rho=rho)
        self.rng = None

        self.info = None

    def seed(self, seed: int = None) -> List[int]:
        if seed is not None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        self.rng = np.random.default_rng(seed=seed)
        return [seed]

    def _process_action(self, action: Union[np.ndarray, np.int32]) -> np.float32:
        if not isinstance(action, (np.ndarray, list, tuple)) and self.action_bins > 0:
            return np.float32(action/self.action_bins*(1+self.epsilon))
        elif self.action_bins > 0:
            return np.float32(action[0] / self.action_bins * (1 + self.epsilon))
        elif isinstance(action, (np.ndarray, list, tuple)):
            return np.float32(action[0])
        else:
            return np.float32(action)

    def reset(self, seed: int = None, options: None = None) -> Tuple[np.ndarray, History]:
        if self.rng is None:
            self.seed()
        strike_price = 100
        self.portfolio = SimplePortfolio(strike_price=strike_price,
                                         expiry_time=self.T,
                                         dt=self.dt)
        self.portfolio.init(self.sigma, 0.5)
        self.info = History(max_size=self.T*int(1/self.dt))

        state = {
            'stock_price': self.portfolio.stock_price,
            'remaining_time': self.portfolio.remaining_time,
            'stock_held': self.portfolio.stock_held,
            'strike_price': self.portfolio.strike_price,
            'option_value': self.portfolio.option_valuation(self.sigma)
        }
        self.info.set(
            transaction_fees=self.transaction_fees,
            reward=0,
            sigma=self.sigma,
            portfolio_value=self.portfolio.portfolio_valuation(self.sigma),
            stock_held=self.portfolio.stock_held,
            stock_price=self.portfolio.stock_price,
            capital=self.portfolio.capital,
            option_value=state['option_value'],
            action=None
        )
        return np.array(list(state.values()), dtype=np.float32), self.info[-1]

    def step(self, action: np.ndarray = None) -> Tuple[np.ndarray, float, bool, bool, History]:
        action = self._process_action(action)
        done, truncated = False, False
        price_change = np.exp(self.rng.normal(0, self.sigma*np.sqrt(self.dt)))
        new_price = self.portfolio.stock_price * price_change
        self.portfolio.update_position(new_price, action)

        state = {
            'stock_price': self.portfolio.stock_price,
            'remaining_time': self.portfolio.remaining_time,
            'stock_held': self.portfolio.stock_held,
            'strike_price': self.portfolio.strike_price,
            'option_value': self.portfolio.option_valuation(self.sigma)
        }
        self.info.add(
            transaction_fees=self.transaction_fees,
            reward=0,
            sigma=self.sigma,
            portfolio_value=self.portfolio.portfolio_valuation(self.sigma),
            stock_held=self.portfolio.stock_held,
            stock_price=self.portfolio.stock_price,
            capital=self.portfolio.capital,
            option_value=state['option_value'],
            action=action
        )
        reward = self.reward_function(self.info)
        if np.isclose(self.portfolio.remaining_time, self.dt):
            reward -= self.portfolio.stock_held * new_price * self.transaction_fees
            done = True

        self.info['reward', -1] = reward
        return np.array(list(state.values()), dtype=np.float32), reward, done, truncated, self.info[-1]

    def render(self) -> None:
        pass


def make_env(epsilon: float,
             sigma: float,
             rho: float,
             action_bins: int,
             T: int,
             rebalance_frequency: int,
             seed: int | None,
             transaction_fees: float = 0.001,
             **kwargs) -> Callable[[], OptionHedgingEnv]:
    def _init() -> OptionHedgingEnv:
        env = gym.make('OptionHedgingEnv',
                       epsilon=epsilon,
                       sigma=sigma,
                       rho=rho,
                       action_bins=action_bins,
                       T=T,
                       rebalance_frequency=rebalance_frequency,
                       transaction_fees=transaction_fees)
        env.seed(seed)
        return env
    return _init

import numpy as np
import gymnasium as gym
from scipy.stats import norm
from utils.history import History
from utils.portfolio import SimplePortfolio
from typing import Union, Tuple, List


def make_reward_function(rho=0., mode='model_free'):
    if mode == 'model_free':
        def reward_function(info):
            stock_amount_delta = info['stock_held', -1] - info['stock_held', -2]
            option_delta = (max(info['stock_price', -1] - info['strike_price', 0], 0) -
                            max(info['stock_price', -2] - info['strike_price', 0], 0))
            transaction_cost = info['transaction_fees', 0] * info['stock_price', - 1] * np.abs(stock_amount_delta)
            stock_delta = info['stock_held', -2] * (info['stock_price', -1] - info['stock_price', -2])
            portfolio_delta = stock_delta - option_delta
            return portfolio_delta - transaction_cost - rho/2*portfolio_delta ** 2
    else:
        def option_valuation(S, K, T, sigma):
            d_plus = 1/(sigma*np.sqrt(T))*np.log(S/K)+sigma**2/2*T
            d_minus = d_plus - sigma*np.sqrt(T)
            return norm.cdf(d_plus)*S-norm.cdf(d_minus)*K

        def reward_function(info):
            stock_amount_delta = info['stock_held', -1] - info['stock_held', -2]
            option_valuation_old = option_valuation(info['stock_price', -2],
                                                    info['strike_price', -2],
                                                    info['remaining_time', -2],
                                                    info['sigma', -2])
            option_valuation_new = option_valuation(info['stock_price', -1],
                                                    info['strike_price', -1],
                                                    info['remaining_time', -1],
                                                    info['sigma', -1])
            option_delta = option_valuation_new - option_valuation_old
            transaction_cost = info['transaction_fees', 0] * info['stock_price', - 1] * np.abs(stock_amount_delta)
            stock_delta = info['stock_held', -2] * (info['stock_price', -1] - info['stock_price', -2])
            portfolio_delta = stock_delta - option_delta
            return portfolio_delta - transaction_cost - rho / 2 * portfolio_delta ** 2

    return reward_function


class OptionHedgingEnv(gym.Env):
    metadata = {'render_modes': ['logs']}

    def __init__(self,
                 epsilon: float = 0.1,
                 sigma: float = 0.1,
                 rho: float = 0.,
                 action_bins: int = 0,
                 duration_bounds: tuple = (2, 30),
                 transaction_fees: float = 0.001,
                 mode: str = 'model_free',
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
        assert mode.lower() in ('model_free', 'gbm')

        self.duration_bounds = duration_bounds
        self.transaction_fees = transaction_fees
        self.benchmark = benchmark
        self.action_bins = action_bins
        self.epsilon = epsilon

        if action_bins == 0:
            self.action_space = gym.spaces.Box(low=0, high=1 + epsilon)
        else:
            self.action_space = gym.spaces.Discrete(action_bins)
        # stock price, remaining time, stock held, strike price
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]).astype(np.float32),
                                                high=np.array([1e3, duration_bounds[1] + 2,
                                                               1 + epsilon, 1e3]).astype(np.float32),
                                                shape=(4,))

        self.sigma = sigma
        self.portfolio = None
        self.reward_function = make_reward_function(rho=rho, mode=mode.lower())
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
        expiry_time = self.rng.integers(low=self.duration_bounds[0], high=self.duration_bounds[1] + 1)
        self.portfolio = SimplePortfolio(strike_price=strike_price,
                                         stock_price=100.,
                                         expiry_time=expiry_time,
                                         transaction_fees=self.transaction_fees)

        self.info = History(max_size=self.duration_bounds[1])

        state = {
            'stock_price': self.portfolio.stock_price,
            'remaining_time': self.portfolio.remaining_time,
            'stock_held': self.portfolio.stock_held,
            'strike_price': self.portfolio.strike_price
        }
        self.info.set(
            transaction_fees=self.transaction_fees,
            reward=0,
            sigma=self.sigma,
            **state
        )
        return np.array(list(state.values()), dtype=np.float32), self.info[-1]

    def step(self, action: np.ndarray = None) -> Tuple[np.ndarray, float, bool, bool, History]:
        action = self._process_action(action)
        done, truncated = False, False
        new_price = self.portfolio.stock_price * np.exp(self.rng.normal(0, self.sigma))
        self.portfolio.update_position(new_price, action)

        state = {
            'stock_price': self.portfolio.stock_price,
            'remaining_time': self.portfolio.remaining_time,
            'stock_held': self.portfolio.stock_held,
            'strike_price': self.portfolio.strike_price
        }
        self.info.add(
            transaction_fees=self.transaction_fees,
            reward=0,
            sigma=self.sigma,
            **state
        )
        reward = self.reward_function(self.info)

        if self.portfolio.remaining_time == 1:
            reward -= self.portfolio.stock_held * new_price * self.transaction_fees
            done = True

        self.info['reward', -1] = reward

        return np.array(list(state.values()), dtype=np.float32), reward, done, truncated, self.info[-1]

    def render(self) -> None:
        pass


def make_env(epsilon, sigma, rho, action_bins, duration_bounds, seed, **kwargs):
    def _init():
        env = gym.make('OptionHedgingEnv',
                       epsilon=epsilon,
                       sigma=sigma,
                       rho=rho,
                       action_bins=action_bins,
                       duration_bounds=duration_bounds)
        env.seed(seed)
        return env
    return _init


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    sigma = 0.05
    env = OptionHedgingEnv(sigma=sigma, duration_bounds=(6, 13))
    env.seed(None)
    trials = 100
    price_series = np.zeros((trials, 20))
    for trial in range(trials):
        done, truncated = False, False
        obs, info = env.reset()
        step = 0
        price_series[trial, step] = obs[0]
        while not done and not truncated:
            step += 1
            hedge = env.portfolio.black_scholes_hedge(sigma)
            obs, reward, done, truncated, info = env.step(np.array([hedge]))
            price_series[trial, step] = obs[0]
    price_series = np.where(price_series == 0, np.nan, price_series)
    plt.plot(price_series.T)
    plt.show()

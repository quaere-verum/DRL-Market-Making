import numpy as np
import gymnasium as gym
from utils.history import History
from utils.portfolio import SimplePortfolio
from typing import Callable, Any


def base_reward_function(info):
    stock_delta = info['stock_held', -1] - info['stock_held', -2]
    option_delta = (max(info['stock_price', -1] - info['strike_price', 0], 0) -
                    max(info['stock_price', -2] - info['strike_price', 0], 0))
    transaction_cost = info['transaction_fees', 0]*info['stock_price', - 1]*np.abs(stock_delta)
    stock_pnl = info['stock_held', -2]*(info['stock_price', -1] - info['stock_price', -2])
    return -transaction_cost - option_delta + stock_pnl


class MarketMakingEnv(gym.Env):
    metadata = {'render_modes': ['logs']}

    def __init__(self, epsilon, reward_function: Callable[[Any], float] = base_reward_function, max_duration: int = 30,
                 transaction_fees: float = 0.001):
        super().__init__()
        assert epsilon >= 0
        
        self.max_duration = max_duration
        self.transaction_fees = transaction_fees
        
        self.action_space = gym.spaces.Box(low=0, high=1+epsilon)
        # self.observation_space = gym.spaces.Dict({
        #     'stock_price': gym.spaces.Box(0, 2),
        #     'remaining_time': gym.spaces.Discrete(start=1, n=max_duration+1),
        #     'stock_held': gym.spaces.Box(0, 1+epsilon),
        #     'strike_price': gym.spaces.Box(0, 2),
        #     'volatility_forecast': gym.spaces.Box(0, 1),
        #     'black_scholes_hedge': gym.spaces.Box(0, 1)
        # })
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0, 0]).astype(np.float32),
                                                high=np.array([2, max_duration+1,
                                                               1+epsilon, 2, 1, 1]).astype(np.float32),
                                                shape=(6,))

        self.sigma = 0.01
        self.portfolio = None
        self.reward_function = reward_function
        
        self.info = None
        
    def reset(self, seed=None, options=None):
        strike_price = np.random.uniform(0.5, 1.5)
        expiry_time = np.random.randint(low=2, high=self.max_duration + 1)
        self.portfolio = SimplePortfolio(strike_price=strike_price,
                                         stock_price=1,
                                         expiry_time=expiry_time,
                                         transaction_fees=self.transaction_fees)
        self.info = History(max_size=self.max_duration)
        
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
            **state
        )
        return np.array(list(state.values()), dtype=np.float32), self.info[-1]
    
    def step(self, action=None):
        action = action[0]
        done, truncated = False, False
        new_price = self.portfolio.stock_price*np.exp(np.random.normal(0, self.sigma))
        self.portfolio.update_position(new_price, action)
        black_scholes_hedge = self.portfolio.black_scholes_hedge(self.sigma)

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
            **state
        )
        reward = self.reward_function(self.info)

        if self.portfolio.remaining_time == 1:
            reward -= self.portfolio.stock_held * new_price * self.transaction_fees
            done = True
        return np.array(list(state.values()), dtype=np.float32), reward, done, truncated, self.info[-1]
    
    def render(self):
        pass

import numpy as np
import gymnasium as gym
import scipy.stats as ss

class MarketMakingEnv(gym.Env):
    metadata = {'render_modes': ['logs']}
    def __init__(self, epsilon, max_duration: int = 30, transaction_fees: float = 0.001):
        super().__init__()
        assert epsilon >= 0
        
        self.max_duration = max_duration
        self.transaction_fees = transaction_fees
        
        self.action_space = gym.spaces.Box(low=0, high=1+epsilon)
        self.observation_space = gym.spaces.Dict({
            'stock_price': gym.spaces.Box(0, 2),
            'remaining_time': gym.spaces.Discrete(max_duration+1),
            'stock_held': gym.spaces.Box(0, 1+epsilon),
            'strike_price': gym.spaces.Box(0, 2),
            'volatility_forecast': gym.spaces.Box(0, 1), 
            'black_scholes_hedge': gym.spaces.Box(0, 1)
        })
        
        self.strike_price = None
        self.stock_price = None
        self.remaining_time = None
        self.stock_held = None
        self.cash = None
        self.option_value = None
        self.sigma = 0.01
        
        self.info = None
        
    def reset(self, seed=None, options=None):
        self.strike_price = np.random.uniform(0.5, 1.5)
        self.stock_price = 1
        self.remaining_time = np.random.randint(low=1, high=self.max_duration + 1)
        self.stock_held = 0
        self.option_value = -np.max(self.stock_price - self.strike_price, 0)
        
        
        d_plus = 1/(self.sigma*np.sqrt(self.remaining_time))*(np.log(self.stock_price/self.strike_price) +
                                                 1/2*self.sigma**2/2*self.remaining_time)
        black_scholes_hedge = ss.norm.cdf(d_plus)
        
        state = {
            'stock_price': self.stock_price,
            'remaining_time': self.remaining_time,
            'stock_held': self.stock_held,
            'strike_price': self.strike_price,
            'volatility_forecast': self.sigma,
            'black_scholes_hedge': black_scholes_hedge
        }
        
        return state, self.info
    
    def step(self, action=None):
        done, truncated = False, False
        new_price = self.stock_price*np.exp(np.random.normal(0, self.sigma))
        new_option_value = -np.max(new_price - self.strike_price, 0)
        self.remaining_time -= 1
        price_delta = self.stock_held*(new_price - self.stock_price)
        option_delta = new_option_value - self.option_value
        transaction_cost = self.transaction_fees*self.stock_price*np.abs(action-self.stock_held)
        reward = -transaction_cost + option_delta + price_delta

        self.stock_price = new_price
        self.option_value = new_option_value
        self.stock_held = action
        
        if self.remaining_time == 1:
            reward -= self.stock_held*new_price*self.transaction_fees
            done = True
        
        d_plus = 1/(self.sigma*np.sqrt(self.remaining_time))*(np.log(self.stock_price/self.strike_price) +
                                                 1/2*self.sigma**2/2*self.remaining_time)
        black_scholes_hedge = ss.norm.cdf(d_plus)
        state = {
            'stock_price': self.stock_price,
            'remaining_time': self.remaining_time,
            'stock_held': self.stock_held,
            'strike_price': self.strike_price,
            'volatility_forecast': self.sigma,
            'black_scholes_hedge': black_scholes_hedge
        }
        
        return state, reward, done, truncated, self.info
    
    def render(self):
        pass
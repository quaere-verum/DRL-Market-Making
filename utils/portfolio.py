import numpy as np
import scipy.stats as ss


class SimplePortfolio:
    def __init__(self,
                 strike_price: float,
                 expiry_time: int,
                 dt: float):
        self.strike_price = strike_price
        self.stock_price = 100.
        self.remaining_time = expiry_time
        self.dt = dt
        self.stock_held = None
        self.capital = None

    def init(self, sigma, initial_stock_held):
        self.capital = self.option_valuation(sigma) - initial_stock_held*self.stock_price
        self.stock_held = initial_stock_held

    def update_position(self, stock_price, stock_held):
        self.remaining_time -= self.dt
        stock_delta = stock_held - self.stock_held
        capital_delta = -stock_delta * stock_price
        self.capital += capital_delta
        self.stock_price = stock_price
        self.stock_held = stock_held

    def black_scholes_hedge(self, sigma):
        d_plus = 1 / (sigma * np.sqrt(self.remaining_time)) * (
                    np.log(self.stock_price / self.strike_price) +
                    sigma ** 2 / 2 * self.remaining_time)
        return ss.norm.cdf(d_plus)

    def option_valuation(self, sigma):
        d_plus = (1 / (sigma * np.sqrt(self.remaining_time)) * np.log(self.stock_price / self.strike_price) +
                  sigma ** 2 / 2 * self.remaining_time)
        d_minus = d_plus - sigma * np.sqrt(self.remaining_time)
        return ss.norm.cdf(d_plus) * self.stock_price - ss.norm.cdf(d_minus) * self.strike_price

    def portfolio_valuation(self, sigma):
        option_value = self.option_valuation(sigma)
        stock_value = self.stock_price * self.stock_held
        return stock_value + self.capital - option_value

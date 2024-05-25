import numpy as np
import scipy.stats as ss


class SimplePortfolio:
    def __init__(self, strike_price, stock_price, expiry_time, dt, transaction_fees):
        self.strike_price = strike_price
        self.stock_price = stock_price
        self.remaining_time = expiry_time
        self.dt = dt
        self.stock_held = 0
        self.option_value = max(stock_price - strike_price, 0)
        self.transaction_fees = transaction_fees

    def update_position(self, stock_price, stock_held):
        self.remaining_time -= self.dt
        self.stock_price = stock_price
        self.stock_held = stock_held
        self.option_value = max(self.stock_price - self.strike_price, 0)

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

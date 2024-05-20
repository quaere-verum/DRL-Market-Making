import numpy as np
import scipy.stats as ss


class SimplePortfolio:
    def __init__(self, strike_price, stock_price, expiry_time, transaction_fees):
        self.strike_price = strike_price
        self.stock_price = stock_price
        self.remaining_time = expiry_time
        self.stock_held = 0
        self.option_value = max(stock_price - strike_price, 0)
        self.transaction_fees = transaction_fees

    def update_position(self, stock_price, stock_held):
        self.remaining_time -= 1
        self.stock_price = stock_price
        self.stock_held = stock_held
        self.option_value = np.max((self.stock_price - self.strike_price, 0))

    def black_scholes_hedge(self, sigma):
        d_plus = 1 / (sigma * np.sqrt(self.remaining_time)) * (
                    np.log(self.stock_price / self.strike_price) +
                    1 / 2 * sigma ** 2 / 2 * self.remaining_time)
        return ss.norm.cdf(d_plus)

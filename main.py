import math
import numpy as np

def simple_return(buy_price, sell_price):
    return -buy_price + sell_price

def simple_rate_of_return(buy_price, sell_price):
    return simple_return(buy_price, sell_price) / buy_price

def present_value(future_cash, fixed_interest_rate):
    return future_cash / (1 + fixed_interest_rate)

def net_present_value(cash_now, future_cash, fixed_interest_rate):
    return -cash_now + present_value(future_cash, fixed_interest_rate)

def set_probability(probability_up, probability_down):
    return np.array([probability_up, probability_down])

def calc_probability_down(probability_up):
    return 1 - probability_up

def set_uncertein_sell_price(sell_price_up, sell_price_down):
    return np.array([sell_price_up, sell_price_down])

# The expectation is the dot product between probability and sell price
# expectation = (probability_up * sell_price_up) + (probability_down * sell_price_down)
def expectation(probability, sell_price):
    return np.dot(probability, sell_price)

def expected_return(buy_price, probability, sell_price):
    simple_return = np.array([sell_price[0] - buy_price, sell_price[1] - buy_price])
    return np.dot(probability, simple_return)

def expected_rate_of_return(buy_price, probability, sell_price):
    return expected_return(buy_price, probability, sell_price) / buy_price

def variance(buy_price, probability, sell_price):
    uncertein_simple_rate_of_return = np.array([simple_rate_of_return(buy_price, sell_price[0]),
                                                simple_rate_of_return(buy_price, sell_price[1])])
    mu = expected_rate_of_return(buy_price, probability, sell_price)
    deviation = (uncertein_simple_rate_of_return - mu) ** 2
    return np.dot(probability, deviation)

# Volatility is standard deviation of the expected rate of return
# Which is square root of variance
def volatility(varian):
    return math.sqrt(varian)

if __name__ == "__main__":
    buy_price = 10
    probability = set_probability(0.3, 0.7)
    future_sell_prices = set_uncertein_sell_price(30, 20)
    var = variance(buy_price, probability, future_sell_prices)
    print(var)
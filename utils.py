""" Utility Functions """
import random

import numpy as np
from scipy.stats import norm

random.seed(1)

def brownian_sim(num_path, num_period, mu, std, init_p, dt):
    z = np.random.normal(size=(num_path, num_period))

    a_price = np.zeros((num_path, num_period))
    a_price[:, 0] = init_p

    for t in range(num_period - 1):
        a_price[:, t + 1] = a_price[:, t] * np.exp(
            (mu - (std ** 2) / 2) * dt + std * np.sqrt(dt) * z[:, t]
        )
    return a_price


# BSM Call Option Pricing Formula & BS Delta formula
# T here is time to maturity
def bs_call(iv, T, S, K, r, q):
    d1 = (np.log(S / K) + (r - q + iv * iv / 2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    bs_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    bs_delta = np.exp(-q * T) * norm.cdf(d1)
    return bs_price, bs_delta


def get_sim_path(M, freq, np_seed, num_sim):
    """ Return simulated data: a tuple of three arrays
        M: initial time to maturity
        freq: trading freq in unit of day, e.g. freq=2: every 2 day; freq=0.5 twice a day;
        np_seed: numpy random seed
        num_sim: number of simulation path

        1) asset price paths (num_path x num_period)
        2) option price paths (num_path x num_period)
        3) delta (num_path x num_period)
    """
    # set the np random seed
    np.random.seed(np_seed)

    # Trading Freq per day; passed from function parameter
    # freq = 2

    # Annual Trading Day
    T = 250

    # Simulation Time Step
    dt = 0.004 * freq

    # Option Day to Maturity; passed from function parameter
    # M = 60

    # Number of period
    num_period = int(M / freq)

    # Number of simulations; passed from function parameter
    # num_sim = 1000000

    # Annual Return
    mu = 0.05

    # Annual Volatility
    vol = 0.2

    # Initial Asset Value
    S = 100

    # Option Strike Price
    K = 100

    # Annual Risk Free Rate
    r = 0

    # Annual Dividend
    q = 0

    # asset price 2-d array
    print("1. generate asset price paths")
    a_price = brownian_sim(num_sim, num_period + 1, mu, vol, S, dt)

    # time to maturity "rank 1" array: e.g. [M, M-1, ..., 0]
    ttm = np.arange(M, -freq, -freq)

    # BS price 2-d array and bs delta 2-d array
    print("2. generate BS price and delta")
    bs_price, bs_delta = bs_call(vol, ttm / T, a_price, K, r, q)

    print("simulation done!")

    return a_price, bs_price, bs_delta


def sabr_sim(num_path, num_period, mu, std, init_p, dt, rho, beta, volvol):
    qs = np.random.normal(size=(num_path, num_period))
    qi = np.random.normal(size=(num_path, num_period))
    qv = rho * qs + np.sqrt(1 - rho * rho) * qi

    vol = np.zeros((num_path, num_period))
    vol[:, 0] = std

    a_price = np.zeros((num_path, num_period))
    a_price[:, 0] = init_p

    for t in range(num_period - 1):
        gvol = vol[:, t] * (a_price[:, t] ** (beta - 1))
        a_price[:, t + 1] = a_price[:, t] * np.exp(
            (mu - (gvol ** 2) / 2) * dt + gvol * np.sqrt(dt) * qs[:, t]
        )
        vol[:, t + 1] = vol[:, t] * np.exp(
            -volvol * volvol * 0.5 * dt + volvol * qv[:, t] * np.sqrt(dt)
        )

    return a_price, vol


def sabr_implied_vol(vol, T, S, K, r, q, beta, volvol, rho):

    F = S * np.exp((r - q) * T)
    x = (F * K) ** ((1 - beta) / 2)
    y = (1 - beta) * np.log(F / K)
    A = vol / (x * (1 + y * y / 24 + y * y * y * y / 1920))
    B = 1 + T * (
        ((1 - beta) ** 2) * (vol * vol) / (24 * x * x)
        + rho * beta * volvol * vol / (4 * x)
        + volvol * volvol * (2 - 3 * rho * rho) / 24
    )
    Phi = (volvol * x / vol) * np.log(F / K)
    Chi = np.log((np.sqrt(1 - 2 * rho * Phi + Phi * Phi) + Phi - rho) / (1 - rho))

    SABRIV = np.where(F == K, vol * B / (F ** (1 - beta)), A * B * Phi / Chi)

    return SABRIV


def bartlett(sigma, T, S, K, r, q, ds, beta, volvol, rho):

    dsigma = ds * volvol * rho / (S ** beta)

    vol1 = sabr_implied_vol(sigma, T, S, K, r, q, beta, volvol, rho)
    vol2 = sabr_implied_vol(sigma + dsigma, T, S + ds, K, r, q, beta, volvol, rho)

    bs_price1, _ = bs_call(vol1, T, S, K, r, q)
    bs_price2, _ = bs_call(vol2, T, S+ds, K, r, q)

    b_delta = (bs_price2 - bs_price1) / ds

    return b_delta

def bs_call(iv, T, S, K, r, q):
    d1 = (np.log(S / K) + (r - q + iv * iv / 2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    bs_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    bs_delta = np.exp(-q * T) * norm.cdf(d1)
    return bs_price, bs_delta


def get_sim_path_sabr(M, freq, np_seed, num_sim):
    """ Return simulated data: a tuple of four arrays
        M: initial time to maturity
        freq: trading freq in unit of day, e.g. freq=2: every 2 day; freq=0.5 twice a day;
        np_seed: numpy random seed
        num_sim: number of simulation path

        1) asset price paths (num_path x num_period)
        2) option price paths (num_path x num_period)
        3) bs delta (num_path x num_period)
        4) bartlett delta (num_path x num_period)
    """
    # set the np random seed
    np.random.seed(np_seed)

    # Trading Freq per day; passed from function parameter
    # freq = 2

    # Annual Trading Day
    T = 250

    # Simulation Time Step
    dt = 0.004 * freq

    # Option Day to Maturity; passed from function parameter
    # M = 60

    # Number of period
    num_period = int(M / freq)

    # Number of simulations; passed from function parameter
    # num_sim = 1000000

    # Annual Return
    mu = 0.05

    # Annual Volatility
    vol = 0.2

    # Initial Asset Value
    S = 100

    # Option Strike Price
    K = 100

    # Annual Risk Free Rate
    r = 0

    # Annual Dividend
    q = 0

    # SABR parameters
    beta = 1
    rho = -0.4
    volvol = 0.6
    ds = 0.001

    # asset price 2-d array; sabr_vol
    print("1. generate asset price paths (sabr)")
    a_price, sabr_vol = sabr_sim(
        num_sim, num_period + 1, mu, vol, S, dt, rho, beta, volvol
    )

    # time to maturity "rank 1" array: e.g. [M, M-1, ..., 0]
    ttm = np.arange(M, -freq, -freq)

    # BS price 2-d array and bs delta 2-d array
    print("2. generate BS price, BS delta, and Bartlett delta")

    # sabr implied vol
    implied_vol = sabr_implied_vol(
        sabr_vol, ttm / T, a_price, K, r, q, beta, volvol, rho
    )

    bs_price, bs_delta = bs_call(implied_vol, ttm / T, a_price, K, r, q)

    bartlett_delta = bartlett(sabr_vol, ttm / T, a_price, K, r, q, ds, beta, volvol, rho)

    print("simulation done!")

    return a_price, bs_price, bs_delta, bartlett_delta

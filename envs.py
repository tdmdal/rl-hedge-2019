"""A trading environment"""
import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

from utils import get_sim_path, get_sim_path_sabr


class TradingEnv(gym.Env):
    """
    trading environment;
    """

    # trade_freq in unit of day, e.g 2: every 2 day; 0.5 twice a day;
    def __init__(self, cash_flow_flag=0, dg_random_seed=1, num_sim=500002, sabr_flag = False,
        continuous_action_flag=False, spread=0, init_ttm=5, trade_freq=1, num_contract=1):

        # simulated data: array of asset price, option price and delta paths (num_path x num_period)
        # generate data now
        if sabr_flag:
            self.path, self.option_price_path, self.delta_path, self.bartlett_delta_path = get_sim_path_sabr(M=init_ttm, freq=trade_freq,
                np_seed=dg_random_seed, num_sim=num_sim)
        else:
            self.path, self.option_price_path, self.delta_path = get_sim_path(M=init_ttm, freq=trade_freq,
                np_seed=dg_random_seed, num_sim=num_sim)

        # other attributes
        self.num_path = self.path.shape[0]

        # set num_period: initial time to maturity * daily trading freq + 1 (see get_sim_path() in utils.py)
        self.num_period = self.path.shape[1]
        # print("***", self.num_period)

        # time to maturity array
        self.ttm_array = np.arange(init_ttm, -trade_freq, -trade_freq)
        # print(self.ttm_array)

        # spread
        self.spread = spread

        # step function initialization depending on cash_flow_flag
        if cash_flow_flag == 1:
            self.step = self.step_cash_flow
        else:
            self.step = self.step_profit_loss

        self.num_contract = num_contract
        self.strike_price = 100

        # track the index of simulated path in use
        self.sim_episode = -1

        # track time step within an episode (it's step)
        self.t = None

        # action space
        if continuous_action_flag:
            self.action_space = spaces.Box(low=np.array([0]), high=np.array([num_contract * 100]), dtype=np.float32)
        else:
            self.num_action = num_contract * 100 + 1
            self.action_space = spaces.Discrete(self.num_action)

        self.num_state = 3

        self.state = []

        # seed and start
        self.seed()
        # self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # repeatedly go through available simulated paths (if needed)
        self.sim_episode = (self.sim_episode + 1) % self.num_path

        self.t = 0

        price = self.path[self.sim_episode, self.t]
        position = 0

        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]

        return self.state

    def step_cash_flow(self, action):
        """
        cash flow period reward
        """

        # do it consistently as in the profit & loss case
        # current prices (at t)
        current_price = self.state[0]

        # current position
        current_position = self.state[1]

        # update time/period
        self.t = self.t + 1

        # get state for tomorrow
        price = self.path[self.sim_episode, self.t]
        position = action
        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]

        # calculate period reward (part 1)
        cash_flow = -(position - current_position) * current_price - np.abs(position - current_position) * current_price * self.spread

        # if tomorrow is end of episode
        if self.t == self.num_period - 1:
            done = True
            # add (stock payoff + option payoff) to cash flow
            reward = cash_flow + price * position - max(price - self.strike_price, 0) * self.num_contract * 100 - position * price * self.spread
        else:
            done = False
            reward = cash_flow

        # for other info
        info = {"path_row": self.sim_episode}

        return self.state, reward, done, info

    def step_profit_loss(self, action):
        """
        profit loss period reward
        """

        # current prices (at t)
        current_price = self.state[0]
        current_option_price = self.option_price_path[self.sim_episode, self.t]

        # current position
        current_position = self.state[1]

        # update time
        self.t = self.t + 1

        # get state for tomorrow (at t + 1)
        price = self.path[self.sim_episode, self.t]
        option_price = self.option_price_path[self.sim_episode, self.t]
        position = action
        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]

        # calculate period reward (part 1)
        reward = (price - current_price) * position - np.abs(current_position - position) * current_price * self.spread

        # if tomorrow is end of episode
        if self.t == self.num_period - 1:
            done = True
            reward = reward - (max(price - self.strike_price, 0) - current_option_price) * self.num_contract * 100 - position * price * self.spread
        else:
            done = False
            reward = reward - (option_price - current_option_price) * self.num_contract * 100

        # for other info later
        info = {"path_row": self.sim_episode}

        return self.state, reward, done, info

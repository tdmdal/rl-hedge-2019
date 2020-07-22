import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# put things common to different algorithms here
class DRL:
    def __init__(self):
        if not os.path.exists('model'):
            os.mkdir('model')

        if not os.path.exists('history'):
            os.mkdir('history')

    def test(self, total_episode, delta_flag=False, bartlett_flag=False):
        """hedge with model.
        """
        print('testing...')

        self.epsilon = -1

        w_T_store = []

        for i in range(total_episode):
            observation = self.env.reset()
            done = False
            action_store = []
            reward_store = []

            while not done:

                # prepare state
                x = np.array(observation).reshape(1, -1)

                if delta_flag:
                    action = self.env.delta_path[i % self.env.num_path, self.env.t] * self.env.num_contract * 100
                elif bartlett_flag:
                    action = self.env.bartlett_delta_path[i % self.env.num_path, self.env.t] * self.env.num_contract * 100
                else:
                    # choose action from epsilon-greedy; epsilon has been set to -1
                    action, _, _ = self.egreedy_action(x)

                # store action to take a look
                action_store.append(action)

                # a step
                observation, reward, done, info = self.env.step(action)
                reward_store.append(reward)

            # get final wealth at the end of episode, and store it.
            w_T = sum(reward_store)
            w_T_store.append(w_T)

            if i % 1000 == 0:
                w_T_mean = np.mean(w_T_store)
                w_T_var = np.var(w_T_store)
                path_row = info["path_row"]
                print(info)
                with np.printoptions(precision=2, suppress=True):
                    print("episode: {} | final wealth: {:.2f}; so far mean and variance of final wealth was {} and {}".format(i, w_T, w_T_mean, w_T_var))
                    print("episode: {} | so far Y(0): {:.2f}".format(i, -w_T_mean + self.ra_c * np.sqrt(w_T_var)))
                    print("episode: {} | rewards: {}".format(i, np.array(reward_store)))
                    print("episode: {} | action taken: {}".format(i, np.array(action_store)))
                    print("episode: {} | deltas {}".format(i, self.env.delta_path[path_row] * 100))
                    print("episode: {} | stock price {}".format(i, self.env.path[path_row]))
                    print("episode: {} | option price {}\n".format(i, self.env.option_price_path[path_row] * 100))

    def plot(self, history):
        pass

    def save_history(self, history, name):
        name = os.path.join('history', name)

        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')
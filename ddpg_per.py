import os
import numpy as np

import tensorflow as tf

from keras.layers import Input, Dense, Lambda, concatenate, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

from drl import DRL
from envs import TradingEnv
from replay_buffer import PrioritizedReplayBuffer
from schedules import LinearSchedule


class DDPG(DRL):
    """
    Deep Deterministic Policy Gradient
    """

    def __init__(self, env):
        super(DDPG, self).__init__()

        self.sess = K.get_session()

        self.env = env
        self.upper_bound = self.env.action_space.high[0]
        self.lower_bound = self.env.action_space.low[0]

        # update rate for target model.
        # for 2nd round training, use 0.000001 
        self.TAU = 0.00001

        # learning rate for actor and critic
        # for 2nd round training, use 1e-5
        self.actor_lr = 1e-4
        self.critic_lr = 1e-4

        # risk averse constant
        self.ra_c = 1.5

        # actor: policy function
        # critic: Q functions; Q_ex, Q_ex2, and Q
        self.actor = self._build_actor(learning_rate=self.actor_lr)
        self.critic_Q_ex, self.critic_Q_ex2, self.critic_Q = self._build_critic(learning_rate=self.critic_lr)

        self.critic_Q.summary()

        # target networks for actor and three critics
        self.actor_hat = self._build_actor(learning_rate=self.actor_lr)
        self.actor_hat.set_weights(self.actor.get_weights())

        self.critic_Q_ex_hat, self.critic_Q_ex2_hat, self.critic_Q_hat = self._build_critic(learning_rate=self.critic_lr)
        self.critic_Q_ex_hat.set_weights(self.critic_Q_ex.get_weights())
        self.critic_Q_ex2_hat.set_weights(self.critic_Q_ex2.get_weights())

        # epsilon of epsilon-greedy
        self.epsilon = 1.0

        # discount rate for epsilon
        self.epsilon_decay = 0.99994
        # self.epsilon_decay = 0.9994

        # min epsilon of epsilon-greedy.
        self.epsilon_min = 0.1

        # memory buffer for experience replay
        buffer_size = 600000
        prioritized_replay_alpha = 0.6
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)

        prioritized_replay_beta0 = 0.4

        # need not be the same as training episode (see schedules.py)
        prioritized_replay_beta_iters = 50001

        self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)

        # for numerical stabiligy
        self.prioritized_replay_eps = 1e-6

        self.t = None

        # memory sample batch size
        self.batch_size = 128

        # may use for 2nd round training
        # self.policy_noise = 5
        # self.noise_clip = 5

        # gradient function
        self.get_critic_grad = self.critic_gradient()
        self.actor_optimizer()

    def load(self, tag=""):
        """load two Qs for test"""
        if tag == "":
            actor_file = "model/ddpg_actor.h5"
            critic_Q_ex_file = "model/ddpg_critic_Q_ex.h5"
            critic_Q_ex2_file = "model/ddpg_critic_Q_ex2.h5"
        else:
            actor_file = "model/ddpg_actor_" + tag + ".h5"
            critic_Q_ex_file = "model/ddpg_critic_Q_ex_" + tag + ".h5"
            critic_Q_ex2_file = "model/ddpg_critic_Q_ex2_" + tag + ".h5"

        if os.path.exists(actor_file):
            self.actor.load_weights(actor_file)
            self.actor_hat.load_weights(actor_file)
        if os.path.exists(critic_Q_ex_file):
            self.critic_Q_ex.load_weights(critic_Q_ex_file)
            self.critic_Q_ex_hat.load_weights(critic_Q_ex_file)
        if os.path.exists(critic_Q_ex2_file):
            self.critic_Q_ex2.load_weights(critic_Q_ex2_file)
            self.critic_Q_ex2_hat.load_weights(critic_Q_ex2_file)

    def _build_actor(self, learning_rate=1e-3):
        """basic NN model.
        """
        inputs = Input(shape=(self.env.num_state,))

        # bn after input
        x = BatchNormalization()(inputs)

        # bn after activation
        x = Dense(32, activation="relu")(x)
        x = BatchNormalization()(x)

        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)

        # no bn for output layer
        x = Dense(1, activation="sigmoid")(x)

        output = Lambda(lambda x: x * self.env.num_contract * 100)(x)

        model = Model(inputs=inputs, outputs=output)

        # compile the model using mse loss, but won't use mse to train
        model.compile(loss="mse", optimizer=Adam(learning_rate))

        return model

    def _build_critic(self, learning_rate=1e-3):
        """basic NN model.
        """
        # inputs
        s_inputs = Input(shape=(self.env.num_state,))
        a_inputs = Input(shape=(1,))

        # combine inputs
        x = concatenate([s_inputs, a_inputs])

        # bn after input
        x = BatchNormalization()(x)
        
        # Q_ex network

        # bn after activation
        x1 = Dense(32, activation="relu")(x)
        x1 = BatchNormalization()(x1)

        x1 = Dense(64, activation="relu")(x1)
        x1 = BatchNormalization()(x1)

        # no bn for output layer
        output1 = Dense(1, activation="linear")(x1)

        model_Q_ex = Model(inputs=[s_inputs, a_inputs], outputs=output1)
        model_Q_ex.compile(loss="mse", optimizer=Adam(learning_rate))

        # Q_ex2 network

        # bn after activation
        x2 = Dense(32, activation="relu")(x)
        x2 = BatchNormalization()(x2)

        # bn after activation
        x2 = Dense(64, activation="relu")(x2)
        x2 = BatchNormalization()(x2)

        # no bn for output layer
        output2 = Dense(1, activation="linear")(x2)

        model_Q_ex2 = Model(inputs=[s_inputs, a_inputs], outputs=output2)
        model_Q_ex2.compile(loss="mse", optimizer=Adam(learning_rate))

        # Q
        output3 = Lambda(lambda o: o[0] - self.ra_c * K.sqrt(K.max(o[1] - o[0] * o[0], 0)))([output1, output2])
        model_Q = Model(inputs=[s_inputs, a_inputs], outputs=output3)
        model_Q.compile(loss="mse", optimizer=Adam(learning_rate))

        return model_Q_ex, model_Q_ex2, model_Q

    def actor_optimizer(self):
        """actor_optimizer.
        Returns:
            function, opt function for actor.
        """
        self.ainput = self.actor.input
        aoutput = self.actor.output
        trainable_weights = self.actor.trainable_weights
        self.action_gradient = tf.placeholder(tf.float32, shape=(None, 1))

        # tf.gradients calculates dy/dx with a initial gradients for y
        # action_gradient is dq/da, so this is dq/da * da/dparams
        params_grad = tf.gradients(aoutput, trainable_weights, -self.action_gradient)
        grads = zip(params_grad, trainable_weights)
        self.opt = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def critic_gradient(self):
        """get critic gradient function.
        Returns:
            function, gradient function for critic.
        """
        cinput = self.critic_Q.input
        coutput = self.critic_Q.output

        # compute the gradient of the action with q value, dq/da.
        action_grads = K.gradients(coutput, cinput[1])

        return K.function([cinput[0], cinput[1]], action_grads)

    def egreedy_action(self, X):
        """get actor action with ou noise.
        Arguments:
            X: state value.
        """
        # do the epsilon greedy way; not using OU
        if np.random.rand() <= self.epsilon:
            action = env.action_space.sample()

            # may use for 2nd round training
            # action = self.actor.predict(X)[0][0]
            # noise = np.clip(np.random.normal(0, self.policy_noise), -self.noise_clip, self.noise_clip)
            # action = np.clip(action + noise, 0, self.env.num_contract * 100)
        else:
            action = self.actor.predict(X)[0][0]

        return action, None, None

    def update_epsilon(self):
        """update epsilon
        """
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        """add data to experience replay.
        Arguments:
            state: observation
            action: action
            reward: reward
            next_state: next_observation
            done: if game is done.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def process_batch(self, batch_size):
        """process batch data
        Arguments:
            batch: batch size
        Returns:
            states: batch of states
            actions: batch of actions
            target_q_ex, target_q_ex2: batch of targets;
            weights: priority weights
        """
        # prioritized sample from experience replay buffer
        experience = self.replay_buffer.sample(batch_size, beta=self.beta_schedule.value(self.t))
        (states, actions, rewards, next_states, dones, weights, batch_idxes) = experience

        actions = actions.reshape(-1, 1)
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        # get next_actions
        next_actions = self.actor_hat.predict(next_states)

        # prepare targets for Q_ex and Q_ex2 training
        q_ex_next = self.critic_Q_ex_hat.predict([next_states, next_actions])
        q_ex2_next = self.critic_Q_ex2_hat.predict([next_states, next_actions])

        target_q_ex = rewards + (1 - dones) * q_ex_next
        target_q_ex2 = rewards ** 2 + (1 - dones) * (2 * rewards * q_ex_next + q_ex2_next)

        # use Q2 TD error as priority weight
        td_errors = self.critic_Q_ex2.predict([states, actions]) - target_q_ex2
        new_priorities = (np.abs(td_errors) + self.prioritized_replay_eps).flatten()
        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

        return states, actions, target_q_ex, target_q_ex2, weights

    def update_model(self, X1, X2, y1, y2, weights):
        """update ddpg model.
        Arguments:
            X1: states
            X2: actions
            y1: target for Q_ex
            y2: target for Q_ex2
            weights: priority weights
        Returns:
            loss_ex: critic Q_ex loss
            loss_ex2: critic Q_ex2 loss
        """
        # flatten to prepare for training with weights
        weights = weights.flatten()

        # default batch size is 32
        loss_ex = self.critic_Q_ex.fit([X1, X2], y1, sample_weight=weights, verbose=0)
        loss_ex = np.mean(loss_ex.history['loss'])

        # default batch size is 32
        loss_ex2 = self.critic_Q_ex2.fit([X1, X2], y2, sample_weight=weights, verbose=0)
        loss_ex2 = np.mean(loss_ex2.history['loss'])

        X3 = self.actor.predict(X1)

        a_grads = np.array(self.get_critic_grad([X1, X3]))[0]
        self.sess.run(self.opt, feed_dict={
            self.ainput: X1,
            self.action_gradient: a_grads
        })

        return loss_ex, loss_ex2

    def update_target_model(self):
        """soft update target model.
        """
        critic_Q_ex_weights = self.critic_Q_ex.get_weights()
        critic_Q_ex2_weights = self.critic_Q_ex2.get_weights()
        actor_weights = self.actor.get_weights()

        critic_Q_ex_hat_weights = self.critic_Q_ex_hat.get_weights()
        critic_Q_ex2_hat_weights = self.critic_Q_ex2_hat.get_weights()
        actor_hat_weights = self.actor_hat.get_weights()

        for i in range(len(critic_Q_ex_weights)):
            critic_Q_ex_hat_weights[i] = self.TAU * critic_Q_ex_weights[i] + (1 - self.TAU) * critic_Q_ex_hat_weights[i]

        for i in range(len(critic_Q_ex2_weights)):
            critic_Q_ex2_hat_weights[i] = self.TAU * critic_Q_ex2_weights[i] + (1 - self.TAU) * critic_Q_ex2_hat_weights[i]

        for i in range(len(actor_weights)):
            actor_hat_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_hat_weights[i]

        self.critic_Q_ex_hat.set_weights(critic_Q_ex_hat_weights)
        self.critic_Q_ex2_hat.set_weights(critic_Q_ex2_hat_weights)
        self.actor_hat.set_weights(actor_hat_weights)

    def train(self, episode):
        """training
        Arguments:
            episode: total episodes to run

        Returns:
            history: training history
        """

        # some statistics
        history = {"episode": [], "episode_w_T": [], "loss_ex": [], "loss_ex2": []}

        for i in range(episode):
            observation = self.env.reset()
            done = False

            # for recording purpose
            y_action = np.empty(0, dtype=int)
            reward_store = np.empty(0)

            self.t = i

            # steps in an episode
            while not done:

                # prepare state
                x = np.array(observation).reshape(1, -1)

                # chocie action from epsilon-greedy.
                action, _, _ = self.egreedy_action(x)

                # one step
                observation, reward, done, info = self.env.step(action)

                # record action and reward
                y_action = np.append(y_action, action)
                reward_store = np.append(reward_store, reward)

                # store to memory
                self.remember(x[0], action, reward, observation, done)

                if len(self.replay_buffer) > self.batch_size:

                    # draw from memory
                    X1, X2, y_ex, y_ex2, weights = self.process_batch(self.batch_size)

                    # update model
                    loss_ex, loss_ex2 = self.update_model(X1, X2, y_ex, y_ex2, weights)

                    # soft update target
                    self.update_target_model()

            # reduce epsilon per episode
            self.update_epsilon()

            # print/store some statistics every 1000 episodes
            if i % 1000 == 0 and i != 0:

            # may want to print/store some statistics every 100 episodes
            # if i % 100 == 0 and i >= 1000:

                # get w_T for statistics
                w_T = np.sum(reward_store)

                history["episode"].append(i)
                history["episode_w_T"].append(w_T)
                history["loss_ex"].append(loss_ex)
                history["loss_ex2"].append(loss_ex2)

                path_row = info["path_row"]
                print(info)
                print(
                    "episode: {} | episode final wealth: {:.3f} | loss_ex: {:.3f} | loss_ex2: {:.3f} | epsilon:{:.2f}".format(
                        i, w_T, loss_ex, loss_ex2, self.epsilon
                    )
                )

                with np.printoptions(precision=2, suppress=True):
                    print("episode: {} | rewards {}".format(i, reward_store))
                    print("episode: {} | actions taken {}".format(i, y_action))
                    print("episode: {} | deltas {}".format(i, self.env.delta_path[path_row] * 100))
                    print("episode: {} | stock price {}".format(i, self.env.path[path_row]))
                    print("episode: {} | option price {}\n".format(i, self.env.option_price_path[path_row] * 100))

                # may want to save model every 100 episode
                # if i % 100 == 0:
                #     self.actor.save_weights("model/ddpg_actor_" + str(int(i/100)) + ".h5")
                #     self.critic_Q_ex.save_weights("model/ddpg_critic_Q_ex_" + str(int(i/100)) + ".h5")
                #     self.critic_Q_ex2.save_weights("model/ddpg_critic_Q_ex2_" + str(int(i/100)) + ".h5")
                self.actor.save_weights("model/ddpg_actor_" + str(int(i/1000)) + ".h5")
                self.critic_Q_ex.save_weights("model/ddpg_critic_Q_ex_" + str(int(i/1000)) + ".h5")
                self.critic_Q_ex2.save_weights("model/ddpg_critic_Q_ex2_" + str(int(i/1000)) + ".h5")

        # save weights once training is done
        self.actor.save_weights("model/ddpg_actor.h5")
        self.critic_Q_ex.save_weights("model/ddpg_critic_Q_ex.h5")
        self.critic_Q_ex2.save_weights("model/ddpg_critic_Q_ex2.h5")

        return history

if __name__ == "__main__":

    # disable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # setup for training
    # use init_ttm, spread and other arguments to train for different scenarios
    env = TradingEnv(continuous_action_flag=True, sabr_flag=True, dg_random_seed=1, init_ttm=20, trade_freq=1, spread=0.01, num_contract=1, num_sim=50002)
    ddpg = DDPG(env)

    # for second round training, specify the tag of weights to load
    # ddpg.load(tag="50")

    # for second round training, may want to start with a specific value of epsilon
    # ddpg.epsilon = 0.1

    # episode for training: 0 to 50000 inclusive
    # cycle through available data paths if number of episode for training > number of sim paths
    history = ddpg.train(50001)
    ddpg.save_history(history, "ddpg.csv")

    # setup for testing; use another instance for testing
    env_test = TradingEnv(continuous_action_flag=True, sabr_flag=True, dg_random_seed=2, init_ttm=20, trade_freq=1, spread=0.01, num_contract=1, num_sim=100001)
    ddpg_test = DDPG(env_test)
    ddpg_test.load()

    # episode for testing: 0 to 100000 inclusive
    ddpg_test.test(100001)
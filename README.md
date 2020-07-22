# Deep Hedging with Reinforcement Learning

## About

This is the companion code for the paper *Deep Hedging of Derivatives Using Reinforcement Learning* by Jay Cao, Jacky Chen, John Hull, and Zissis Poulos. The paper is available [here](https://ssrn.com/abstract=3514586) at SSRN.

## Requirement

The code requires gym (0.12.1), tensorflow (1.13.1), and keras (2.3.1).

## Usage

Run `python ddpg_per.py` to start training. Run `python ddpg_test.py` to test a trained model.

To setup a trading scenario for training and testing, modify the trading environment instantiation parameter values in the code accordingly (`env = TradingEnv(...)` and `env_test = TradingEnv(...)` ). 

## Weights files

Trained weights for all trading scenarios in the paper are provided in the `weights` folder.

Each set of weights are obtained after 2 or 3 rounds of trainings. Later round of trainings start with the best weights obtained from the previous round together with manually fine-tuned hyper-parameter values (learning rate, target network soft update rate, etc. See comments in the code for details.)

## Credits

* The code structure is adapted from [@xiaochus](https://github.com/xiaochus)'s github project [Deep-Reinforcement-Learning-Practice](https://github.com/xiaochus/Deep-Reinforcement-Learning-Practice).

* The implementation of prioritized experience replay buffer is taken from OpenAI [Baselines](https://github.com/openai/baselines).

"""
test and visualize trained cartpole agents
"""
import numpy as np
import torch
import pandas as pd
import functools
import os

from environment import ForexEnv
from agents import RandomAgent
from agents import DQNAgent
from agents import Forex_reward_function
from feature import ForexIdentityFeature

def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_episode_reward(index_history, price_record_history, action_history):
    action_history.insert(0,1)
    print(len(action_history) == len(index_history))
    print(len(price_record_history) == len(index_history))
    for time in range(1,len(action_history)):
        prev_position = action_history[time - 1] -1
        if time == len(action_history) - 1:
            position = 0
        else:
            position = action_history[time] -1
        bid, ask, next_bid, next_ask = price_record_history[time]

    return 0

def test(agent, environment, num_episodes, max_timesteps, log_path = ''):
    action_data = []
    rewards = []
    print_actions = []

    for episode in range(num_episodes):
        agent.reset_cumulative_reward()
        new_env = environment.reset_fixed(episode * 3600)
        observation_history = [(new_env[0], new_env[1], new_env[2], False)]
        price_record_history =[new_env[2]]
        index_history = [new_env[0]]
        action_history = []

        t = 0
        done = False
        while not done:
            action = agent.act(observation_history, action_history)
            timestamp, state, price_record, done = environment.step(action)
            index_history.append(timestamp)
            price_record_history.append(price_record)
            action_history.append(action)
            observation_history.append((timestamp, state, price_record, done))
            t += 1
            done = done or (t == max_timesteps)



        logging('ep id  '+ str(episode) + '    1-hour est. reward ' +str(np.sum(np.array(rewards))),log_path)
        logging('short  ' + str(print_actions.count(0)), log_path)
        logging('neutral  '+ str(print_actions.count(1)), log_path)
        logging('long  '+ str(print_actions.count(2)), log_path)

        print_actions = []

    return observation_data, action_data, rewards

if __name__=='__main__':
    dqn_model_path = './AUDUSD/agents/20190526-174236/dqn|0.pt'
    log_path = dqn_model_path[:-8] +'test.txt'

    eps = 50
    max_timesteps = 3600
    hidden = [50,50] # Or [64,32]

    np.random.seed(321)
    torch.manual_seed(123)

    env = ForexEnv(mode = 'eval')

    agent = DQNAgent(
        action_set=[0, 1, 2],
        reward_function=functools.partial(Forex_reward_function),
        feature_extractor=ForexIdentityFeature(),
        hidden_dims=hidden,
        test_model_path=dqn_model_path)

    observation_data, action_data, rewards = test(agent=agent,
                                                  environment=env,
                                                  num_episodes=eps,
                                                  max_timesteps=max_timesteps,
                                                  log_path = log_path)




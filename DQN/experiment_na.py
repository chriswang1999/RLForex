import numpy as np
import torch
import os
import functools

# import matplotlib.pyplot as plt
from tqdm import trange

from live import live
from environment import ForexEnv
from agents_na import RandomAgent
from agents_na import DQNAgent
from agents_na import Forex_reward_function
from feature import ForexIdentityFeature
import time

if __name__ == '__main__':
    cur = 'AUDUSD'
    reward_path = './'+ cur +'/results/'+ time.strftime("%Y%m%d-%H%M%S") +'/'
    agent_path = './'+ cur +'/agents/' + time.strftime("%Y%m%d-%H%M%S") +'/'
    log_path = './'+ cur +'/log/' + time.strftime("%Y%m%d-%H%M%S") +'/'

    if not os.path.exists(reward_path):
        os.makedirs(reward_path)
    if not os.path.exists(agent_path):
        os.makedirs(agent_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    env = ForexEnv(mode = 'train')
    env_test = ForexEnv(mode = 'eval')

    # train dqn agents
    number_seeds = 3
    for seed in trange(number_seeds):
        np.random.seed(seed+1)
        torch.manual_seed(seed+2)

        agent = DQNAgent(
            action_set=[0, 1, 2],
            reward_function=functools.partial(Forex_reward_function),
            feature_extractor=ForexIdentityFeature(),
            hidden_dims=[64, 32],
            learning_rate=0.000025,
            buffer_size=5000,
            batch_size=16,
            num_batches=50,
            starts_learning=1000,
            final_epsilon=0.02,
            discount=0.9,
            target_freq=10,
            verbose=True,
            print_every=10,
            log_path = log_path)

        _, _, rewards = live(
            agent=agent,
            environment=env,
            test_environment=env_test,
            num_episodes=3000,
            max_timesteps=3600,
            verbose=True,
            print_every=100,
            log_path = log_path)

        file_name = '|'.join(['dqn', str(seed)])
        np.save(os.path.join(reward_path, file_name), rewards)
        agent.save(path=os.path.join(agent_path, file_name+'.pt'))

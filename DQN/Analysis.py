import os
import numpy as np
import pandas as pd
import functools
import torch
from environment_16 import ForexEnv
from agents import DQNAgent
from agents import Forex_reward_function
from feature_16 import ForexIdentityFeature

def get_reward(index_history, price_record_history, action_history):
    accumulative_reward = 0
    assert (len(action_history) == len(index_history)-1)
    assert (len(price_record_history) == len(index_history))
    timespan = len(action_history)
    for t in range(timespan):
        if t == 0:
            prev_position = 0
        else:
            prev_position = action_history[t - 1] -1
        if t == len(action_history) - 1:
            position = 0
        else:
            position = action_history[t] -1
        bid, ask, next_bid, next_ask = price_record_history[t]
        
        action = position - prev_position
        price = 0
        
        if action > 0:
            price = next_ask
        elif action < 0:
            price = next_bid
        reward = torch.sum(torch.tensor(-1.).float() * action * price).to(device)
        accumulative_reward += reward
    return index_history[0], accumulative_reward 

if __name__ == '__main__':

	HOURLY_RF_RATE = 2.68 * 10 / 365.0 / 24.0

	num_episodes = 50
	env = ForexEnv(mode = 'eval')
	max_timesteps = 3600
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	agent = DQNAgent(
	        action_set=[0, 1, 2],
	        reward_function=functools.partial(Forex_reward_function),
	        feature_extractor=ForexIdentityFeature(),
	        hidden_dims=[50,50],
	        test_model_path='AUDUSD/agents/20190602-043734/dqn|0.pt')

	_eps = []
	_idxs = []
	rewards = []
	# g_mat = np.zeros((num_episodes, 296))

	for episode in range(num_episodes):
	    agent.reset_cumulative_reward()
	    new_env = env.reset_fixed(episode * 3600)
	    observation_history = [(new_env[0], new_env[1], new_env[2], False)]
	    price_record_history =[new_env[2]]
	    index_history = [new_env[0]]
	    action_history = []

	    t = 0
	    done = False
	#     sensitivity = [0] * 296
	    while not done:
	        action = agent.act(observation_history, action_history)
	        timestamp, state, price_record, done = env.step(action)
	        index_history.append(timestamp)
	        price_record_history.append(price_record)
	        action_history.append(action)
	        observation_history.append((timestamp, state, price_record, done))
	        t += 1
	        done = done or (t == max_timesteps)
	        
	#         obs = observation_history[-1][1].data.numpy()
	#         for i in range(len(obs)):
	#             obs[i] += 0.00001
	#             tmp_1, _, tmp_3, tmp_4 = observation_history[-1]
	#             observation_history[-1] = (tmp_1, torch.tensor(obs).to(device), tmp_3, tmp_4)
	#             sens_action = agent.act(observation_history, action_history)
	#             if action != sens_action:
	#                 sensitivity[i] += 1
	#             obs[i] -= 0.00001
	#             observation_history[-1] = (tmp_1, torch.tensor(obs).to(device), tmp_3, tmp_4)
	            
	#     sensitivity = [elem/3600.0 for elem in sensitivity]
	#     g_mat[episode] = np.array(sensitivity)
	        
	    _id , reward = get_reward(index_history, price_record_history, action_history)
	    _eps.append(episode)
	    _idxs.append(_id)
	    rewards.append(reward.item())
	    print("Testing on the {} datapoint drawing from {} th data and return is {}".format(episode,_id,reward))
	    print("{} short positions".format(action_history.count(0)))
	    print("{} neutral positions".format(action_history.count(1)))
	    print("{} long positions".format(action_history.count(2)))

	print("Testing on the {} datapoint and average return is {}".format(num_episodes,np.mean(np.asarray(rewards))))
	output = pd.DataFrame({'num_episode': _eps, 'data_idx': _idxs, 'reward': rewards})
	output.to_csv('AUDUSD/results/20190602-043734/test_rewards.csv', index = False)

	_mean = np.mean(np.asarray(rewards))
	_std = np.std(np.asarray(rewards))
	sharpe_ratio = (_mean - HOURLY_RF_RATE) / _std
	print("Mean {}, std {} and Sharpe ratio {}".format(_mean, _std, sharpe_ratio))
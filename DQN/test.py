"""
test and visualize trained cartpole agents
"""
import numpy as np
import torch
# from gym.envs.classic_control import rendering
# import time
# import skvideo.io
import functools

from environment import ForexEnv
from agents import RandomAgent
from agents import DQNAgent
from agents import Forex_reward_function
from feature import ForexIdentityFeature

def test(agent, environment, max_timesteps):
    """
    return observation and action data for one episode
    """
    # observation_history is a list of tuples (observation, termination signal)
    observation_history = [(environment.reset()[0],environment.reset()[1],environment.reset()[2], False)]
    action_history = []

    t = 0
    done = False
    while not done:
        action = agent.act(observation_history, action_history)
        timestamp, state, price_record, done = environment.step(action)
        action_history.append(action)
        observation_history.append((timestamp, state, price_record, done))
        t += 1
        done = done or (t == max_timesteps)

    return observation_history, action_history


if __name__=='__main__':
    dqn_model_path = './agents/dqn|0.pt'

    np.random.seed(321)
    torch.manual_seed(123)

    env = ForexEnv(mode = 'eval')
    eps = 10
    rewards = []

    agent = DQNAgent(
        action_set=[0, 1, 2],
        reward_function=functools.partial(Forex_reward_function),
        feature_extractor=ForexIdentityFeature(),
        test_model_path=dqn_model_path)

    for e in range(eps):
        observation_history, action_history = test(
            agent=agent,
            environment=env,
            max_timesteps=3600)
        r = torch.sum(agent.get_episode_reward(observation_history, action_history))
        print('reward %.2f' % r)
        rewards.append(r)
        if e == eps -1:
            print(action_history)
            print(agent.get_episode_reward(observation_history, action_history))

    reward = torch.mean(torch.stack(rewards))

    print('agent %s, cumulative reward %.2f' % (str(agent), reward))



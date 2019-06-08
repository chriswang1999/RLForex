import numpy as np
import torch
import matplotlib.pyplot as plt
import functools
import os


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(log_path, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')

def live(agent, environment, test_environment, num_episodes, max_timesteps,
    verbose=False, print_every=10, log_path = ''):
    """
    Logic for operating over episodes. 
    max_timesteps is maximum number of time steps per episode. 
    """


    # observation_data = [] #(self.timestamp, self.state, self.price_record)
    # action_data = []
    rewards = []
    print_rewards = []
    print_actions = []
    if verbose:
        logging('agent: ' + str(agent) + '  number of episodes: '+ str(num_episodes), log_path)
    
    for episode in range(num_episodes):
        agent.reset_cumulative_reward()
        new_env = environment.reset()
        observation_history = [(new_env[0],new_env[1],new_env[2], False)]
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

        agent.update_buffer(observation_history, action_history)
        agent.learn_from_buffer()

        # observation_data.append(observation_history)
        # action_data.append(action_history)
        rewards.append(agent.cummulative_reward)
        print_rewards.append(agent.cummulative_reward)

        print_actions += action_history


        if verbose and (episode % print_every == 0):
            logging('----------------------------------------', log_path)
            logging('ep id  '+ str(episode) + '    100-ep  ' +str(np.sum(np.array(print_rewards))),log_path)
            logging('short  ' + str(print_actions.count(0)), log_path)
            logging('neutral  '+ str(print_actions.count(1)), log_path)
            logging('long  '+ str(print_actions.count(2)), log_path)
            logging('----------------------------------------', log_path)
            print_rewards = []
            print_actions = []

        if episode % (2 * print_every) == 0:
            test(agent, test_environment, 10, max_timesteps, True, 20, log_path)

    return rewards

def test(agent, environment, num_episodes, max_timesteps, verbose=True, print_every=20, log_path = ''):
    # observation_data = [] #(self.timestamp, self.state, self.price_record)
    # action_data = []
    rewards = []
    print_actions = []

    agent.test_mode = True
    logging("start testing on eval set...", log_path)
    for episode in range(20):
        agent.reset_cumulative_reward()
        new_env = environment.reset_fixed(episode * 3600)
        observation_history = [(new_env[0], new_env[1], new_env[2], False)]
        # observation_history = [(environment.reset()[0],environment.reset()[1],environment.reset()[2], False)]
        action_history = []

        t = 0
        done = False
        while not done:
            action = agent.act(observation_history, action_history)
            # action = 0
            timestamp, state, price_record, done = environment.step(action)
            action_history.append(action)
            observation_history.append((timestamp, state, price_record, done))
            t += 1
            done = done or (t == max_timesteps)

        agent.update_buffer(observation_history, action_history)
        # observation_data.append(observation_history)
        # action_data.append(action_history)
        rewards.append(agent.cummulative_reward)
        print_actions += action_history

        if verbose and ((episode+1) % print_every == 0):
            logging('----------------------------------------', log_path)
            logging('ep id  '+ str(episode) + '    100-ep  ' +str(np.sum(np.array(rewards))),log_path)
            logging('short  ' + str(print_actions.count(0)), log_path)
            logging('neutral  '+ str(print_actions.count(1)), log_path)
            logging('long  '+ str(print_actions.count(2)), log_path)
            logging('----------------------------------------', log_path)

            print_actions = []
    logging( "finishing testing on eval set...",log_path)
    agent.test_mode = False
    return rewards

### Example of usage
from environment import ForexEnv
from agents import RandomAgent
from agents import DQNAgent
from agents import Forex_reward_function
from feature import ForexIdentityFeature

if __name__=='__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    env = ForexEnv()

    agent = DQNAgent(
        action_set=[0, 1, 2],
        reward_function=functools.partial(Forex_reward_function),
        feature_extractor=ForexIdentityFeature(),
        hidden_dims=[10, 10],
        learning_rate=0.000025,
        buffer_size=5000,
        batch_size=12,
        num_batches=20,
        starts_learning=200,
        final_epsilon=0.02, 
        discount=0.9,
        target_freq=10,
        verbose=False, 
        print_every=10)
    rewards = live(agent=agent,environment=env,num_episodes=5,max_timesteps=5,verbose=True,print_every=50)
    # observation_data, action_data, rewards = live(
    #                         agent=agent,
    #                         environment=env,
    #                         num_episodes=5,
    #                         max_timesteps=5,
    #                         verbose=True,
    #                         print_every=50)

    agent.save('./dqn.pt')

import numpy as np
import pandas as pd
import torch

from run_deep import Policy
from utils_full import draw_eval_episode


def test(config):
    policy = Policy()
    policy.load_state_dict(torch.load(config.model_path))

    d = {'bid_price': [0], 'ask_price': [0], 'action':[0]}
    df = pd.DataFrame(data=d)
    rewards_over_time = []

    with torch.no_grad():
        accumulative_reward_test = 0
        for j in range(config.num_of_test):
            current_reward = 0
            ask = np.zeros((1, 1))
            bid = np.zeros((1, 1))
            previous_action = torch.tensor([0.0])
            while ask.shape[0] <= config.timespan and bid.shape[0]<=3600:
                target_bid, target_ask, feature_span = draw_eval_episode(config.week_num, config.lag, config.currency,
                                                                         config.min_history, j, config.offset)
                bid, ask, feature_span = target_bid[config.lag:]*1e3, target_ask[config.lag:]*1e3, feature_span
            for t in range(config.timespan):  # Don't infinite loop while learning
                state = feature_span[t]
                save_action = policy(torch.from_numpy(state).float(),0.1*previous_action)

                if t == config.timespan-1:
                    save_action = 0
                action = save_action - previous_action

                price = 0
                if action > 0:
                    price = ask[t]
                elif action < 0:
                    price = bid[t]
                reward = torch.sum(-1 * action * price)
                accumulative_reward_test += reward
                current_reward  += reward

                d = {'bid_price': [bid[t]], 'ask_price': [ask[t]], 'action':[action.item()]}
                temp_df = pd.DataFrame(data=d)
                df = df.append(temp_df)
                previous_action = save_action
            print("episode_reward",current_reward)
        print ("Testing on {} datapoint and return is {}".format(config.num_of_test, accumulative_reward_test))
        rewards_over_time.append(accumulative_reward_test)

    # Save the csv file
    currency_pair = config.model_path[-27:-21]
    saved_path = 'deep/result/' + currency_pair + 'shift' +'.csv'
    print('Saving the csv file ...')
    df.to_csv(saved_path, index = False)

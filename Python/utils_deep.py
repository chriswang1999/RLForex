import os
import numpy as np
import pandas as pd
import torch
from pro_data_drl import CreateFeature
np.random.seed(1)
torch.manual_seed(1)

T = 3617
m = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_file(week_num, lag, cur, mode):
    final = None
    if mode == 'train':
        trainname = './data/train_' + cur + '_lag_' + str(lag) + '_week' + str(week_num) + '.csv'
        if os.path.exists(trainname) == False:
            CreateFeature(cur, lag, week_num)
        final = pd.read_csv(trainname).reset_index(drop = True)
    elif mode == 'eval':
        evalname = './data/eval_' + cur + '_lag_' + str(lag) + '_week' + str(week_num) + '.csv'
        if os.path.exists(evalname) == False:
            CreateFeature(cur, lag, week_num)
        final = pd.read_csv(evalname).reset_index(drop = True)
    return final

def draw_train_episode(week_num, lag, cur, min_history):
    '''
    Input:
        m, number of lag returns z_1,...z_m
        cur, currency pair that we target to trade
        min_history, min length of a valid episode
    '''
    final = get_file(week_num, lag, cur, 'train')
    to_draw = np.sort(final['timestamp'].unique())
    n = np.random.randint(to_draw.shape[0] - min_history)
    _max = to_draw.shape[0]
    _end = min(n+T, _max)
    timeframe = to_draw[n:_end]
    train = final[final.timestamp.isin(timeframe)]
    target_bid = train['bid price'].values
    target_ask = train['ask price'].values
    feature_span = train.iloc[:,-256:].values
    normalized = (feature_span-feature_span.mean())/feature_span.std()
    return torch.tensor(target_bid).to(device), torch.tensor(target_ask).to(device), torch.tensor(normalized).to(device)

def draw_eval_episode(week_num, lag, cur, min_history, factor, offset):
    '''
    Input:
        m, number of lag returns z_1,...z_m
        cur, currency pair that we target to trade
        min_history, min length of a valid episode
    '''
    final = get_file(week_num, lag, cur, 'eval')
    to_draw = np.sort(final['timestamp'].unique())
    n = (factor * 3600) % int(to_draw.shape[0]- min_history) + offset
    _max = to_draw.shape[0]
    _end = min(n+T, _max)
    timeframe = to_draw[n:_end]
    eval = final[final.timestamp.isin(timeframe)]
    target_bid = eval['bid price'].values
    target_ask = eval['ask price'].values
    feature_span = eval.iloc[:,-256:].values
    normalized = (feature_span-feature_span.mean())/feature_span.std()
    return torch.tensor(target_bid).to(device), torch.tensor(target_ask).to(device), torch.tensor(normalized).to(device)


import numpy as np
import pandas as pd
# import datetime
# from pro_data import CreateFeature

# Pad = pd.read_csv('PadData_v2.csv')
# currency = list(np.sort(Pad['currency pair'].unique()))

class Environment(object):
    """
    generic class for environments
    """
    def reset(self):
        """
        returns initial observation
        """
        pass

    def step(self, action):
        """
        returns (observation, termination signal)
        """
        pass


class ForexEnv(Environment):
    """
    Observation: 
        self.timestamp
        self.state = 4 + 144 + 3 = 151
            time sin(2 pi t/T) 4
            log returns: bid, ask price of target currency 2*16
            log returns: mid price of non-target currency 7*16
            position 3
        self.price_record
            bid, ask price of target currency

    Actions:
        0  short
        1   neutral
        2   long

    Starting State:
        random start within training set

    Episode Termination:
        none
    """

    def __init__(self, cur = 'AUDUSD', lag = 16, min_history = 1000):
        self.ccy = cur
        self.lag = lag
        self.min_history = min_history
        self.index = None
        self.state = None
        self.price_record = None
        # self.df = CreateFeature(self.ccy, self.lag).reset_index(drop = True)
        filename = self.ccy + '_lag_' + str(self.lag) + '.csv'
        self.df = pd.read_csv(filename).reset_index(drop = True)
        self.totalframe = self.df.index.values.tolist()
        self.train = self.totalframe[:int(0.6*len(self.totalframe))-self.min_history]
        self.eval = self.totalframe[int(0.6*len(self.totalframe)):int(0.8*len(self.totalframe))-self.min_history]
        self.test = self.totalframe[int(0.8*len(self.totalframe)):]

    def get_features(self,_idx):
        colindex = range(9,9 + self.lag * 9 + 4)
        bid = self.df['bid price'].values[_idx]
        ask = self.df['ask price'].values[_idx]
        # feature_span = self.df[colindex].to_numpy()[_idx,:]
        feature_span = self.df.iloc[_idx,9:].values
        return bid, ask, feature_span

    def step(self, action, mode = 'train'):
        assert action in [0, 1, 2], "invalid action"
        self.index += 1
        position = np.zeros(3)
        position[action] = 1

        bid, ask, feature_span = self.get_features(self.index)
        self.state = np.append(feature_span,position, axis = 0)
        self.price_record = (bid,ask)
        return self.index, self.state, self.price_record, False

    def reset(self, mode = 'train'):
        if mode == 'train':
            to_draw = self.train
        elif mode == 'eval':
            to_draw = self.eval
        elif mode == 'test':
            to_draw = self.test
        n = np.random.choice(len(to_draw))
        self.index = to_draw[n]

        position = np.zeros(3)
        action = np.random.choice(3)
        position[action] = 1

        bid, ask, feature_span = self.get_features(self.index)
        self.state = np.append(feature_span,position, axis = 0)
        self.price_record = (bid,ask)
        return self.index, self.state, self.price_record


### test
if __name__=='__main__':
    nsteps = 5
    np.random.seed(0)

    env = ForexEnv()
    time, obs, price = env.reset()
    t = 0
    print(time)
    print(obs.shape)
    print(price)

    done = False
    while not done:
        action = np.random.randint(3)
        time,obs, price, done = env.step(action)
        t += 1
        print(time)
        print(obs.shape)
        print(price)
        done = done or t==nsteps



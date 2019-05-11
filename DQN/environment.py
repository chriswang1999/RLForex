import numpy as np
import pandas as pd
import datetime

Pad = pd.read_csv('PadData_v2.csv')
to_draw = np.sort(Pad['timestamp'].unique())
to_draw_train = to_draw[:int(to_draw.shape[0]*0.6)]
to_draw_eval = to_draw[int(to_draw.shape[0]*0.6):int(to_draw.shape[0]*0.8)]
to_draw_test = to_draw[int(to_draw.shape[0]*0.8):]
currency = list(np.sort(Pad['currency pair'].unique()))

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
        -1  short
        0   neutral
        1   long

    Starting State:
        random start within training set

    Episode Termination:
        none
    """

    def __init__(self, cur = 'AUDUSD', lag = 16, min_history = 1000):
        self.ccy = cur
        self.lag = lag
        self.min_history = min_history
        self.timestamp = None
        self.state = None
        self.price_record = None

    def features(self,price_path):
        features = np.zeros(self.lag)
        for i in range(self.lag):
            features[i] = np.log(price_path[-1]) - np.log(price_path[-(i+2)])
        return features

    def get_features(self, timestamp, currency = currency):
        tmp = Pad[Pad['currency pair'] == self.ccy]
        idx = to_draw_train.tolist().index(timestamp)
        timeframe = to_draw_train[idx - self.lag:idx+1]
        b_target = tmp[tmp.timestamp.isin(timeframe)]['bid price'].values
        a_target = tmp[tmp.timestamp.isin(timeframe)]['ask price'].values
        feature_span = self.features(b_target)
        feature_span = np.append(feature_span, self.features(a_target), axis = 0)
        for cur in currency:
            if cur == self.ccy:
                pass
            else:
                _tmp = Pad[Pad['currency pair'] == cur]
                b_other = _tmp[_tmp.timestamp.isin(timeframe)]['bid price'].values
                a_other = _tmp[_tmp.timestamp.isin(timeframe)]['ask price'].values
                mid_other = np.mean(np.asarray([b_other,a_other]), axis=0)
                feature_span = np.append(feature_span, self.features(mid_other), axis = 0)
        return feature_span

    def step(self, action):
        assert action in [0, 1, 2], "invalid action"
        time = self.timestamp
        idx = to_draw_train.tolist().index(time)+1
        timestamp = to_draw_train[idx]

        tmp = Pad[Pad['currency pair'] == self.ccy]
        position = np.zeros(3)
        position[action] = 1

        dow = datetime.datetime(int(timestamp[6:10]),int(timestamp[0:2]),int(timestamp[3:5])).weekday()
        h = int(timestamp[11:13])
        m = int(timestamp[14:16])
        s = int(timestamp[17:19])
        state_0 = np.asarray([np.sin(np.pi*s/60),np.sin(np.pi*m/60),np.sin(np.pi*h/24),np.sin(np.pi*dow/7)])
        state_1 = self.get_features(timestamp)
        state_2 = position

        self.timestamp = timestamp
        self.state = np.append(np.append(state_0,state_1,axis = 0),state_2, axis = 0)
        self.price_record = (tmp[tmp.timestamp == timestamp]['bid price'].values[0],
                             tmp[tmp.timestamp == timestamp]['ask price'].values[0])
        
        return self.timestamp, self.state, self.price_record, False

    def reset(self):
        n = np.random.random_integers(self.lag+1,to_draw_train.shape[0] - self.min_history)
        timestamp = to_draw_train[n]

        tmp = Pad[Pad['currency pair'] == self.ccy]
        position = np.zeros(3)
        position[np.random.randint(3)] = 1

        dow = datetime.datetime(int(timestamp[6:10]),int(timestamp[0:2]),int(timestamp[3:5])).weekday()
        h = int(timestamp[11:13])
        m = int(timestamp[14:16])
        s = int(timestamp[17:19])
        state_0 = np.asarray([np.sin(np.pi*s/60),np.sin(np.pi*m/60),np.sin(np.pi*h/24),np.sin(np.pi*dow/7)])
        state_1 = self.get_features(timestamp)
        state_2 = position

        self.timestamp = timestamp
        self.state = np.append(np.append(state_0,state_1,axis = 0),state_2, axis = 0)
        self.price_record = (tmp[tmp.timestamp == timestamp]['bid price'].values[0],
                             tmp[tmp.timestamp == timestamp]['ask price'].values[0])
        return self.timestamp, self.state, self.price_record


### test
if __name__=='__main__':
    nsteps = 2
    np.random.seed(0)

    env = ForexEnv()
    time, obs,price = env.reset()
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



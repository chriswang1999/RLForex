import numpy as np
import pandas as pd
import datetime

def CreateFeature(cur, lag):
    Pad = pd.read_csv('PadData_v2.csv')
    currency = list(np.sort(Pad['currency pair'].unique()))
    tmp = Pad[Pad['currency pair'] == cur].sort_values(by=['timestamp'])
    for i in range(1,lag+1):
        colname1 = 'bid_lag_' + str(i)
        colname2 = 'ask_lag_' + str(i)
        tmp[colname1] = np.log(tmp['bid price']) - np.log(tmp['bid price'].shift(i))
        tmp[colname2] = np.log(tmp['ask price']) - np.log(tmp['ask price'].shift(i))
    for ccy in currency:
        if ccy == cur:
            pass
        else:
            _tmp = Pad[Pad['currency pair'] == ccy].sort_values(by=['timestamp'])
            mid =  pd.DataFrame(np.mean(np.asarray([_tmp['bid price'].values,_tmp['ask price'].values]), axis=0))
            for i in range(1,lag+1):
                colname3 = ccy + '_lag_' + str(i)
                tmp[colname3] = np.log(mid) - np.log(mid.shift(i))
    tmp['date'] = tmp['timestamp'].astype(str).str[0:10]
    tmp['dow'] = pd.to_datetime(tmp['date']).dt.dayofweek
    tmp['hh'] = tmp['timestamp'].astype(str).str[11:13]
    tmp['mm'] = tmp['timestamp'].astype(str).str[14:16]
    tmp['ss'] = tmp['timestamp'].astype(str).str[17:19]
    tmp['time_1'] = np.sin(np.pi*tmp['dow'].values/7)
    tmp['time_2'] = np.sin(np.pi*tmp['hh'].astype('int64').values/24)
    tmp['time_3'] = np.sin(np.pi*tmp['mm'].astype('int64').values/60)
    tmp['time_4'] = np.sin(np.pi*tmp['ss'].astype('int64').values/60)
    tmp = tmp.drop(['date', 'dow','hh','mm','ss'], axis=1)
    tmp = tmp.reset_index(drop=True)
    tmp = tmp[lag:]
    filename = './data/' + cur + '_lag_' + str(lag) + '.csv'
    tmp.to_csv(filename ,index=False)

if __name__=='__main__':
    CreateFeature('AUDUSD', 16)

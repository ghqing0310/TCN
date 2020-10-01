import scipy.io as scio
import numpy as np
from functools import reduce

pars = {'batch':[40], 'day_pred':20, 'day_step':5, 'train_p':0.8, 'hidden':10, 'lr':1e-3, 'frq':2500, 'model': 'tcn', 
        'level': 11, 'kernel':6}

def normalize(data):
    mu = np.mean(data)
    std = np.std(data)
    return (data-mu) / std

# 输入：前batch天收益率序列，滚动窗口day_step天
# 输出：预测后day_pred天的累积收益率
def get_data(batch, day_pred=1, day_step=5, train_p=0.9):
    # data
    path = '../华泰金工/数据/alpha.mat'
    data = scio.loadmat(path)

    close_raw = data['indexinfo']['SH000001'][0][0][0][0][0][0] # 5446
    rate_raw = close_raw[1:] / close_raw[:-1] - 1 # 5445
    # trim
    length = int( (len(rate_raw)-day_pred) / batch ) * batch + day_pred
    rate_raw = rate_raw[:length]
    # rate today
    rate = rate_raw[:-day_pred]
    rate = [ rate[i:i+batch] for i in range(0, len(rate)-batch+1, day_step) ] # rolling window
    rate = normalize(rate).reshape((-1,batch,1))
    # rate days after batch
    if day_pred == 1:
        rate1_raw = [ rate_raw[i] for i in range(batch, length, day_step) ]
    else:
        rate1_raw = np.array( [reduce( lambda x,y:x*y, (rate_raw[i:i+day_pred]+1) ) 
                               for i in range(batch, length-day_pred+1, day_step)] ) - 1
    rate1 = normalize(rate1_raw).reshape((-1,1))

    # split
    assert rate.shape[0] == rate1.shape[0]
    train_num = int(rate.shape[0]*train_p)
    rate_train = rate[:train_num,:,:]
    rate_test = rate[train_num:,:,:]
    rate1_train = rate1[:train_num,:]
    rate1_test = rate1[train_num:,:]

    return rate_train, rate_test, rate1_train, rate1_test


import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from data import get_data,pars,normalize
import torch
from torch import nn
from functools import reduce

mu = 1.00010640674316012939
std = 0.014122747695978287

def plot_rate(py, ry, name):
    plt.plot(py, 'r', label='prediction')
    plt.plot(ry, 'b', label='real')
    plt.legend(loc='best')
    plt.title(name)
    plt.savefig(name)
    plt.show()

def plot_accu_rate(py, ry, name):
    accu_py = accumulate(py)
    accu_ry = accumulate(ry)
    plot_rate(accu_py, accu_ry, name)

def accumulate(rate):
    accu_r = [ reduce(lambda x,y:x*y, (rate[:i+1]*std+mu)) for i in range(1, len(rate)) ]
    accu_r.insert(0, rate[0]*std+mu)
    accu_r.insert(0, 1)
    accu_r = np.array(accu_r)*1000
    return accu_r

def print_loss(py, ry, batch):
    criterion = nn.MSELoss()
    loss = criterion(torch.from_numpy(py).float().cuda(), torch.from_numpy(ry).float().cuda())
    print('batch size %s, test loss %s' % (batch,loss.item()))

if __name__ == "__main__":
    for batch in pars['batch']:
        rate_train, rate_test, rate1_train, rate1_test = get_data(batch, pars['day_pred'], pars['day_step'], pars['train_p'])
        ry = rate1_test.reshape(-1)
        py = np.load('results/%s_pred_%s.npy' % (pars['model'], batch))
        # py = normalize(py)
        plot_rate(py, ry, 'results/%s_pred_%s.png' % (pars['model'], batch))
        accu_py = accumulate(py)
        accu_ry = accumulate(ry)
        plot_rate(accu_py, accu_ry, 'results/%s_accu_pred_%s.png' % (pars['model'], batch))

        print_loss(py, ry, batch)

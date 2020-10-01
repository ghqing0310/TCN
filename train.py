import torch
from torch import nn,optim
from torch.autograd import Variable
import numpy as np
from model.LSTM import lstmModel
from model.TCN import TCN
from data import get_data,pars
from test import plot_rate,print_loss,plot_accu_rate
import random
import os

def seed_torch(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(model, rate_train_ts, rate1_train_ts):
    device = torch.device("cuda")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr = pars['lr'])

    frq, sec = pars['frq'], 50
    loss_set = []

    for epoch in range(1, frq + 1) :
        inputs = Variable(rate_train_ts)
        target = Variable(rate1_train_ts)
        # forward
        output = model(inputs)
        assert target.shape[0] == output.shape[0]
        loss = criterion(output, target)
        # update paramters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print training information
        print_loss = loss.item()
        loss_set.append([epoch, print_loss])
        if epoch % sec == 0 :
            print('Epoch[{}/{}], Loss: {:.5f}'.format(epoch, frq, print_loss))
        if print_loss <= 0.015:
            print('Epoch[{}/{}], Loss: {:.5f}'.format(epoch, frq, print_loss))
            break

    return model, loss_set

if __name__ == "__main__":
    seed_torch(1024)
    for batch in pars['batch']:
        # train
        rate_train, rate_test, rate1_train, rate1_test = get_data(batch, pars['day_pred'], pars['day_step'], pars['train_p'])
        if pars['model'] == 'lstm':
            # data
            model = lstmModel(in_dim=1, hidden_dim=pars['hidden'], out_dim=1, batch=batch)
        elif pars['model'] == 'tcn':
            rate_train = rate_train.reshape((-1,1,batch))
            rate_test = rate_test.reshape((-1,1,batch))
            model = TCN(num_inputs=1, num_channels=[pars['hidden']]*pars['level'], num_outputs=1, batch=batch, kernel_size=pars['kernel'])

        # tensor
        rate_train_ts = torch.from_numpy(rate_train).float().cuda()
        rate_test_ts = torch.from_numpy(rate_test).float().cuda()
        rate1_train_ts = torch.from_numpy(rate1_train).float().cuda()
        rate1_test_ts = torch.from_numpy(rate1_test).float()
        model, loss_set = train(model, rate_train_ts, rate1_train_ts)
        
        # test
        model = model.eval()
        px = Variable(rate_test_ts)
        py = model(px).data
        py = np.array(py.cpu()).reshape(-1)
        ry = np.array(rate1_test_ts).reshape(-1)
        assert py.shape[0] == ry.shape[0]

        # save loss and pred
        np.save('results/%s/%s_loss_%s_%s' % (pars['model'], pars['model'], batch, pars['day_pred']), loss_set)
        np.save('results/%s/%s_pred_%s_%s' % (pars['model'], pars['model'], batch, pars['day_pred']), py)

        # plot
        # plot_rate(py, ry, 'results/%s/%s_pred_%s_%s.png' % (pars['model'], pars['model'], batch, pars['day_pred']))
        plot_accu_rate(py, ry, 'results/%s/%s_accu_pred_%s_%s.png' % (pars['model'], pars['model'], batch, pars['day_pred']))
        print_loss(py, ry, batch)
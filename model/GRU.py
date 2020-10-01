import torch
from torch import nn,optim
from torch.autograd import Variable
from  torch.nn import init
import numpy as np
from seqInit import toTs
from seqInit import input_size,train,real

#定义GRU模型
class gruModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num, dropout_p):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=layer_num)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_p)
        self.gru2 = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=layer_num)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=out_dim)

        self.init_weights()

    def forward(self, x):
        out,_ = self.gru(x)
        out = self.tanh(out)
        out = self.dropout(out)

        out,_ = self.gru2(out)
        out = out[:,-1,:] # 取消batch
        out = self.tanh(out)
        out = self.dropout(out)

        out = self.dense(out)
        return out

    #初始化权重
    def init_weights(self):
        self.lstm1.weight.data.normal_(0, 0.01)
        self.lstm2.weight.data.normal_(0, 0.01)


import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, batch, kernel_size, stride, dilation, padding, dropout_p=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.chomp1d = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.layernorm = nn.LayerNorm(normalized_shape=(n_outputs,batch), elementwise_affine=False)
        self.batchnorm = nn.BatchNorm1d(num_features=batch, affine=False)

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1d(out) # 前后长度保持相同
        out = self.layernorm(out)
        out = self.relu(out)
        # out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.chomp1d(out)
        out = self.layernorm(out)
        out = self.relu(out)
        # out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(out+res)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, batch, kernel_size=2, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential()

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layer = TemporalBlock(in_channels, out_channels, batch, kernel_size, stride=1, dilation=dilation_size,
                                  padding=(kernel_size-1) * dilation_size, dropout_p=dropout)
            self.network.add_module('layer%s'%i, layer)
        
    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, num_outputs, batch, kernel_size=2, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, batch, kernel_size, dropout)
        self.dense = nn.Linear(in_features=num_channels[-1], out_features=num_outputs)
        self.init_weights()

    def init_weights(self):
        self.dense.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.tcn(x)
        out = self.dense(out[:, :, -1])
        return out
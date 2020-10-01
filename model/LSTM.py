import torch
from torch import nn
from torch.nn import init

#定义LSTM 模型
class lstmModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, batch, layer_num=2, dropout_p=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=layer_num)
        self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=layer_num)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=out_dim)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_p)
        self.layernorm = nn.LayerNorm(normalized_shape=(batch,hidden_dim), elementwise_affine=False) # 不加affine效果比较好
        self.batchnorm = nn.BatchNorm1d(num_features=batch, affine=False)

        self.init_weights()

    def forward(self, x):
        # layernorm、batchnorm和dropout选其一，layernorm效果最好
        out,_ = self.lstm1(x)
        out = self.layernorm(out)
        out = self.tanh(out)
        # out = self.dropout(out)

        out,_ = self.lstm2(out)
        out = self.layernorm(out)
        out = out[:,-1,:] # 取消batch
        out = self.tanh(out)
        # out = self.dropout(out)

        out = self.dense(out)
        return out

    #初始化权重
    def init_weights(self):
        self.dense.weight.data.normal_(0, 0.01)
        for name, param in self.named_parameters():
            if 'lstm1.weight' or 'lstm2.weight' in name:
                init.normal_(param, mean=0, std=1e-2)
            if 'batchnorm.weight' in name:
                init.constant_(param, 1)
            if 'batchnorm.bias' in name:
                init.constant_(param, 0)
            if 'layernorm.weight' in name:
                init.constant_(param, 1)
            if 'layernorm.bias' in name:
                init.constant_(param, 0)



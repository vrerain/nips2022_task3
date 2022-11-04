import torch
from torch import nn
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, num_questions, hidden_size, num_layers):        #num_questions代表知识点数量
        super(Net, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.rnn = nn.RNN(num_questions * 2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, num_questions)
        
    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))  #Variable：包装一个tensor,并记录用在它身上的operations
        out, _ = self.rnn(x, h0)   #out:保存在最后一层的输出特征   _:保存着最后一个时刻隐状态
        res = torch.sigmoid(self.fc(out))
        return res
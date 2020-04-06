import torch
import torch.nn as nn
import torch.nn.functional as F
from diConstants import DEVICE


class Attention(nn.Module):
    def __init__(self, hidden_size, key_size, query_size, value_size):
        super(Attention, self).__init__()

        self.n = torch.sqrt(torch.tensor([float(hidden_size)])).to(DEVICE)
        # self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        # self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        # self.value_layer = nn.Linear(value_size, hidden_size, bias=False)
        # 
        # nn.init.xavier_normal_(self.key_layer.weight)
        # nn.init.xavier_normal_(self.query_layer.weight)
        # nn.init.xavier_normal_(self.value_layer.weight)

    def forward(self, query, key, value, mask):
        # print("query", query.repeat(1, proj_key.shape[1], 1).shape)
        # print("proj_key", proj_key.shape)
        # print("query + proj_key", torch.cat([query.repeat(1, proj_key.shape[1], 1), proj_key], dim=2).shape)

        # query = self.query_layer(query)
        # scores = self.energy_layer(torch.tanh(torch.cat([query.expand(-1, proj_key.shape[1], -1), proj_key], dim=2)))
        scores = torch.bmm(key, query.view(key.shape[0], -1, 1)) / self.n
        
        scores = scores.squeeze(2).unsqueeze(1)
        scores = F.softmax(scores, dim=-1)
        context = torch.bmm(scores, value)

        return context

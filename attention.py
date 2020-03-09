import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size, key_size, query_size):
        super(Attention, self).__init__()

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, proj_key, value, mask):
        query = self.query_layer(query)

        # scores = self.energy_layer(torch.tanh(proj_key + cur_embed?))
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        # scores.data.masked_fill_(mask == 0, -float('inf'))
        scores = F.softmax(scores, dim=-1)
        context = torch.bmm(scores, value)

        return context

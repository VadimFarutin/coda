import torch
import torch.nn as nn


class KerasFormatConverter(nn.Module):
    def __init__(self):
        super(KerasFormatConverter, self).__init__()

    def forward(self, data):
        if isinstance(data, torch.Tensor):
            return torch.transpose(data, 1, 2)
        else:
            return data.transpose(0, 2, 1)

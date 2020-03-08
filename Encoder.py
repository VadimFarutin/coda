import torch
import torch.nn as nn

from kerasFormatConverter import KerasFormatConverter


class Encoder(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 hidden_size, num_layers, bidirectional, dropout):
        super(Encoder, self).__init__()

        # self.embedding = nn.Embedding(in_channels, hidden_size)

        self.lstm = nn.LSTM(input_size=in_channels,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout)

    def forward(self, x):
        # output = self.embedding(x)
        # print("encoder input ", x.shape)
        output, final_hidden = self.lstm(x)
        # print("encoder output ", output.shape, final_hidden[0].shape, final_hidden[1].shape)

        return output, final_hidden

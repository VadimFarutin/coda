import torch
import torch.nn as nn

from kerasFormatConverter import KerasFormatConverter


class LSTMModel(nn.Module):
    def __init__(self, predict_binary_output, in_channels, out_channels, hidden_size, num_layers, bidirectional, p_dropout):
        super(LSTMModel, self).__init__()

        self.converter = KerasFormatConverter()
        self.lstm = nn.LSTM(input_size=in_channels,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.dropout = nn.Dropout(p_dropout)
        self.fc = nn.Linear(in_features=hidden_size * 2 if bidirectional else hidden_size,
                            out_features=out_channels)
        if predict_binary_output:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.last_activation(x)

        return x

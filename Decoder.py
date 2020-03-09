import torch
import torch.nn as nn

from kerasFormatConverter import KerasFormatConverter
from diConstants import DEVICE


class Decoder(nn.Module):
    def __init__(self, predict_binary_output,
                 in_channels, out_channels,
                 hidden_size, num_layers, bidirectional, p_dropout,
                 teacher_forcing):
        super(Decoder, self).__init__()

        # peak calls in [0..1], signal in [-1..1] after DataNormalizer in mode "01"
        SOS_value = -1.0 if predict_binary_output else -2
        EOS_value = 2.0 if predict_binary_output else 2

        self.SOS = torch.tensor([SOS_value]).repeat(out_channels).to(DEVICE)
        self.EOS = torch.tensor([EOS_value]).repeat(out_channels).to(DEVICE)

        self.teacher_forcing = teacher_forcing

        # self.embedding = nn.Embedding(out_channels, hidden_size)

        self.lstm = nn.LSTM(input_size=out_channels,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=p_dropout)

        self.fc = nn.Linear(in_features=hidden_size * 2 if bidirectional else hidden_size,
                            out_features=out_channels)

        if predict_binary_output:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = nn.ReLU()


    def forward_step(self, input, hidden):
        # embed_output = self.embedding(input)
        y_hat, last_hidden = self.lstm(input, hidden)
        y_hat = self.fc(y_hat)
        y_hat = self.last_activation(y_hat)

        return y_hat, last_hidden

    def forward(self, encoder_output, hidden, target=None, seq_length=0):
        # print("decoder input ", hidden[0].shape, hidden[1].shape, target.shape)

        last_hidden = hidden
        batch_size = hidden[0].shape[1]
        y_hat = torch.tensor([]).to(DEVICE)
        y = self.SOS.repeat(batch_size, 1, 1).float()

        for i in range(seq_length):
            # print("y ", y.shape)
            y_hat_i, last_hidden = self.forward_step(y, last_hidden)
            # print("y_hat_i ", y_hat_i.shape)
            y_hat = torch.cat((y_hat, y_hat_i), dim=1)

            if target is None or not self.teacher_forcing:
                y = y_hat_i  # .detach()
            else:
                y = target.gather(1, torch.tensor([i] * batch_size).view(-1, 1, 1))

        # print("decoder output ", y_hat.shape)
        return y_hat

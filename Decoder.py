import torch
import torch.nn as nn

from kerasFormatConverter import KerasFormatConverter


class Decoder(nn.Module):
    def __init__(self, predict_binary_output,
                 in_channels, out_channels,
                 hidden_size, num_layers, bidirectional, p_dropout,
                 teacher_forcing, seq_length):
        super(Decoder, self).__init__()

        # peak calls in [0..1], signal in [-1..1] after DataNormalizer in mode "01"
        SOS_value = -1 if predict_binary_output else -2
        EOS_value = 2 if predict_binary_output else 2

        self.SOS = torch.tensor([SOS_value]).repeat(out_channels)
        self.EOS = torch.tensor([EOS_value]).repeat(out_channels)

        self.teacher_forcing = teacher_forcing
        self.seq_length = seq_length

        self.embedding = nn.Embedding(out_channels, hidden_size)

        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=p_dropout)

        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=out_channels)

        if predict_binary_output:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = nn.ReLU()


    def forward_step(self, input, hidden):
        embed_output = self.embedding(input)
        y_hat, last_hidden = self.lstm(embed_output, hidden)
        y_hat = self.fc(y_hat)
        y_hat = self.last_activation(y_hat)

        return y_hat, last_hidden

    def forward(self, hidden, target=None):
        last_hidden = hidden
        y_hat = []
        y = self.SOS

        for i in range(self.seq_length):
            y_hat_i, last_hidden = self.forward_step(y, last_hidden)
            y_hat.append(y_hat_i)

            if target is None or not self.teacher_forcing:
                y = y_hat_i.detach()
            else:
                y = target[i]

        y_hat = np.array(y_hat)
        return y_hat

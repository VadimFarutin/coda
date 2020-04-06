import numpy as np

import torch
import torch.nn as nn

from kerasFormatConverter import KerasFormatConverter
from diConstants import DEVICE

from attention import Attention


class AttentionDecoder(nn.Module):
    def __init__(self, predict_binary_output,
                 in_channels, out_channels,
                 hidden_size, num_layers, bidirectional, p_dropout,
                 teacher_forcing):
        super(AttentionDecoder, self).__init__()

        # peak calls in [0..1], signal in [-1..1] after DataNormalizer in mode "01"
        SOS_value = 0.0 if predict_binary_output else 0.0
        EOS_value = 2.0 if predict_binary_output else 2

        self.SOS = torch.tensor([SOS_value]).expand(out_channels).to(DEVICE)
        self.EOS = torch.tensor([EOS_value]).expand(out_channels).to(DEVICE)

        self.teacher_forcing_p = teacher_forcing
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.D = 50

        # self.embedding = nn.Embedding(out_channels, hidden_size)
        self.attention = Attention(hidden_size=hidden_size * 2 if bidirectional else hidden_size,
                                   key_size=hidden_size * 2 if bidirectional else hidden_size,
                                   query_size=hidden_size,
                                   value_size=hidden_size * 2 if bidirectional else hidden_size)

        self.lstm = nn.LSTM(input_size=out_channels + (hidden_size * 2 if bidirectional else hidden_size),
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=False,
                            batch_first=True,
                            dropout=p_dropout)

        self.dropout = nn.Dropout(p_dropout)

        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=out_channels)

        if predict_binary_output:
            self.last_activation = nn.Sigmoid()
        else:
            nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
            self.last_activation = nn.ReLU()

        self.device = DEVICE

    def forward_step(self, input, hidden, key, value, projection_key_mask, encoder_output_mask):
        query = hidden[0][0].unsqueeze(1)
        # query = hidden[0][-1].unsqueeze(1)
        # projection_key_masked = projection_key.gather(1, projection_key_mask)
        # encoder_output_masked = encoder_output.gather(1, encoder_output_mask)
        # projection_key_masked = projection_key
        # encoder_output_masked = encoder_output
        # print(encoder_output_masked.shape)
        context = self.attention(query=query, key=key, value=value, mask=None)
        lstm_input = torch.cat([input.float(), context], dim=2)
        # print("lstm_input", lstm_input.shape)
        # print("hidden", hidden[0].shape)
        y_hat, last_hidden = self.lstm(lstm_input, hidden)
        y_hat = self.dropout(y_hat)
        y_hat = self.fc(y_hat)
        y_hat = self.last_activation(y_hat)

        return y_hat, last_hidden

    def forward(self, encoder_output, hidden, target=None, seq_length=0):
        # last_hidden = ((hidden[0][0] + hidden[0][1]).unsqueeze(0), (hidden[1][0] + hidden[1][1]).unsqueeze(0))
        # last_hidden = ((hidden[0][-1]).unsqueeze(0), (hidden[1][-1]).unsqueeze(0))
        last_hidden = hidden

        batch_size = hidden[0].shape[1]
        y_hat = []
        y = self.SOS.view(1, 1, -1).expand(batch_size, -1, -1).float()
        key = self.attention.key_layer(encoder_output)
        value = self.attention.value_layer(encoder_output)

        teacher_forcing = np.random.uniform(0.0, 1.0) <= self.teacher_forcing_p
        one = torch.tensor([1.0]).to(self.device)
        
        if target is None or not teacher_forcing:
            for i in range(seq_length):
                # projection_key_mask2 = torch.from_numpy(np.arange(i - self.D, i + self.D + 1)).to(DEVICE)\
                #     .repeat(self.hidden_size, 1)\
                #     .transpose(0, 1)\
                #     .repeat(batch_size, 1, 1).clamp(0, seq_length - 1).long()
                #
                # encoder_output_mask2 = torch.from_numpy(np.arange(i - self.D, i + self.D + 1)).to(DEVICE)\
                #     .repeat(self.hidden_size * 2 if self.bidirectional else self.hidden_size, 1)\
                #     .transpose(0, 1)\
                #     .repeat(batch_size, 1, 1).clamp(0, seq_length - 1).long()

                y_hat_i, last_hidden = self.forward_step(y, last_hidden, key, value, None, None)
                # y_hat = torch.cat((y_hat, y_hat_i), dim=1)
                y_hat.append(y_hat_i)
                
                y = one + y_hat_i  # .detach()
        else:
            for i in range(seq_length):
                # projection_key_mask2 = torch.from_numpy(np.arange(i - self.D, i + self.D + 1)).to(DEVICE) \
                #     .repeat(self.hidden_size, 1) \
                #     .transpose(0, 1) \
                #     .repeat(batch_size, 1, 1).clamp(0, seq_length - 1).long()
                #
                # encoder_output_mask2 = torch.from_numpy(np.arange(i - self.D, i + self.D + 1)).to(DEVICE) \
                #     .repeat(self.hidden_size * 2 if self.bidirectional else self.hidden_size, 1) \
                #     .transpose(0, 1) \
                #     .repeat(batch_size, 1, 1).clamp(0, seq_length - 1).long()

                y_hat_i, last_hidden = self.forward_step(y, last_hidden, key, value, None, None)
                # y_hat = torch.cat((y_hat, y_hat_i), dim=1)
                y_hat.append(y_hat_i)

                y = one + target.gather(1, torch.tensor([i] * batch_size).view(-1, 1, 1).to(self.device))

        y_hat = torch.cat(y_hat, dim=1)
        return y_hat

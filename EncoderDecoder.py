import torch
import torch.nn as nn

from kerasFormatConverter import KerasFormatConverter
from Encoder import Encoder
from Decoder import Decoder


class EncoderDecoder(nn.Module):
    def __init__(self, predict_binary_output,
                 in_channels, out_channels,
                 hidden_size, num_layers, bidirectional, p_dropout,
                 teacher_forcing, seq_length):
        super(EncoderDecoder, self).__init__()

        self.converter = KerasFormatConverter()

        self.encoder = Encoder(in_channels, out_channels,
                               hidden_size, num_layers, bidirectional, p_dropout)
        self.decoder = Decoder(predict_binary_output,
                               in_channels, out_channels,
                               hidden_size, num_layers, bidirectional, p_dropout,
                               teacher_forcing, seq_length)

        # self.dropout = nn.Dropout(p_dropout)
        # self.fc = nn.Linear(in_features=hidden_size,
        #                     out_features=out_channels)
        # if predict_binary_output:
        #     self.last_activation = nn.Sigmoid()
        # else:
        #     self.last_activation = nn.ReLU()

    def forward(self, x, y):
        # x = self.converter(x)
        # print(x.shape)

        encoder_output, encoder_final_hidden = self.encoder(x)
        decoder_output = self.decoder(hidden=encoder_final_hidden, target=y)

        # x = self.dropout(x)
        # output = self.fc(decoder_output)
        # output = self.last_activation(output)
        # x = self.converter(x)

        return output

    def encode(self, x):
        encoder_out = self.encoder(x)
        return encoder_out

    def decode(self, x):
        decoder_out = self.decoder(hidden=x)
        return decoder_out

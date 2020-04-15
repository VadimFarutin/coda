import torch
import torch.nn as nn

from CnnEncoder import CnnEncoder
from CnnDecoder import CnnDecoder


class CnnEncoderDecoder(nn.Module):
    def __init__(self, predict_binary_output,
                in_channels, out_channels,
                hidden_size, num_layers,
                kernel_size, stride, dilation,
                residual, p_dropout,
                seq_length, seq2seq):
        super(CnnEncoderDecoder, self).__init__()

        self.encoder = CnnEncoder(in_channels, out_channels,
                                  hidden_size, num_layers,
                                  kernel_size, stride, dilation,
                                  residual,
                                  p_dropout)
        self.decoder = CnnDecoder(predict_binary_output,
                                  in_channels, out_channels,
                                  hidden_size, num_layers,
                                  kernel_size, stride, dilation,
                                  residual, p_dropout,
                                  seq_length, seq2seq)

    def __call__(self, x, y=None):
        return self.forward(x, y)

    def forward(self, x, y=None):
        encoder_output, residual_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output=encoder_output,
                                      residual_output=residual_output)

        return decoder_output

    def encode(self, x):
        pass
    
    def decode(self, encoder_output, x):
        pass

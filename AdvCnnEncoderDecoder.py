import torch
import torch.nn as nn

from CnnEncoder import CnnEncoder
from CnnDecoder import CnnDecoder
from CnnEncoderDecoder import CnnEncoderDecoder


class AdvCnnEncoderDecoder(nn.Module):
    def __init__(self, predict_binary_output,
                in_channels, out_channels,
                hidden_size, num_layers,
                kernel_size, stride, dilation,
                residual, p_dropout,
                disc_hidden_size, disc_num_layers,
                disc_kernel_size, disc_dilation):
        super(AdvCnnEncoderDecoder, self).__init__()

        self.model = CnnEncoderDecoder(predict_binary_output,
                                       in_channels, out_channels,
                                       hidden_size, num_layers,
                                       kernel_size, stride, dilation,
                                       residual, p_dropout)

        self.decoder = Discriminator(hidden_size, disc_hidden_size, disc_num_layers,
                                     disc_kernel_size, disc_dilation,
                                     p_dropout)

    def __call__(self, x, y=None):
        return self.forward(x, y)

    def forward(self, x, y=None):
        encoder_output, residual_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output=encoder_output,
                                      residual_output=residual_output)

        return decoder_output

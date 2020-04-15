import torch
import torch.nn as nn

from CnnEncoder import CnnEncoder
from CnnDecoder import CnnDecoder
from CnnEncoderDecoder import CnnEncoderDecoder


class Discriminator(nn.Module):
    def __init__(self,
                in_channels, hidden_size, num_layers,
                kernel_size, dilation,
                p_dropout):
        super(Discriminator, self).__init__()

        self.layers = [nn.Conv1d(in_channels=in_channels, 
                                 out_channels=in_channels,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=0).to(DEVICE)]

        for _ in range(num_layers):
            self.layers.append(nn.LinearLayer(in_channels=hidden_size, 
                                              out_channels=hidden_size).to(DEVICE))
            self.layers.append(nn.ReLU().to(DEVICE))

        self.layers.append(nn.LinearLayer(in_channels=hidden_size, 
                                          out_channels=1).to(DEVICE))
        self.layers.append(nn.Sigmoid().to(DEVICE))

                                 
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            
        return x


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

        self.discriminator = Discriminator(hidden_size, disc_hidden_size, disc_num_layers,
                                           disc_kernel_size, disc_dilation,
                                           p_dropout)

    def __call__(self, x, return_latent=False):
        return self.forward(x, return_latent)

    def forward(self, x, return_latent):
        return self.model(x, return_latent)

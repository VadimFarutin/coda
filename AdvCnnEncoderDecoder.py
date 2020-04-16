import torch
import torch.nn as nn

from CnnEncoder import CnnEncoder
from CnnDecoder import CnnDecoder
from CnnEncoderDecoder import CnnEncoderDecoder

from diConstants import DEVICE

class Discriminator(nn.Module):
    def __init__(self,
                in_channels, hidden_size, num_layers,
                kernel_size, dilation,
                p_dropout):
        super(Discriminator, self).__init__()

        layers = [nn.Conv1d(in_channels=in_channels, 
                                 out_channels=in_channels,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=0).to(DEVICE)]

        for _ in range(num_layers):
            layers.append(nn.Linear(in_features=hidden_size, 
                                         out_features=hidden_size).to(DEVICE))
            layers.append(nn.ReLU().to(DEVICE))

        layers.append(nn.Linear(in_features=hidden_size, 
                                     out_features=1).to(DEVICE))
        layers.append(nn.Sigmoid().to(DEVICE))

        self.layers = nn.ModuleList(layers)
                                 
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
                seq_length, seq2seq,
                disc_hidden_size, disc_num_layers,
                disc_kernel_size, disc_dilation):
        super(AdvCnnEncoderDecoder, self).__init__()

        self.model = CnnEncoderDecoder(predict_binary_output,
                                       in_channels, out_channels,
                                       hidden_size, num_layers,
                                       kernel_size, stride, dilation,
                                       residual, p_dropout,
                                       seq_length, seq2seq)

        self.discriminator = Discriminator(hidden_size, disc_hidden_size, disc_num_layers,
                                           disc_kernel_size, disc_dilation,
                                           p_dropout)

    def __call__(self, x, return_latent=False):
        return self.forward(x, return_latent)

    def forward(self, x, return_latent):
        return self.model(x, return_latent)

    def parameters(self):
        return self.model.parameters()

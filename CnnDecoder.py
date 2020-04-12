import numpy as np

import torch
import torch.nn as nn

from diConstants import DEVICE

class CnnDecoder(nn.Module):
    def __init__(self, predict_binary_output, 
                 in_channels, out_channels,
                 hidden_size, num_layers,
                 kernel_size, stride, dilation,
                 residual,
                 p_dropout):
        super(CnnDecoder, self).__init__()

        self.residual = residual
        self.num_layers = num_layers
        self.deconv_layers = []
        self.conv_layers = []
        
        for _ in range(num_layers):
            self.deconv_layers.append(nn.ConvTranspose1d(in_channels=hidden_size, 
                                                         out_channels=hidden_size,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=kernel_size // 2,
                                                         dilation=dilation).to(DEVICE))
            self.conv_layers.append(nn.Conv1d(in_channels=hidden_size, 
                                              out_channels=hidden_size,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=kernel_size // 2).to(DEVICE))
                               
        #self.last_conv = nn.Conv1d(in_channels=hidden_size, 
        #                           out_channels=out_channels,
        #                           kernel_size=kernel_size, #1
        #                           stride=stride, #1
        #                           padding=kernel_size // 2).to(DEVICE) #0
        
        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=out_channels)

        if predict_binary_output:
            self.last_activation = nn.Sigmoid()
        else:
            nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
            self.last_activation = nn.ReLU()

    def forward(self, encoder_output, residual_output):
        out = encoder_output
        #print("############################")
        #print(f"Decoder input {out.shape}")
        #print("##############")

        for i in range(self.num_layers):
            #print("##############")
            #print(f"Layer input  {out.shape}")
            out = nn.functional.relu(self.deconv_layers[i](out))
            #print(f"Layer output {out.shape}")
            #print("##############")
            if self.residual:
                out = out + residual_output[self.num_layers - 1 - i]             
                out = nn.functional.relu(self.conv_layers[i](out))
            
        out = torch.transpose(out, 1, 2)
        #print("##############")
        #print(f"Last layer input  {out.shape}")
        out = self.fc(out)
        out = self.last_activation(out)
        #print(f"Last layer output {out.shape}")
        #print("##############")

        return out
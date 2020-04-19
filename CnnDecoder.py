import numpy as np

import torch
import torch.nn as nn

from diConstants import DEVICE

class CnnDecoder(nn.Module):
    def __init__(self, predict_binary_output, 
                 in_channels, out_channels,
                 hidden_size, num_layers,
                 kernel_size, stride, dilation,
                 residual, p_dropout,
                 seq_length, seq2seq):
        super(CnnDecoder, self).__init__()

        self.residual = residual
        self.num_layers = num_layers
        deconv_layers = []
        conv_layers = []
        bn_layers = []
        
        deconv_layers.append(nn.ConvTranspose1d(in_channels=hidden_size // 2, 
                                                out_channels=hidden_size,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=dilation * 2 * (kernel_size - 1) // 2,
                                                dilation=dilation * 2).to(DEVICE))
        conv_layers.append(nn.Conv1d(in_channels=hidden_size, 
                                     out_channels=hidden_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=(kernel_size - 1) // 2).to(DEVICE))
        bn_layers.append(nn.BatchNorm1d(num_features=hidden_size).to(DEVICE))
                                     
        for _ in range(num_layers - 2):
            deconv_layers.append(nn.ConvTranspose1d(in_channels=hidden_size, 
                                                    out_channels=hidden_size,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=dilation * (kernel_size - 1) // 2,
                                                    dilation=dilation).to(DEVICE))
            conv_layers.append(nn.Conv1d(in_channels=hidden_size, 
                                         out_channels=hidden_size,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=(kernel_size - 1) // 2).to(DEVICE))
            bn_layers.append(nn.BatchNorm1d(num_features=hidden_size).to(DEVICE))

        deconv_layers.append(nn.ConvTranspose1d(in_channels=hidden_size, 
                                                out_channels=hidden_size * 2,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=dilation * (kernel_size - 1) // 2,
                                                dilation=dilation).to(DEVICE))
        conv_layers.append(nn.Conv1d(in_channels=hidden_size * 2, 
                                     out_channels=hidden_size * 2,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=(kernel_size - 1) // 2).to(DEVICE))
        bn_layers.append(nn.BatchNorm1d(num_features=hidden_size * 2).to(DEVICE))

        self.deconv_layers = nn.ModuleList(deconv_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
                                              
        # padding = (seq_length - 1) // 2 if seq2seq else 0
        # self.fc = nn.Conv1d(in_channels=hidden_size * 2, 
        #                     out_channels=out_channels,
        #                     kernel_size=seq_length,
        #                     stride=1,
        #                     padding=padding).to(DEVICE)
        # self.last_linear = False
        
        self.last_linear = True
        self.fc = nn.Linear(in_features=hidden_size * 2,
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
                out = self.conv_layers[i](out)
                out = self.bn_layers[i](out)
                out = nn.functional.relu(out)
            
        #print("##############")
        #print(f"Last layer input  {out.shape}")
        if self.last_linear:
            out = torch.transpose(out, 1, 2)
        out = self.fc(out)
        out = self.last_activation(out)
        #print(f"Last layer output {out.shape}")
        #print("##############")
        if not self.last_linear:
            out = torch.transpose(out, 1, 2)

        return out

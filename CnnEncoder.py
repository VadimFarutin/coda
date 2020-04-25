import torch
import torch.nn as nn

from diConstants import DEVICE

class CnnEncoder(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 hidden_size, num_layers,
                 kernel_size, stride, dilation,
                 residual,
                 p_dropout):
        super(CnnEncoder, self).__init__()

        layers = [nn.Conv1d(in_channels=in_channels, 
                            out_channels=hidden_size * 4,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size - 1) // 2
                            ).to(DEVICE)]
        bn_layers = [nn.BatchNorm1d(num_features=hidden_size * 4).to(DEVICE)]

        layers.append(nn.Conv1d(in_channels=hidden_size * 4, 
                                out_channels=hidden_size,
                                kernel_size=kernel_size,
                                stride=stride,
                                #padding=dilation * (kernel_size - 1) // 2,
                                padding=0,
                                dilation=dilation).to(DEVICE))
        bn_layers.append(nn.BatchNorm1d(num_features=hidden_size).to(DEVICE))
                            
        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(in_channels=hidden_size, 
                                    out_channels=hidden_size,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    #padding=dilation * (kernel_size - 1) // 2,
                                    padding=0,
                                    dilation=dilation).to(DEVICE))
            bn_layers.append(nn.BatchNorm1d(num_features=hidden_size).to(DEVICE))

        layers.append(nn.Conv1d(in_channels=hidden_size, 
                                out_channels=hidden_size // 4,
                                kernel_size=kernel_size,
                                stride=stride,
                                #padding=dilation * 2 * (kernel_size - 1) // 2,
                                padding=0,
                                dilation=dilation * 2).to(DEVICE))
        bn_layers.append(nn.BatchNorm1d(num_features=hidden_size // 4).to(DEVICE))

        self.layers = nn.ModuleList(layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        
    def forward(self, x):
        #print("############################")
        #print(f"Encoder input  {x.shape}")
        x = torch.transpose(x, 1, 2)
        #print(f"Transposed {x.shape}")
        #print("##############")

        residual_output = []
    
        for i, layer in enumerate(self.layers):
            #print("##############")
            #print(f"Layer input  {x.shape}")
            x = layer(x)
            #x = self.bn_layers[i](x)
            if i != len(self.layers) - 1:
                x = nn.functional.relu(x)
            else:
                x = nn.functional.tanh(x)                
            #print(f"Layer output {x.shape}")
            #print("##############")
            residual_output.append(x)
        
        return x, residual_output

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
                            out_channels=hidden_size * 2,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size - 1) // 2).to(DEVICE)]

        layers.append(nn.Conv1d(in_channels=hidden_size * 2, 
                                out_channels=hidden_size,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=dilation * (kernel_size - 1) // 2,
                                dilation=dilation).to(DEVICE))
                            
        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(in_channels=hidden_size, 
                                    out_channels=hidden_size,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=dilation * (kernel_size - 1) // 2,
                                    dilation=dilation).to(DEVICE))

        layers.append(nn.Conv1d(in_channels=hidden_size, 
                                out_channels=hidden_size // 4,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=dilation * 2 * (kernel_size - 1) // 2,
                                dilation=dilation * 2).to(DEVICE))

        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        #print("############################")
        #print(f"Encoder input  {x.shape}")
        x = torch.transpose(x, 1, 2)
        #print(f"Transposed {x.shape}")
        #print("##############")

        residual_output = []
    
        for layer in self.layers:
            #print("##############")
            #print(f"Layer input  {x.shape}")
            x = nn.functional.relu(layer(x))
            #print(f"Layer output {x.shape}")
            #print("##############")
            residual_output.append(x)
        
        return x, residual_output

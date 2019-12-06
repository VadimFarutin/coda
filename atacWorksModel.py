import torch
import torch.nn as nn

from kerasFormatConverter import KerasFormatConverter


class ZeroSamePad1d(nn.Module):
    def __init__(self, interval_size, kernel_size, stride, dilation):
        super(ZeroSamePad1d, self).__init__()

        required_total_padding = ZeroSamePad1d._get_total_same_padding(
            interval_size, kernel_size, stride, dilation)
        padding_left = required_total_padding // 2
        padding_right = required_total_padding - padding_left
        self.pad = nn.ConstantPad1d((padding_left, padding_right), 0)

    @staticmethod
    def _get_total_same_padding(interval_size, kernel_size, stride, dilation):
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        required_total_padding = (interval_size - 1) * \
            stride + effective_kernel_size - interval_size
        return required_total_padding

    def forward(self, x):
        return self.pad(x)


class Activation(nn.Module):
    def __init__(self, afunc='relu'):
        super(Activation, self).__init__()
        self.act_layer = nn.modules.Identity()
        if afunc == 'relu':
            self.act_layer = nn.ReLU()

    def forward(self, x):
        return self.act_layer(x)


class ConvAct1d(nn.Module):
    def __init__(self, interval_size, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False, afunc='relu'):
        self.interval_size = interval_size
        super(ConvAct1d, self).__init__()

        self.padding_layer = ZeroSamePad1d(
            interval_size, kernel_size, stride, dilation)
        self.conv_layer = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, bias=bias)
        self.act_layer = Activation(afunc)

    def forward(self, x):
        x = self.padding_layer(x)
        x = self.conv_layer(x)
        x = self.act_layer(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, interval_size, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False, afunc='relu', conv_input=False):
        super(ResBlock, self).__init__()

        self.conv_input = nn.modules.Identity()
        if conv_input:
            self.conv_input = ConvAct1d(interval_size, in_channels,  out_channels, kernel_size=1, afunc=afunc)
        self.conv_act1 = ConvAct1d(
            interval_size, in_channels,  out_channels, kernel_size, stride, dilation, bias, afunc)
        self.conv_act2 = ConvAct1d(
            interval_size, out_channels, out_channels, kernel_size, stride, dilation, bias, afunc)
        self.conv_act3 = ConvAct1d(
            interval_size, out_channels, out_channels, kernel_size, stride, dilation, bias, afunc=None)
        self.activation = nn.ReLU()

    def forward(self, input):
        x = self.conv_act1(input)
        x = self.conv_act2(x)
        x = self.conv_act3(x)
        x = x + self.conv_input(input)
        x = self.activation(x)

        return x


class AtacWorksModel(nn.Module):
    def __init__(self, predict_binary_output, interval_size, in_channels, out_channels, num_filters=15, num_blocks=5,
                 kernel_size=50, dilation=8, afunc='relu', num_blocks_class=2):
        super(AtacWorksModel, self).__init__()

        self.res_blocks = nn.ModuleList()

        self.res_blocks.append(KerasFormatConverter())
        self.res_blocks.append(ResBlock(interval_size, in_channels, num_filters, kernel_size,
                                        dilation=dilation, afunc=afunc, conv_input=True))
        for _ in range(num_blocks - 1):
            self.res_blocks.append(ResBlock(interval_size, num_filters, num_filters, kernel_size,
                                            dilation=dilation, afunc=afunc, conv_input=False))

        if predict_binary_output:
            self.res_blocks.append(ConvAct1d(interval_size, in_channels=num_filters,
                                             out_channels=out_channels, kernel_size=1, dilation=1, afunc=afunc))

        if not predict_binary_output:
            self.res_blocks.append(nn.Conv1d(
                                        in_channels=num_filters,
                                        out_channels=out_channels,
                                        kernel_size=interval_size,
                                        padding=0))
            self.res_blocks.append(nn.ReLU())

        if predict_binary_output:
            self.res_blocks.append(ResBlock(interval_size, in_channels=out_channels, out_channels=num_filters,
                                            kernel_size=kernel_size, dilation=dilation, afunc=afunc, conv_input=True, bias=True))
            for _ in range(num_blocks_class-1):
                self.res_blocks.append(ResBlock(interval_size, num_filters, num_filters,
                                                kernel_size, dilation=dilation, afunc=afunc, conv_input=False, bias=True))
            # self.res_blocks.append(ConvAct1d(interval_size, in_channels=num_filters,
            #                                  out_channels=out_channels, kernel_size=1, dilation=1, afunc=None, bias=True))
            self.res_blocks.append(nn.Conv1d(
                                        in_channels=num_filters,
                                        out_channels=out_channels,
                                        kernel_size=interval_size,
                                        padding=0))
            self.res_blocks.append(nn.Sigmoid())

        self.res_blocks.append(KerasFormatConverter())

    def forward(self, x):
        for res_block in self.res_blocks:
            x = res_block(x)

        return x

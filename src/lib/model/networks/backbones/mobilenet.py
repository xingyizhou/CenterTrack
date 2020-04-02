from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.utils import load_state_dict_from_url

BN_MOMENTUM = 0.1

model_urls = {
     'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, opt,
                 width_mult=1.0,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super().__init__()
        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1], # 1
            [6, 24, 2, 2], # 2
            [6, 32, 3, 2], # 3
            [6, 64, 4, 2], # 4
            [6, 96, 3, 1], # 5
            [6, 160, 3, 2],# 6
            [6, 320, 1, 1],# 7
        ]
        
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        # self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        if opt.pre_img:
            print('adding pre_img layer...')
            self.pre_img_layer = nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(input_channel))
        if opt.pre_hm:
            print('adding pre_hm layer...')
            self.pre_hm_layer = nn.Sequential(
            nn.Conv2d(1, input_channel, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(input_channel))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        self.key_block = [True]
        all_channels = [input_channel]
        self.channels = [input_channel]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                if stride == 2:
                    self.key_block.append(True)
                else:
                    self.key_block.append(False)
                all_channels.append(output_channel)
        # building last several layers
        # features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # self.key_block.append(False)
        # all_channels.append(self.last_channel)
        for i in range(len(self.key_block) - 1):
          if self.key_block[i + 1]:
            self.key_block[i] = True
            self.key_block[i + 1] = False
            self.channels.append(all_channels[i])
        self.key_block[-1] = True
        self.channels.append(all_channels[-1])
        print('channels', self.channels)
        # make it nn.Sequential
        self.features = nn.ModuleList(features)
        print('len(self.features)', len(self.features))
        # self.channels = [, ]

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'])
        self.load_state_dict(state_dict, strict=False)

    def forward(self, inputs, pre_img=None, pre_hm=None):
        x = self.features[0](inputs)
        if pre_img is not None:
            x = x + self.pre_img_layer(pre_img)
        if pre_hm is not None:
            x = x + self.pre_hm_layer(pre_hm)
        y = [x]
        for i in range(1, len(self.features)):
            x = self.features[i](x)
            # print('i, shape, is_key', i, x.shape, self.key_block[i])
            if self.key_block[i]:
                y.append(x)
        return y


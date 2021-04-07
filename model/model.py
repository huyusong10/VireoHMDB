# Author: hys

import torch
from torch import nn
from torch.nn import functional as F

from functools import partial

from .utils import get_conv
from .attention3d import Attention3d


class Bellii3D(nn.Module):

    def __init__(self, inc, ouc, kernel_size, stride, expand_ratio=4, bn_mom = 0.99, bn_eps = 0.001, se=False):
        super().__init__()
        self.inc = inc
        self.ouc = ouc
        self.stride = stride
        self.expand_ratio = expand_ratio
        self._bn_mom = 1 - bn_mom
        self._bn_eps = bn_eps
        self._se = se
        _, self.conv2d, self.conv3d = get_conv()

        expc = inc * self.expand_ratio
        if self.expand_ratio != 1:
            self._expand_conv = self.conv3d(in_channels=inc, out_channels=expc, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm3d(num_features=expc, momentum=self._bn_mom, eps=self._bn_eps)

        # depthwise conv
        self._depthwise_conv = self.conv3d(
            in_channels=expc, out_channels=expc, groups=expc,
            kernel_size=kernel_size, stride=stride, bias=False
        )
        self._bn1 = nn.BatchNorm3d(num_features=expc, momentum=self._bn_mom, eps=self._bn_eps)

        if self._se:
            self._attention = Attention3d(expc)

        # pointwise conv
        self._pointwise_conv = self.conv3d(
            in_channels=expc, out_channels=ouc, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm3d(num_features=ouc, momentum=self._bn_mom, eps=self._bn_eps)

        self._activate = nn.ReLU()

    def forward(self, inputs):
        x = inputs

        # expand dim
        if self.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self._activate(x)
        
        # depthwise conv
        # print('depthwise1',x.shape)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._activate(x)
        # print('depthwise2',x.shape)

        # se
        if self._se:
            self._attention(x)

        # pointwise conv
        x = self._pointwise_conv(x)
        x = self._bn2(x)

        # skip connection
        if self.inc == self.ouc and self.stride == 1:
            x = x + inputs
        return x


class VireoNet(nn.Module):
    
    def __init__(self, num_classes, bn_mom = 0.99, bn_eps=0.001):
        super().__init__()
        _, self.conv2d, self.conv3d = get_conv()

        repeats = [1, 2, 3, 4, 4, 2]
        strides = [
            (1, 2, 2),
            (2, 2, 2),
            (1, 2, 2),
            (2, 2, 2),
            (1, 1, 1),
            (2, 2, 2),
        ]
        kernel_size = [5, 3, 3, 5, 3, 3]
        channels = [3, 32, 56, 80, 104, 128, 160]
        se = [False, True, True, True, True, False]

        self._blocks = []
        for idx in range(6):
            block = Bellii3D(channels[idx], channels[idx+1], kernel_size=kernel_size[idx], stride=strides[idx])
            self._blocks.append(block)
            for _ in range(repeats[idx] - 1):
                block = Bellii3D(channels[idx+1], channels[idx+1], kernel_size=kernel_size[idx], stride=1, se=se[idx])
                self._blocks.append(block)
        self._blocks = nn.Sequential(*self._blocks)

        head_chann = 640
        self._conv_head = self.conv3d(channels[-1], head_chann, kernel_size=1, bias=False)
        self._bn_head = nn.BatchNorm3d(num_features=head_chann, momentum=bn_mom, eps=bn_eps)

        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(head_chann, num_classes)
        self._activate = nn.ReLU()

    def forward(self, input):
        x = self._blocks(input)
        x = self._activate(self._bn_head(self._conv_head(x)))
        x = self._avg_pooling(x).flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)

        return x


if __name__ == '__main__':
    import sys
    print(sys.path)
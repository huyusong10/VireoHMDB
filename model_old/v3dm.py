import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from functools import partial

from .utils import get_conv, get_act
from .attention3d import Attention3d, Attention3d_up, SE

class V3DTransform(nn.Module):

    def __init__(self, inc, ouc, kernel_size, stride, expand_ratio, bn_mom=0.99, bn_eps=0.001, se=False):
        super().__init__()
        self.inc = inc
        self.ouc = ouc
        self.stride = stride
        self.expand_ratio = expand_ratio
        self._bn_mom = 1 - bn_mom
        self._bn_eps = bn_eps
        self._se = se
        _, _, self.conv3d = get_conv()

        # depthwise
        self._depthwise_conv = self.conv3d(
            in_channels=inc, out_channels=inc, groups=inc,
            kernel_size=kernel_size, stride=stride, bias=False
        )
        self._bn0 = nn.BatchNorm3d(num_features=inc, momentum=self._bn_mom, eps=self._bn_eps)

        # expand
        expc = round(ouc * expand_ratio)
        if expand_ratio != 1:
            self._expand_conv = self.conv3d(in_channels=inc, out_channels=expc, kernel_size=1, bias=False)
            self._bn1 = nn.BatchNorm3d(num_features=expc, momentum=self._bn_mom, eps=self._bn_eps)

        if self._se:
            self._attention = Attention3d(expc)

        # projec conv
        self._project_conv = self.conv3d(
            in_channels=expc, out_channels=ouc, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm3d(num_features=ouc, momentum=self._bn_mom, eps=self._bn_eps)
        self._activate = get_act('ReLU')

    def forward(self, inputs):
        x = inputs

        x = self._depthwise_conv(x)
        x = self._bn0(x)

        if self.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn1(x)
            x = self._activate(x)

        if self._se:
            self._attention(x)

        x = self._project_conv(x)
        x = self._bn2(x)

        if self.inc == self.ouc and self.stride == 1:
            x = x + inputs

        return x


class V3Dm(nn.Module):

    def __init__(self, num_classes, expand_ratio=4, bn_mom = 0.9, bn_eps=0.001):
        super().__init__()
        _, _, self.conv3d = get_conv()
        self._bn_mom = 1 - bn_mom
        self._bn_eps = bn_eps

        repeats = [3, 5, 11, 7]
        strides = [
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
        ]
        channels = [24, 48, 96, 192]
        se = [True] * 6

        input_chann = 3
        stem_chann = 24
        self._stem_conv_xy = self.conv3d(input_chann, stem_chann, stride=(1, 2, 2), kernel_size=(1, 3, 3), bias=False)
        self._stem_conv_t = self.conv3d(stem_chann, stem_chann, stride=(1, 1, 1), kernel_size=(5, 1, 1), bias=False)
        self._stem_bn = nn.BatchNorm3d(num_features=24, momentum=self._bn_mom, eps=self._bn_eps)

        self._res_blocks = []
        channels = [stem_chann] + channels
        for idx in range(len(repeats)):
            block = V3DTransform(channels[idx], channels[idx+1], kernel_size=3, stride=strides[idx], expand_ratio=expand_ratio)
            self._res_blocks.append(block)
            for _ in range(repeats[idx] - 1):
                block = V3DTransform(channels[idx+1], channels[idx+1], kernel_size=3, stride=1, se=se[idx], expand_ratio=expand_ratio)
                self._res_blocks.append(block)
        self._res_blocks = nn.Sequential(*self._res_blocks)

        expand_chann = round(channels[-1] * expand_ratio)
        self._expand_conv = self.conv3d(channels[-1], expand_chann, kernel_size=1, bias=False)
        self._epxand_bn = nn.BatchNorm3d(num_features=expand_chann, momentum=self._bn_mom, eps=self._bn_eps)

        head_chann = 2048
        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        self._dropout = nn.Dropout(0.2)
        self._fc1 = nn.Linear(expand_chann, head_chann)
        self._fc2 = nn.Linear(head_chann, num_classes)

        self._activate = get_act('ReLU')

    def forward(self, inputs):
        x = self._stem_conv_xy(inputs)
        x = self._stem_conv_t(x)
        x = self._stem_bn(x)
        x = self._activate(x)

        x = self._res_blocks(x)
        x = self._expand_conv(x)
        x = self._epxand_bn(x)
        x = self._activate(x)

        x = self._avg_pooling(x).flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc1(x)
        x = self._fc2(x)

        return x
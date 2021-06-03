# Author: hys

import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from functools import partial

from .utils import get_conv, get_act
from .attention3d import Attention3d, Attention3d_up, Attention3d_dev, SE


class Transform3D(nn.Module):

    def __init__(self, inc, ouc, kernel_size, stride, expand=0, bn_mom=0.9, bn_eps=0.001, se=False):
        super().__init__()
        self.inc = inc
        self.ouc = ouc
        self.stride = stride
        self._bn_mom = 1 - bn_mom
        self._bn_eps = bn_eps
        self._se = se
        _, self.conv2d, self.conv3d = get_conv()

        expc = inc + expand
        self.expc = expc

        # depthwise conv
        self._depthwise_conv = self.conv3d(
            in_channels=inc, out_channels=inc, groups=inc,
            kernel_size=kernel_size, stride=stride, bias=False
        )
        self._bn1 = nn.BatchNorm3d(num_features=inc, momentum=self._bn_mom, eps=self._bn_eps)

        if self._se:
            self._attention = Attention3d(expc)

        # project conv
        self._project_conv = self.conv3d(
            in_channels=expc, out_channels=ouc, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm3d(num_features=ouc, momentum=self._bn_mom, eps=self._bn_eps)

        self._activate = get_act('ReLU')

    def forward(self, inputs):

        x = inputs[:, :self.inc]
        shortcut = inputs[:, self.inc:] if self.stride == 1 else None

        # depthwise conv
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._activate(x)

        if shortcut is not None:
            x = torch.cat((x, shortcut), dim=1)
        # se
        if self._se:
            self._attention(x)

        # project conv
        x = self._project_conv(x)
        x = self._bn2(x)

        if self.stride == 1:
            x = torch.cat((x, inputs), dim=1)
        return x

class Fused_Transform3D(nn.Module):

    def __init__(self, inc, ouc, kernel_size, stride, expand_ratio=1, bn_mom=0.9, bn_eps=0.001, se=False):
        super().__init__()
        self.inc = inc
        self.ouc = ouc
        self.stride = stride
        self.expand_ratio = expand_ratio
        self._bn_mom = 1 - bn_mom
        self._bn_eps = bn_eps
        self._se = se
        _, _, self.conv3d = get_conv()

        # expand phase
        expc = round(ouc * self.expand_ratio)

        # fused conv
        self._fused_conv = self.conv3d(
            in_channels=inc, out_channels=expc, groups=1,
            kernel_size=kernel_size, stride=stride, bias=False
        )
        self._bn1 = nn.BatchNorm3d(num_features=expc, momentum=self._bn_mom, eps=self._bn_eps)

        if self._se:
            self._attention = Attention3d(expc)

        # project conv
        self._project_conv = self.conv3d(
            in_channels=expc, out_channels=ouc, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm3d(num_features=ouc, momentum=self._bn_mom, eps=self._bn_eps)

        self._activate = get_act('ReLU')

    def forward(self, inputs):
        x = inputs

        # fused conv
        x = self._fused_conv(x)
        x = self._bn1(x)
        x = self._activate(x)

        # se
        if self._se:
            self._attention(x)

        # project conv
        x = self._project_conv(x)
        x = self._bn2(x)

        # skip connection
        if self.inc == self.ouc and self.stride == 1:
            x = x + inputs
        return x


class DenseVireoV3(nn.Module):
    
    def __init__(self, num_classes, bn_mom = 0.9, bn_eps=0.001):
        super().__init__()
        _, self.conv2d, self.conv3d = get_conv()
        self._bn_mom = 1 - bn_mom
        self._bn_eps = bn_eps
        self.Transform3D = Transform3D
        self.Fused_Transform3D = Fused_Transform3D

        repeats = [3, 4, 4, 4, 4, 4, 4, 4]
        kernel_size = [3, 3, 3, 3, 3, 3, 3, 3]
        strides = [
            (1, 2, 2),  # 56
            (1, 2, 2),  # 28
            (1, 1, 1),
            (1, 2, 2),  # 14
            (1, 1, 1),
            (2, 2, 2),  # 7
            (1, 1, 1),
            (1, 1, 1), 
        ]
        se = [True, True, True, True, True, True, True, True]
        input_chann, stem_chann = 3, 24
        channels = [24, 48, 72, 96, 120, 144, 168, 196]

        self._stem = self.Fused_Transform3D(input_chann, stem_chann, kernel_size=(3, 3, 3), stride=(1, 2, 2), expand_ratio=2)
        self._stem_bn = nn.BatchNorm3d(num_features=stem_chann, momentum=self._bn_mom, eps=self._bn_eps)

        self._blocks = []
        channels = [stem_chann] + channels
        self.last_chann = channels[-1]
        for idx in range(len(kernel_size)):
            transform = self.Transform3D
            block = transform(channels[idx], channels[idx+1], kernel_size=kernel_size[idx], stride=strides[idx])
            self._blocks.append(block)
            for i in range(repeats[idx]-1):
                block = transform(channels[idx+1], channels[idx+1], expand=channels[idx+1]*i, kernel_size=kernel_size[idx], stride=1, se=se[idx])
                self._blocks.append(block)
        self._blocks = nn.Sequential(*self._blocks)

        head_chann = 640
        self._head_conv = self.conv3d(channels[-1], head_chann, kernel_size=1, bias=False)
        self._head_bn = nn.BatchNorm3d(num_features=head_chann, momentum=self._bn_mom, eps=self._bn_eps)

        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(head_chann, num_classes)
        self._activate = get_act('ReLU')


    def forward(self, inputs):
        x = self._stem(inputs)
        x = self._stem_bn(x)
        x = self._activate(x)

        x = self._blocks(x)
        x = x[:, :self.last_chann]

        x = self._activate(self._head_bn(self._head_conv(x)))
        x = self._avg_pooling(x).flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)

        return x


if __name__ == '__main__':
    import sys
    print(sys.path)
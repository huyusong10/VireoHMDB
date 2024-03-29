# Author: hys

import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from functools import partial

from .utils import get_conv, get_act
from .attention3d import Attention3d, Attention3d_up, Attention3d_dev, SE


class Transform3D(nn.Module):

    def __init__(self, inc, ouc, kernel_size, stride, expand_ratio, bn_mom=0.9, bn_eps=0.001, se=False):
        super().__init__()
        self.inc = inc
        self.ouc = ouc
        self.stride = stride
        self.expand_ratio = expand_ratio
        self._bn_mom = 1 - bn_mom
        self._bn_eps = bn_eps
        self._se = se
        _, self.conv2d, self.conv3d = get_conv()

        # expand phase
        expc = round(ouc * self.expand_ratio)
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

        # project conv
        self._project_conv = self.conv3d(
            in_channels=expc, out_channels=ouc, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm3d(num_features=ouc, momentum=self._bn_mom, eps=self._bn_eps)

        self._activate = get_act('ReLU') if self.expand_ratio != 1 else get_act('swish')

    def forward(self, inputs):
        x = inputs

        # expand dim
        if self.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self._activate(x)
        
        # depthwise conv
        x = self._depthwise_conv(x)
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

class Fused_Transform3D(nn.Module):

    def __init__(self, inc, ouc, kernel_size, stride, expand_ratio, bn_mom=0.9, bn_eps=0.001, se=False):
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


class VireoPro(nn.Module):
    
    def __init__(self, num_classes, bn_mom = 0.9, bn_eps=0.001, expand_ratio=4):
        super().__init__()
        _, self.conv2d, self.conv3d = get_conv()
        self._bn_mom = 1 - bn_mom
        self._bn_eps = bn_eps
        self.Transform3D = Transform3D
        self.Fused_Transform3D = Fused_Transform3D

        repeats = [2, 3, 4, 4, 5]
        kernel_size = [3, 3, 5, 3, 3]
        fused = [False, False, False, False, False, False]
        strides = [
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
            (1, 1, 1),
            (2, 2, 2),
        ]
        se = [True, True, True, True, True, True]
        input_chann, stem_chann = 3, 16

        reduction = 1 / 2
        chann_expand = 2.0
        natural_reduce, downsample_expand = chann_expand ** (1 / 3), chann_expand ** (2 / 3)

        reduce_pow = (reduction) ** (1 / (len(repeats) - 1))
        natural_reduce_pow = (1 / natural_reduce) ** (1 / (len(repeats) - 1))

        stage_channels = [stem_chann]
        stage_expand_ratios = []
        instage_expand_ratios = []

        for i, block_repeats in enumerate(repeats):
            ratio = reduce_pow ** i

            stage_ratio = expand_ratio * ratio
            stage_expand_ratios.append(stage_ratio)

            natural_reduce_ratio = natural_reduce * natural_reduce_pow ** i
            if sum(strides[i]) > 3:
                max_channels = round(stage_channels[-1] * (downsample_expand * natural_reduce_ratio))
            else:
                max_channels = round(stage_channels[-1] * natural_reduce_ratio)
            stage_channels.append(max_channels)

            max_reduction = reduction * (1 / reduce_pow) ** i
            instage_channel_reduce_pow = max_reduction ** (1 / (max(repeats[i] - 1, 1)))
            instage_expand_ratio = [stage_ratio * instage_channel_reduce_pow ** x for x in range(repeats[i])]
            instage_expand_ratios.append(instage_expand_ratio)

        print(stage_channels)
        print(stage_expand_ratios)
        print(instage_expand_ratios)

        self._stem = self.Fused_Transform3D(input_chann, stage_channels[0], kernel_size=(5, 5, 5), stride=(1, 2, 2), expand_ratio=2)
        self._stem_bn = nn.BatchNorm3d(num_features=stage_channels[0], momentum=self._bn_mom, eps=self._bn_eps)

        self._blocks = []
        for idx in range(len(kernel_size)):
            transform = self.Fused_Transform3D if fused[idx] else self.Transform3D
            block = transform(stage_channels[idx], stage_channels[idx+1], kernel_size=kernel_size[idx], stride=strides[idx], expand_ratio=instage_expand_ratios[idx][0])
            self._blocks.append(block)
            for i in range(repeats[idx]-1):
                block = transform(stage_channels[idx+1], stage_channels[idx+1], kernel_size=kernel_size[idx], stride=1, se=se[idx], expand_ratio=instage_expand_ratios[idx][i+1])
                self._blocks.append(block)
        self._blocks = nn.Sequential(*self._blocks)

        head_chann = 640
        self._head_conv = self.conv3d(stage_channels[-1], head_chann, kernel_size=1, bias=False)
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

        x = self._activate(self._head_bn(self._head_conv(x)))
        x = self._avg_pooling(x).flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)

        return x


if __name__ == '__main__':
    import sys
    print(sys.path)
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from functools import partial

from .utils import get_conv, get_act, MaxPool3dStaticSamePadding
from .attention3d import Attention3d, Attention3d_up, SE


class PartialDWC(nn.Module):

    def __init__(self, c, stride, kernel_size, ratio=1):
        super().__init__()
        _, _, self.conv3d = get_conv()

        c_depth = round(c * ratio)
        self.c_depth = c_depth
        c_identity = c - c_depth

        self._depthwise_conv = self.conv3d(
            in_channels=c_depth, out_channels=c_depth, groups=c_depth,
            kernel_size=kernel_size, stride=stride, bias=False
        )

        if stride != 1 and c_identity != 0:
            self._identity = nn.Sequential(
                # self.conv3d(
                # in_channels=c_identity, out_channels=c_identity, groups=c_identity,
                # kernel_size=kernel_size, stride=stride, bias=False
                # )
                MaxPool3dStaticSamePadding(kernel_size=stride, stride=stride)
            )
        else:
            self._identity = nn.Identity()


    def forward(self, inputs):

        x_depth = self._depthwise_conv(inputs[:, :self.c_depth])
        x_identity = self._identity(inputs[:, self.c_depth:])

        b, _, d, h, w = x_depth.shape
        x_identity = x_identity.view((b, -1, d, h, w))

        return torch.cat((x_depth, x_identity), dim=1)


class L3DTransform(nn.Module):

    def __init__(self, inc, ouc, kernel_size, stride, expand_ratio, se=False):
        super().__init__()
        self.inc = inc
        self.ouc = ouc
        self.stride = stride
        self.expand_ratio = expand_ratio
        self._se = se
        _, _, self.conv3d = get_conv()

        # expand phase
        expc = round(ouc * expand_ratio)
        if expand_ratio != 1:
            self._expand_conv = self.conv3d(in_channels=inc, out_channels=expc, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm3d(num_features=expc)

        # depthwise conv
        ratio = 2 / expand_ratio
        self._partial_conv = PartialDWC(c=expc, stride=stride, kernel_size=kernel_size, ratio=ratio)
        self._bn1 = nn.BatchNorm3d(num_features=expc)

        if self._se:
            self._attention = Attention3d(expc)

        # project conv
        self._project_conv = self.conv3d(
            in_channels=expc, out_channels=ouc, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm3d(num_features=ouc)
        self._activate = get_act('ReLU')
        
        if inc == ouc and stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                self.conv3d(in_channels=inc, out_channels=inc, groups=inc, kernel_size=kernel_size, stride=stride, bias=False),
                self.conv3d(in_channels=inc, out_channels=ouc, kernel_size=1, bias=False),
                nn.BatchNorm3d(num_features=ouc)
                # MaxPool3dStaticSamePadding(kernel_size=stride, stride=stride)
            )

    def forward(self, inputs):
        x = inputs

        if self.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self._activate(x)

        x = self._partial_conv(x)
        x = self._bn1(x)
        x = self._activate(x)

        if self._se:
            x = self._attention(x)

        x = self._project_conv(x)
        x = self._bn2(x)

        x = x + self.shortcut(inputs)

        return x

class Stem(nn.Module):

    def __init__(self, inc, ouc, kernel_size, stride, expand_ratio):
        super().__init__()
        self.inc = inc
        self.ouc = ouc
        self.stride = stride
        self.expand_ratio = expand_ratio
        _, _, self.conv3d = get_conv()

        # expand phase
        expc = round(ouc * self.expand_ratio)

        # fused conv
        self._fused_conv = self.conv3d(
            in_channels=inc, out_channels=expc, groups=1,
            kernel_size=kernel_size, stride=stride, bias=False
        )
        self._bn1 = nn.BatchNorm3d(num_features=expc)

        # project conv
        self._project_conv = self.conv3d(
            in_channels=expc, out_channels=ouc, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm3d(num_features=ouc)

        self._activate = get_act('ReLU')

    def forward(self, inputs):
        x = inputs

        # fused conv
        x = self._fused_conv(x)
        x = self._bn1(x)
        x = self._activate(x)

        # project conv
        x = self._project_conv(x)
        x = self._bn2(x)

        return x

class Light3DV2(nn.Module):

    def __init__(self, num_classes, expand_ratio=4):
        super().__init__()
        _, _, self.conv3d = get_conv()

        repeats = [2, 3, 4, 8, 12]
        strides = [
            (1, 2, 2),
            (1, 2, 2),
            (1, 2, 2),
            (1, 1, 1),
            (2, 2, 2),
        ]
        channels = [24, 48, 96, 120, 144]
        se = [True] * 5

        input_chann = 3
        stem_chann = 24
        # self._stem = Stem(input_chann, stem_chann, kernel_size=(3, 3, 3), stride=(1, 2, 2), expand_ratio=1)
        self._stem_conv_xy = self.conv3d(input_chann, stem_chann, stride=(1, 2, 2), kernel_size=(1, 3, 3), bias=False)
        self._stem_conv_t = self.conv3d(stem_chann, stem_chann, stride=(1, 1, 1), kernel_size=(5, 1, 1), bias=False)
        self._stem_bn = nn.BatchNorm3d(num_features=stem_chann)

        reduction = 1 / 2
        reduce_pow = (reduction) ** (1 / (len(repeats) - 1))
        stage_expand_ratios = []
        instage_expand_ratios = []

        for i, block_repeats in enumerate(repeats):
            ratio = reduce_pow ** i

            stage_ratio = expand_ratio * ratio
            stage_expand_ratios.append(stage_ratio)

            max_reduction = reduction * (1 / reduce_pow) ** i
            instage_channel_reduce_pow = max_reduction ** (1 / (max(repeats[i] - 1, 1)))
            instage_expand_ratio = [stage_ratio * instage_channel_reduce_pow ** x for x in range(repeats[i])]
            instage_expand_ratios.append(instage_expand_ratio)

        print(stage_expand_ratios)
        print(instage_expand_ratios)

        self._blocks = []
        channels = [stem_chann] + channels
        for idx in range(len(repeats)):
            block = L3DTransform(channels[idx], channels[idx+1], kernel_size=3, stride=strides[idx], expand_ratio=instage_expand_ratios[idx][0])
            self._blocks.append(block)
            for i in range(repeats[idx] - 1):
                block = L3DTransform(channels[idx+1], channels[idx+1], kernel_size=3, stride=1, se=se[idx], expand_ratio=instage_expand_ratios[idx][i+1])
                self._blocks.append(block)
        self._blocks = nn.Sequential(*self._blocks)

        head_chann = 1280
        self._head_conv = self.conv3d(channels[-1], head_chann, kernel_size=1, bias=False)
        self._head_bn = nn.BatchNorm3d(num_features=head_chann)
        
        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        self._dropout = nn.Dropout(0.2)
        self._fc = nn.Linear(head_chann, num_classes)

        self._activate = get_act('ReLU')

    def forward(self, inputs):
        x = self._stem_conv_xy(inputs)
        x = self._stem_conv_t(x)
        x = self._stem_bn(x)

        x = self._blocks(x)

        x = self._head_conv(x)
        x = self._head_bn(x)
        x = self._activate(x)

        x = self._avg_pooling(x).flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)

        return x
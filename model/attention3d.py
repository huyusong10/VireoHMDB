# Author: hys

import math
import torch
from torch import nn
from torch.nn import functional as F
from .utils import get_conv

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class Attention3d_mode1(nn.Module):

    def __init__(self, channels, kernel_size=3, bn_mom=0.99, bn_eps=0.001, reduction=32):
        super().__init__()

        self._channels = channels
        self._bn_mom = bn_mom
        self._bn_eps = bn_eps
        _, self.conv2d, _ = get_conv()

        # depthwise conv
        depth_chann = channels * 3
        self._depthwise_conv = self.conv2d(
            in_channels=depth_chann, out_channels=depth_chann, groups=depth_chann,
            kernel_size=kernel_size, bias=False
        )
        self._bn1 = nn.BatchNorm2d(num_features=depth_chann, momentum=self._bn_mom, eps=self._bn_eps)

        # se attention
        se_chann = max(1, depth_chann//reduction)
        self._se_reduce = self.conv2d(in_channels=depth_chann, out_channels=se_chann, kernel_size=1, bias=True)

        # project conv
        self._project_conv = self.conv2d(
            in_channels=se_chann, out_channels=depth_chann,
            kernel_size=1, bias=True
        )
        self._activate = h_swish()

    def forward(self, inputs):

        _, _, d, h, w = inputs.shape
        min_dim = min(d, h, w)

        # reduce dimensionality to 2d
        dh = inputs
        dh = F.adaptive_avg_pool3d(dh, (min_dim, min_dim, 1)).squeeze(dim=-1)
        hw = inputs.transpose(2, 4)
        hw = F.adaptive_avg_pool3d(hw, (min_dim, min_dim, 1)).squeeze(dim=-1)
        dw = inputs.transpose(3, 4)
        dw = F.adaptive_avg_pool3d(dw, (min_dim, min_dim, 1)).squeeze(dim=-1)

        # stack three group of 2D feature maps
        x = torch.cat((dh, hw, dw), dim=1)

        # depthwise conv
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._activate(x)

        # reduce dimensionality to 1d
        x = self._se_reduce(x)
        x = self._activate(x)

        # pointwise conv
        x = self._project_conv(x).sigmoid()

        dh, hw, dw = torch.split(x, self._channels, dim=1)
        dh = dh.unsqueeze(dim=-1)
        dw = dw.unsqueeze(dim=-2)
        hw = hw.unsqueeze(dim=-3)
        mask = F.interpolate((dh * dw * hw), size=(d, h, w))
        out = inputs * mask
        return out


class Attention3d(nn.Module):

    def __init__(self, channels, kernel_size=3, bn_mom=0.99, bn_eps=0.001, reduction=32):
        super().__init__()

        self._channels = channels
        self._bn_mom = bn_mom
        self._bn_eps = bn_eps
        self.conv1d, _, _ = get_conv()

        # depthwise conv
        depth_chann = channels * 3
        self._depthwise_conv = self.conv1d(
            in_channels=depth_chann, out_channels=depth_chann, groups=depth_chann,
            kernel_size=kernel_size, bias=False
        )
        self._bn1 = nn.BatchNorm1d(num_features=depth_chann, momentum=self._bn_mom, eps=self._bn_eps)

        # se attention
        se_chann = max(1, depth_chann//reduction)
        self._se_reduce = self.conv1d(in_channels=depth_chann, out_channels=se_chann, kernel_size=1, bias=True)

        # project conv
        self._project_conv = self.conv1d(
            in_channels=se_chann, out_channels=depth_chann,
            kernel_size=1, bias=True
        )
        self._activate = h_swish()

    def forward(self, inputs):

        _b, _c, _d, _h, _w = inputs.shape
        min_dim = min(_d, _h, _w)

        # reduce dimensionality to 2d
        d = inputs
        d = F.adaptive_avg_pool3d(d, (min_dim, 1, 1)).flatten(start_dim=2)
        h = inputs.transpose(2, 4)
        h = F.adaptive_avg_pool3d(h, (min_dim, 1, 1)).flatten(start_dim=2)
        w = inputs.transpose(3, 4)
        w = F.adaptive_avg_pool3d(w, (min_dim, 1, 1)).flatten(start_dim=2)

        # stack three group of 2D feature maps
        x = torch.cat((d, h, w), dim=1)

        # depthwise conv
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._activate(x)

        # reduce dimensionality to 1d
        x = self._se_reduce(x)
        x = self._activate(x)

        # pointwise conv
        x = self._project_conv(x).sigmoid()

        d, h, w = torch.split(x, self._channels, dim=1)
        d = d.view((_b, _c, min_dim, 1, 1))
        h = h.view((_b, _c, 1, min_dim, 1))
        w = w.view((_b, _c, 1, 1, min_dim))
        mask = F.interpolate((d * h * w), size=(_d, _h, _w))
        out = inputs * mask

        return out

        



        

        
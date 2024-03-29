# Author: hys

import math
import torch
from torch import nn
from torch.nn import functional as F
from .utils import get_conv, get_act

class Attention3d_deprecated(nn.Module):

    def __init__(self, channels, kernel_size=3, bn_mom=0.99, bn_eps=0.001, reduction=48):
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
        self._activate = get_act('ReLU')

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

    def __init__(self, channels, kernel_size=3, bn_mom=0.99, bn_eps=0.001, reduction=48):
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
        self._activate = get_act('ReLU')

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


class Attention3d_up(nn.Module):

    def __init__(self, channels, kernel_size=3, bn_mom=0.99, bn_eps=0.001, reduction=48):
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
        self._activate = get_act('ReLU')

    def forward(self, inputs):

        _b, _c, _d, _h, _w = inputs.shape
        max_dim = max(_d, _h, _w)
        unified = F.interpolate(inputs, size=(max_dim, max_dim, max_dim))

        # reduce dimensionality to 2d
        d = unified
        d = F.adaptive_avg_pool3d(d, (max_dim, 1, 1)).flatten(start_dim=2)
        h = unified.transpose(2, 4)
        h = F.adaptive_avg_pool3d(h, (max_dim, 1, 1)).flatten(start_dim=2)
        w = unified.transpose(3, 4)
        w = F.adaptive_avg_pool3d(w, (max_dim, 1, 1)).flatten(start_dim=2)

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
        d = d.view((_b, _c, max_dim, 1, 1))
        h = h.view((_b, _c, 1, max_dim, 1))
        w = w.view((_b, _c, 1, 1, max_dim))
        mask = F.interpolate((d * h * w), size=(_d, _h, _w))
        out = inputs * mask

        return out

class SE(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self._channels = channels
        _, _, self.conv3d = get_conv()
        self._pool = self._pool3d = nn.AdaptiveAvgPool3d(1)

        inc = max(1, channels // reduction)
        self._reduce_conv = self.conv3d(channels, inc, 1, bias=False)
        self._expand_conv = self.conv3d(inc, channels, 1, bias=False)

        self._activate = nn.ReLU()

    def forward(self, inputs):

        x = self._pool(inputs)
        x = self._reduce_conv(x)
        x = self._activate(x)
        x = self._expand_conv(x)

        return torch.sigmoid(x) * inputs


class Attention3d_dev(nn.Module):

    def __init__(self, channels, bn_mom=0.99, bn_eps=0.001, reduction=48):
        super().__init__()

        self._channels = channels
        self._bn_mom = bn_mom
        self._bn_eps = bn_eps
        _, _, self.conv3d = get_conv()

        # depthwise conv
        depth_chann = channels * 3
        self._depthwise_conv = self.conv3d(
            in_channels=depth_chann, out_channels=depth_chann, groups=depth_chann,
            kernel_size=(3, 1, 1), bias=False
        )
        self._bn1 = nn.BatchNorm3d(num_features=depth_chann, momentum=self._bn_mom, eps=self._bn_eps)

        # se attention
        se_chann = max(1, depth_chann//reduction)
        self._se_reduce = self.conv3d(in_channels=depth_chann, out_channels=se_chann, kernel_size=1, bias=True)

        # project conv
        self._project_conv = self.conv3d(
            in_channels=se_chann, out_channels=depth_chann,
            kernel_size=1, bias=True
        )
        self._activate = get_act('ReLU')

    def forward(self, inputs):

        _b, _c, _d, _h, _w = inputs.shape
        min_dim = min(_d, _h, _w)

        # reduce dimensionality to 2d
        d = inputs
        d = F.adaptive_avg_pool3d(d, (min_dim, 1, 1))
        h = inputs.transpose(2, 4)
        h = F.adaptive_avg_pool3d(h, (min_dim, 1, 1))
        w = inputs.transpose(3, 4)
        w = F.adaptive_avg_pool3d(w, (min_dim, 1, 1))

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
        h = h.transpose(2, 4)
        w = w.transpose(3, 4)
        mask = F.interpolate((d * h * w), size=(_d, _h, _w))
        out = inputs * mask

        return out
        

        
# Author: hys

import math

import torch
from torch import nn
import torch.nn.functional as F


class Conv1dStaticSamePadding(nn.Module):
    """
    modified by hys
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        l = x.shape[-1]    
        extra_l = (math.ceil(l / self.stride) - 1) * self.stride - l + self.kernel_size
        l = extra_l // 2
        x = F.pad(x, [l, l])

        x = self.conv(x)
        return x

class Conv2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x

class Conv3dStaticSamePadding(nn.Module):
    """
    modified by hys
    3D version of Conv2dStaticSamePadding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 3
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 3

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 3
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 3

    def forward(self, x):
        d, h, w = x.shape[-3:]
        
        extra_h = (math.ceil(w / self.stride[2]) - 1) * self.stride[2] - w + self.kernel_size[2]
        extra_v = (math.ceil(h / self.stride[1]) - 1) * self.stride[1] - h + self.kernel_size[1]
        extra_d = (math.ceil(d / self.stride[0]) - 1) * self.stride[0] - d + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        front = extra_d // 2
        back = extra_d - front

        x = F.pad(x, [left, right, top, bottom, front, back])

        x = self.conv(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x

class MaxPool3dStaticSamePadding(nn.Module):
    """
    modified by hys
    3D version of MaxPool2dStaticSamePadding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool3d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 3
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 3

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 3
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 3

    def forward(self, x):
        d, h, w = x.shape[-3:]
        
        extra_h = (math.ceil(w / self.stride[2]) - 1) * self.stride[2] - w + self.kernel_size[2]
        extra_v = (math.ceil(h / self.stride[1]) - 1) * self.stride[1] - h + self.kernel_size[1]
        extra_d = (math.ceil(d / self.stride[0]) - 1) * self.stride[0] - d + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        front = extra_d // 2
        back = extra_d - front

        x = F.pad(x, [left, right, top, bottom, front, back])

        x = self.pool(x)
        return x


def get_conv():
    
    return Conv1dStaticSamePadding, Conv2dStaticSamePadding, Conv3dStaticSamePadding


class h_sigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def get_act(act):

    if act == 'h_swish':
        return h_swish()
    if act == 'swish':
        return MemoryEfficientSwish()
    else:
        return nn.ReLU()
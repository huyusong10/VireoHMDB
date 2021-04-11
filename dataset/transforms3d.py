import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

class Fitframe(nn.Module):

    def _fit_diff(self, frames_to_change, diff):

        if len(frames_to_change) != diff:
            frames_to_change = frames_to_change[:-1]
        return frames_to_change

class ReduceFrame(Fitframe):
    
    def __init__(self, target_frame):
        super().__init__()
        self.target = target_frame

    def forward(self, frames):
        n = frames.shape[0]
        if n > self.target:
            diff = n - self.target
            stride = n / diff
            frames_to_del = np.floor(np.arange(0, n, stride))
            frames_to_del = self._fit_diff(frames_to_del, diff)
            frames = [f for idx, f in enumerate(frames) if idx not in frames_to_del]

            return torch.stack(frames)
        else:
            return frames


class Expandframe(Fitframe):

    def __init__(self, target_frame):
        super().__init__()
        self.target = target_frame

    def forward(self, frames):
        n = frames.shape[0]
        if n < self.target:
            diff = self.target - n
            stride = n / diff
            frames_to_add = np.floor(np.arange(0, n, stride)).astype(int)
            frames_to_add = self._fit_diff(frames_to_add, diff)
            frames = frames.tolist()
            for i, frame in enumerate(frames_to_add):
                frames.insert(i+frame, frames[i+frame])
            return torch.tensor(np.stack(frames))

        else:
            return frames


class RandomCenterCrop(nn.Module):

    def __init__(self, p=0.5, size=[224, 224]):
        super().__init__(size)
        self.p = p
        self.size = size

    def forward(self, frames):

        if np.random.rand < p:
            transform = transforms.CenterCrop(self.size)
            frames = transform(frames)

        return frames

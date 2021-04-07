import os
from glob import glob
import random

import numpy as np
import torch
import cv2
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, KFold

import transforms3d

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(img_size=224, target_frame=56):
    return transforms.Compose([
        transforms.RandomChoice([
            transforms.GaussianBlur(kernel_size=3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        ]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(img_size, scale=(0.66, 1.0), ratio=(3. / 4., 4. / 3.)),
        transforms.Normalize(mean=[0.471, 0.448, 0.408], std=[0.234, 0.239, 0.242]),
        transforms3d.Fitframe(target_frame=target_frame),
    ])

def val(img_size=224, target_frame=56):
    return transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Normalize(mean=[0.471, 0.448, 0.408], std=[0.234, 0.239, 0.242]),
        transforms3d.Fitframe(target_frame=target_frame),
    ])

class HMDBDataset(Dataset):

    def __init__(self, video_label, transforms):
        super().__init__()
        self.ids = video_label['ids']
        self.labels = video_label['labels']
        assert len(self.ids) == len(self.labels)
        self.transforms = transforms

    def _imread(self, video_path):

        video = cv2.VideoCapture(video_path)
        frames = []
        has_frame = video.isOpened()

        while(has_frame):
            has_frame, frame = video.read()
            if has_frame:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video.release()

        # stack and convert to 0~1 tensor
        frames = torch.tensor(np.stack(frames)) / 255
        # (d, h, w, c) -> (d, c, h, w)  d: depth or frames
        frames = frames.permute(0, 3, 1, 2)

        return frames

    def __getitem__(self, index):

        frames = self._imread(self.ids[index])
        label = self.labels[index]

        if self.transforms:
            frames = self.transforms(frames)
        
        # (d, c, h, w) -> (c, d, h, w)
        frames = frames.permute(1, 0, 2, 3)

        return {
            'frames': frames,
            'label': label
        }

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    frames = [s['frames'] for s in data]
    labels = [s['label'] for s in data]

    # (b, c, d, h, w)
    frames = torch.stack(frames)
    labels = torch.tensor(labels)

    return {
        'frames': frames,
        'labels': labels
    }


def get_loaders(video_dir, batch_size=4, num_workers=4, frames, img_size, folder=0):

    init_seed(10086)

    ids = []
    labels = []
    for class_id, folder_name in enumerate(os.listdir(video_dir)):
        folder = os.path.join(video_dir, folder_name)
        if os.path.isdir(folder):
            # class_name = os.path.split(folder)[-1]
            videos = glob(os.path.join(folder, '*.avi'))
            if videos:
                ids += videos
                labels += [class_id] * len(videos)

    
    skf = StratifiedKFold(n_splits=5)
    ti, vi = list(skf.split(ids, labels))[fold]
    trains = {
        'ids': [ids[x] for x in ti],
        'labels': [labels[x] for x in ti]
    }
    vals = {
        'ids': [ids[x] for x in vi],
        'labels': [labels[x] for x in vi]
    }

    train_set = HMDBDataset(
        trains,
        train(img_size, frames),
    )
    # print(train_set[0]['frames'].shape)
    val_set = HMDBDataset(
        vals,
        val(img_size, frames),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return {
        'train': train_loader,
        'val': val_loader
    }




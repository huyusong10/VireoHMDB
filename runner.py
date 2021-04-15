import os
import copy
from glob import glob
from tqdm import tqdm

import numpy as np

import torch
from torch.cuda import amp


class Runner():

    def __init__(self, params, net, optim, torch_device, loss, writer, scheduler):
        self.params = params
        self.result = os.path.join(self.params.save_dir, 'results.txt')
        self.writer = writer
        self.torch_device = torch_device
        self.net = net
        self.loss = loss
        self.optim = optim
        self.scheduler = scheduler
        self.start_epoch = 0
        self.top1 = 0.6
        self.save = True if self.params.mode == 'train' else False

    def save(self, epoch, filename="train"):
            torch.save({"epoch": epoch,
                        "network": self.net.module.state_dict(),
                        "top1": self.top1,
                        }, self.params.save_dir + "/%s.pth" % (filename))
            print("Model saved %d epoch" % (epoch))

    def train(self, loaders):

        scaler = amp.GradScaler(enabled=True)
        self.optim.zero_grad()
        for epoch in range(self.start_epoch, self.params.epoch):

            self.net.train()
            loss_ls = []

            nb = len(loaders['train'])
            pbar = tqdm(enumerate(loaders['train']), total=nb)
            for i, (frames_, labels_) in pbar:
                ni = i + nb * epoch
                frames_ = frames_.to(self.torch_device, non_blocking=True)
                labels_ = labels_.to(self.torch_device, non_blocking=True)
                with amp.autocast():
                    preds = self.net(frames_)
                    loss = self.loss(preds, labels_)

                loss_ls.append(loss.item())
                scaler.scale(loss).backward()

                if ni % self.params.accumulate == 0:

                    scaler.step(self.optim)
                    scaler.update()
                    self.optim.zero_grad()

                if self.writer:
                    self.writer.add_scalars('Loss', {'train set': loss.item()}, ni)
                pbar.set_description('Step{}. Epoch: {}/{}. Iteration: {}/{}. Loss: {:.5f}. '
                        .format(ni, epoch, self.params.epoch-1, i,
                                nb-1, loss.item()))

            self.scheduler.step()
            if loaders['val'] is not None:
                val_loss, top1_acc, top5_acc = self.valid(epoch, loaders['val'])
                print('valid loss: {:.4f}, top1: {:.4f}, top5: {:.4f}'.format(val_loss, top1_acc, top5_acc))
            else:
                raise RuntimeError('val_loader not existed!')

            if self.writer:
                lr = self.optim.state_dict()['param_groups'][0]['lr']
                self.writer.add_scalar('lr', lr, epoch)
                self.writer.add_scalars('Loss', {'val set': val_loss}, ni)
                self.writer.add_scalar('Top1', top1_acc, epoch)
                self.writer.add_scalar('Top5', top5_acc, epoch)
                with open(self.result, 'a') as f:
                    f.write('Epoch: {}/{}  LR: {:.5f}  Train loss: {:.4f}  Val loss: {:.4f}  Top1:{:.4f}  Top5: {:.4f}'
                            .format(epoch, self.params.epoch-1, lr, np.mean(loss_ls), val_loss, top1_acc, top5_acc) + '\n')
            
    def valid(self, epoch, loader):
        
        print('validating...')
        loss_ls = []
        acc1, acc5, total = 0, 0, 0

        self.net.eval()
        with torch.no_grad():
            for frames_, labels_ in tqdm(loader, total=len(loader)):

                frames_ = frames_.to(self.torch_device, non_blocking=True)
                labels_ = labels_.to(self.torch_device, non_blocking=True)
                preds = self.net(frames_)

                loss_ls.append(self.loss(preds, labels_).item())

                _, top5 = preds.topk(5)
                total += preds.shape[0]
                labels_ = labels_.view(-1,1)
                acc1 += (top5[:, :1] == labels_).sum().item()
                acc5 += (top5 == labels_).sum().item()

        top1_acc = acc1 / total
        top5_acc = acc5 / total
                
        if top1_acc > self.top1 and self.save:
            self.top1 = top1_acc
            self.save(epoch, "ckpt_%d_%.4f" % (
                epoch, top1_acc))

        return np.mean(loss_ls), top1_acc, top5_acc
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
        self.best_metric = 0.5
        self.save = True if self.params.mode == 'train' else False

    def save(self, epoch, filename="train"):
            torch.save({"epoch": epoch,
                        "network": self.net.module.state_dict(),
                        "best_metric": self.best_metric,
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
                res = self.valid(epoch, loaders['val'], ni)
                print('valid loss: {:.4f}, acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, metric: {:.4f}'.format(res['loss'], res['acc'], res['precision'], res['recall'], res['metric']))
            else:
                raise RuntimeError('val_loader not existed!')

            if self.writer:
                self.writer.add_scalar('lr', self.optim.state_dict()['param_groups'][0]['lr'], epoch)

            with open(self.result, 'a') as f:
                f.write('Epoch: {}/{}  Train loss: {:.4f}  Val loss: {:.4f}  Accuracy:{:.4f}  Presicion: {:.4f}  Recall: {:.4f}  Metric: {:.4f}'
                        .format(epoch, self.params.epoch-1, np.mean(loss_ls), res['loss'], res['acc'], res['precision'], res['recall'], res['metric']) + '\n')
            
            if res['singlecls'] is not None and self.save:
                save_str = ''
                for i in res['singlecls']:
                    save_str += '({:d},{:.4f},{:.4f},{:.4f})  '.format(i[0], i[1], i[2], i[3])
                with open(os.path.join(self.params.save_dir, 'singlecls_results.txt'), 'a') as f:
                    f.write(save_str + '\n')

    def valid(self, epoch, val_loader, step):
        
        print('validating...')
        self.net.eval()
        res = self._get_acc(val_loader, epoch=epoch)

        metric = res['precision']
        res.update({'metric': metric})

        if self.writer:
            self.writer.add_scalars('Loss', {'val set': res['loss']}, step)
            self.writer.add_scalar('Accuracy', res['acc'], epoch)
            self.writer.add_scalar('Precision', res['precision'], epoch)
            self.writer.add_scalar('Recall', res['recall'], epoch)
            self.writer.add_scalar('Metric', res['metric'], epoch)

        if metric > self.best_metric and self.save:
            self.best_metric = metric
            self.save(epoch, "ckpt_%d_%.4f" % (
                epoch, metric))
        return res

    def _get_acc(self, loader, epoch):
        
        loss_ls = []
        pred_ls = []
        label_ls = []

        t_tp = 0  # total true positive
        t_fp = 0  # total false positive
        t_tn = 0  # totaltrue negative
        t_fn = 0  # total false negative

        with torch.no_grad():
            for frames_, labels_ in tqdm(loader, total=len(loader)):
                frames_ = frames_.to(self.torch_device, non_blocking=True)
                labels_gpu = labels_.to(self.torch_device, non_blocking=True)
                preds = self.net(frames_)
                loss_ls.append(self.loss(preds, labels_gpu).item())

                for i, pred in enumerate(preds.cpu()):
                    pred = np.where(pred==pred.max(), 1., 0.)
                    label = [1. if x==labels_[i] else 0. for x in range(self.params.num_classes)]
                    pred_ls.append(pred)
                    label_ls.append(label)

        result_sc = []
        preds = np.array(pred_ls)
        labels = np.array(label_ls)
        assert preds.shape == labels.shape

        for j in range(preds.shape[1]):
            tp, fp, tn, fn = 0, 0, 0, 0

            for i in range(preds.shape[0]):
                if preds[i, j] == 1 and labels[i, j] == 1:
                    tp += 1
                elif preds[i, j] == 1 and labels[i, j] == 0:
                    fp += 1
                elif preds[i, j] == 0 and labels[i, j] == 0:
                    tn += 1
                elif preds[i, j] == 0 and labels[i, j] == 1:
                    fn += 1
                else:
                    raise ValueError('preds is not binary!')
            
            a = (tp+tn)/(tp+fp+tn+fn) if tp+fp+tn+fn!=0 else 0
            p = tp/(tp+fp) if tp+fp!=0 else 0
            r = tp/(tp+fn) if tp+fn!=0 else 0

            t_tp += tp
            t_fp += fp
            t_tn += tn
            t_fn += fn
            result_sc.append([a, p, r])

        singlecls_ls = []
        for i, (a, p, r) in enumerate(result_sc):
            print('{}st class: acc {:.3f} precision {:.3f} recall {:.3f}'.format(i, a, p, r))
            singlecls_ls.append([i, a, p, r])
            if self.writer:
                self.writer.add_scalars('Class_precision', {f'class{i}': p}, epoch)
                self.writer.add_scalars('Class_recall', {f'class{i}': r}, epoch)
                self.writer.add_scalars('Class_acc', {f'class{i}': a}, epoch)

        # mean loss, acc, p, r
        ret = {
            'loss': np.mean(loss_ls),
            'acc': (t_tp+t_tn)/(t_tp+t_fp+t_tn+t_fn),
            'precision': t_tp/(t_tp+t_fp),
            'recall': t_tp/(t_tp+t_fn),
            'singlecls': singlecls_ls
        }

        return ret

    def _metric(self, precision, recall):
        metric = 2 * (precision * recall) / (precision + recall)    # F1 value
        return metric
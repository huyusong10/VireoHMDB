import os
import math
import yaml
import datetime
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
from tensorboardX import SummaryWriter

from runner import Runner
from model.model import VireoNet
from model.x3dm import X3Dm
from model.x3dv import X3DV
from model.x3dv2 import X3DV2
from model.x3dv3 import X3DV3
from model.x3dv4 import X3DV4
from model.x3dc import X3Dc
from model.x3dc_v2 import X3DcV2
from model.light3d import Light3D
from model.light3d_v2 import Light3DV2
from model.light3d_v3 import Light3DV3
from model.light3d_v4 import Light3DV4
from dataset.hmdb_data import get_loaders
from utils.utils import check_file, select_device, init_seeds

class Params:
    def __init__(self, file=r'./params.yml'):
        with open(file) as f:
            self.params = yaml.safe_load(f)

    def __getattr__(self, item):
        return self.params.get(item, None)

    def get_original(self):
        return self.params

def get_model(path, num_classes, model_name):

    model_cls = {
        # 'v3dm': V3Dm,
        # 'v3dmv2': V3DmV2,
        # 'vireodev': VireoDev,
        # 'vireonew': VireoNew,
        # 'vireou': VireoU,
        # 'vireopro': VireoPro,
        # 'vireomax': VireoMax,
        # 'densevireo': DenseVireo,
        # 'densevireov2': DenseVireoV2,
        # 'densevireov3': DenseVireoV3,
        # 'densevireov4': DenseVireoV4,
        'vireonet': VireoNet,
        'x3dm': X3Dm,
        'x3dv': X3DV,
        'x3dv2': X3DV2,
        'x3dv3': X3DV3,
        'x3dv4': X3DV4,
        'x3dc': X3Dc,
        'x3dcv2': X3DcV2,
        'light3d': Light3D,
        'light3dv2': Light3DV2,
        'light3dv3': Light3DV3,
        'light3dv4': Light3DV4,
    }[model_name]

    print('using {}'.format(model_name))
    if path:
        path = check_file(path)
        ckpt = torch.load(path)
        model = model_cls(num_classes=num_classes)
        model.load_state_dict(ckpt['network'], strict=False, map_location='cpu')
    else:
        model = model_cls(num_classes=num_classes)

    return model

def get_loss():
    return nn.CrossEntropyLoss()

def get_scheduler(optim, epoch, warm_up=10):

    def sche_with_warmup(x):
        if x < warm_up:
            lr = 0.9 * (x / warm_up) + 0.1
        else:
            lr = ((1 + math.cos(x * math.pi / epoch)) / 2) * (1 - params.lrf) + params.lrf
        return lr

    return LambdaLR(optim, sche_with_warmup)



if __name__ == "__main__":
    params_file = 'params.yml'
    params_file = check_file(params_file)
    params = Params('params.yml')

    debug = True if params.mode == 'debug' else False

    params.save_dir = os.path.join(os.getcwd(), params.save_dir)
    os.makedirs(params.save_dir, exist_ok=True)

    device = select_device(params.device, batch_size=params.batch_size)
    init_seeds(10086)

    loaders = get_loaders(params.input_dir, params.batch_size, params.num_workers, params.frames, params.img_size, debug=debug)
    net = get_model(params.weights, params.num_classes, params.model)
    # net = nn.DataParallel(net).to(device, non_blocking=True)
    net = net.to(device, non_blocking=True)
    loss = get_loss()

    params.accumulate = max(round(params.nbs / params.batch_size), 1)
    params.weight_decay_nbs = (params.batch_size * params.accumulate / params.nbs) * params.weight_decay

    pg0, pg1, pg2 = [], [], []
    for k, v in net.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm3d) or isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, 'weight'):
            pg1.append(v.weight)

    optim = {
        "adamw" : lambda : torch.optim.AdamW(pg0, lr=params.lr, betas=eval(params.beta), weight_decay=params.weight_decay),
        "SGD": lambda : torch.optim.SGD(pg0, lr=params.lr, momentum=params.momentum, nesterov=True, weight_decay=params.weight_decay),
    }[params.optim]()

    optim.add_param_group({'params': pg1, 'weight_decay': params.weight_decay_nbs})
    optim.add_param_group({'params': pg2})
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    net_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(net_params)
    del pg0, pg1, pg2

    scheduler = get_scheduler(optim, params.epoch, warm_up=params.warmup)

    if debug:
        writer = None
    else:
        writer = SummaryWriter(params.save_dir + '/' + params.note.replace(' ', '') + f'/{datetime.datetime.now().strftime("%Y%m%d:%H%M")}')
        params.save_dir = writer.logdir
        with open(os.path.join(params.save_dir, 'params.yml'), 'w') as f:
            p = params.get_original()
            p.update({'params': net_params})
            yaml.dump(p, f, sort_keys=False)

    model = Runner(params, net, optim, device, loss, writer, scheduler)
    model.train(loaders)  

    if writer:
        writer.close()

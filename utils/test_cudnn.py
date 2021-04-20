import os, sys
import time
import torch
print(torch)
torch.backends.cudnn.benchmark = True
print(torch.backends.cudnn.version())
import torch.nn as nn

feat_size = [
    (8, 64, 32, 56, 56),
    (8, 128, 32, 28, 28),
    (8, 256, 32, 14, 14),
    (8, 512, 32, 7, 7),
    (8, 54, 13, 40, 40),
    (8, 108, 13, 20, 20),
    (8, 216, 13, 10, 10),
    (8, 432, 13, 5, 5),
    (8, 54, 16, 56, 56),
    (8, 108, 16, 28, 28),
    (8, 216, 16, 14, 14),
    (8, 432, 16, 7, 7),
    (8, 54, 16, 78, 78),
    (8, 108, 16, 39, 39),
    (8, 216, 16, 20, 20),
    (8, 432, 16, 10, 10),
    ]
max_iter = 50

print('Running %d times for each config.' % max_iter)
for tcase, size in enumerate(feat_size):
  data = torch.randn(size, requires_grad=True, device='cuda')
  conv = nn.Conv3d(data.size(1), data.size(1),
      kernel_size=(3, 3, 3),
      padding=(1, 1, 1),
      #stride=(1, 2, 2), # may use stride 1 or stride 2
      stride=(1, 1, 1),
      groups=data.size(1),
      bias=False)
  nn.init.kaiming_normal_(conv.weight)
  conv.cuda()

  #data = data.half() # may or may not use fp16.
  #conv.half()

  with torch.backends.cudnn.flags(enabled=True):
    for i in range(50):
      conv(data).sum().backward() # warmup
    torch.cuda.synchronize()
    time_st = time.time()
    for i in range(max_iter):
      conv(data).sum().backward()
    torch.cuda.synchronize()
  time_ed = time.time()
  print('Input size: %s, total_time: %.6f s, avg_time: %.6f s' % (
    str(size), time_ed - time_st, (time_ed - time_st) / max_iter))
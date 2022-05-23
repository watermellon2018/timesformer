import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, p=0.5, phase='train'):
        super().__init__()
        self.p = p
        self.phase = phase

    def forward(self, x):
        if self.phase != 'train' and int(self.p) == 0:
            return x

        prob = torch.rand(x.shape[0])
        ind = torch.where(prob < self.p)
        x = x.div(1 - self.p)
        x[ind] = 0

        return x
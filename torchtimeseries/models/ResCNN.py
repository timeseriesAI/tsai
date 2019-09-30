# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@gmail.com based on:

# Zou, X., Wang, Z., Li, Q., & Sheng, W. (2019). Integration of residual network and convolutional neural network along with various activation functions and global pooling for time series classification. Neurocomputing.
# No official implementation found


import torch
import torch.nn as nn
from .layers import *

__all__ = ['ResCNN']


class Block(nn.Module):
    def __init__(self, ni, nf, ks=[7, 5, 3], act_fn='relu'):
        super().__init__()
        self.conv1 = convlayer(ni, nf, ks[0], act_fn=act_fn)
        self.conv2 = convlayer(nf, nf, ks[1], act_fn=act_fn)
        self.conv3 = convlayer(nf, nf, ks[2], act_fn=False)

        # expand channels for the sum if necessary
        self.shortcut = noop if ni == nf else convlayer(ni, nf, ks=1, act_fn=False)
        self.act_fn = get_act_layer(act_fn)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        sc = self.shortcut(res)
        x += sc
        x = self.act_fn(x)
        return x


class ResCNN(nn.Module):
    def __init__(self, c_in, c_out):
        nf = 64
        super().__init__()
        self.block = Block(c_in, nf, ks=[7, 5, 3], act_fn='relu')
        self.conv1 = convlayer(nf, nf * 2, ks=3, act_fn='leakyrelu', negative_slope=.2)
        self.conv2 = convlayer(nf * 2, nf * 4, ks=3, act_fn='prelu')
        self.conv3 = convlayer(nf * 4, nf * 2, ks=3, act_fn='elu', alpha=.3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.lin = nn.Linear(nf * 2, c_out)

    def forward(self, x):
        x = self.block(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x).squeeze(-1)
        return self.lin(x)
# This is an unofficial Multivariate, GPU (PyTorch) implementation by Ignacio Oguiza - oguiza@gmail.com based on:

# Dempster, A., Petitjean, F., & Webb, G. I. (2019). ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. arXiv preprint arXiv:1910.13051.
# GitHub code: https://github.com/angus924/rocket

import torch
import torch.nn as nn
import numpy as np

class ROCKET(nn.Module):
    def __init__(self, c_in, seq_len, n_kernels=10000, kss=[7, 9, 11]):
        
        '''
        ROCKET is a GPU Pytorch implementation of the original ROCKET methods generate_kernels and apply_kernels that can be used with univariate and multivariate time series.
        Input: is a 3d torch tensor of type torch.float32. When used with univariate TS, make sure you transform the 2d to 3d by adding unsqueeze(1)
        c_in: number of channels in (features). For univariate c_in is 1.
        seq_len: sequence length (is the last dimension of the input)
        '''
        super().__init__()
        kss = [ks for ks in kss if ks < seq_len]
        convs = nn.ModuleList()
        for i in range(n_kernels):
            ks = np.random.choice(kss)
            dilation = 2**np.random.uniform(0, np.log2((seq_len - 1) // (ks - 1)))
            padding = int((ks - 1) * dilation // 2) if np.random.randint(2) == 1 else 0
            weight = torch.normal(0, 1, (1, c_in, ks))
            weight -= weight.mean()
            bias = 2 * (torch.rand(1) - .5)
            layer = nn.Conv1d(c_in, 1, ks, padding=2 * padding, dilation=int(dilation), bias=True)
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
            convs.append(layer)
        self.convs = convs
        self.n_kernels = n_kernels
        self.kss = kss

    def forward(self, x):
        for i in range(self.n_kernels):
            out = self.convs[i](x)
            _max = out.max(dim=-1).values
            _ppv = torch.gt(out, 0).sum(dim=-1).float() / out.shape[-1]
            cat = torch.cat((_max, _ppv), dim=-1)
            output = cat if i == 0 else torch.cat((output, cat), dim=-1)
        return output
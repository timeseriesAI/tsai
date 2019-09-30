# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@gmail.com based on:

# Wang, Z., Yan, W., & Oates, T. (2017, May). Time series classification from scratch with deep neural networks: A strong baseline. In 2017 international joint conference on neural networks (IJCNN) (pp. 1578-1585). IEEE.

# Fawaz, H. I., Forestier, G., Weber, J., Idoumghar, L., & Muller, P. A. (2019). Deep learning for time series classification: a review. Data Mining and Knowledge Discovery, 33(4), 917-963.
# FCN TensorFlow implementation: https://github.com/hfawaz/dl-4-tsc

# 👀 kernel filter size 8 has been replaced by 7 (I believe it's a bug)

import torch
import torch.nn as nn
from .layers import *

__all__ = ['FCN']

class FCN(nn.Module):
    def __init__(self,c_in,c_out,layers=[128,256,128],kss=[7,5,3]):
        super().__init__()
        self.conv1 = convlayer(c_in,layers[0],kss[0])
        self.conv2 = convlayer(layers[0],layers[1],kss[1])
        self.conv3 = convlayer(layers[1],layers[2],kss[2])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(layers[-1],c_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)
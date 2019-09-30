
#original FTSwish = https://arxiv.org/abs/1812.06247

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.torch_core import Module
from typing import Callable
from functools import partial
from . import *
#from . import FCN, ResNet, Res2Net, InceptionTime, InceptionTimePlus, ResNetPlus2


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def noop(x): return x

class LambdaPlus(Module):
    def __init__(self, func, *args, **kwargs): self.func,self.args,self.kwargs=func,args,kwargs
    def forward(self, x): return self.func(x, *self.args, **self.kwargs)
    
    
def get_act_layer(act_fn, **kwargs):
    act_fn = act_fn.lower()
    assert act_fn in ['relu', 'leakyrelu', 'prelu', 'elu', 
                      'mish', 'swish'], 'incorrect act_fn'
    if act_fn == 'relu': return nn.ReLU()
    elif act_fn == 'leakyrelu': return nn.LeakyReLU(**kwargs)
    elif act_fn == 'prelu': return nn.PReLU(**kwargs)
    elif act_fn == 'elu': return nn.ELU(**kwargs)
    elif act_fn == 'mish': return Mish()
    elif act_fn == 'swish': return Swish()
    
    
def same_padding1d(seq_len,ks,stride=1,dilation=1):
    assert stride > 0
    assert dilation >= 1
    effective_ks = (ks - 1) * dilation + 1
    out_dim = (seq_len + stride - 1) // stride
    p = max(0, (out_dim - 1) * stride + effective_ks - seq_len)
    padding_before = p // 2
    padding_after = p - padding_before
    return padding_before, padding_after

class ZeroPad1d(nn.ConstantPad1d):
    def __init__(self, padding):
        super().__init__(padding, 0.)

class ConvSP1d(nn.Module):
    "Conv1d padding='same'"
    def __init__(self,c_in,c_out,ks,stride=1,padding='same',dilation=1,bias=True):
        super().__init__()
        self.ks, self.stride, self.dilation = ks, stride, dilation
        self.conv = nn.Conv1d(c_in,c_out,ks,stride=stride,padding=0,dilation=dilation,bias=bias)
        self.zeropad = ZeroPad1d
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, x):
        padding = same_padding1d(x.shape[-1],self.ks,stride=self.stride,dilation=self.dilation)
        return self.conv(self.zeropad(padding)(x))

def convlayer(c_in,c_out,ks=3,padding='same',bias=True,stride=1,
              bn_init=False,zero_bn=False,bn_before=True,
              act_fn='relu', **kwargs):
    '''conv layer (padding="same") + bn + act'''
    if ks % 2 == 1 and padding == 'same': padding = ks // 2
    layers = [ConvSP1d(c_in,c_out, ks, bias=bias, stride=stride) if padding == 'same' else \
    nn.Conv1d(c_in,c_out, ks, stride=stride, padding=padding, bias=bias)]
    bn = nn.BatchNorm1d(c_out)
    if bn_init: nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    if bn_before: layers.append(bn)
    if act_fn: layers.append(get_act_layer(act_fn, **kwargs))
    if not bn_before: layers.append(bn)
    return nn.Sequential(*layers)


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)
    
class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x): return x.squeeze(dim=self.dim)
    
class Unsqueeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x): return x.unsqueeze(dim=self.dim)
    
class YRange(nn.Module):
    def __init__(self, y_range:tuple): 
        super().__init__()
        self.y_range = y_range
    def forward(self, x):
        x = F.sigmoid(x)
        x = x * (self.y_range[1] - self.y_range[0])
        return x + self.y_range[0]
    
    
class Mult(nn.Module):
    def __init__(self, mult, trainable=True):
        self.mult,self.trainable = mult,trainable
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(1).fill_(self.mult).to(device), requires_grad=self.trainable)
    def forward(self, x):
        return x.mul_(self.weight)
    
    
class Exp(nn.Module):
    def __init__(self, exp, trainable=True):
        self.exp,self.trainable = exp,trainable
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(1).fill_(self.exp).to(device), requires_grad=self.trainable)
    def forward(self, x):
        return x**self.weight
    

def get_cls(model, c_in, seq_len, c_out, **kwargs):
    if model.lower() == 'fcn': return FCN(c_in, c_out, **kwargs)
    elif model.lower() == 'resnet': return ResNet.ResNet(c_in, c_out, **kwargs)
    elif model.lower() == 'inceptiontime': return InceptionTime(c_in, c_out, **kwargs)
    elif model.lower() == 'rescnn': return ResCNN(c_in, c_out, **kwargs)
    else: print('Model not available!!')
        

# ACTIVATION LAYERS
class FTSwishPlus(nn.Module):
    def __init__(self, threshold=-.25, sub=-.1, **kwargs):
        super().__init__()
        self.threshold,self.sub = threshold,sub

    def forward(self, x): 
        x = F.relu(x) * torch.sigmoid(x) + self.threshold
        if self.sub is not None: x.sub_(self.sub)
        return x


class Swish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)
    
    
class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=0., maxv=None, **kwargs):
        super().__init__()
        self.leak,self.sub,self.maxv = leak,sub,maxv

    def forward(self, x): 
        x = F.leaky_relu(x,self.leak) if self.leak is not None else F.relu(x)
        x = x.sub_(self.sub)
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x
    

#Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
#https://arxiv.org/abs/1908.08681v1
#implemented for PyTorch / FastAI by lessw2020 
#github: https://github.com/lessw2020/mish
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x *( torch.tanh(F.softplus(x)))
        return x

def get_act_fn_norm(act_fn):
    x = torch.randn(1000000)
    x = act_fn(x)
    x1 = x / x.std()
    x2 = x1- x1.mean()
    return 1/x.std(), x1.mean()


class AFN(nn.Module):
    def __init__(self, act_fn=F.relu, trainable=True, **kwargs):
        super().__init__()
        self.act_fn = partial(act_fn,**kwargs)
        mul, add = get_act_fn_norm(self.act_fn)
        self.weight = nn.Parameter(torch.Tensor(1).fill_(mul).to(device), requires_grad=trainable)
        self.bias = nn.Parameter(torch.Tensor(1).fill_(add).to(device), requires_grad=trainable)
        
    def forward(self, x): 
        x = self.act_fn(x)
        x.mul_(self.weight)
        x.add_(self.bias)
        return x

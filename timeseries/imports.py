import fastai2
import fastcore
import torch
from fastai2.imports import *
from fastai2.data.all import *
from fastai2.torch_core import *
from fastai2.learner import *
from fastai2.metrics import *
from fastai2.callback.all import *
from fastai2.vision.data import *
from fastai2.interpret import *
import pprint
import psutil
import scipy as sp

PATH = Path(os.getcwd())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from IPython.display import display, HTML
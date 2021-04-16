import fastai
from fastai.imports import *
from fastai.data.all import *
from fastai.torch_core import *
from fastai.learner import *
from fastai.metrics import *
from fastai.callback.all import *
from fastai.vision.data import *
from fastai.interpret import *
from fastai.optimizer import *
from fastai.torch_core import Module
from fastai.data.transforms import get_files
from fastai.tabular.all import *
import fastcore
from fastcore.test import *
from fastcore.utils import *
import torch
import torch.nn as nn
import scipy as sp
import sklearn.metrics as skm
from sklearn.metrics import make_scorer
import gc
import os
from numbers import Integral
from pathlib import Path
import time
from time import gmtime, strftime
import pytz # timezone
import sklearn
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV # needed by rocket!
from IPython.display import Audio, display, HTML, Javascript, clear_output
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

PATH = Path(os.getcwd())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cpus = defaults.cpus

import sys
IS_COLAB = 'google.colab' in sys.modules
if IS_COLAB:
    from numba import config
    config.THREADING_LAYER = 'omp'

def save_nb(verbose=False):
    display(Javascript('IPython.notebook.save_checkpoint();'))
    time.sleep(3)
    pv('\nCurrent notebook saved.\n', verbose)


def last_saved(max_elapsed=60):
    print('\n')
    lib_path = Path(os.getcwd()).parent
    folder = lib_path / 'tsai'
    print('Checking folder:', folder)
    counter = 0
    elapsed = 0
    current_time = time.time()
    for fp in get_files(folder):
        fp = str(fp)
        fn = fp.split('/')[-1]
        if not fn.endswith(".py") or fn.startswith("_") or fn.startswith(
                ".") or fn in ['imports.py', 'all.py']:
            continue
        elapsed_time = current_time - os.path.getmtime(fp)
        if elapsed_time > max_elapsed:
            print(f"{fn:30} saved {elapsed_time:10.0f} s ago ***")
            counter += 1
        elapsed += elapsed_time
    if counter == 0:
        print('Correct conversion! 😃')
        output = 1
    else:
        print('Incorrect conversion! 😔')
        output = 0
    print(f'Total time elapsed {elapsed:.0f} s')
    print(strftime("%A %d/%m/%y %T %Z"))
    return output


def beep(inp=1, duration=.1, n=1):
    rate = 10000
    mult = 1.6 * inp if inp else .08
    wave = np.sin(mult*np.arange(rate*duration))
    for i in range(n): 
        display(Audio(wave, rate=10000, autoplay=True))
        time.sleep(duration / .1)


def create_scripts(max_elapsed=60):
    from nbdev.export import notebook2script
    save_nb()
    notebook2script()
    return last_saved(max_elapsed)


class Timer:
    def start(self, verbose=True): 
        self.all_elapsed = 0
        self.n = 0
        self.verbose = verbose
        self.start_dt = datetime.now()
        self.start_dt0 = self.start_dt

    def elapsed(self):
        end_dt = datetime.now()
        self.n += 1
        assert hasattr(self, "start_dt0"), "You need to first use timer.start()"
        elapsed = end_dt - self.start_dt
        if self.all_elapsed == 0: self.all_elapsed = elapsed
        else: self.all_elapsed += elapsed
        pv(f'Elapsed time ({self.n:3}): {elapsed}', self.verbose)
        self.start_dt = datetime.now()
        if not self.verbose: return elapsed

    def stop(self):
        end_dt = datetime.now()
        self.n += 1
        assert hasattr(self, "start_dt0"), "You need to first use timer.start()"
        elapsed = end_dt - self.start_dt
        if self.all_elapsed == 0: self.all_elapsed = elapsed
        else: self.all_elapsed += elapsed
        total_elapsed = end_dt - self.start_dt0
        delattr(self, "start_dt0")
        delattr(self, "start_dt")
        if self.verbose:
            if self.n > 1:
                print(f'Elapsed time ({self.n:3}): {elapsed}')
                print(f'Total time        : {self.all_elapsed}')
            else: 
                print(f'Total time        : {total_elapsed}')
        else: return total_elapsed

timer = Timer()
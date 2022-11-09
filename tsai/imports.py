import platform, os
if platform.system()=='Darwin':
    # workaround "OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized"
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor    

import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable,Generator,Sequence,Iterator,List,Set,Dict,Union,Optional
from functools import partial
import math
import random
import gc
import sys
from numbers import Integral
from pathlib import Path
import time
from time import strftime
from IPython.display import HTML, Javascript, display, Audio
import importlib
import warnings
from warnings import warn
import psutil

import fastcore
from fastcore.imports import *
from fastcore.basics import *
from fastcore.xtras import *
from fastcore.test import *
from fastcore.foundation import *
from fastcore.meta import *
from fastcore.dispatch import *

import fastai
from fastai.basics import *
from fastai.imports import *
from fastai.torch_core import *

import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from configparser import ConfigParser
config = ConfigParser(delimiters=['='])
config.read('../settings.ini')
cfg = config['DEFAULT']
lib_name = cfg.get('lib_name')

def get_gpu_memory():
    import subprocess
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_values = [round(int(x.split()[0])/1024, 2) for i, x in enumerate(memory_info)]
    return memory_values

def get_ram_memory():
    nbytes = psutil.virtual_memory().total
    return round(nbytes / 1024**3, 2)

def is_installed(
    module_name:str # name of the module that is being checked
    ):
    "Determines if a module is installed without importing it"
    return importlib.util.find_spec(module_name) is not None

def is_lab():
    import re
    return any(re.search('jupyter-lab', x) for x in psutil.Process().parent().cmdline())

def is_nb():
    from IPython.core import getipython
    return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'

def is_colab():
    from IPython.core import getipython
    return 'google.colab' in str(getipython.get_ipython())

def to_local_time(t, time_format='%Y-%m-%d %H:%M:%S'):
    return time.strftime(time_format, time.localtime(t))

def _save_nb():
    """Save and checkpoints current jupyter notebook."""

    if is_lab():
        script = """
        this.nextElementSibling.focus();
        this.dispatchEvent(new KeyboardEvent('keydown', {key:'s', keyCode: 83, metaKey: true}));
        """
        display(HTML(('<img src onerror="{}" style="display:none">'
                      '<input style="width:0;height:0;border:0">').format(script)))
    else:
        display(Javascript('IPython.notebook.save_checkpoint();'))

def save_nb(nb_name=None, attempts=1, verbose=True, wait=2):
    """
    Save and checkpoints current jupyter notebook. 1 attempt per second.
    """
    if is_colab():
        if verbose:
            print('cannot save the notebook in Google Colab. You should save it manually 👋')
        return
    if nb_name is None:
        _save_nb()
    else:
        saved = False
        current_time = time.time()
        for i in range(attempts):
            _save_nb()
            # confirm it's saved. This takes come variable time.
            for j in range(20):
                time.sleep(.5)
                saved_time = os.path.getmtime(nb_name)
                if  saved_time >= current_time: break
            if saved_time >= current_time:
                saved = True
                break
        if verbose:
            if saved:
                print(f'{nb_name} saved at {to_local_time(saved_time)}')
            else:
                print(f"{nb_name} couldn't be saved automatically. You should save it manually 👋")
    time.sleep(wait)

def maybe_mount_gdrive():
    if is_colab():
        from google.colab.drive import mount
        if not Path("/content/drive").exists(): mount("/content/drive")
    else:
        print("You cannot mount google drive because you are not using Colab")

def py_last_saved(nb_name, max_elapsed=1):
    if nb_name is None:
        print("Couldn't get nb_name")
        output = 0
    else:
        lib_path = Path(os.getcwd()).parent
        folder = Path(lib_path / lib_name)
        if nb_name == "index.ipynb":
            script_name = nb_name
        else:
            script_name = str(folder/".".join([str(nb_name).split("_", 1)[1:][0].replace(".ipynb", "").replace(".", "/"), "py"]))
        elapsed_time = time.time() - os.path.getmtime(script_name)
        if elapsed_time < max_elapsed:
            print('Correct notebook to script conversion! 😃')
            output = 1
        else:
            print(f"{script_name:30} saved {elapsed_time:10.0f} s ago ***")
            print('Incorrect notebook to script conversion! 😔')
            output = 0
#         print(f'Total time elapsed {elapsed_time:.3f} s')
        print(strftime("%A %d/%m/%y %T %Z"))
    return output

def beep(inp=1, duration=.1, n=1):
    rate = 10000
    mult = 1.6 * inp if inp else .08
    wave = np.sin(mult*np.arange(rate*duration))
    for i in range(n):
        display(Audio(wave, rate=10000, autoplay=True))
        time.sleep(duration / .1)
    
def create_scripts(nb_name, max_elapsed=60, wait=2):
    "Function that saves a notebook, converts it to .py and checks it's been correctly converted"
    from nbdev.export import nb_export
    save_nb(nb_name, wait=wait)
    if nb_name is not None:
        path = Path.cwd()
        nb_name = Path(nb_name).name
        full_nb_name = str(path/nb_name)
        nb_export(full_nb_name)
        output = py_last_saved(nb_name, max_elapsed)
        beep(output)
    else:
        nb_export()

class Timer:
    def __init__(self, verbose=True, return_seconds=True, instance=None):
        self.verbose, self.return_seconds, self.instance = verbose, return_seconds, instance

    def start(self, verbose=None, return_seconds=None, instance=None):
        if verbose is not None:
            self.verbose = verbose
        if return_seconds is not None:
            self.return_seconds = return_seconds
        if instance is not None:
            self.instance = instance
        self.all_elapsed = 0
        self.n = 0
        self.start_dt = datetime.datetime.now()
        self.start_dt0 = self.start_dt
        return self

    def elapsed(self):
        end_dt = datetime.datetime.now()
        self.n += 1
        assert hasattr(self, "start_dt0"), "You need to first use timer.start()"
        elapsed = end_dt - self.start_dt
        if self.all_elapsed == 0: self.all_elapsed = elapsed
        else: self.all_elapsed += elapsed
        if self.return_seconds:
            elapsed = elapsed.total_seconds()
        if self.instance is not None:
            pv(f'Elapsed time ({self.n:3}) ({self.instance:3}): {elapsed}', self.verbose)
        else:
            pv(f'Elapsed time ({self.n:3}): {elapsed}', self.verbose)
        self.start_dt = datetime.datetime.now()
        return elapsed

    def stop(self):
        end_dt = datetime.datetime.now()
        self.n += 1
        assert hasattr(self, "start_dt0"), "You need to first use timer.start()"
        elapsed = end_dt - self.start_dt
        if self.all_elapsed == 0:
            self.all_elapsed = elapsed
        else:
            self.all_elapsed += elapsed
        total_elapsed = end_dt - self.start_dt0
        if self.return_seconds:
            total_elapsed = total_elapsed.total_seconds()
        delattr(self, "start_dt0")
        delattr(self, "start_dt")

        if self.verbose:
            if self.n > 1:
                if self.return_seconds:
                    elapsed = elapsed.total_seconds()
                    self.all_elapsed = self.all_elapsed.total_seconds()
                if self.instance is not None:
                    print(f'Elapsed time ({self.n:3}) ({self.instance:3}): {elapsed}')
                    print(f'Total time         ({self.instance:3}): {self.all_elapsed}')
                else:
                    print(f'Elapsed time ({self.n:3}): {elapsed}')
                    print(f'Total time              : {self.all_elapsed}')
            else:
                if self.instance is not None:
                    print(f'Total time         ({self.instance:3}): {total_elapsed}')
                else:
                    print(f'Total time              : {total_elapsed}')
        return total_elapsed

timer = Timer()

def import_file_as_module(filepath, return_path=False):
    filepath = Path(filepath)
    sys.path.append("..")
    if str(filepath.parent) != ".":
        sys.path.append(str(filepath.parent))
        mod_path = ".".join([str(filepath.parent).replace("/", "."), filepath.stem])
        package, name = mod_path.rsplit(".", 1)
    else:
        mod_path = filepath.stem
        name, package = mod_path, None
    try:
        module = importlib.import_module(mod_path)
    except:
        module = importlib.import_module(name, package)
    if return_path:
        return module, mod_path
    else: return module

def my_setup(*pkgs):
    warnings.filterwarnings("ignore")
    try:
        print(f'os              : {platform.platform()}')
    except:
        pass
    try:
        from platform import python_version
        print(f'python          : {python_version()}')
    except:
        pass
    try:
        import tsai
        print(f'tsai            : {tsai.__version__}')
    except:
        print(f'tsai            : N/A')
    try:
        import fastai
        print(f'fastai          : {fastai.__version__}')
    except:
        print(f'fastai          : N/A')
    try:
        import fastcore
        print(f'fastcore        : {fastcore.__version__}')
    except:
        print(f'fastcore        : N/A')

    if pkgs:
        for pkg in pkgs:
            try: print(f'{pkg.__name__:15} : {pkg.__version__}')
            except: pass
    try:
        import torch
        print(f'torch           : {torch.__version__}')
        try:
            import torch_xla
            print(f'device          : TPU')
        except:
            iscuda = torch.cuda.is_available()
            if iscuda:
                device_count = torch.cuda.device_count()
                gpu_text = 'gpu' if device_count == 1 else 'gpus'
                print(f'device          : {device_count} {gpu_text} ({[torch.cuda.get_device_name(i) for i in range(device_count)]})')
            else:
                print(f'device          : {device}')
    except: pass
    try:
        print(f'cpu cores       : {psutil.cpu_count(logical=False)}')
    except:
        print(f'cpu cores       : N/A')
    try:
        print(f'threads per cpu : {psutil.cpu_count() // psutil.cpu_count(logical=False)}')
    except:
        print(f'threads per cpu : N/A')
    try:
        print(f'RAM             : {get_ram_memory()} GB')
    except:
        print(f'RAM             : N/A')
    try:
        print(f'GPU memory      : {get_gpu_memory()} GB')
    except:
        print(f'GPU memory      : N/A')


computer_setup = my_setup
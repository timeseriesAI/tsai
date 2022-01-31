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
from IPython.core.display import Javascript, display, HTML    
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cpus = defaults.cpus

import sys
IS_COLAB = 'google.colab' in sys.modules
if IS_COLAB:
    from numba import config
    config.THREADING_LAYER = 'omp'

def is_lab():
    import re
    import psutil
    return any(re.search('jupyter-lab', x) for x in psutil.Process().parent().cmdline())

def is_colab():
    from IPython.core import getipython
    return 'google.colab' in str(getipython.get_ipython())

def is_lab():
    import re
    import psutil
    return any(re.search('jupyter-lab', x) for x in psutil.Process().parent().cmdline())

def is_colab():
    from IPython.core import getipython
    return 'google.colab' in str(getipython.get_ipython())

def to_local_time(t, time_format='%Y-%m-%d %H:%M:%S'):
    return time.strftime(time_format, time.localtime(t))

def _save_nb():
    """
    Save and checkpoints current jupyter notebook.
    """
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
    
    if nb_name is None:
        if is_colab(): 
            if verbose: 
                print('cannot save the notebook in Google Colab. Save it manually.')
        else:
            _save_nb()
    else:
        saved = False
        current_time = time.time()
        if is_colab(): 
            if verbose: print(f'cannot save the notebook in Google Colab. Last saved {to_local_time(os.path.getmtime(nb_name))}.')
        else: 
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
        assert saved, f"{nb_name} couldn't be saved."
        if verbose:
            print(f'{nb_name} saved at {to_local_time(saved_time)}.')
    time.sleep(wait)

def maybe_mount_drive():
    from pathlib import Path
    from google.colab.drive import mount
    if not Path("/content/drive").exists(): mount("/content/drive")

def all_last_saved(max_elapsed=60):
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
                ".") or fn in ['imports.py', 'all.py']: # add here files without a notebook
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
    print(f'Total time elapsed {elapsed:.3f} s')
    print(strftime("%A %d/%m/%y %T %Z"))
    return output

def py_last_saved(nb_name, max_elapsed=1):
    print('\n')
    lib_path = Path(os.getcwd()).parent
    folder = Path(lib_path / 'tsai')
    script_name = str(folder/".".join([str(nb_name).split("_", 1)[1:][0].replace(".ipynb", "").replace(".", "/"), "py"]))
    elapsed_time = time.time() - os.path.getmtime(script_name)
    if elapsed_time < max_elapsed:
        print('Correct conversion! 😃')
        output = 1
    else:
        print(f"{script_name:30} saved {elapsed_time:10.0f} s ago ***")
        print('Incorrect conversion! 😔')
        output = 0
    print(f'Total time elapsed {elapsed_time:.3f} s')
    print(strftime("%A %d/%m/%y %T %Z"))
    return output

def beep(inp=1, duration=.1, n=1):
    rate = 10000
    mult = 1.6 * inp if inp else .08
    wave = np.sin(mult*np.arange(rate*duration))
    for i in range(n): 
        display(Audio(wave, rate=10000, autoplay=True))
        time.sleep(duration / .1)

def create_scripts(nb_name=None, max_elapsed=60, wait=2):
    from nbdev.export import notebook2script
    if nb_name is not None: wait = 0
    try: save_nb(nb_name)
    except: save_nb(wait=wait)
    time.sleep(0.5)
    notebook2script(nb_name)
    if nb_name is None: output = all_last_saved(max_elapsed=max_elapsed)
    else: output = py_last_saved(nb_name=nb_name, max_elapsed=max_elapsed)
    beep(output)
    return output

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

def import_file_as_module(filepath, return_path=False):
    from pathlib import Path
    import sys
    import importlib
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
    if return_path: return module, module_path
    else: return module

def my_setup(*pkgs):
    import warnings
    warnings.filterwarnings("ignore")
    try: 
        import platform
        print(f'os             : {platform.platform()}')
    except: 
        pass
    try: 
        from platform import python_version
        print(f'python         : {python_version()}')
    except: 
        pass
    try: 
        import tsai
        print(f'tsai           : {tsai.__version__}')
    except: 
        print(f'tsai           : N/A')
    try: 
        import fastai
        print(f'fastai         : {fastai.__version__}')
    except: 
        print(f'fastai         : N/A')
    try: 
        import fastcore
        print(f'fastcore       : {fastcore.__version__}')
    except: 
        print(f'fastcore       : N/A')
    
    if pkgs is not None: 
        for pkg in listify(pkgs):
            try: print(f'{pkg.__name__:15}: {pkg.__version__}')
            except: pass 
    try: 
        import torch
        print(f'torch          : {torch.__version__}')
        try:
            import torch_xla
            print(f'device         : TPU')
        except:
            iscuda = torch.cuda.is_available()
            print(f'n_cpus         : {cpus}')
            print(f'device         : {device} ({torch.cuda.get_device_name(0)})' if iscuda else f'device         : {device}')
    except: print(f'torch          : N/A')
        
computer_setup = my_setup

#export
#This function will be available in fastai 2.5.4
def ismin_torch(min_version):
    from packaging.version import parse
    "Check if `torch.__version__` >= `min_version` using packaging.version"
    return parse(torch.__version__) >= parse(min_version)
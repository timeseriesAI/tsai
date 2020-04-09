__all__ = ['save_nb', 'last_saved']

from IPython.display import display, Javascript
import os
from pathlib import Path
import time
from time import gmtime, strftime
from fastai2.data.transforms import get_files

# Cell
def save_nb():
    display(Javascript('IPython.notebook.save_checkpoint()'))
    time.sleep(1)
    print('\nCurrent notebook saved.\n')

# Cell
def last_saved():
    print()
    lib_path = Path(os.getcwd()).parent
    folder = lib_path/str(lib_path).split('/')[-1]
    current_time = time.time()
    elapsed = 0
    for fp in get_files(folder):
        fp = str(fp)
        fn = fp.split('/')[-1]
        if not fn.endswith(".py") or fn.startswith("_") or fn.startswith(".") or fn in ['imports.py']: continue
        elapsed_time = current_time - os.path.getmtime(fp)
        print(f"{fn:30} saved {elapsed_time:10.0f} s ago")
        elapsed += elapsed_time
    print()
    if elapsed < 1: print('Correct conversion!')
    else: print(f'Total elapsed time {elapsed:.0f} s')
    print(strftime("%d-%m-%Y %H:%M:%S", gmtime()))
    return int(elapsed)
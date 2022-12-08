#!/usr/bin/env python
"Rename ipynb files starting at 01"

import re
import os
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

i = 1
for p in sorted(list(Path().glob('*.ipynb'))):
    pre = re.findall(r'^(\d+)[a-z]?[A-Z]?_(.*)', str(p))
    if not pre:
        continue
    new = f'{i:03d}_{pre[0][1]}'
    print(p, new)
    os.rename(p, new)
    i+=1
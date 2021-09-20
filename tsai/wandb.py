# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Script to run wandb sweeps."""

import os
from tsai.imports import *
from tsai.utils import *
from fastcore.script import *
from fastcore.xtras import *
import argparse

def run_sweep():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str)
    parser.add_argument('--program', type=str)
    parser.add_argument('--launch', action='store_false')
    parser.add_argument('--count', type=int)
    parser.add_argument('--entity', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--sweep_id', type=str)
    parser.add_argument('--relogin', action='store_true')
    parser.add_argument('--login_key', type=str)
    args = parser.parse_args() 

    try:
        import wandb
    except ImportError:
        raise ImportError('You need to install wandb to run sweeps!')

    # Login to W&B
    if args.relogin:
        wandb.login(relogin=True)
    elif args.login_key:
        wandb.login(key=args.login_key)

    # Sweep id
    if not args.sweep_id:
        # Load the sweep config
        assert os.path.isfile(args.sweep), f"can't find file {args.sweep}"
        if isinstance(args.sweep, str):
            sweep = yaml2dict(args.sweep)
        else: 
            sweep = args.sweep
        program = sweep["program"] if args.program is None else args.progam
        # Initialize the sweep
        print('Initializing sweep...')
        sweep_id = wandb.sweep(sweep=sweep, entity=args.entity, project=args.project)
        print('...sweep initialized')
    else: 
        sweep_id = args.sweep_id

    # Load your training script
    print('Loading training script...')
    assert program is not None, "you need to pass either a sweep or program path"
    assert os.path.isfile(program), f"can't find file program = {program}"
    train_script, file_path = import_file_as_module(program, True)
    assert hasattr(train_script, "train")
    train_fn = train_script.train
    print('...training script loaded')

    # Launch agent
    if args.launch:
        print('\nRun additional sweep agents with:\n')
    else:
        print('\nRun sweep agent with:\n')
    print('    from a notebook:')
    print('        import wandb')
    print(f'        from {file_path} import train')
    print(f"        wandb.agent('{sweep_id}', function=train, count=None)\n")
    print('    from a terminal:')
    print(
        f"        wandb agent {os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/{sweep_id}\n")
    if args.launch:
        print('Running agent...')
        wandb.agent(sweep_id, function=train_fn, count=args.count)
        
if __name__ == '__main__':
    run_sweep()
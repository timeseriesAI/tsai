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

@call_parse
def run_sweep(
    sweep: Param("Path to YAML file with the sweep config", str) = None,
    program: Param("Path to Python training script", str) = None,
    launch: Param("Launch wanbd agent.", store_false) = True,
    count: Param("Number of runs to execute", int) = None,
    entity: Param("username or team name where you're sending runs", str) = None,
    project: Param("The name of the project where you're sending the new run.", str) = None,
    sweep_id: Param("Sweep ID. This option omits `sweep`", str) = None,
    relogin: Param("Relogin to wandb.", store_true) = False,
    login_key: Param("Login key for wandb", str) = None,
):

    # import wandb
    try:
        import wandb
    except ImportError:
        raise ImportError('You need to install wandb to run sweeps!')

    # Login to W&B
    if relogin:
        wandb.login(relogin=True)
    elif login_key:
        wandb.login(key=login_key)

    # Sweep id
    if not sweep_id:
        # Load the sweep config
        assert os.path.isfile(sweep), f"can't find file {sweep}"
        if isinstance(sweep, str):
            sweep = yaml2dict(sweep)
        if program is None:
            program = sweep["program"]
        # Initialize the sweep
        print('Initializing sweep...')
        sweep_id = wandb.sweep(sweep=sweep, entity=entity, project=project)
        print('...sweep initialized')

    # Load your training script
    print('Loading training script...')
    assert program is not None, "you need to pass either a sweep or program path"
    assert os.path.isfile(program), f"can't find file program = {program}"
    train_script, file_path = import_file_as_module(program, True)
    assert hasattr(train_script, "train")
    train_fn = train_script.train
    print('...training script loaded')

    # Launch agent
    if launch:
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
    if launch:
        print('Running agent...')
        wandb.agent(sweep_id, function=train_fn, count=count)

if __name__ == '__main__':
    run_sweep()
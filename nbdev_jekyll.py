#!/usr/bin/env python
# coding: utf-8

from fastcore.all import *
from shutil import rmtree
import yaml

rm=['_includes','_layouts','css','fonts','js','licenses','tooltips.json','Gemfile','Gemfile.lock']
raw='https://raw.githubusercontent.com/fastai/nbdev_template/master/docs/'

def rm_files(pth:Path):
    "Removes files controlled by theme"
    for f in rm:
        p = pth/f'{f}'
        if p.is_dir(): rmtree(p, ignore_errors=True)
        elif p.is_file() and p.exists(): p.unlink()
    urlsave(f'{raw}Gemfile', dest=str((pth/'Gemfile')))
    urlsave(f'{raw}Gemfile.lock', dest=str((pth/'Gemfile.lock')))

def config(pth:Path):
    """Edit config file to include remote theme."""
    p = pth/'_config.yml'
    cfg = yaml.safe_load(p.read_text())
    cfg.pop('theme', None)
    cfg['plugins'] = ['jekyll-remote-theme']
    cfg['remote_theme']= 'fastai/nbdev-jekyll-theme'
    p.write_text(yaml.safe_dump(cfg))


@call_parse
def use_theme(docs_dir:Param('nbdev docs directory', str)='docs'):
    pth = Path(docs_dir)
    assert pth.exists(), f'Could not find directory: {pth.absolute()}'
    rm_files(pth)
    config(pth)

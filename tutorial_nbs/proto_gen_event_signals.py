#!/usr/bin/env python
# coding: utf-8

# Sample Signals with Events

from fastcore.foundation import L
import numpy as np
import random



def get_sample_data(length=1000, n_sig=10, dims=2, n_fold=None, split_pct=None):
    assert dims in (2,3)
    X = _get_sample_signals(length=length, n_sig=n_sig)
    y = np.array([f'ID{i:06d}'for i in range(n_sig)])
    events = _get_sample_events(X)
    
    if dims==3:
        X = np.expand_dims(X, 1)
        
    if n_fold is not None:
        folds = _get_sample_folds(n_sig, n_fold)
        return X, y, events, folds
    elif split_pct is not None:
        splits = _get_sample_splits(n_sig, split_pct)
        return X, y, events, splits
    else:
        return X, y, events


def _get_sample_signals(length=1000, n_sig=10):
    if (isinstance(length, tuple) or isinstance(length, tuple)):
        assert len(length) == 2
        lengths = np.random.randint(*length, n_sig)
    else:
        lengths = [length]*n_sig
        
    all_sig = []
    for i, scale_x in enumerate(2.0*np.pi/100*np.linspace(0.8, 1.2, n_sig)):
        n = lengths[i]
        offset = 2.0*np.pi*random.random()
        sig = np.sin(offset + scale_x*np.arange(n))*np.linspace(1.5, 0.5, n)
        all_sig.append(sig)
        
    if (isinstance(length, tuple) or isinstance(length, tuple)):
        return np.array(all_sig, dtype=object)
    else:
        return np.array(all_sig)


def _get_sample_events(data):
    """Use signal peaks for sample events"""
    results = []
    for sig in data:
        deriv = np.diff(sig)
        peaks = np.logical_and(deriv[:-1]> 0, deriv[1:]<= 0)
        results.append(np.where(peaks)[0])
    return results


def _get_sample_folds(n_sig, n_fold):
    folds = np.array([*range(n_fold)]*((n_sig+n_fold-1)//n_fold))
    np.random.shuffle(folds)
    folds = folds[:n_sig]
    assert len(folds) == n_sig
    return folds


def _get_sample_splits(n_sig, split_pct):
    assert isinstance(split_pct, float) or isinstance(split_pct, tuple) or isinstance(split_pct, list)
    all_indices = [*range(n_sig)]
    random.shuffle(all_indices)
    if isinstance(split_pct, float):
        i_val = int(round(split_pct*n_sig))
        splits = L(sorted(all_indices[i_val:]), sorted(all_indices[:i_val]))
    else:
        assert len(split_pct) == 2
        i_val = int(round(split_pct[0]*n_sig))
        i_test = i_val+int(round(split_pct[0]*n_sig))
        splits = L(sorted(all_indices[i_test:]), sorted(all_indices[:i_val]), sorted(all_indices[i_val:i_test]))
    return splits


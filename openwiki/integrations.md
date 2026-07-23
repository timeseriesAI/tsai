---
type: Integration Guide
title: Framework and External Integrations
description: Maps tsai's runtime frameworks, sklearn ecosystem, datasets, optional feature libraries, experiment tools, documentation stack, and OpenWiki automation.
tags: [integrations, dependencies, optuna, wandb, sklearn, nbdev]
---

# Framework and external integrations

## Runtime foundations

### PyTorch and fastai

PyTorch provides tensor operations, neural modules, devices, serialization, and compiler paths such as TorchScript. fastai provides `Learner`, callbacks/events, optimizers, losses, transforms, data foundations, and training methods. `tsai` adapts these systems to time-series shapes and model families; it does not replace their training runtime. See the [architecture overview](/openwiki/architecture/overview.md) and [data and learning flow](/openwiki/architecture/data-and-learning.md).

Current package floors are declared in `pyproject.toml`, including PyTorch `>=2.12,<3`, fastai `>=2.8.7`, and fastcore `>=1.13.2`. Device compatibility changes in recent history include MPS support and TorchScript-safe reshaping, so integration-sensitive layers should be tested on the relevant backend.

### scikit-learn and imbalanced-learn

scikit-learn is a hard dependency used for validation, metrics, preprocessing, calibration, and ROCKET/MiniRocket estimator pipelines. imbalanced-learn supports over/undersampling in split utilities. scikit-learn is capped below 1.8 because of imbalanced-learn compatibility, as recorded in `CHANGELOG.md`.

ROCKET and MiniRocket wrappers also expose sklearn-style pipelines. A recent fix explicitly initialized those pipelines, making estimator initialization and `fit`/`transform` lifecycle checks important.

## Data and feature sources

### Built-in external dataset helpers

`tsai/data/external.py` provides catalogs and download/preparation helpers for UCR classification, Monash regression/forecasting, and long-term forecasting datasets. These feed the [domain workflow](/openwiki/domain-concepts.md) with benchmark data, but normal user data can enter as arrays, tensors, or DataFrames without any download integration.

Network behavior is intentionally separated from default tests. Download errors—especially UCR failures—should be explicit, and notebook tests should prefer local fixtures unless validating the integration itself.

### Optional scientific libraries

The `extras` group declares:

- `sktime` for selected ROCKET/MiniRocket wrappers;
- `tsfresh` for feature extraction in `tsai/data/features.py` and conversion helpers;
- `PyWavelets` for wavelet-related functionality;
- `nbformat` for the `nb2py` notebook conversion command.

These features generally import dependencies lazily and provide targeted installation errors. Cells requiring them should use the configured `extras` test flag. ONNX-related notebook cells use an `onnx` flag, but ONNX packages are not currently in the extras group.

## Experiment orchestration

### Optuna

`run_optuna_study` in `tsai/optuna.py` lazily imports Optuna and supports Bayesian, grid, random, and supplied samplers; resuming joblib-persisted studies; queued evaluations; pruning; saving; visualization; and both single- and multi-objective directions. Recent changes choose multi-objective study creation appropriately and tolerate resumed studies with no finished trials.

Optuna is not declared in `pyproject.toml` extras. Users must install it separately. Tests should cover single and multiple directions, fresh and resumed studies, empty/interrupted state, and saving paths.

### Weights & Biases

`tsai/wandb.py` lazily imports `wandb` when creating or running sweep agents. It can set the target program, create a sweep, run an agent, update known configuration keys, and load a sweep from YAML/dict configuration. If entity or project arguments are omitted, `WANDB_ENTITY` and `WANDB_PROJECT` are read from the environment.

W&B is also not declared in package extras. Do not record actual credentials or secret values in documentation or tests; only the environment variable names are relevant.

Optuna and W&B extend experiment workflows around learners but do not change the core [learner construction path](/openwiki/architecture/data-and-learning.md).

## Development and publishing stack

### nbdev and Quarto

nbdev 3 turns notebooks into generated Python modules, executable tests, and API documentation. Quarto renders and previews the site. These tools define the [development workflow](/openwiki/workflows/development.md), while `.github/workflows/test.yaml` and `.github/workflows/deploy.yaml` connect them to CI and GitHub Pages.

### Packaging and PyPI

setuptools builds wheels and source distributions from `pyproject.toml`. Distribution is PyPI-only for current releases; older conda artifacts remain but are no longer updated. Artifact integrity is checked in CI because the 1.0.0 package discovery configuration omitted subpackages. See the [operations runbook](/openwiki/operations/testing-and-release.md).

### OpenWiki automation

The current working tree includes `.github/workflows/openwiki-update.yml`, a scheduled/manual workflow that installs a pinned OpenWiki release, runs a code-mode update, and opens a documentation pull request. It is untracked at this snapshot, so treat it as proposed local automation rather than established history. Generated wiki pages live only under `openwiki/`; `openwiki/INSTRUCTIONS.md` is the user-authored scope brief and should not be regenerated.

## Integration checklist

When adding an integration:

1. keep third-party imports lazy unless the dependency becomes mandatory;
2. provide an actionable installation error;
3. decide whether it belongs in a named optional dependency group;
4. add flagged notebook tests that exercise the real lifecycle;
5. ensure `tsai.all` still imports without optional packages;
6. document environment variable names without exposing values;
7. test persistence/network failure paths where applicable;
8. update the [source map](/openwiki/source-map.md) and operations guidance if CI or packaging changes.

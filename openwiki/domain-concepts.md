---
type: Domain Guide
title: Time-Series Domain Concepts
description: Defines the task, shape, validation, transform, and model-family concepts that drive tsai behavior and extension decisions.
tags: [domain, time-series, forecasting, classification, models]
---

# Time-series domain concepts

## Supported problem families

`tsai` targets sequential classification, regression, forecasting, imputation, and representation/feature workflows (`README.md`). The high-level learner classes distinguish three common supervised intents:

- `TSClassifier` expects categorical or encoded targets and commonly uses `TSClassification` plus cross entropy and classification metrics.
- `TSRegressor` treats targets as continuous and defaults to mean squared error when no loss is inferred.
- `TSForecaster` shares much of the regression construction path but makes forecasting intent explicit; preparation and split selection carry the critical temporal semantics.

These task labels do not create independent training engines. All three assemble the same [data-to-fastai runtime](/openwiki/architecture/data-and-learning.md) with different defaults and user intent.

## Shape vocabulary

The central X convention is:

```text
samples × variables × sequence length
```

`TSDatasets` calls `to3d`, `TSTensor` exposes variable and length properties, and inference's `new_dl` requires 3D input. Targets may be scalar, vector, or higher-dimensional. Loader property `d` records non-scalar target dimensions so `build_ts_model` can select forecasting arguments or inject an n-dimensional head.

Shape is therefore part of the API, not an implementation detail. A transform that changes rank can alter model construction, default loss behavior, and inference decoding. The [architecture overview](/openwiki/architecture/overview.md) identifies this metadata coupling as a primary pressure point.

## Windowing, horizons, and leakage

A time-series sample often starts as a long sequence rather than a pre-cut tensor. `SlidingWindow` defines:

- **lookback** through `window_len`;
- **sampling overlap** through `stride`;
- **prediction position(s)** through `horizon`;
- **feature and target columns** through `get_x` and `get_y`;
- **incomplete sequence behavior** through `pad_remainder`, padding direction/value, and an optional padding feature.

For forecasting, random splits may leak future structure. `tsai/data/validation.py` includes time, walk-forward, forecasting, and long-term forecasting split helpers in addition to random splitters. `check_overlap`, `check_splits_overlap`, and `leakage_finder` guard index-level overlap; `SlidingWindow` separately guards X/y column overlap for non-positive horizons.

## Transforms and target semantics

Transforms operate at item and batch stages inherited from fastai. In the common classification path:

- X has no item transform;
- `TSClassification` establishes target vocabulary/loss semantics;
- `TSStandardize` is applied as a batch transform.

`TSDatasets` distinguishes no transform, vectorized `TSTfmdLists`, and general item-wise `TfmdLists`. This affects whether preprocessing can be eagerly stacked with `inplace=True`. Target transforms can supply vocabulary, loss behavior, and decoding, linking training and inference.

Data capabilities extend beyond dense time-series tensors into tabular, image, mixed, unwindowed, augmentation, and feature-extraction modules. These are aggregated by `tsai/data/all.py` and mapped in the [source map](/openwiki/source-map.md).

## Model families

The repository includes several distinct design families rather than one canonical network:

| Family | Representative modules | Typical role |
|---|---|---|
| Convolutional/residual | `FCN*`, `ResNet*`, `InceptionTime*`, `XceptionTime*`, `XCM*`, `TCN` | Learned temporal feature extraction for classification/regression. |
| Recurrent and hybrids | `RNN*`, `RNNAttention*`, `RNN_FCN*`, `TransformerRNNPlus` | Sequential recurrence, attention, or convolution-recurrence combinations. |
| Transformer/patch | `TST*`, `PatchTST`, `TSiTPlus`, `TSSequencerPlus`, `ConvTranPlus`, `TSPerceiver` | Attention-based sequence modeling and long-term forecasting. |
| Random convolutional features | `ROCKET*`, `MINIROCKET*`, `MultiRocketPlus`, `Hydra*` | Fast feature transforms, sklearn pipelines, or PyTorch variants. |
| Tabular/multimodal | `TabModel`, `TabTransformer`, `TabFusionTransformer`, `MultiInputNet`, `multimodal.py` | Combine sequential inputs with static, observed, categorical, or continuous features. |
| Specialized | `mWDN`, `OmniScaleCNN`, `gMLP`, `MLP`, `XResNet1d*` | Wavelet, multi-scale, MLP, and 1D residual alternatives. |

The README lists research provenance for many architectures. The runtime registry is not plugin-based: names are explicitly resolved in `tsai/learner.py`, and constructor shapes are dispatched in `tsai/models/utils.py`. Use the [development workflow](/openwiki/workflows/development.md) when extending a family.

## Multimodal feature roles

The learner and model factory accept indexes and embeddings for:

- static categorical and continuous features;
- observed/time-dependent categorical and continuous features;
- patch length and stride;
- fusion-layer dimensions and behavior.

When any such specification is supplied, `build_ts_model` routes through `MultInputWrapper` in `tsai/models/multimodal.py`. This relationship means a nominally standard architecture may execute inside a wrapper with different head and fusion behavior. Changelog entries for sequence-prediction fixes in the multimodal head show that shaped outputs must be included in tests.

## Dataset acquisition and offline behavior

`tsai/data/external.py` exposes classification, regression, forecasting, and long-term forecasting dataset catalogs and download helpers. These are convenience sources for examples and benchmarks, not a runtime requirement for user-owned arrays. Recent test fixes replaced NATOPS and Adult Sample downloads in default tests and made UCR download failures explicit, reflecting two operational rules:

1. default tests should avoid unnecessary network dependence;
2. failed downloads should surface as failures rather than silently yielding invalid data.

External data and optional feature libraries are detailed in [integrations](/openwiki/integrations.md).

<div align="center">
    <img width="60%" src="./docs/images/tsai_logo.svg">
</div>

-----------------

# tsai



![CI](https://github.com/timeseriesai/tsai/workflows/CI/badge.svg) [![PyPI](https://img.shields.io/pypi/v/tsai?color=blue&label=pypi%20version)](https://pypi.org/project/tsai/#description) ![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

## Description
> State-of-the-art Deep Learning library for Time Series and Sequences. 

`tsai` is an open-source deep learning package built on top of Pytorch & fastai focused on state-of-the-art techniques for time series tasks like classification, regression, forecasting, imputation...

`tsai` is currently under active development by timeseriesAI.

## New in tsai:

#### September, 2021
* 🚀🚀 New tutorial notebook on how to **train your model with larger-than-memory datasets in less time achieving 100% GPU usage!!** 🚀🚀   <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/11_How_to_train_big_arrays_fast_in_tsai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

* **`tsai` supports now more input formats**: np.array, np.memmap, zarr, xarray, dask, list, L, ...

#### Previously

* **MINIROCKET** a SOTA Time Series Classification model (now available in Pytorch):
You can now check MiniRocket's performance in our new tutorial notebook <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
> "Using this method, it is possible to train and test a classifier on all of 109 datasets from the UCR archive to state-of-the-art accuracy in less than 10 minutes." A. Dempster et al. (Dec 2020)

* **Multi-class and multi-label time series classification notebook:** you can also check our new tutorial notebook: <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/01a_MultiClass_MultiLabel_TSClassification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* **Self-supervised learning:** Learn how to leverage your unlabeled datasets <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/08_Self_Supervised_TSBERT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

* **New visualization:**
We've also added a new PredictionDynamics callback that will display the predictions during training. This is the type of output you would get in a classification task for example:
<p align="center">
    <img src="https://github.com/timeseriesAI/tsai/blob/main/nbs/multimedia/LSST_PD.gif?raw=true">
</p>

## Installation

You can install the **latest stable** version from pip using:
```
pip install tsai
```

Or you can install the cutting edge version of this library from github by doing:
```
pip install -Uqq git+https://github.com/timeseriesAI/tsai.git
```

Once the install is complete, you should restart your runtime and then run: 

```
from tsai.all import *
```

## Documentation

Here's the link to the [documentation](https://timeseriesai.github.io/tsai/).

## Available models:

Here's a list with some of the state-of-the-art models available in `tsai`:

- [LSTM](https://ieeexplore.ieee.org/abstract/document/6795963/) (Hochreiter, 1997)
- [GRU](https://arxiv.org/abs/1412.3555) (Cho, 2014)
- [MLP](https://arxiv.org/abs/1611.06455) - Multilayer Perceptron (Wang, 2016)
- [FCN](https://arxiv.org/abs/1611.06455) - Fully Convolutional Network (Wang, 2016)
- [ResNet](https://arxiv.org/abs/1611.06455) - Residual Network (Wang, 2016)
- [LSTM-FCN](https://arxiv.org/abs/1709.05206) (Karim, 2017)
- [GRU-FCN](https://arxiv.org/abs/1812.07683) (Elsayed, 2018)
- [MLSTM-FCN](https://www.sciencedirect.com/science/article/abs/pii/S0893608019301200) - Multivariate LSTM-FCN (Karim, 2019)
- [InceptionTime](https://arxiv.org/abs/1909.04939) (Fawaz, 2019)
- [Rocket](https://arxiv.org/abs/1910.13051) (Dempster, 2019)
- [OmniScale](https://arxiv.org/abs/2002.10061) - Omni-Scale 1D-CNN (Tang, 2020)
- [MiniRocket](https://arxiv.org/abs/2102.00457) (Dempster, 2021)

- [ResCNN](https://www.sciencedirect.com/science/article/abs/pii/S0925231220305944) - 1D-ResCNN (Sun , 2020)
- [TCN](https://arxiv.org/abs/1803.01271) - Temporal Convolutional Network (Bai, 2018)
- [TST](https://dl.acm.org/doi/abs/10.1145/3447548.3467401) - Time Series Transformer (Zerveas, 2021)
- [TabModel](https://docs.fast.ai/tabular.model.html#TabularModel) - modified from fastai's TabularModel
- [TabTransformer](https://arxiv.org/pdf/2012.06678) (Huang, 2020)
- [XCM](https://arxiv.org/abs/2005.03645) - Explainable Convolutional Neural Network) (Fauvel, 2020)
- [XceptionTime](https://arxiv.org/abs/1911.03803) (Rahimian, 2019)
- [mWDN](https://dl.acm.org/doi/abs/10.1145/3219819.3220060) - Multilevel wavelet decomposition network (Wang, 2018)

among others!

## How to start using tsai?

To get to know the tsai package, we'd suggest you start with this notebook in Google Colab: **[01_Intro_to_Time_Series_Classification](https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/01_Intro_to_Time_Series_Classification.ipynb)**
It provides an overview of a time series classification task.

We have also develop many other [tutorial notebooks](https://github.com/timeseriesAI/tsai/tree/main/tutorial_nbs). 

To use tsai in your own notebooks, the only thing you need to do after you have installed the package is to run this:

`from tsai.all import *`

## Examples

These are just a few examples of how you can use `tsai`:

### Binary, univariate classification

```
from tsai.all import *
X, y, splits = get_classification_data('ECG200', split_data=False)
tfms = [None, TSClassification()]
batch_tfms = TSStandardize()
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
learn = ts_learner(dls, InceptionTimePlus, metrics=accuracy, cbs=ShowGraph())
learn.fit_one_cycle(100, 3e-4)
```

### Multi-class, multivariate classification

```
from tsai.all import *
X, y, splits = get_classification_data('LSST', split_data=False)
tfms = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
learn = ts_learner(dls, InceptionTimePlus, metrics=accuracy, cbs=ShowGraph())
learn.fit_one_cycle(10, 1e-2)
```

### Multivariate Regression

```
from tsai.all import *
from sklearn.metrics import mean_squared_error
X_train, y_train, X_test, y_test = get_regression_data('AppliancesEnergy')
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
reg = MiniRocketRegressor(scoring=rmse_scorer)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mean_squared_error(y_test, y_pred, squared=False)
```

### Univariate Forecasting

```
from tsai.all import *
ts = get_forecasting_time_series("Sunspots").values
X, y = SlidingWindow(60, horizon=1)(ts)
splits = TimeSplitter(235)(y) 
batch_tfms = TSStandardize()
learn = TSForecaster(X, y, splits=splits, batch_tfms=batch_tfms, bs=512, arch=TST, metrics=mae, cbs=ShowGraph())
learn.fit_one_cycle(50, 1e-3)
```

## How to contribute to tsai?

We welcome contributions of all kinds. Development of enhancements, bug fixes, documentation, tutorial notebooks, ... 

We have created a guide to help you start contributing to tsai. You can read it [here](https://github.com/timeseriesAI/tsai/blob/main/CONTRIBUTING.md).

## Citing tsai

If you use tsai in your research please use the following BibTeX entry:

```text
@Misc{tsai,
    author =       {Ignacio Oguiza},
    title =        {tsai - A state-of-the-art deep learning library for time series and sequential data},
    howpublished = {Github},
    year =         {2020},
    url =          {https://github.com/timeseriesAI/tsai}
}
```

<div align="center">
    <img width="60%" src="./docs/images/tsai_logo.svg">
</div>

-----------------

# Title



![CI](https://github.com/timeseriesai/tsai/workflows/CI/badge.svg) [![PyPI](https://img.shields.io/pypi/v/tsai?color=blue&label=pypi%20version)](https://pypi.org/project/tsai/#description) ![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

## Description
> State-of-the-art Deep Learning library for Time Series and Sequences. 

`tsai` is an open-source deep learning package built on top of Pytorch & fastai focused on state-of-the-art techniques for time series tasks like classification, regression, forecasting, imputation...

`tsai` is currently under active development by timeseriesAI.

## New in tsai:

* `tsai` is now available in conda. 
* 🚀🚀 **MINIROCKET** a SOTA Time Series Classification model (now available in Pytorch):
You can now check MiniRocket's performance in our new tutorial notebook [10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb](https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb)
> "Using this method, it is possible to train and test a classifier on all of 109 datasets from the UCR archive to state-of-the-art accuracy in less than 10 minutes." A. Dempster et al. (Dec 2020)

* **Multi-class and multi-label time series classification notebook:** you can also check our new tutorial notebook: [01a_MultiClass_MultiLabel_TSClassification.ipynb](https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/01a_MultiClass_MultiLabel_TSClassification.ipynb)
* **Self-supervised learning:**
If you are interested in applying self-supervised learning to time series, you may check our new tutorial notebook: [08_Self_Supervised_MVP.ipynb](https://github.com/timeseriesAI/tsai/blob/master/tutorial_nbs/08_Self_Supervised_TSBERT.ipynb)

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

## How to start using tsai?

To get to know the tsai package, we'd suggest you start with this notebook in Google Colab: **[01_Intro_to_Time_Series_Classification](https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/01_Intro_to_Time_Series_Classification.ipynb)**
It provides an overview of a time series classification task.

We have also develop many other [tutorial notebooks](https://github.com/timeseriesAI/tsai/tree/main/tutorial_nbs). 

To use tsai in your own notebooks, the only thing you need to do after you have installed the package is to run this:

`from tsai.all import *`

## Examples

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

We welcome contributions of all kinds. Development of features, bug fixes, and other improvements. 

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

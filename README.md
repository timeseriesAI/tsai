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

## What's new:

#### September, 2021
* See our new tutorial notebook on how to **track your experiments with Weights & Biases**
<a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/12_Experiment_tracking_with_W%26B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* `tsai` just got easier to use with the new sklearn-like APIs: `TSClassifier`, `TSRegressor`, and `TSForecaster`!! See [this](https://timeseriesai.github.io/tsai/tslearner.html) for more info.
* 🚀🚀 New tutorial notebook on how to **train your model with larger-than-memory datasets in less time achieving up to 100% GPU usage!!** 🚀🚀   <a href="https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/11_How_to_train_big_arrays_faster_with_tsai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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

- [LSTM](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py) (Hochreiter, 1997) ([paper](https://ieeexplore.ieee.org/abstract/document/6795963/))
- [GRU](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py) (Cho, 2014) ([paper](https://arxiv.org/abs/1412.3555))
- [MLP](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/MLP.py) - Multilayer Perceptron (Wang, 2016) ([paper](https://arxiv.org/abs/1611.06455))
- [FCN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/FCN.py) - Fully Convolutional Network (Wang, 2016) ([paper](https://arxiv.org/abs/1611.06455))
- [ResNet](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResNet.py) - Residual Network (Wang, 2016) ([paper](https://arxiv.org/abs/1611.06455))
- [LSTM-FCN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py) (Karim, 2017) ([paper](https://arxiv.org/abs/1709.05206))
- [GRU-FCN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py) (Elsayed, 2018) ([paper](https://arxiv.org/abs/1812.07683))
- [mWDN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/mWDN.py) - Multilevel wavelet decomposition network (Wang, 2018) ([paper](https://dl.acm.org/doi/abs/10.1145/3219819.3220060))
- [TCN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TCN.py) - Temporal Convolutional Network (Bai, 2018) ([paper](https://arxiv.org/abs/1803.01271))
- [MLSTM-FCN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py) - Multivariate LSTM-FCN (Karim, 2019) ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608019301200))
- [InceptionTime](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTime.py) (Fawaz, 2019) ([paper](https://arxiv.org/abs/1909.04939))
- [Rocket](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ROCKET.py) (Dempster, 2019) ([paper](https://arxiv.org/abs/1910.13051))
- [XceptionTime](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/XceptionTime.py) (Rahimian, 2019) ([paper](https://arxiv.org/abs/1911.03803))
- [ResCNN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResCNN.py) - 1D-ResCNN (Zou , 2019) ([paper](https://www.sciencedirect.com/science/article/pii/S0925231219311506))
- [TabModel](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TabModel.py) - modified from fastai's [TabularModel](https://docs.fast.ai/tabular.model.html#TabularModel)
- [OmniScale](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/OmniScaleCNN.py) - Omni-Scale 1D-CNN (Tang, 2020) ([paper](https://arxiv.org/abs/2002.10061))
- [TST](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py) - Time Series Transformer (Zerveas, 2020) ([paper](https://dl.acm.org/doi/abs/10.1145/3447548.3467401))
- [TabTransformer](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TabTransformer.py) (Huang, 2020) ([paper](https://arxiv.org/pdf/2012.06678))
- [XCM](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/XCM.py) - Explainable Convolutional Neural Network) (Fauvel, 2020) ([paper](https://arxiv.org/abs/2005.03645))
- [MiniRocket](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/MINIROCKET.py) (Dempster, 2021) ([paper](https://arxiv.org/abs/2102.00457))


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
batch_tfms = TSStandardize()
clf = TSClassifier(X, y, splits=splits, arch=InceptionTimePlus, batch_tfms=batch_tfms, metrics=accuracy, cbs=ShowGraph(), verbose=True)
clf.fit_one_cycle(100, 3e-4)
```

### Multi-class, multivariate classification

```
from tsai.all import *
X, y, splits = get_classification_data('LSST', split_data=False)
batch_tfms = TSStandardize(by_sample=True)
clf = TSClassifier(X, y, splits=splits, arch=InceptionTimePlus, batch_tfms=batch_tfms, metrics=accuracy, cbs=ShowGraph(), verbose=True)
clf.fit_one_cycle(10, 1e-2)
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
fcst = TSForecaster(X, y, splits=splits, batch_tfms=batch_tfms, bs=512, arch=TST, metrics=mae, cbs=ShowGraph())
fcst.fit_one_cycle(50, 1e-3)
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

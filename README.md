<div align="center">
    <img width="60%" src="./docs/images/tsai_logo.svg">
</div>

-----------------

# tsai
> State-of-the-art Deep Learning library for Time Series and Sequences. 


![CI](https://github.com/timeseriesai/tsai/workflows/CI/badge.svg) 
[![PyPI](https://img.shields.io/pypi/v/tsai?color=blue&label=pypi%20version)](https://pypi.org/project/tsai/#description)
[![Downloads](https://pepy.tech/badge/tsai)](https://pepy.tech/project/tsai)

`tsai` is an open-source deep learning package built on top of Pytorch & fastai focused on state-of-the-art techniques for time series tasks like classification, regression, forecasting, imputation...

## New in tsai:

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

It provides an overview of a time series classification problem using fastai v2.

If you want more details, you can get them in nbs 00 and 00a.

To use tsai in your own notebooks, the only thing you need to do after you have installed the package is to add this:

`from tsai.all import *`

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

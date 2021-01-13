<div align="center">
    <img width="60%" src="./docs/images/tsai_logo.svg">
</div>

-----------------

# tsai
> State-of-the-art Deep Learning for Time Series and Sequence Modeling. `tsai` is currently under active development by timeseriesAI.


![CI](https://github.com/timeseriesai/tsai/workflows/CI/badge.svg) 
[![PyPI](https://img.shields.io/pypi/v/tsai?color=blue&label=pypi%20version)](https://pypi.org/project/tsai/#description)
[![Downloads](https://pepy.tech/badge/tsai)](https://pepy.tech/project/tsai)

`tsai`is an open-source deep learning package built on top of Pytorch & fastai focused on state-of-the-art techniques for time series classification, regression and forecasting.

⚠️ If you are interested in applying self-supervised learning to time series, you may want to check our new tutorial notebook: [08_Self_Supervised_TSBERT.ipynb](https://github.com/timeseriesAI/tsai/blob/master/tutorial_nbs/08_Self_Supervised_TSBERT.ipynb)

Here's the link to the [documentation](https://timeseriesai.github.io/tsai/).

## Install

You can install the **latest stable** version from pip:

`pip install tsai`

Or you can install the **bleeding edge** version of this library from github by doing:

`pip install git+https://github.com/timeseriesAI/tsai.git@master`

## How to get started

To get to know the `tsai` package, I'd suggest you start with this notebook in Google Colab: **[01_Intro_to_Time_Series_Classification](https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/01_Intro_to_Time_Series_Classification.ipynb)**

It provides an overview of a time series classification problem using fastai v2.

If you want more details, you can get them in nbs 00 and 00a.

To use tsai in your own notebooks, the only thing you need to do after you have installed the package is to add this:

`from tsai.all import *`

## Citing tsai

If you use `tsai` in your research please use the following BibTeX entry:

```text
@Misc{tsai,
    author =       {Ignacio Oguiza},
    title =        {tsai - A state-of-the-art deep learning library for time series and sequential data},
    howpublished = {Github},
    year =         {2020},
    url =          {https://github.com/timeseriesAI/tsai}
}
```

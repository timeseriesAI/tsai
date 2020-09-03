# tsai
> Practical Deep Learning for Time Series / Sequential Data package built with fastai v2/ Pytorch.


`tsai`is a deep learning package built on top of fastai v2 / Pytorch focused on state-of-the-art methods for time series classification and regression.

If you are looking for **timeseriesAI based on fastai v1**, it's been moved to **[timeseriesAI1](https://github.com/timeseriesAI/timeseriesAI1)**.

## Install

You can install the **latest stable** version from pip:

`pip install tsai`

Or you can install the **bleeding edge** version of this library from github by doing:

`pip install git+https://github.com/timeseriesAI/timeseriesAI.git@master`

In the latter case, you may also want to use install the bleeding egde fastai & fastcore libraries, in which case you need to do this:

`pip install git+https://github.com/fastai/fastcore.git@master`

`pip install git+https://github.com/fastai/fastai.git@master`

## How to get started

To get to know the `tsai` package, I'd suggest you start with this notebook:

**[01_Intro_to_Time_Series_Classification](https://github.com/timeseriesAI/timeseriesAI/blob/master/tutorial_nbs/01_Intro_to_Time_Series_Classification.ipynb)**

It provides an overview of a time series classification problem using fastai v2.

If you want more details, you can get them in nbs 00 and 00a.

To use tsai in your own notebooks, the only thing you need to do after you have installed the package is to add this:

`from tsai.all import *`

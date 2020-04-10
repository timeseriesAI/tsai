# tsai
> Practical Deep Learning for Time Series / Sequential Data library based on fastai v2/ Pytorch.


`tsai`is a deep learning library built on top of fastai v2 / Pytorch focused on state-of-the-art methods for time series classification and regression.

## Install

You can install the **latest stable** version from pip:

`pip install tsa`

Or you can install the **bleeding edge** version of this library from github by doing:

`pip install git+https://github.com/timeseriesAI/timeseriesAI.git@master`

In the latter case, you may also want to use install the bleeding egde fastai & fastcore libraries, in which case you need to do this:

`pip install git+https://github.com/fastai/fastcore.git@master`

`pip install git+https://github.com/fastai/fastai2.git@master`

## How to use

The only thing you need to do after you have installed the library is to add this to your notebook:

`from tsai.all import *`

To get familiarized with the library, I'd suggest you start with this notebook:

[01_Intro_to_Time_Series_Classification](https://github.com/timeseriesAI/timeseriesAI/blob/master/tutorial_nbs/01_Intro_to_Time_Series_Classification.ipynb)

It provides an overview of a time series classification problem using fastai v2.

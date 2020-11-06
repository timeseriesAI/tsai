# tsai
> Practical Deep Learning for Time Series / Sequential Data package built with fastai v2/ Pytorch.


`tsai`is a deep learning package built on top of fastai v2 / Pytorch focused on state-of-the-art methods for time series classification and regression.

If you are looking for **timeseriesAI based on fastai v1**, it's been moved to **[timeseriesAI1](https://github.com/timeseriesAI/timeseriesAI1)**.

## What's new?

### tsai: 2.0 (Nov 3rd, 2020)

`tsai` 2.0 is a major update to the tsai library. These are the major changes made to the library:

* New tutorial nbs have been added to demonstrate the use of new functionality like: 
    * Time series data preparation
    * Intro to time series regression
    * TS archs comparison
    * TS to image classification
    * TS classification with transformers
* Also some tutorial nbs have been updated like Time Series transforms
* More ts data transforms have been added, including ts to images.
* New callbacks, like the state of the art noisy_student that will allow you to use unlabeled data.
* New time series, state-of-the-art models are now available like XceptionTime, RNN_FCN (like LSTM_FCN, GRU_FCN), TransformerModel, TST (Transformer), OmniScaleCNN, mWDN (multi-wavelet decomposition network), XResNet1d.
* Some of the models (those finishing with an plus) have additional, experimental functionality (like coordconv, zero_norm, squeeze and excitation, etc).

The best way to discocer and understand how to use this new functionality is to use the tutorial nbs. I encourage you to use them!

## Install

You can install the **latest stable** version from pip:

`pip install tsai`

Or you can install the **bleeding edge** version of this library from github by doing:

`pip install git+https://github.com/timeseriesAI/timeseriesAI.git@master`

## How to get started

To get to know the `tsai` package, I'd suggest you start with this notebook:

**[01_Intro_to_Time_Series_Classification](https://github.com/timeseriesAI/timeseriesAI/blob/master/tutorial_nbs/01_Intro_to_Time_Series_Classification.ipynb)**

It provides an overview of a time series classification problem using fastai v2.

If you want more details, you can get them in nbs 00 and 00a.

To use tsai in your own notebooks, the only thing you need to do after you have installed the package is to add this:

`from tsai.all import *`

# timeseriesAI


timeseriesAI is a library built on top of fastai/ Pytorch to help you apply Deep Learning to your time series/ sequential datasets, in particular Time Series Classification (TSC) and Time Series Regression (TSR) problems.


The library contains 3 major components: 

1. **Notebooks** 📒: they are very practical, and show you how certain techniques can be easily applied. 

2. **fastai_timeseries** 🏃🏽‍♀️: it's an extension of fastai's library that focuses on time series/ sequential problems. 

3. **torchtimeseries.models** 👫: it's a collection of some state-of-the-art time series/ sequential models.


The 3 components of this library will keep growing in the future as new techniques are added and/or new state-of-the-art models appear. In those cases, I will keep adding notebooks to demonstrate how you can apply them in a practical way.


## Notebooks

#### 1. Introduction to Time Series Classification (TSC) 🤝: 
- This is an intro that nb that shows you how you can achieve high performance in 4 simple steps.

#### 2. UCR_TCS 🧪:
- The UCR datasets are broadly used in TSC problems as s bechmark to measure performance. This notebook will allow you to test any of the available datasets, with the model of your choice and any training scheme. You can easily tweak any of them to try to beat a SOTA.

#### 3. New TS data augmentations 🔎: 
- You will see how you can apply successful data augmentation techniques (like mixup, cutout, and cutmix) to time series problems.

#### 4. The importance of scaling ⚖️: 
- In this notebook you'll learn more about the options to scale your data and the impact it may have on performance, which is huge!

#### 5. Multivariate ROCKET on GPU 🚀: 
- On October 29, 2019, there was a major milestone in the area of Time Series Classification. 
A new method, called ROCKET (RandOm Convolutional KErnel Transform) was released (Dempster A, Petitjean F, Webb GI (2019) [ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels](https://arxiv.org/pdf/1910.13051)) together with the code they used.
This new method not only beat the previous recognized state of the art (HIVE-COTE) on a TSC benchmark, but it does it in record time, many orders of magnitude faster than any other method. 
I’ve been using it for a couple of days and the results are IMPRESSIVE!! 
The release code however has 2 limitations:
  - it can only handle univariate time series
  - it doesn’t support GPU

- I have developed ROCKET with GPU support in Pytorch that you can now use it with univariate of multivariate time series. In this notebook you will see how you can use ROCKET in your time series problems.

#### 6. TS data augmentations: single-item transforms 👯 - UPDATED with new tfms, GPU tfms and RandAugment
- In this notebook, you'll find some TS transforms that can be used to augment your data. Most of the transforms are adapted from/ inspired by research papers on time series augmentation. Transforms include:  TSjittering, TSmagscale, TSmagwarp, TStimenoise, TStimewarp, TSlookback, TStimestepsout, TSchannelout, TScutout, TScrop, TSwindowslice and TSzoom.

- UPDATED: I have now updated all tfms so that we can use them as regular tfms, or pass them to a DataLoader and use them as a batch transform (which is much faster). I have also included RandAugment (a new technique developed by Google to eliminate the data augmentation search phase). This applies a random data augmentation to each individual batch.


# timeseriesAI


timeseriesAI is a library built on top of fastai/ Pytorch to help you apply Deep Learning to your time series/ sequential datasets, in particular Time Series Classification (TSC) and Time Series Regression (TSR) problems.


The library contains 3 major components: 

1. **Notebooks** 📒: they are very practical, and show you how certain techniques can be easily applied. 

2. **fastai_timeseries** 🏃🏽‍♀️: it's an extension of fastai's library that focuses on time series/ sequential problems. 

3. **torchtimeseries.models** 👫: it's a collection of some state-of-the-art time series/ sequential models.


The 3 components of this library will keep growing in the future as new techniques are added and/or new state-of-the-art models appear. In those cases, I will keep adding notebooks to demonstrate how you can apply them in a practical way.


## Notebooks

#### 1. Introduction to Time Series Classification (TSC): 
- This is an intro that nb that shows you how you can achieve high performance in 4 simple steps.

#### 2. UCR_TCS:
- The UCR datasets are broadly used in TSC problems as s bechmark to measure performance. This notebook will allow you to test any of the available datasets, with the model of your choice and any training scheme. You can easily tweak any of them to try to beat a SOTA.

#### 3. New TS data augmentations: 
- You will see how you can apply successful data augmentation techniques (like mixup, cutout, and cutmix) to time series problems.


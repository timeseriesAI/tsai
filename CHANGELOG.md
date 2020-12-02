# Release notes

<!-- do not remove -->

## 0.2.8

### New Features

* Data: 
    * data preparation: a new `SlidingWindowPanel` function has been added to help prepare the input from panel data. `SlidingWindow` has also been enhanced.
    * new preprocessors: TSRobustScaler, TSClipOutliers, TSDiff, TSLog, TSLogReturn
* Models: 
    * `MLP` and `TCN` (Temporal Convolutional Network) have been added.
* Training:
    * Callback: Uncertainty-based data augmentation
    * Label-mixing transforms (data augmentation): MixUp1D, CutMix1D callbacks
* Utility functions: build_ts_model, build_tabular_model, get_ts_dls, get_tabular_dls, ts_learner


## 0.2.4

### New Features

* Added support to Pytorch 1.7.


## 0.2.0

`tsai` 0.2.0 is a major update to the tsai library. These are the major changes made to the library:

* New tutorial nbs have been added to demonstrate the use of new functionality like: 
    * Time series **data preparation**
    * Intro to **time series regression**
    * TS archs comparison
    * **TS to image** classification
    * TS classification with **transformers**
    
### New Features
* More ts data transforms have been added, including ts to images.
* New callbacks, like the state of the art noisy_student that will allow you to use unlabeled data.
* New time series, state-of-the-art models are now available like XceptionTime, RNN_FCN (like LSTM_FCN, GRU_FCN), TransformerModel, TST (Transformer), OmniScaleCNN, mWDN (multi-wavelet decomposition network), XResNet1d.
* Some of the models (those finishing with an plus) have additional, experimental functionality (like coordconv, zero_norm, squeeze and excitation, etc).
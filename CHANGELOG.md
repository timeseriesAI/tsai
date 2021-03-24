# Release notes

<!-- do not remove -->

## 0.2.17

### Bug Fixes
* Models: 
    * Fixed an issue in TST and TSTPlus related to encoder layer creation.

## New features:
* Models:
    * Added TabTransformer, a state-of-the-art tabular transformer released in Dec 2020.

### Breaking change:
* Data:
    * train_perc in get_splits has been changed to train_size to allow both floats or integers.

### Updates
* Learner:
    * Updated 3 new learner APIs: TSClassifier, TSRegressor, TSForecaster. 
    
* ShowGraph callback:
    * Callback optionally plots all metrics at the end of training.


## 0.2.16

### Bug Fixes
* Data: 
    * Updated df2xy function to fix a bug.

### Updates
* Tutorial notebooks:
    * Updated 04 (regression) to use the recently released Monash, UEA & UCR Time Series Extrinsic Regression Repository (2020).
    
## New features:
* Models:
    * Added new pooling layers and 3 new heads: attentional_pool_head, universal_pool_head, gwa_pool_head
    
    
    
## 0.2.15

### New Features
* General:
    * Added 3 new sklearn-type APIs: TSClassifier, TSRegressor and TSForecaster.
    
* Data:
    * External: added a new function get_forecasting_data to access some forecasting datasets.
    * Modified TimeSplitter to also allow passing testing_size.
    * Utilities: add a simple function (standardize) to scale any data using splits.
    * Preprocessing: added a new class (Preprocess) to be able to preprocess data before creating the datasets/ dataloaders. This is mainly to test different target preprocessing techniques.
    * Utils added Nan2Value batch transform to remove any nan values in the dataset.
    * Added a new utility function to easy the creation of a single TSDataLoader when no splits are used (for example with unlabeled datasets).
    * Added a new function to quickly create empty arrays on disk or in memory (create_empty_array). 
    
* Models: 
    * TST: Added option to visualize self-attention maps. 
    * Added 3 new SOTA models: MiniRocketClassifier and MiniRocketRegressor for datasets <10k samples, and MiniRocket (Pytorch) which supports any dataset size. 
    * Added a simple function to create a naive forecast.
    * Added future_mask to TSBERT to be able to train forecasting models. 
    * Added option to pass any custom mask to TSBERT.
    
* Training:
    * PredictionDynamics callback: allows you to visualize predictions during training.
    
* Tutorial notebooks: 
    * New notebook demonstrating the new PredictionDynamics callback.
    
### Bug Fixes
* Models: 
    * Fixed bug that prevented models to freeze or unfreeze. Now all models that end with Plus can take predefined weights and learn.freeze()/ learn.unfreeze() will work as expected.

## 0.2.14

### New Features
* Data:
    * External: added a new function get_Monash_data to get extrinsic regression data.
* Models: 
    * Added show_batch functionality to TSBERT. 
    

## 0.2.13

### New Features
* General: Added min requirements for all package dependencies.
* Data:
    * Validation: added split visualization (show_plot=True by default).
    * Data preprocessing: add option to TSStandardize or TSNormalize by_step.
    * Featurize time series: added tsfresh library to allow the creation of features from time series.     
* Models: 
    * Updated ROCKET to speed up feature creation and allow usage of large datasets.
    * Added change_model_head utility function to ease the process of changing an instantiated models head.
    * conv_lin_3d_head function to allow generation of 3d output tensors. This may be useful for multivariate, multi-horizon direct (non-recursive) time series forecasting, multi-output regression tasks, etc.
    * Updated TST (Time series transformer) to allow the use of residual attention (based on He, R., Ravula, A., Kanagal, B., & Ainslie, J. (2020). Realformer: Transformer Likes Informed Attention. arXiv preprint arXiv:2012.11747.)
    * provided new functionality to transfer model's weights (useful when using pre-trained models). 
    * updated build_ts_model to be able to use pretrained model weights.
* Training:
    * TSBERT: a new callback has been added to be able to train a model in a self-supervised manner (similar to BERT).
* Tutorial notebooks: 
    * I've added a new tutorial notebook to demonstrate how to apply TSBERT (self-supervised method for time series).

### Bug Fixes
* Data: 
    * ROCKET: fixed a bug in `create_rocket_features`.

    

## 0.2.12

### New Features

* Data: 
    * core: `get_subset_dl` and `get_subset_dls`convenience function have been added.
    * data preparation: `SlidingWindow` and `SlidingWindowPanel` functions are now vectorized, and are at least an order of magnitude faster. 
* Models: 
    * `XCM`: An Explainable Convolutional Neural Network for Multivariate Time Series Classification have been added. Official code not released yet. This is a stete-of-the-art time series model that combines Conv1d and Conv2d and has good explainability.
* Training:
    * learner: `ts_learner` and `tsimage_learner` convenience functions have been added, as well as a `get_X_preds` methods to facilitate the generation of predictions.


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
# Release notes

<!-- do not remove -->

## 0.3.0

### New Features

- Added function that pads sequences to same length ([#410](https://github.com/timeseriesAI/tsai/issues/410))

- Added TSRandomStandardize preprocessing technique ([#396](https://github.com/timeseriesAI/tsai/issues/396))

- New visualization techniques: model's feature importance and step importance ([#393](https://github.com/timeseriesAI/tsai/issues/393))

- Allow from tsai.basics import * to speed up loading ([#320](https://github.com/timeseriesAI/tsai/issues/320))


### Bugs Squashed

- Separate core from non-core dependencies in tsai - pip install tsai[extras]([#389](https://github.com/timeseriesAI/tsai/issues/318)). This is an important change that:
   - reduces the time to ```pip install tsai``` 
   - avoid errors during installation
   - reduces the time to load tsai using ```from tsai.all import *```



## 0.2.25
### Breaking Changes

- updated forward_gaps removing nan_to_num ([#331](https://github.com/timeseriesAI/tsai/issues/331))

- TSRobustScaler only applied by_var ([#329](https://github.com/timeseriesAI/tsai/issues/329))

- remove add_na arg from TSCategorize ([#327](https://github.com/timeseriesAI/tsai/issues/327))

### New Features

- added IntraClassCutMix1d ([#384](https://github.com/timeseriesAI/tsai/issues/384))

- added learn.calibrate_model method ([#379](https://github.com/timeseriesAI/tsai/issues/379))

- added analyze_array function ([#378](https://github.com/timeseriesAI/tsai/issues/378))

- Added TSAddNan transform ([#376](https://github.com/timeseriesAI/tsai/issues/376))

- added dummify function to create dummy data from original data ([#366](https://github.com/timeseriesAI/tsai/issues/366))

- added Locality Self Attention to TSiT ([#363](https://github.com/timeseriesAI/tsai/issues/363))

- added sel_vars argument to MVP callback ([#349](https://github.com/timeseriesAI/tsai/issues/349))

- added sel_vars argument to TSNan2Value ([#348](https://github.com/timeseriesAI/tsai/issues/348))

- added multiclass, weighted FocalLoss ([#346](https://github.com/timeseriesAI/tsai/issues/346))

- added TSRollingMean batch transform ([#343](https://github.com/timeseriesAI/tsai/issues/343))

- added recall_at_specificity metric ([#342](https://github.com/timeseriesAI/tsai/issues/342))

- added train_metrics argument to ts_learner ([#341](https://github.com/timeseriesAI/tsai/issues/341))

- added hist to PredictionDynamics for binary classification ([#339](https://github.com/timeseriesAI/tsai/issues/339))

- add padding_idxs to MultiEmbedding ([#330](https://github.com/timeseriesAI/tsai/issues/330))

### Bugs Squashed

- sort_by data may be duplicated in SlidingWindowPanel ([#389](https://github.com/timeseriesAI/tsai/issues/389))

- create_script splits the nb name if multiple underscores are used ([#385](https://github.com/timeseriesAI/tsai/issues/385))

- added torch functional dependency to plot_calibration_curve ([#383](https://github.com/timeseriesAI/tsai/issues/383))

- issue when setting horizon to 0 in SlidingWindow ([#382](https://github.com/timeseriesAI/tsai/issues/382))

- replace learn by self in calibrate_model patch ([#381](https://github.com/timeseriesAI/tsai/issues/381))

- Argument `d_head` is not used in TSiTPlus ([#380](https://github.com/timeseriesAI/tsai/issues/380))
  - https://github.com/timeseriesAI/tsai/blob/6baf0ba2455895b57b54bf06744633b81cdcb2b3/tsai/models/TSiTPlus.py#L63

- replace default relu activation by gelu in TSiT ([#361](https://github.com/timeseriesAI/tsai/issues/361))

- sel_vars and sel_steps in TSDatasets and TSDalaloaders don't work when used simultaneously ([#347](https://github.com/timeseriesAI/tsai/issues/347))

- ShowGraph fails when recoder.train_metrics=True ([#340](https://github.com/timeseriesAI/tsai/issues/340))

- fixed 'se' always equal to 16 in MLSTM_FCN ([#337](https://github.com/timeseriesAI/tsai/issues/337))

- ShowGraph doesn't work well when train_metrics=True ([#336](https://github.com/timeseriesAI/tsai/issues/336))

- TSPositionGaps doesn't work on cuda ([#333](https://github.com/timeseriesAI/tsai/issues/333))

- XResNet object has no attribute 'backbone' ([#332](https://github.com/timeseriesAI/tsai/issues/332))

- import InceptionTimePlus in tsai.learner ([#328](https://github.com/timeseriesAI/tsai/issues/328))

- df2Xy: Format correctly without the need to specify sort_by ([#324](https://github.com/timeseriesAI/tsai/issues/324))

- bug in MVP code learn.model --> self.learn.model ([#323](https://github.com/timeseriesAI/tsai/issues/323))

- Colab install issues: importing the lib takes forever ([#315](https://github.com/timeseriesAI/tsai/issues/315))

- Calling learner.feature_importance on larger than memory dataset causes OOM ([#310](https://github.com/timeseriesAI/tsai/issues/310))


## 0.2.24
### Breaking Changes

- removed InceptionTSiT, InceptionTSiTPlus, ConvTSiT & ConvTSiTPlus ([#276](https://github.com/timeseriesAI/tsai/issues/276))

### New Features

- add stateful custom sklearn API type tfms: TSShrinkDataFrame, TSOneHotEncoder, TSCategoricalEncoder ([#313](https://github.com/timeseriesAI/tsai/issues/313))

- Pytorch 1.10 compatibility ([#311](https://github.com/timeseriesAI/tsai/issues/311))

- ability to pad at the start/ end of sequences and filter results in SlidingWindow ([#307](https://github.com/timeseriesAI/tsai/issues/307))

- added bias_init to TSiT ([#288](https://github.com/timeseriesAI/tsai/issues/288))

- plot permutation feature importance after a model's been trained ([#286](https://github.com/timeseriesAI/tsai/issues/286))

- added separable as an option to MultiConv1d ([#285](https://github.com/timeseriesAI/tsai/issues/285))

- Modified TSiTPlus to accept a feature extractor and/or categorical variables ([#278](https://github.com/timeseriesAI/tsai/issues/278))

### Bugs Squashed

- learn modules takes too long to load ([#312](https://github.com/timeseriesAI/tsai/issues/312))

- error in roll2d and roll3d when passing index 2 ([#304](https://github.com/timeseriesAI/tsai/issues/304))

- TypeError: unhashable type: 'numpy.ndarray' ([#302](https://github.com/timeseriesAI/tsai/issues/302))

- ValueError: only one element tensors can be converted to Python scalars ([#300](https://github.com/timeseriesAI/tsai/issues/300))

- unhashable type: 'numpy.ndarray' when using multiclass multistep labels ([#298](https://github.com/timeseriesAI/tsai/issues/298))

- incorrect data types in NumpyDatasets subset ([#297](https://github.com/timeseriesAI/tsai/issues/297))

- create_future_mask creates a mask in the past ([#293](https://github.com/timeseriesAI/tsai/issues/293))

- NameError: name 'X' is not defined in learner.feature_importance ([#291](https://github.com/timeseriesAI/tsai/issues/291))

- TSiT test fails on cuda ([#287](https://github.com/timeseriesAI/tsai/issues/287))

- MultiConv1d breaks when ni == nf ([#284](https://github.com/timeseriesAI/tsai/issues/284))

- WeightedPerSampleLoss reported an error when used with LDS_weights ([#281](https://github.com/timeseriesAI/tsai/issues/281))

- pos_encoding transfer weight in TSiT fails ([#280](https://github.com/timeseriesAI/tsai/issues/280))

- MultiEmbedding cat_pos and cont_pos are not in state_dict() ([#277](https://github.com/timeseriesAI/tsai/issues/277))

- fixed issue with MixedDataLoader  ([#229](https://github.com/timeseriesAI/tsai/pull/229)), thanks to [@Wabinab](https://github.com/Wabinab)



## 0.2.23
### Breaking Changes

- removed torch-optimizer dependency ([#228](https://github.com/timeseriesAI/tsai/issues/228))

### New Features

- added option to train MVP on random sequence lengths ([#252](https://github.com/timeseriesAI/tsai/issues/252))

- added ability to pass an arch name (str) to learner instead of class ([#217](https://github.com/timeseriesAI/tsai/issues/217))

- created convenience fns create_directory and delete_directory in utils ([#213](https://github.com/timeseriesAI/tsai/issues/213))

- added option to create random array of given shapes and dtypes ([#212](https://github.com/timeseriesAI/tsai/issues/212))

- my_setup() print your main system and package versions ([#202](https://github.com/timeseriesAI/tsai/issues/202))

- added a new tutorial on how to train large datasets using tsai ([#199](https://github.com/timeseriesAI/tsai/issues/199))

- added a new function to load any file as a module ([#196](https://github.com/timeseriesAI/tsai/issues/196))

### Bugs Squashed

- Loading code just for inference takes too long ([#273](https://github.com/timeseriesAI/tsai/issues/273))

- Fixed out-of-memory issue with large datasets on disk ([#126](https://github.com/timeseriesAI/tsai/issues/126))

- AttributeError: module 'torch' has no attribute 'nan_to_num' ([#262](https://github.com/timeseriesAI/tsai/issues/262))

- Fixed TypeError: unhashable type: 'numpy.ndarray' ([#250](https://github.com/timeseriesAI/tsai/issues/250))

- Wrong link in paper references ([#249](https://github.com/timeseriesAI/tsai/issues/249))

- remove default PATH which overwrites custom PATH ([#238](https://github.com/timeseriesAI/tsai/issues/238))

- Predictions where not properly decoded when using with_decoded. ([#237](https://github.com/timeseriesAI/tsai/issues/237))

- SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame ([#221](https://github.com/timeseriesAI/tsai/issues/221))

- InceptionTimePlus wasn't imported by TSLearners ([#218](https://github.com/timeseriesAI/tsai/issues/218))

- get_subset_dl fn is not properly creating a subset dataloader ([#211](https://github.com/timeseriesAI/tsai/issues/211))

- Bug in WeightedPersSampleLoss ([#203](https://github.com/timeseriesAI/tsai/issues/203))


## 0.2.19

### New Features

- implemented src_key_padding_mask in TST & TSTPlus ([#79](https://github.com/timeseriesAI/tsai/issues/79))

### Bugs Squashed

- Problem with get_minirocket_features while using CUDA in training ([#153](https://github.com/timeseriesAI/tsai/issues/153))


## 0.2.19

### New Features
* Models: 
    - implement src_key_padding_mask in TST & TSTPlus ([#79](https://github.com/timeseriesAI/tsai/issues/79))

### Bugs Squashed
* Models:
    - Problem with get_minirocket_features while using CUDA in training ([#153](https://github.com/timeseriesAI/tsai/issues/153))


## 0.2.18

## New features:
* Data:
    * Update TSStandardize to accept some variables and/or groups of variables when using by_var.
    * added option to pad labeled and unlabed datasets with SlidingWindow with a padding value
    * added split_idxs and idxs to mixed_dls
    * added sklearn preprocessing tfms
    * added functions to measure sequence gaps
    * added decodes to TSStandardize
* Callbacks:
    * change mask return values in MVP to True then mask
    * updated MVP to accept nan values
* Models:
    * updated mWDN to take either model or arch
    * added padding_var to TST
    * added MiniRocketFeatures in Pytorch
* Losses & metrics:
    * added WeightedPerSampleLoss
    * added mean_per_class_accuracy to metrics
    * added mape metric
    * added HuberLoss and LogCoshLoss
* Learner:
    * added Learner.remove_all_cbs
    * updated get_X_preds to work with multilabel datasets
* Miscellaneous:
    * added rotate_axis utility functions
    
### Bug Fixes:   
* Callbacks:
    * fixed and issue with inconsistency in show_preds in MVP
* Models: 
    * Fixed an issue in InceptionTimePlus with stochastic depth regularization (stoch_depth parameter)
    * Fixed issue with get_X_preds (different predictions when executed multiple times)
    * fixed stoch_depth issue in InceptionTimePlus
    * fixed kwargs issue in MultiInceptionTimePlus
* Data:
    * fixed issue in delta gap normalize
* Learner:
    * fixed bug in get_X_preds device
    * updated get_X_preds to decode classification and regression outputs


## 0.2.17

### Bug Fixes:
* Models: 
    * Fixed an issue in TST and TSTPlus related to encoder layer creation.
    * Fixed issue in TSStandardize when passing tensor with nan values

## New features:
* Models:
    * Added TabTransformer, a state-of-the-art tabular transformer released in Dec 2020.
    * TSTPlus now supports padding masks (passed as nan values) by default.
* Data:
    * Added a Nan2Value batch transform that removes any nan value in the tensor by zero or median.
    * Faster dataloader when suffle == True.
    * Added TSUndindowedDataset and TSUnwindowedDatasets, which apply window slicing online to prepare time series data. 
    * Added TSMetaDataset and TSMetaDatasets, which allow you to use one or multiple X (and y) arrays as input. In this way, you won't need to merge all data into a single array. This will allow you to work with larger than memory datasets. 
    * Added a new tutorial notebook that demonstrates both multi-class and multi-label classification using tsai.
    * Upgraded df2Xy to accept y_func that allows calculation of different types of targets
* Callbacks: 
    * MVP is now much faster as masks are now created directly as cuda tensors. This has increased speed by 2.5x in some tests.

### Breaking changes:
* Data:
    * train_perc in get_splits has been changed to train_size to allow both floats or integers.
    * df2Xy API has been modified

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

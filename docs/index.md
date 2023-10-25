# flexcv
__Flexible Cross Validation and Machine Learning for Tabular Data__

`flexcv` is a Python package that implements flexible cross validation and machine learning for tabular data. It provides a range of features for comparing machine learning models on different datasets with different sets of predictors, customizing just about everything around cross validations. It supports both fixed and random effects, as well as random slopes.

## Features

The `flexcv` package provides the following features:

1. Cross-validation of model performance using different cross-validation splits that are dependent or independent of the clustering structures in your data.
2. Fair selection of model parameters without data leakage using an inner cross-validation loop and a state-of-the-art optimization package. We use the `optuna` package for this purpose.
3. Customization of objective functions for optimization to select meaningful model parameters.
4. Scaling of inner and outer cross-validation loops separately.
5. Easy usage of the state-of-the-art logging dashboard `neptune` to track all of your experiments.
6. Easy local summary of all evaluation metrics in a single table.

## Contributions

We welcome contributions to this repository. If you have any questions, please contact us at rosenthal.fabian@gmail.com.
# flexcv

__Flexible Cross Validation and Machine Learning for Regression on Tabular Data__

[Repository](https://github.com/radlfabs/flexcv)

`flexcv` is a Python package that implements flexible cross validation and machine learning for tabular data. It provides a range of features for comparing machine learning models on different datasets with different sets of predictors, customizing just about everything around cross validations. It supports both fixed and random effects, as well as random slopes.

Since we ware in a very early phase and currently preparing the package release, this documentation is under construction and is currently being edited.

## Features

The `flexcv` package provides the following features:

1. Cross-validation of model performance using different cross-validation splits that are dependent or independent of the clustering structures in your data.
2. Selection of model parameters fairly without data leakage using an inner cross-validation loop and a state-of-the-art optimization package. We use the `optuna` package for this purpose.
3. Customization of objective functions for optimization to select meaningful model parameters.
4. Scaling of inner and outer cross-validation loops separately.
5. Include random effects and random slopes in your models. We use the `merf` package for this purpose
6. Easy usage of the state-of-the-art logging dashboard `neptune` to track all of your experiments.
7. Easy local summary of all evaluation metrics in a single table.
8. Wrapper classes for the R `earth` package to use the powerful regression splines in Python.
9. Wrapper classes for the `statsmodels` package to use their mixed effects models inside of a `sklearn` Pipeline.
10. Inner cross validation implementation that let's you push groups to the inner split in order to apply GroupKFold.

## Contributions

We welcome contributions to this repository. If you have any questions, please don't hesitate to get in contact.

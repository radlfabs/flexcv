<table>
  <tr>
    <td><img src="images/logo_colored.png" width="200"></td>
    <td><h1>Flexible Cross Validation and Machine Learning for Regression on Tabular Data
</h1></td>
  </tr>
</table>


Find the repository [here](https://github.com/radlfabs/flexcv).

`flexcv` is a Python package that implements flexible cross validation and machine learning for tabular data. It provides a range of features for comparing machine learning models on different datasets with different sets of predictors, customizing just about everything around cross validations. It supports both fixed and random effects, as well as random slopes.

Since we ware in a very early phase and currently preparing the package release, this documentation is under construction and is currently being edited.

## Features

The `flexcv` package provides the following features:

1. Cross-validation of model performance (generalization estimation)
2. Selection of model hyperparameters using an inner cross-validation and a state-of-the-art optimization provided by `optuna`.
3. Customization of objective functions for optimization to select meaningful model parameters.
4. Fixed and mixed effects modeling (random intercepts and slopes).
5. Scaling of inner and outer cross-validation folds separately.
6. Easy usage of the state-of-the-art MLops platform `neptune` to track all of your experiments. Have a look at their [website](https://neptune.ai/) or explore our [neptune project](https://ui.neptune.ai/radlfabs/flexcv-testing) that we used for testing this package. Also check out the [neptune integration guide](guides/neptune-integration.md).
7. Integrates the `merf` package to apply correction for clustered data using the expectation maximization algorithm and supporting any `sklearn` BaseEstimator. Read more about that package [here](https://github.com/manifoldai/merf).
8. Adaptations for cross validation splits with stratification for continuous target variables.
9. Easy local summary of all evaluation metrics in a single table.
10. Wrapper classes for the `statsmodels` package to use their mixed effects models inside of a `sklearn` Pipeline. Read more about that package [here](https://github.com/manifoldai/merf).
11. Inner cross validation implementation that let's you push groups to the inner split, e. g. to apply GroupKFold.
12. Customizable ObjectiveScorer function for hyperparameter tuning, that let's you make a trade-off between under- and overfitting.

These are the core packages used under the hood in `flexcv`:

1. `sklearn` - A very popular machine learning library. We use their Estimator API for models, the pipeline module, the StandardScaler, metrics and of course wrap around their cross validation split methods. Learn more [here](https://scikit-learn.org/stable/).
2. `Optuna` - A state-of-the-art optimization package. We use it for parameter selection in the inner loop of our nested cross validation. Learn more about theoretical background and opportunities [here](https://optuna.org/).
3. `neptune` - Awesome logging dashboard with lots of integrations. It is a charm in combination with `Optuna`. We used it to track all of our experiments. `Neptune` is quite deeply integrated into `flexcv`. Learn more about this great library [here](https://neptune.ai/).
4. `merf` - Mixed Effects for Random Forests. Applies correction terms on the predictions of clustered data. Works not only with random forest but with every `sklearn` BaseEstimator.

## Why would you use `flexcv`?

Working with cross validation in Python usually starts with creating a sklearn pipeline. Pipelines are super useful to combine preprocessing steps with model fitting and prevent data leakage. 
However, there are limitations, e. g. if you want to push the training part of your clustering variable to the inner cross validation split. For some of the features, you would have to write a lot of boilerplate code to get it working, and you end up with a lot of code duplication.
As soon as you want to use a linear mixed effects model, you have to use the `statsmodels` package, which is not compatible with the `sklearn` pipeline.
`flexcv` solves these problems and provides a lot of useful features for cross validation and machine learning on tabular data, so you can focus on your data and your models.

## Earth Extension

An wrapper implementation of the Earth Regression package for R exists which you can use with flexcv. It is called [flexcv-earth](https:github.com/radlfabs/flexcv-earth). It is not yet available on PyPI, but you can install it from GitHub with the command `pip install git+https://github.com/radlfabs/flexcv-earth.git`. You can then use the `EarthModel` class in your `flexcv` configuration by importing it from `flexcv_earth`. Further information is available in the [documentation](https://radlfabs.github.io/flexcv-earth/).

## Contributions

We welcome contributions to this repository. If you have any questions, please don't hesitate to get in contact by reaching out or filing a [github issue](https://github.com/radlfabs/flexcv/issues).

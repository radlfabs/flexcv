<table>
  <tr>
    <td><img src="images/logo_colored.png" width="200"></td>
    <td><h1>Flexible Cross Validation and Machine Learning for Regression on Tabular Data
</h1></td>
  </tr>
</table>

Find the repository [here](https://github.com/radlfabs/flexcv) or the documentation [here](https://radlfabs.github.io/flexcv/). Quickly get started with `flexcv` by reading our notes on [installation](start/getting-started.md) or take a quick [tutorial](start/tutorial.md) to get familiar with the package. If you want to dive deeper into the package, read our [guides](user-guide.md) or take a look at the [reference](reference.md).

`flexcv` is a Python package that implements flexible cross validation and machine learning for tabular data. It provides a range of features for comparing machine learning models on different datasets with different sets of predictors, customizing just about everything around cross validations. It supports both fixed and random effects, as well as random slopes.

Since we ware in a very early phase and currently preparing the package release, this documentation is under construction and is currently being edited.

## Features

The `flexcv` package provides the following features:

1. Cross-validation of model performance (generalization estimation)
2. Selection of model hyperparameters using an inner cross-validation and a state-of-the-art optimization provided by `optuna`.
3. Customization of objective functions for optimization to select meaningful model parameters.
4. Fixed and mixed effects modeling (random intercepts and slopes).
5. Scaling of inner and outer cross-validation folds separately.
6. Easy usage of the state-of-the-art logging dashboard `neptune` to track all of your experiments.
7. Adaptations for cross validation splits with stratification for continuous target variables.
8. Easy local summary of all evaluation metrics in a single table.
9. Wrapper classes for the R `earth` package to use the powerful regression splines in Python. Read more about that package [here](https://www.rdocumentation.org/packages/earth/versions/5.3.2).
10. Wrapper classes for the `statsmodels` package to use their mixed effects models inside of a `sklearn` Pipeline. Read more about that package [here](https://github.com/manifoldai/merf).
11. Uses the `merf` package to apply correction for clustered data using the expectation maximization algorithm and supporting any `sklearn` BaseEstimator. Read more about that package [here](https://github.com/manifoldai/merf).
12. Inner cross validation implementation that let's you push groups to the inner split, e. g. to apply GroupKFold.
13. Customizable ObjectiveScorer function for hyperparameter tuning, that let's you make a trade-off between under- and overfitting.

These are the core packages used under the hood in `flexcv`:

1. `sklearn` - A very popular machine learning library. We use their Estimator API for models, the pipeline module, the StandardScaler, metrics and of course wrap around their cross validation split methods. Learn more [here](https://scikit-learn.org/stable/).
2. `Optuna` - A state-of-the-art optimization package. We use it for parameter selection in the inner loop of our nested cross validation. Learn more about theoretical background and opportunities [here](https://optuna.org/).
3. `neptune` - Awesome logging dashboard with lots of integrations. It is a charm in combination with `Optuna`. We used it to track all of our experiments. `Neptune` is quite deeply integrated into `flexcv`. Learn more about this great library [here](https://neptune.ai/).
4. `merf` - Mixed Effects for Random Forests. Applies correction terms on the predictions of clustered data. Works not only with random forest but with every `sklearn` BaseEstimator.

## Contributions

We welcome contributions to this repository. If you have any questions, please don't hesitate to get in contact by reaching out or filing a [github issue](https://github.com/radlfabs/flexcv/issues).

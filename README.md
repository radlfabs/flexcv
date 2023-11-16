<table>
  <tr>
    <td><img src="docs/images/logo_colored.png" width="200"></td>
    <td><h1>Flexible Cross Validation and Machine Learning for Regression on Tabular Data
</h1></td>
  </tr>
</table>

Authors: Fabian Rosenthal, Patrick Blättermann and Siegbert Versümer

## Introduction
This repository contains the code for the python package `flexcv` which implements flexible cross validation and machine learning for tabular data. It's code is used for the machine learning evaluations in Versümer et al. (2023).
The core functionality has been developed in the course of a research project at Düsseldorf University of Applied Science, Germany.

`flexcv` is a method comparison package for Python that wraps around popular libraries to easily taylor complex cross validation code to your needs.

It provides a range of features for comparing machine learning models on different datasets with different sets of predictors customizing just about everything around cross validations. It supports both fixed and random effects, as well as random slopes.

Install the package and give it a try:

`pip install flexcv`

You can find our documentation [here](https://radlfabs.github.io/flexcv/).

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
9. Wrapper classes for the `statsmodels` package to use their mixed effects models inside of a `sklearn` Pipeline. Read more about that package [here](https://github.com/manifoldai/merf).
10. Uses the `merf` package to apply correction for clustered data using the expectation maximization algorithm and supporting any `sklearn` BaseEstimator. Read more about that package [here](https://github.com/manifoldai/merf).
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

## Getting Started


Let's set up a minimal working example using a LinearRegression estimator and some randomly generated regression data.

```py
# import the interface class, a data generator and our model
from flexcv import CrossValidation
from flexcv.synthesizer import generate_regression
from flexcv.models import LinearModel
  
# generate some random sample data that is clustered
X, y, group, _ = generate_regression(10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42)
```

The `CrossValidation` class is the core of this package. It holds all the information about the data, the models, the cross validation splits and the results. It is also responsible for performing the cross validation and logging the results. Setting up the `CrossValidation` object is easy. We can use method chaining to set up our configuration and perform the cross validation. You might be familiar with this pattern from `pandas` and other packages. The set-methods all return the `CrossValidation` object itself, so we can chain them together. The `perform` method then performs the cross validation and returns the `CrossValidation` object again. The `get_results` method returns a `CrossValidationResults` object which holds all the results of the cross validation. It has a `summary` property which returns a `pandas.DataFrame` with all the results. We can then use the `to_excel` method of the `DataFrame` to save the results to an excel file.

```python
# instantiate our cross validation class
cv = CrossValidation()

# now we can use method chaining to set up our configuration perform the cross validation
results = (
    cv
    .set_data(X, y, group, dataset_name="ExampleData")
    # configure our split strategies. Lets go for a GroupKFold since our data is clustered
    .set_splits(
        method_outer_split=flexcv.CrossValMethod.GROUP
    # add the model class
    .add_model(LinearModel)
    .perform()
    .get_results()
)

# results has a summary property which returns a dataframe
# we can simply call the pandas method "to_excel"
results.summary.to_excel("my_cv_results.xlsx")
```

You can then use the various functions and classes provided by the framework to compare machine learning models on your data.
Additional info on how to get started working with this package will be added here soon as well as to the (documentation)[radlfabs.github.io/flexcv/].

## Documentation

Have a look at our [documentation](https://radlfabs.github.io/flexcv/). We currently add lots of additional guides and tutorials to help you get started with `flexcv`.

## Conclusion

`flexcv` is a powerful tool for comparing machine learning models on different datasets with different sets of predictors. It provides a range of features for cross-validation, parameter selection, and experiment tracking. With its state-of-the-art optimization package and logging dashboard, it is a valuable addition to any machine learning workflow.

## Acknowledgements

We would like to thank the developers of `sklearn`, `optuna`, `neptune` and `merf` for their great work. Without their awesome packages and dedication, this project would not have been possible. The logo design was generated by [Microsoft Bing Chat Image Creator](https://www.bing.com/images/create) using the prompt "Generate a logo graphic where a line graph becomes the letters 'c' and 'v'. Be as simple and simplistic as possible."

## Contributions

We welcome contributions to this repository. Feel free to open an issue or pull request if you have any suggestions, problems or questions. Since the project is maintained as a side project, we cannot guarantee a quick response or fix. However, we will try to respond as soon as possible. We strongly welcome contributions to the documentation and tests. If you have any questions about contributing, feel free to contact us.

## About

`flexcv` was developed at the [Institute of Sound and Vibration Engineering](https://isave.hs-duesseldorf.de/) at the University of Applied Science Düsseldorf, Germany. However, it's maintainers are not affiliated with the university anymore. The package is maintained by Fabian Rosenthal.
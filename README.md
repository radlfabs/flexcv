# flexcv - Flexible Cross Validation and Machine Learning for Regression on Tabular Data

Authors: Fabian Rosenthal, Patrick Blättermann and Siegbert Versümer

This repository contains the code for the python package `flexcv` which implements flexible cross validation and machine learning for tabular data. It's code is used for the machine learning evaluations in Versümer et al. (2023).

`flexcv` is a method comparison package for Python that wraps around popular libraries to easily taylor complex cross validation code to your needs.

It provides a range of features for comparing machine learning models on different datasets with different sets of predictors customizing just about everything around cross validations. It supports both fixed and random effects, as well as random slopes.

These are the core packages used under the hood in `flexcv`:

1. `sklearn` - A very popular machine learning library. We use their Estimator API for models, the pipeline module, the StandardScaler, metrics and of course wrap around their cross validation split methods. Learn more [here](https://scikit-learn.org/stable/).
2. `Optuna` - A state-of-the-art optimization package. We use it for parameter selection in the inner loop of our nested cross validation. Learn more about theoretical background and opportunities [here](https://optuna.org/).
3. `neptune` - Awesome logging dashboard with lots of integrations. It is a charm in combination with `Optuna`. We used it to track all of our experiments. `Neptune` is quite deeply integrated into `flexcv`. Learn more about this great library [here](https://neptune.ai/).

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

## Installation

First, clone this repository.

To use `flexcv` you will need Python 3.10 or 3.11. Some dependencies are not yet compatible with Python version 3.12. As soon as they update their compatibility we can support Python 3.12 as well.

##### Using conda

You can easily install this version of Python using conda (Anaconda or Miniconda). We recommend using a fresh environment for cleanly holding all relevant packages corresponding to this repo. With conda installed you can create a new Python 3.10 environment, activate it and install our requirements by running the following lines from the command line:

```bash
conda create --name flexcv python=3.10
conda activate flexcv
conda install pip
cd path/to/this/repo
pip install -r flexcv/requirements.txt
```

##### Using env

To separate Python environments on your system, you can also use the `venv` package from the standard library.

```bash
cd path/to/this/repo
python -m venv my_env_name
my_env_name/Scripts/activate
pip install flexcv/requirements.txt
```

##### Additional dependencies `rpy2`

Some of our model classes are actually wrapping around `rpy2` code and are using `R` under the hood. To use them, you should use a recent `R` version and run our `install_rpackages.py` script:

```bash
cd flexcv
python install_rpackages.py
```

Now you have installed everything you need to perform flexible cross validation and machine learning on your tabular data.

## Getting Started

Note: The interface is currently under review and may be changed soon. Therefore, the getting started may be a off at the moment.

Let's set up a minimal working example using a LinearRegression estimator and some randomly generated regression data.

```py
# import the most important object
from flexcv import CrossValidation
# import the function for data generation
from flexcv.data_generation import generate_regression
# import the model class
from flexcv.models import LinearModel
  
# make sample data
X, y, group, random_slopes = generate_regression(10, 100, n_slopes=1, noise=9.1e-2)
  
# create a model mapping
model_map = ModelMappingDict({
    "LinearModel": ModelConfigDict({
        "model": LinearModel,
    }),
})

# instantiate our cross validation class
cv = CrossValidation()

# now we can use method chaining to set up our configuration perform the cross validation
results = (
    cv
    .set_data(X, y, group, dataset_name="ExampleData")
    .set_splits(method_outer_split=flexcv.CrossValMethod.GROUP, method_inner_split=flexcv.CrossValMethod.KFOLD)
    .set_models(model_map)
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

Have a look at our [documentation](https://radlfabs.github.io/flexcv/). Besides a guide to get started, we will add tutorials, a more detailled user guide and an API reference in the future.

## Conclusion

`flexcv` is a powerful tool for comparing machine learning models on different datasets with different sets of predictors. It provides a range of features for cross-validation, parameter selection, and experiment tracking. With its state-of-the-art optimization package and logging dashboard, it is a valuable addition to any machine learning workflow.

## Contributions

We welcome contributions to this repository. Feel free to open an issue or pull request if you have any suggestions or questions. We are happy to help you get started with `flexcv` and are looking forward to your contributions.

## Acknowledgements

We would like to thank the developers of `sklearn`, `optuna`, `neptune` and `merf` for their great work. Without their awesome packages and dedication, this project would not have been possible.

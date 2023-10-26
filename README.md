# flexcv - Flexible Cross Validation and Machine Learning for Tabular Data

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
5. Easy usage of the state-of-the-art logging dashboard `neptune` to track all of your experiments.
6. Easy local summary of all evaluation metrics in a single table.

## Installation

First, clone this repository.

To use `flexcv` you will need Python v3.10. You can easily install this version of Python using conda (Anaconda or Miniconda). We recommend using a fresh environment for cleanly holding all relevant packages corresponding to this repo. With conda installed you can create a new Python 3.10 environment, activate it and install our requirements by running the following lines from the command line:

```bash
conda create --n flexcv python=3.10
conda activate flexcv
conda install pip
cd path/to/this/repo
pip install -r requirements.txt
```

Some of our model classes are actually wrapping around `rpy2` code and are using `R` under the hood. To use them, you should use a recent `R` version and run our `install_rpackages.py` script:

```bash
cd flexcv
python install_rpackages.py
```

Now you have installed everything you need to perform flexible cross validation and machine learning on your tabular data.

## Getting Started

To get started with the ML Method Comparison Framework, you can import it into your Python script or notebook:

```py
from flexcv import CrossValidation
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

We would like to thank the developers of `sklearn`, `optuna` and `neptune` for their great work. Without their awesome packages and dedication, this project would not have been possible.
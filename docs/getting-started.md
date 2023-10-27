# Getting Started with flexcv

This guide will help you get started with the `flexcv` package quickly. It will show you how to install the package and how to use it to compare machine learning models on your data.

## Installation

Clone our [repository on github](https://github.com/radlfabs/flexcv) to your computer.

`flexcv` was tested using Python v3.10. You can easily install this version of Python using conda (Anaconda or Miniconda). We recommend using a fresh environment for cleanly holding all relevant packages corresponding to this repo. With conda installed, you can create a new Python 3.10 environment, activate it, and install our requirements by running the following lines from the command line:

```bash
conda create --n flexcv python=3.10
conda activate flexcv
conda install pip
cd path/to/this/repo
pip install -r requirements.txt
```

Now you have installed everything you need to perform flexible cross validation and machine learning on your tabular data.

## Getting Started

To get started with `flexcv`, we take you through a quick example using a LinearModel on a randomly generated regression dataset.

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
    .set_dataframes(X, y, group, dataset_name="ExampleData")
    .set_splits(method_outer_split=flexcv.CrossValMethod.GROUP, method_inner_split=flexcv.CrossValMethod.KFOLD)
    .set_models(model_map)
    .perform()
    .get_results()
)

# results has a summary property which returns a dataframe
# we can simply call the pandas method "to_excel"
results.summary.to_excel("my_cv_results.xlsx")

```

You can then use the various configuration classes provided by the package to compare machine learning models on your data.

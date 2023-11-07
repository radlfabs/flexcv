# Getting Started with flexcv

This guide will help you get started with the `flexcv` package quickly. It will show you how to install the package and how to use it to compare machine learning models on your data.

## Installation

Clone our [repository on github](https://github.com/radlfabs/flexcv) to your computer.

##### Usind conda

`flexcv` was tested with Python 3.10 and 3.11. You can easily install this version of Python using conda (Anaconda or Miniconda). We recommend using a fresh environment for cleanly holding all relevant packages corresponding to this repo. With conda installed, you can create a new Python 3.11 environment, activate it, and install our requirements by running the following lines from the command line:

```bash
conda create --n flexcv python=3.11
conda activate flexcv
conda install pip
cd path/to/this/repo
pip install -r requirements.txt
```

##### Using venv

To separate Python environments on your system, you can also use the `venv` package from the standard library.

```bash
cd path/to/this/repo
python -m venv my_env_name
my_env_name/Scripts/activate
pip install flexcv/requirements.txt
```

##### Additional dependencies for `rpy2`

Some of our model classes are actually wrapping around `rpy2` code and are using `R` under the hood. To use them, you should use a recent `R` version and run our `install_rpackages.py` script:

```bash
cd flexcv
python install_rpackages.py
```

Now you have installed everything you need to perform flexible cross validation and machine learning on your tabular data.

## Getting Started

To get started with `flexcv`, we take you through a couple of quick and basic code examples.

###### Linear Model

First we will use a LinearModel on a randomly generated regression dataset. Because Linear Models do not have any hyperparameters, we naturally don't need an inner cross validation loop.

```py
# import the most important object
from flexcv import CrossValidation
# import the function for data generation
from flexcv.synthesizer import generate_regression
# import the model class
from flexcv.models import LinearModel
  
# make sample data
X, y, group, random_slopes = generate_regression(10, 100, n_slopes=1, noise_level=9.1e-2)
  
# create a model mapping
model_map = ModelMappingDict({
    "LinearModel": ModelConfigDict({
	# pass the model class but NOT the instance ;)
        "model": LinearModel,
	# specify if your model needs a R-style formula for the fit
	"requires_formula": True,
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

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

To get started with `flexcv`, we take you through a couple of quick and basic code examples.

###### Linear Model

First we will use a LinearModel on a randomly generated regression dataset. Because Linear Models do not have any hyperparameters, we naturally don't need an inner cross validation loop.

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

###### Random Forest Regressor

Next, we will use a Random Forest Regressor. We will tune a single hyperparameter, the max_depth of the trees, in the inner cross validation and evaluate the best estimator's performance in the outer cross validation loop. We will use the randomly generated dataset just as in the example above.

```python
import optuna
from sklearn.ensemble import RandomForestRegressor
import flexcv.model_postprocessing as mp

model_map = ModelMappingDict({
    "RandomForest": ModelConfigDict({
	# now we specify, that we do want to evaluate the inner cross validation loop
        "inner_cv": True,
	# let's specify the model's ability to run in parallel
        "n_jobs_model": {"n_jobs": -1},
        "n_jobs_cv": -1,
	# pass the model class
        "model": RandomForestRegressor,
	# pass the parameter distribution
        "params": {
            "max_depth": optuna.distributions.IntDistribution(5, 100), 
        },
	# pass a model post processing function
	# this can be useful for plotting, logging or additional results routines...
        "post_processor": mp.rf_post,
    }),
})
 

cv = CrossValidation()
# just before pass everything to CrossValidation using method chaining
results = (
    cv.set_data(X, y)
    .set_models(model_map)
    .set_inner_cv(3)
    .set_splits(n_splits_out=3)
    .set_run(Run())
    .perform()
    .get_results()
)
# Print the averaged R²
n_values = len(results["RandomForest"]["metrics"])
r2_values = [results["RandomForest"]["metrics"][k]["r2"] for k in range(n_values)]
print(np.mean(r2_values))
```

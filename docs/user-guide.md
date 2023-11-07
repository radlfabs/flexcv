# User Guide

This user guide will give you a detailled guide on how to use `flexcv` functions and objects.

## Fit a Random Forest Regressor

In this section, we will use a Random Forest Regressor to predict a target variable. We use cross validation to estimate the regressor's ability to generalize on unseen data. Also, we want to tune a single hyperparameter, the max_depth of the trees, in the inner cross validation and evaluate the best estimator's performance in the outer cross validation loop. We will use the randomly generated dataset just as in the example in our [Getting-Started](getting-started.md) guide.

Note how we use Optuna distributions to specify the hyperparameter search space. In the same syntax, you can add all kinds of hyperparameter distributions to the params dictionary in the mapping.

Also, we use a model post processing function to extract the feature importances and plot them as SHAP beeswarm plots in Neptune. This is a very powerful feature of `flexcv` and you can use it to implement any kind of post processing you want. You can also use it to log additional metrics or results to Neptune.

```python
import optuna
from sklearn.ensemble import RandomForestRegressor
import flexcv.model_postprocessing as mp

model_map = ModelMappingDict({
    "RandomForest": ModelConfigDict({
	# now we specify, that we do want to evaluate the inner cross validation loop
        "requires_inner_cv": True,
	# let's specify the model's ability to run in parallel
        "n_jobs_model": -1,
        "n_jobs_cv": -1,
	# pass the model class
        "model": RandomForestRegressor,
	# pass the parameter distribution
        "params": {
            # we use optuna distributions to specify the hyperparameter search space
            # let's tune the max_depth of the trees between 5 and 100
            "max_depth": optuna.distributions.IntDistribution(5, 100), 
        },
	# pass a model post processing function
	# this can be useful for plotting, logging or additional results routines...
        "post_processor": mp.rf_post,
    }),
})
 
# instantiate the CrossValidation class
cv = CrossValidation()

# pass everything to CrossValidation using method chaining
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

# save the results table to an excel file
results.summary.to_excel("results.xlsx")
```

Note: If you have a model that does not have the `n_jobs` or `random_state` arguments in it's signature, you can prevent errors by specifying this in the `ModelConfigDict` like so:

```python
model_map = ModelMappingDict({
    "SVR": ModelConfigDict({
	# now we specify, that we do want to evaluate the inner cross validation loop
        "requires_inner_cv": True,
	# If a model can not run in parallel and does not allow a random_state specify
	"allows_seed": False,
	"allows_n_jobs": False,
	# pass the model class
        "model": SupportVectorRegressor,
        "post_processor": mp.svr_post,
    }),
})
```

## Repeated Cross Validation

Some of the cross validation splits are performed with shuffling the data before dividing in train and test splits. Therefore, you might wonder if your evaluation varies for multiple runs.

In the standard configuration, you would seed every run to make it absolutely reproducible. Now we want to explore, how different seeds influence the cross validation results. This is call repeated cross validation. We can still seed this process though by randomly generating a number of seeds. This makes even the repeated CV reproducible.

First, we create our random data set and a basic model mapping just as in a single run.

Second, we instantiate a `RepeatedCV` object. This class not only has the `set`-methods just as CrossValidation but also implements `set_n_repeats()` and `set_neptune()`. We can chain these methods because they also return the class `self` and we use them to set the number of repetitions as well as passing the credentials for Neptune runs. `RepeatedCV` then takes care of instantiating the desired number of runs and logs every single cross validation to it's own neptune run.

Most importantly `RepeatedCV` implements the iteration over single cross validation runs in it's `perform()` method. We can chain `perform()` in the same manner as we are now used to. The last element of our chain should also be `get_results`. This will allow us to inspect summary statistics as a measure of variance in the runs.

Here is the full code to perform cross validation 3 times and get summary statistics for all folds and models.

```python
from flexcv.synthesizer import generate_regression
from flexcv.models import LinearModel
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.repeated import RepeatedCV

# make sample data
X, y, group, random_slopes =generate_regression(10,100,n_slopes=1,noise_level=9.1e-2)

# create a basic model mapping
model_map = ModelMappingDict(
    {
    	"LinearModel": ModelConfigDict(
    	{
    		"model": LinearModel,
    	}
    ),
    }
)

credentials = {}

rcv = (
    RepeatedCV()
    .set_data(X, y, group,dataset_name="ExampleData")
    .set_models(model_map)
    .set_n_repeats(3)
    .set_neptune(credentials)
    .perform()
    .get_results()
)

rcv.summary.to_excel("repeated_cv.xlsx")  # save dataframe to excel file
```

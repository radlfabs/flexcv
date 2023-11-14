## Fit a Random Forest Regressor

In this section, we will use a Random Forest Regressor to predict a target variable. We use cross validation to estimate the regressor's ability to generalize on unseen data. Also, we want to tune a single hyperparameter, the max_depth of the trees, in the inner cross validation and evaluate the best estimator's performance in the outer cross validation loop. We will use the randomly generated dataset just as in the example in our [Getting-Started](getting-started.md) guide.

Note how we use Optuna distributions to specify the hyperparameter search space. In the same syntax, you can add all kinds of hyperparameter distributions to the params dictionary in the mapping.

Also, we use a model post processing function to extract the feature importances and plot them as SHAP beeswarm plots in Neptune. This is a very powerful feature of `flexcv` and you can use it to implement any kind of post processing you want. You can also use it to log additional metrics or results to Neptune.

```python
import optuna
from sklearn.ensemble import RandomForestRegressor
from flexcv.model_postprocessing import RandomForestModelPostProcessor

# lets start with generating some clustered data
X, y, group, random_slopes =generate_regression(
    10, 100, n_slopes=1 ,noise_level=9.1e-2
)

# define a set of hyperparameters to tune
params = {
    "max_depth": optuna.distributions.IntDistribution(5, 100),
    "n_estimators": optuna.distributions.CategoricalDistribution([10]),
}

# instantiate the CrossValidation class
cv =CrossValidation()

# pass everything to CrossValidation using method chaining
results = (
    cv.set_data(X, y)
    .add_model(
        model=RandomForestRegressor,
        requires_inner_cv=True,
        params=params,
        post_processor=RandomForestModelPostProcessor,
    )
    .set_inner_cv(3)
    .set_splits(n_splits_out=3)
    .set_run(Run())
    .perform()
    .get_results()
)

# Print the averaged R²
n_values =len(results["RandomForestRegressor"]["metrics"])
r2_values =[results["RandomForestRegressor"]["metrics"][k]["r2"]for k inrange(n_values)]
print(np.mean(r2_values))
# save the results table to an excel file
results.summary.to_excel("results.xlsx")

```

Note: If you have a model that does not have the `n_jobs` or `random_state` arguments in it's signature, you can prevent errors by specifying this in the `ModelConfigDict` like so:

```python

model_map =ModelMappingDict({

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

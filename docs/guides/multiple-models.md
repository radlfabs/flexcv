## Evaluating multiple models

`flexcv` offers two ways of passing multiple models in set-up to our CrossValidation interface class. You either can call `.add_model` multiple times on the class instance or you can pass a `ModelMappingDict` to the class instance. The latter may be the preferred way of doing it when the number of models gets larger and you want to reuse the configuration. We will discuss both ways in this guide.
For both ways of interacting with the `CrossValidation` class instance, a `ModelMappingDict` is created internally and stored to the instance's `config` attribute. The core function `cross_validate` then just iterates over the `ModelMappingDict` and fits every model to the data. As additional benefit, this provides extensive logging, results summaries and useful information such as progress bars for all layers of processes.

So let's start with the way of adding two models the way we learned before. Say, we want to compare a LinearMixedEffectsModel to a RandomForestRegressor and a MERF correction for clustered data.
Thats as simple as this:

```python
import optuna
from sklearn.ensemble import RandomForestRegressor
from flexcv import CrossValidation
from flexcv.models import LinearMixedEffectsModel
from flexcv.merf import MERF
from flexcv.model_postprocessing import RandomForestModelPostProcessor, LinearMixedEffectsModelPostProcessor
from flexcv.synthesizer import generate_regression


# lets start with generating some clustered data
X, y, group, random_slopes =generate_regression(
    10,100,n_slopes=1,noise_level=9.1e-2
)
# define our hyperparameters
params = {
    "max_depth": optuna.distributions.IntDistribution(5,100),
    "n_estimators": optuna.distributions.CategoricalDistribution([10]),
}

cv =CrossValidation()
results = (
    cv.set_data(X, y, group, random_slopes)
    .set_models(model_map)
    .set_inner_cv(3)
    .set_splits(n_splits_out=3)
    .add_model(model=LinearMixedEffectsModel, post_processor=LinearMixedEffectsModelPostProcessor)
    .add_model(model=RandomForestRegressor, requires_inner_cv=True, params=params, post_processor=RandomForestModelPostProcessor, add_merf=True)
    .perform()
    .get_results()
)
```
Now when you want to compare a larger number of models it could make sense to pass the mapping directly to the `.set_models` method.

```python
model_map = ModelMappingDict(
    {
        "LinearModel": ModelConfigDict(
            {
                "model": LinearModel,
                "post_processor": mp.LinearModelPostProcessor,
                "requires_inner_cv": False,
            }
            ),
        "LinearMixedEffectsModel": ModelConfigDict(
            {
                "model": LinearMixedEffectsModel,
                "post_processor": mp.LMERPostProcessor,
                "requires_inner_cv": False,
            }
        ),
        "RandomForest": ModelConfigDict(
            {
                "model": RandomForestRegressor,
                "params": {
                    "max_depth": optuna.distributions.IntDistribution(5,100),
                    "n_estimators": optuna.distributions.CategoricalDistribution(
                        [10]
                    ),
                "post_processor": mp.RandomForestModelPostProcessor
                "requires_inner_cv": True,
                "add_merf": True,

                },
            }
        )
    }
)

# and then call .set_models on your CrossValidation instance
cv = CrossValidation()
cv.set_models(model_map)

```

Have a look at this section from `flexcv.model_mapping_template` for how to add multiple models to a `ModelMappingDict`:

```python # TODO redo this template
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm importSVR
from xgboost import XGBRegressor

from . import model_postprocessing as mp
from .merf importMERF
from .model_mapping import ModelConfigDict, ModelMappingDict
from .models import LinearMixedEffectsModel, LinearModel

MODEL_MAPPING=ModelMappingDict(
    {
        "LinearModel": ModelConfigDict(
            {
                "requires_inner_cv": False,
                "n_trials": 100,
                "n_jobs_model": 1,
                "n_jobs_cv": 1,
                "model": LinearModel,
                "params": {},
                "post_processor": mp.LinearModelPostProcessor,
            }
        ),
        "LinearMixedEffectsModel": ModelConfigDict(
            {
                "requires_inner_cv": False,
                "n_trials": 100,
                "n_jobs_model": 1,
                "n_jobs_cv": 1,
                "model": LinearMixedEffectsModel,
                "params": {},
                "post_processor": mp.LMERPostProcessor,
            }
        ),
        "RandomForest": ModelConfigDict(
            {
                "requires_inner_cv": True,
                "n_trials": 400,
                "n_jobs_model": 1,
                "n_jobs_cv": -1,
                "model": RandomForestRegressor,
                "params": {
                    "max_depth": optuna.distributions.IntDistribution(5,100),
                    "min_samples_split": optuna.distributions.IntDistribution(
                        2,1000,log=True
                    ),
                    "min_samples_leaf": optuna.distributions.IntDistribution(
                        2,5000,log=True
                    ),
                    "max_samples": optuna.distributions.FloatDistribution(0.0021,0.9),
                    "max_features": optuna.distributions.IntDistribution(1,10),
                    "max_leaf_nodes": optuna.distributions.IntDistribution(10,40000),
                    "min_impurity_decrease": optuna.distributions.FloatDistribution(
                        1e-8,0.02,log=True
                    ),  # >>>> can be (1e-8, .01, log=True)
                    "min_weight_fraction_leaf": optuna.distributions.FloatDistribution(
                        0,0.5
                   ),  # must be a float in the range [0.0, 0.5]
                    "ccp_alpha": optuna.distributions.FloatDistribution(1e-8,0.01),
                    "n_estimators": optuna.distributions.IntDistribution(2,7000),
                },
            }
        ),
        "XGBoost": ModelConfigDict(
            {
                "requires_inner_cv": True,
                "n_trials": 300,
                "n_jobs_model": 1,
                "n_jobs_cv": -1,
                "model": XGBRegressor,
                "params": {
                    "max_depth": optuna.distributions.IntDistribution(2,700),
                    "learning_rate": optuna.distributions.FloatDistribution(0.01,0.8),
                    "n_estimators": optuna.distributions.IntDistribution(5,5000),
                    "min_child_weight": optuna.distributions.IntDistribution(2,100),
                    # "max_delta_step": optuna.distributions.FloatDistribution(0.1, 10.0),
                    # "gamma": optuna.distributions.FloatDistribution(0.0, 50),
                    "subsample": optuna.distributions.FloatDistribution(0.005,0.97),
                    "colsample_bytree": optuna.distributions.FloatDistribution(
                        0.1,1,step=0.1
                    ),
                    "colsample_bylevel": optuna.distributions.FloatDistribution(
                        0.1,1,step=0.1
                    ),
                    "colsample_bynode": optuna.distributions.FloatDistribution(
                        0.1,1,step=0.1
                    ),
                    "colsample_bytree": optuna.distributions.FloatDistribution(
                        0.1,1,step=0.1
                    ),
                    "colsample_bylevel": optuna.distributions.FloatDistribution(
                        0.1,1,step=0.1
                    ),
                    "colsample_bynode": optuna.distributions.FloatDistribution(
                        0.1,1,step=0.1
                    ),
                    "reg_alpha": optuna.distributions.FloatDistribution(0.1,500),
                    "reg_lambda": optuna.distributions.FloatDistribution(0.001,800),
                },
            }
        ),
        "SVR": ModelConfigDict(
            {
                "requires_inner_cv": True,
                "n_trials": 450,
                "allows_n_jobs": False,
                "n_jobs_cv": -1,
                "model": SVR,
                "params": {
                    # Most Important: Kernel + C
                    # "kernel": default "rbf" yielded best results
                    # "degree": # for poly only
                    "C": optuna.distributions.FloatDistribution(0.001,50,log=True),
                    "epsilon": optuna.distributions.FloatDistribution(0.1,1.3),
                    "gamma": optuna.distributions.FloatDistribution(
                        1e-5,0.1,log=True
                    ),  # better than default "scale"
                    # "tol": optuna.distributions.FloatDistribution(1e-4, 10),
                    # "shrinking": default "True" yielded best restults
                },
            }
        ),
    }
)


```

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
from flexcv.model_postprocessing import RandomForestModelPostProcessor, LMERModelPostProcessor
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
    .set_inner_cv(3)
    .set_splits(n_splits_out=3)
    .add_model(model_class=LinearMixedEffectsModel, post_processor=LMERModelPostProcessor)
    .add_model(model_class=RandomForestRegressor, requires_inner_cv=True, params=params, post_processor=RandomForestModelPostProcessor, add_merf=True)
    .perform()
    .get_results()
)
```
Now when you want to compare a larger number of models you can assign your customized ModelMappingDict as below and pass the mapping directly to the `.set_models` method.

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

## Configuration using yaml

As another and convenient method to configure multiple models you can pass yaml-code to the interface.
This is especially useful when you want to reuse the configuration for multiple runs.

As a hidden gem, we implemented a yaml-parser that can take care of imports of model classes and postprocessors.
It also takes care of instantiating the optuna distributions for hyperparameter optimization.

Just use the following yaml tags:

- `!Int` for `optuna.distributions.IntDistribution`
- `!Float` for `optuna.distributions.FloatDistribution`
- `!Categorical` for `optuna.distributions.CategoricalDistribution`

Use the following syntax to define the distribution:

```yaml
!Int
  low: 5
  high: 100
  step: 1
  log: true
```
Note: You have to provide keys for the distribution parameters. Also you have to provide low and high values. 
Exceptions are the `step` parameter for `IntDistribution` which defaults to 1 and the `log` parameter which defaults to False.

With our yaml configuration we could define models like this:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm importSVR
from xgboost import XGBRegressor

from flexcv import CrossValidation
from flexcv.merf import MERF
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.models import LinearMixedEffectsModel, LinearModel

yaml_mapping = """
LinearModel:
    requires_inner_cv: False
    n_jobs_model: 1
    n_jobs_cv: 1
    model: flexcv.models.LinearModel
    post_processor: flexcv.model_postprocessing.LinearModelPostProcessor

LMER:
    requires_inner_cv: False
    n_jobs_model: 1
    n_jobs_cv: 1
    model: flexcv.models.LinearMixedEffectsModel
    post_processor: flexcv.model_postprocessing.LMERModelPostProcessor

RandomForest:
  requires_inner_cv: true
  n_trials: 400
  n_jobs_model: -1
  n_jobs_cv: 1
  model: sklearn.ensemble.RandomForestRegressor
  params:
    max_depth: !Int
      low: 5
      high: 100
    min_samples_split: !Int
      low: 2
      high: 1000
      log: true
    min_samples_leaf: !Int
      low: 2
      high: 5000
      log: true
    max_samples: !Float
      low: 0.0021
      high: 0.9
    max_features: !Int
      low: 1
      high: 10
    max_leaf_nodes: !Int
      low: 10
      high: 40000
    min_impurity_decrease: !Float
      low: 0.0000000008
      high: 0.02
      _kwargs:
        log: true
    min_weight_fraction_leaf: !Float
      low: 0, 
      high: 0.5
    ccp_alpha: !Float
      low: 1e-08
      high: 0.01
    n_estimators: !Int
      low: 2
      high: 7000
  post_processor: flexcv.model_postprocessing.RandomForestModelPostProcessor
  add_merf: true
"""

# and then call .set_models on your CrossValidation instance
cv = CrossValidation()
cv.set_models(yaml_string=yaml_mapping)

# if you have your yaml 

```




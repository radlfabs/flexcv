## Evaluating multiple models

`flexcv` offers two ways of passing multiple models in set-up to our CrossValidation interface class. You either can call `add_model()` multiple times on the class instance or you can pass multiple models to `set_models()` to set a configuration for multiple models at once. The latter may be the preferred way of doing it when the number of models gets larger and you want to reuse the configuration. We will discuss both ways in this guide.

For both ways of interacting with the `CrossValidation` class instance, a `ModelMappingDict` is created internally and stored to the instance's `config` attribute. 
Since both `add_model()` and `set_models()` are updating the same attribute of the class instance, you can use both ways in combination. This is especially useful when you want to add a model to a configuration that you already set up using `set_models()`.

In `CrossValidation.perform` the core function `cross_validate` is called and iterates over the keys in `ModelMappingDict` and fits every model to the data. As additional benefit, this provides extensive logging, results summaries and useful information such as progress bars for all layers of processes.

### Using add_model()

So let's start with the way of adding two models the way we learned before. Say, we want to compare a LinearModel to a RandomForestRegressor.
Thats as simple as this:

```python
import optuna
from sklearn.ensemble import RandomForestRegressor
from flexcv import CrossValidation
from flexcv.models import LinearModel
from flexcv.merf import MERF
from flexcv.model_postprocessing import RandomForestModelPostProcessor, LinearModelPostProcessor
from flexcv.synthesizer import generate_regression


# lets start with generating some clustered data
X, y, group, random_slopes =generate_regression(
    3,100,n_slopes=1,noise_level=9.1e-2
)
# define our hyperparameters for the random forest
params = {
    "max_depth": optuna.distributions.IntDistribution(5,100),
}

cv =CrossValidation()
(
  cv.set_data(X, y, group, random_slopes)
  .set_inner_cv(3)
  .set_splits(n_splits_out=3)
  .add_model(model_class=LinearModel, post_processor=LinearModelPostProcessor)
  .add_model(model_class=RandomForestRegressor, requires_inner_cv=True, params=params, post_processor=RandomForestModelPostProcessor)
)
```

### Configuration using yaml

A great and convenient method to configure multiple models at once is passing yaml-code to the interface.
This is especially useful when you want to reuse the configuration for multiple runs and save it to a file.
`.set_models` takes either a yaml-string or a path to a yaml-file.

As a hidden gem, we implemented a yaml-parser that can take care of imports of model classes and postprocessors.
It also takes care of instantiating the optuna distributions for hyperparameter optimization.

Just use the following yaml tags:

- `!Int` for `optuna.distributions.IntDistribution`
- `!Float` for `optuna.distributions.FloatDistribution`
- `!Categorical` for `optuna.distributions.CategoricalDistribution`

Note: Don't put commas to end the lines of the distributions in the yaml file. This will break the instantiation of the distributions since the yaml parser will interpret the comma as part of the distribution and cast it to float.

Please also note, that the yaml parser does not allow scientific notation at the moment. This is due to the fact that the yaml parser will interpret the scientific notation as a string and not as a float. This may be improved in future versions when pyaml is updating their regex that constructs floats.

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
from sklearn.svm import SVR
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
  n_trials: 10
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
      log: true
    min_weight_fraction_leaf: !Float
      low: 0
      high: 0.5
    ccp_alpha: !Float
      low: 0.000008
      high: 0.01
    n_estimators: !Int
      low: 2
      high: 7000
  post_processor: flexcv.model_postprocessing.RandomForestModelPostProcessor
"""

# and then call .set_models on your CrossValidation instance
cv = CrossValidation()
cv.set_models(yaml_string=yaml_mapping)

```
In addition to the yaml configuration, you can also pass a path to a yaml file to the `.set_models` method by using the `yaml_path` keyword argument.

```python	
import yaml
from flexcv import CrossValidation

yaml_code = """
LinearModel:
  requires_inner_cv: False
  n_jobs_model: 1
  n_jobs_cv: 1
  model: flexcv.models.LinearModel
  post_processor: flexcv.model_postprocessing.LinearModelPostProcessor
"""
with open("my_yaml.yaml", "w") as f:
    yaml.safe_dump(yaml_code, f)
  
cv = CrossValidation()
cv.set_models(yaml_path="my_yaml.yaml")

```

### Configuration using a ModelMappingDict

Of course, when you want to compare a larger number of models you can assign them to a customized ModelMappingDict directly and pass the mapping directly to the `.set_models` method.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import optuna

from flexcv import CrossValidation
from flexcv.merf import MERF
from flexcv.model_mapping import ModelConfigDict, ModelMappingDict
from flexcv.models import LinearMixedEffectsModel, LinearModel
import flexcv.model_postprocessing as mp

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
                "post_processor": mp.LMERModelPostProcessor,
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
                "post_processor": mp.RandomForestModelPostProcessor,
                "requires_inner_cv": True,
                },
            }
        )
    }
)

# and then call .set_models on your CrossValidation instance
cv = CrossValidation()
cv.set_models(model_map)

```

In this guide you learned several ways to set up your models for cases where you want to compare multiple models in the same run. You have seen how to use the `add_model()` method, how to use yaml configuration and how to use a `ModelMappingDict` to set up your models. You can use all of these methods in combination to set up your models and to fully customize your cross validation setup.
This makes it easy to compare multiple models on your data and to find the best model for your use case. A big help is the neptune integration that we provide. You can find a more detailled guide on how to use it [here](neptune-integration.md).
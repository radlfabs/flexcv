from typing import Dict, Type
import optuna
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

from .models import LinearModel
from .models import LinearMixedEffectsModel
from .models import EarthRegressor
from .merf_adaptation import MERF
from . import model_postprocessing as mp

"""
- Model mapping is a dictionary of dictionaries, where the keys of the outer dictionary are the names of the models
- it is used to map the model names to the model classes, hyperparameter distributions, post-processing functions, etc.
- it is used in the cross_val.py file to create the models by iterating over the outer dictionary

MODEL_MAPPING = {
    "model_name_1": {
        "inner_cv": bool,  # determines whether to use inner CV for hyperparameter tuning
        "n_jobs_model": dict,  # dict {"n_jobs": int} or empty {} to pass to model constructor
        "n_jobs_cv": int,  # number of jobs to use for inner CV, should be -1 or 1
        "model": model_class,  # model class to use
        "params": dict,  # dict of hyperparameter distributions to use for optuna
        "post_processor": post_processor_function,  # function to use for post-processing
        "level_4_model": level_4_model_class,  # model class to use for level 4 model
        "level_4_post_processor": level_4_post_processor_function  # function to use for level 4 post-processing
    },
    "model_name_2": {
        "inner_cv": bool,
        "n_jobs_model": dict,
        "n_jobs_cv": int,
        "model": model_class,
        "params": dict,
        "post_processor": post_processor_function,
        "level_4_model": level_4_model_class,
        "level_4_post_processor": level_4_post_processor_function
    },
    # ...
}
"""


class ModelConfigDict(Dict[str, Type]):
    """A dictionary that maps model configuration names to their corresponding types.
    Usage:
        ```py
        {
            "inner_cv": bool,
                    # this flag can be set to control if a model is used in the inner cross validation.
                    # if set to False, the model will be instantiated in the outer cross validation without hyper parameter optimization.
            "n_trials": int,
                    # number of trials to be used in hyper parameter optimization.
            "n_jobs_model": {"n_jobs": 1},
                    # number of jobs to be used in the model. We use the sklearn convention here.
                    # n_jobs_model is passed to the model constructor as **n_jobs_model
                    # therefore, it MUST be a dictionary with the key "n_jobs" and the value being an integer
                    # if you want to leave it empty, you can pass the empty dict {}.
            "n_jobs_cv": 1,
                    # number of jobs to be used in the inner cross validation/hyper parameter tuning. We use the sklearn convention here.
            "model": BaseEstimator,
                    # pass your sklearn model here. It must be a class, not an instance.
            "params": {},
                    # pass the parameters to be used in the model here. It must be a dictionary of optuna distributions or an empty dict.
            "post_processor": mp.lm_post,
                    # pass the post processor function to be used here. It must be a callable.
            "mixed_model": BaseEstimator,
                    # pass the mixed effects model to be used here. It must be a class, not an instance.
                    # it's fit method must have the same signature as the fit method of the sklearn models.
            "mixed_post_processor": mp.lmer_post,
                    # pass the post processor function to be used here. It must be a callable.
            "mixed_name": "MixedLM"
                    # name of the mixed effects model. It is used to identify the model in the results dictionary.
        }```
    """

    pass


class ModelMappingDict(Dict[str, ModelConfigDict]):
    """A dictionary that maps model names to  model configuration dicts.
    Usage:
    ```py
    model_mapping = ModelMappingDict({
        "LinearModel": ModelConfigDict(
            {...}
        ),
        "SecondModel": ModelConfigDict(
            {...}
        ),
        )
    ```
    """

    pass


from sklearn.base import BaseEstimator


def make_model_config_from_estimator(
    fixed_effects_model: BaseEstimator,
    mixed_effects_model: BaseEstimator = None,
    fixed_post_processing_fn: callable = None,
    mixed_post_processing_fn: callable = None,
) -> ModelConfigDict:
    """Creates a model configuration dictionary from a sklearn model.
    This function is used to create a model configuration dictionary from a sklearn model.
    It is used in the `CrossValidation` class to create the model configuration dictionary from the model mapping dictionary.
    Args:
        model: A sklearn model.
    Returns:
        A model configuration dictionary.
    """

    def empty_func(*args, **kwargs):
        pass

    if not mixed_effects_model:
        mixed_effects_model = None

    return ModelConfigDict(
        {
            "inner_cv": False,
            "n_trials": 100,
            "n_jobs_model": {"n_jobs": 1},
            "n_jobs_cv": 1,
            "model": fixed_effects_model.__class__,
            "params": {},
            "post_processor": fixed_post_processing_fn
            if fixed_post_processing_fn
            else empty_func,
            "level_4_model": mixed_effects_model
            if mixed_effects_model
            else BaseEstimator,
            "level_4_post_processor": mixed_post_processing_fn
            if mixed_post_processing_fn
            else empty_func,
            "level_4_name": mixed_effects_model.__class__.__name__
            if mixed_effects_model
            else "MixedModel",
        }
    )


MODEL_MAPPING: Dict[str, Dict] = {
    "LinearModel": {
        "inner_cv": False,
        "n_trials": 100,
        "n_jobs_model": {"n_jobs": 1},
        "n_jobs_cv": 1,
        "model": LinearModel,
        "params": {},
        "post_processor": mp.lm_post,
        "level_4_model": LinearMixedEffectsModel,
        "level_4_post_processor": mp.lmer_post,
        "level_4_name": "MixedLM",
    },
    "RandomForest": {
        # https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
        "inner_cv": True,
        "n_trials": 400,
        "n_jobs_model": {"n_jobs": -1},
        "n_jobs_cv": -1,
        "model": RandomForestRegressor,
        "params": {
            #    https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
            "max_depth": optuna.distributions.IntDistribution(5, 100),
            "min_samples_split": optuna.distributions.IntDistribution(
                2, 1000, log=True
            ),
            "min_samples_leaf": optuna.distributions.IntDistribution(2, 5000, log=True),
            "max_samples": optuna.distributions.FloatDistribution(0.0021, 0.9),
            "max_features": optuna.distributions.IntDistribution(1, 10),
            "max_leaf_nodes": optuna.distributions.IntDistribution(10, 40000),
            "min_impurity_decrease": optuna.distributions.FloatDistribution(
                1e-8, 0.02, log=True
            ),  # >>>> can be (1e-8, .01, log=True)
            "min_weight_fraction_leaf": optuna.distributions.FloatDistribution(
                0, 0.5
            ),  # must be a float in the range [0.0, 0.5]
            "ccp_alpha": optuna.distributions.FloatDistribution(1e-8, 0.01),
            "n_estimators": optuna.distributions.IntDistribution(2, 7000),
        },
        "post_processor": mp.rf_post,
        "level_4_model": MERF,
        "level_4_post_processor": mp.expectation_maximation_post,
        "level_4_name": "MERF",
    },
    "XGBoost": {
        # https://www.kaggle.com/code/andreshg/xgboost-optuna-hyperparameter-tunning
        # https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning
        "inner_cv": True,
        "n_trials": 300,
        "n_jobs_model": {"n_jobs": -1},
        "n_jobs_cv": -1,
        "model": XGBRegressor,
        "params": {
            "max_depth": optuna.distributions.IntDistribution(2, 700),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.8),
            "n_estimators": optuna.distributions.IntDistribution(5, 5000),
            "min_child_weight": optuna.distributions.IntDistribution(2, 100),
            # "max_delta_step": optuna.distributions.FloatDistribution(0.1, 10.0),
            # "gamma": optuna.distributions.FloatDistribution(0.0, 50),
            "subsample": optuna.distributions.FloatDistribution(0.005, 0.97),
            "colsample_bytree": optuna.distributions.FloatDistribution(0.1, 1, step=0.1),
            "colsample_bylevel": optuna.distributions.FloatDistribution(0.1, 1, step=0.1),
            "colsample_bynode": optuna.distributions.FloatDistribution(0.1, 1, step=0.1),
            "colsample_bytree": optuna.distributions.FloatDistribution(0.1, 1, step=0.1),
            "colsample_bylevel": optuna.distributions.FloatDistribution(0.1, 1, step=0.1),
            "colsample_bynode": optuna.distributions.FloatDistribution(0.1, 1, step=0.1),
            "reg_alpha": optuna.distributions.FloatDistribution(0.1, 500),
            "reg_lambda": optuna.distributions.FloatDistribution(0.001, 800),
        },
        "post_processor": mp.xgboost_post,
        "level_4_model": MERF,
        "level_4_post_processor": mp.expectation_maximation_post,
        "level_4_name": "XGBEM",
    },
    "MARS": {
        "inner_cv": True,
        "n_trials": 200,
        "n_jobs_model": {},
        "n_jobs_cv": 1,
        "model": EarthRegressor,
        "params": {  # 'degree', 'endspan', 'fast_beta', 'fast_k', 'minspan', 'newvar_penalty', 'nk', 'nprune', 'pmethod', 'random_state', 'thresh'
            "degree": optuna.distributions.IntDistribution(1, 5),
            "nprune": optuna.distributions.IntDistribution(1, 300),
            # "fast_k": optuna.distributions.CategoricalDistribution([0, 1, 5, 10, 20]),  #
            "fast_k": optuna.distributions.IntDistribution(0, 20),  #
            # "nk": does not help
            "newvar_penalty": optuna.distributions.FloatDistribution(0.01, 0.2),
            # "pmethod": # use default: backward
            # "fast_beta": # default(=1) yielded best results
        },
        "post_processor": mp.mars_post,
        "level_4_model": MERF,
        "level_4_post_processor": mp.expectation_maximation_post,
        "level_4_name": "EarthEM",
    },
    "SVR": {
        "inner_cv": True,
        "n_trials": 450,
        "n_jobs_model": {},
        "n_jobs_cv": -1,
        "model": SVR,
        "params": {
            # Most Important: Kernel + C
            # "kernel": default "rbf" yielded best results
            # "degree": # for poly only
            "C": optuna.distributions.FloatDistribution(0.001, 50, log=True),
            "epsilon": optuna.distributions.FloatDistribution(0.1, 1.3),
            "gamma": optuna.distributions.FloatDistribution(1e-5, 0.1, log=True),  # better than default "scale"
            # "tol": optuna.distributions.FloatDistribution(1e-4, 10),
            # "shrinking": default "True" yielded best restults
        },
        "post_processor": mp.svr_post,
        "level_4_model": MERF,
        "level_4_post_processor": mp.expectation_maximation_post,
        "level_4_name": "SVREM",
    },
}

# EM parameters as fine tuned by SV and FR in call on 2023-09-04
EM_MAX_ITER_PER_DATASET: Dict[str, int] = {"HSDD": 200, "ARAUSD": 50, "ISD": 300}
EM_WINDOW_PER_DATASET: Dict[str, int] = {"HSDD": 50, "ARAUSD": 15, "ISD": 25}
EM_THRESH_PER_DATASET: Dict[str, float] = {"HSDD": 0.025, "ARAUSD": 0.005, "ISD": 0.01}

MIXED_TO_BASE_MODEL_MAPPING: Dict[str, str] = {
    "MixedLM": "LinearModel",
    "MERF": "RandomForest",
    "XGBEM": "XGBoost",
    "EarthEM": "MARS",
    "SVREM": "SVR",
}
